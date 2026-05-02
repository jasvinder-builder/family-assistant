"""
Clip recording — segment watcher + clip cutter + finalised-clip index.

The inference container's sibling ffmpeg writes rolling .ts segments under
{CLIPS_DIR}/{cam_id}/segs/. This module:
  - watches each registered camera's segs/ dir and updates CameraRuntime.seg_ring
  - extends a per-camera clip-trigger window on each detection
  - cuts a single .mp4 (concat-stream-copy) when a window expires
  - registers the finalised clip in the global ClipIndex

State split:
  - per-camera: CameraRuntime.seg_ring + CameraRuntime.clip_triggers
                (owned by services.camera_runtime)
  - global:     ClipIndex singleton (the list backing GET /clips)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from services.camera_runtime import registry

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

CLIPS_DIR           = Path(os.environ.get("CLIPS_DIR", "./clips"))
SEG_DURATION_S      = int(os.environ.get("SEG_DURATION_S", "30"))
PRE_BUFFER_S        = int(os.environ.get("PRE_BUFFER_S", "5"))
POST_BUFFER_S       = int(os.environ.get("POST_BUFFER_S", "5"))
MAX_CLIPS_PER_CAM   = int(os.environ.get("MAX_CLIPS_PER_CAM", "100"))
MAX_CLIP_DURATION_S = int(os.environ.get("MAX_CLIP_DURATION_S", "60"))
MIN_CLIP_DETECTIONS = int(os.environ.get("MIN_CLIP_DETECTIONS", "5"))


@dataclass
class SegBoundary:
    """Metadata for one completed rolling segment file."""
    path:       str    # absolute path to the .ts segment file
    start_wall: float  # time.time() when this segment started recording
    end_wall:   float  # time.time() when this segment ended (next segment started)


# ── ClipIndex (the list backing /clips) ──────────────────────────────────────


class ClipIndex:
    """Thread-safe in-memory list of finalised clips.

    Populated at startup by scan_disk(), then by register() as new clips are
    cut. Pruned per-camera when MAX_CLIPS_PER_CAM is exceeded."""

    def __init__(self) -> None:
        self._items: list[dict] = []
        self._lock = threading.Lock()

    def list_all(self, cam_id: Optional[str] = None) -> list[dict]:
        with self._lock:
            items = list(self._items)
        if cam_id:
            items = [c for c in items if c["cam_id"] == cam_id]
        return list(reversed(items))   # newest first

    def register(self, cam_id: str, out_path: Path, trigger: dict) -> None:
        entry = {
            "cam_id":    cam_id,
            "filename":  out_path.name,
            "path":      str(out_path),
            "timestamp": datetime.fromtimestamp(
                trigger["first_detect_wall"], tz=timezone.utc
            ).isoformat(timespec="seconds"),
            "query":     trigger["query"],
            "url":       f"/cameras/clips/file/{cam_id}/{out_path.name}",
        }
        with self._lock:
            self._items.append(entry)
        self._prune(cam_id)

    def _prune(self, cam_id: str) -> None:
        cam_dir = CLIPS_DIR / cam_id
        try:
            clips = sorted(cam_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
            for old in clips[:-MAX_CLIPS_PER_CAM]:
                old.unlink(missing_ok=True)
                with self._lock:
                    self._items[:] = [c for c in self._items if c["path"] != str(old)]
        except Exception as exc:
            logger.warning("Clip prune error for %s: %s", cam_id, exc)

    def scan_disk(self) -> None:
        """Initial scan — call once at startup to populate from existing clips."""
        try:
            CLIPS_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        with self._lock:
            self._items.clear()
            try:
                for cam_dir in sorted(CLIPS_DIR.iterdir()):
                    if not cam_dir.is_dir():
                        continue
                    for mp4 in sorted(cam_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime):
                        stem_parts = mp4.stem.split("_", 2)
                        query = stem_parts[2].replace("_", " ") if len(stem_parts) >= 3 else ""
                        self._items.append({
                            "cam_id":    cam_dir.name,
                            "filename":  mp4.name,
                            "path":      str(mp4),
                            "timestamp": datetime.fromtimestamp(mp4.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
                            "query":     query,
                            "url":       f"/cameras/clips/file/{cam_dir.name}/{mp4.name}",
                        })
            except Exception as exc:
                logger.warning("Failed to scan clips dir: %s", exc)
            count = len(self._items)
        logger.info("Clip index loaded: %d clips", count)


clip_index = ClipIndex()


# ── ffmpeg picker ─────────────────────────────────────────────────────────────


def _ffmpeg_exe() -> Optional[str]:
    """Return an ffmpeg binary suitable for clip CUTTING (local-file work).

    The deepstream image's apt ffmpeg has libavcodec.so.58 missing on disk
    despite dpkg metadata. The imageio_ffmpeg static binary works fine for
    stream-copy of local .ts files — the WORKLOG segfault was specific to
    imageio_ffmpeg reading RTSP, which is now done in the inference container."""
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


# ── Trigger management (per-camera, called from /internal/trigger and /event) ─


def trigger_clip_on_detection(cam_id: str, query: str, wall_now: float) -> None:
    """Start or extend a clip trigger for this camera. Idempotent — safe to
    call on every per-detection trigger POST. No-op if the camera is not
    registered (e.g. cleanup race during DELETE /streams/{cam_id})."""
    cam = registry.get(cam_id)
    if cam is None:
        return
    started_new = False
    with cam.lock:
        if cam.clip_triggers:
            t = cam.clip_triggers[-1]
            if wall_now - t["first_detect_wall"] <= MAX_CLIP_DURATION_S:
                t["last_detect_wall"] = wall_now
                t["detection_count"] += 1
                return
        cam.clip_triggers.append({
            "query":             query,
            "first_detect_wall": wall_now,
            "last_detect_wall":  wall_now,
            "detection_count":   1,
        })
        started_new = True
    if started_new:
        logger.info("Clip trigger started: cam=%s query=%r", cam_id, query)


# ── Segment watcher (one thread, scans all cameras) ──────────────────────────


class _SegWatcher:
    """Background thread that polls each registered camera's segs/ directory
    every 5s and appends new completed segments to CameraRuntime.seg_ring."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="ds-seg-watcher",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=8)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(timeout=5)
            if self._stop.is_set():
                break
            self._scan()

    def _scan(self) -> None:
        for cam in registry.all():
            segs_dir = CLIPS_DIR / cam.cam_id / "segs"
            if not segs_dir.exists():
                continue
            try:
                seg_files = sorted(segs_dir.glob("seg_*.ts"),
                                   key=lambda p: p.stat().st_mtime)
            except Exception:
                continue
            with cam.lock:
                known = {s.path for s in cam.seg_ring}
            new_segs: list[SegBoundary] = []
            for seg_path in seg_files:
                path_str = str(seg_path)
                if path_str in known:
                    continue
                try:
                    mtime = seg_path.stat().st_mtime
                    size  = seg_path.stat().st_size
                except OSError:
                    continue
                if size < 1024:
                    continue
                new_segs.append(SegBoundary(
                    path=path_str,
                    start_wall=mtime - SEG_DURATION_S,
                    end_wall=mtime,
                ))
            if new_segs:
                with cam.lock:
                    cam.seg_ring.extend(new_segs)


seg_watcher = _SegWatcher()


# ── Clip manager (one thread, polls all cameras' triggers) ───────────────────


class _ClipManager:
    """Background thread that watches each camera's clip_triggers list and
    spawns a cutter daemon when a trigger expires."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="ds-clip-manager",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(timeout=1.0)
            if self._stop.is_set():
                break

            now_wall = time.time()
            ready: list[tuple[str, dict]] = []

            for cam in registry.all():
                with cam.lock:
                    active: list[dict] = []
                    for t in cam.clip_triggers:
                        if now_wall - t["last_detect_wall"] >= POST_BUFFER_S:
                            ready.append((cam.cam_id, t))
                        else:
                            active.append(t)
                    cam.clip_triggers = active

            for cam_id, trigger in ready:
                logger.info("Clip trigger expired: cam=%s query=%r detections=%d — cutting clip",
                            cam_id, trigger["query"], trigger["detection_count"])
                threading.Thread(
                    target=_process_trigger,
                    args=(cam_id, trigger),
                    daemon=True,
                    name=f"ds-clip-cut-{cam_id}",
                ).start()
        logger.info("Clip manager stopped")


clip_manager = _ClipManager()


# ── Cutter (per-trigger daemon thread) ──────────────────────────────────────


def _process_trigger(cam_id: str, trigger: dict) -> None:
    """Cut one clip from segment files. Runs in a daemon thread spawned by
    _ClipManager when a trigger expires."""
    if trigger["detection_count"] < MIN_CLIP_DETECTIONS:
        logger.info("Clip discarded: cam=%s query=%r detections=%d < min=%d",
                    cam_id, trigger["query"],
                    trigger["detection_count"], MIN_CLIP_DETECTIONS)
        return

    clip_end_wall = trigger["last_detect_wall"] + POST_BUFFER_S
    deadline      = clip_end_wall + SEG_DURATION_S + 5.0
    segs_dir      = CLIPS_DIR / cam_id / "segs"

    def _latest_finalized_end_wall() -> float:
        """Latest segment end mtime visible on disk OR in the ring. Disk
        fallback handles the case where the camera was removed mid-cut."""
        candidates: list[float] = []
        cam = registry.get(cam_id)
        if cam is not None:
            with cam.lock:
                candidates.extend(s.end_wall for s in cam.seg_ring)
        try:
            for p in segs_dir.glob("seg_*.ts"):
                try:
                    candidates.append(p.stat().st_mtime)
                except OSError:
                    continue
        except Exception:
            pass
        return max(candidates) if candidates else 0.0

    while time.time() < deadline:
        if _latest_finalized_end_wall() >= clip_end_wall:
            break
        time.sleep(2.0)

    out_path = _extract_and_save_clip(cam_id, trigger)
    if out_path:
        clip_index.register(cam_id, out_path, trigger)


def _extract_and_save_clip(cam_id: str, trigger: dict) -> Optional[Path]:
    """Cut a clip from rolling segment files via ffmpeg stream-copy."""
    ffmpeg = _ffmpeg_exe()
    if ffmpeg is None:
        logger.error("ffmpeg unavailable — clip cut for %s disabled", cam_id)
        return None

    query           = trigger["query"]
    first_wall      = trigger["first_detect_wall"]
    last_wall       = trigger["last_detect_wall"]
    clip_start_wall = first_wall - PRE_BUFFER_S
    clip_end_wall   = last_wall  + POST_BUFFER_S

    cam = registry.get(cam_id)
    if cam is not None:
        with cam.lock:
            ring: list[SegBoundary] = list(cam.seg_ring)
    else:
        ring = []

    # Disk fallback: the watcher may be behind, or the camera may have been
    # removed between trigger fire and this cut.
    if not ring:
        segs_dir = CLIPS_DIR / cam_id / "segs"
        try:
            disk_files = sorted(segs_dir.glob("seg_*.ts"),
                                key=lambda p: p.stat().st_mtime)
        except Exception:
            disk_files = []
        for seg_path in disk_files:
            try:
                mtime = seg_path.stat().st_mtime
                size  = seg_path.stat().st_size
            except OSError:
                continue
            if size < 1024:
                continue
            ring.append(SegBoundary(
                path=str(seg_path),
                start_wall=mtime - SEG_DURATION_S,
                end_wall=mtime,
            ))

    segs = [s for s in ring
            if s.end_wall > clip_start_wall and s.start_wall < clip_end_wall]
    segs.sort(key=lambda s: s.start_wall)
    segs = [s for s in segs if Path(s.path).exists()]

    if not segs:
        logger.warning("Clip cut skipped: no segment files available for cam=%s query=%r",
                       cam_id, query)
        return None

    actual_start = max(clip_start_wall, segs[0].start_wall)
    actual_end   = min(clip_end_wall,   segs[-1].end_wall)
    if actual_end <= actual_start:
        logger.warning("Clip cut skipped: empty time window for cam=%s", cam_id)
        return None

    out_dir = CLIPS_DIR / cam_id
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.error("Cannot create clips dir %s: %s", out_dir, exc)
        return None

    ts_str   = datetime.fromtimestamp(first_wall).strftime("%Y%m%d_%H%M%S")
    safe_q   = "".join(c if c.isalnum() or c in "-_" else "_" for c in query)
    out_path = out_dir / f"{ts_str}_{safe_q}.mp4"
    concat_file: Optional[Path] = None

    try:
        if len(segs) == 1:
            seg = segs[0]
            ss  = max(0.0, actual_start - seg.start_wall)
            dur = actual_end - actual_start
            cmd = [ffmpeg, "-y",
                   "-ss", f"{ss:.3f}", "-i", seg.path,
                   "-t", f"{dur:.3f}",
                   "-c", "copy", "-movflags", "+faststart",
                   str(out_path)]
        else:
            concat_file = out_dir / f".concat_{ts_str}.txt"
            with open(concat_file, "w") as cf:
                for seg in segs:
                    cf.write(f"file '{seg.path}'\n")
            ss  = max(0.0, actual_start - segs[0].start_wall)
            dur = actual_end - actual_start
            cmd = [ffmpeg, "-y",
                   "-f", "concat", "-safe", "0", "-i", str(concat_file),
                   "-ss", f"{ss:.3f}", "-t", f"{dur:.3f}",
                   "-c", "copy", "-movflags", "+faststart",
                   str(out_path)]

        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=120,
        )
        if result.returncode != 0:
            logger.error("ffmpeg clip cut failed for cam=%s (rc=%d): %s",
                         cam_id, result.returncode,
                         result.stderr.decode(errors="replace")[-300:])
            return None
    except Exception as exc:
        logger.error("Clip cut error for cam=%s: %s", cam_id, exc)
        return None
    finally:
        if concat_file and concat_file.exists():
            concat_file.unlink(missing_ok=True)

    if not out_path.exists() or out_path.stat().st_size == 0:
        logger.error("Clip file empty after cut: %s", out_path)
        return None

    logger.info("Clip saved: cam=%s query=%r path=%s segs=%d dur=%.1fs",
                cam_id, query, out_path.name, len(segs), actual_end - actual_start)
    return out_path
