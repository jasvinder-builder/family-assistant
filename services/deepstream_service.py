"""
DeepStream service — control plane + clip recording + REST API.

Phase 1 step B: ingress and inference live elsewhere now.
  - MediaMTX (bianca-mediamtx) is the single RTSP/HLS hub. Browser pulls
    LL-HLS from :8888 via main.py's /cameras/hls/* proxy. Inference workers
    pull RTSP from :8554.
  - The inference container (bianca-inference, port 8091) owns one
    subprocess per camera; the subprocess does NVDEC decode + Triton
    inference + tracking + event firing, and POSTs each event back here.

This service still owns:
  - cameras.json (the camera registry — source of truth)
  - The events deque + /events API
  - Rolling segment recording from MediaMTX RTSP (system ffmpeg)
  - Clip cutting on detection windows + /clips API
  - Queries / threshold / ROI / cameras CRUD via the public REST API

Public stream management:
  add_stream(cam_id, uri) / remove_stream(cam_id) / get_streams()
  get_queries / add_query / remove_query / set_roi
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

# MediaMTX HTTP API (path mgmt) and RTSP re-stream (clip recorder source)
MEDIAMTX_API_URL  = os.environ.get("MEDIAMTX_API_URL",  "http://bianca-mediamtx:9997")
MEDIAMTX_RTSP_URL = os.environ.get("MEDIAMTX_RTSP_URL", "rtsp://bianca-mediamtx:8554")

# Inference control plane (subprocess-per-camera)
INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://bianca-inference:8091")

META_JSON_PATH    = os.environ.get("META_JSON_PATH",    "/app/models/yoloworld.meta.json")
CAMERAS_JSON_PATH = os.environ.get("CAMERAS_JSON_PATH", "./cameras.json")

CLIPS_DIR           = Path(os.environ.get("CLIPS_DIR", "./clips"))
SEG_DURATION_S      = int(os.environ.get("SEG_DURATION_S", "30"))    # rolling segment length
MAX_SEG_FILES       = int(os.environ.get("MAX_SEG_FILES", "25"))
PRE_BUFFER_S        = int(os.environ.get("PRE_BUFFER_S", "5"))
POST_BUFFER_S       = int(os.environ.get("POST_BUFFER_S", "5"))
MAX_CLIPS_PER_CAM   = int(os.environ.get("MAX_CLIPS_PER_CAM", "100"))
MAX_CLIP_DURATION_S = int(os.environ.get("MAX_CLIP_DURATION_S", "60"))
MIN_CLIP_DETECTIONS = int(os.environ.get("MIN_CLIP_DETECTIONS", "5"))


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class CameraEvent:
    timestamp:  str
    query:      str
    track_id:   int
    confidence: float
    image_b64:  str
    cam_id:     str = "cam0"


@dataclass
class _SegBoundary:
    """Metadata for one completed rolling segment file."""
    path:       str    # absolute path to the .ts segment file
    start_wall: float  # time.time() when this segment started recording
    end_wall:   float  # time.time() when this segment ended (next segment started)


# ── Module-level state ────────────────────────────────────────────────────────

_pipeline_lock = threading.Lock()
_rebuild_lock  = threading.Lock()      # serialises concurrent _rebuild calls
_streams: dict[str, str] = {}          # cam_id → uri (ordered by insertion)
_rois:    dict[str, Optional[dict]] = {}  # cam_id → {x,y,w,h} | None

# Segment watcher thread: polls clips/ directory to update _seg_ring
_seg_watcher_stop   = threading.Event()
_seg_watcher_thread: Optional[threading.Thread] = None

_events:     deque = deque(maxlen=500)
_events_lock = threading.Lock()

# Segment ring buffer (disk-based rolling segments from ffmpeg)
_seg_ring:      dict[str, deque] = {}   # cam_id → deque[_SegBoundary], maxlen=20
_seg_ring_lock  = threading.Lock()

# Pending clip triggers: each entry represents one detection event window.
_clip_triggers:      dict[str, list] = {}
_clip_triggers_lock  = threading.Lock()

# Finalised clip index
_clip_index:      list[dict] = []
_clip_index_lock  = threading.Lock()

_clip_manager_stop    = threading.Event()
_clip_manager_thread: Optional[threading.Thread] = None

_queries:      list[str] = []
_queries_lock  = threading.Lock()

_threshold:      float = 0.3
_debug_overlay:  bool  = False

_query_update_lock  = threading.Lock()


# ── MediaMTX path management ─────────────────────────────────────────────────

def _mediamtx_path_url(cam_id: str, action: str) -> str:
    return f"{MEDIAMTX_API_URL}/v3/config/paths/{action}/{urllib.parse.quote(cam_id, safe='')}"


def _mediamtx_post(cam_id: str, action: str, body: Optional[dict]) -> bool:
    url = _mediamtx_path_url(cam_id, action)
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=5).close()
        return True
    except Exception as exc:
        logger.warning("MediaMTX %s %s failed: %s", action, cam_id, exc)
        return False


def _mediamtx_add_path(cam_id: str, source: str) -> bool:
    """Register a path that pulls from `source` and re-streams on RTSP/HLS/WebRTC.
    Idempotent: if the path already exists we delete and re-add to apply the new source."""
    # Field shape matches mediamtx.yml `paths:` entries (PathConf).
    # rtspTransport=tcp matches the camera RTSP transport used by every camera
    # in this project; keeping sourceOnDemand=false so the inference worker can
    # connect even before a viewer is watching.
    body = {
        "source":         source,
        "sourceOnDemand": False,
        "rtspTransport":  "tcp",
    }
    if _mediamtx_post(cam_id, "add", body):
        return True
    _mediamtx_post(cam_id, "delete", None)
    return _mediamtx_post(cam_id, "add", body)


def _mediamtx_remove_path(cam_id: str) -> None:
    _mediamtx_post(cam_id, "delete", None)


# ── Inference control plane ─────────────────────────────────────────────────

def _inference_add_camera(cam_id: str) -> bool:
    """Tell the inference service to spawn a worker subprocess for this camera.
    The worker pulls RTSP from MediaMTX's re-stream, not from the original source."""
    rtsp_url = f"{MEDIAMTX_RTSP_URL}/{urllib.parse.quote(cam_id, safe='')}"
    body = json.dumps({"cam_id": cam_id, "rtsp_url": rtsp_url}).encode()
    req = urllib.request.Request(
        f"{INFERENCE_URL}/cameras", data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=10).close()
        return True
    except Exception as exc:
        logger.warning("Inference add_camera %s failed: %s", cam_id, exc)
        return False


def _inference_remove_camera(cam_id: str) -> None:
    req = urllib.request.Request(
        f"{INFERENCE_URL}/cameras/{urllib.parse.quote(cam_id, safe='')}",
        method="DELETE",
    )
    try:
        urllib.request.urlopen(req, timeout=10).close()
    except Exception as exc:
        logger.debug("Inference remove_camera %s: %s", cam_id, exc)


# ── Clip recording from MediaMTX RTSP ────────────────────────────────────────

def _ffmpeg_exe() -> Optional[str]:
    """Return an ffmpeg binary suitable for clip CUTTING (local-file work).

    Uses the imageio_ffmpeg bundled static binary by default — the deepstream
    base image's apt ffmpeg is broken (libavcodec.so.58 missing despite dpkg
    metadata), and the WORKLOG's segfault was specific to imageio_ffmpeg
    *reading RTSP*, which is now done by the inference container's system
    ffmpeg, not here.  System ffmpeg is preferred only if it actually runs."""
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


# Clip recording (rolling .ts segments) is owned by the inference container —
# its savant-deepstream base has a working system ffmpeg, while the deepstream
# image's apt ffmpeg is missing libavcodec.so.58.  We still scan + cut +
# register clips here because that's pure disk work and keeps the public
# /clips API in one place.


# ── Segment watcher — polls clips/ directory to update _seg_ring ─────────────

def _segment_watcher_loop() -> None:
    """Poll each camera's segs/ directory every 5s to detect new completed segments."""
    while not _seg_watcher_stop.is_set():
        _seg_watcher_stop.wait(timeout=5)
        if _seg_watcher_stop.is_set():
            break
        _scan_segments()


def _scan_segments() -> None:
    """Scan all active camera segment directories and populate _seg_ring."""
    for cam_id in list(_streams):
        segs_dir = CLIPS_DIR / cam_id / "segs"
        if not segs_dir.exists():
            continue
        try:
            seg_files = sorted(segs_dir.glob("seg_*.ts"), key=lambda p: p.stat().st_mtime)
        except Exception:
            continue
        with _seg_ring_lock:
            ring = _seg_ring.setdefault(cam_id, deque(maxlen=MAX_SEG_FILES))
            known = {s.path for s in ring}
        new_segs = []
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
                continue  # skip empty/incomplete segments
            new_segs.append(_SegBoundary(
                path=path_str,
                start_wall=mtime - SEG_DURATION_S,
                end_wall=mtime,
            ))
        if new_segs:
            with _seg_ring_lock:
                for seg in new_segs:
                    ring.append(seg)


# ── Queries / meta.json ───────────────────────────────────────────────────────

def _load_queries_from_meta() -> list[str]:
    try:
        return json.loads(open(META_JSON_PATH).read())["queries"]
    except Exception:
        return ["person", "car", "dog"]


def _save_queries_to_meta(new_queries: list[str]) -> None:
    try:
        existing = json.loads(open(META_JSON_PATH).read())
    except Exception:
        existing = {"queries": [], "imgsz": 640}
    existing["queries"] = new_queries
    tmp = META_JSON_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(existing, f, indent=2)
    os.replace(tmp, META_JSON_PATH)


def _commit_queries(new_queries: list[str]) -> None:
    """Update in-memory query list and persist to meta.json.
    Called by add_query/remove_query immediately, and also by main.py after
    a successful TRT re-export to keep the local label-mapping in sync."""
    with _query_update_lock:
        _queries.clear()
        _queries.extend(new_queries)
        _save_queries_to_meta(new_queries)
    logger.info("Queries committed: %s", new_queries)


# ── Camera persistence (cameras.json) ────────────────────────────────────────

def _load_cameras() -> tuple[dict[str, str], dict[str, Optional[dict]]]:
    try:
        with open(CAMERAS_JSON_PATH) as f:
            data = json.load(f)
        streams: dict[str, str] = {}
        rois:    dict[str, Optional[dict]] = {}
        for k, v in data.items():
            if not isinstance(k, str):
                continue
            if isinstance(v, dict) and "url" in v:
                streams[k] = v["url"]
                rois[k]    = v.get("roi")
        return streams, rois
    except FileNotFoundError:
        return {}, {}
    except Exception as exc:
        logger.warning("Failed to load cameras.json: %s", exc)
        return {}, {}


def _save_cameras(streams: dict[str, str], rois: dict[str, Optional[dict]]) -> None:
    """Write camera list to cameras.json. Note: atomic tmp+rename fails with
    EBUSY on Docker bind-mounted single files; write directly instead."""
    data = {cam_id: {"url": uri, "roi": rois.get(cam_id)} for cam_id, uri in streams.items()}
    try:
        with open(CAMERAS_JSON_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logger.warning("Failed to save cameras.json: %s", exc)


# ── Clip recording helpers ────────────────────────────────────────────────────

def _trigger_clip_on_detection(cam_id: str, query: str, wall_now: float) -> None:
    """Start or extend a clip trigger for this camera. Called when an event arrives
    via /internal/event from an inference worker subprocess."""
    with _clip_triggers_lock:
        triggers = _clip_triggers.setdefault(cam_id, [])
        if triggers:
            t = triggers[-1]
            if wall_now - t["first_detect_wall"] <= MAX_CLIP_DURATION_S:
                t["last_detect_wall"] = wall_now
                t["detection_count"] += 1
                return
        triggers.append({
            "query":             query,
            "first_detect_wall": wall_now,
            "last_detect_wall":  wall_now,
            "detection_count":   1,
        })
        logger.info("Clip trigger started: cam=%s query=%r", cam_id, query)


def _extract_and_save_clip(cam_id: str, trigger: dict) -> Optional[Path]:
    """Cut a clip from rolling segment files using ffmpeg stream-copy (zero re-encode)."""
    ffmpeg = _ffmpeg_exe()
    if ffmpeg is None:
        logger.error("ffmpeg unavailable — clip cut for %s disabled", cam_id)
        return None

    query           = trigger["query"]
    first_wall      = trigger["first_detect_wall"]
    last_wall       = trigger["last_detect_wall"]
    clip_start_wall = first_wall - PRE_BUFFER_S
    clip_end_wall   = last_wall  + POST_BUFFER_S

    with _seg_ring_lock:
        ring = list(_seg_ring.get(cam_id, []))

    # Disk fallback: if the ring was cleared (e.g. camera removed mid-cut)
    # or the watcher is behind, derive boundaries directly from on-disk
    # mtimes.  Same heuristic as _scan_segments.
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
            ring.append(_SegBoundary(
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
            seg    = segs[0]
            ss     = max(0.0, actual_start - seg.start_wall)
            dur    = actual_end - actual_start
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


def _prune_clips(cam_id: str) -> None:
    cam_dir = CLIPS_DIR / cam_id
    try:
        clips = sorted(cam_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        for old in clips[:-MAX_CLIPS_PER_CAM]:
            old.unlink(missing_ok=True)
            with _clip_index_lock:
                _clip_index[:] = [c for c in _clip_index if c["path"] != str(old)]
    except Exception as exc:
        logger.warning("Clip prune error for %s: %s", cam_id, exc)


def _register_clip(cam_id: str, out_path: Path, trigger: dict) -> None:
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
    with _clip_index_lock:
        _clip_index.append(entry)
    _prune_clips(cam_id)


def _process_trigger(cam_id: str, trigger: dict) -> None:
    """Worker function run in a daemon thread to cut one clip from segments."""
    if trigger["detection_count"] < MIN_CLIP_DETECTIONS:
        logger.info("Clip discarded: cam=%s query=%r detections=%d < min=%d",
                    cam_id, trigger["query"],
                    trigger["detection_count"], MIN_CLIP_DETECTIONS)
        return

    clip_end_wall = trigger["last_detect_wall"] + POST_BUFFER_S
    deadline = clip_end_wall + SEG_DURATION_S + 5.0

    segs_dir = CLIPS_DIR / cam_id / "segs"

    def _latest_finalized_end_wall() -> float:
        """Latest segment end mtime visible on disk OR in the ring.  A segment is
        'finalized' once a newer file exists alongside it (its mtime is fixed)."""
        candidates: list[float] = []
        with _seg_ring_lock:
            candidates.extend(s.end_wall for s in _seg_ring.get(cam_id, []))
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
        _register_clip(cam_id, out_path, trigger)


def _clip_manager() -> None:
    while not _clip_manager_stop.is_set():
        _clip_manager_stop.wait(timeout=1.0)
        if _clip_manager_stop.is_set():
            break

        now_wall = time.time()
        ready: list[tuple[str, dict]] = []

        with _clip_triggers_lock:
            for cam_id in list(_clip_triggers):
                active = []
                for t in _clip_triggers[cam_id]:
                    if now_wall - t["last_detect_wall"] >= POST_BUFFER_S:
                        ready.append((cam_id, t))
                    else:
                        active.append(t)
                _clip_triggers[cam_id] = active
                if not active:
                    del _clip_triggers[cam_id]

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


def _start_clip_manager() -> None:
    global _clip_manager_thread
    _clip_manager_stop.clear()
    _clip_manager_thread = threading.Thread(
        target=_clip_manager, daemon=True, name="ds-clip-manager"
    )
    _clip_manager_thread.start()


def _stop_clip_manager() -> None:
    global _clip_manager_thread
    _clip_manager_stop.set()
    if _clip_manager_thread and _clip_manager_thread.is_alive():
        _clip_manager_thread.join(timeout=5)
    _clip_manager_thread = None


def _start_seg_watcher() -> None:
    global _seg_watcher_thread
    _seg_watcher_stop.clear()
    _seg_watcher_thread = threading.Thread(
        target=_segment_watcher_loop, daemon=True, name="ds-seg-watcher"
    )
    _seg_watcher_thread.start()


def _stop_seg_watcher() -> None:
    global _seg_watcher_thread
    _seg_watcher_stop.set()
    if _seg_watcher_thread and _seg_watcher_thread.is_alive():
        _seg_watcher_thread.join(timeout=8)
    _seg_watcher_thread = None


def _init_clip_index() -> None:
    """Scan clips directory on startup to populate the in-memory index."""
    try:
        CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        for cam_dir in sorted(CLIPS_DIR.iterdir()):
            if not cam_dir.is_dir():
                continue
            for mp4 in sorted(cam_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime):
                stem_parts = mp4.stem.split("_", 2)
                query = stem_parts[2].replace("_", " ") if len(stem_parts) >= 3 else ""
                _clip_index.append({
                    "cam_id":    cam_dir.name,
                    "filename":  mp4.name,
                    "path":      str(mp4),
                    "timestamp": datetime.fromtimestamp(mp4.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
                    "query":     query,
                    "url":       f"/cameras/clips/file/{cam_dir.name}/{mp4.name}",
                })
    except Exception as exc:
        logger.warning("Failed to scan clips dir: %s", exc)
    logger.info("Clip index loaded: %d clips", len(_clip_index))


# ── Stream lifecycle ─────────────────────────────────────────────────────────

def _start_camera(cam_id: str, uri: str) -> None:
    """Wire up MediaMTX path + inference subprocess (which also spawns the
    sibling ffmpeg clip recorder) for one camera."""
    _mediamtx_add_path(cam_id, uri)
    _inference_add_camera(cam_id)


def _stop_camera(cam_id: str) -> None:
    _inference_remove_camera(cam_id)
    _mediamtx_remove_path(cam_id)


def _rebuild(new_streams: dict[str, str]) -> None:
    """Reconcile to the desired stream set. Currently a stop-everything-then-restart
    pass for simplicity; Phase 2 plan replaces this with per-camera CameraRuntime
    objects that don't require global teardown."""
    global _streams

    if not _rebuild_lock.acquire(blocking=False):
        logger.debug("_rebuild: already in progress, skipping concurrent call")
        return

    try:
        _stop_clip_manager()
        _stop_seg_watcher()

        for cam_id in list(_streams):
            _stop_camera(cam_id)

        removed = set(_streams) - set(new_streams)
        for cam_id in removed:
            with _seg_ring_lock:
                _seg_ring.pop(cam_id, None)
            with _clip_triggers_lock:
                _clip_triggers.pop(cam_id, None)
            _rois.pop(cam_id, None)

        _streams = dict(new_streams)

        if _streams:
            for cam_id, uri in _streams.items():
                _start_camera(cam_id, uri)
            _start_seg_watcher()
            _start_clip_manager()
            logger.info("DeepStream service running: %s", list(_streams.keys()))
        else:
            logger.info("DeepStream service idle — no streams")
    finally:
        _rebuild_lock.release()


# ── Public stream management ──────────────────────────────────────────────────

def add_stream(cam_id: str, uri: str) -> None:
    with _pipeline_lock:
        new = dict(_streams)
        new[cam_id] = uri
        _rebuild(new)
        _save_cameras(_streams, _rois)


def remove_stream(cam_id: str) -> None:
    with _pipeline_lock:
        new = {k: v for k, v in _streams.items() if k != cam_id}
        _rebuild(new)
        _save_cameras(_streams, _rois)


def set_roi(cam_id: str, roi: Optional[dict]) -> None:
    """Set or clear the inference ROI for a camera. Phase 1 step B note: the new
    inference worker does not yet honour ROI; this stores it for Phase 2."""
    with _pipeline_lock:
        if cam_id not in _streams:
            raise KeyError(cam_id)
        _rois[cam_id] = roi
        _save_cameras(_streams, _rois)


def get_streams() -> dict[str, dict]:
    return {
        cam_id: {"url": uri, "roi": _rois.get(cam_id)}
        for cam_id, uri in _streams.items()
    }


# ── Detection / event getters ─────────────────────────────────────────────────

def get_latest_detections(cam_id: Optional[str] = None) -> list:
    """Live overlay detections moved into the inference subprocess (Phase 1).
    Kept as an empty stub so any leftover callers stay quiet until /diag arrives in Phase 3."""
    return []


def get_events(max_age_hours: float = 1.0,
               cam_id: Optional[str] = None) -> list[dict]:
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    with _events_lock:
        return [
            {
                "timestamp":  e.timestamp,
                "query":      e.query,
                "confidence": round(e.confidence, 3),
                "image_b64":  e.image_b64,
                "cam_id":     e.cam_id,
            }
            for e in _events
            if (datetime.fromisoformat(e.timestamp) >= cutoff
                and (cam_id is None or e.cam_id == cam_id))
        ]


# ── Query management ──────────────────────────────────────────────────────────

def get_queries() -> list[str]:
    with _queries_lock:
        return list(_queries)


def add_query(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    with _queries_lock:
        if text in _queries:
            return False
        new = list(_queries) + [text]
    _commit_queries(new)
    return True


def remove_query(index: int) -> bool:
    with _queries_lock:
        if not (0 <= index < len(_queries)):
            return False
        new = [q for i, q in enumerate(_queries) if i != index]
    _commit_queries(new)
    return True


# ── Threshold / overlay ───────────────────────────────────────────────────────

def get_threshold() -> float:
    return _threshold


def set_threshold(value: float) -> None:
    global _threshold
    _threshold = max(0.0, min(1.0, value))


def get_debug_overlay() -> bool:
    return _debug_overlay


def set_debug_overlay(enabled: bool) -> None:
    global _debug_overlay
    _debug_overlay = enabled


def get_pad_factor() -> float:
    return 0.0


def set_pad_factor(_: float) -> None:
    pass


# ── Backward-compatible single-camera API ────────────────────────────────────

def set_stream_url(url: str) -> None:
    url = url.strip() if url else ""
    if url:
        add_stream("cam0", url)
    else:
        remove_stream("cam0")


def get_stream_url() -> Optional[str]:
    return _streams.get("cam0")


def start_analysis(url: str) -> None:
    add_stream("cam0", url)


def stop_analysis() -> None:
    remove_stream("cam0")


def push_frame(frame) -> None:
    """No-op — inference container reads RTSP directly. Kept for API compat."""
    pass


# ── Startup: load persisted queries, cameras, and clip index ─────────────────

def _init_queries() -> None:
    persisted = _load_queries_from_meta()
    with _queries_lock:
        if not _queries:
            _queries.extend(persisted)


def _init_cameras() -> None:
    cameras, rois = _load_cameras()
    for cam_id, uri in cameras.items():
        logger.info("Restoring camera from cameras.json: %s → %s (roi=%s)",
                    cam_id, uri, rois.get(cam_id))
        _rois[cam_id] = rois.get(cam_id)
        add_stream(cam_id, uri)


_init_queries()
_init_clip_index()
_init_cameras()


# ── FastAPI app (served by the deepstream container on port 8090) ─────────────

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse as _JSONResponse  # noqa: E402

app = FastAPI(title="DeepStream Service")


@app.get("/health")
async def _health():
    return {"ok": True}


@app.post("/internal/trigger")
async def _internal_trigger(payload: dict):
    """Per-detection trigger ping from an inference worker.  Extends or starts
    the clip-recording window for this camera.  Called at the worker's frame
    rate (≤INFER_FPS); does NOT append to /events (events are rate-limited and
    posted via /internal/event)."""
    cam_id = (payload.get("cam_id") or "").strip()
    query  = (payload.get("query")  or "").strip()
    if not cam_id or not query:
        return _JSONResponse({"error": "cam_id and query required"}, status_code=400)
    _trigger_clip_on_detection(cam_id, query, time.time())
    return _JSONResponse({"ok": True})


@app.post("/internal/event")
async def _internal_event(payload: dict):
    """Receive a (rate-limited) detection event from an inference worker subprocess
    and fold it into the events deque.  Also extends the clip-trigger window so
    a single rare event still produces a clip if MIN_CLIP_DETECTIONS allows."""
    cam_id = (payload.get("cam_id") or "").strip()
    query  = (payload.get("query")  or "").strip()
    if not cam_id or not query:
        return _JSONResponse({"error": "cam_id and query required"}, status_code=400)

    try:
        track_id = int(payload.get("track_id") or 0)
        confidence = float(payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return _JSONResponse({"error": "track_id and confidence must be numeric"}, status_code=400)

    image_b64 = str(payload.get("image_b64") or "")
    ts        = str(payload.get("timestamp")
                    or datetime.now().isoformat(timespec="seconds"))

    with _events_lock:
        _events.append(CameraEvent(
            timestamp=ts, query=query, track_id=track_id,
            confidence=confidence, image_b64=image_b64, cam_id=cam_id,
        ))
    _trigger_clip_on_detection(cam_id, query, time.time())
    return _JSONResponse({"ok": True})


@app.get("/streams")
async def _list_streams():
    return _JSONResponse({"streams": get_streams()})


@app.post("/streams")
async def _add_stream(payload: dict):
    cam_id = payload.get("cam_id", "").strip()
    url    = payload.get("url", "").strip()
    if not cam_id:
        return _JSONResponse({"error": "cam_id is required"}, status_code=400)
    if not url:
        return _JSONResponse({"error": "url is required"}, status_code=400)
    await asyncio.to_thread(add_stream, cam_id, url)
    return _JSONResponse({"ok": True, "streams": get_streams()})


@app.delete("/streams/{cam_id}")
async def _remove_stream(cam_id: str):
    if cam_id not in _streams:
        return _JSONResponse({"error": f"stream '{cam_id}' not found"}, status_code=404)
    await asyncio.to_thread(remove_stream, cam_id)
    return _JSONResponse({"ok": True, "streams": get_streams()})


@app.patch("/streams/{cam_id}/roi")
async def _set_roi(cam_id: str, payload: dict):
    if cam_id not in _streams:
        return _JSONResponse({"error": f"stream '{cam_id}' not found"}, status_code=404)
    roi = None
    if payload.get("roi") is not None:
        roi = payload["roi"]
    elif all(k in payload for k in ("x", "y", "w", "h")):
        roi = {k: int(payload[k]) for k in ("x", "y", "w", "h")}
        if "frame_w" in payload and "frame_h" in payload:
            roi["frame_w"] = int(payload["frame_w"])
            roi["frame_h"] = int(payload["frame_h"])
    try:
        await asyncio.to_thread(set_roi, cam_id, roi)
    except KeyError:
        return _JSONResponse({"error": f"stream '{cam_id}' not found"}, status_code=404)
    logger.info("ROI updated: cam=%s roi=%s", cam_id, roi)
    return _JSONResponse({"ok": True, "cam_id": cam_id, "roi": roi, "streams": get_streams()})


@app.get("/queries")
async def _list_queries():
    return _JSONResponse({"queries": get_queries()})


@app.post("/queries")
async def _add_query(payload: dict):
    text = payload.get("text", "").strip()
    if not text:
        return _JSONResponse({"error": "text is required"}, status_code=400)
    added = add_query(text)
    return _JSONResponse({"ok": True, "added": added, "queries": get_queries()})


@app.delete("/queries/{index}")
async def _remove_query(index: int):
    removed = remove_query(index)
    if not removed:
        return _JSONResponse({"error": "index out of range"}, status_code=404)
    return _JSONResponse({"ok": True, "queries": get_queries()})


@app.get("/queries/status")
async def _queries_status_route():
    return _JSONResponse({"state": "ready", "eta_s": 0, "queries": get_queries()})


@app.post("/queries/commit")
async def _queries_commit_route(payload: dict):
    queries = payload.get("queries", [])
    if not isinstance(queries, list):
        return _JSONResponse({"error": "queries must be a list"}, status_code=400)
    _commit_queries(queries)
    return _JSONResponse({"ok": True, "queries": get_queries()})


@app.get("/events")
async def _events_route():
    return _JSONResponse({"events": get_events()})


@app.get("/threshold")
async def _get_threshold():
    return _JSONResponse({"threshold": get_threshold()})


@app.post("/threshold")
async def _set_threshold(payload: dict):
    try:
        value = float(payload.get("value"))
    except (TypeError, ValueError):
        return _JSONResponse({"error": "value must be a number"}, status_code=400)
    set_threshold(value)
    return _JSONResponse({"ok": True, "threshold": get_threshold()})


@app.post("/debug-overlay")
async def _debug_overlay_route(payload: dict):
    enabled = bool(payload.get("enabled", False))
    set_debug_overlay(enabled)
    return _JSONResponse({"ok": True, "enabled": enabled})


@app.post("/set-stream")
async def _set_stream(payload: dict):
    url = payload.get("url", "").strip()
    set_stream_url(url)
    return _JSONResponse({"ok": True})


@app.get("/clips")
async def _list_clips(cam_id: Optional[str] = None):
    with _clip_index_lock:
        clips = list(_clip_index)
    if cam_id:
        clips = [c for c in clips if c["cam_id"] == cam_id]
    return _JSONResponse({"clips": list(reversed(clips))})  # newest first


@app.get("/clips/file/{cam_id}/{filename}")
async def _serve_clip(cam_id: str, filename: str):
    if "/" in cam_id or "/" in filename or ".." in cam_id or ".." in filename:
        return _JSONResponse({"error": "invalid path"}, status_code=400)
    path = CLIPS_DIR / cam_id / filename
    if not path.exists():
        return _JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(path), media_type="video/mp4")
