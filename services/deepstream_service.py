"""
DeepStream service — replaces camera_service.py + scene_service.py.

Handles:
  - RTSP/file ingestion via NVDEC hardware decode (DeepStream 7.0)
  - Multi-camera batching via nvstreammux + nvmultistreamtiler
  - Frame broadcast to WebSocket/MJPEG consumers (25fps cap)
  - YOLO-World TRT inference via Triton HTTP at ~10fps per camera
  - Per-camera IoU tracking + event firing
  - Query management (update meta.json + Triton unload/load)

Public API is backward-compatible with camera_service + scene_service:
  set_stream_url / get_stream_url / ws_frame_generator / mjpeg_generator
  get_queries / add_query / remove_query
  get_threshold / set_threshold / get_debug_overlay / set_debug_overlay
  get_latest_detections / get_events / start_analysis / stop_analysis

New multi-camera additions:
  add_stream(cam_id, uri) / remove_stream(cam_id) / get_streams()
"""

import asyncio
import base64
import json
import logging
import os
import queue
import subprocess
import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

# Derive Triton HTTP host from TRITON_URL env var (default port gRPC 8001 → HTTP 8002)
_triton_raw  = os.environ.get("TRITON_URL", "localhost:8001")
_triton_host = _triton_raw.split(":")[0]
TRITON_HTTP_URL  = f"{_triton_host}:8002"

META_JSON_PATH    = os.environ.get("META_JSON_PATH", "/app/models/yoloworld.meta.json")
CAMERAS_JSON_PATH  = os.environ.get("CAMERAS_JSON_PATH", "./cameras.json")
CLIPS_DIR          = Path(os.environ.get("CLIPS_DIR", "./clips"))
PRE_BUFFER_S       = int(os.environ.get("PRE_BUFFER_S", "5"))
POST_BUFFER_S      = int(os.environ.get("POST_BUFFER_S", "5"))
MAX_CLIPS_PER_CAM  = int(os.environ.get("MAX_CLIPS_PER_CAM", "100"))
CLIP_FPS           = int(os.environ.get("CLIP_FPS", "25"))      # clip recording FPS (matches display pipeline)
MAX_CLIP_DURATION_S = int(os.environ.get("MAX_CLIP_DURATION_S", "60"))  # hard cap per clip chunk

MOTION_GATE       = os.environ.get("MOTION_GATE", "true").lower() == "true"
MOTION_THRESHOLD  = float(os.environ.get("MOTION_THRESHOLD", "2.0"))  # mean pixel diff (0-255) to trigger inference
MOTION_SCALE      = (320, 180)   # downscale to this for cheap frame differencing

DISPLAY_FPS       = 25
INFER_FPS         = 10       # max inference calls per camera per second
RECHECK_INTERVAL_S = 30      # minimum seconds between events for same (track, query)
TRACK_PRUNE_AGE_S  = 300
CAM_WIDTH         = 1280
CAM_HEIGHT        = 720


# ── IoU tracker (zero extra deps, CPU-only) ───────────────────────────────────

class _SimpleTracker:
    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 10):
        self._iou_thr  = iou_threshold
        self._max_lost = max_lost
        self._next_id  = 1
        self._tracks: dict[int, dict] = {}

    @staticmethod
    def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
        iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
        ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
        iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
        inter  = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        union  = area_a[:, None] + area_b[None, :] - inter
        return np.where(union > 0, inter / union, 0.0)

    def update(self, boxes, scores, class_ids):
        _empty = (
            np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
            np.zeros((0,), int),          np.zeros((0,), int),
        )
        for t in self._tracks.values():
            t["lost"] += 1

        if len(boxes) == 0:
            self._tracks = {k: v for k, v in self._tracks.items()
                            if v["lost"] <= self._max_lost}
            return _empty

        out_b, out_s, out_c, out_t = [], [], [], []
        tids      = list(self._tracks)
        matched_t = set()
        matched_d = set()

        if tids:
            t_boxes = np.array([self._tracks[i]["box"] for i in tids], np.float32)
            iou     = self._iou_matrix(t_boxes, boxes)
            ti_arr, di_arr = np.where(iou >= self._iou_thr)
            if len(ti_arr):
                order = np.argsort(-iou[ti_arr, di_arr])
                for idx in order:
                    ti, di = int(ti_arr[idx]), int(di_arr[idx])
                    if ti in matched_t or di in matched_d:
                        continue
                    matched_t.add(ti); matched_d.add(di)
                    tid = tids[ti]
                    self._tracks[tid] = {
                        "box": boxes[di].tolist(), "class_id": int(class_ids[di]), "lost": 0
                    }
                    out_b.append(boxes[di]); out_s.append(scores[di])
                    out_c.append(class_ids[di]); out_t.append(tid)

        for di in range(len(boxes)):
            if di in matched_d:
                continue
            tid = self._next_id; self._next_id += 1
            self._tracks[tid] = {
                "box": boxes[di].tolist(), "class_id": int(class_ids[di]), "lost": 0
            }
            out_b.append(boxes[di]); out_s.append(scores[di])
            out_c.append(class_ids[di]); out_t.append(tid)

        self._tracks = {k: v for k, v in self._tracks.items()
                        if v["lost"] <= self._max_lost}

        if not out_b:
            return _empty
        return (
            np.array(out_b, np.float32), np.array(out_s, np.float32),
            np.array(out_c, int),        np.array(out_t, int),
        )


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    track_id: int
    box:      tuple        # (x1, y1, x2, y2) pixel coords
    scores:   dict         # {query: confidence}
    label:    str = ""
    cam_id:   str = "cam0"


@dataclass
class CameraEvent:
    timestamp:  str
    query:      str
    track_id:   int
    confidence: float
    image_b64:  str
    cam_id:     str = "cam0"


@dataclass
class _ClipSession:
    cam_id:          str
    query:           str
    pre_frames:      list   # [(ts: float, jpeg: bytes)] — snapshot of pre-buffer at session start
    live_frames:     list   # [(ts: float, jpeg: bytes)] — appended by frame reader
    first_detection: float  # monotonic time of first detection — used for duration checks
    last_detection:  float  # monotonic time of most recent detection — used for gap checks
    wall_start:      float  # time.time() at session start — used for filename/timestamp


# ── Module-level state ────────────────────────────────────────────────────────

_pipeline_lock = threading.Lock()
_rebuild_lock  = threading.Lock()      # serialises concurrent _rebuild calls
_streams: dict[str, str] = {}          # cam_id → uri (ordered by insertion)

# Subprocess that owns the GStreamer pipeline (isolated from asyncio)
_worker_proc:         Optional[subprocess.Popen] = None
_frame_reader_thread: Optional[threading.Thread] = None

_inference_thread: Optional[threading.Thread] = None
_infer_stop        = threading.Event()

# Leaky inference slots: cam_id → (jpeg_bytes | ndarray, timestamp) | None
_infer_slots:      dict[str, Optional[tuple]] = {}
_infer_slots_lock  = threading.Lock()
_infer_event       = threading.Event()

# Display subscribers: cam_id → set[queue.Queue]
_subscribers:      dict[str, set] = {}
_subscribers_lock  = threading.Lock()

# Rate limiting per camera
_last_display: dict[str, float] = {}
_last_infer:   dict[str, float] = {}

# Per-camera trackers and detections
_trackers:    dict[str, _SimpleTracker] = {}
_detections:  dict[str, list]           = {}
_det_lock     = threading.Lock()

# Motion gate: last downscaled grayscale frame per camera for frame differencing
_motion_prev: dict[str, np.ndarray] = {}

_events:     deque = deque(maxlen=500)
_events_lock = threading.Lock()

# Per-camera rolling pre-event frame buffers (last PRE_BUFFER_S seconds of full frames)
_pre_buffers: dict[str, deque] = {}   # cam_id → deque of (ts: float, jpeg: bytes)
_last_clip_frame: dict[str, float] = {}  # rate-limit pre-buffer + session writes to CLIP_FPS

# Active clip recording sessions and finalized clip index
_clip_sessions:       dict[str, _ClipSession] = {}
_clip_sessions_lock   = threading.Lock()
_clip_index:          list[dict] = []
_clip_index_lock      = threading.Lock()
_clip_manager_stop    = threading.Event()
_clip_manager_thread: Optional[threading.Thread] = None

_queries:      list[str] = []
_queries_lock  = threading.Lock()

_threshold:      float = 0.3
_debug_overlay:  bool  = False

_query_update_lock  = threading.Lock()   # serialises concurrent add/remove calls


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
    # Only update active queries — never touch engine_queries (baked TRT text embeddings).
    # model.py compares queries vs engine_queries to decide whether to use TRT or PyTorch.
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

def _load_cameras() -> dict[str, str]:
    """Load persisted camera list from cameras.json. Returns {} on missing/corrupt file."""
    try:
        with open(CAMERAS_JSON_PATH) as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        logger.warning("Failed to load cameras.json: %s", exc)
        return {}


def _save_cameras(streams: dict[str, str]) -> None:
    """Write camera list to cameras.json.

    Note: atomic tmp+rename fails with EBUSY on Docker bind-mounted single files
    (rename(2) can't replace a bind-mount inode). Write directly instead.
    """
    try:
        with open(CAMERAS_JSON_PATH, "w") as f:
            json.dump(streams, f, indent=2)
    except Exception as exc:
        logger.warning("Failed to save cameras.json: %s", exc)


# ── Clip recording helpers ────────────────────────────────────────────────────

def _update_clip_session(cam_id: str, query: str, now: float) -> None:
    """Start a new clip session or extend the existing one for this camera."""
    with _clip_sessions_lock:
        if cam_id in _clip_sessions:
            _clip_sessions[cam_id].last_detection = now
        else:
            # Snapshot the current pre-buffer as the pre-event footage
            pre_frames = list(_pre_buffers.get(cam_id, []))
            _clip_sessions[cam_id] = _ClipSession(
                cam_id=cam_id,
                query=query,
                pre_frames=pre_frames,
                live_frames=[],
                first_detection=now,
                last_detection=now,
                wall_start=time.time(),
            )
            logger.info("Clip session started: cam=%s query=%r pre_frames=%d",
                        cam_id, query, len(pre_frames))


def _encode_clip(session: _ClipSession) -> Optional[Path]:
    """Encode all frames into an H.264 MP4 using imageio-ffmpeg's bundled static binary.
    Produces browser-playable H.264 with faststart (moov atom at front)."""
    import imageio_ffmpeg
    all_frames = session.pre_frames + session.live_frames
    if not all_frames:
        return None
    out_dir = CLIPS_DIR / session.cam_id
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.error("Cannot create clips dir %s: %s", out_dir, exc)
        return None
    ts_str   = datetime.fromtimestamp(session.wall_start).strftime("%Y%m%d_%H%M%S")
    safe_q   = "".join(c if c.isalnum() or c in "-_" else "_" for c in session.query)
    out_path = out_dir / f"{ts_str}_{safe_q}.mp4"

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        proc = subprocess.Popen(
            [
                ffmpeg_exe, "-y",
                "-f", "image2pipe", "-framerate", str(CLIP_FPS),
                "-i", "pipe:",
                "-c:v", "libx264", "-crf", "23", "-preset", "fast",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(out_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        written = 0
        for _, jpeg in all_frames:
            proc.stdin.write(jpeg)
            written += 1
        proc.stdin.close()
        proc.wait(timeout=120)
        if proc.returncode != 0:
            logger.error("ffmpeg failed for clip %s (rc=%d)", out_path, proc.returncode)
            return None
    except Exception as exc:
        logger.error("Clip encoding error for %s: %s", out_path, exc)
        return None

    if not out_path.exists() or out_path.stat().st_size == 0:
        logger.error("Clip file empty after encoding: %s", out_path)
        return None

    logger.info("Clip saved: %s (%d frames, %.1fs)", out_path, written, written / CLIP_FPS)
    return out_path


def _prune_clips(cam_id: str) -> None:
    """Delete oldest clips when per-camera limit is exceeded."""
    cam_dir = CLIPS_DIR / cam_id
    try:
        clips = sorted(cam_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        for old in clips[:-MAX_CLIPS_PER_CAM]:
            old.unlink(missing_ok=True)
            with _clip_index_lock:
                _clip_index[:] = [c for c in _clip_index if c["path"] != str(old)]
    except Exception as exc:
        logger.warning("Clip prune error for %s: %s", cam_id, exc)


def _finalize_session(session: _ClipSession) -> None:
    """Encode session frames, update clip index, prune old clips."""
    out_path = _encode_clip(session)
    if out_path is None:
        return
    entry = {
        "cam_id":    session.cam_id,
        "filename":  out_path.name,
        "path":      str(out_path),
        "timestamp": datetime.fromtimestamp(session.wall_start).isoformat(timespec="seconds"),
        "query":     session.query,
        "url":       f"/cameras/clips/file/{session.cam_id}/{out_path.name}",
    }
    with _clip_index_lock:
        _clip_index.append(entry)
    _prune_clips(session.cam_id)


def _clip_manager() -> None:
    """Background thread: finalize clip sessions whose POST_BUFFER_S window has elapsed."""
    while not _clip_manager_stop.is_set():
        _clip_manager_stop.wait(timeout=1.0)
        if _clip_manager_stop.is_set():
            break
        now = time.monotonic()
        expired = []
        with _clip_sessions_lock:
            for cam_id in list(_clip_sessions):
                session = _clip_sessions[cam_id]
                if now - session.last_detection > POST_BUFFER_S:
                    expired.append(_clip_sessions.pop(cam_id))
                elif now - session.first_detection > MAX_CLIP_DURATION_S:
                    # Hard duration cap: finalize this chunk; inference will start a new session
                    expired.append(_clip_sessions.pop(cam_id))
                    logger.info("Clip duration cap reached for cam=%s — finalizing chunk", cam_id)
        for session in expired:
            logger.info("Clip session ended: cam=%s query=%r live_frames=%d",
                        session.cam_id, session.query, len(session.live_frames))
            threading.Thread(
                target=_finalize_session,
                args=(session,),
                daemon=True,
                name=f"ds-clip-encode-{session.cam_id}",
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
                    "timestamp": datetime.fromtimestamp(mp4.stat().st_mtime).isoformat(timespec="seconds"),
                    "query":     query,
                    "url":       f"/cameras/clips/file/{cam_dir.name}/{mp4.name}",
                })
    except Exception as exc:
        logger.warning("Failed to scan clips dir: %s", exc)
    logger.info("Clip index loaded: %d clips", len(_clip_index))


# ── Frame-reader helpers (subprocess stdout → broadcast + inference) ──────────

def _read_exactly(buf, n: int) -> Optional[bytes]:
    """Read exactly n bytes from a binary file-like object. Returns None on EOF."""
    data = b""
    while len(data) < n:
        chunk = buf.read(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def _apply_debug_overlay(cam_id: str, jpeg_bytes: bytes) -> bytes:
    """Decode JPEG, draw detection boxes, re-encode. Used when debug overlay is on."""
    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jpeg_bytes
    with _det_lock:
        dets = list(_detections.get(cam_id, []))
    for det in dets:
        x1, y1, x2, y2 = det.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, f"{det.label} #{det.track_id}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
    ok_jpg, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes() if ok_jpg else jpeg_bytes


def _frame_reader(proc: subprocess.Popen, streams_snapshot: dict) -> None:
    """
    Read JPEG frames from the pipeline subprocess stdout.
    Each frame: [4B cam_id len][cam_id][4B jpeg len][jpeg]
    EOS marker:  [4B = 0]

    On subprocess exit (including EOS), trigger a pipeline restart so that
    file sources loop and RTSP sources reconnect.
    """
    src = proc.stdout
    try:
        while True:
            hdr = _read_exactly(src, 4)
            if hdr is None:
                break
            cam_id_len = struct.unpack(">I", hdr)[0]
            if cam_id_len == 0:
                logger.info("Pipeline worker sent EOS")
                break

            cam_id_bytes = _read_exactly(src, cam_id_len)
            if cam_id_bytes is None:
                break
            cam_id = cam_id_bytes.decode("utf-8")

            jpeg_len_hdr = _read_exactly(src, 4)
            if jpeg_len_hdr is None:
                break
            jpeg_len = struct.unpack(">I", jpeg_len_hdr)[0]

            jpeg_bytes = _read_exactly(src, jpeg_len)
            if jpeg_bytes is None or len(jpeg_bytes) != jpeg_len:
                break

            # Optional debug overlay (decode → draw → re-encode)
            if _debug_overlay:
                jpeg_bytes = _apply_debug_overlay(cam_id, jpeg_bytes)

            _broadcast(cam_id, jpeg_bytes)

            # Pre-event buffer + clip session feed — rate-limited to CLIP_FPS to bound RAM/disk.
            # At CLIP_FPS=10: pre-buffer holds 50 frames (5s), a 60s clip uses ≤600 frames (~30MB JPEG).
            ts_frame = time.monotonic()
            if ts_frame - _last_clip_frame.get(cam_id, 0.0) >= 1.0 / CLIP_FPS:
                _last_clip_frame[cam_id] = ts_frame
                if cam_id not in _pre_buffers:
                    _pre_buffers[cam_id] = deque(maxlen=PRE_BUFFER_S * CLIP_FPS)
                _pre_buffers[cam_id].append((ts_frame, jpeg_bytes))

                # Feed active clip session
                with _clip_sessions_lock:
                    active_session = _clip_sessions.get(cam_id)
                if active_session is not None:
                    active_session.live_frames.append((ts_frame, jpeg_bytes))

            # Inference slot throttled to INFER_FPS
            now = time.monotonic()
            if now - _last_infer.get(cam_id, 0.0) >= 1.0 / INFER_FPS:
                _last_infer[cam_id] = now
                with _infer_slots_lock:
                    _infer_slots[cam_id] = (jpeg_bytes, now)
                _infer_event.set()

    except Exception as exc:
        logger.error("Frame reader error: %s", exc)
    finally:
        try:
            proc.wait(timeout=5)
        except Exception:
            pass

    # Restart if this is still the active worker (not a superseded one)
    global _worker_proc
    if _worker_proc is proc and streams_snapshot:
        logger.info("Pipeline worker exited — restarting for file-loop / RTSP reconnect")
        threading.Thread(
            target=_rebuild,
            args=(streams_snapshot,),
            daemon=True,
            name="ds-eos-restart",
        ).start()


# ── Subscriber management ─────────────────────────────────────────────────────

def _broadcast(cam_id: str, jpeg_bytes: bytes) -> None:
    with _subscribers_lock:
        subs = set(_subscribers.get(cam_id, set()))
        subs |= set(_subscribers.get("*", set()))   # wildcard "any camera"
    for q in subs:
        try:
            q.put_nowait(jpeg_bytes)
        except queue.Full:
            pass


def subscribe_frames(cam_id: str = "cam0") -> queue.Queue:
    q: queue.Queue = queue.Queue(maxsize=4)
    with _subscribers_lock:
        _subscribers.setdefault(cam_id, set()).add(q)
    return q


def unsubscribe_frames(q: queue.Queue, cam_id: str = "cam0") -> None:
    with _subscribers_lock:
        _subscribers.get(cam_id, set()).discard(q)
        _subscribers.get("*", set()).discard(q)
    try:
        q.put_nowait(None)
    except queue.Full:
        pass


# ── Inference loop ────────────────────────────────────────────────────────────

def _inference_loop() -> None:
    try:
        import tritonclient.http as triton_http
    except Exception as exc:
        logger.error("Failed to import tritonclient.http — inference disabled: %s", exc)
        return

    logger.info("Inference loop starting, connecting to Triton at %s", TRITON_HTTP_URL)
    client = triton_http.InferenceServerClient(TRITON_HTTP_URL)

    for _ in range(60):
        if _infer_stop.is_set():
            return
        try:
            if client.is_server_ready() and client.is_model_ready("yoloworld"):
                logger.info("Triton ready")
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        logger.error("Triton not ready after 60s — inference disabled")
        return

    queries = _load_queries_from_meta()
    with _queries_lock:
        if not _queries:
            _queries.extend(queries)
        else:
            queries = list(_queries)

    fired:        dict[tuple, float] = {}   # (cam_id, track_id, q_idx) → last_fired
    score_buf:    dict[tuple, deque] = {}

    while not _infer_stop.is_set():
        _infer_event.wait(timeout=0.5)
        _infer_event.clear()

        with _infer_slots_lock:
            slots = {k: v for k, v in _infer_slots.items() if v is not None}
            for k in slots:
                _infer_slots[k] = None

        if not slots:
            continue

        with _queries_lock:
            queries = list(_queries)

        if not queries:
            continue

        for cam_id, (frame_or_jpeg, _ts) in slots.items():
            if _infer_stop.is_set():
                return

            # _infer_slots stores either JPEG bytes (subprocess path) or numpy
            # arrays (legacy in-process path).  Normalise to jpeg_bytes for Triton;
            # keep frame=None so numpy is only decoded when actually needed (crops).
            if isinstance(frame_or_jpeg, bytes):
                jpeg_bytes = frame_or_jpeg
                frame      = None   # lazy-decoded below only if needed
            else:
                frame = frame_or_jpeg
                ok, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    continue
                jpeg_bytes = jpeg_buf.tobytes()

            # Motion gate: skip Triton if the scene hasn't changed enough.
            # Diff is computed on a small grayscale thumbnail — ~0.5ms CPU, no GPU needed.
            if MOTION_GATE:
                gray = cv2.resize(
                    cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE),
                    MOTION_SCALE,
                )
                prev = _motion_prev.get(cam_id)
                _motion_prev[cam_id] = gray
                if prev is not None and cv2.absdiff(gray, prev).mean() < MOTION_THRESHOLD:
                    continue   # scene is static — skip Triton for this frame

            img_in = triton_http.InferInput("IMAGE", [len(jpeg_bytes)], "UINT8")
            img_in.set_data_from_numpy(
                np.frombuffer(jpeg_bytes, dtype=np.uint8).copy()
            )
            thr_in = triton_http.InferInput("THRESHOLD", [1], "FP32")
            thr_in.set_data_from_numpy(
                np.array([_threshold], dtype=np.float32)
            )

            try:
                resp = client.infer("yoloworld", inputs=[img_in, thr_in])
            except Exception as exc:
                logger.warning("Triton infer error (%s): %s", cam_id, exc)
                continue

            boxes     = resp.as_numpy("BOXES")
            scores    = resp.as_numpy("SCORES")
            label_ids = resp.as_numpy("LABEL_IDS")

            if len(boxes) == 0:
                with _det_lock:
                    _detections[cam_id] = []
                continue

            class_ids = label_ids.astype(int)
            tracker = _trackers.setdefault(cam_id, _SimpleTracker())
            t_boxes, t_scores, t_cids, t_tids = tracker.update(
                boxes.astype(np.float32),
                scores.astype(np.float32),
                class_ids,
            )

            # Decode JPEG to numpy only if we need pixel dimensions / crops
            if frame is None:
                frame = cv2.imdecode(
                    np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                )
            if frame is None:
                continue
            h, w = frame.shape[:2]
            now  = time.monotonic()
            dets: list[Detection] = []

            for i in range(len(t_boxes)):
                track_id = int(t_tids[i])
                x1, y1, x2, y2 = map(int, t_boxes[i])
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                conf  = float(t_scores[i])
                q_idx = int(t_cids[i])
                label = queries[q_idx] if q_idx < len(queries) else ""
                scores_dict = {label: conf} if label else {}

                key = (cam_id, track_id, q_idx)

                # Clip session: start/extend whenever detection is above threshold,
                # independent of the event-firing cooldown (RECHECK_INTERVAL_S).
                if label and conf >= _threshold:
                    _update_clip_session(cam_id, label, now)

                if label and now - fired.get(key, 0.0) > RECHECK_INTERVAL_S:
                    score_buf.setdefault(key, deque(maxlen=3)).append(conf)
                    if len(score_buf[key]) >= 1:
                        mean_conf = sum(score_buf[key]) / len(score_buf[key])
                        if mean_conf >= _threshold:
                            fired[key] = now
                            score_buf.pop(key, None)
                            crop        = frame[y1:y2, x1:x2]
                            ok_j, j_buf = cv2.imencode(
                                ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80]
                            )
                            img_b64 = (
                                base64.b64encode(j_buf.tobytes()).decode()
                                if ok_j else ""
                            )
                            with _events_lock:
                                _events.append(CameraEvent(
                                    timestamp=datetime.now().isoformat(timespec="seconds"),
                                    query=label,
                                    track_id=track_id,
                                    confidence=round(mean_conf, 3),
                                    image_b64=img_b64,
                                    cam_id=cam_id,
                                ))
                            logger.info(
                                "Event: cam=%s track=%d query=%r conf=%.3f",
                                cam_id, track_id, label, mean_conf,
                            )

                dets.append(Detection(
                    track_id=track_id,
                    box=(x1, y1, x2, y2),
                    scores=scores_dict,
                    label=label,
                    cam_id=cam_id,
                ))

            with _det_lock:
                _detections[cam_id] = dets

            # Prune fired/score_buf for this camera
            active = {int(tid) for tid in t_tids}
            score_buf = {k: v for k, v in score_buf.items()
                         if k[0] != cam_id or k[1] in active}

        # Prune stale fired entries globally
        cutoff = time.monotonic() - TRACK_PRUNE_AGE_S
        fired = {k: v for k, v in fired.items() if v > cutoff}

    logger.info("Inference loop stopped")


# ── Pipeline subprocess management ───────────────────────────────────────────

# Path to pipeline_worker.py — same directory as this service file
_WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "pipeline_worker.py")


def _start_pipeline_worker(stream_map: dict[str, str]) -> None:
    """Spawn pipeline_worker.py as a subprocess, start the frame-reader thread."""
    global _worker_proc, _frame_reader_thread

    if not stream_map:
        return

    cmd = [sys.executable, _WORKER_SCRIPT, json.dumps(stream_map)]
    logger.info("Spawning pipeline worker: %s", cmd)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        # stderr inherits — logs appear in `docker logs bianca-deepstream`
    )
    _worker_proc = proc
    logger.info("Pipeline worker started PID=%d", proc.pid)

    _frame_reader_thread = threading.Thread(
        target=_frame_reader,
        args=(proc, dict(stream_map)),
        daemon=True,
        name="ds-frame-reader",
    )
    _frame_reader_thread.start()


def _stop_pipeline() -> None:
    global _worker_proc, _frame_reader_thread

    if _worker_proc is not None:
        proc = _worker_proc
        _worker_proc = None   # clear first so frame_reader doesn't restart
        try:
            if proc.stdin:
                proc.stdin.write(b"STOP\n")
                proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
        logger.info("Pipeline worker stopped")

    if _frame_reader_thread and _frame_reader_thread.is_alive():
        _frame_reader_thread.join(timeout=3)
    _frame_reader_thread = None


def _stop_inference() -> None:
    global _inference_thread
    _infer_stop.set()
    _infer_event.set()
    if _inference_thread and _inference_thread.is_alive():
        _inference_thread.join(timeout=5)
    _inference_thread = None
    _infer_stop.clear()


def _start_inference() -> None:
    global _inference_thread
    _inference_thread = threading.Thread(
        target=_inference_loop, daemon=True, name="ds-inference"
    )
    _inference_thread.start()


def _rebuild(new_streams: dict[str, str]) -> None:
    """Stop everything, update state, restart with new stream set."""
    global _streams

    # Prevent concurrent rebuilds (e.g. EOS restart racing with add_stream)
    if not _rebuild_lock.acquire(blocking=False):
        logger.debug("_rebuild: already in progress, skipping concurrent call")
        return

    try:
        _stop_clip_manager()
        _stop_inference()
        _stop_pipeline()

        removed = set(_streams) - set(new_streams)

        with _infer_slots_lock:
            _infer_slots.clear()
        _last_display.clear()
        _last_infer.clear()

        for cam_id in removed:
            _trackers.pop(cam_id, None)
            with _det_lock:
                _detections.pop(cam_id, None)
            _pre_buffers.pop(cam_id, None)
            _last_clip_frame.pop(cam_id, None)
            _motion_prev.pop(cam_id, None)
            with _clip_sessions_lock:
                _clip_sessions.pop(cam_id, None)

        _streams = dict(new_streams)

        if _streams:
            _start_pipeline_worker(_streams)
            _start_inference()
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
        _save_cameras(_streams)


def remove_stream(cam_id: str) -> None:
    with _pipeline_lock:
        new = {k: v for k, v in _streams.items() if k != cam_id}
        _rebuild(new)
        _save_cameras(_streams)


def get_streams() -> dict[str, str]:
    return dict(_streams)


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
    """No-op — DeepStream feeds frames internally. Kept for API compat."""
    pass


# ── Detection / event getters ─────────────────────────────────────────────────

def get_latest_detections(cam_id: Optional[str] = None) -> list:
    with _det_lock:
        if cam_id is not None:
            return list(_detections.get(cam_id, []))
        return [d for dets in _detections.values() for d in dets]


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
    """Add a query and persist to meta.json. Re-export is orchestrated by the app."""
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
    """Remove a query and persist to meta.json. Re-export is orchestrated by the app."""
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


# ── Frame generators (consumed by FastAPI routes) ─────────────────────────────

async def ws_frame_generator(cam_id: str = "cam0"):
    """Async generator yielding raw JPEG bytes for WebSocket streaming."""
    q = subscribe_frames(cam_id)
    try:
        while True:
            try:
                chunk = q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.02)
                continue
            if chunk is None:
                break
            yield chunk
    finally:
        unsubscribe_frames(q, cam_id)


async def mjpeg_generator(rtsp_url: str, cam_id: str = "cam0"):
    """Async generator yielding MJPEG multipart chunks."""
    q = subscribe_frames(cam_id)
    try:
        while True:
            try:
                chunk = q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.02)
                continue
            if chunk is None:
                break
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + chunk + b"\r\n"
            )
    finally:
        unsubscribe_frames(q, cam_id)


# ── Startup: load persisted queries, cameras, and clip index ─────────────────

def _init_queries() -> None:
    persisted = _load_queries_from_meta()
    with _queries_lock:
        if not _queries:
            _queries.extend(persisted)


def _init_cameras() -> None:
    cameras = _load_cameras()
    for cam_id, uri in cameras.items():
        logger.info("Restoring camera from cameras.json: %s → %s", cam_id, uri)
        add_stream(cam_id, uri)


_init_queries()
_init_clip_index()
_init_cameras()


# ── FastAPI app (served by the deepstream container on port 8090) ─────────────

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse as _JSONResponse  # noqa: E402

app = FastAPI(title="DeepStream Service")


@app.get("/health")
async def _health():
    return {"ok": True}


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
    if cam_id not in get_streams():
        return _JSONResponse({"error": f"stream '{cam_id}' not found"}, status_code=404)
    await asyncio.to_thread(remove_stream, cam_id)
    return _JSONResponse({"ok": True, "streams": get_streams()})


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
    # Re-export status is owned by the app container (main.py). Return current queries only.
    return _JSONResponse({"state": "ready", "eta_s": 0, "queries": get_queries()})


@app.post("/queries/commit")
async def _queries_commit_route(payload: dict):
    """Called by main.py after a successful TRT re-export to sync the label mapping."""
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
async def _debug_overlay(payload: dict):
    enabled = bool(payload.get("enabled", False))
    set_debug_overlay(enabled)
    return _JSONResponse({"ok": True, "enabled": enabled})


@app.post("/set-stream")
async def _set_stream(payload: dict):
    url = payload.get("url", "").strip()
    set_stream_url(url)
    return _JSONResponse({"ok": True})


@app.websocket("/ws/{cam_id}")
async def _ws_cam(websocket: WebSocket, cam_id: str):
    await websocket.accept()
    try:
        async for jpeg_bytes in ws_frame_generator(cam_id):
            await websocket.send_bytes(jpeg_bytes)
    except WebSocketDisconnect:
        pass


@app.websocket("/ws")
async def _ws_default(websocket: WebSocket):
    await websocket.accept()
    try:
        async for jpeg_bytes in ws_frame_generator("cam0"):
            await websocket.send_bytes(jpeg_bytes)
    except WebSocketDisconnect:
        pass


@app.get("/clips")
async def _list_clips(cam_id: Optional[str] = None):
    with _clip_index_lock:
        clips = list(_clip_index)
    if cam_id:
        clips = [c for c in clips if c["cam_id"] == cam_id]
    return _JSONResponse({"clips": list(reversed(clips))})  # newest first


@app.get("/clips/file/{cam_id}/{filename}")
async def _serve_clip(cam_id: str, filename: str):
    # Sanitise inputs — only allow safe path components
    if "/" in cam_id or "/" in filename or ".." in cam_id or ".." in filename:
        return _JSONResponse({"error": "invalid path"}, status_code=400)
    path = CLIPS_DIR / cam_id / filename
    if not path.exists():
        return _JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(path), media_type="video/mp4")
