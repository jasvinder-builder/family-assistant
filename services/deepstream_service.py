"""
Deepstream inference service — Triton inference + clip recording + REST API.

Handles:
  - JPEG frame ingestion from Savant pyfunc via POST /internal/frame
  - YOLO-World TRT inference via Triton at ~10fps per camera
  - Per-camera IoU tracking + event firing
  - Segment-based clip recording (ffmpeg rolling segments from Savant Always-On Sink RTSP)
  - Clip extraction on detection events (ffmpeg stream-copy)
  - Savant RTSP adapter container lifecycle (Docker SDK)

Live display is handled entirely by Savant Always-On Sink (LL-HLS on port 888).

Public stream management:
  add_stream(cam_id, uri) / remove_stream(cam_id) / get_streams()
  get_queries / add_query / remove_query
"""

import asyncio
import base64
import json
import logging
import os
import queue
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

# Derive Triton HTTP host from TRITON_URL env var (default port gRPC 8001 → HTTP 8002)
_triton_raw  = os.environ.get("TRITON_URL", "localhost:8001")
_triton_host = _triton_raw.split(":")[0]
TRITON_HTTP_URL  = f"{_triton_host}:8002"

META_JSON_PATH    = os.environ.get("META_JSON_PATH", "/app/models/yoloworld.meta.json")
CAMERAS_JSON_PATH  = os.environ.get("CAMERAS_JSON_PATH", "./cameras.json")

# Savant adapter / sink configuration
SAVANT_MODULE_HOST = os.environ.get("SAVANT_MODULE_HOST", "bianca-savant-module")
SAVANT_SINK_HOST   = os.environ.get("SAVANT_SINK_HOST", "bianca-savant-sink")
DOCKER_NETWORK     = os.environ.get("DOCKER_NETWORK", "family-assistant_bianca-net")
RTSP_ADAPTER_IMAGE = os.environ.get(
    "RTSP_ADAPTER_IMAGE",
    "ghcr.io/insight-platform/savant-adapters-gstreamer:latest"
)
SINK_IMAGE = os.environ.get(
    "SINK_IMAGE",
    "ghcr.io/insight-platform/savant-adapters-deepstream:latest"
)
STUB_FILE_HOST_PATH = os.environ.get("STUB_FILE_HOST_PATH", "/app/stub.jpg")
GO2RTC_URL = os.environ.get("GO2RTC_URL", "http://bianca-go2rtc:1984")
GO2RTC_RTSP_HOST = os.environ.get("GO2RTC_RTSP_HOST", "bianca-go2rtc")
SEG_DURATION_S = int(os.environ.get("SEG_DURATION_S", "30"))
MAX_SEG_FILES  = int(os.environ.get("MAX_SEG_FILES", "25"))
CLIPS_DIR          = Path(os.environ.get("CLIPS_DIR", "./clips"))
PRE_BUFFER_S       = int(os.environ.get("PRE_BUFFER_S", "5"))
POST_BUFFER_S      = int(os.environ.get("POST_BUFFER_S", "5"))
MAX_CLIPS_PER_CAM  = int(os.environ.get("MAX_CLIPS_PER_CAM", "100"))
MAX_CLIP_DURATION_S = int(os.environ.get("MAX_CLIP_DURATION_S", "60"))  # hard cap per clip chunk
MIN_CLIP_DETECTIONS = int(os.environ.get("MIN_CLIP_DETECTIONS", "5"))   # discard clips with fewer detections
SEG_DURATION_S     = int(os.environ.get("SEG_DURATION_S", "30"))        # rolling segment length (must match worker)

MOTION_GATE       = os.environ.get("MOTION_GATE", "true").lower() == "true"
MOTION_THRESHOLD  = float(os.environ.get("MOTION_THRESHOLD", "2.0"))  # mean pixel diff (0-255) to trigger inference
MOTION_SCALE      = (320, 180)   # downscale to this for cheap frame differencing

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

# ffmpeg clip recorders: one subprocess per camera recording rolling segments
# from the Savant Always-On Sink RTSP output
_ffmpeg_procs: dict[str, subprocess.Popen] = {}

# Segment watcher thread: polls clips/ directory to update _seg_ring
_seg_watcher_stop   = threading.Event()
_seg_watcher_thread: Optional[threading.Thread] = None

_inference_thread: Optional[threading.Thread] = None
_infer_stop        = threading.Event()

# Leaky inference slots: cam_id → (jpeg_bytes | ndarray, timestamp) | None
_infer_slots:      dict[str, Optional[tuple]] = {}
_infer_slots_lock  = threading.Lock()
_infer_event       = threading.Event()

# Rate limiting per camera
_last_infer:   dict[str, float] = {}

# Per-camera trackers and detections
_trackers:    dict[str, _SimpleTracker] = {}
_detections:  dict[str, list]           = {}
_det_lock     = threading.Lock()

# Motion gate: last downscaled grayscale frame per camera for frame differencing
_motion_prev: dict[str, np.ndarray] = {}

_events:     deque = deque(maxlen=500)
_events_lock = threading.Lock()

# ── Segment ring buffer (disk-based rolling segments from ffmpeg) ─────────────
# Populated by _segment_watcher_loop polling clips/<cam>/segs/ every 5s.
_seg_ring:      dict[str, deque] = {}   # cam_id → deque[_SegBoundary], maxlen=20
_seg_ring_lock  = threading.Lock()

# Current in-flight segment per camera (not yet in _seg_ring — still being written).
_current_seg:      dict[str, Optional[dict]] = {}   # cam_id → {path, start_wall} | None
_current_seg_lock  = threading.Lock()

# Pending clip triggers: each entry represents one detection event window.
# The clip manager inspects these every second and cuts a clip once POST_BUFFER_S
# has elapsed since the last detection.
_clip_triggers:      dict[str, list] = {}   # cam_id → list of trigger-dicts
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

_query_update_lock  = threading.Lock()   # serialises concurrent add/remove calls


# ── Savant RTSP adapter management (Docker SDK) ───────────────────────────────

_docker_client = None


def _safe_name(cam_id: str) -> str:
    """Sanitize cam_id for use in Docker container names (only [a-zA-Z0-9_.-] allowed)."""
    import re
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", cam_id)


def _get_docker():
    global _docker_client
    if _docker_client is None:
        try:
            import docker
            _docker_client = docker.DockerClient(base_url="unix://var/run/docker.sock")
        except Exception as exc:
            logger.warning("Docker SDK unavailable — adapter management disabled: %s", exc)
    return _docker_client


def _go2rtc_register(cam_id: str, rtsp_url: str) -> bool:
    """Register a stream with go2rtc via HTTP API.  Returns True on success."""
    import urllib.parse
    params = urllib.parse.urlencode({"name": cam_id, "src": rtsp_url})
    req = urllib.request.Request(
        f"{GO2RTC_URL}/api/streams?{params}",
        method="PUT",
    )
    try:
        urllib.request.urlopen(req, timeout=5).close()
        logger.info("Registered stream %s in go2rtc", cam_id)
        return True
    except Exception as exc:
        logger.warning("go2rtc register failed for %s: %s — using direct RTSP", cam_id, exc)
        return False


def _go2rtc_unregister(cam_id: str) -> None:
    import urllib.parse
    req = urllib.request.Request(
        f"{GO2RTC_URL}/api/streams?name={urllib.parse.quote(cam_id)}",
        method="DELETE",
    )
    try:
        urllib.request.urlopen(req, timeout=5).close()
    except Exception:
        pass


def _start_rtsp_adapter(cam_id: str, rtsp_url: str) -> None:
    """Start a Savant RTSP adapter container for this camera.

    Registers the camera with go2rtc first (DTS normalization proxy), then
    starts the adapter pointing at go2rtc's clean re-stream.  Falls back to
    direct RTSP if go2rtc is unavailable.
    """
    client = _get_docker()
    if client is None:
        logger.warning("Docker client unavailable — cannot start RTSP adapter for %s", cam_id)
        return
    import docker
    container_name = f"bianca-rtsp-{_safe_name(cam_id)}"

    # Use go2rtc as RTSP proxy; request video-only to drop problematic audio DTS issues
    if _go2rtc_register(cam_id, rtsp_url):
        adapter_rtsp_url = f"rtsp://{GO2RTC_RTSP_HOST}:8554/{cam_id}?video=h264"
    else:
        adapter_rtsp_url = rtsp_url

    try:
        try:
            old = client.containers.get(container_name)
            old.stop(timeout=5)
            old.remove()
            logger.info("Removed existing RTSP adapter %s", container_name)
        except docker.errors.NotFound:
            pass
        client.containers.run(
            RTSP_ADAPTER_IMAGE,
            command="/opt/savant/adapters/gst/sources/rtsp.sh",
            detach=True,
            name=container_name,
            network=DOCKER_NETWORK,
            environment={
                "ZMQ_ENDPOINT": f"dealer+connect:tcp://{SAVANT_MODULE_HOST}:5555",
                "SOURCE_ID": cam_id,
                "RTSP_URI": adapter_rtsp_url,
                "RTSP_TRANSPORT": "tcp",
                "SYNC_OUTPUT": "False",
                "BUFFER_LEN": "50",
                "FFMPEG_TIMEOUT_MS": "120000",
            },
            restart_policy={"Name": "unless-stopped"},
        )
        logger.info("Started RTSP adapter %s → %s", cam_id, rtsp_url)
    except Exception as exc:
        logger.error("Failed to start RTSP adapter for %s: %s", cam_id, exc)


def _stop_rtsp_adapter(cam_id: str) -> None:
    """Stop the Savant RTSP adapter container for a camera."""
    client = _get_docker()
    if client is None:
        return
    import docker
    container_name = f"bianca-rtsp-{_safe_name(cam_id)}"
    try:
        container = client.containers.get(container_name)
        container.stop(timeout=5)
        logger.info("Stopped RTSP adapter %s", container_name)
    except docker.errors.NotFound:
        pass
    except Exception as exc:
        logger.warning("Failed to stop RTSP adapter %s: %s", container_name, exc)


def _stop_all_rtsp_adapters() -> None:
    """Stop all running bianca-rtsp-* adapter containers."""
    client = _get_docker()
    if client is None:
        return
    try:
        for container in client.containers.list(filters={"name": "bianca-rtsp-"}):
            try:
                container.stop(timeout=5)
                logger.info("Stopped RTSP adapter %s", container.name)
            except Exception as exc:
                logger.warning("Failed to stop adapter %s: %s", container.name, exc)
    except Exception as exc:
        logger.warning("Error listing adapter containers: %s", exc)


# ── Per-camera Always-On Sink container management ───────────────────────────

def _start_sink_container(cam_id: str) -> None:
    """Start an always_on_rtsp sink container for one camera.

    The sink subscribes to the Savant module ZMQ pub (port 5556), filters to
    SOURCE_ID=cam_id, re-encodes H.264, and serves:
      - HLS   via embedded MediaMTX :888  path /stream/{cam_id}/
      - RTSP  via embedded MediaMTX :554  path /stream/{cam_id}
    No host port mapping — access via Docker network by container name.
    """
    client = _get_docker()
    if client is None:
        logger.warning("Docker client unavailable — cannot start sink for %s", cam_id)
        return
    import docker
    container_name = f"bianca-sink-{_safe_name(cam_id)}"
    try:
        try:
            old = client.containers.get(container_name)
            old.stop(timeout=5)
            old.remove()
            logger.info("Removed existing sink %s", container_name)
        except docker.errors.NotFound:
            pass
        client.containers.run(
            SINK_IMAGE,
            command="python -m adapters.ds.sinks.always_on_rtsp",
            detach=True,
            name=container_name,
            network=DOCKER_NETWORK,
            volumes={STUB_FILE_HOST_PATH: {"bind": "/stub.jpg", "mode": "ro"}},
            environment={
                "ZMQ_ENDPOINT": f"sub+connect:tcp://{SAVANT_MODULE_HOST}:5556",
                "SOURCE_ID": cam_id,
                "DEV_MODE": "True",
                "STUB_FILE_LOCATION": "/stub.jpg",
                "FRAMERATE": "25/1",
                "ENCODER_PROFILE": "High",
                "ENCODER_BITRATE": "3000000",
            },
            device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
            restart_policy={"Name": "unless-stopped"},
        )
        logger.info("Started sink container %s", container_name)
    except Exception as exc:
        logger.error("Failed to start sink for %s: %s", cam_id, exc)


def _stop_sink_container(cam_id: str) -> None:
    client = _get_docker()
    if client is None:
        return
    import docker
    container_name = f"bianca-sink-{_safe_name(cam_id)}"
    try:
        client.containers.get(container_name).stop(timeout=5)
        logger.info("Stopped sink %s", container_name)
    except docker.errors.NotFound:
        pass
    except Exception as exc:
        logger.warning("Failed to stop sink %s: %s", container_name, exc)


def _stop_all_sink_containers() -> None:
    client = _get_docker()
    if client is None:
        return
    try:
        for c in client.containers.list(filters={"name": "bianca-sink-"}):
            try:
                c.stop(timeout=5)
                logger.info("Stopped sink %s", c.name)
            except Exception as exc:
                logger.warning("Failed to stop sink %s: %s", c.name, exc)
    except Exception as exc:
        logger.warning("Error listing sink containers: %s", exc)


# ── Clip recording via ffmpeg from Always-On Sink RTSP ───────────────────────

def _start_clip_recorder(cam_id: str) -> None:
    """Start ffmpeg to record rolling 30s segments from the Savant Always-On Sink."""
    segs_dir = CLIPS_DIR / cam_id / "segs"
    try:
        segs_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.error("Cannot create segs dir %s: %s", segs_dir, exc)
        return

    import imageio_ffmpeg
    rtsp_url = f"rtsp://bianca-sink-{cam_id}:554/stream/{cam_id}"
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-c", "copy",
        "-f", "segment",
        "-segment_time", str(SEG_DURATION_S),
        "-segment_wrap", str(MAX_SEG_FILES),
        "-reset_timestamps", "1",
        "-segment_format", "mpegts",
        "-segment_atclocktime", "1",
        str(segs_dir / "seg_%05d.ts"),
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _ffmpeg_procs[cam_id] = proc
        logger.info("Started clip recorder for %s (PID=%d) reading %s",
                    cam_id, proc.pid, rtsp_url)
    except Exception as exc:
        logger.error("Failed to start clip recorder for %s: %s", cam_id, exc)


def _stop_clip_recorder(cam_id: str) -> None:
    """Stop the ffmpeg clip recorder for a camera."""
    proc = _ffmpeg_procs.pop(cam_id, None)
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    logger.info("Stopped clip recorder for %s", cam_id)


def _stop_all_clip_recorders() -> None:
    for cam_id in list(_ffmpeg_procs):
        _stop_clip_recorder(cam_id)


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
            logger.debug("Segment watcher: %d new segments for %s", len(new_segs), cam_id)


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

def _load_cameras() -> tuple[dict[str, str], dict[str, Optional[dict]]]:
    """Load persisted camera list from cameras.json.
    Supports both old format {"cam": "rtsp://..."} and new {"cam": {"url":..., "roi":...}}.
    Returns (streams, rois) — both default to {} on missing/corrupt file."""
    try:
        with open(CAMERAS_JSON_PATH) as f:
            data = json.load(f)
        streams: dict[str, str] = {}
        rois:    dict[str, Optional[dict]] = {}
        for k, v in data.items():
            if not isinstance(k, str):
                continue
            if isinstance(v, str):
                streams[k] = v
                rois[k]    = None
            elif isinstance(v, dict) and "url" in v:
                streams[k] = v["url"]
                rois[k]    = v.get("roi")  # {x,y,w,h} or None
        return streams, rois
    except FileNotFoundError:
        return {}, {}
    except Exception as exc:
        logger.warning("Failed to load cameras.json: %s", exc)
        return {}, {}


def _save_cameras(streams: dict[str, str], rois: dict[str, Optional[dict]]) -> None:
    """Write camera list to cameras.json in new {url, roi} format.

    Note: atomic tmp+rename fails with EBUSY on Docker bind-mounted single files
    (rename(2) can't replace a bind-mount inode). Write directly instead.
    """
    data = {cam_id: {"url": uri, "roi": rois.get(cam_id)} for cam_id, uri in streams.items()}
    try:
        with open(CAMERAS_JSON_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logger.warning("Failed to save cameras.json: %s", exc)


# ── Clip recording helpers (Phase 2: disk-based segment recording) ────────────

def _trigger_clip_on_detection(cam_id: str, query: str, wall_now: float) -> None:
    """Start or extend a clip trigger for this camera.

    Called from the inference loop whenever a detection fires above threshold.
    wall_now must be time.time() so it correlates with segment boundary timestamps.
    Multiple detections within MAX_CLIP_DURATION_S extend the same trigger window.
    """
    with _clip_triggers_lock:
        triggers = _clip_triggers.setdefault(cam_id, [])
        # Extend the most recent active trigger if it's within the hard cap window
        if triggers:
            t = triggers[-1]
            if wall_now - t["first_detect_wall"] <= MAX_CLIP_DURATION_S:
                t["last_detect_wall"] = wall_now
                t["detection_count"] += 1
                return
        # New trigger
        triggers.append({
            "query":             query,
            "first_detect_wall": wall_now,
            "last_detect_wall":  wall_now,
            "detection_count":   1,
        })
        logger.info("Clip trigger started: cam=%s query=%r", cam_id, query)


def _extract_and_save_clip(cam_id: str, trigger: dict) -> Optional[Path]:
    """Cut a clip from rolling segment files using ffmpeg stream-copy (zero re-encode).

    Segments are H.264 MP4 files written by splitmuxsink.  We use either:
      - Single segment: ffmpeg -ss {offset} -i {seg} -t {dur} -c copy
      - Multiple segments: ffmpeg -f concat -safe 0 -i {list} -ss {offset} -t {dur} -c copy
    """
    import imageio_ffmpeg

    query           = trigger["query"]
    first_wall      = trigger["first_detect_wall"]
    last_wall       = trigger["last_detect_wall"]
    clip_start_wall = first_wall - PRE_BUFFER_S
    clip_end_wall   = last_wall  + POST_BUFFER_S

    with _seg_ring_lock:
        segs = [s for s in _seg_ring.get(cam_id, [])
                if s.end_wall > clip_start_wall and s.start_wall < clip_end_wall]
    segs.sort(key=lambda s: s.start_wall)

    # Filter to files that actually exist on disk (splitmuxsink max-files may have pruned)
    segs = [s for s in segs if Path(s.path).exists()]

    if not segs:
        logger.warning("Clip cut skipped: no segment files available for cam=%s query=%r",
                       cam_id, query)
        return None

    # Clamp the clip window to actual segment coverage
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

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    concat_file: Optional[Path] = None

    try:
        if len(segs) == 1:
            seg    = segs[0]
            ss     = max(0.0, actual_start - seg.start_wall)
            dur    = actual_end - actual_start
            cmd = [ffmpeg_exe, "-y",
                   "-ss", f"{ss:.3f}", "-i", seg.path,
                   "-t", f"{dur:.3f}",
                   "-c", "copy", "-movflags", "+faststart",
                   str(out_path)]
        else:
            # Concat all relevant segments, then seek/trim
            concat_file = out_dir / f".concat_{ts_str}.txt"
            with open(concat_file, "w") as cf:
                for seg in segs:
                    cf.write(f"file '{seg.path}'\n")
            ss  = max(0.0, actual_start - segs[0].start_wall)
            dur = actual_end - actual_start
            cmd = [ffmpeg_exe, "-y",
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

    logger.info("Clip saved (stream-copy): cam=%s query=%r path=%s segs=%d dur=%.1fs",
                cam_id, query, out_path.name, len(segs), actual_end - actual_start)
    return out_path


def _prune_clips(cam_id: str) -> None:
    """Delete oldest extracted clips when per-camera limit is exceeded."""
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
    """Add a newly saved clip to the in-memory index and prune old ones."""
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

    # Check whether we already have segment coverage up to clip_end_wall.
    # If not, force a segment cut and wait up to SEG_DURATION + 5 s for the
    # boundary event to arrive.
    deadline = clip_end_wall + SEG_DURATION_S + 5.0

    while time.time() < deadline:
        with _seg_ring_lock:
            segs = list(_seg_ring.get(cam_id, []))
        if any(s.end_wall >= clip_end_wall for s in segs):
            break
        time.sleep(2.0)

    out_path = _extract_and_save_clip(cam_id, trigger)
    if out_path:
        _register_clip(cam_id, out_path, trigger)


def _clip_manager() -> None:
    """Background thread: watch for expired trigger windows and cut clips."""
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
            now      = time.monotonic()
            wall_now = time.time()    # wall clock for segment correlation
            dets: list[Detection] = []

            # ROI offset: boxes from Triton are in ROI-crop coordinate space.
            # Translate to full-frame coordinates for overlay and clip recording.
            roi = _rois.get(cam_id)
            roi_ox = roi["x"] if roi else 0
            roi_oy = roi["y"] if roi else 0

            for i in range(len(t_boxes)):
                track_id = int(t_tids[i])
                x1, y1, x2, y2 = map(int, t_boxes[i])
                # Clamp to inference-frame bounds first (crop or full frame)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                conf  = float(t_scores[i])
                q_idx = int(t_cids[i])
                label = queries[q_idx] if q_idx < len(queries) else ""
                scores_dict = {label: conf} if label else {}

                key = (cam_id, track_id, q_idx)

                # Clip trigger: start/extend whenever detection is above threshold,
                # independent of the event-firing cooldown (RECHECK_INTERVAL_S).
                # wall_now is used so timestamps match pipeline segment boundaries.
                if label and conf >= _threshold:
                    _trigger_clip_on_detection(cam_id, label, wall_now)

                # Translate box from inference-frame coords to full-frame coords
                fx1 = x1 + roi_ox; fy1 = y1 + roi_oy
                fx2 = x2 + roi_ox; fy2 = y2 + roi_oy

                if label and now - fired.get(key, 0.0) > RECHECK_INTERVAL_S:
                    score_buf.setdefault(key, deque(maxlen=3)).append(conf)
                    if len(score_buf[key]) >= 1:
                        mean_conf = sum(score_buf[key]) / len(score_buf[key])
                        if mean_conf >= _threshold:
                            fired[key] = now
                            score_buf.pop(key, None)
                            # Crop from inference frame (ROI-relative coords are still valid here)
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

                # Store full-frame coordinates for overlay + clip crop
                dets.append(Detection(
                    track_id=track_id,
                    box=(fx1, fy1, fx2, fy2),
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

    if not _rebuild_lock.acquire(blocking=False):
        logger.debug("_rebuild: already in progress, skipping concurrent call")
        return

    try:
        _stop_clip_manager()
        _stop_inference()
        _stop_seg_watcher()
        _stop_all_clip_recorders()
        _stop_all_rtsp_adapters()
        _stop_all_sink_containers()

        removed = set(_streams) - set(new_streams)

        with _infer_slots_lock:
            _infer_slots.clear()
        _last_infer.clear()

        for cam_id in removed:
            _go2rtc_unregister(cam_id)
            _trackers.pop(cam_id, None)
            with _det_lock:
                _detections.pop(cam_id, None)
            _motion_prev.pop(cam_id, None)
            _rois.pop(cam_id, None)
            with _seg_ring_lock:
                _seg_ring.pop(cam_id, None)
            with _current_seg_lock:
                _current_seg.pop(cam_id, None)
            with _clip_triggers_lock:
                _clip_triggers.pop(cam_id, None)

        _streams = dict(new_streams)

        if _streams:
            for cam_id, uri in _streams.items():
                _start_rtsp_adapter(cam_id, uri)
                _start_sink_container(cam_id)
                _start_clip_recorder(cam_id)
            _start_seg_watcher()
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
        _save_cameras(_streams, _rois)


def remove_stream(cam_id: str) -> None:
    with _pipeline_lock:
        new = {k: v for k, v in _streams.items() if k != cam_id}
        _rebuild(new)
        _save_cameras(_streams, _rois)


def set_roi(cam_id: str, roi: Optional[dict]) -> None:
    """Set or clear the inference ROI for a camera.
    ROI is applied dynamically in the Savant pyfunc — no pipeline rebuild needed."""
    with _pipeline_lock:
        if cam_id not in _streams:
            raise KeyError(cam_id)
        _rois[cam_id] = roi
        _save_cameras(_streams, _rois)


def get_streams() -> dict[str, dict]:
    """Return {cam_id: {url, roi}} for all active streams."""
    return {
        cam_id: {"url": uri, "roi": _rois.get(cam_id)}
        for cam_id, uri in _streams.items()
    }


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

import httpx  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse as _JSONResponse, StreamingResponse  # noqa: E402

app = FastAPI(title="DeepStream Service")


@app.get("/health")
async def _health():
    return {"ok": True}


@app.post("/internal/frame")
async def _internal_frame(request: Request):
    """Receive a JPEG frame from the Savant pyfunc for Triton inference."""
    from starlette.requests import ClientDisconnect
    cam_id = request.headers.get("X-Cam-Id", "cam0")
    try:
        jpeg_bytes = await request.body()
    except ClientDisconnect:
        return _JSONResponse({"ok": True})
    if not jpeg_bytes:
        return _JSONResponse({"error": "empty body"}, status_code=400)
    now = time.monotonic()
    with _infer_slots_lock:
        _infer_slots[cam_id] = (jpeg_bytes, now)
    _infer_event.set()
    return _JSONResponse({"ok": True})


@app.get("/hls/{cam_id}/{path:path}")
async def _hls_proxy(request: Request, cam_id: str, path: str):
    """Proxy HLS/RTSP segment requests to the per-camera always_on_rtsp sink container."""
    if ".." in cam_id or "/" in cam_id:
        return _JSONResponse({"error": "invalid cam_id"}, status_code=400)
    target = f"http://bianca-sink-{_safe_name(cam_id)}:888/stream/{cam_id}/{path}"
    # LL-HLS blocking playlist requests carry _HLS_msn/_HLS_part — must forward.
    params = dict(request.query_params)
    client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=30.0))
    try:
        req = client.build_request("GET", target, params=params)
        r = await client.send(req, stream=True)
    except Exception as exc:
        await client.aclose()
        logger.debug("HLS proxy error for %s/%s: %s", cam_id, path, exc)
        return _JSONResponse({"error": "sink unavailable"}, status_code=503)

    async def _stream():
        try:
            async for chunk in r.aiter_bytes(65536):
                yield chunk
        finally:
            await r.aclose()
            await client.aclose()

    passthrough_headers = {
        k: v for k, v in r.headers.items()
        if k.lower() not in ("content-encoding", "transfer-encoding", "content-length")
    }
    return StreamingResponse(
        _stream(),
        status_code=r.status_code,
        media_type=r.headers.get("content-type", "application/octet-stream"),
        headers=passthrough_headers,
    )


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
    """Set or clear the inference ROI for a camera.
    Body: {"x": int, "y": int, "w": int, "h": int} to set, {} or {"roi": null} to clear."""
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
