"""
Scene analysis service — YOLOv8 + ByteTrack + CLIP.

Runs an independent RTSP reader in a background thread, detects persons with
YOLOv8n, tracks them with ByteTrack, and matches each new track against
user-defined natural-language queries using CLIP cosine similarity.

Models are loaded lazily on the first start_analysis() call so startup is
unaffected when the cameras page is not in use.
"""

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import cv2

# Keep ultralytics config and weights inside the project directory rather than
# writing to ~/.config/Ultralytics — set before any ultralytics import.
_project_root = Path(__file__).parent.parent
_yolo_cfg_dir = _project_root / ".ultralytics"
_yolo_cfg_dir.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_yolo_cfg_dir))

# Force TCP transport for RTSP — UDP is often blocked by firewalls/NAT.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000",
)
import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_similarity_threshold = 0.15  # mutable at runtime via set_threshold()
_debug_overlay: bool = False   # mirrored from camera_service


def get_threshold() -> float:
    return _similarity_threshold


def set_threshold(value: float) -> None:
    global _similarity_threshold
    _similarity_threshold = max(0.0, min(1.0, value))


def get_debug_overlay() -> bool:
    return _debug_overlay


def set_debug_overlay(enabled: bool) -> None:
    global _debug_overlay
    _debug_overlay = enabled
RECHECK_INTERVAL_S   = 30     # seconds before re-checking same (track, query)
ANALYSIS_FPS         = 3      # frames per second to analyse
TRACK_PRUNE_AGE_S    = 300    # prune fired-cache entries older than this

# ── Latest detections (shared with camera_service for debug overlay) ──────────

@dataclass
class Detection:
    track_id: int
    box: tuple        # (x1, y1, x2, y2)
    scores: dict      # {query: similarity}


_latest_detections: list[Detection] = []
_detections_lock = threading.Lock()


def get_latest_detections() -> list[Detection]:
    with _detections_lock:
        return list(_latest_detections)


def _set_latest_detections(detections: list[Detection]) -> None:
    with _detections_lock:
        _latest_detections.clear()
        _latest_detections.extend(detections)


# ── Event log ────────────────────────────────────────────────────────────────

@dataclass
class CameraEvent:
    timestamp: str   # ISO seconds
    query: str
    track_id: int
    confidence: float
    image_b64: str   # base64-encoded JPEG crop, empty string if unavailable


_events: deque[CameraEvent] = deque(maxlen=500)
_events_lock = threading.Lock()


def get_events(max_age_hours: float = 1.0) -> list[dict]:
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    with _events_lock:
        return [
            {
                "timestamp": e.timestamp,
                "query": e.query,
                "confidence": round(e.confidence, 3),
                "image_b64": e.image_b64,
            }
            for e in _events
            if datetime.fromisoformat(e.timestamp) >= cutoff
        ]


def _append_event(ev: CameraEvent) -> None:
    with _events_lock:
        _events.append(ev)


# ── Query management ──────────────────────────────────────────────────────────

_queries: list[str] = []
_queries_lock = threading.Lock()


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
        _queries.append(text)
    return True


def remove_query(index: int) -> bool:
    with _queries_lock:
        if 0 <= index < len(_queries):
            _queries.pop(index)
            return True
    return False


# ── Model loading ─────────────────────────────────────────────────────────────

_models_lock   = threading.Lock()
_models_loaded = False
_yolo          = None
_clip_model    = None
_clip_processor = None
_device        = "cpu"  # resolved at load time


def _load_models() -> None:
    global _models_loaded, _yolo, _clip_model, _clip_processor, _device

    with _models_lock:
        if _models_loaded:
            return

        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading scene analysis models on %s", _device)

        from ultralytics import YOLO
        _yolo = YOLO("yolov8n.pt")  # ~6 MB, auto-downloads on first use
        logger.info("YOLOv8n loaded")

        from transformers import CLIPModel, CLIPProcessor
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_device)
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        logger.info("CLIP ViT-B/32 loaded on %s", _device)

        _models_loaded = True


# ── CLIP inference ────────────────────────────────────────────────────────────

def _clip_similarities(crop_bgr: np.ndarray, queries: list[str]) -> list[float]:
    """Return cosine similarity of a BGR crop against each query string."""
    from PIL import Image
    import torch

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img  = Image.fromarray(crop_rgb)

    inputs = _clip_processor(
        text=queries,
        images=pil_img,
        return_tensors="pt",
        padding=True,
    ).to(_device)

    with torch.no_grad():
        out     = _clip_model(**inputs)
        img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
        sims    = (img_emb @ txt_emb.T).squeeze(0)

    return sims.tolist() if sims.dim() > 0 else [sims.item()]


# ── Shared frame (pushed by camera_service reader) ────────────────────────────

_shared_frame: Optional[np.ndarray] = None
_shared_frame_event = threading.Event()
_shared_frame_lock  = threading.Lock()


def push_frame(frame: np.ndarray) -> None:
    """Called by camera_service each time a new frame is decoded."""
    global _shared_frame
    with _shared_frame_lock:
        _shared_frame = frame
    _shared_frame_event.set()


def _get_shared_frame() -> Optional[np.ndarray]:
    with _shared_frame_lock:
        return _shared_frame


# ── Analysis loop ─────────────────────────────────────────────────────────────

_stop_event      = threading.Event()
_analysis_thread: Optional[threading.Thread] = None


def _analysis_loop() -> None:
    logger.info("Scene analysis starting (shared-frame mode)")
    try:
        _load_models()
    except Exception:
        logger.exception("Failed to load scene analysis models — analysis disabled")
        return

    # (track_id, query_index) → monotonic time of last event fired
    fired: dict[tuple[int, int], float] = {}
    frame_interval = 1.0 / ANALYSIS_FPS

    while not _stop_event.is_set():
        t0 = time.monotonic()

        # Wait up to frame_interval for a new frame from the display reader
        _shared_frame_event.wait(timeout=frame_interval)
        _shared_frame_event.clear()

        frame = _get_shared_frame()
        if frame is None:
            continue

        queries = get_queries()
        debug = get_debug_overlay()
        if not queries and not debug:
            time.sleep(max(0.0, frame_interval - (time.monotonic() - t0)))
            continue

        # YOLOv8 + ByteTrack — persons only (class 0)
        results = _yolo.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            verbose=False,
            device=_device,
        )

        if results and results[0].boxes is not None:
            now = time.monotonic()
            h, w = frame.shape[:2]
            frame_detections: list[Detection] = []

            for box in results[0].boxes:
                if box.id is None:
                    continue
                track_id = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                scores: dict[str, float] = {}
                if queries:
                    pending = [
                        i for i, _ in enumerate(queries)
                        if now - fired.get((track_id, i), 0.0) > RECHECK_INTERVAL_S
                    ]
                    if pending:
                        try:
                            sims = _clip_similarities(crop, [queries[i] for i in pending])
                        except Exception:
                            logger.exception("CLIP inference failed for track %d", track_id)
                            sims = []
                        for q_idx, sim in zip(pending, sims):
                            scores[queries[q_idx]] = sim
                            if sim >= _similarity_threshold:
                                fired[(track_id, q_idx)] = now
                                import base64
                                ok_jpg, jpg_buf = cv2.imencode(
                                    ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80]
                                )
                                img_b64 = base64.b64encode(jpg_buf.tobytes()).decode() if ok_jpg else ""
                                _append_event(CameraEvent(
                                    timestamp=datetime.now().isoformat(timespec="seconds"),
                                    query=queries[q_idx],
                                    track_id=track_id,
                                    confidence=sim,
                                    image_b64=img_b64,
                                ))
                                logger.info(
                                    "Camera event: track=%d query=%r sim=%.3f",
                                    track_id, queries[q_idx], sim,
                                )

                frame_detections.append(Detection(
                    track_id=track_id,
                    box=(x1, y1, x2, y2),
                    scores=scores,
                ))

            _set_latest_detections(frame_detections)

            # Prune stale fired-cache entries
            cutoff = now - TRACK_PRUNE_AGE_S
            fired = {k: v for k, v in fired.items() if v > cutoff}
        else:
            _set_latest_detections([])

        elapsed = time.monotonic() - t0
        time.sleep(max(0.0, frame_interval - elapsed))

    logger.info("Scene analysis stopped")


# ── Public control ────────────────────────────────────────────────────────────

def start_analysis(rtsp_url: str) -> None:
    global _analysis_thread, _shared_frame
    stop_analysis()
    with _shared_frame_lock:
        _shared_frame = None
    _shared_frame_event.clear()
    _stop_event.clear()
    _analysis_thread = threading.Thread(
        target=_analysis_loop,
        daemon=True,
        name="scene-analysis",
    )
    _analysis_thread.start()


def stop_analysis() -> None:
    global _analysis_thread
    _stop_event.set()
    if _analysis_thread and _analysis_thread.is_alive():
        _analysis_thread.join(timeout=5)
    _analysis_thread = None
