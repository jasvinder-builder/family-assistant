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
_crop_padding = 0.3            # fractional padding added around each crop before CLIP
_debug_overlay: bool = False   # mirrored from camera_service

# YOLO COCO class IDs we detect and their human-readable labels
_DETECT_CLASSES = [0, 1, 2, 3, 5, 7, 14, 15, 16]
_CLASS_LABELS = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck", 14: "bird", 15: "cat", 16: "dog",
}


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


def get_pad_factor() -> float:
    return _crop_padding


def set_pad_factor(value: float) -> None:
    global _crop_padding
    _crop_padding = max(0.0, min(2.0, value))


RECHECK_INTERVAL_S   = 30     # seconds before re-checking same (track, query)
ANALYSIS_FPS         = 5      # frames per second to analyse (up from 3)
TRACK_PRUNE_AGE_S    = 300    # prune fired-cache entries older than this
VOTE_FRAMES          = 3      # frames to average before firing an event (multi-frame voting)
MIN_CROP_PIXELS      = 3000   # skip CLIP for detections smaller than this (px²)

# Prompt templates for ensemble — averaged to form each query's text embedding.
# CLIP was trained on captions, so descriptive templates outperform bare labels.
_PROMPT_TEMPLATES = [
    "{}",
    "a photo of {}",
    "a picture of {}",
    "an image of {}",
]

# ── Latest detections (shared with camera_service for debug overlay) ──────────

@dataclass
class Detection:
    track_id: int
    box: tuple        # (x1, y1, x2, y2)
    scores: dict      # {query: similarity}
    label: str = "person"  # YOLO class label


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

_models_lock    = threading.Lock()
_models_loaded  = False
_yolo           = None
_clip_model     = None
_clip_processor = None
_device         = "cpu"  # resolved at load time

# Text embedding cache: query string → averaged, normalized CPU tensor.
# Populated on first use; reused until the server restarts.
_text_emb_cache: dict = {}


def _load_models() -> None:
    global _models_loaded, _yolo, _clip_model, _clip_processor, _device

    with _models_lock:
        if _models_loaded:
            return

        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading scene analysis models on %s", _device)

        from ultralytics import YOLO
        _yolo = YOLO("yolov8s.pt")  # small model — better recall than nano
        logger.info("YOLOv8s loaded")

        from transformers import CLIPModel, CLIPProcessor
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(_device)
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        logger.info("CLIP ViT-L/14 loaded on %s", _device)

        _models_loaded = True


# ── CLIP inference ────────────────────────────────────────────────────────────

def _get_text_embeddings(queries: list[str]):
    """Return stacked normalized text embeddings for queries, using cache.

    Each query is expanded into _PROMPT_TEMPLATES variants; their embeddings
    are averaged and re-normalized (prompt ensembling from the CLIP paper).
    Computed embeddings are stored in _text_emb_cache (CPU tensors) so
    repeated calls for the same queries are free.

    Returns a tensor of shape [len(queries), dim] on _device.
    """
    import torch

    uncached = [q for q in queries if q not in _text_emb_cache]
    if uncached:
        all_texts = [tmpl.format(q) for q in uncached for tmpl in _PROMPT_TEMPLATES]
        n_tmpl = len(_PROMPT_TEMPLATES)
        txt_inputs = _clip_processor(
            text=all_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=77,
        ).to(_device)
        with torch.no_grad():
            text_out = _clip_model.text_model(**txt_inputs)
            feats = _clip_model.text_projection(text_out.pooler_output)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        # Average over templates per query and re-normalize
        feats = feats.view(len(uncached), n_tmpl, -1).mean(dim=1)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        for i, q in enumerate(uncached):
            _text_emb_cache[q] = feats[i].cpu()

    return torch.stack([_text_emb_cache[q] for q in queries]).to(_device)


def _clip_similarities(crop_bgr: np.ndarray, queries: list[str]) -> list[float]:
    """Return cosine similarity of a BGR crop against each query string.

    Uses cached + ensembled text embeddings (see _get_text_embeddings).
    """
    from PIL import Image
    import torch

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img  = Image.fromarray(crop_rgb)

    img_inputs = _clip_processor(images=pil_img, return_tensors="pt").to(_device)
    with torch.no_grad():
        vision_out = _clip_model.vision_model(pixel_values=img_inputs["pixel_values"])
        img_emb = _clip_model.visual_projection(vision_out.pooler_output)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

    txt_emb = _get_text_embeddings(queries)
    sims = (img_emb @ txt_emb.T).squeeze(0)
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
    # (track_id, query_index) → deque of recent per-frame similarity scores
    score_buffer: dict[tuple[int, int], deque] = {}
    frame_interval = 1.0 / ANALYSIS_FPS

    import base64

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

        # YOLOv8s + ByteTrack — persons, vehicles, and animals
        # conf=0.15 lowers the detection threshold vs the default 0.25,
        # improving recall for partially occluded or distant objects.
        results = _yolo.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=_DETECT_CLASSES,
            conf=0.15,
            verbose=False,
            device=_device,
        )

        if results and results[0].boxes is not None:
            now = time.monotonic()
            h, w = frame.shape[:2]
            frame_detections: list[Detection] = []
            pad_factor = get_pad_factor()
            seen_track_ids: set[int] = set()

            for box in results[0].boxes:
                if box.id is None:
                    continue
                track_id = int(box.id.item())
                seen_track_ids.add(track_id)
                cls_id = int(box.cls[0].item())
                label = _CLASS_LABELS.get(cls_id, "object")
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                # Expand crop by pad_factor for better CLIP context
                pad_x = int((x2 - x1) * pad_factor)
                pad_y = int((y2 - y1) * pad_factor)
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(w, x2 + pad_x)
                cy2 = min(h, y2 + pad_y)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue

                scores: dict[str, float] = {}
                if queries:
                    # Skip CLIP on tiny detections — upscaling a 40×60px crop
                    # to 224×224 produces blurry embeddings that hurt accuracy.
                    crop_px = (cx2 - cx1) * (cy2 - cy1)
                    if crop_px < MIN_CROP_PIXELS:
                        frame_detections.append(Detection(
                            track_id=track_id, box=(x1, y1, x2, y2),
                            scores={}, label=label,
                        ))
                        continue

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

                            # Multi-frame voting: accumulate scores in a rolling
                            # window; only fire when the window mean >= threshold.
                            key = (track_id, q_idx)
                            if key not in score_buffer:
                                score_buffer[key] = deque(maxlen=VOTE_FRAMES)
                            score_buffer[key].append(sim)

                            if len(score_buffer[key]) < VOTE_FRAMES:
                                continue  # not enough frames yet

                            mean_sim = sum(score_buffer[key]) / len(score_buffer[key])
                            if mean_sim >= _similarity_threshold:
                                fired[key] = now
                                score_buffer.pop(key, None)
                                ok_jpg, jpg_buf = cv2.imencode(
                                    ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80]
                                )
                                img_b64 = base64.b64encode(jpg_buf.tobytes()).decode() if ok_jpg else ""
                                _append_event(CameraEvent(
                                    timestamp=datetime.now().isoformat(timespec="seconds"),
                                    query=queries[q_idx],
                                    track_id=track_id,
                                    confidence=round(mean_sim, 3),
                                    image_b64=img_b64,
                                ))
                                logger.info(
                                    "Camera event: track=%d label=%s query=%r mean_sim=%.3f",
                                    track_id, label, queries[q_idx], mean_sim,
                                )

                frame_detections.append(Detection(
                    track_id=track_id,
                    box=(x1, y1, x2, y2),
                    scores=scores,
                    label=label,
                ))

            _set_latest_detections(frame_detections)

            # Prune score_buffer for tracks no longer visible (they'll reset
            # when the track reappears and ByteTrack reassigns an ID).
            score_buffer = {k: v for k, v in score_buffer.items()
                            if k[0] in seen_track_ids}

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
