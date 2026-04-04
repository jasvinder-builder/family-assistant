"""
Scene analysis service — Grounding DINO + ByteTrack (supervision).

Replaces the former YOLO + CLIP two-stage pipeline with a single
open-vocabulary object detector: Grounding DINO localises objects described by
natural-language queries in one forward pass, producing well-calibrated
confidence scores instead of raw cosine similarities.  Supervision's
ByteTracker assigns persistent track IDs.

Models load lazily on the first start_analysis() call.
"""

import base64
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
# box_threshold  — minimum GDINO box confidence to keep a detection
# text_threshold — per-token text-image alignment (kept below box_threshold)
_box_threshold:  float = 0.35
_text_threshold: float = 0.25
_debug_overlay:  bool  = False

RECHECK_INTERVAL_S = 30
ANALYSIS_FPS       = 5
VOTE_FRAMES        = 2    # GDINO confidence is calibrated; 2 frames enough
TRACK_PRUNE_AGE_S  = 300


def get_threshold() -> float:
    return _box_threshold


def set_threshold(value: float) -> None:
    global _box_threshold, _text_threshold
    _box_threshold  = max(0.0, min(1.0, value))
    # Keep text threshold a fixed margin below box threshold
    _text_threshold = max(0.05, _box_threshold - 0.10)


def get_debug_overlay() -> bool:
    return _debug_overlay


def set_debug_overlay(enabled: bool) -> None:
    global _debug_overlay
    _debug_overlay = enabled


# Retained for API compatibility with cameras.html pad slider — no-op since
# GDINO returns tight boxes without needing extra crop context.
def get_pad_factor() -> float:
    return 0.0


def set_pad_factor(value: float) -> None:
    pass


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    track_id: int
    box: tuple    # (x1, y1, x2, y2) pixel coords
    scores: dict  # {matched_query: gdino_confidence}
    label: str = ""


@dataclass
class CameraEvent:
    timestamp: str
    query: str
    track_id: int
    confidence: float
    image_b64: str


# ── Shared state ──────────────────────────────────────────────────────────────

_latest_detections: list[Detection] = []
_detections_lock = threading.Lock()

_events: deque[CameraEvent] = deque(maxlen=500)
_events_lock = threading.Lock()

_queries: list[str] = []
_queries_lock = threading.Lock()

_shared_frame: Optional[np.ndarray] = None
_shared_frame_event = threading.Event()
_shared_frame_lock  = threading.Lock()

_stop_event       = threading.Event()
_analysis_thread: Optional[threading.Thread] = None


def get_latest_detections() -> list[Detection]:
    with _detections_lock:
        return list(_latest_detections)


def _set_latest_detections(d: list[Detection]) -> None:
    with _detections_lock:
        _latest_detections.clear()
        _latest_detections.extend(d)


def get_events(max_age_hours: float = 1.0) -> list[dict]:
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    with _events_lock:
        return [
            {
                "timestamp": e.timestamp,
                "query":     e.query,
                "confidence": round(e.confidence, 3),
                "image_b64": e.image_b64,
            }
            for e in _events
            if datetime.fromisoformat(e.timestamp) >= cutoff
        ]


def _append_event(ev: CameraEvent) -> None:
    with _events_lock:
        _events.append(ev)


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


def push_frame(frame: np.ndarray) -> None:
    """Called by camera_service for every decoded frame."""
    global _shared_frame
    with _shared_frame_lock:
        _shared_frame = frame
    _shared_frame_event.set()


def _get_shared_frame() -> Optional[np.ndarray]:
    with _shared_frame_lock:
        return _shared_frame


# ── Model loading ─────────────────────────────────────────────────────────────

_models_lock   = threading.Lock()
_models_loaded = False
_gdino_model     = None
_gdino_processor = None
_device          = "cpu"


def _load_models() -> None:
    global _models_loaded, _gdino_model, _gdino_processor, _device

    with _models_lock:
        if _models_loaded:
            return

        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Grounding DINO Tiny on %s", _device)

        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        _gdino_processor = AutoProcessor.from_pretrained(
            "IDEA-Research/grounding-dino-tiny"
        )
        dtype = torch.float16 if _device == "cuda" else torch.float32
        _gdino_model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained("IDEA-Research/grounding-dino-tiny", torch_dtype=dtype)
            .to(_device)
            .eval()
        )
        logger.info(
            "Grounding DINO Tiny loaded on %s (%s)",
            _device, "fp16" if _device == "cuda" else "fp32",
        )
        _models_loaded = True


# ── Inference helpers ─────────────────────────────────────────────────────────

def _build_prompt(queries: list[str]) -> str:
    """Format query list as 'thing1 . thing2 . thing3 .' for Grounding DINO."""
    return " . ".join(q.strip().lower().rstrip(" .") for q in queries) + " ."


def _match_label_to_query(label: str, queries: list[str]) -> int:
    """Return the index of the best-matching query for a GDINO label string.

    Tries exact match, then substring containment, then returns 0 as fallback.
    GDINO labels closely mirror the input phrases, so exact match usually hits.
    """
    label_l = label.lower().strip()
    for i, q in enumerate(queries):
        if q.lower().strip() == label_l:
            return i
    for i, q in enumerate(queries):
        ql = q.lower().strip()
        if label_l in ql or ql in label_l:
            return i
    return 0


# ── Analysis loop ─────────────────────────────────────────────────────────────

def _analysis_loop() -> None:
    logger.info("Scene analysis starting (Grounding DINO + ByteTrack)")
    try:
        _load_models()
    except Exception:
        logger.exception("Failed to load Grounding DINO — analysis disabled")
        return

    import torch
    import supervision as sv
    from PIL import Image

    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=ANALYSIS_FPS,
    )

    fired:        dict[tuple[int, int], float]        = {}
    score_buffer: dict[tuple[int, int], deque]        = {}
    frame_interval = 1.0 / ANALYSIS_FPS

    while not _stop_event.is_set():
        t0 = time.monotonic()

        _shared_frame_event.wait(timeout=frame_interval)
        _shared_frame_event.clear()

        frame = _get_shared_frame()
        if frame is None:
            continue

        queries = get_queries()
        debug   = get_debug_overlay()

        if not queries and not debug:
            time.sleep(max(0.0, frame_interval - (time.monotonic() - t0)))
            continue

        h, w = frame.shape[:2]

        # Use real queries or a generic fallback when debug mode is on with no queries
        prompt_queries = queries if queries else ["person", "vehicle", "animal"]

        try:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            text    = _build_prompt(prompt_queries)

            inputs = _gdino_processor(
                images=pil_img, text=text, return_tensors="pt",
            ).to(_device)

            with torch.no_grad():
                outputs = _gdino_model(**inputs)

            results = _gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=_box_threshold,
                text_threshold=_text_threshold,
                target_sizes=[(h, w)],
            )[0]

        except Exception:
            logger.exception("Grounding DINO inference failed")
            _set_latest_detections([])
            time.sleep(max(0.0, frame_interval - (time.monotonic() - t0)))
            continue

        boxes  = results["boxes"].cpu().float().numpy()   # [N, 4] xyxy pixels
        confs  = results["scores"].cpu().float().numpy()  # [N]
        labels = results["labels"]                        # [N] matched phrase strings

        if len(boxes) == 0:
            tracker.update_with_detections(sv.Detections.empty())
            _set_latest_detections([])
            time.sleep(max(0.0, frame_interval - (time.monotonic() - t0)))
            continue

        # Map each GDINO label to a query index; stored as class_id for tracker
        class_ids = np.array(
            [_match_label_to_query(lbl, prompt_queries) for lbl in labels],
            dtype=int,
        )

        sv_dets = sv.Detections(
            xyxy=boxes.astype(np.float32),
            confidence=confs.astype(np.float32),
            class_id=class_ids,
        )
        tracked = tracker.update_with_detections(sv_dets)

        now = time.monotonic()
        frame_detections: list[Detection] = []

        if tracked.tracker_id is not None and len(tracked) > 0:
            for i in range(len(tracked)):
                track_id = int(tracked.tracker_id[i])
                x1, y1, x2, y2 = map(int, tracked.xyxy[i])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                conf  = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
                q_idx = int(tracked.class_id[i])     if tracked.class_id   is not None else 0
                label = prompt_queries[q_idx] if q_idx < len(prompt_queries) else ""

                scores_dict: dict[str, float] = {}
                if queries and q_idx < len(queries):
                    scores_dict[queries[q_idx]] = conf

                    if now - fired.get((track_id, q_idx), 0.0) > RECHECK_INTERVAL_S:
                        key = (track_id, q_idx)
                        score_buffer.setdefault(key, deque(maxlen=VOTE_FRAMES)).append(conf)

                        if len(score_buffer[key]) >= VOTE_FRAMES:
                            mean_conf = sum(score_buffer[key]) / len(score_buffer[key])
                            if mean_conf >= _box_threshold:
                                fired[key] = now
                                score_buffer.pop(key, None)
                                crop = frame[y1:y2, x1:x2]
                                ok_jpg, jpg_buf = cv2.imencode(
                                    ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80]
                                )
                                img_b64 = (
                                    base64.b64encode(jpg_buf.tobytes()).decode()
                                    if ok_jpg else ""
                                )
                                _append_event(CameraEvent(
                                    timestamp=datetime.now().isoformat(timespec="seconds"),
                                    query=queries[q_idx],
                                    track_id=track_id,
                                    confidence=round(mean_conf, 3),
                                    image_b64=img_b64,
                                ))
                                logger.info(
                                    "Camera event: track=%d query=%r conf=%.3f",
                                    track_id, queries[q_idx], mean_conf,
                                )

                frame_detections.append(Detection(
                    track_id=track_id,
                    box=(x1, y1, x2, y2),
                    scores=scores_dict,
                    label=label,
                ))

        _set_latest_detections(frame_detections)

        # Prune score_buffer for tracks no longer in the active set
        active_ids = {int(tid) for tid in tracked.tracker_id} if tracked.tracker_id is not None else set()
        score_buffer = {k: v for k, v in score_buffer.items() if k[0] in active_ids}

        # Prune stale fired entries
        cutoff = now - TRACK_PRUNE_AGE_S
        fired = {k: v for k, v in fired.items() if v > cutoff}

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
        target=_analysis_loop, daemon=True, name="scene-analysis",
    )
    _analysis_thread.start()


def stop_analysis() -> None:
    global _analysis_thread
    _stop_event.set()
    if _analysis_thread and _analysis_thread.is_alive():
        _analysis_thread.join(timeout=5)
    _analysis_thread = None
