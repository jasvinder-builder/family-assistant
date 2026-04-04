"""
Scene analysis service — Grounding DINO + ByteTrack via Triton.

Frames are sent as JPEG bytes to a Triton Inference Server (Python backend
running Grounding DINO Tiny on GPU) over gRPC.  Boxes, scores and labels are
returned in full-resolution pixel coordinates.  ByteTrack tracking and all
event-firing logic run locally in this process (CPU-only, no PyTorch here).

Set TRITON_URL in .env (default: localhost:8001) to point at the Triton
gRPC port.
"""

import base64
import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

TRITON_URL = os.getenv("TRITON_URL", "localhost:8001")

# ── Thresholds ────────────────────────────────────────────────────────────────
_box_threshold:  float = 0.35
_text_threshold: float = 0.25
_debug_overlay:  bool  = False

RECHECK_INTERVAL_S = 30
ANALYSIS_FPS       = 5
VOTE_FRAMES        = 1
TRACK_PRUNE_AGE_S  = 300


def get_threshold() -> float:
    return _box_threshold


def set_threshold(value: float) -> None:
    global _box_threshold, _text_threshold
    _box_threshold  = max(0.0, min(1.0, value))
    _text_threshold = max(0.05, _box_threshold - 0.10)


def get_debug_overlay() -> bool:
    return _debug_overlay


def set_debug_overlay(enabled: bool) -> None:
    global _debug_overlay
    _debug_overlay = enabled


# Retained for API compatibility with cameras.html pad slider — no-op.
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
                "timestamp":  e.timestamp,
                "query":      e.query,
                "confidence": round(e.confidence, 3),
                "image_b64":  e.image_b64,
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


# ── Triton gRPC client ────────────────────────────────────────────────────────

def _connect_triton(max_wait_s: float = 30.0, retry_interval: float = 2.0):
    """Wait for Triton to become ready, return a connected client or None."""
    import tritonclient.grpc as grpcclient

    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        if _stop_event.is_set():
            return None
        try:
            client = grpcclient.InferenceServerClient(url=TRITON_URL, verbose=False)
            if client.is_server_ready() and client.is_model_ready("gdino"):
                logger.info("Connected to Triton at %s (gdino model ready)", TRITON_URL)
                return client
        except Exception as exc:
            logger.debug("Triton not ready yet: %s", exc)
        time.sleep(retry_interval)
    logger.error(
        "Triton at %s did not become ready within %.0fs — camera analysis disabled",
        TRITON_URL, max_wait_s,
    )
    return None


def _triton_infer(
    client,
    jpeg_bytes: bytes,
    queries: list[str],
    threshold: float,
    text_threshold: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Send a frame to Triton, return (boxes [N,4], scores [N], labels [N])."""
    import tritonclient.grpc as grpcclient

    img_input = grpcclient.InferInput("IMAGE", [1], "BYTES")
    img_input.set_data_from_numpy(np.array([jpeg_bytes], dtype=object))

    qry_input = grpcclient.InferInput("QUERIES", [1], "BYTES")
    qry_input.set_data_from_numpy(
        np.array([json.dumps(queries).encode()], dtype=object)
    )

    thr_input = grpcclient.InferInput("THRESHOLD", [1], "FP32")
    thr_input.set_data_from_numpy(np.array([threshold], dtype=np.float32))

    txt_input = grpcclient.InferInput("TEXT_THRESHOLD", [1], "FP32")
    txt_input.set_data_from_numpy(np.array([text_threshold], dtype=np.float32))

    outputs = [
        grpcclient.InferRequestedOutput("BOXES"),
        grpcclient.InferRequestedOutput("SCORES"),
        grpcclient.InferRequestedOutput("LABELS"),
    ]

    response = client.infer(
        model_name="gdino",
        inputs=[img_input, qry_input, thr_input, txt_input],
        outputs=outputs,
    )

    boxes  = response.as_numpy("BOXES")   # float32 [N, 4] or [0, 4]
    scores = response.as_numpy("SCORES")  # float32 [N]
    labels_raw = response.as_numpy("LABELS")
    labels = [
        l.decode() if isinstance(l, bytes) else str(l)
        for l in labels_raw
    ]
    return boxes, scores, labels


# ── Label-to-query matching ───────────────────────────────────────────────────

def _match_label_to_query(label: str, queries: list[str]) -> int:
    """Return the index of the best-matching query for a GDINO label string."""
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
    logger.info("Scene analysis starting (Grounding DINO via Triton + ByteTrack)")

    client = _connect_triton()
    if client is None:
        return

    import supervision as sv

    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=ANALYSIS_FPS,
    )

    fired:        dict[tuple[int, int], float] = {}
    score_buffer: dict[tuple[int, int], deque] = {}
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
        prompt_queries = queries if queries else ["person", "vehicle", "animal"]

        # JPEG-encode frame for Triton transport
        ok, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue
        jpeg_bytes = jpeg_buf.tobytes()

        try:
            boxes, scores, labels = _triton_infer(
                client, jpeg_bytes, prompt_queries, _box_threshold, _text_threshold,
            )
        except Exception:
            logger.exception("Triton GDINO inference failed")
            # Try to reconnect on next iteration
            try:
                import tritonclient.grpc as grpcclient
                client = grpcclient.InferenceServerClient(url=TRITON_URL, verbose=False)
            except Exception:
                pass
            _set_latest_detections([])
            time.sleep(max(0.0, frame_interval - (time.monotonic() - t0)))
            continue

        if len(boxes) == 0:
            tracker.update_with_detections(sv.Detections.empty())
            _set_latest_detections([])
            time.sleep(max(0.0, frame_interval - (time.monotonic() - t0)))
            continue

        # Map each GDINO label to a query index for ByteTrack class_id
        class_ids = np.array(
            [_match_label_to_query(lbl, prompt_queries) for lbl in labels],
            dtype=int,
        )

        sv_dets = sv.Detections(
            xyxy=boxes.astype(np.float32),
            confidence=scores.astype(np.float32),
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

        # Prune score_buffer for inactive tracks
        active_ids = (
            {int(tid) for tid in tracked.tracker_id}
            if tracked.tracker_id is not None else set()
        )
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
