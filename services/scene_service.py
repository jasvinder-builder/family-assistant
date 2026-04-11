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
from config import settings

logger = logging.getLogger(__name__)

TRITON_URL = settings.triton_url

# ── Thresholds ────────────────────────────────────────────────────────────────
_box_threshold:  float = 0.35
_text_threshold: float = 0.25
_debug_overlay:  bool  = False

RECHECK_INTERVAL_S = 30
ANALYSIS_FPS       = 5
VOTE_FRAMES        = 1


# ── Minimal IoU tracker (replaces supervision.ByteTrack) ─────────────────────
# supervision pulls in all of matplotlib (including the pybind11 C extension
# ft2font).  GLib's initialisation (Gst.init) corrupts pybind11's GC traversal
# handler, causing std::terminate() inside Python's cyclic GC.
# This self-contained class has zero extra dependencies (pure numpy).

class _SimpleTracker:
    """Greedy IoU multi-object tracker sufficient for ANALYSIS_FPS=5."""

    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 10):
        self._iou_thr  = iou_threshold
        self._max_lost = max_lost
        self._next_id  = 1
        # {track_id: {'box': [x1,y1,x2,y2], 'class_id': int, 'lost': int}}
        self._tracks: dict[int, dict] = {}

    @staticmethod
    def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute IoU between every pair in boxes a [N,4] and b [M,4] → [N,M]."""
        inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
        inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
        inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
        inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])
        inter    = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
        area_a   = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_b   = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        union    = area_a[:, None] + area_b[None, :] - inter
        return np.where(union > 0, inter / union, 0.0)

    def update(
        self,
        boxes:     np.ndarray,   # [N, 4] float32 xyxy
        scores:    np.ndarray,   # [N]    float32
        class_ids: np.ndarray,   # [N]    int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (boxes, scores, class_ids, track_ids) for active detections."""
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

        out_boxes, out_scores, out_cids, out_tids = [], [], [], []
        track_ids  = list(self._tracks)
        matched_t  = set()
        matched_d  = set()

        if track_ids:
            t_boxes = np.array([self._tracks[i]["box"] for i in track_ids], np.float32)
            iou     = self._iou_matrix(t_boxes, boxes)           # [T, N]
            ti_arr, di_arr = np.where(iou >= self._iou_thr)
            if len(ti_arr):
                order = np.argsort(-iou[ti_arr, di_arr])
                for idx in order:
                    ti, di = int(ti_arr[idx]), int(di_arr[idx])
                    if ti in matched_t or di in matched_d:
                        continue
                    matched_t.add(ti); matched_d.add(di)
                    tid = track_ids[ti]
                    self._tracks[tid] = {
                        "box": boxes[di].tolist(), "class_id": int(class_ids[di]), "lost": 0
                    }
                    out_boxes.append(boxes[di]); out_scores.append(scores[di])
                    out_cids.append(class_ids[di]); out_tids.append(tid)

        for di in range(len(boxes)):
            if di in matched_d:
                continue
            tid = self._next_id; self._next_id += 1
            self._tracks[tid] = {
                "box": boxes[di].tolist(), "class_id": int(class_ids[di]), "lost": 0
            }
            out_boxes.append(boxes[di]); out_scores.append(scores[di])
            out_cids.append(class_ids[di]); out_tids.append(tid)

        self._tracks = {k: v for k, v in self._tracks.items()
                        if v["lost"] <= self._max_lost}

        if not out_boxes:
            return _empty
        return (
            np.array(out_boxes,  np.float32),
            np.array(out_scores, np.float32),
            np.array(out_cids,   int),
            np.array(out_tids,   int),
        )
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

def _connect_triton(max_wait_s: float = 60.0, retry_interval: float = 2.0):
    """Wait for the GDINO FastAPI server to become ready, return an httpx.Client.

    The 'Triton' service was replaced with a plain FastAPI/uvicorn server running
    GDINO directly.  This avoids a Triton 24.04 Python backend bug where every
    tensor element is overwritten with the first element's value via shm IPC.
    """
    import httpx

    base_url = f"http://{TRITON_URL}"
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        if _stop_event.is_set():
            return None
        try:
            r = httpx.get(f"{base_url}/health", timeout=3.0)
            if r.status_code == 200:
                client = httpx.Client(base_url=base_url, timeout=15.0)
                logger.info("Connected to GDINO server at %s", base_url)
                return client
        except Exception as exc:
            logger.debug("GDINO server not ready yet: %s", exc)
        time.sleep(retry_interval)
    logger.error(
        "GDINO server at %s did not become ready within %.0fs — camera analysis disabled",
        base_url, max_wait_s,
    )
    return None


def _triton_infer(
    client,
    jpeg_bytes: bytes,
    queries: list[str],
    threshold: float,
    text_threshold: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Send a frame to the GDINO FastAPI server, return (boxes, scores, labels)."""
    response = client.post(
        "/infer",
        files={"image": ("frame.jpg", jpeg_bytes, "image/jpeg")},
        data={
            "queries": json.dumps(queries),
            "threshold": str(threshold),
            "text_threshold": str(text_threshold),
        },
    )
    response.raise_for_status()
    result = response.json()

    if result["boxes"]:
        boxes  = np.array(result["boxes"],  dtype=np.float32).reshape(-1, 4)
        scores = np.array(result["scores"], dtype=np.float32)
    else:
        boxes  = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,),   dtype=np.float32)
    labels = result["labels"]
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
    logger.info("Scene analysis starting (Grounding DINO via Triton + IoU tracker)")

    client = _connect_triton()
    if client is None:
        return

    tracker = _SimpleTracker(iou_threshold=0.3, max_lost=10)

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
            logger.exception("GDINO inference failed")
            _set_latest_detections([])
            time.sleep(max(0.0, frame_interval - (time.monotonic() - t0)))
            continue

        if len(boxes) == 0:
            tracker.update(
                np.zeros((0, 4), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0,), int),
            )
            _set_latest_detections([])
            time.sleep(max(0.0, frame_interval - (time.monotonic() - t0)))
            continue

        # Map each GDINO label to a query index for tracker class_id
        class_ids = np.array(
            [_match_label_to_query(lbl, prompt_queries) for lbl in labels],
            dtype=int,
        )

        t_boxes, t_scores, t_class_ids, t_track_ids = tracker.update(
            boxes.astype(np.float32),
            scores.astype(np.float32),
            class_ids,
        )

        now = time.monotonic()
        frame_detections: list[Detection] = []

        if len(t_boxes) > 0:
            for i in range(len(t_boxes)):
                track_id = int(t_track_ids[i])
                x1, y1, x2, y2 = map(int, t_boxes[i])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                conf  = float(t_scores[i])
                q_idx = int(t_class_ids[i])
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
        active_ids = {int(tid) for tid in t_track_ids}
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
