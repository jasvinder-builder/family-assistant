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
  get_queries / add_query / remove_query / get_query_status
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
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

# Derive Triton HTTP host from TRITON_URL env var (default port gRPC 8001 → HTTP 8002)
_triton_raw  = os.environ.get("TRITON_URL", "localhost:8001")
_triton_host = _triton_raw.split(":")[0]
TRITON_HTTP_URL = f"{_triton_host}:8002"

META_JSON_PATH    = os.environ.get("META_JSON_PATH", "/app/models/yoloworld.meta.json")

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


# ── Module-level state ────────────────────────────────────────────────────────

_pipeline_lock = threading.Lock()
_streams: dict[str, str] = {}          # cam_id → uri (ordered by insertion)
_pipeline      = None                   # Gst.Pipeline | None
_pipeline_stop = threading.Event()

_inference_thread: Optional[threading.Thread] = None
_infer_stop        = threading.Event()

# Leaky inference slots: cam_id → (frame, timestamp) | None
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

_events:     deque = deque(maxlen=500)
_events_lock = threading.Lock()

_queries:      list[str] = []
_queries_lock  = threading.Lock()

_threshold:      float = 0.3
_debug_overlay:  bool  = False

_query_status:      dict = {"state": "ready", "eta_s": 0}
_query_status_lock  = threading.Lock()
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
    existing["queries"] = new_queries
    tmp = META_JSON_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(existing, f, indent=2)
    os.replace(tmp, META_JSON_PATH)


def _triton_reload_model() -> None:
    """Unload then load the yoloworld model so Triton re-reads meta.json."""
    import urllib.request
    base = f"http://{TRITON_HTTP_URL}/v2/repository/models/yoloworld"
    for action in ("unload", "load"):
        req = urllib.request.Request(
            f"{base}/{action}",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=30)
            logger.info("Triton model yoloworld %s OK", action)
        except Exception as exc:
            logger.warning("Triton model %s failed: %s", action, exc)
        time.sleep(1)


def _query_update_worker(new_queries: list[str]) -> None:
    """Background thread: update meta.json and reload Triton model.
    Serialised by _query_update_lock so concurrent calls queue up."""
    with _query_update_lock:
        _do_query_update(new_queries)


def _do_query_update(new_queries: list[str]) -> None:
    with _query_status_lock:
        _query_status["state"] = "updating"
        _query_status["eta_s"] = 10

    logger.info("Updating queries to %s", new_queries)
    _save_queries_to_meta(new_queries)
    _triton_reload_model()

    with _query_status_lock:
        _query_status["state"] = "ready"
        _query_status["eta_s"] = 0

    logger.info("Query update complete: %s", new_queries)




def get_query_status() -> dict:
    with _query_status_lock:
        return dict(_query_status)


# ── GStreamer appsink callback ─────────────────────────────────────────────────

def _on_new_sample(appsink, _userdata):
    """
    Receives one tiled frame (height × width*N_cams).
    Slices into per-camera frames.
    Throttled broadcast at DISPLAY_FPS; throttled inference push at INFER_FPS.
    """
    sample = appsink.emit("pull-sample")
    if not sample:
        from gi.repository import Gst
        return Gst.FlowReturn.OK

    buf  = sample.get_buffer()
    caps = sample.get_caps()
    s    = caps.get_structure(0)
    w    = s.get_int("width").value
    h    = s.get_int("height").value

    from gi.repository import Gst
    ok, minfo = buf.map(Gst.MapFlags.READ)
    if not ok:
        return Gst.FlowReturn.OK

    tiled = (
        np.frombuffer(minfo.data, dtype=np.uint8)
        .reshape(h, w, 4)[:, :, :3]
        .copy()
    )
    buf.unmap(minfo)

    cam_ids = list(_streams.keys())
    n       = len(cam_ids)
    if n == 0:
        return Gst.FlowReturn.OK

    now     = time.monotonic()
    cam_w   = CAM_WIDTH   # each slice is CAM_WIDTH wide

    for i, cam_id in enumerate(cam_ids):
        frame = tiled[:, i * cam_w : (i + 1) * cam_w, :]

        # ── Display broadcast (25fps) ────────────────────────────────────────
        if now - _last_display.get(cam_id, 0.0) >= 1.0 / DISPLAY_FPS:
            _last_display[cam_id] = now
            disp = frame
            if _debug_overlay:
                disp = frame.copy()
                with _det_lock:
                    dets = list(_detections.get(cam_id, []))
                for det in dets:
                    x1, y1, x2, y2 = det.box
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        disp, f"{det.label} #{det.track_id}",
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )
            ok_jpg, jpeg = cv2.imencode(
                ".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
            if ok_jpg:
                _broadcast(cam_id, jpeg.tobytes())

        # ── Inference slot (10fps) ───────────────────────────────────────────
        if now - _last_infer.get(cam_id, 0.0) >= 1.0 / INFER_FPS:
            _last_infer[cam_id] = now
            with _infer_slots_lock:
                _infer_slots[cam_id] = (frame.copy(), now)
            _infer_event.set()

    return Gst.FlowReturn.OK


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
    import tritonclient.http as triton_http

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

        for cam_id, (frame, _ts) in slots.items():
            if _infer_stop.is_set():
                return

            ok, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue
            jpeg_bytes = jpeg_buf.tobytes()

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


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _build_and_start_pipeline(stream_map: dict[str, str]) -> None:
    """Build a new DeepStream pipeline for all streams and start it."""
    global _pipeline

    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    Gst.init(None)

    n = len(stream_map)
    if n == 0:
        return

    cam_ids = list(stream_map.keys())

    pipeline   = Gst.Pipeline.new("ds-pipeline")
    streammux  = Gst.ElementFactory.make("nvstreammux",        "mux")
    tiler      = Gst.ElementFactory.make("nvmultistreamtiler", "tiler")
    convert    = Gst.ElementFactory.make("nvvideoconvert",     "convert")
    capsfilter = Gst.ElementFactory.make("capsfilter",         "caps")
    sink       = Gst.ElementFactory.make("appsink",            "sink")

    for name, el in [("nvstreammux", streammux), ("nvmultistreamtiler", tiler),
                     ("nvvideoconvert", convert), ("capsfilter", capsfilter),
                     ("appsink", sink)]:
        if el is None:
            logger.error("Could not create GStreamer element '%s'", name)
            return

    streammux.set_property("batch-size",           n)
    streammux.set_property("width",                CAM_WIDTH)
    streammux.set_property("height",               CAM_HEIGHT)
    streammux.set_property("batched-push-timeout", 40000)

    tiler.set_property("rows",    1)
    tiler.set_property("columns", n)
    tiler.set_property("width",   CAM_WIDTH * n)
    tiler.set_property("height",  CAM_HEIGHT)

    capsfilter.set_property(
        "caps", Gst.Caps.from_string("video/x-raw,format=RGBA")
    )
    sink.set_property("emit-signals", True)
    sink.set_property("max-buffers",  2)
    sink.set_property("drop",         True)
    sink.set_property("sync",         False)
    sink.connect("new-sample", _on_new_sample, None)

    for el in [streammux, tiler, convert, capsfilter, sink]:
        pipeline.add(el)
    streammux.link(tiler)
    tiler.link(convert)
    convert.link(capsfilter)
    capsfilter.link(sink)

    for i, (cam_id, uri) in enumerate(stream_map.items()):
        src = Gst.ElementFactory.make("nvurisrcbin", f"src-{cam_id}")
        if src is None:
            logger.error("Could not create nvurisrcbin for %s", cam_id)
            return
        src.set_property("uri", uri)

        def on_pad_added(src_bin, new_pad, sink_id=i):
            try:
                sp = streammux.request_pad_simple(f"sink_{sink_id}")
            except AttributeError:
                sp = streammux.get_request_pad(f"sink_{sink_id}")
            if sp and not sp.is_linked():
                ret = new_pad.link(sp)
                if ret == Gst.PadLinkReturn.OK:
                    logger.info("cam%d → nvstreammux:sink_%d linked", sink_id, sink_id)

        src.connect("pad-added", on_pad_added)
        pipeline.add(src)

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        logger.error("DeepStream pipeline failed to start")
        pipeline.set_state(Gst.State.NULL)
        return

    _pipeline = pipeline
    logger.info("DeepStream pipeline started (%d camera(s))", n)

    # Watch bus for EOS — restart pipeline so file sources loop
    def _bus_watch(bus, msg, pipeline_ref):
        from gi.repository import Gst as _Gst
        if msg.type == _Gst.MessageType.EOS:
            logger.info("Pipeline EOS — seeking to start for loop")
            pipeline_ref.seek_simple(
                _Gst.Format.TIME,
                _Gst.SeekFlags.FLUSH | _Gst.SeekFlags.KEY_UNIT,
                0,
            )
        elif msg.type == _Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            logger.warning("Pipeline error: %s (%s)", err, dbg)
        return True  # keep watching

    bus = pipeline.get_bus()
    bus.add_watch(0, _bus_watch, pipeline)


def _stop_pipeline() -> None:
    global _pipeline
    if _pipeline is not None:
        try:
            _pipeline.set_state(__import__("gi").repository.Gst.State.NULL)
        except Exception:
            pass
        _pipeline = None
        logger.info("DeepStream pipeline stopped")


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

    _stop_inference()
    _stop_pipeline()

    removed = set(_streams) - set(new_streams)

    with _infer_slots_lock:
        _infer_slots.clear()
    _last_display.clear()
    _last_infer.clear()

    # Clean up state for cameras that are no longer active
    for cam_id in removed:
        _trackers.pop(cam_id, None)
        with _det_lock:
            _detections.pop(cam_id, None)

    _streams = dict(new_streams)

    if _streams:
        _build_and_start_pipeline(_streams)
        _start_inference()
        logger.info("DeepStream service running: %s", list(_streams.keys()))
    else:
        logger.info("DeepStream service idle — no streams")


# ── Public stream management ──────────────────────────────────────────────────

def add_stream(cam_id: str, uri: str) -> None:
    with _pipeline_lock:
        new = dict(_streams)
        new[cam_id] = uri
        _rebuild(new)


def remove_stream(cam_id: str) -> None:
    with _pipeline_lock:
        new = {k: v for k, v in _streams.items() if k != cam_id}
        _rebuild(new)


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
    text = text.strip()
    if not text:
        return False
    with _queries_lock:
        if text in _queries:
            return False
        new = list(_queries) + [text]
        _queries.clear()
        _queries.extend(new)   # update in-memory immediately
    threading.Thread(
        target=_query_update_worker, args=(new,), daemon=True, name="query-update"
    ).start()
    return True


def remove_query(index: int) -> bool:
    with _queries_lock:
        if not (0 <= index < len(_queries)):
            return False
        new = [q for i, q in enumerate(_queries) if i != index]
        _queries.clear()
        _queries.extend(new)   # update in-memory immediately
    threading.Thread(
        target=_query_update_worker, args=(new,), daemon=True, name="query-update"
    ).start()
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


# ── Startup: load persisted queries ──────────────────────────────────────────

def _init_queries() -> None:
    persisted = _load_queries_from_meta()
    with _queries_lock:
        if not _queries:
            _queries.extend(persisted)


_init_queries()
