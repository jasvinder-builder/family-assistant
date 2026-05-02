"""
DeepStream service — slim FastAPI router.

Phase 2 split (2026-05-02): real work lives in:
  - services.camera_runtime  — CameraRuntime + Registry + cameras.json
  - services.ingress         — MediaMTX + inference HTTP clients
  - services.clips           — segment watcher + clip mgr + cutter + index

This module is the FastAPI app, the events feed, the queries+threshold globals,
and the per-camera lifecycle wrappers (add_stream / remove_stream / set_roi).
No more _rebuild — every operation touches one CameraRuntime under its own
lock; background loops (seg watcher, clip manager) run continuously and
iterate the registry each tick.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse as _JSONResponse

from services.camera_runtime import (
    CameraRuntime,
    load_cameras_json,
    registry,
    save_cameras_json,
)
from services.clips import (
    CLIPS_DIR,
    clip_index,
    clip_manager,
    seg_watcher,
    trigger_clip_on_detection,
)
from services.ingress import (
    inference_add_camera,
    inference_remove_camera,
    mediamtx_add_path,
    mediamtx_remove_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

META_JSON_PATH = os.environ.get("META_JSON_PATH", "/app/models/yoloworld.meta.json")


# ── Events feed (global, not per-camera) ─────────────────────────────────────


@dataclass
class CameraEvent:
    timestamp:  str
    query:      str
    track_id:   int
    confidence: float
    image_b64:  str
    cam_id:     str = "cam0"


class _EventLog:
    """Thread-safe rolling deque of recent events. Single global instance —
    events are a unified feed across all cameras."""

    def __init__(self, maxlen: int = 500) -> None:
        self._events: deque[CameraEvent] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, e: CameraEvent) -> None:
        with self._lock:
            self._events.append(e)

    def list_recent(self, max_age_hours: float = 1.0,
                    cam_id: Optional[str] = None) -> list[dict]:
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        with self._lock:
            return [
                {
                    "timestamp":  e.timestamp,
                    "query":      e.query,
                    "confidence": round(e.confidence, 3),
                    "image_b64":  e.image_b64,
                    "cam_id":     e.cam_id,
                }
                for e in self._events
                if (datetime.fromisoformat(e.timestamp) >= cutoff
                    and (cam_id is None or e.cam_id == cam_id))
            ]


event_log = _EventLog()


# ── Queries (the YOLO-World prompt list, persisted in meta.json) ─────────────


class _Queries:
    """Thread-safe query list backed by yoloworld.meta.json on disk.

    `engine_queries` (the baked TRT text embeddings) is left untouched here;
    only `queries` (the active list) is updated. main.py orchestrates the
    re-export; this class just keeps the local label-mapping in sync."""

    def __init__(self) -> None:
        self._items: list[str] = []
        self._lock = threading.Lock()

    def get(self) -> list[str]:
        with self._lock:
            return list(self._items)

    def commit(self, new: list[str]) -> None:
        with self._lock:
            self._items.clear()
            self._items.extend(new)
            try:
                existing = json.loads(open(META_JSON_PATH).read())
            except Exception:
                existing = {"queries": [], "imgsz": 640}
            existing["queries"] = new
            tmp = META_JSON_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(existing, f, indent=2)
            os.replace(tmp, META_JSON_PATH)
        logger.info("Queries committed: %s", new)

    def add(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        with self._lock:
            if text in self._items:
                return False
            new = list(self._items) + [text]
        self.commit(new)
        return True

    def remove(self, index: int) -> bool:
        with self._lock:
            if not (0 <= index < len(self._items)):
                return False
            new = [q for i, q in enumerate(self._items) if i != index]
        self.commit(new)
        return True

    def init_from_meta(self) -> None:
        try:
            persisted = json.loads(open(META_JSON_PATH).read())["queries"]
        except Exception:
            persisted = ["person", "car", "dog"]
        with self._lock:
            if not self._items:
                self._items.extend(persisted)


queries = _Queries()

# Small global state — float / bool, not collections
threshold:     float = 0.3
debug_overlay: bool  = False


# ── Per-camera lifecycle ─────────────────────────────────────────────────────


def add_stream(cam_id: str, uri: str, roi: Optional[dict] = None) -> None:
    """Register a camera or update its source URL.

    Touches one CameraRuntime under its own lock; no global rebuild. External
    HTTP calls (MediaMTX + inference) happen outside the cam lock so we don't
    block on the network while holding it."""
    cam = registry.get(cam_id)
    if cam is None:
        cam = CameraRuntime(cam_id=cam_id, rtsp_url=uri, roi=roi)
        registry.add(cam)
    else:
        with cam.lock:
            cam.rtsp_url = uri
            if roi is not None:
                cam.roi = roi

    mediamtx_add_path(cam_id, uri)
    inference_add_camera(cam_id)
    save_cameras_json()


def remove_stream(cam_id: str) -> None:
    """Unregister a camera. Idempotent — silent on missing cam_id."""
    cam = registry.remove(cam_id)
    if cam is None:
        save_cameras_json()
        return
    inference_remove_camera(cam_id)
    mediamtx_remove_path(cam_id)
    save_cameras_json()


def set_roi(cam_id: str, roi: Optional[dict]) -> None:
    """Update the ROI for a camera. The inference worker does not yet honour
    ROI on the new path; this stores it for Phase 3 plumbing and persists it
    to cameras.json so it survives restart."""
    cam = registry.get(cam_id)
    if cam is None:
        raise KeyError(cam_id)
    with cam.lock:
        cam.roi = roi
    save_cameras_json()


def get_streams() -> dict[str, dict]:
    return registry.to_dict()


# ── Backward-compatible wrappers (tests + scripts call these) ────────────────


def get_queries() -> list[str]:
    return queries.get()


def add_query(text: str) -> bool:
    return queries.add(text)


def remove_query(index: int) -> bool:
    return queries.remove(index)


def get_threshold() -> float:
    return threshold


def set_threshold(value: float) -> None:
    global threshold
    threshold = max(0.0, min(1.0, value))


def get_debug_overlay() -> bool:
    return debug_overlay


def set_debug_overlay(enabled: bool) -> None:
    global debug_overlay
    debug_overlay = bool(enabled)


def get_pad_factor() -> float:
    return 0.0


def set_pad_factor(_: float) -> None:
    pass


def get_latest_detections(cam_id: Optional[str] = None) -> list:
    """Live overlay detections moved into the inference subprocess (Phase 1).
    Empty stub kept until /diag in Phase 3 exposes per-cam state."""
    return []


def get_events(max_age_hours: float = 1.0,
               cam_id: Optional[str] = None) -> list[dict]:
    return event_log.list_recent(max_age_hours=max_age_hours, cam_id=cam_id)


# Single-camera shims (legacy — pre-multi-camera API)
def set_stream_url(url: str) -> None:
    url = url.strip() if url else ""
    if url:
        add_stream("cam0", url)
    else:
        remove_stream("cam0")


def get_stream_url() -> Optional[str]:
    cam = registry.get("cam0")
    return cam.rtsp_url if cam else None


def start_analysis(url: str) -> None:
    add_stream("cam0", url)


def stop_analysis() -> None:
    remove_stream("cam0")


def push_frame(_frame) -> None:
    """No-op — inference container reads RTSP directly. Kept for API compat."""
    pass


# ── Startup ──────────────────────────────────────────────────────────────────


def _startup() -> None:
    queries.init_from_meta()
    clip_index.scan_disk()
    for cam in load_cameras_json():
        logger.info("Restoring camera from cameras.json: %s → %s (roi=%s)",
                    cam.cam_id, cam.rtsp_url, cam.roi)
        registry.add(cam)
        mediamtx_add_path(cam.cam_id, cam.rtsp_url)
        inference_add_camera(cam.cam_id)
    seg_watcher.start()
    clip_manager.start()
    if registry.list_ids():
        logger.info("DeepStream service running: %s", registry.list_ids())
    else:
        logger.info("DeepStream service idle — no streams")


_startup()


# ── FastAPI ──────────────────────────────────────────────────────────────────


app = FastAPI(title="DeepStream Service")


@app.get("/health")
async def _health():
    return {"ok": True}


@app.post("/internal/trigger")
async def _internal_trigger(payload: dict):
    """Per-detection trigger ping from an inference worker. Extends the clip
    window for this camera. Called at the worker's frame rate; does NOT
    append to /events (events are rate-limited and posted via /internal/event)."""
    cam_id = (payload.get("cam_id") or "").strip()
    query  = (payload.get("query")  or "").strip()
    if not cam_id or not query:
        return _JSONResponse({"error": "cam_id and query required"}, status_code=400)
    trigger_clip_on_detection(cam_id, query, time.time())
    return _JSONResponse({"ok": True})


@app.post("/internal/event")
async def _internal_event(payload: dict):
    """Receive a (rate-limited) detection event from an inference worker
    subprocess and fold it into the events feed. Also extends the clip window."""
    cam_id = (payload.get("cam_id") or "").strip()
    query  = (payload.get("query")  or "").strip()
    if not cam_id or not query:
        return _JSONResponse({"error": "cam_id and query required"}, status_code=400)
    try:
        track_id   = int(payload.get("track_id") or 0)
        confidence = float(payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return _JSONResponse(
            {"error": "track_id and confidence must be numeric"}, status_code=400,
        )

    image_b64 = str(payload.get("image_b64") or "")
    ts        = str(payload.get("timestamp")
                    or datetime.now().isoformat(timespec="seconds"))

    event_log.push(CameraEvent(
        timestamp=ts, query=query, track_id=track_id,
        confidence=confidence, image_b64=image_b64, cam_id=cam_id,
    ))
    trigger_clip_on_detection(cam_id, query, time.time())
    return _JSONResponse({"ok": True})


@app.get("/streams")
async def _list_streams():
    return _JSONResponse({"streams": registry.to_dict()})


@app.post("/streams")
async def _add_stream(payload: dict):
    cam_id = payload.get("cam_id", "").strip()
    url    = payload.get("url", "").strip()
    if not cam_id:
        return _JSONResponse({"error": "cam_id is required"}, status_code=400)
    if not url:
        return _JSONResponse({"error": "url is required"}, status_code=400)
    await asyncio.to_thread(add_stream, cam_id, url)
    return _JSONResponse({"ok": True, "streams": registry.to_dict()})


@app.delete("/streams/{cam_id}")
async def _remove_stream(cam_id: str):
    if registry.get(cam_id) is None:
        return _JSONResponse({"error": f"stream '{cam_id}' not found"}, status_code=404)
    await asyncio.to_thread(remove_stream, cam_id)
    return _JSONResponse({"ok": True, "streams": registry.to_dict()})


@app.patch("/streams/{cam_id}/roi")
async def _set_roi(cam_id: str, payload: dict):
    if registry.get(cam_id) is None:
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
    return _JSONResponse(
        {"ok": True, "cam_id": cam_id, "roi": roi, "streams": registry.to_dict()},
    )


@app.get("/queries")
async def _list_queries():
    return _JSONResponse({"queries": queries.get()})


@app.post("/queries")
async def _add_query(payload: dict):
    text = payload.get("text", "").strip()
    if not text:
        return _JSONResponse({"error": "text is required"}, status_code=400)
    added = queries.add(text)
    return _JSONResponse({"ok": True, "added": added, "queries": queries.get()})


@app.delete("/queries/{index}")
async def _remove_query(index: int):
    removed = queries.remove(index)
    if not removed:
        return _JSONResponse({"error": "index out of range"}, status_code=404)
    return _JSONResponse({"ok": True, "queries": queries.get()})


@app.get("/queries/status")
async def _queries_status_route():
    return _JSONResponse({"state": "ready", "eta_s": 0, "queries": queries.get()})


@app.post("/queries/commit")
async def _queries_commit_route(payload: dict):
    qs = payload.get("queries", [])
    if not isinstance(qs, list):
        return _JSONResponse({"error": "queries must be a list"}, status_code=400)
    queries.commit(qs)
    return _JSONResponse({"ok": True, "queries": queries.get()})


@app.get("/events")
async def _events_route():
    return _JSONResponse({"events": event_log.list_recent()})


@app.get("/threshold")
async def _get_threshold():
    return _JSONResponse({"threshold": threshold})


@app.post("/threshold")
async def _set_threshold(payload: dict):
    try:
        value = float(payload.get("value"))
    except (TypeError, ValueError):
        return _JSONResponse({"error": "value must be a number"}, status_code=400)
    set_threshold(value)
    return _JSONResponse({"ok": True, "threshold": threshold})


@app.post("/debug-overlay")
async def _debug_overlay_route(payload: dict):
    set_debug_overlay(bool(payload.get("enabled", False)))
    return _JSONResponse({"ok": True, "enabled": debug_overlay})


@app.post("/set-stream")
async def _set_stream(payload: dict):
    url = payload.get("url", "").strip()
    set_stream_url(url)
    return _JSONResponse({"ok": True})


@app.get("/clips")
async def _list_clips(cam_id: Optional[str] = None):
    return _JSONResponse({"clips": clip_index.list_all(cam_id)})


@app.get("/clips/file/{cam_id}/{filename}")
async def _serve_clip(cam_id: str, filename: str):
    if "/" in cam_id or "/" in filename or ".." in cam_id or ".." in filename:
        return _JSONResponse({"error": "invalid path"}, status_code=400)
    path = CLIPS_DIR / cam_id / filename
    if not path.exists():
        return _JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(path), media_type="video/mp4")
