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
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse as _JSONResponse

from services.ingress import INFERENCE_URL

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

# Phase 3 observability — tunables for /diag and /health.
HEARTBEAT_STALE_S       = float(os.environ.get("HEARTBEAT_STALE_S",       "15"))
DECODE_STALE_S          = float(os.environ.get("DECODE_STALE_S",          "10"))
INFERENCE_DIAG_TIMEOUT  = float(os.environ.get("INFERENCE_DIAG_TIMEOUT",  "2"))
STARTUP_GRACE_S         = float(os.environ.get("STARTUP_GRACE_S",         "30"))


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


# ── Phase 3 observability ────────────────────────────────────────────────────


def _inference_diag(cam_id: str) -> Optional[dict]:
    """Fetch the inference container's per-cam diag (worker pid + ffmpeg pid).
    Returns None on transport error so the caller can degrade gracefully."""
    url = f"{INFERENCE_URL}/diag/{urllib.parse.quote(cam_id, safe='')}"
    try:
        with urllib.request.urlopen(url, timeout=INFERENCE_DIAG_TIMEOUT) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001
        logger.debug("inference /diag/%s failed: %s", cam_id, exc)
        return None


def _diag_for_cam(cam_id: str) -> Optional[dict]:
    """Aggregate per-camera diag JSON. Returns None if the camera isn't registered.

    health = ok | degraded | down, with `reasons` listing every failed check
    so that a single GET shows which subsystem is sick without grepping logs."""
    cam = registry.get(cam_id)
    if cam is None:
        return None

    now = time.time()
    with cam.lock:
        hb            = cam.last_heartbeat or {}
        hb_wall       = cam.last_heartbeat_wall
        rtsp_url      = cam.rtsp_url
        triggers      = list(cam.clip_triggers)
        ring_size     = len(cam.seg_ring)
        registered_at = cam.registered_at

    in_grace    = (now - registered_at) < STARTUP_GRACE_S
    inf_diag    = _inference_diag(cam_id)
    worker_alive = bool(inf_diag.get("alive")) if inf_diag else None
    worker_pid   = inf_diag.get("pid") if inf_diag else None
    ffmpeg_alive = inf_diag.get("ffmpeg_alive") if inf_diag else None
    ffmpeg_pid   = inf_diag.get("ffmpeg_pid")   if inf_diag else None

    last_decode  = float(hb.get("last_decode_wall", 0) or 0)
    last_triton  = float(hb.get("last_triton_wall", 0) or 0)
    last_event   = float(hb.get("last_event_wall",  0) or 0)
    decoded      = int(hb.get("decoded", 0) or 0)
    skipped      = int(hb.get("motion_skipped", 0) or 0)
    motion_pct   = round(100.0 * skipped / decoded, 1) if decoded else 0.0

    cam_clips = [c for c in clip_index.list_all() if c["cam_id"] == cam_id]
    clips_total   = len(cam_clips)
    last_clip_ts  = cam_clips[0]["timestamp"] if cam_clips else None

    reasons: list[str] = []
    if hb_wall == 0:
        if not in_grace:
            reasons.append("no heartbeat received yet")
    elif now - hb_wall > HEARTBEAT_STALE_S:
        reasons.append(f"heartbeat stale ({now - hb_wall:.1f}s)")
    if last_decode and now - last_decode > DECODE_STALE_S:
        reasons.append(f"no decoded frames for {now - last_decode:.1f}s")
    if worker_alive is False:
        reasons.append("inference worker subprocess dead")
    if ffmpeg_alive is False:
        reasons.append("ffmpeg clip recorder dead")

    if not hb_wall and worker_alive is None:
        # No signal yet from either the heartbeat thread or inference container.
        # Suppress "down" during the startup grace window — worker may still be
        # connecting to Triton and MediaMTX.
        health = "ok" if in_grace else "down"
    elif reasons:
        health = "degraded"
    else:
        health = "ok"

    return {
        "cam_id": cam_id,
        "ingress": {
            "rtsp_url":         rtsp_url,
            "last_frame_ts":    last_decode or None,
            "last_frame_age_s": (now - last_decode) if last_decode else None,
            "reconnect_count":  int(hb.get("reconnect_count", 0) or 0),
            "bytes_total":      int(hb.get("bytes_total", 0) or 0),
            "ffmpeg_alive":     ffmpeg_alive,
            "ffmpeg_pid":       ffmpeg_pid,
        },
        "inference": {
            "worker_alive":         worker_alive,
            "worker_pid":           worker_pid,
            "last_triton_ts":       last_triton or None,
            "last_triton_age_s":    (now - last_triton) if last_triton else None,
            "last_event_ts":        last_event or None,
            "last_event_age_s":     (now - last_event) if last_event else None,
            "triton_ms_p50":        float(hb.get("triton_ms_p50", 0.0) or 0.0),
            "triton_ms_p99":        float(hb.get("triton_ms_p99", 0.0) or 0.0),
            "motion_gate_skip_pct": motion_pct,
            "triton_errors":        int(hb.get("triton_errors", 0) or 0),
            "decoded":              decoded,
            "inferred":             int(hb.get("inferred", 0) or 0),
        },
        "clips": {
            "active_sessions": len(triggers),
            "last_clip_ts":    last_clip_ts,
            "clips_total":     clips_total,
            "seg_ring_size":   ring_size,
        },
        "heartbeat_age_s": (now - hb_wall) if hb_wall else None,
        "health":          health,
        "reasons":         reasons,
        "startup_grace":   in_grace,
        "registered_at":   registered_at,
    }


def _compute_health() -> tuple[str, list[str]]:
    """Aggregate per-camera health into a single status. Returns (status, reasons).

    `down` only if every registered camera is down. `degraded` if any camera is
    not ok. `ok` otherwise (including when no cameras are registered)."""
    cam_ids = registry.list_ids()
    if not cam_ids:
        return "ok", []

    reasons: list[str] = []
    statuses: list[str] = []
    for cid in cam_ids:
        diag = _diag_for_cam(cid)
        if diag is None:
            continue
        statuses.append(diag["health"])
        for r in diag["reasons"]:
            reasons.append(f"{cid}: {r}")

    if statuses and all(s == "down" for s in statuses):
        return "down", reasons
    if any(s != "ok" for s in statuses):
        return "degraded", reasons
    return "ok", []


# ── FastAPI ──────────────────────────────────────────────────────────────────


app = FastAPI(title="DeepStream Service")


@app.get("/health")
async def _health():
    """Aggregate health across all registered cameras + Triton + MediaMTX.
    Returns 200 when ok, 503 when degraded/down, with named reasons."""
    status, reasons = await asyncio.to_thread(_compute_health)
    body = {"status": status, "reasons": reasons}
    if status == "ok":
        return _JSONResponse(body)
    return _JSONResponse(body, status_code=503)


@app.get("/diag/{cam_id}")
async def _diag(cam_id: str):
    diag = await asyncio.to_thread(_diag_for_cam, cam_id)
    if diag is None:
        return _JSONResponse({"error": f"cam_id '{cam_id}' not found"}, status_code=404)
    return _JSONResponse(diag)


@app.post("/internal/heartbeat")
async def _internal_heartbeat(payload: dict):
    """Inference worker stats heartbeat (every ~5s).  Stored on the camera's
    runtime for /diag aggregation; never appended to events."""
    cam_id = (payload.get("cam_id") or "").strip()
    if not cam_id:
        return _JSONResponse({"error": "cam_id required"}, status_code=400)
    cam = registry.get(cam_id)
    if cam is None:
        # Worker may briefly outlive its DELETE — accept silently to avoid
        # spamming worker logs with 404s during teardown.
        return _JSONResponse({"ok": True, "registered": False})
    with cam.lock:
        cam.last_heartbeat      = payload
        cam.last_heartbeat_wall = time.time()
    return _JSONResponse({"ok": True, "registered": True})


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
