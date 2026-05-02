"""
Ingress — HTTP clients for MediaMTX (path mgmt) and inference (worker spawn).

Stateless: all per-camera state is in CameraRuntime. These functions just make
the HTTP calls; failures log and return False/None — callers decide whether to
proceed (we proceed: cameras.json is the source of truth, mediamtx and
inference reconnect on their own once they're healthy).
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

MEDIAMTX_API_URL  = os.environ.get("MEDIAMTX_API_URL",  "http://bianca-mediamtx:9997")
MEDIAMTX_RTSP_URL = os.environ.get("MEDIAMTX_RTSP_URL", "rtsp://bianca-mediamtx:8554")
INFERENCE_URL     = os.environ.get("INFERENCE_URL",     "http://bianca-inference:8091")


# ── MediaMTX path management ─────────────────────────────────────────────────


def _mediamtx_call(method: str, action: str, cam_id: str,
                   body: Optional[dict]) -> tuple[int, str]:
    """Call the MediaMTX HTTP API.  Returns (status, body) — 0 on transport error.
    Note action verbs vary by method: POST add/replace/patch, DELETE delete."""
    url = f"{MEDIAMTX_API_URL}/v3/config/paths/{action}/{urllib.parse.quote(cam_id, safe='')}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read().decode(errors="replace")
    except urllib.request.HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace") if exc.fp else ""
    except Exception as exc:
        logger.warning("MediaMTX %s %s %s failed: %s", method, action, cam_id, exc)
        return 0, str(exc)


def mediamtx_add_path(cam_id: str, source: str) -> bool:
    """Register a path that pulls from `source` and re-streams on RTSP/HLS/WebRTC.

    Idempotent: if the path already exists, replace it.  rtspTransport=tcp
    matches every camera in this project; sourceOnDemand=false so the inference
    worker can connect even before a viewer is watching.
    """
    body = {"source": source, "sourceOnDemand": False, "rtspTransport": "tcp"}
    status, _ = _mediamtx_call("POST", "add", cam_id, body)
    if 200 <= status < 300:
        return True
    if status == 400:
        # Most common reason is "path already exists" — replace it instead.
        status2, msg = _mediamtx_call("POST", "replace", cam_id, body)
        if 200 <= status2 < 300:
            return True
        logger.warning("MediaMTX replace %s failed (status=%d): %s",
                       cam_id, status2, msg[:200])
        return False
    logger.warning("MediaMTX add %s failed (status=%d)", cam_id, status)
    return False


def mediamtx_remove_path(cam_id: str) -> None:
    """Remove a path. Note the verb is `delete` and the HTTP method is DELETE
    (POST returns 404 — silent failure mode that bit Phase 1)."""
    status, msg = _mediamtx_call("DELETE", "delete", cam_id, None)
    if status not in (200, 204, 404):
        logger.warning("MediaMTX remove %s status=%d: %s", cam_id, status, msg[:200])


# ── Inference control plane ──────────────────────────────────────────────────


def inference_add_camera(cam_id: str) -> bool:
    """Tell the inference service to spawn a worker subprocess for this camera.
    The worker pulls RTSP from MediaMTX's re-stream, not from the original source."""
    rtsp_url = f"{MEDIAMTX_RTSP_URL}/{urllib.parse.quote(cam_id, safe='')}"
    body = json.dumps({"cam_id": cam_id, "rtsp_url": rtsp_url}).encode()
    req = urllib.request.Request(
        f"{INFERENCE_URL}/cameras", data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=10).close()
        return True
    except Exception as exc:
        logger.warning("Inference add_camera %s failed: %s", cam_id, exc)
        return False


def inference_remove_camera(cam_id: str) -> None:
    req = urllib.request.Request(
        f"{INFERENCE_URL}/cameras/{urllib.parse.quote(cam_id, safe='')}",
        method="DELETE",
    )
    try:
        urllib.request.urlopen(req, timeout=10).close()
    except Exception as exc:
        logger.debug("Inference remove_camera %s: %s", cam_id, exc)
