"""
CameraRuntime — per-camera state object owned by the Registry.

Phase 2 split: replaces the per-camera module-level dicts (_streams, _rois,
_seg_ring, _clip_triggers) that used to live in deepstream_service.py with a
single object guarded by a per-camera lock. The Registry is the single source
of truth for active cameras.

Concurrency model: threading. Per-camera lock serialises mutation of one
runtime; the registry's lock guards add/remove/list of cameras and never
blocks on a runtime lock (no nested-lock deadlock path).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

CAMERAS_JSON_PATH = os.environ.get("CAMERAS_JSON_PATH", "./cameras.json")
SEG_RING_MAXLEN   = int(os.environ.get("MAX_SEG_FILES", "25"))


@dataclass
class CameraRuntime:
    """All per-camera state owned by the deepstream service.

    What lives here vs. elsewhere:
      - here: cam_id, rtsp_url, roi, rolling-segment ring, pending clip
        triggers, latest worker heartbeat (Phase 3)
      - inference container: tracker, motion_prev, detections, raw frame counters
        (heartbeated here every ~5s)
      - global (singletons in deepstream_service): event log, clip index, queries

    The lock guards mutation of fields on this dataclass. It is a regular
    threading.Lock — the watcher and clip manager only need brief snapshots,
    so contention is negligible at expected scale.
    """
    cam_id:        str
    rtsp_url:      str
    roi:           Optional[dict]   = None
    seg_ring:      deque            = field(default_factory=lambda: deque(maxlen=SEG_RING_MAXLEN))
    clip_triggers: list[dict]       = field(default_factory=list)
    lock:          threading.Lock   = field(default_factory=threading.Lock)
    # Phase 3 (observability): last heartbeat snapshot from inference worker.
    # Wire-format dict (decoded, motion_skipped, inferred, events, triton_errors,
    # last_decode_wall, last_triton_wall, last_event_wall, triton_ms_p50,
    # triton_ms_p99, reconnect_count, bytes_total).  Updated by
    # POST /internal/heartbeat; read by GET /diag/{cam_id} and /health.
    last_heartbeat:      Optional[dict] = None
    last_heartbeat_wall: float          = 0.0
    # Timestamp of first registration — used by /diag to suppress "down" during
    # the startup grace window (STARTUP_GRACE_S) while the worker connects.
    registered_at:       float          = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Serialise for /streams API and cameras.json persistence."""
        return {"url": self.rtsp_url, "roi": self.roi}


class _Registry:
    """Thread-safe registry of active CameraRuntimes.

    Single registry-level lock covers add/remove/list of cameras; the
    per-camera lock on each CameraRuntime covers mutation of its own state.
    The two locks NEVER need to be held simultaneously — registry holders
    take a snapshot under their lock, then release before touching runtimes.
    """

    def __init__(self) -> None:
        self._cams: dict[str, CameraRuntime] = {}
        self._lock = threading.Lock()

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._cams)

    def get(self, cam_id: str) -> Optional[CameraRuntime]:
        with self._lock:
            return self._cams.get(cam_id)

    def all(self) -> list[CameraRuntime]:
        with self._lock:
            return list(self._cams.values())

    def add(self, cam: CameraRuntime) -> None:
        with self._lock:
            self._cams[cam.cam_id] = cam

    def remove(self, cam_id: str) -> Optional[CameraRuntime]:
        with self._lock:
            return self._cams.pop(cam_id, None)

    def to_dict(self) -> dict[str, dict]:
        with self._lock:
            return {cid: cam.to_dict() for cid, cam in self._cams.items()}


# Module-level singleton — the one Registry the deepstream service uses
registry = _Registry()


# ── cameras.json persistence ─────────────────────────────────────────────────


def load_cameras_json() -> list[CameraRuntime]:
    """Read cameras.json on startup. Returns CameraRuntime list (not yet
    registered — callers do registry.add() + external wiring)."""
    try:
        with open(CAMERAS_JSON_PATH) as f:
            data = json.load(f)
    except FileNotFoundError:
        return []
    except Exception as exc:
        logger.warning("Failed to load cameras.json: %s", exc)
        return []
    out: list[CameraRuntime] = []
    for cam_id, v in data.items():
        if not isinstance(cam_id, str) or not isinstance(v, dict) or "url" not in v:
            continue
        out.append(CameraRuntime(cam_id=cam_id, rtsp_url=v["url"], roi=v.get("roi")))
    return out


def save_cameras_json() -> None:
    """Persist current registry to cameras.json.

    Direct write (not tmp+rename) because rename(2) fails with EBUSY on Docker
    bind-mounted single files — the inode change can't replace a bind mount.
    """
    data = registry.to_dict()
    try:
        with open(CAMERAS_JSON_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logger.warning("Failed to save cameras.json: %s", exc)
