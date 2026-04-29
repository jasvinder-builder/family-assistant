"""
Phase 1 inference control plane — FastAPI + subprocess-per-camera.

Replaces, at the architectural level: bianca-savant-module + per-camera
bianca-rtsp-{cam_id} + per-camera bianca-sink-{cam_id} + the inference
loop inside bianca-deepstream.

Each camera runs as its own subprocess executing services/inference_worker.py.
The parent tracks subprocess lifecycle and exposes /diag for observability.

Endpoints:
  GET    /                  → smoke
  GET    /cameras           → list registry { cam_id: {pid, alive, started_at} }
  POST   /cameras           → {cam_id, rtsp_url} → spawn subprocess
  DELETE /cameras/{cam_id}  → SIGTERM subprocess, await exit
  GET    /diag/{cam_id}     → {pid, alive, returncode, started_at,
                                last_stderr_lines}

Subprocess-per-camera is the deliberate concurrency model — Phase 1.3
prototype proved threading + h264_cuvid + cv2 + Triton in one process
starves all but the first decoder.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse


# ── Config ────────────────────────────────────────────────────────────────────

WORKER_SCRIPT = os.environ.get(
    "WORKER_SCRIPT", "/app/services/inference_worker.py"
)
TRITON_URL = os.environ.get("PROTOTYPE_TRITON_URL", "bianca-triton:8002")
META_PATH = os.environ.get(
    "PROTOTYPE_META", "/data/models/yoloworld.meta.json"
)
# A sentinel "very long" duration — the parent owns lifecycle, not the child.
WORKER_DURATION_S = int(os.environ.get("WORKER_DURATION_S", "86400"))


# ── Per-camera registry entry ────────────────────────────────────────────────


@dataclass
class _CameraProc:
    cam_id: str
    rtsp_url: str
    proc: subprocess.Popen
    started_at: float
    stderr_buf: deque = field(default_factory=lambda: deque(maxlen=20))
    stderr_thread: Optional[threading.Thread] = None

    def alive(self) -> bool:
        return self.proc.poll() is None

    def returncode(self) -> Optional[int]:
        return self.proc.returncode if not self.alive() else None


# ── Registry ─────────────────────────────────────────────────────────────────


class _Registry:
    """Thread-safe per-camera subprocess registry.

    Single global lock — fine at this scale (handful of cameras, slow ops).
    The Phase 2 plan calls for one lock per CameraRuntime; for the prototype's
    tiny surface area the global lock is simpler and adequate.
    """

    def __init__(self) -> None:
        self._procs: dict[str, _CameraProc] = {}
        self._lock = threading.Lock()

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._procs)

    def get(self, cam_id: str) -> Optional[_CameraProc]:
        with self._lock:
            return self._procs.get(cam_id)

    def add(self, cam_id: str, rtsp_url: str) -> _CameraProc:
        cmd = [
            sys.executable, WORKER_SCRIPT,
            "--rtsp", rtsp_url,
            "--triton", TRITON_URL,
            "--meta", META_PATH,
            "--duration", str(WORKER_DURATION_S),
        ]
        # Tag stdout/stderr with cam_id so the parent's logs are readable
        env = dict(os.environ)
        env["PROTOTYPE_CAM_ID"] = cam_id
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,        # events not consumed in prototype
            stderr=subprocess.PIPE,           # for crash diagnostics
            text=True,
            env=env,
        )
        cp = _CameraProc(
            cam_id=cam_id, rtsp_url=rtsp_url,
            proc=proc, started_at=time.time(),
        )

        def _drain_stderr():
            assert proc.stderr is not None
            for line in proc.stderr:
                cp.stderr_buf.append(line.rstrip())

        cp.stderr_thread = threading.Thread(
            target=_drain_stderr, daemon=True, name=f"stderr-{cam_id}",
        )
        cp.stderr_thread.start()

        with self._lock:
            existing = self._procs.get(cam_id)
        if existing is not None:
            self._terminate(existing)
        with self._lock:
            self._procs[cam_id] = cp
        return cp

    def remove(self, cam_id: str) -> bool:
        with self._lock:
            cp = self._procs.pop(cam_id, None)
        if cp is None:
            return False
        self._terminate(cp)
        return True

    def shutdown_all(self) -> None:
        with self._lock:
            ids = list(self._procs)
        for cid in ids:
            self.remove(cid)

    @staticmethod
    def _terminate(cp: _CameraProc, timeout_s: int = 5) -> None:
        if cp.proc.poll() is not None:
            return
        try:
            cp.proc.terminate()
            cp.proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            cp.proc.kill()
            cp.proc.wait(timeout=2)
        except Exception:
            pass


# ── App ──────────────────────────────────────────────────────────────────────


_registry = _Registry()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Forward Ctrl-C to subprocess teardown
    def _on_signal(sig, _frame):
        print(f"[svc] signal {sig} — shutting children down", flush=True)
        _registry.shutdown_all()
        # Default handler will exit the process

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    yield
    _registry.shutdown_all()


app = FastAPI(title="Phase 1.4 inference control plane", lifespan=_lifespan)


@app.get("/")
def root():
    return {"status": "ok", "cameras": _registry.list_ids()}


@app.get("/cameras")
def list_cameras():
    out = {}
    for cid in _registry.list_ids():
        cp = _registry.get(cid)
        if cp is None:
            continue
        out[cid] = {
            "pid": cp.proc.pid,
            "alive": cp.alive(),
            "returncode": cp.returncode(),
            "started_at": cp.started_at,
            "rtsp_url": cp.rtsp_url,
        }
    return {"cameras": out}


@app.post("/cameras")
def add_camera(payload: dict):
    cam_id = (payload.get("cam_id") or "").strip()
    rtsp_url = (payload.get("rtsp_url") or "").strip()
    if not cam_id or not rtsp_url:
        raise HTTPException(status_code=400, detail="cam_id and rtsp_url required")

    cp = _registry.add(cam_id, rtsp_url)
    # Quick sanity: the process should still be alive ~0.5s after spawn.
    # If not, surface the stderr immediately so the caller can debug.
    time.sleep(0.5)
    if not cp.alive():
        return JSONResponse(
            {
                "ok": False,
                "cam_id": cam_id,
                "returncode": cp.returncode(),
                "stderr_tail": list(cp.stderr_buf),
            },
            status_code=500,
        )
    return {"ok": True, "cam_id": cam_id, "pid": cp.proc.pid}


@app.delete("/cameras/{cam_id}")
def remove_camera(cam_id: str):
    if not _registry.remove(cam_id):
        raise HTTPException(status_code=404, detail="cam_id not found")
    return {"ok": True, "cam_id": cam_id}


@app.get("/diag/{cam_id}")
def diag(cam_id: str):
    cp = _registry.get(cam_id)
    if cp is None:
        raise HTTPException(status_code=404, detail="cam_id not found")
    return {
        "cam_id": cam_id,
        "pid": cp.proc.pid,
        "alive": cp.alive(),
        "returncode": cp.returncode(),
        "started_at": cp.started_at,
        "rtsp_url": cp.rtsp_url,
        "stderr_tail": list(cp.stderr_buf),
    }


# ── Standalone entrypoint ────────────────────────────────────────────────────


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8091"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
