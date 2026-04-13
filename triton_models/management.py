"""
Triton management sidecar — TRT re-export on demand.

Runs on port 8004 alongside tritonserver inside the triton container.
Called by deepstream_service.py when queries change.

POST /reexport          {"queries": [...], "model": "yolov8m-worldv2", "imgsz": 640}
GET  /reexport/status   {"state": "idle"|"running"|"done"|"error", "eta_s": N, "queries": [...]}
GET  /health            {"ok": true}
"""

import json
import pathlib
import shutil
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

ENGINE_PATH = pathlib.Path("/trt_engines/yoloworld.engine")
META_PATH   = pathlib.Path("/trt_engines/yoloworld.meta.json")

_status: dict[str, Any] = {"state": "idle", "eta_s": 0, "queries": []}
_lock = threading.Lock()


class ReexportRequest(BaseModel):
    queries: list[str]
    model:   str = "yolov8m-worldv2"
    imgsz:   int = 640


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/reexport/status")
def reexport_status():
    with _lock:
        return dict(_status)


@app.post("/reexport")
def start_reexport(req: ReexportRequest):
    with _lock:
        if _status["state"] == "running":
            raise HTTPException(409, "Re-export already in progress")
        _status.update({"state": "running", "eta_s": 90, "queries": req.queries})

    threading.Thread(
        target=_run_export,
        args=(req.queries, req.model, req.imgsz),
        daemon=True,
        name="trt-reexport",
    ).start()
    return {"state": "running", "eta_s": 90}


def _run_export(queries: list[str], model_name: str, imgsz: int) -> None:
    t0 = time.monotonic()
    try:
        from ultralytics import YOLOWorld

        print(f"[management] TRT re-export started: queries={queries}", flush=True)

        model = YOLOWorld(f"{model_name}.pt")
        model.set_classes(queries)

        raw_path = model.export(
            format="engine",
            half=True,
            imgsz=imgsz,
            simplify=True,
            dynamic=False,
            workspace=4,
            verbose=True,
        )

        # Atomic swap — keep a .bak of the previous engine
        new_engine = ENGINE_PATH.with_suffix(".engine.new")
        shutil.move(raw_path, new_engine)
        bak = ENGINE_PATH.with_suffix(".engine.bak")
        bak.unlink(missing_ok=True)
        if ENGINE_PATH.exists():
            ENGINE_PATH.rename(bak)
        new_engine.rename(ENGINE_PATH)

        # Update meta.json: keep all existing fields, sync both query fields
        try:
            meta: dict = json.loads(META_PATH.read_text())
        except Exception:
            meta = {"imgsz": imgsz}
        meta["queries"]        = queries
        meta["engine_queries"] = queries   # now in sync with the new TRT engine
        meta["model"]          = model_name
        meta["exported_at"]    = time.strftime("%Y-%m-%dT%H:%M:%S")
        tmp = META_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2))
        tmp.rename(META_PATH)

        elapsed = time.monotonic() - t0
        print(f"[management] TRT re-export done in {elapsed:.0f}s — queries={queries}", flush=True)
        with _lock:
            _status.update({"state": "done", "eta_s": 0, "queries": queries})

    except Exception as exc:
        import traceback
        traceback.print_exc()
        with _lock:
            _status.update({"state": "error", "eta_s": 0, "queries": queries, "error": str(exc)})
