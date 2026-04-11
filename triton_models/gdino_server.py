"""
Grounding DINO inference server — FastAPI/uvicorn on GPU.

Replaces the Triton Python backend, which has a bug in Triton 24.04 where
every tensor element is overwritten with the first element's value when data
is transferred via the Python backend shared-memory IPC channel.

Endpoints
---------
POST /infer
    multipart/form-data:
        image           JPEG bytes
        queries         JSON list of query strings, e.g. '["person","dog"]'
        threshold       float, box confidence threshold (default 0.35)
        text_threshold  float, per-token threshold (default 0.25)
    Returns JSON: {boxes: [[x1,y1,x2,y2],...], scores: [...], labels: [...]}

GET /health
    Returns {"status": "ok"}
"""

import json
import os

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

app = FastAPI()

_MAX_W = 800
_device: str = "cpu"
_processor = None
_model = None


@app.on_event("startup")
async def _load_model():
    global _processor, _model, _device
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "IDEA-Research/grounding-dino-tiny"
    print(f"[gdino] Loading on {_device}...", flush=True)
    _processor = AutoProcessor.from_pretrained(model_id)
    dtype = torch.float16 if _device == "cuda" else torch.float32
    _model = (
        AutoModelForZeroShotObjectDetection
        .from_pretrained(model_id, torch_dtype=dtype)
        .to(_device)
        .eval()
    )
    print(
        f"[gdino] Ready on {_device} "
        f"({'fp16' if _device == 'cuda' else 'fp32'})",
        flush=True,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/infer")
async def infer(
    image: UploadFile = File(...),
    queries: str = Form(...),
    threshold: float = Form(0.35),
    text_threshold: float = Form(0.25),
):
    queries_list: list[str] = json.loads(queries)

    jpeg_bytes = await image.read()
    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        return {"boxes": [], "scores": [], "labels": [], "error": "bad JPEG"}

    orig_h, orig_w = bgr.shape[:2]
    if orig_w > _MAX_W:
        scale = _MAX_W / orig_w
        bgr = cv2.resize(
            bgr, (_MAX_W, int(orig_h * scale)), interpolation=cv2.INTER_AREA
        )

    pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    text = " . ".join(q.strip().lower().rstrip(" .") for q in queries_list) + " ."

    inputs = _processor(images=pil_img, text=text, return_tensors="pt").to(_device)

    with torch.autocast(device_type=_device, enabled=(_device == "cuda")):
        with torch.no_grad():
            outputs = _model(**inputs)

    results = _processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[(orig_h, orig_w)],
    )[0]

    boxes  = results["boxes"].cpu().float().numpy().tolist()
    scores = results["scores"].cpu().float().numpy().tolist()
    raw_labels = results.get("labels", [])
    labels = [lbl if isinstance(lbl, str) else lbl.decode() for lbl in raw_labels]

    return {"boxes": boxes, "scores": scores, "labels": labels}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="warning")
