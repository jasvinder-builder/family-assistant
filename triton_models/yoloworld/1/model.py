"""
Triton Python backend — YOLO-World inference.

Queries are loaded from /trt_engines/yoloworld.meta.json at model init.
To change queries: update meta.json, then call Triton unload+load API.
This sidesteps Triton 24.09 / tritonclient 2.67 variable-length UINT8
tensor incompatibility for per-request query passing.

Inputs:
  IMAGE     UINT8 [N]   JPEG-encoded frame bytes (flat)
  THRESHOLD FP32  [1]   Box confidence threshold

Outputs:
  BOXES     FP32  [M, 4]  xyxy pixel coordinates
  SCORES    FP32  [M]     Detection confidences
  LABEL_IDS FP32  [M]     Class index per detection; map to string via meta.json queries list
"""

import json
import os
import pathlib

import cv2
import numpy as np
import torch
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self._model = None
        self._current_queries: list[str] = []
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[yoloworld] Initialising on {self._device}", flush=True)
        self._load_model()

    def _load_model(self) -> None:
        model_name = os.environ.get("YOLOWORLD_MODEL", "yolov8m-worldv2")
        engine_path = pathlib.Path("/trt_engines/yoloworld.engine")
        meta_path   = pathlib.Path("/trt_engines/yoloworld.meta.json")

        if engine_path.exists():
            from ultralytics import YOLO
            print(f"[yoloworld] Loading TRT engine: {engine_path}", flush=True)
            self._model = YOLO(str(engine_path), task="detect")
            self._using_trt = True
            if meta_path.exists():
                self._current_queries = json.loads(meta_path.read_text()).get(
                    "queries", ["person", "dog", "car"]
                )
            else:
                self._current_queries = ["person", "dog", "car"]
        else:
            from ultralytics import YOLOWorld
            default_queries = ["person", "dog", "car"]
            if meta_path.exists():
                default_queries = json.loads(meta_path.read_text()).get(
                    "queries", default_queries
                )
            print(f"[yoloworld] Loading PyTorch model: {model_name}", flush=True)
            self._model = YOLOWorld(f"{model_name}.pt")
            self._model.set_classes(default_queries)
            self._current_queries = default_queries
            self._using_trt = False

        print(f"[yoloworld] Ready. Queries: {self._current_queries}", flush=True)

    def execute(self, requests):
        responses = []

        for request in requests:
            # ── Decode inputs ──────────────────────────────────────────────────
            image_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE")
            thresh_tensor = pb_utils.get_input_tensor_by_name(request, "THRESHOLD")

            jpeg_bytes = image_tensor.as_numpy().tobytes()
            threshold  = float(thresh_tensor.as_numpy().flatten()[0])

            # ── Decode JPEG ────────────────────────────────────────────────────
            nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if bgr is None:
                boxes     = np.zeros((0, 4), dtype=np.float32)
                scores    = np.zeros((0,),   dtype=np.float32)
                label_ids = np.zeros((0,),   dtype=np.float32)
            else:
                # ── Run inference ──────────────────────────────────────────────
                results = self._model(
                    bgr,
                    conf=threshold,
                    verbose=False,
                    half=True,
                    device=self._device,
                )

                r = results[0]
                if len(r.boxes) > 0:
                    boxes     = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                    scores    = r.boxes.conf.cpu().numpy().astype(np.float32)
                    label_ids = r.boxes.cls.cpu().numpy().astype(np.float32)
                else:
                    boxes     = np.zeros((0, 4), dtype=np.float32)
                    scores    = np.zeros((0,),   dtype=np.float32)
                    label_ids = np.zeros((0,),   dtype=np.float32)

            # ── Build response ─────────────────────────────────────────────────
            # LABEL_IDS: FP32 class indices — client maps to strings via meta.json
            out_boxes     = pb_utils.Tensor("BOXES",     boxes)
            out_scores    = pb_utils.Tensor("SCORES",    scores)
            out_label_ids = pb_utils.Tensor("LABEL_IDS", label_ids)

            responses.append(
                pb_utils.InferenceResponse(output_tensors=[out_boxes, out_scores, out_label_ids])
            )

        return responses

    def finalize(self):
        print("[yoloworld] Shutting down", flush=True)
        del self._model
