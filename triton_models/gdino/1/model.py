"""
Triton Python backend — Grounding DINO Tiny (fp16 on GPU).

Inputs
------
IMAGE         TYPE_UINT8  [-1]  JPEG-encoded frame bytes
QUERIES       TYPE_UINT8  [-1]  JSON-encoded list of query strings
THRESHOLD     TYPE_FP32   [1]   box confidence threshold
TEXT_THRESHOLD TYPE_FP32  [1]   per-token text-image threshold

Outputs
-------
BOXES   TYPE_FP32  [-1, 4]  xyxy in full-resolution pixel coords
SCORES  TYPE_FP32  [-1]     detection confidence scores
LABELS  TYPE_FP32  [-1]     newline-delimited label string, byte→float32 encoded

Note: the app uses tritonclient.http (not gRPC).  Triton 24.04's gRPC shared-
memory path has a bug that broadcasts the first tensor element to every position
for all dtypes.  The HTTP transport delivers data correctly.

LABELS uses byte→float32 encoding on output (each byte value stored as fp32)
to sidestep any similar issue on the output path.

Resize strategy: frame is downscaled to at most _MAX_W wide before GDINO
to avoid Swin-T processing large frames slowly.  target_sizes is set to
the ORIGINAL image dimensions so the returned boxes are already in
full-resolution pixel coords — the caller does not need to scale them.
"""

import json
import numpy as np
import triton_python_backend_utils as pb_utils

_MAX_W = 800  # pre-resize width cap (same as scene_service constant)


def _get_bytes(request, name: str) -> bytes:
    """Extract bytes from a TYPE_UINT8 flat input tensor."""
    arr = pb_utils.get_input_tensor_by_name(request, name).as_numpy()
    return arr.tobytes()


class TritonPythonModel:
    def initialize(self, args):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "IDEA-Research/grounding-dino-tiny"

        print(f"[gdino] Loading on {self._device}...", flush=True)
        self._processor = AutoProcessor.from_pretrained(model_id)
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(model_id, dtype=dtype)
            .to(self._device)
            .eval()
        )
        self._torch = torch
        print(
            f"[gdino] Ready on {self._device} "
            f"({'fp16' if self._device == 'cuda' else 'fp32'})",
            flush=True,
        )

    def execute(self, requests):
        import cv2
        from PIL import Image

        responses = []
        for request in requests:
            try:
                jpeg_bytes    = _get_bytes(request, "IMAGE")
                queries_bytes = _get_bytes(request, "QUERIES")
                queries       = json.loads(queries_bytes.decode())
                threshold     = float(
                    pb_utils.get_input_tensor_by_name(request, "THRESHOLD").as_numpy()[0]
                )
                text_threshold = float(
                    pb_utils.get_input_tensor_by_name(request, "TEXT_THRESHOLD").as_numpy()[0]
                )

                # Decode JPEG → BGR numpy
                nparr  = np.frombuffer(jpeg_bytes, np.uint8)
                bgr    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError("cv2.imdecode returned None -- bad JPEG?")
                orig_h, orig_w = bgr.shape[:2]

                # Downscale wide frames to avoid Swin-T slowness
                if orig_w > _MAX_W:
                    scale = _MAX_W / orig_w
                    small = cv2.resize(
                        bgr, (_MAX_W, int(orig_h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    small = bgr

                pil_img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                text    = " . ".join(
                    q.strip().lower().rstrip(" .") for q in queries
                ) + " ."

                inputs = self._processor(
                    images=pil_img, text=text, return_tensors="pt",
                ).to(self._device)

                torch = self._torch
                with torch.autocast(device_type=self._device, enabled=(self._device == "cuda")):
                    with torch.no_grad():
                        outputs = self._model(**inputs)

                # target_sizes = ORIGINAL dims → boxes in full-res pixel coords.
                results = self._processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=threshold,
                    text_threshold=text_threshold,
                    target_sizes=[(orig_h, orig_w)],
                )[0]

                boxes  = results["boxes"].cpu().float().numpy()   # [N, 4]
                scores = results["scores"].cpu().float().numpy()  # [N]
                labels = results.get("labels", [])

                if len(boxes) == 0:
                    boxes  = np.zeros((0, 4), dtype=np.float32)
                    scores = np.zeros((0,),   dtype=np.float32)
                    # LABELS: empty → single zero float32
                    labels_fp32 = np.zeros((1,), dtype=np.float32)
                else:
                    joined = "\n".join(
                        lbl if isinstance(lbl, str) else lbl.decode()
                        for lbl in labels
                    ).encode()
                    labels_fp32 = np.frombuffer(joined, dtype=np.uint8).astype(np.float32)

                responses.append(pb_utils.InferenceResponse(output_tensors=[
                    pb_utils.Tensor("BOXES",  boxes),
                    pb_utils.Tensor("SCORES", scores),
                    pb_utils.Tensor("LABELS", labels_fp32),
                ]))

            except Exception as exc:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("BOXES",  np.zeros((0, 4), dtype=np.float32)),
                        pb_utils.Tensor("SCORES", np.zeros((0,),   dtype=np.float32)),
                        pb_utils.Tensor("LABELS", np.zeros((1,),   dtype=np.float32)),
                    ],
                    error=pb_utils.TritonError(str(exc)),
                ))

        return responses

    def finalize(self):
        del self._model
        del self._processor
