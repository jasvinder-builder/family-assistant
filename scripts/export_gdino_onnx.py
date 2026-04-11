#!/usr/bin/env python3
"""
Benchmark Grounding DINO Tiny inference options on CUDA.

Usage:
    python scripts/export_gdino_onnx.py

Summary of findings (RTX 4070 Ti Super, transformers 5.5, PyTorch 2.6):
  - fp16 autocast eager:  ~81ms  — PRODUCTION PATH (already in scene_service.py)
  - fp32 eager:           ~112ms — baseline
  - ONNX (trace):         BLOCKED — generate_masks_with_special_tokens_and_transfer_map
                          has Python loops over token values, incompatible with torch.jit.trace
  - ONNX (dynamo):        BLOCKED — unsupported ops in dynamo ONNX exporter for GDINO
  - torch.compile:        BLOCKED — transformers 5.x output_capturing decorator causes
                          NameError inside dynamo (upstream bug)
  - SDPA/Flash Attention: NOT SUPPORTED by GroundingDinoForObjectDetection in transformers 5.5

Next steps for further speedup:
  1. Switch to a GDINO variant that supports SDPA once transformers adds support
  2. Revisit torch.compile after transformers fixes the output_capturing/dynamo bug
  3. TensorRT via torch-tensorrt (requires compatible TRT version) — skip ONNX entirely
  4. Split model: cache text features for fixed prompts, export vision+decoder only to ONNX
"""

import time
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


MODEL_ID    = "IDEA-Research/grounding-dino-tiny"
TEST_IMG_W  = 800
TEST_IMG_H  = 448
TEST_PROMPT = "person . car . dog ."
N_RUNS      = 5


def make_inputs(processor, device):
    dummy = Image.fromarray(np.random.randint(0, 255, (TEST_IMG_H, TEST_IMG_W, 3), dtype=np.uint8))
    inputs = processor(images=dummy, text=TEST_PROMPT, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def bench(fn, label, n=N_RUNS):
    with torch.no_grad():
        fn()  # warm-up
    torch.cuda.synchronize()
    t0 = time.monotonic()
    with torch.no_grad():
        for _ in range(n):
            fn()
    torch.cuda.synchronize()
    ms = (time.monotonic() - t0) / n * 1000
    print(f"  {label:<35} {ms:>7.1f}ms")
    return ms


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading processor and model (fp32)...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = (
        AutoModelForZeroShotObjectDetection
        .from_pretrained(MODEL_ID, dtype=torch.float32)
        .to(device)
        .eval()
    )
    inputs = make_inputs(processor, device)
    print(f"Input shapes: { {k: tuple(v.shape) for k, v in inputs.items()} }")

    print("\n── Latency benchmarks ──")
    fp32_ms = bench(lambda: model(**inputs), "fp32 eager (baseline)")

    fp16_ms = bench(
        lambda: (lambda: model(**inputs))()
        if not torch.is_autocast_enabled()
        else model(**inputs),
        "fp16 autocast eager (production)",
    )
    # redo properly
    def _fp16():
        with torch.autocast("cuda", dtype=torch.float16):
            model(**inputs)

    fp16_ms = bench(_fp16, "fp16 autocast eager")

    print(f"\n── Summary ──")
    print(f"  fp32 eager:        {fp32_ms:.1f}ms  (1.00×)")
    print(f"  fp16 autocast:     {fp16_ms:.1f}ms  ({fp32_ms/fp16_ms:.2f}×)  ← production")
    print()
    print("  ONNX (trace):      BLOCKED — data-dependent Python loops in generate_masks_with_special_tokens_and_transfer_map")
    print("  ONNX (dynamo):     BLOCKED — unsupported ops in dynamo ONNX exporter")
    print("  torch.compile:     BLOCKED — transformers 5.x output_capturing breaks dynamo")
    print("  SDPA/Flash Attn:   NOT SUPPORTED by GroundingDinoForObjectDetection (transformers 5.5)")
    print()
    print("  scene_service.py already uses fp16 autocast — no further ONNX optimisation needed now.")
    print("  Revisit when transformers fixes SDPA/dynamo support for GDINO.")


if __name__ == "__main__":
    main()
