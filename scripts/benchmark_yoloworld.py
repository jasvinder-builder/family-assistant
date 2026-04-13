"""
Phase 1 — YOLO-World inference speed benchmark.

Usage:
  python3 scripts/benchmark_yoloworld.py [--model yolov8s-worldv2] [--img 640] [--iters 200]

Reports median / p95 / p99 latency and VRAM delta.
Run with Qwen + Whisper + GDINO loaded to simulate production VRAM pressure.
"""

import argparse
import time

import numpy as np
import torch


def benchmark(model_name: str, img_size: int, iters: int) -> None:
    from ultralytics import YOLOWorld

    print(f"\nLoading {model_name}...")
    vram_before = torch.cuda.memory_allocated()
    model = YOLOWorld(f"{model_name}.pt")
    model.set_classes(["person", "dog", "car", "bicycle", "cat"])

    frame = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

    # Warmup — first call sets up AutoBackend (fp16 on CUDA) and fuses layers
    print("  Warming up (10 frames, fp16)...")
    for _ in range(10):
        model(frame, conf=0.3, verbose=False, half=True, device="cuda")
    torch.cuda.synchronize()

    vram_after = torch.cuda.memory_allocated()
    print(f"  VRAM delta:  {(vram_after - vram_before)/1e6:.0f} MB")
    print(f"  VRAM total:  {vram_after/1e9:.2f} GB allocated")

    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(frame, conf=0.3, verbose=False, half=True, device="cuda")
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000)

    times_ms.sort()
    n = len(times_ms)
    print(f"\nResults — model: {model_name}  img: {img_size}px  iters: {iters}")
    print(f"  Median : {times_ms[n // 2]:.1f} ms")
    print(f"  p95    : {times_ms[int(n * 0.95)]:.1f} ms")
    print(f"  p99    : {times_ms[int(n * 0.99)]:.1f} ms")
    print(f"  Min    : {times_ms[0]:.1f} ms")
    print(f"  Max    : {times_ms[-1]:.1f} ms")

    # Quick real-frame detection test
    real_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    real_img[img_size // 3 : 2 * img_size // 3, img_size // 4 : 3 * img_size // 4] = 200
    results = model(real_img, conf=0.1, verbose=False, half=True, device="cuda")
    print(f"\n  Detection test (synthetic rectangle, conf=0.1):")
    print(f"    Boxes found: {len(results[0].boxes)}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8s-worldv2")
    p.add_argument("--img", type=int, default=640)
    p.add_argument("--iters", type=int, default=200)
    args = p.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Free VRAM: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.2f} GB")

    benchmark(args.model, args.img, args.iters)
