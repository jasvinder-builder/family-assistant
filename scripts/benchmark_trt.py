"""
Phase 2 — Benchmark a YOLO-World TRT engine.

Usage:
  python3 scripts/benchmark_trt.py [--engine models/yoloworld.engine] [--iters 200]
"""

import argparse
import json
import pathlib
import time

import numpy as np
import torch


def benchmark(engine_path: str, iters: int) -> None:
    from ultralytics import YOLO

    meta_path = pathlib.Path(engine_path).with_suffix(".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"Engine metadata: {meta}")
    else:
        print(f"No metadata file found at {meta_path}")

    print(f"\nLoading TRT engine: {engine_path}")
    model = YOLO(engine_path, task="detect")

    img_size = meta.get("imgsz", 640) if meta_path.exists() else 640
    frame = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

    print(f"Warming up (10 frames)...")
    for _ in range(10):
        model(frame, verbose=False)
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(frame, verbose=False)
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000)

    times_ms.sort()
    n = len(times_ms)
    print(f"\nTRT engine benchmark  iters={iters}  img={img_size}px")
    print(f"  Median : {times_ms[n // 2]:.1f} ms")
    print(f"  p95    : {times_ms[int(n * 0.95)]:.1f} ms")
    print(f"  p99    : {times_ms[int(n * 0.99)]:.1f} ms")
    print(f"  Min    : {times_ms[0]:.1f} ms")
    print(f"  Max    : {times_ms[-1]:.1f} ms")

    # Sanity check: try a real frame with an object-shaped blob
    real = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    real[img_size // 3:2 * img_size // 3, img_size // 4:3 * img_size // 4] = 180
    results = model(real, conf=0.1, verbose=False)
    print(f"\n  Detection test (synthetic rectangle, conf=0.1): {len(results[0].boxes)} boxes")
    if results[0].boxes:
        for b in results[0].boxes:
            print(f"    cls={int(b.cls.item())} conf={b.conf.item():.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default="models/yoloworld.engine")
    p.add_argument("--iters", type=int, default=200)
    args = p.parse_args()

    if not pathlib.Path(args.engine).exists():
        print(f"Engine not found: {args.engine}")
        print("Run: python3 scripts/export_yoloworld_trt.py")
        raise SystemExit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    benchmark(args.engine, args.iters)
