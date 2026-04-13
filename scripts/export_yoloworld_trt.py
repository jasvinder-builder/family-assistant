"""
Phase 2 — Export YOLO-World to TensorRT engine.

Bakes current text queries into the engine. Re-run whenever queries change.
Output: models/yoloworld.engine + models/yoloworld.meta.json

Usage:
  python3 scripts/export_yoloworld_trt.py \
      --queries '["person","dog","car"]' \
      --model yolov8m-worldv2 \
      --img 640

NOTE: Requires ~4GB free VRAM for TRT workspace.
      Stop Qwen (docker compose stop ollama) before running if VRAM is tight.
      Re-start after export (docker compose start ollama).
"""

import argparse
import json
import pathlib
import shutil
import time

import torch


def export(model_name: str, queries: list[str], img_size: int) -> None:
    from ultralytics import YOLOWorld

    out_dir = pathlib.Path("models")
    out_dir.mkdir(exist_ok=True)

    print(f"Free VRAM before export: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f} GB")
    print(f"Model:   {model_name}")
    print(f"Queries: {queries}")
    print(f"ImgSize: {img_size}")
    print()

    print("Loading model and setting classes...")
    model = YOLOWorld(f"{model_name}.pt")
    model.set_classes(queries)

    print(f"Exporting to TensorRT (fp16, imgsz={img_size})...")
    print("This takes 60-120 seconds. Progress shown below.\n")
    t0 = time.monotonic()

    engine_path = model.export(
        format="engine",
        half=True,
        imgsz=img_size,
        simplify=True,
        dynamic=False,   # fixed batch=1 — change to True for variable batch (slower export)
        workspace=4,     # GB TRT workspace; reduce to 2 if OOM
        verbose=True,
    )

    elapsed = time.monotonic() - t0
    print(f"\nExport done in {elapsed:.0f}s")

    dest_engine = out_dir / "yoloworld.engine"
    dest_meta   = out_dir / "yoloworld.meta.json"

    if dest_engine.exists():
        backup = out_dir / "yoloworld.engine.bak"
        dest_engine.rename(backup)
        print(f"Previous engine backed up to {backup}")

    shutil.move(engine_path, dest_engine)
    dest_meta.write_text(json.dumps({
        "model":   model_name,
        "queries": queries,
        "imgsz":   img_size,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))

    size_mb = dest_engine.stat().st_size / 1e6
    print(f"\nSaved: {dest_engine}  ({size_mb:.0f} MB)")
    print(f"Meta:  {dest_meta}")
    print(f"\nRun benchmark: python3 scripts/benchmark_trt.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--queries", default='["person","dog","car","bicycle","bag","cat"]',
                   help="JSON list of detection queries")
    p.add_argument("--model", default="yolov8m-worldv2",
                   help="Base model to export (default: yolov8m-worldv2)")
    p.add_argument("--img", type=int, default=640,
                   help="Input image size (default: 640)")
    args = p.parse_args()

    queries = json.loads(args.queries)
    if not isinstance(queries, list) or not queries:
        raise ValueError("--queries must be a non-empty JSON list, e.g. '[\"person\",\"car\"]'")

    export(args.model, queries, args.img)
