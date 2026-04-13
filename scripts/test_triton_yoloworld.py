"""
Phase 3b — Triton YOLO-World backend standalone test.

Queries are loaded from models/yoloworld.meta.json at model init.
Query-change test: updates meta.json then calls Triton unload+load.

Requires Triton to be running:
  docker compose up triton

Usage:
  python3 scripts/test_triton_yoloworld.py \\
      [--server localhost:8002] \\
      [--image path/to/frame.jpg] \\
      [--threshold 0.3] \\
      [--iters 50]
"""

import argparse
import json
import time
import urllib.request

import cv2
import numpy as np


def triton_reload(server: str) -> None:
    """Unload then load yoloworld model via Triton management API."""
    for action in ("unload", "load"):
        url = f"http://{server}/v2/repository/models/yoloworld/{action}"
        req = urllib.request.Request(url, method="POST")
        urllib.request.urlopen(req, timeout=10)
    # Wait for model to be ready
    for _ in range(60):
        time.sleep(1)
        try:
            url = f"http://{server}/v2/models/yoloworld/ready"
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            pass
    raise RuntimeError("Model not ready after reload")


def run_test(server: str, image_path: str | None, threshold: float, iters: int) -> None:
    import tritonclient.http as triton_http

    print(f"Connecting to Triton at {server}...")
    client = triton_http.InferenceServerClient(server)

    if not client.is_server_ready():
        raise RuntimeError("Triton server is not ready")
    if not client.is_model_ready("yoloworld"):
        raise RuntimeError("Model 'yoloworld' is not ready — check container logs")
    print("Triton OK\n")

    # Load a real image or generate a synthetic one
    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        print(f"Image: {image_path}  shape={frame.shape}")
    else:
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        frame[150:500, 280:360] = [200, 180, 160]
        print("Image: synthetic 640×640 frame (no --image provided)")

    ok, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    jpeg_bytes = jpeg_buf.tobytes()
    # Load queries from meta.json for label mapping
    meta_path = "models/yoloworld.meta.json"
    with open(meta_path) as f:
        queries = json.load(f)["queries"]
    print(f"Queries (from meta.json): {queries}")
    print(f"Threshold: {threshold}\n")

    def make_request():
        img_in = triton_http.InferInput("IMAGE", [len(jpeg_bytes)], "UINT8")
        img_in.set_data_from_numpy(np.frombuffer(jpeg_bytes, dtype=np.uint8).copy())

        thr_in = triton_http.InferInput("THRESHOLD", [1], "FP32")
        thr_in.set_data_from_numpy(np.array([threshold], dtype=np.float32))

        return client.infer("yoloworld", inputs=[img_in, thr_in])

    # First request (includes model init)
    print("First inference (includes backend init)...")
    t0 = time.perf_counter()
    resp = make_request()
    first_ms = (time.perf_counter() - t0) * 1000
    print(f"  Time: {first_ms:.0f}ms")

    boxes     = resp.as_numpy("BOXES")
    scores    = resp.as_numpy("SCORES")
    label_ids = resp.as_numpy("LABEL_IDS")
    labels    = [queries[int(i)] if int(i) < len(queries) else str(int(i)) for i in label_ids]

    print(f"  Detections: {len(boxes)}")
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = map(int, box)
        print(f"    [{i}] {label!r:20s}  conf={score:.3f}  box=({x1},{y1},{x2},{y2})")

    # Latency benchmark
    if iters > 0:
        print(f"\nBenchmark ({iters} iters, after warmup)...")
        for _ in range(5):
            make_request()

        times_ms = []
        for _ in range(iters):
            t0 = time.perf_counter()
            make_request()
            times_ms.append((time.perf_counter() - t0) * 1000)

        times_ms.sort()
        n = len(times_ms)
        print(f"  Median : {times_ms[n // 2]:.1f} ms")
        print(f"  p95    : {times_ms[int(n * 0.95)]:.1f} ms")
        print(f"  p99    : {times_ms[int(n * 0.99)]:.1f} ms")
        print(f"  Min    : {times_ms[0]:.1f} ms")

    # Query change test — update meta.json and reload model
    print(f"\nTesting query change (updates meta.json + Triton reload)...")
    meta_path = "models/yoloworld.meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    original_queries = meta["queries"]
    new_queries = original_queries + ["bicycle"] if "bicycle" not in original_queries else original_queries
    meta["queries"] = new_queries
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    t0 = time.perf_counter()
    triton_reload(server)
    reload_ms = (time.perf_counter() - t0) * 1000
    print(f"  Reload time: {reload_ms:.0f}ms")

    with open(meta_path) as f:
        queries = json.load(f)["queries"]
    resp2 = make_request()
    label_ids2 = resp2.as_numpy("LABEL_IDS")
    labels2 = [queries[int(i)] if int(i) < len(queries) else str(int(i)) for i in label_ids2]
    print(f"  Detections after reload: {len(labels2)}  labels={labels2[:5]}")

    # Restore original queries
    meta["queries"] = original_queries
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPhase 3b: PASS ✓")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--server",    default="localhost:8002")
    p.add_argument("--image",     default=None,
                   help="Path to a JPEG/PNG image for testing")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--iters",     type=int,   default=50)
    args = p.parse_args()

    run_test(
        server=args.server,
        image_path=args.image,
        threshold=args.threshold,
        iters=args.iters,
    )
