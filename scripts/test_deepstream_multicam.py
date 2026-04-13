"""
Phase 5 — DeepStream multi-camera test.

Topology:
  src-0 ─┐
  src-1 ─┤→ nvstreammux(batch=N) → nvmultistreamtiler(1 row × N cols)
               → nvvideoconvert → appsink
                   └─ appsink callback splits tiled frame into N camera slices

nvmultistreamtiler consumes the batched NVMM buffer and outputs one regular
frame (1×N tile grid). nvvideoconvert can convert that to system RGBA without
issue. We then slice the tiled image by column to recover per-camera frames.

Tiled frame layout (2 cameras, width=1280, height=720):
  ┌──────────────────────┬──────────────────────┐
  │       cam0           │       cam1           │  height=720
  │  cols 0..1279        │  cols 1280..2559     │  total width=2560
  └──────────────────────┴──────────────────────┘

Usage (inside the DeepStream container):
  python3 /workspace/scripts/test_deepstream_multicam.py \\
      file:///workspace/test.mp4 \\
      file:///workspace/test1.mp4 \\
      [--triton HOST:PORT] \\
      [--threshold 0.3] \\
      [--duration 30]
"""

import argparse
import json
import queue
import sys
import threading
import time

import cv2
import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst


# ── Shared state ──────────────────────────────────────────────────────────────
_frame_qs      = {}   # cam_id → Queue(maxsize=1)
_decode_counts = {}
_decode_lock   = threading.Lock()

_CAM_WIDTH  = 0   # set in main after args parsed
_CAM_HEIGHT = 0
_N_CAMS     = 0


def on_new_sample(appsink, _userdata):
    """
    Receives one tiled frame (height × width*N_CAMS).
    Slices it into N per-camera BGR frames and pushes each to its queue.
    """
    sample = appsink.emit("pull-sample")
    if not sample:
        return Gst.FlowReturn.OK

    buf  = sample.get_buffer()
    caps = sample.get_caps()
    s    = caps.get_structure(0)
    w    = s.get_int("width").value    # tiled total width = CAM_WIDTH * N_CAMS
    h    = s.get_int("height").value

    ok, minfo = buf.map(Gst.MapFlags.READ)
    if not ok:
        return Gst.FlowReturn.OK

    tiled = (
        np.frombuffer(minfo.data, dtype=np.uint8)
        .reshape(h, w, 4)[:, :, :3]   # drop alpha
        .copy()
    )
    buf.unmap(minfo)

    cam_w = _CAM_WIDTH
    for i in range(_N_CAMS):
        frame = tiled[:, i * cam_w : (i + 1) * cam_w, :]
        q = _frame_qs[i]
        try:
            q.get_nowait()       # drop stale
        except queue.Empty:
            pass
        try:
            q.put_nowait(frame)
        except queue.Full:
            pass
        with _decode_lock:
            _decode_counts[i] = _decode_counts.get(i, 0) + 1

    return Gst.FlowReturn.OK


# ── Inference thread ──────────────────────────────────────────────────────────

def inference_loop(triton_url: str, queries: list, threshold: float,
                   n_cams: int, stop_event: threading.Event,
                   infer_times: dict, infer_counts: dict) -> None:
    import tritonclient.http as triton_http

    print(f"  [inference] Connecting to Triton at {triton_url}...", flush=True)
    client = triton_http.InferenceServerClient(triton_url)

    for _ in range(30):
        try:
            if client.is_server_ready() and client.is_model_ready("yoloworld"):
                print("  [inference] Triton ready", flush=True)
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        print("  [inference] ERROR: Triton not ready after 30s", flush=True)
        return

    idx = 0
    while not stop_event.is_set():
        cam_id = idx % n_cams
        idx   += 1

        try:
            frame = _frame_qs[cam_id].get(timeout=0.1)
        except queue.Empty:
            continue

        t0 = time.perf_counter()

        ok, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue
        jpeg_bytes = jpeg_buf.tobytes()

        img_in = triton_http.InferInput("IMAGE", [len(jpeg_bytes)], "UINT8")
        img_in.set_data_from_numpy(np.frombuffer(jpeg_bytes, dtype=np.uint8).copy())

        thr_in = triton_http.InferInput("THRESHOLD", [1], "FP32")
        thr_in.set_data_from_numpy(np.array([threshold], dtype=np.float32))

        try:
            resp = client.infer("yoloworld", inputs=[img_in, thr_in])
        except Exception as e:
            print(f"  [inference] cam{cam_id} error: {e}", flush=True)
            continue

        dt_ms = (time.perf_counter() - t0) * 1000
        infer_times.setdefault(cam_id, []).append(dt_ms)
        infer_counts[cam_id] = infer_counts.get(cam_id, 0) + 1

        label_ids = resp.as_numpy("LABEL_IDS")
        scores    = resp.as_numpy("SCORES")
        labels    = [queries[int(i)] if int(i) < len(queries) else str(int(i))
                     for i in label_ids]
        det_str   = ", ".join(
            f"{lbl}({sc:.2f})" for lbl, sc in zip(labels[:4], scores[:4])
        )
        total = sum(infer_counts.values())
        print(
            f"  [cam{cam_id}] #{total:4d}  {dt_ms:5.1f}ms"
            f"  dets={len(labels):2d}  {det_str}",
            flush=True,
        )

    for cam_id in sorted(infer_times):
        times = sorted(infer_times[cam_id])
        n     = len(times)
        if n == 0:
            continue
        print(f"\n[inference] cam{cam_id} summary ({n} frames):")
        print(f"  Median : {times[n // 2]:.1f} ms")
        print(f"  p95    : {times[int(n * 0.95)]:.1f} ms")
        print(f"  p99    : {times[int(n * 0.99)]:.1f} ms")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global _CAM_WIDTH, _CAM_HEIGHT, _N_CAMS

    p = argparse.ArgumentParser()
    p.add_argument("sources", nargs="+", help="RTSP URLs or file URIs")
    p.add_argument("--triton",    default="localhost:8002")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--duration",  type=int,   default=30)
    p.add_argument("--width",     type=int,   default=1280)
    p.add_argument("--height",    type=int,   default=720)
    args = p.parse_args()

    meta_path = "/workspace/models/yoloworld.meta.json"
    try:
        queries = json.loads(open(meta_path).read())["queries"]
    except Exception:
        queries = ["person", "car", "dog"]

    n_cams = len(args.sources)
    _CAM_WIDTH  = args.width
    _CAM_HEIGHT = args.height
    _N_CAMS     = n_cams

    print(f"Cameras  : {n_cams}")
    for i, s in enumerate(args.sources):
        print(f"  cam{i}  : {s}")
    print(f"Queries  : {queries}")
    print(f"Triton   : {args.triton}")
    print(f"Duration : {args.duration}s")
    print(f"Tiled output: {args.width * n_cams} × {args.height}\n")

    for i in range(n_cams):
        _frame_qs[i] = queue.Queue(maxsize=1)

    Gst.init(None)

    # ── Build pipeline ────────────────────────────────────────────────────────
    pipeline   = Gst.Pipeline.new("multicam-pipeline")
    streammux  = Gst.ElementFactory.make("nvstreammux",        "mux")
    tiler      = Gst.ElementFactory.make("nvmultistreamtiler", "tiler")
    convert    = Gst.ElementFactory.make("nvvideoconvert",     "convert")
    capsfilter = Gst.ElementFactory.make("capsfilter",         "caps")
    sink       = Gst.ElementFactory.make("appsink",            "sink")

    for name, el in [("nvstreammux", streammux), ("nvmultistreamtiler", tiler),
                     ("nvvideoconvert", convert), ("capsfilter", capsfilter),
                     ("appsink", sink)]:
        if el is None:
            print(f"ERROR: Could not create element '{name}'")
            sys.exit(1)

    streammux.set_property("batch-size",           n_cams)
    streammux.set_property("width",                args.width)
    streammux.set_property("height",               args.height)
    streammux.set_property("batched-push-timeout", 40000)

    tiler.set_property("rows",    1)
    tiler.set_property("columns", n_cams)
    tiler.set_property("width",   args.width * n_cams)
    tiler.set_property("height",  args.height)

    capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw,format=RGBA"))
    sink.set_property("emit-signals", True)
    sink.set_property("max-buffers",  2)
    sink.set_property("drop",         True)
    sink.set_property("sync",         False)
    sink.connect("new-sample", on_new_sample, None)

    for el in [streammux, tiler, convert, capsfilter, sink]:
        pipeline.add(el)
    streammux.link(tiler)
    tiler.link(convert)
    convert.link(capsfilter)
    capsfilter.link(sink)

    # Add one nvurisrcbin per source
    for i, uri in enumerate(args.sources):
        src = Gst.ElementFactory.make("nvurisrcbin", f"src-{i}")
        if src is None:
            print(f"ERROR: Could not create nvurisrcbin for cam{i}")
            sys.exit(1)
        src.set_property("uri", uri)

        def on_pad_added(src_bin, new_pad, sink_id=i):
            try:
                sink_pad = streammux.request_pad_simple(f"sink_{sink_id}")
            except AttributeError:
                sink_pad = streammux.get_request_pad(f"sink_{sink_id}")
            if sink_pad.is_linked():
                return
            ret = new_pad.link(sink_pad)
            status = "OK" if ret == Gst.PadLinkReturn.OK else str(ret)
            print(f"  cam{sink_id} → nvstreammux:sink_{sink_id} [{status}]", flush=True)

        src.connect("pad-added", on_pad_added)
        pipeline.add(src)

    # ── Start inference thread ────────────────────────────────────────────────
    infer_times  = {}
    infer_counts = {}
    stop_event   = threading.Event()
    infer_thread = threading.Thread(
        target=inference_loop,
        args=(args.triton, queries, args.threshold, n_cams,
              stop_event, infer_times, infer_counts),
        daemon=True,
        name="inference",
    )
    infer_thread.start()

    # ── Start pipeline ────────────────────────────────────────────────────────
    print("Starting pipeline...")
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("ERROR: Pipeline failed to start")
        stop_event.set()
        sys.exit(1)

    ret, state, _ = pipeline.get_state(timeout=10 * Gst.SECOND)
    if state != Gst.State.PLAYING:
        print(f"WARNING: Pipeline state={state}, expected PLAYING")
    else:
        print("Pipeline PLAYING\n")

    t0 = time.time()
    while time.time() - t0 < args.duration:
        time.sleep(2.0)
        elapsed = time.time() - t0
        with _decode_lock:
            counts = dict(_decode_counts)
        parts = []
        for i in range(n_cams):
            c   = counts.get(i, 0)
            fps = c / elapsed if elapsed > 0 else 0
            parts.append(f"cam{i}: {c}fr {fps:.1f}fps")
        total_infer = sum(infer_counts.values())
        print(
            f"  [pipeline] t={elapsed:5.1f}s  "
            + "  |  ".join(parts)
            + f"  |  infer_total={total_infer}",
            flush=True,
        )

    print("\nStopping...")
    stop_event.set()
    pipeline.set_state(Gst.State.NULL)
    infer_thread.join(timeout=5)

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    with _decode_lock:
        counts = dict(_decode_counts)
    for i in range(n_cams):
        c       = counts.get(i, 0)
        avg_fps = c / elapsed if elapsed > 0 else 0
        inf     = infer_counts.get(i, 0)
        print(f"cam{i}: decode={c} frames  avg={avg_fps:.1f} fps  inferred={inf} frames")

    passed = True
    for i in range(n_cams):
        if counts.get(i, 0) == 0:
            print(f"FAIL: cam{i} — no frames decoded")
            passed = False
        avg_fps = counts.get(i, 0) / elapsed if elapsed > 0 else 0
        if avg_fps < 15:
            print(f"WARN: cam{i} fps={avg_fps:.1f} is low")

    if sum(infer_counts.values()) == 0:
        print("FAIL: No inference frames processed")
        passed = False

    print(f"\nPhase 5: {'PASS ✓' if passed else 'FAIL ✗'}")


if __name__ == "__main__":
    main()
