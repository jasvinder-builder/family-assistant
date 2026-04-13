"""
Phase 4 — DeepStream NVDEC → Triton YOLO-World inference integration test.

Connects:
  RTSP/file source → nvurisrcbin → nvstreammux → appsink (display, 30fps)
                                               → leaky queue (max 1 frame)
                                                 → inference thread → Triton gRPC

Prints detection results and timing stats. No FastAPI, no WebSocket.

Requires:
  - Triton running with yoloworld model:  docker compose up triton
  - DeepStream container (or bare-metal DeepStream + GStreamer plugins)

Usage (run inside deepstream container):
  python3 scripts/test_deepstream_inference.py \\
      <rtsp://url | file:///path/to/video.mp4> \\
      [--queries '["person","car","dog"]'] \\
      [--threshold 0.3] \\
      [--triton localhost:8001] \\
      [--duration 60]

Run inside container:
  docker run --rm --gpus all \\
    --network host \\
    -v $(pwd)/scripts:/scripts \\
    nvcr.io/nvidia/deepstream:7.1-devel \\
    bash -c "pip install tritonclient[grpc] opencv-python-headless numpy && \\
             python3 /scripts/test_deepstream_inference.py rtsp://your-url"
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


# ── Frame queue: leaky, max 1 frame — decoder never waits on inference ────────
_frame_q: queue.Queue = queue.Queue(maxsize=1)

_frame_count   = 0
_first_frame_t = None


def on_new_sample(appsink, _userdata):
    """GStreamer appsink callback — runs in the GStreamer streaming thread."""
    global _frame_count, _first_frame_t

    sample = appsink.emit("pull-sample")
    if not sample:
        return Gst.FlowReturn.OK

    buf  = sample.get_buffer()
    caps = sample.get_caps()
    s    = caps.get_structure(0)
    w    = s.get_int("width").value
    h    = s.get_int("height").value

    ok, minfo = buf.map(Gst.MapFlags.READ)
    if ok:
        # RGBA from nvvideoconvert → drop alpha channel → BGR for cv2/Triton
        frame = (
            np.frombuffer(minfo.data, dtype=np.uint8)
            .reshape(h, w, 4)[:, :, :3]
            .copy()          # copy before unmap
        )
        buf.unmap(minfo)

        # Leaky: drop old frame if inference hasn't consumed it yet
        try:
            _frame_q.get_nowait()
        except queue.Empty:
            pass
        try:
            _frame_q.put_nowait(frame)
        except queue.Full:
            pass

    if _first_frame_t is None:
        _first_frame_t = time.monotonic()
        print("  [pipeline] First frame received", flush=True)

    _frame_count += 1
    return Gst.FlowReturn.OK


def inference_loop(triton_url: str, queries: list[str], threshold: float,
                   stop_event: threading.Event) -> None:
    """Runs in a background thread. Pulls frames, sends to Triton, prints results."""
    import tritonclient.http as triton_http

    print(f"  [inference] Connecting to Triton at {triton_url}...", flush=True)
    client = triton_http.InferenceServerClient(triton_url)

    # Wait for Triton to be ready (up to 30s)
    for attempt in range(30):
        try:
            if client.is_server_ready() and client.is_model_ready("yoloworld"):
                print("  [inference] Triton ready", flush=True)
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        print("  [inference] ERROR: Triton not ready after 30s — exiting thread",
              flush=True)
        return

    infer_times_ms: list[float] = []
    frames_processed = 0

    while not stop_event.is_set():
        try:
            frame = _frame_q.get(timeout=2.0)
        except queue.Empty:
            continue

        t0 = time.perf_counter()

        # Encode frame as JPEG for transport (matches model.py expectation)
        ok, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue
        jpeg_bytes = jpeg_buf.tobytes()

        # Build Triton inputs
        img_in = triton_http.InferInput("IMAGE", [len(jpeg_bytes)], "UINT8")
        img_in.set_data_from_numpy(np.frombuffer(jpeg_bytes, dtype=np.uint8).copy())

        thr_in = triton_http.InferInput("THRESHOLD", [1], "FP32")
        thr_in.set_data_from_numpy(np.array([threshold], dtype=np.float32))

        try:
            resp = client.infer("yoloworld", inputs=[img_in, thr_in])
        except Exception as e:
            print(f"  [inference] gRPC error: {e}", flush=True)
            continue

        dt_ms = (time.perf_counter() - t0) * 1000
        infer_times_ms.append(dt_ms)
        frames_processed += 1

        boxes     = resp.as_numpy("BOXES")
        scores    = resp.as_numpy("SCORES")
        label_ids = resp.as_numpy("LABEL_IDS")
        labels    = [queries[int(i)] if int(i) < len(queries) else str(int(i)) for i in label_ids]

        det_str = ", ".join(
            f"{lbl}({sc:.2f})" for lbl, sc in zip(labels[:5], scores[:5])
        )
        print(
            f"  [inference] frame={frames_processed:4d}  "
            f"infer={dt_ms:6.1f}ms  dets={len(labels):2d}  {det_str}",
            flush=True,
        )

    # Summary
    if infer_times_ms:
        infer_times_ms.sort()
        n = len(infer_times_ms)
        print(f"\n[inference] Summary ({n} frames processed):")
        print(f"  Median : {infer_times_ms[n // 2]:.1f} ms")
        print(f"  p95    : {infer_times_ms[int(n * 0.95)]:.1f} ms")
        print(f"  p99    : {infer_times_ms[int(n * 0.99)]:.1f} ms")
        print(f"  Min    : {infer_times_ms[0]:.1f} ms")
        print(f"  Max    : {infer_times_ms[-1]:.1f} ms")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("source",      help="RTSP URL or file URI (file:///path/to/video.mp4)")
    p.add_argument("--queries",   default='["person","car","dog"]')
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--triton",    default="localhost:8002",
                   help="Triton HTTP endpoint host:port")
    p.add_argument("--duration",  type=int,   default=60,
                   help="Test duration in seconds")
    p.add_argument("--width",     type=int,   default=1280)
    p.add_argument("--height",    type=int,   default=720)
    args = p.parse_args()

    queries = json.loads(args.queries)
    print(f"Source   : {args.source}")
    print(f"Triton   : {args.triton}")
    print(f"Queries  : {queries}")
    print(f"Threshold: {args.threshold}")
    print(f"Duration : {args.duration}s\n")

    Gst.init(None)

    # Build pipeline with explicit elements — nvurisrcbin uses dynamic pads
    # and cannot be connected via parse_launch shorthand.
    pipeline   = Gst.Pipeline.new("infer-pipeline")
    source     = Gst.ElementFactory.make("nvurisrcbin",    "src")
    streammux  = Gst.ElementFactory.make("nvstreammux",    "mux")
    convert    = Gst.ElementFactory.make("nvvideoconvert", "convert")
    capsfilter = Gst.ElementFactory.make("capsfilter",     "caps")
    sink       = Gst.ElementFactory.make("appsink",        "sink")

    for name, el in [("nvurisrcbin", source), ("nvstreammux", streammux),
                     ("nvvideoconvert", convert), ("capsfilter", capsfilter),
                     ("appsink", sink)]:
        if el is None:
            print(f"ERROR: Could not create element '{name}'")
            sys.exit(1)

    source.set_property("uri", args.source)
    streammux.set_property("batch-size",           1)
    streammux.set_property("width",                args.width)
    streammux.set_property("height",               args.height)
    streammux.set_property("batched-push-timeout", 40000)
    capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw,format=RGBA"))
    sink.set_property("emit-signals", True)
    sink.set_property("max-buffers",  2)
    sink.set_property("drop",         True)
    sink.set_property("sync",         False)
    sink.connect("new-sample", on_new_sample, None)

    def on_pad_added(src_bin, new_pad):
        sink_pad = streammux.get_request_pad("sink_0")
        if sink_pad.is_linked():
            return
        new_pad.link(sink_pad)

    source.connect("pad-added", on_pad_added)

    for el in [source, streammux, convert, capsfilter, sink]:
        pipeline.add(el)
    streammux.link(convert)
    convert.link(capsfilter)
    capsfilter.link(sink)

    # Start inference thread
    stop_event = threading.Event()
    infer_thread = threading.Thread(
        target=inference_loop,
        args=(args.triton, queries, args.threshold, stop_event),
        daemon=True,
        name="inference",
    )
    infer_thread.start()

    # Start pipeline
    print("Starting pipeline...")
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("ERROR: Pipeline failed to start")
        stop_event.set()
        sys.exit(1)

    # Wait for PLAYING state
    ret, state, _ = pipeline.get_state(timeout=10 * Gst.SECOND)
    if state != Gst.State.PLAYING:
        print(f"WARNING: Pipeline state is {state}, expected PLAYING")

    print(f"Running for {args.duration}s...\n")
    t0 = time.time()

    while time.time() - t0 < args.duration:
        time.sleep(2.0)
        elapsed = time.time() - t0
        fps = _frame_count / elapsed if elapsed > 0 else 0
        qsize = _frame_q.qsize()
        print(
            f"  [pipeline] t={elapsed:5.1f}s  frames={_frame_count:6d}  "
            f"fps={fps:5.1f}  queue={qsize}",
            flush=True,
        )

    print("\nStopping...")
    stop_event.set()
    pipeline.set_state(Gst.State.NULL)
    infer_thread.join(timeout=5)

    elapsed = time.time() - t0
    avg_fps = _frame_count / elapsed if elapsed > 0 else 0

    print(f"\n{'='*55}")
    print("RESULTS")
    print(f"{'='*55}")
    print(f"Decode frames  : {_frame_count}")
    print(f"Duration       : {elapsed:.1f}s")
    print(f"Avg decode fps : {avg_fps:.1f}")

    # Acceptance criteria
    passed = True
    if _frame_count == 0:
        print("FAIL: No frames decoded")
        passed = False
    if avg_fps < 15:
        print(f"WARN: fps={avg_fps:.1f} is low — check NVDEC availability")

    print(f"\nPhase 4: {'PASS ✓' if passed else 'FAIL ✗'}")


if __name__ == "__main__":
    main()
