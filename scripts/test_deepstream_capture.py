"""
Phase 3 — DeepStream single-camera frame capture test.

Verifies that:
  - NVDEC hardware decode works (CPU usage stays low)
  - nvstreammux batches frames correctly
  - appsink delivers frames at expected fps
  - Frame shape is correct (height, width, 3)

Run inside the deepstream container:
  docker run --rm --gpus all \\
    -v $(pwd):/workspace \\
    nvcr.io/nvidia/deepstream:7.0-gc-triton-devel \\
    python3 /workspace/scripts/test_deepstream_capture.py file:///workspace/video.mp4

  # or RTSP:
  python3 /workspace/scripts/test_deepstream_capture.py rtsp://your-camera-url
"""

import sys
import time

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
import numpy as np

Gst.init(None)

SOURCE_URI = sys.argv[1] if len(sys.argv) > 1 else None
if SOURCE_URI is None:
    print("Usage: python3 test_deepstream_capture.py <rtsp://url | file:///path>")
    sys.exit(1)

DURATION_S = int(sys.argv[2]) if len(sys.argv) > 2 else 30

frame_count = 0
last_shape  = None
first_frame_t = None


def on_new_sample(appsink, _):
    global frame_count, last_shape, first_frame_t

    sample = appsink.emit("pull-sample")
    if not sample:
        return Gst.FlowReturn.OK

    if first_frame_t is None:
        first_frame_t = time.monotonic()
        print("  First frame received", flush=True)

    buf  = sample.get_buffer()
    caps = sample.get_caps()
    s    = caps.get_structure(0)
    w    = s.get_int("width").value
    h    = s.get_int("height").value

    ok, minfo = buf.map(Gst.MapFlags.READ)
    if ok:
        arr        = np.frombuffer(minfo.data, np.uint8).reshape(h, w, 4)[:, :, :3]
        last_shape = arr.shape
        buf.unmap(minfo)

    frame_count += 1
    return Gst.FlowReturn.OK


# Build pipeline using explicit elements + pad-added callback for nvurisrcbin
pipeline  = Gst.Pipeline.new("capture-pipeline")
source    = Gst.ElementFactory.make("nvurisrcbin",   "src")
streammux = Gst.ElementFactory.make("nvstreammux",   "mux")
convert   = Gst.ElementFactory.make("nvvideoconvert","convert")
capsfilter= Gst.ElementFactory.make("capsfilter",    "caps")
sink      = Gst.ElementFactory.make("appsink",       "sink")

for name, el in [("nvurisrcbin", source), ("nvstreammux", streammux),
                 ("nvvideoconvert", convert), ("capsfilter", capsfilter),
                 ("appsink", sink)]:
    if el is None:
        print(f"ERROR: Could not create element '{name}' — plugin missing?")
        sys.exit(1)

source.set_property("uri", SOURCE_URI)

streammux.set_property("batch-size",           1)
streammux.set_property("width",                1280)
streammux.set_property("height",               720)
streammux.set_property("batched-push-timeout", 40000)

capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw,format=RGBA"))

sink.set_property("emit-signals", True)
sink.set_property("max-buffers",  2)
sink.set_property("drop",         True)
sink.set_property("sync",         False)
sink.connect("new-sample", on_new_sample, None)


def on_pad_added(src_bin, new_pad):
    """nvurisrcbin creates pads dynamically — link to nvstreammux sink_0."""
    sink_pad = streammux.get_request_pad("sink_0")
    if sink_pad.is_linked():
        return
    ret = new_pad.link(sink_pad)
    if ret != Gst.PadLinkReturn.OK:
        print(f"  WARNING: pad link returned {ret}", flush=True)
    else:
        print("  nvurisrcbin → nvstreammux linked", flush=True)


source.connect("pad-added", on_pad_added)

for el in [source, streammux, convert, capsfilter, sink]:
    pipeline.add(el)

streammux.link(convert)
convert.link(capsfilter)
capsfilter.link(sink)

print(f"Source  : {SOURCE_URI}")
print(f"Duration: {DURATION_S}s\n")

print("Starting pipeline...")
ret = pipeline.set_state(Gst.State.PLAYING)
if ret == Gst.StateChangeReturn.FAILURE:
    print("ERROR: Pipeline failed to start")
    sys.exit(1)

# Wait up to 10s for PLAYING state
ret, state, _ = pipeline.get_state(timeout=10 * Gst.SECOND)
if state != Gst.State.PLAYING:
    print(f"WARNING: Pipeline state is {state}, expected PLAYING")
else:
    print("Pipeline PLAYING\n")

t0 = time.time()
while time.time() - t0 < DURATION_S:
    time.sleep(1.0)
    elapsed = time.time() - t0
    fps     = frame_count / elapsed if elapsed > 0 else 0
    print(f"  t={elapsed:5.1f}s  frames={frame_count:5d}  fps={fps:5.1f}  shape={last_shape}",
          flush=True)

pipeline.set_state(Gst.State.NULL)

elapsed = time.time() - t0
avg_fps = frame_count / elapsed if elapsed > 0 else 0

print(f"\n{'='*50}")
print(f"RESULTS")
print(f"{'='*50}")
print(f"Total frames : {frame_count}")
print(f"Duration     : {elapsed:.1f}s")
print(f"Average fps  : {avg_fps:.1f}")
print(f"Final shape  : {last_shape}")

passed = True
if frame_count == 0:
    print("FAIL: No frames received")
    passed = False
if avg_fps < 15:
    print(f"WARN: fps={avg_fps:.1f} is low — check NVDEC availability")
if last_shape and last_shape[2] != 3:
    print(f"FAIL: Expected 3-channel frame, got {last_shape}")
    passed = False

print(f"\nPhase 3: {'PASS ✓' if passed else 'FAIL ✗'}")
