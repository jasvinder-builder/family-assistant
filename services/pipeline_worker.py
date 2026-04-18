#!/usr/bin/env python3
"""
Standalone GStreamer/DeepStream pipeline worker.

Run as a subprocess of deepstream_service.py to completely isolate
rtspsrc / GStreamer from the asyncio event loop.  When asyncio runs in the
same process as rtspsrc, the GLib socket watcher races with asyncio's epoll
and causes SIGABRT on RTSP sources.  Spawning this as a subprocess removes
the interference entirely.

Invocation:
    python3 pipeline_worker.py '<streams_json>'
where streams_json is a JSON object: {"cam0": "rtsp://...", "cam1": "file:///..."}

Protocol — stdout (binary, big-endian):
  Normal frame:
    [4B: cam_id UTF-8 byte length]
    [N bytes: cam_id]
    [4B: JPEG byte length]
    [M bytes: JPEG]
  EOS / shutdown signal:
    [4B = 0x00000000]   (4 zero bytes — cam_id length of zero)

Control — stdin (text lines):
  "STOP\\n"  →  clean shutdown

Pipeline topology — one chain per camera, no tiler:
  RTSP:  rtspsrc → rtph264/5depay → h264/5parse → nvv4l2decoder
              → nvvideoconvert → capsfilter → jpegenc → appsink
  File:  uridecodebin → nvvideoconvert → capsfilter → jpegenc → appsink

JPEG encoding is handled by GStreamer's jpegenc element (no cv2/numpy needed).
"""

import json
import logging
import os
import struct
import sys
import threading
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[pipeline_worker] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

Gst.init(None)
_glib_loop = GLib.MainLoop()
threading.Thread(target=_glib_loop.run, daemon=True, name="glib-main-loop").start()

DISPLAY_FPS = int(os.environ.get("DISPLAY_FPS", "25"))
INFER_FPS   = int(os.environ.get("INFER_FPS",   "10"))

_stop_flag = threading.Event()
_last_ts: dict[str, float] = {}
_stdout = sys.stdout.buffer


# ── Wire protocol helpers ─────────────────────────────────────────────────────

def _send_frame(cam_id: str, jpeg_bytes: bytes, frame_type: int = 0) -> None:
    """Wire protocol: [4B cam_id_len][cam_id][1B frame_type][4B jpeg_len][jpeg]
    frame_type 0 = display, 1 = inference (ROI-cropped)."""
    cid = cam_id.encode("utf-8")
    msg = (
        struct.pack(">I", len(cid))
        + cid
        + struct.pack(">B", frame_type)
        + struct.pack(">I", len(jpeg_bytes))
        + jpeg_bytes
    )
    try:
        _stdout.write(msg)
        _stdout.flush()
    except BrokenPipeError:
        _stop_flag.set()


def _send_eos() -> None:
    try:
        _stdout.write(struct.pack(">I", 0))
        _stdout.flush()
    except BrokenPipeError:
        pass


# ── Per-camera appsink callback ───────────────────────────────────────────────

def _make_sample_cb(cam_id: str, frame_type: int = 0):
    """Return a new-sample callback bound to cam_id and frame_type.
    frame_type 0 = display (DISPLAY_FPS), 1 = inference/ROI-cropped (INFER_FPS)."""
    min_interval = 1.0 / (DISPLAY_FPS if frame_type == 0 else INFER_FPS)
    ts_key = f"{cam_id}:{frame_type}"

    def cb(appsink, _userdata):
        if _stop_flag.is_set():
            return Gst.FlowReturn.EOS

        now = time.monotonic()
        # FPS throttle: drain the buffer without sending if we're ahead of schedule
        if now - _last_ts.get(ts_key, 0.0) < min_interval:
            appsink.emit("pull-sample")
            return Gst.FlowReturn.OK
        _last_ts[ts_key] = now

        sample = appsink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        ok, minfo = buf.map(Gst.MapFlags.READ)
        if ok:
            _send_frame(cam_id, bytes(minfo.data), frame_type)
            buf.unmap(minfo)

        return Gst.FlowReturn.OK

    return cb


# ── Per-camera downstream chain ───────────────────────────────────────────────

def _attach_downstream(pipeline: Gst.Pipeline, decoded_src_pad: Gst.Pad,
                       cam_id: str, roi: dict | None = None) -> bool:
    """
    Build the downstream chain from decoded_src_pad.

    Without ROI (single branch):
        nvv4l2decoder → nvvideoconvert → capsfilter(I420,system-mem)
          → jpegenc → appsink[display, frame_type=0]

    With ROI (tee after NVMM exit — safe because system-memory buffers are
    copied by tee, not shared by reference like NVMM buffers):
        nvv4l2decoder → nvvideoconvert → capsfilter(I420,system-mem) → tee
          ├─ queue → jpegenc → appsink[display,   frame_type=0, DISPLAY_FPS]
          └─ queue → videocrop → jpegenc → appsink[inference, frame_type=1, INFER_FPS]

    videocrop left/top are set immediately; right/bottom are computed once caps
    are negotiated (pad notify::caps signal) so we don't need to know frame dims upfront.
    """
    # ── Shared head: NVMM → I420 system memory ──────────────────────────────
    convert    = Gst.ElementFactory.make("nvvideoconvert", f"convert-{cam_id}")
    capsfilter = Gst.ElementFactory.make("capsfilter",     f"caps-{cam_id}")

    if not all([convert, capsfilter]):
        logger.error("Failed to create convert chain for %s", cam_id)
        return False

    # Request I420 WITHOUT memory:NVMM → nvvideoconvert outputs system memory.
    capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw,format=I420"))

    for el in [convert, capsfilter]:
        pipeline.add(el)
        el.sync_state_with_parent()

    if not convert.link(capsfilter):
        logger.error("convert → capsfilter link failed for %s", cam_id)
        return False

    if roi:
        # ── Tee → display branch + inference branch ──────────────────────────
        tee = Gst.ElementFactory.make("tee", f"tee-{cam_id}")

        # Display branch — leaky queue so a slow consumer never blocks the tee
        q_d    = Gst.ElementFactory.make("queue",   f"q-disp-{cam_id}")
        jpeg_d = Gst.ElementFactory.make("jpegenc", f"jpeg-disp-{cam_id}")
        sink_d = Gst.ElementFactory.make("appsink", f"sink-disp-{cam_id}")

        # Inference branch — leaky queue so inference backlog never starves display
        q_i    = Gst.ElementFactory.make("queue",     f"q-inf-{cam_id}")
        vcrop  = Gst.ElementFactory.make("videocrop", f"vcrop-{cam_id}")
        jpeg_i = Gst.ElementFactory.make("jpegenc",   f"jpeg-inf-{cam_id}")
        sink_i = Gst.ElementFactory.make("appsink",   f"sink-inf-{cam_id}")

        if not all([tee, q_d, jpeg_d, sink_d, q_i, vcrop, jpeg_i, sink_i]):
            logger.error("Failed to create tee branch elements for %s", cam_id)
            return False

        # videocrop: left/top are pixels to remove from each edge.
        # right/bottom = frame_w/h - (roi_x/y + roi_w/h).
        # Use stored frame dimensions if available (set by browser at ROI save time)
        # to compute right/bottom immediately — avoids wrong aspect ratio during the
        # caps-negotiation window.  Fall back to notify::caps for legacy roi dicts.
        vcrop.set_property("left", roi["x"])
        vcrop.set_property("top",  roi["y"])

        frame_w_known = roi.get("frame_w")
        frame_h_known = roi.get("frame_h")
        if frame_w_known and frame_h_known:
            vcrop.set_property("right",  max(0, frame_w_known - (roi["x"] + roi["w"])))
            vcrop.set_property("bottom", max(0, frame_h_known - (roi["y"] + roi["h"])))
            logger.info("%s videocrop: left=%d top=%d right=%d bottom=%d (from roi dims)",
                        cam_id, roi["x"], roi["y"],
                        max(0, frame_w_known - (roi["x"] + roi["w"])),
                        max(0, frame_h_known - (roi["y"] + roi["h"])))
        else:
            # Fallback: set right/bottom once caps are negotiated
            def _on_vcrop_caps(pad, _pspec, _vcrop=vcrop, _roi=roi):
                caps = pad.get_current_caps()
                if not caps or caps.get_size() == 0:
                    return
                s = caps.get_structure(0)
                ok_w, fw = s.get_int("width")
                ok_h, fh = s.get_int("height")
                if ok_w and ok_h and fw and fh:
                    right  = max(0, fw - (_roi["x"] + _roi["w"]))
                    bottom = max(0, fh - (_roi["y"] + _roi["h"]))
                    _vcrop.set_property("right",  right)
                    _vcrop.set_property("bottom", bottom)
                    logger.info("%s videocrop: left=%d top=%d right=%d bottom=%d (from caps)",
                                cam_id, _roi["x"], _roi["y"], right, bottom)
            vcrop.get_static_pad("sink").connect("notify::caps", _on_vcrop_caps)

        # leaky=downstream: drop the newest buffer when queue is full rather than
        # blocking the tee pad.  Prevents a slow inference consumer from starving
        # the display branch — both branches run independently.
        for q in [q_d, q_i]:
            q.set_property("leaky", 2)        # GST_QUEUE_LEAK_DOWNSTREAM = 2
            q.set_property("max-size-buffers", 2)

        # Configure appsinks
        for sink, ft in [(sink_d, 0), (sink_i, 1)]:
            sink.set_property("emit-signals", True)
            sink.set_property("max-buffers",  2)
            sink.set_property("drop",         True)
            sink.set_property("sync",         False)
            sink.connect("new-sample", _make_sample_cb(cam_id, ft), None)

        for el in [tee, q_d, jpeg_d, sink_d, q_i, vcrop, jpeg_i, sink_i]:
            pipeline.add(el)
            el.sync_state_with_parent()

        if not capsfilter.link(tee):
            logger.error("capsfilter → tee link failed for %s", cam_id)
            return False

        # Request two src pads from tee (tee.src_%u template)
        tee_src_d = tee.get_request_pad("src_%u")
        tee_src_i = tee.get_request_pad("src_%u")
        if tee_src_d is None or tee_src_i is None:
            logger.error("Could not get tee src pads for %s", cam_id)
            return False

        if tee_src_d.link(q_d.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
            logger.error("tee → display queue link failed for %s", cam_id)
            return False
        if not (q_d.link(jpeg_d) and jpeg_d.link(sink_d)):
            logger.error("Display branch link failed for %s", cam_id)
            return False

        if tee_src_i.link(q_i.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
            logger.error("tee → inference queue link failed for %s", cam_id)
            return False
        if not (q_i.link(vcrop) and vcrop.link(jpeg_i) and jpeg_i.link(sink_i)):
            logger.error("Inference branch link failed for %s", cam_id)
            return False

        logger.info("%s downstream: tee → [display] + [videocrop%s→infer]",
                    cam_id, repr(roi))
    else:
        # ── Single branch (no ROI) ────────────────────────────────────────────
        jpegenc = Gst.ElementFactory.make("jpegenc",  f"jpeg-{cam_id}")
        sink    = Gst.ElementFactory.make("appsink",  f"sink-{cam_id}")

        if not all([jpegenc, sink]):
            logger.error("Failed to create single-branch elements for %s", cam_id)
            return False

        sink.set_property("emit-signals", True)
        sink.set_property("max-buffers",  2)
        sink.set_property("drop",         True)
        sink.set_property("sync",         False)
        sink.connect("new-sample", _make_sample_cb(cam_id, 0), None)

        for el in [jpegenc, sink]:
            pipeline.add(el)
            el.sync_state_with_parent()

        if not (capsfilter.link(jpegenc) and jpegenc.link(sink)):
            logger.error("Single-branch link failed for %s", cam_id)
            return False

        logger.info("%s downstream: single branch (nvvideoconvert→jpegenc→appsink)", cam_id)

    ret = decoded_src_pad.link(convert.get_static_pad("sink"))
    if ret != Gst.PadLinkReturn.OK:
        logger.error("%s → nvvideoconvert link failed: %s", cam_id, ret)
        return False

    return True


# ── Source helpers ────────────────────────────────────────────────────────────

def _add_rtsp_source(pipeline: Gst.Pipeline, cam_id: str, uri: str,
                     roi: dict | None = None) -> None:
    """
    Add an RTSP source using rtspsrc directly (bypasses nvurisrcbin which
    requires a config file in DS7.0).  Decode chain per camera:
      rtspsrc → rtph264/5depay → h264/5parse → nvv4l2decoder → queue
            → nvvideoconvert → capsfilter(I420,system-mem) → [tee] → appsink(s)
    Audio pads from rtspsrc are silently ignored.
    """
    src = Gst.ElementFactory.make("rtspsrc", f"src-{cam_id}")
    if src is None:
        logger.error("Could not create rtspsrc for %s", cam_id)
        sys.exit(1)
    src.set_property("location", uri)
    src.set_property("latency",  0)
    pipeline.add(src)

    def on_rtp_pad(rtspsrc_el, new_pad, _cid=cam_id, _pl=pipeline, _roi=roi):
        caps = new_pad.get_current_caps() or new_pad.query_caps()
        if not caps or caps.get_size() == 0:
            return
        struct_ = caps.get_structure(0)
        media   = struct_.get_string("media") or ""
        enc     = (struct_.get_string("encoding-name") or "").upper()

        if media != "video":
            logger.debug("rtspsrc pad skipped (media=%s)", media)
            return

        if enc in ("H265", "HEVC"):
            depay_name, parse_name = "rtph265depay", "h265parse"
        else:
            depay_name, parse_name = "rtph264depay", "h264parse"

        depay   = Gst.ElementFactory.make(depay_name,     f"depay-{_cid}")
        parse   = Gst.ElementFactory.make(parse_name,     f"parse-{_cid}")
        decoder = Gst.ElementFactory.make("nvv4l2decoder", f"dec-{_cid}")
        q       = Gst.ElementFactory.make("queue",         f"q-{_cid}")

        if not all([depay, parse, decoder, q]):
            logger.error("Failed to create decode chain for %s", _cid)
            return

        for el in [depay, parse, decoder, q]:
            _pl.add(el)
            el.sync_state_with_parent()

        if not (new_pad.link(depay.get_static_pad("sink")) == Gst.PadLinkReturn.OK
                and depay.link(parse)
                and parse.link(decoder)
                and decoder.link(q)):
            logger.error("Decode chain link failed for %s", _cid)
            return

        _attach_downstream(_pl, q.get_static_pad("src"), _cid, _roi)

    src.connect("pad-added", on_rtp_pad)


def _add_file_source(pipeline: Gst.Pipeline, cam_id: str, uri: str,
                     roi: dict | None = None) -> None:
    """Add a file source using uridecodebin."""
    src = Gst.ElementFactory.make("uridecodebin", f"src-{cam_id}")
    if src is None:
        logger.error("Could not create uridecodebin for %s", cam_id)
        sys.exit(1)
    src.set_property("uri", uri)

    def on_pad_added(src_bin, new_pad, _cid=cam_id, _pl=pipeline, _roi=roi):
        caps = new_pad.get_current_caps() or new_pad.query_caps()
        if not caps or caps.get_size() == 0:
            return
        if "video" not in caps.get_structure(0).get_name():
            return
        _attach_downstream(_pl, new_pad, _cid, _roi)

    src.connect("pad-added", on_pad_added)
    pipeline.add(src)


# ── Pipeline build ────────────────────────────────────────────────────────────

def _build_and_run(stream_map: dict) -> None:
    cam_ids = list(stream_map.keys())
    n       = len(cam_ids)

    pipeline = Gst.Pipeline.new("worker-pipeline")

    for cam_id, cam_info in stream_map.items():
        # Support both old format {"cam": "rtsp://..."} and new {"cam": {"url":..., "roi":...}}
        if isinstance(cam_info, str):
            uri, roi = cam_info, None
        else:
            uri = cam_info.get("url", "")
            roi = cam_info.get("roi")  # dict {x,y,w,h} or None

        if uri.lower().startswith("rtsp://"):
            _add_rtsp_source(pipeline, cam_id, uri, roi)
        else:
            _add_file_source(pipeline, cam_id, uri, roi)

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        logger.error("Pipeline failed to start")
        sys.exit(1)

    logger.info("Pipeline started: %d camera(s) %s", n, cam_ids)

    # Bus poll loop (blocks until EOS / error / stop)
    bus = pipeline.get_bus()
    while not _stop_flag.is_set():
        msg = bus.timed_pop_filtered(
            100 * Gst.MSECOND,
            Gst.MessageType.EOS | Gst.MessageType.ERROR,
        )
        if msg is None:
            continue
        if msg.type == Gst.MessageType.EOS:
            logger.info("EOS received — exiting for file-loop restart")
        elif msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            logger.error("Pipeline error: %s (%s)", err, dbg)
        break

    pipeline.set_state(Gst.State.NULL)
    _send_eos()
    logger.info("Pipeline stopped, EOS sent")


# ── stdin watcher ─────────────────────────────────────────────────────────────

def _stdin_watcher() -> None:
    try:
        for line in sys.stdin:
            if line.strip() == "STOP":
                logger.info("Received STOP — shutting down")
                _stop_flag.set()
                break
    except Exception:
        pass
    _stop_flag.set()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: pipeline_worker.py '<streams_json>'")
        sys.exit(1)

    try:
        stream_map: dict = json.loads(sys.argv[1])
    except json.JSONDecodeError as exc:
        logger.error("Invalid streams JSON: %s", exc)
        sys.exit(1)

    if not stream_map:
        logger.error("Empty stream map — nothing to do")
        sys.exit(1)

    threading.Thread(target=_stdin_watcher, daemon=True, name="stdin-watcher").start()
    _build_and_run(stream_map)
