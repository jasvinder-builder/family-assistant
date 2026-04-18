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
    frame_type 0 = display, 1 = inference (ROI-cropped)."""
    min_interval = 1.0 / DISPLAY_FPS
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


# ── Per-camera downstream chains ──────────────────────────────────────────────

def _make_jpeg_chain(pipeline: Gst.Pipeline, cam_id: str, suffix: str,
                     roi: dict | None, frame_type: int) -> tuple:
    """
    Build nvvideoconvert → capsfilter(I420) → jpegenc → appsink.
    If roi is set, configures nvvideoconvert src-crop for GPU-side crop.
    Returns (convert_element, appsink_element) or (None, None) on failure.
    """
    convert    = Gst.ElementFactory.make("nvvideoconvert", f"convert-{suffix}")
    capsfilter = Gst.ElementFactory.make("capsfilter",     f"caps-{suffix}")
    jpegenc    = Gst.ElementFactory.make("jpegenc",        f"jpeg-{suffix}")
    sink       = Gst.ElementFactory.make("appsink",        f"sink-{suffix}")

    if not all([convert, capsfilter, jpegenc, sink]):
        logger.error("Failed to create chain elements for %s", suffix)
        return None, None

    if roi:
        convert.set_property(
            "src-crop",
            f"{roi['x']}:{roi['y']}:{roi['w']}:{roi['h']}",
        )

    capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw,format=I420"))

    sink.set_property("emit-signals", True)
    sink.set_property("max-buffers",  2)
    sink.set_property("drop",         True)
    sink.set_property("sync",         False)
    sink.connect("new-sample", _make_sample_cb(cam_id, frame_type), None)

    for el in [convert, capsfilter, jpegenc, sink]:
        pipeline.add(el)
        el.sync_state_with_parent()

    if not (convert.link(capsfilter)
            and capsfilter.link(jpegenc)
            and jpegenc.link(sink)):
        logger.error("Chain link failed for %s", suffix)
        return None, None

    return convert, sink


def _attach_downstream(pipeline: Gst.Pipeline, decoded_src_pad: Gst.Pad,
                       cam_id: str, roi: dict | None = None) -> bool:
    """
    Attach downstream chain(s) to decoded_src_pad.

    No ROI: single display chain (frame_type=0) — existing behaviour.
    ROI set: tee → display branch (frame_type=0) + inference branch with
             GPU src-crop (frame_type=1).

    nvvideoconvert downloads NVMM frames to system memory; jpegenc (software)
    encodes to JPEG so appsink can map the buffer with READ access.
    Returns True on success.
    """
    if roi is None:
        # Single branch: display + inference on same frame (no tee)
        convert, _ = _make_jpeg_chain(pipeline, cam_id, cam_id, None, frame_type=0)
        if convert is None:
            return False
        ret = decoded_src_pad.link(convert.get_static_pad("sink"))
        if ret != Gst.PadLinkReturn.OK:
            logger.error("%s → nvvideoconvert link failed: %s", cam_id, ret)
            return False
        logger.info("%s display chain ready (no ROI)", cam_id)
        return True

    # Two branches: display (full frame) + inference (ROI crop) via tee
    tee = Gst.ElementFactory.make("tee", f"tee-{cam_id}")
    if tee is None:
        logger.error("Failed to create tee for %s", cam_id)
        return False
    pipeline.add(tee)
    tee.sync_state_with_parent()

    ret = decoded_src_pad.link(tee.get_static_pad("sink"))
    if ret != Gst.PadLinkReturn.OK:
        logger.error("%s → tee link failed: %s", cam_id, ret)
        return False

    # Branch A — display (full frame, frame_type=0)
    q_disp = Gst.ElementFactory.make("queue", f"q-disp-{cam_id}")
    if q_disp is None:
        logger.error("Failed to create display queue for %s", cam_id)
        return False
    pipeline.add(q_disp)
    q_disp.sync_state_with_parent()
    tee_src0 = tee.get_request_pad("src_%u")
    if tee_src0 is None or tee_src0.link(q_disp.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
        logger.error("tee → display queue link failed for %s", cam_id)
        return False
    conv_disp, _ = _make_jpeg_chain(
        pipeline, cam_id, f"{cam_id}-disp", None, frame_type=0
    )
    if conv_disp is None:
        return False
    if not q_disp.get_static_pad("src").link(conv_disp.get_static_pad("sink")) == Gst.PadLinkReturn.OK:
        logger.error("display queue → nvvideoconvert link failed for %s", cam_id)
        return False

    # Branch B — inference (ROI crop, frame_type=1)
    q_infer = Gst.ElementFactory.make("queue", f"q-infer-{cam_id}")
    if q_infer is None:
        logger.error("Failed to create infer queue for %s", cam_id)
        return False
    pipeline.add(q_infer)
    q_infer.sync_state_with_parent()
    tee_src1 = tee.get_request_pad("src_%u")
    if tee_src1 is None or tee_src1.link(q_infer.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
        logger.error("tee → infer queue link failed for %s", cam_id)
        return False
    conv_infer, _ = _make_jpeg_chain(
        pipeline, cam_id, f"{cam_id}-infer", roi, frame_type=1
    )
    if conv_infer is None:
        return False
    if not q_infer.get_static_pad("src").link(conv_infer.get_static_pad("sink")) == Gst.PadLinkReturn.OK:
        logger.error("infer queue → nvvideoconvert link failed for %s", cam_id)
        return False

    logger.info(
        "%s tee ready: display (full) + inference (ROI %s:%s:%s:%s)",
        cam_id, roi["x"], roi["y"], roi["w"], roi["h"],
    )
    return True


# ── Source helpers ────────────────────────────────────────────────────────────

def _add_rtsp_source(pipeline: Gst.Pipeline, cam_id: str, uri: str,
                     roi: dict | None = None) -> None:
    """
    Add an RTSP source using rtspsrc directly (bypasses nvurisrcbin which
    requires a config file in DS7.0).  Decode chain per camera:
      rtspsrc → rtph264/5depay → h264/5parse → nvv4l2decoder → queue
            → [tee →] nvvideoconvert → capsfilter → jpegenc → appsink
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
            roi = cam_info.get("roi")   # {x, y, w, h} or None

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
