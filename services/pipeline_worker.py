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
    [1B: frame_type  0=display  1=inference]
    [4B: JPEG byte length]
    [M bytes: JPEG]
  EOS / shutdown signal:
    [4B = 0x00000000]   (4 zero bytes — cam_id length of zero)

Control — stdin (text lines):
  "STOP\\n"  →  clean shutdown

Pipeline topology — one chain per camera, no tiler:

  Live display is handled by go2rtc (reads RTSP directly from go2rtc's re-stream).
  No display branch here — only inference + recording.

  With ROI:
    rtspsrc → nvv4l2decoder → nvvideoconvert → capsfilter(I420,sys) → tee
      ├─ queue(leaky) → videocrop(ROI) → jpegenc → appsink[inference, ft=1, INFER_FPS]
      └─ queue        → nvvideoconvert(NVMM) → nvh264enc → h264parse → splitmuxsink[segs]

  Without ROI:
    rtspsrc → nvv4l2decoder → nvvideoconvert → capsfilter(I420,sys) → tee
      ├─ queue(leaky) → jpegenc → appsink[inference, ft=1, INFER_FPS]
      └─ queue        → nvvideoconvert(NVMM) → nvh264enc → h264parse → splitmuxsink[segs]

JPEG encoding: GStreamer jpegenc for inference crops only.
Segment recording: nvh264enc → splitmuxsink writes rolling .ts segments via mpegtsmux.
  format-location fires on new segment; frame_type=2 JSON event sent to deepstream_service.
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

DISPLAY_FPS    = int(os.environ.get("DISPLAY_FPS",    "25"))
INFER_FPS      = int(os.environ.get("INFER_FPS",      "10"))
SEG_DURATION_S = int(os.environ.get("SEG_DURATION_S", "30"))   # rolling segment length
MAX_SEG_FILES  = int(os.environ.get("MAX_SEG_FILES",  "25"))   # ~12.5 min pre-buffer
CLIPS_DIR      = os.environ.get("CLIPS_DIR", "/app/clips")

_stop_flag = threading.Event()
_last_ts: dict[str, float] = {}
_stdout = sys.stdout.buffer
_stdout_lock = threading.Lock()   # serialise writes from concurrent appsink threads

# splitmuxsink elements indexed by cam_id — used by the SPLIT stdin command
_splitmuxers: dict[str, object] = {}
_splitmuxers_lock = threading.Lock()

# Per-camera current in-flight segment: {path, start_wall}
_seg_info: dict[str, dict] = {}


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
        with _stdout_lock:
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


# ── Recording branch helpers ──────────────────────────────────────────────────

def _build_recording_branch(pipeline: Gst.Pipeline, cam_id: str) -> tuple:
    """
    Build and return (q_r, mux_r) — the recording tee branch elements.

    Chain: queue → nvvideoconvert(NVMM) → nvh264enc → h264parse → splitmuxsink

    nvvideoconvert re-uploads the I420 system-memory buffer (output from the
    upstream capsfilter) back to NVMM so nvh264enc can GPU-encode it.  The
    double copy (NVMM→sys for jpegenc, sys→NVMM for nvh264enc) is acceptable
    because the recording branch runs at the same rate as the source (~25fps)
    but the encode is fast on hardware.

    splitmuxsink writes rolling SEG_DURATION_S-second MP4 segments to
    CLIPS_DIR/{cam_id}/segs/seg_%05d.mp4.  It emits:
      - format-location: when creating a new file (we record start_wall)
      - finished-file:   when a segment is done (we send frame_type=2 to parent)
    """
    segs_dir = os.path.join(CLIPS_DIR, cam_id, "segs")
    try:
        os.makedirs(segs_dir, exist_ok=True)
    except Exception as exc:
        logger.error("Cannot create segs dir %s: %s", segs_dir, exc)
        return None, None

    q_r      = Gst.ElementFactory.make("queue",           f"q-rec-{cam_id}")
    conv_r   = Gst.ElementFactory.make("nvvideoconvert",  f"conv-rec-{cam_id}")
    caps_r   = Gst.ElementFactory.make("capsfilter",      f"caps-rec-{cam_id}")
    enc_r    = Gst.ElementFactory.make("nvv4l2h264enc",   f"enc-{cam_id}")
    parse_r  = Gst.ElementFactory.make("h264parse",       f"h264parse-{cam_id}")
    mux_r    = Gst.ElementFactory.make("splitmuxsink",    f"splitmux-{cam_id}")

    if not all([q_r, conv_r, caps_r, enc_r, parse_r, mux_r]):
        logger.error("Failed to create recording branch elements for %s", cam_id)
        return None, None

    # Recording queue: leaky=2 (drop oldest) to prevent the non-draining encoder
    # from deadlocking the tee.  GStreamer tee pushes synchronously — a blocked
    # branch blocks ALL branches, including the leaky inference branch.  Making
    # this queue leaky means the tee never waits; occasional frame drops during
    # the initial go2rtc burst are acceptable.
    q_r.set_property("leaky",            2)     # GST_QUEUE_LEAK_DOWNSTREAM
    q_r.set_property("max-size-buffers", 600)   # ~40 s at 15 fps before leaking
    q_r.set_property("max-size-time",    int(10 * Gst.SECOND))

    # nvvideoconvert must output NVMM — nvv4l2h264enc only accepts memory:NVMM
    caps_r.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=I420"))

    # nvv4l2h264enc: bitrate is in bits/s (not kbps)
    enc_r.set_property("bitrate",        3000000)   # 3 Mbps — good quality for 1080p/720p
    enc_r.set_property("iframeinterval", 30)        # keyframe every 30 frames (~1s)

    # h264parse: resend SPS/PPS with every IDR so each segment is self-contained.
    parse_r.set_property("config-interval", -1)

    # splitmuxsink: use mpegtsmux instead of the default mp4mux.
    # nvv4l2h264enc output buffers have no PTS — mp4mux rejects them with
    # "Buffer has no PTS"; mpegtsmux is designed for streaming and tolerates
    # missing / inconsistent timestamps without error.
    tsmux = Gst.ElementFactory.make("mpegtsmux", f"tsmux-{cam_id}")
    if tsmux is None:
        logger.error("mpegtsmux not available — recording branch disabled for %s", cam_id)
        return None, None
    mux_r.set_property("muxer",         tsmux)
    mux_r.set_property("location",      os.path.join(segs_dir, "seg_%05d.ts"))
    mux_r.set_property("max-size-time", SEG_DURATION_S * Gst.SECOND)
    mux_r.set_property("max-files",     MAX_SEG_FILES)

    # ── Signal handlers ────────────────────────────────────────────────────────
    # splitmuxsink in GStreamer < 1.18 has no finished-file signal.
    # Instead, format-location fires when a NEW segment opens — at that point
    # the PREVIOUS segment is complete.  We emit the frame_type=2 boundary event
    # for the old segment when the next segment starts.

    def _on_format_location(mux, frag_id, _cam=cam_id):
        """Called when splitmuxsink opens a new segment file."""
        now  = time.time()
        path = os.path.join(CLIPS_DIR, _cam, "segs", f"seg_{frag_id:05d}.ts")

        # Close out the PREVIOUS segment (frag_id > 0 means there was one)
        prev = _seg_info.get(_cam)
        if prev and prev.get("start_wall"):
            payload = json.dumps({
                "path":       prev["path"],
                "start_wall": prev["start_wall"],
                "end_wall":   now,
            }).encode()
            _send_frame(_cam, payload, frame_type=2)
            logger.info("Segment finished: cam=%s path=%s dur=%.1fs",
                        _cam, os.path.basename(prev["path"]), now - prev["start_wall"])

        # Record the new segment
        _seg_info[_cam] = {"path": path, "start_wall": now}
        logger.info("Segment started: cam=%s frag=%d path=%s", _cam, frag_id, path)
        return path   # splitmuxsink uses the returned path as the actual filename

    mux_r.connect("format-location", _on_format_location)

    with _splitmuxers_lock:
        _splitmuxers[cam_id] = mux_r

    # Add and link
    for el in [q_r, conv_r, caps_r, enc_r, parse_r, mux_r]:
        pipeline.add(el)
        el.sync_state_with_parent()

    if not (q_r.link(conv_r) and conv_r.link(caps_r) and caps_r.link(enc_r)
            and enc_r.link(parse_r) and parse_r.link(mux_r)):
        logger.error("Recording branch link failed for %s", cam_id)
        return None, None

    logger.info("%s recording branch: queue→nvh264enc→mpegtsmux→splitmuxsink (%s, %ds segs)",
                cam_id, segs_dir, SEG_DURATION_S)
    return q_r, mux_r


# ── Per-camera downstream chain ───────────────────────────────────────────────

def _attach_downstream(pipeline: Gst.Pipeline, decoded_src_pad: Gst.Pad,
                       cam_id: str, roi: dict | None = None) -> bool:
    """
    Build the downstream chain from decoded_src_pad.

    No display branch — go2rtc handles live display directly from RTSP.

    With ROI:
        → capsfilter(I420,sys) → tee
          ├─ queue(leaky) → videocrop(ROI) → jpegenc → appsink[inference, ft=1]
          └─ queue        → nvvideoconvert(NVMM) → nvh264enc → h264parse → splitmuxsink

    Without ROI:
        → capsfilter(I420,sys) → tee
          ├─ queue(leaky) → jpegenc → appsink[inference, ft=1]
          └─ queue        → nvvideoconvert(NVMM) → nvh264enc → h264parse → splitmuxsink
    """
    # ── Shared head: NVMM → I420 system memory ──────────────────────────────
    convert    = Gst.ElementFactory.make("nvvideoconvert", f"convert-{cam_id}")
    capsfilter = Gst.ElementFactory.make("capsfilter",     f"caps-{cam_id}")
    tee        = Gst.ElementFactory.make("tee",            f"tee-{cam_id}")

    if not all([convert, capsfilter, tee]):
        logger.error("Failed to create head elements for %s", cam_id)
        return False

    capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw,format=I420"))

    for el in [convert, capsfilter, tee]:
        pipeline.add(el)
        el.sync_state_with_parent()

    if not (convert.link(capsfilter) and capsfilter.link(tee)):
        logger.error("head chain link failed for %s", cam_id)
        return False

    # ── Branch A: inference ──────────────────────────────────────────────────
    # With ROI: videocrop → jpegenc → appsink (ft=1, cropped region)
    # Without ROI: jpegenc → appsink (ft=1, full frame)
    q_i    = Gst.ElementFactory.make("queue",   f"q-inf-{cam_id}")
    jpeg_i = Gst.ElementFactory.make("jpegenc", f"jpeg-inf-{cam_id}")
    sink_i = Gst.ElementFactory.make("appsink", f"sink-inf-{cam_id}")

    if not all([q_i, jpeg_i, sink_i]):
        logger.error("Failed to create inference branch elements for %s", cam_id)
        return False

    q_i.set_property("leaky",            2)   # GST_QUEUE_LEAK_DOWNSTREAM
    q_i.set_property("max-size-buffers", 2)
    sink_i.set_property("emit-signals", True)
    sink_i.set_property("max-buffers",  2)
    sink_i.set_property("drop",         True)
    sink_i.set_property("sync",         False)
    sink_i.connect("new-sample", _make_sample_cb(cam_id, 1), None)

    if roi:
        vcrop = Gst.ElementFactory.make("videocrop", f"vcrop-{cam_id}")
        if vcrop is None:
            logger.error("Failed to create videocrop for %s", cam_id)
            return False

        vcrop.set_property("left", roi["x"])
        vcrop.set_property("top",  roi["y"])

        frame_w_known = roi.get("frame_w")
        frame_h_known = roi.get("frame_h")
        if frame_w_known and frame_h_known:
            vcrop.set_property("right",  max(0, frame_w_known - (roi["x"] + roi["w"])))
            vcrop.set_property("bottom", max(0, frame_h_known - (roi["y"] + roi["h"])))
            logger.info("%s videocrop: left=%d top=%d right=%d bottom=%d",
                        cam_id, roi["x"], roi["y"],
                        max(0, frame_w_known - (roi["x"] + roi["w"])),
                        max(0, frame_h_known - (roi["y"] + roi["h"])))
        else:
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
                    logger.info("%s videocrop: left=%d top=%d right=%d bottom=%d (caps)",
                                cam_id, _roi["x"], _roi["y"], right, bottom)
            vcrop.get_static_pad("sink").connect("notify::caps", _on_vcrop_caps)

        for el in [q_i, vcrop, jpeg_i, sink_i]:
            pipeline.add(el)
            el.sync_state_with_parent()
        infer_chain_ok = q_i.link(vcrop) and vcrop.link(jpeg_i) and jpeg_i.link(sink_i)
    else:
        for el in [q_i, jpeg_i, sink_i]:
            pipeline.add(el)
            el.sync_state_with_parent()
        infer_chain_ok = q_i.link(jpeg_i) and jpeg_i.link(sink_i)

    if not infer_chain_ok:
        logger.error("Inference branch link failed for %s", cam_id)
        return False

    tee_src_i = tee.request_pad_simple("src_%u")
    if tee_src_i is None or tee_src_i.link(q_i.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
        logger.error("tee → inference queue link failed for %s", cam_id)
        return False

    # ── Branch B: recording (splitmuxsink) ────────────────────────────────────
    q_r, mux_r = _build_recording_branch(pipeline, cam_id)
    if q_r is None:
        logger.warning("%s recording branch unavailable — skipping", cam_id)
    else:
        tee_src_r = tee.request_pad_simple("src_%u")
        if tee_src_r is None or tee_src_r.link(q_r.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
            logger.error("tee → recording queue link failed for %s", cam_id)
            return False

    logger.info("%s downstream: tee → [inference%s] + [recording]",
                cam_id, "/crop" if roi else "")

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
    # 5 s jitter buffer: go2rtc sends buffered H.264 (up to ~40 s of content) in a burst
    # when GStreamer first connects.  The rtpjitterbuffer resets when the measured skew
    # exceeds latency/2.  With latency=2000, the tolerance is 1000 ms; go2rtc's burst
    # causes up to 1.2 s of skew → reset → byte freeze.  With latency=5000 the tolerance
    # is 2500 ms, safely above the observed burst skew.
    src.set_property("latency",  5000)
    src.set_property("do-rtcp", False)   # go2rtc doesn't forward RTCP SR; without this, the
                                          # jitter buffer times out the video RTP source at ~30s
                                          # and silently stops pulling video from go2rtc (zombie stall).
    pipeline.add(src)

    # Guard: rtspsrc can fire pad-added multiple times for the same video track
    # (e.g. after a jitter-buffer reset or RTSP renegotiation).  Only build the
    # decode chain once — subsequent firings would fail with duplicate element names.
    _video_connected = [False]

    def on_rtp_pad(rtspsrc_el, new_pad, _cid=cam_id, _pl=pipeline, _roi=roi):
        caps = new_pad.get_current_caps() or new_pad.query_caps()
        if not caps or caps.get_size() == 0:
            return
        struct_ = caps.get_structure(0)
        media   = struct_.get_string("media") or ""
        enc     = (struct_.get_string("encoding-name") or "").upper()

        if media != "video":
            # Drain non-video (audio) pads into a fakesink so the rtspsrc
            # jitter buffer doesn't time out and send an EOS.
            sink = Gst.ElementFactory.make("fakesink", None)
            if sink:
                sink.set_property("sync", False)
                _pl.add(sink)
                sink.sync_state_with_parent()
                new_pad.link(sink.get_static_pad("sink"))
            return

        if _video_connected[0]:
            logger.debug("%s: video pad-added fired again — skipping duplicate decode chain", _cid)
            return
        _video_connected[0] = True

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
            cmd = line.strip()
            if cmd == "STOP":
                logger.info("Received STOP — shutting down")
                _stop_flag.set()
                break
            elif cmd.startswith("SPLIT "):
                cam_id = cmd[6:].strip()
                with _splitmuxers_lock:
                    mux = _splitmuxers.get(cam_id)
                if mux:
                    mux.emit("split-now")
                    logger.info("Received SPLIT %s — forcing segment cut", cam_id)
                else:
                    logger.warning("Received SPLIT %s but no splitmuxer found", cam_id)
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
