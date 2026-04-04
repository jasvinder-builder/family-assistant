"""
Camera reader service — GStreamer NVDEC hardware decode + subscriber broadcast.

Uses nvv4l2decoder for zero-copy H.264/H.265 decode directly on GPU via NVDEC,
then nvvidconv to produce BGRx frames for appsink.  Replaces the former PyAV
software decode path (Step 3 in improvement_ideas.md).

Falls back to PyAV software decode if GStreamer / NVDEC is unavailable.

One singleton reader thread per active stream; all WebSocket/MJPEG consumers
subscribe to a shared JPEG broadcast queue.  Frames are also pushed to
scene_service for Grounding DINO analysis.
"""

import asyncio
import logging
import queue
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_stream_url:    str | None = None
_debug_overlay: bool       = False

# ── Singleton reader + subscriber broadcast ───────────────────────────────────

_subscribers:      set[queue.Queue] = set()
_subscribers_lock  = threading.Lock()
_reader_thread:    threading.Thread | None = None
_reader_stop       = threading.Event()


def _broadcast(jpeg_bytes: bytes) -> None:
    """Push a JPEG frame to every subscribed consumer queue, dropping if full."""
    with _subscribers_lock:
        for q in _subscribers:
            try:
                q.put_nowait(jpeg_bytes)
            except queue.Full:
                pass  # slow consumer — drop frame


def subscribe_frames() -> queue.Queue:
    q: queue.Queue = queue.Queue(maxsize=4)
    with _subscribers_lock:
        _subscribers.add(q)
    return q


def unsubscribe_frames(q: queue.Queue) -> None:
    with _subscribers_lock:
        _subscribers.discard(q)
    try:
        q.put_nowait(None)
    except queue.Full:
        pass


# ── GStreamer NVDEC reader ─────────────────────────────────────────────────────

def _gst_pipeline_str(url: str) -> str:
    """Build a GStreamer pipeline string using CPU (software) decode.

    nvh264dec outputs video/x-raw(memory:CUDAMemory) and conflicts with
    PyTorch's CUDA context when both initialise on different threads.
    We use avdec_h264 (libav software decode) to avoid the conflict.
    NVDEC re-enablement requires initialising PyTorch CUDA before GStreamer
    starts and inserting cudadownload after nvh264dec — tracked in improvement_ideas.md.

    For RTSP streams:
      rtspsrc → rtph264depay → h264parse → avdec_h264 → videoconvert → BGRx → appsink

    For local files:
      filesrc → decodebin (nvh264dec rank set to NONE at init) → videoconvert → appsink
    """
    appsink = (
        "appsink name=sink emit-signals=false "
        "max-buffers=2 drop=true sync=false"
    )
    if url.startswith("rtsp://"):
        return (
            f"rtspsrc location={url} latency=100 protocols=tcp ! "
            "rtph264depay ! h264parse ! "
            "avdec_h264 ! "
            "videoconvert ! video/x-raw,format=BGRx ! "
            f"{appsink}"
        )
    else:
        # Local file — decodebin picks avdec_h264 after nvh264dec rank is lowered
        return (
            f"filesrc location={url} ! decodebin ! "
            "videoconvert ! video/x-raw,format=BGRx ! "
            f"{appsink}"
        )


def _bgrx_to_bgr(data, width: int, height: int) -> np.ndarray:
    """Convert a raw BGRx (4-byte-per-pixel) buffer to a BGR numpy array.

    Accepts bytes or a GI memory buffer (map_info.data).
    Returns a contiguous BGR array safe for cv2/numpy operations.
    """
    arr = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))
    return np.ascontiguousarray(arr[:, :, :3])  # drop X channel, ensure contiguous


def _reader_loop_gst(url: str) -> None:
    """GStreamer reader loop (avdec_h264 software decode, no CUDA in decode path).

    Pulls frames from appsink using pull-sample (blocking, no signal overhead),
    converts BGRx → BGR, draws optional debug overlay, encodes JPEG, broadcasts.
    Reconnects on error for live RTSP streams.
    """
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)

    is_live  = url.startswith("rtsp://") or url.startswith("http://")
    pipeline_str = _gst_pipeline_str(url)
    logger.info("Camera reader (GStreamer NVDEC) starting — %s", url)
    logger.debug("GST pipeline: %s", pipeline_str)

    display_interval = 1.0 / 25.0
    last_display_t   = 0.0
    frame_count      = 0

    while not _reader_stop.is_set():
        pipeline = None
        try:
            pipeline = Gst.parse_launch(pipeline_str)
            sink = pipeline.get_by_name("sink")
            if sink is None:
                logger.error("GStreamer: appsink not found in pipeline")
                break

            pipeline.set_state(Gst.State.PLAYING)

            # Wait for pipeline to reach PLAYING (up to 5s)
            ret = pipeline.get_state(timeout=5 * Gst.SECOND)
            if ret[0] != Gst.StateChangeReturn.SUCCESS:
                logger.warning("GStreamer: pipeline did not reach PLAYING — %s", ret)
                pipeline.set_state(Gst.State.NULL)
                if is_live:
                    time.sleep(2)
                    continue
                else:
                    break

            logger.info("Camera reader (GStreamer NVDEC): pipeline PLAYING")

            while not _reader_stop.is_set():
                # pull_sample blocks until a frame arrives or EOS/error
                sample = sink.emit("pull-sample")
                if sample is None:
                    # EOS or pipeline stopped
                    break

                t0 = time.monotonic()
                if t0 - last_display_t < display_interval:
                    continue
                last_display_t = t0

                buf    = sample.get_buffer()
                caps   = sample.get_caps()
                struct = caps.get_structure(0)
                width  = struct.get_int("width").value
                height = struct.get_int("height").value

                ok, map_info = buf.map(Gst.MapFlags.READ)
                if not ok:
                    logger.warning("GStreamer: failed to map buffer")
                    continue
                try:
                    bgr = _bgrx_to_bgr(map_info.data, width, height)
                finally:
                    buf.unmap(map_info)

                # Push raw frame to scene analysis
                from services import scene_service as _ss
                _ss.push_frame(bgr)

                if _debug_overlay:
                    from services import scene_service
                    for det in scene_service.get_latest_detections():
                        x1, y1, x2, y2 = det.box
                        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            bgr, f"{det.label} #{det.track_id}",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                        )
                        for row, (query, sim) in enumerate(det.scores.items()):
                            color = (
                                (0, 255, 0) if sim >= scene_service.get_threshold()
                                else (0, 165, 255)
                            )
                            cv2.putText(
                                bgr, f"{query[:24]}: {sim:.2f}",
                                (x1, y1 + 20 + row * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                            )

                ok_jpg, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok_jpg:
                    frame_count += 1
                    if frame_count <= 3:
                        logger.info(
                            "Camera reader: broadcasting frame %d to %d subscriber(s)",
                            frame_count, len(_subscribers),
                        )
                    _broadcast(jpeg.tobytes())

            # Inner loop exited
            if _reader_stop.is_set():
                break

            msg = pipeline.get_bus().pop_filtered(
                Gst.MessageType.ERROR | Gst.MessageType.EOS
            )
            if msg and msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                logger.warning("GStreamer error: %s (%s)", err, debug)
            elif not is_live:
                logger.info("Camera reader: EOF — looping")
                # fall through: while loop will reopen the pipeline

            if is_live:
                logger.info("Camera reader: stream ended — reconnecting in 2s")
                time.sleep(2)

        except Exception as exc:
            if _reader_stop.is_set():
                break
            logger.warning("Camera reader (GStreamer): %s — reconnecting in 2s", exc)
            time.sleep(2)
        finally:
            if pipeline is not None:
                try:
                    pipeline.set_state(Gst.State.NULL)
                except Exception:
                    pass

    logger.info("Camera reader stopped")
    with _subscribers_lock:
        for q in _subscribers:
            try:
                q.put_nowait(None)
            except queue.Full:
                pass


# ── PyAV fallback reader ───────────────────────────────────────────────────────

def _reader_loop_pyav(url: str) -> None:
    """PyAV software-decode fallback — used when GStreamer/NVDEC is unavailable."""
    import av

    is_live = url.startswith("rtsp://") or url.startswith("http://") or url.startswith("https://")
    open_opts: dict[str, str] = {}
    if url.startswith("rtsp://"):
        open_opts = {"rtsp_transport": "tcp", "stimeout": "5000000"}

    logger.info("Camera reader (PyAV fallback) starting — %s", url)

    display_interval = 1.0 / 25.0
    last_display_t   = 0.0
    frame_count      = 0

    while not _reader_stop.is_set():
        container = None
        try:
            container = av.open(url, options=open_opts)
            video_stream = next(
                (s for s in container.streams if s.type == "video"), None
            )
            if video_stream is None:
                logger.error("Camera reader: no video stream in %s", url)
                break

            try:
                video_stream.codec_context.thread_count = 4
            except Exception:
                pass

            fps = float(video_stream.average_rate or video_stream.base_rate or 25.0)
            frame_interval = 1.0 / fps if not is_live else None
            logger.info("Camera reader: opened at %.1f fps (live=%s)", fps, is_live)

            for av_frame in container.decode(video_stream):
                if _reader_stop.is_set():
                    break
                t0 = time.monotonic()

                if t0 - last_display_t < display_interval:
                    if frame_interval:
                        elapsed = time.monotonic() - t0
                        time.sleep(max(0.0, frame_interval - elapsed))
                    continue

                last_display_t = t0
                bgr = av_frame.to_ndarray(format="bgr24")

                from services import scene_service as _ss
                _ss.push_frame(bgr)

                if _debug_overlay:
                    from services import scene_service
                    for det in scene_service.get_latest_detections():
                        x1, y1, x2, y2 = det.box
                        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            bgr, f"{det.label} #{det.track_id}",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                        )
                        for row, (query, sim) in enumerate(det.scores.items()):
                            color = (
                                (0, 255, 0) if sim >= scene_service.get_threshold()
                                else (0, 165, 255)
                            )
                            cv2.putText(
                                bgr, f"{query[:24]}: {sim:.2f}",
                                (x1, y1 + 20 + row * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                            )

                ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    frame_count += 1
                    if frame_count <= 3:
                        logger.info(
                            "Camera reader: broadcasting frame %d to %d subscriber(s)",
                            frame_count, len(_subscribers),
                        )
                    _broadcast(jpeg.tobytes())

                if frame_interval is not None:
                    elapsed = time.monotonic() - t0
                    time.sleep(max(0.0, display_interval - elapsed))

            if _reader_stop.is_set():
                break
            if not is_live:
                logger.info("Camera reader: EOF — looping")
            else:
                logger.info("Camera reader: stream ended — reconnecting in 2s")
                time.sleep(2)

        except Exception as exc:
            if _reader_stop.is_set():
                break
            logger.warning("Camera reader: %s — reconnecting in 2s", exc)
            time.sleep(2)
        finally:
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass

    logger.info("Camera reader stopped")
    with _subscribers_lock:
        for q in _subscribers:
            try:
                q.put_nowait(None)
            except queue.Full:
                pass


# ── Backend selection ──────────────────────────────────────────────────────────

def _gstreamer_available() -> bool:
    """Return True if GStreamer (with avdec_h264 software decode) is available.

    We deliberately avoid nvh264dec here: it outputs CUDA memory which conflicts
    with PyTorch's CUDA context when both initialise concurrently on different threads.
    avdec_h264 (libav) is used instead — no CUDA involvement in the decode path.

    When GStreamer is available, nvh264dec and nvh265dec ranks are lowered to NONE
    so that decodebin also avoids them for local file sources.
    """
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)
        reg = Gst.Registry.get()
        if reg.find_feature("avdec_h264", Gst.ElementFactory) is None:
            return False
        # Lower nvcodec decoder ranks so decodebin won't pick them
        for elem_name in ("nvh264dec", "nvh265dec", "nvav1dec", "nvvp8dec", "nvvp9dec"):
            feat = reg.find_feature(elem_name, Gst.ElementFactory)
            if feat:
                feat.set_rank(Gst.Rank.NONE)
        return True
    except Exception:
        return False


# GStreamer is available on this system but disabled for now.
# Root cause: _reader_loop_gst() calls Gst.init(None) on the camera-reader
# thread, but GStreamer must only be initialised from the main thread.
# Calling Gst.init() from a worker thread while CTranslate2 owns the CUDA
# context causes a segfault.  Fix requires pre-creating the GStreamer pipeline
# on the main thread and passing it to the reader thread — tracked in
# improvement_ideas.md Step 3.
_gst_available: bool = False
logger.info("Camera decode backend: PyAV (GStreamer deferred — see improvement_ideas.md Step 3)")


def _reader_loop(url: str) -> None:
    _reader_loop_pyav(url)


# ── Public control ────────────────────────────────────────────────────────────

def set_debug_overlay(enabled: bool) -> None:
    global _debug_overlay
    _debug_overlay = enabled
    from services import scene_service
    scene_service.set_debug_overlay(enabled)


def get_debug_overlay() -> bool:
    return _debug_overlay


def set_stream_url(url: str) -> None:
    global _stream_url, _reader_thread
    from services import scene_service

    _stream_url = url.strip() if url.strip() else None

    _reader_stop.set()
    if _reader_thread and _reader_thread.is_alive():
        _reader_thread.join(timeout=3)
    _reader_stop.clear()
    _reader_thread = None

    if _stream_url:
        scene_service.start_analysis(_stream_url)
        _reader_thread = threading.Thread(
            target=_reader_loop,
            args=(_stream_url,),
            daemon=True,
            name="camera-reader",
        )
        _reader_thread.start()
    else:
        scene_service.stop_analysis()


def get_stream_url() -> str | None:
    return _stream_url


# ── Frame generators (consumed by FastAPI routes) ─────────────────────────────

async def mjpeg_generator(rtsp_url: str):
    """Async generator yielding MJPEG multipart chunks (local HTTP fallback)."""
    q = subscribe_frames()
    try:
        while True:
            try:
                chunk = q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.02)
                continue
            if chunk is None:
                break
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + chunk + b"\r\n"
            )
    finally:
        unsubscribe_frames(q)


async def ws_frame_generator():
    """Async generator yielding raw JPEG bytes for WebSocket streaming."""
    q = subscribe_frames()
    try:
        while True:
            try:
                chunk = q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.02)
                continue
            if chunk is None:
                break
            yield chunk
    finally:
        unsubscribe_frames(q)
