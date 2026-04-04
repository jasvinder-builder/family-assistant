"""
Camera reader service — PyAV (FFmpeg) decode + subscriber broadcast.

Replaces the former OpenCV VideoCapture with PyAV for cleaner RTSP handling,
proper frame timestamps, and multi-threaded H.264/H.265 software decode.
Hardware-accelerated decode (NVDEC via GStreamer nvv4l2decoder) is Step 3.

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


# ── Reader loop ───────────────────────────────────────────────────────────────

def _reader_loop(url: str) -> None:
    """Singleton background thread: decodes frames via PyAV, broadcasts JPEGs.

    Handles two cases:
      Local file (e.g. /path/to/video.mp4): rate-controlled playback at source
        fps; loops back to start on EOF.
      Live stream (rtsp:// or http://): reads at network pace; reconnects on
        error or stream end.

    Step 3 upgrade path: replace this reader with a GStreamer pipeline using
    nvv4l2decoder for zero-copy NVDEC hardware decode.
    """
    import av

    is_live = url.startswith("rtsp://") or url.startswith("http://") or url.startswith("https://")
    open_opts: dict[str, str] = {}
    if url.startswith("rtsp://"):
        open_opts = {"rtsp_transport": "tcp", "stimeout": "5000000"}

    logger.info("Camera reader starting — %s", url)

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

            # Enable multi-threaded decode (frame-level parallelism for H.264)
            try:
                video_stream.codec_context.thread_count = 4
            except Exception:
                pass  # not all codecs support this

            fps = float(video_stream.average_rate or video_stream.base_rate or 25.0)
            frame_interval   = 1.0 / fps if not is_live else None
            # Display at 25 fps regardless of source rate — limits the expensive
            # to_ndarray + JPEG encode path and reduces GIL pressure significantly.
            display_interval = 1.0 / 25.0
            last_display_t   = 0.0
            logger.info("Camera reader: opened at %.1f fps (live=%s)", fps, is_live)

            frame_count = 0
            for av_frame in container.decode(video_stream):
                if _reader_stop.is_set():
                    break
                t0 = time.monotonic()

                # Only convert + encode + push on display frames; for the rest just
                # honour the source-fps sleep so file playback runs at real-time speed.
                if t0 - last_display_t < display_interval:
                    if frame_interval:
                        elapsed = time.monotonic() - t0
                        time.sleep(max(0.0, frame_interval - elapsed))
                    continue

                last_display_t = t0
                bgr = av_frame.to_ndarray(format="bgr24")

                # Share raw frame with scene analysis
                from services import scene_service as _ss
                _ss.push_frame(bgr)

                # Draw debug overlay on a copy so the original stays clean
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

                # For display frames, sleep whatever remains of the display interval
                # so the file does not race ahead of real time.
                if frame_interval is not None:
                    elapsed = time.monotonic() - t0
                    time.sleep(max(0.0, display_interval - elapsed))

            # Inner decode loop exited
            if _reader_stop.is_set():
                break
            if not is_live:
                logger.info("Camera reader: EOF — looping")
                # container.close() in finally; while loop reopens
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
    # Signal all waiting subscribers that the stream has ended
    with _subscribers_lock:
        for q in _subscribers:
            try:
                q.put_nowait(None)
            except queue.Full:
                pass


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
