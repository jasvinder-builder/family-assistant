import asyncio
import logging
import os
import queue
import threading
import time

logger = logging.getLogger(__name__)

# Force TCP transport for RTSP — UDP is often blocked by firewalls/NAT.
# stimeout is the socket timeout in microseconds (5s).
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000",
)

_stream_url: str | None = None
_debug_overlay: bool = False

# ── Singleton reader + subscriber broadcast ───────────────────────────────────

_subscribers: set[queue.Queue] = set()
_subscribers_lock = threading.Lock()
_reader_thread: threading.Thread | None = None
_reader_stop = threading.Event()


def _broadcast(jpeg_bytes: bytes) -> None:
    """Push a JPEG frame to every subscribed consumer queue, dropping if full."""
    with _subscribers_lock:
        for q in _subscribers:
            try:
                q.put_nowait(jpeg_bytes)
            except queue.Full:
                pass  # slow consumer — drop frame


def subscribe_frames() -> queue.Queue:
    """Register a new consumer and return its dedicated frame queue."""
    q: queue.Queue = queue.Queue(maxsize=4)
    with _subscribers_lock:
        _subscribers.add(q)
    return q


def unsubscribe_frames(q: queue.Queue) -> None:
    """Deregister a consumer queue and signal it with a None sentinel."""
    with _subscribers_lock:
        _subscribers.discard(q)
    try:
        q.put_nowait(None)
    except queue.Full:
        pass


def _reader_loop(rtsp_url: str) -> None:
    """Singleton background thread: reads frames and broadcasts JPEG bytes."""
    import cv2

    logger.info("Camera reader starting — %s", rtsp_url)
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error("Camera reader: failed to open stream %s", rtsp_url)
        return
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        logger.info("Camera reader: opened at %.1f fps", fps)
        frame_interval = 1.0 / fps
        frame_count = 0
        while not _reader_stop.is_set():
            t0 = time.monotonic()
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Share frame with scene analysis (single source of truth)
            from services import scene_service as _ss
            _ss.push_frame(frame)

            if _debug_overlay:
                from services import scene_service
                for det in scene_service.get_latest_detections():
                    x1, y1, x2, y2 = det.box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{det.label} #{det.track_id}"
                    cv2.putText(frame, label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    for row, (query, sim) in enumerate(det.scores.items()):
                        color = (0, 255, 0) if sim >= scene_service.get_threshold() else (0, 165, 255)
                        text = f"{query[:20]}: {sim:.2f}"
                        cv2.putText(frame, text, (x1, y1 + 20 + row * 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                frame_count += 1
                if frame_count <= 3:
                    logger.info("Camera reader: broadcasting frame %d to %d subscriber(s)", frame_count, len(_subscribers))
                _broadcast(jpeg.tobytes())

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, frame_interval - elapsed))
    except Exception:
        logger.exception("Camera reader crashed")
    finally:
        cap.release()
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

    # Stop existing singleton reader
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


# ── Frame generators ──────────────────────────────────────────────────────────

async def mjpeg_generator(rtsp_url: str):
    """Async generator yielding MJPEG multipart chunks (for local HTTP fallback)."""
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
