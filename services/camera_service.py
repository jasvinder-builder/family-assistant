import asyncio
import os
import queue
import threading

# Force TCP transport for RTSP — UDP is often blocked by firewalls/NAT.
# stimeout is the socket timeout in microseconds (5s).
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000",
)

_stream_url: str | None = None


def set_stream_url(url: str) -> None:
    global _stream_url
    from services import scene_service

    _stream_url = url.strip() if url.strip() else None
    if _stream_url:
        scene_service.start_analysis(_stream_url)
    else:
        scene_service.stop_analysis()


def get_stream_url() -> str | None:
    return _stream_url


async def mjpeg_generator(rtsp_url: str):
    """Async generator that yields MJPEG multipart frames from an RTSP URL."""
    import cv2

    frame_q: queue.Queue = queue.Queue(maxsize=4)
    stop_event = threading.Event()

    def _reader():
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        try:
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                ok, jpeg = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                )
                if not ok:
                    continue
                try:
                    frame_q.put_nowait(jpeg.tobytes())
                except queue.Full:
                    pass  # drop frame — client is slow
        finally:
            cap.release()
            try:
                frame_q.put_nowait(None)  # sentinel
            except queue.Full:
                pass

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    try:
        while True:
            try:
                chunk = frame_q.get_nowait()
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
        stop_event.set()
