import asyncio
import os
import queue
import threading
import time

# Force TCP transport for RTSP — UDP is often blocked by firewalls/NAT.
# stimeout is the socket timeout in microseconds (5s).
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000",
)

_stream_url: str | None = None
_debug_overlay: bool = False


def set_debug_overlay(enabled: bool) -> None:
    global _debug_overlay
    _debug_overlay = enabled
    from services import scene_service
    scene_service.set_debug_overlay(enabled)


def get_debug_overlay() -> bool:
    return _debug_overlay


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
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_interval = 1.0 / fps
            while not stop_event.is_set():
                t0 = time.monotonic()
                ret, frame = cap.read()
                if not ret:
                    # End of file — loop back to start
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
                        label = f"#{det.track_id}"
                        cv2.putText(frame, label, (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        for row, (query, sim) in enumerate(det.scores.items()):
                            color = (0, 255, 0) if sim >= scene_service.get_threshold() else (0, 165, 255)
                            text = f"{query[:20]}: {sim:.2f}"
                            cv2.putText(frame, text, (x1, y1 + 20 + row * 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                ok, jpeg = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                )
                if not ok:
                    continue
                try:
                    frame_q.put_nowait(jpeg.tobytes())
                except queue.Full:
                    pass  # drop frame — client is slow
                # Throttle to native FPS (no-op for live RTSP which self-paces)
                elapsed = time.monotonic() - t0
                time.sleep(max(0.0, frame_interval - elapsed))
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
