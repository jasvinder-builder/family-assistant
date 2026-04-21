"""
Savant pyfunc: extract JPEG frames and forward to deepstream_service for Triton inference.

Each camera's frame is rate-limited to INFER_FPS, encoded as JPEG, and HTTP-POSTed to
deepstream_service at /internal/frame. Inference, tracking, and event logic all live in
deepstream_service — this module is a thin frame-extraction bridge.

ROI crop is applied here (before encoding) so that Triton receives the same cropped JPEG
it would have received from the videocrop GStreamer element in the old pipeline.
"""

import logging
import queue
import threading
import time
import urllib.request

import cv2
import numpy as np

from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.deepstream.utils.surface import get_nvds_buf_surface

logger = logging.getLogger("bianca.pyfunc")


class BiancaFrameForwarder(NvDsPyFuncPlugin):
    """Forward rate-limited, ROI-cropped JPEG frames to deepstream_service for inference."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.deepstream_url: str = kwargs.get("deepstream_url", "http://bianca-deepstream:8090")
        self.infer_fps: float = float(kwargs.get("infer_fps", 10))
        self._min_interval: float = 1.0 / max(self.infer_fps, 1)
        self._last_send: dict[str, float] = {}
        # Per-camera leaky queues: at most 1 pending JPEG per camera
        self._queues: dict[str, queue.Queue] = {}
        self._threads: dict[str, threading.Thread] = {}
        # ROI cache: cam_id → {x,y,w,h} | None; refreshed from deepstream_service
        self._rois: dict[str, dict | None] = {}
        self._roi_refresh_stop = threading.Event()

    def on_start(self) -> bool:
        super().on_start()  # sets self._video_pipeline from gst_element property
        t = threading.Thread(target=self._roi_refresh_loop, daemon=True, name="roi-refresh")
        t.start()
        logger.info("BiancaFrameForwarder started, deepstream_url=%s", self.deepstream_url)
        return True

    def on_stop(self) -> bool:
        self._roi_refresh_stop.set()
        # Signal all forwarder threads to stop
        for q in self._queues.values():
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        return True

    def process_frame(self, buffer, frame_meta) -> None:
        cam_id: str = frame_meta.source_id
        now = time.monotonic()

        # FPS gate — only forward INFER_FPS frames per second per camera
        if now - self._last_send.get(cam_id, 0.0) < self._min_interval:
            return
        self._last_send[cam_id] = now

        # Extract frame as RGBA numpy array from NVMM buffer
        try:
            with get_nvds_buf_surface(buffer, frame_meta.frame_meta) as rgba:
                # Drop alpha channel; keep RGB
                rgb = rgba[:, :, :3].copy()
        except Exception as exc:
            logger.warning("[%s] get_nvds_buf_surface failed: %s", cam_id, exc)
            return

        # Apply ROI crop if configured
        roi = self._rois.get(cam_id)
        if roi:
            x, y, w, h = int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
            fh, fw = rgb.shape[:2]
            x = max(0, min(x, fw - 1))
            y = max(0, min(y, fh - 1))
            w = min(w, fw - x)
            h = min(h, fh - y)
            if w > 0 and h > 0:
                rgb = rgb[y:y + h, x:x + w]

        # Encode to JPEG
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, jpg_buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return
        jpeg_bytes = jpg_buf.tobytes()

        # Ensure per-camera forwarder thread exists
        if cam_id not in self._queues:
            self._ensure_forwarder(cam_id)

        # Leaky put: if queue full, drain and replace
        q = self._queues[cam_id]
        try:
            q.put_nowait(jpeg_bytes)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(jpeg_bytes)
            except queue.Full:
                pass

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_forwarder(self, cam_id: str) -> None:
        if cam_id in self._queues:
            return
        q: queue.Queue = queue.Queue(maxsize=1)
        self._queues[cam_id] = q
        t = threading.Thread(
            target=self._forwarder_loop, args=(cam_id, q),
            daemon=True, name=f"fwd-{cam_id}"
        )
        self._threads[cam_id] = t
        t.start()

    def _forwarder_loop(self, cam_id: str, q: queue.Queue) -> None:
        """Background thread: drain queue and POST JPEG to deepstream_service."""
        url = f"{self.deepstream_url}/internal/frame"
        while True:
            item = q.get()
            if item is None:
                break
            jpeg_bytes: bytes = item
            try:
                req = urllib.request.Request(
                    url,
                    data=jpeg_bytes,
                    method="POST",
                    headers={
                        "Content-Type": "image/jpeg",
                        "X-Cam-Id": cam_id,
                    },
                )
                resp = urllib.request.urlopen(req, timeout=2.0)
                resp.read()
                resp.close()
            except Exception as exc:
                logger.debug("[%s] frame POST failed: %s", cam_id, exc)

    def _roi_refresh_loop(self) -> None:
        """Poll deepstream_service every 10s to refresh ROI config per camera."""
        while not self._roi_refresh_stop.is_set():
            try:
                with urllib.request.urlopen(
                    f"{self.deepstream_url}/streams", timeout=3
                ) as resp:
                    import json
                    data = json.loads(resp.read())
                streams: dict = data.get("streams", {})
                new_rois = {cam_id: info.get("roi") for cam_id, info in streams.items()}
                self._rois.update(new_rois)
            except Exception as exc:
                logger.debug("ROI refresh failed: %s", exc)
            self._roi_refresh_stop.wait(timeout=10)
