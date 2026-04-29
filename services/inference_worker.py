"""
Phase 1 inference worker — one process per camera.

Spawned by services/inference_service.py with `--rtsp rtsp://mediamtx:8554/{cam_id}`.
Decodes via PyAV h264_cuvid (NVDEC), applies a motion gate, calls Triton for
YOLO-World inference, prints `[event] ...` lines on stdout.

One subprocess per camera is deliberate: Phase 1.3 prototype found that two
h264_cuvid decode loops in the same Python process starve the second loop.
Subprocess isolation sidesteps this entirely and matches the pre-Savant
`pipeline_worker.py` pattern.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

import av  # type: ignore[import-not-found]
import cv2
import numpy as np
import tritonclient.http as triton_http  # type: ignore[import-not-found]


# ── Config ────────────────────────────────────────────────────────────────────

INFER_FPS = 10                  # max calls into Triton per camera per second
RECHECK_INTERVAL_S = 30          # min seconds between events for same (track, query)
MOTION_SCALE = (320, 180)        # cheap motion-gate thumbnail size
MOTION_THRESHOLD = 2.0           # mean abs-diff threshold to bypass the gate
QUERIES_FALLBACK = ["animal", "bird", "person", "vehicle"]


# ── Tracker (copied verbatim from services/deepstream_service.py:_SimpleTracker)
# Phase 2 will move this into a shared module; for now duplicate is fine —
# the prototype exists precisely to validate it still works on the new ingress.

class _SimpleTracker:
    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 10):
        self._iou_thr = iou_threshold
        self._max_lost = max_lost
        self._next_id = 1
        self._tracks: dict[int, dict] = {}

    @staticmethod
    def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
        iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
        ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
        iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter
        return np.where(union > 0, inter / union, 0.0)

    def update(self, boxes, scores, class_ids):
        empty = (
            np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
            np.zeros((0,), int), np.zeros((0,), int),
        )
        for t in self._tracks.values():
            t["lost"] += 1

        if len(boxes) == 0:
            self._tracks = {k: v for k, v in self._tracks.items()
                            if v["lost"] <= self._max_lost}
            return empty

        out_b: list = []
        out_s: list = []
        out_c: list = []
        out_t: list = []
        tids = list(self._tracks)
        matched_t: set = set()
        matched_d: set = set()

        if tids:
            t_boxes = np.array([self._tracks[i]["box"] for i in tids], np.float32)
            iou = self._iou_matrix(t_boxes, boxes)
            ti_arr, di_arr = np.where(iou >= self._iou_thr)
            if len(ti_arr):
                order = np.argsort(-iou[ti_arr, di_arr])
                for idx in order:
                    ti, di = int(ti_arr[idx]), int(di_arr[idx])
                    if ti in matched_t or di in matched_d:
                        continue
                    matched_t.add(ti)
                    matched_d.add(di)
                    tid = tids[ti]
                    self._tracks[tid] = {
                        "box": boxes[di].tolist(),
                        "class_id": int(class_ids[di]),
                        "lost": 0,
                    }
                    out_b.append(boxes[di])
                    out_s.append(scores[di])
                    out_c.append(class_ids[di])
                    out_t.append(tid)

        for di in range(len(boxes)):
            if di in matched_d:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {
                "box": boxes[di].tolist(),
                "class_id": int(class_ids[di]),
                "lost": 0,
            }
            out_b.append(boxes[di])
            out_s.append(scores[di])
            out_c.append(class_ids[di])
            out_t.append(tid)

        self._tracks = {k: v for k, v in self._tracks.items()
                        if v["lost"] <= self._max_lost}

        if not out_b:
            return empty
        return (
            np.array(out_b, np.float32), np.array(out_s, np.float32),
            np.array(out_c, int), np.array(out_t, int),
        )


# ── Worker ───────────────────────────────────────────────────────────────────


@dataclass
class _Stats:
    decoded: int = 0
    motion_skipped: int = 0
    inferred: int = 0
    events: int = 0
    triton_errors: int = 0
    last_event_ts: float = 0.0


def _open_pyav_cuvid(rtsp_url: str) -> tuple:
    """Open RTSP source and return (container, stream, decoder_ctx).
    decoder_ctx is a CodecContext bound to h264_cuvid (NVDEC)."""
    container = av.open(
        rtsp_url,
        options={
            "rtsp_transport": "tcp",     # matches production worker config
            "stimeout": "5000000",       # 5s socket timeout
        },
    )
    stream = container.streams.video[0]
    cuvid = av.codec.CodecContext.create("h264_cuvid", "r")
    if stream.codec_context.extradata:
        cuvid.extradata = stream.codec_context.extradata
    return container, stream, cuvid


def _wait_for_triton(client: triton_http.InferenceServerClient, timeout_s: int = 60) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if client.is_server_ready() and client.is_model_ready("yoloworld"):
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _load_queries(meta_path: str) -> list[str]:
    """Load the same query list the production service uses (from meta.json),
    falling back to QUERIES_FALLBACK if the file is unreadable."""
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        qs = meta.get("queries") or meta.get("engine_queries") or []
        if isinstance(qs, list) and qs:
            return [str(q) for q in qs]
    except Exception as exc:
        print(f"[proto] meta.json unreadable, using fallback queries: {exc}",
              file=sys.stderr)
    return list(QUERIES_FALLBACK)


def run(rtsp_url: str, triton_url: str, queries: list[str],
        duration_s: int, threshold: float) -> _Stats:
    print(f"[proto] connecting to Triton at {triton_url}", flush=True)
    client = triton_http.InferenceServerClient(triton_url)
    if not _wait_for_triton(client):
        raise RuntimeError(f"Triton at {triton_url} did not become ready")
    print(f"[proto] Triton ready; queries={queries}", flush=True)

    print(f"[proto] opening RTSP {rtsp_url}", flush=True)
    container, stream, cuvid = _open_pyav_cuvid(rtsp_url)
    print(f"[proto] RTSP open; codec={stream.codec_context.name} "
          f"size={stream.codec_context.width}x{stream.codec_context.height}",
          flush=True)

    tracker = _SimpleTracker()
    fired: dict[tuple, float] = {}
    score_buf: dict[tuple, deque] = {}

    motion_prev: np.ndarray | None = None
    last_send_ts = 0.0
    stats = _Stats()
    deadline = time.monotonic() + duration_s

    try:
        for packet in container.demux(stream):
            if time.monotonic() > deadline:
                break

            for frame in cuvid.decode(packet):
                stats.decoded += 1

                # FPS gate — match production INFER_FPS so we don't out-run Triton
                now = time.monotonic()
                if now - last_send_ts < 1.0 / INFER_FPS:
                    continue
                last_send_ts = now

                # NV12 → BGR for cv2-compatible processing.  PyAV handles this
                # via reformat() which keeps the conversion on whatever device
                # the codec chose.
                img = frame.to_ndarray(format="bgr24")

                # Motion gate
                gray = cv2.resize(
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), MOTION_SCALE
                )
                if motion_prev is not None and \
                        cv2.absdiff(gray, motion_prev).mean() < MOTION_THRESHOLD:
                    stats.motion_skipped += 1
                    motion_prev = gray
                    continue
                motion_prev = gray

                # JPEG-encode (Triton's IMAGE input is UINT8 JPEG bytes)
                ok, jpeg_buf = cv2.imencode(
                    ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                if not ok:
                    continue
                jpeg_bytes = jpeg_buf.tobytes()

                img_in = triton_http.InferInput("IMAGE", [len(jpeg_bytes)], "UINT8")
                img_in.set_data_from_numpy(
                    np.frombuffer(jpeg_bytes, dtype=np.uint8).copy()
                )
                thr_in = triton_http.InferInput("THRESHOLD", [1], "FP32")
                thr_in.set_data_from_numpy(np.array([threshold], dtype=np.float32))

                try:
                    resp = client.infer("yoloworld", inputs=[img_in, thr_in])
                except Exception as exc:
                    stats.triton_errors += 1
                    print(f"[proto] Triton infer error: {exc}", file=sys.stderr)
                    continue
                stats.inferred += 1

                boxes = resp.as_numpy("BOXES")
                scores = resp.as_numpy("SCORES")
                label_ids = resp.as_numpy("LABEL_IDS")

                if len(boxes) == 0:
                    continue

                class_ids = label_ids.astype(int)
                t_boxes, t_scores, t_cids, t_tids = tracker.update(
                    boxes.astype(np.float32), scores.astype(np.float32), class_ids
                )

                wall = time.time()
                for i in range(len(t_boxes)):
                    track_id = int(t_tids[i])
                    q_idx = int(t_cids[i])
                    label = queries[q_idx] if q_idx < len(queries) else ""
                    if not label:
                        continue
                    conf = float(t_scores[i])
                    key = (track_id, q_idx)

                    # Same RECHECK_INTERVAL_S + 3-frame mean as production
                    if now - fired.get(key, 0.0) <= RECHECK_INTERVAL_S:
                        continue
                    score_buf.setdefault(key, deque(maxlen=3)).append(conf)
                    mean_conf = sum(score_buf[key]) / len(score_buf[key])
                    if mean_conf < threshold:
                        continue
                    fired[key] = now
                    score_buf.pop(key, None)

                    stats.events += 1
                    stats.last_event_ts = wall
                    print(
                        f"[event] t={wall:.2f} track={track_id} "
                        f"query={label!r} conf={conf:.3f}",
                        flush=True,
                    )
    finally:
        container.close()

    return stats


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Phase 1.1 inference prototype")
    p.add_argument(
        "--rtsp",
        default=os.environ.get("PROTOTYPE_RTSP",
                               "rtsp://bianca-mediamtx-prototype:8554/test"),
    )
    p.add_argument(
        "--triton",
        default=os.environ.get("PROTOTYPE_TRITON_URL", "bianca-triton:8002"),
        help="Triton HTTP host:port",
    )
    p.add_argument(
        "--meta",
        default=os.environ.get("PROTOTYPE_META",
                               "/data/yoloworld.meta.json"),
        help="Path to yoloworld.meta.json (for query list)",
    )
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--duration", type=int, default=60,
                   help="Seconds to run before exiting cleanly")
    args = p.parse_args(argv)

    queries = _load_queries(args.meta)
    stats = run(args.rtsp, args.triton, queries, args.duration, args.threshold)

    # Final summary, machine-friendly JSON line at the end
    motion_pct = (
        100.0 * stats.motion_skipped / max(1, stats.decoded)
        if stats.decoded else 0.0
    )
    summary = {
        "decoded": stats.decoded,
        "motion_skipped": stats.motion_skipped,
        "motion_skip_pct": round(motion_pct, 1),
        "inferred": stats.inferred,
        "events": stats.events,
        "triton_errors": stats.triton_errors,
    }
    print(f"[summary] {json.dumps(summary)}", flush=True)
    return 0 if stats.triton_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
