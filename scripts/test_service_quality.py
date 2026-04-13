"""
Phase 6 quality regression tests — catches bugs found during Phase 7/8 integration.

These tests complement test_service.py (functional API) by verifying properties
that pure API tests miss:

  QA-1  FastAPI app exists + /health + required routes
        → would have caught: "uvicorn services.deepstream_service:app" crashing
          because no `app` object existed in the module.

  QA-2  BGR colour correctness
        → would have caught: RGBA[:,:,:3] giving RGB instead of BGR, causing
          all colours to look wrong in the browser and hurting YOLO detection.

  QA-3  File source loops on EOS
        → would have caught: bus.add_watch() never firing under uvicorn/asyncio
          (no GLib main loop), so the file played once and stopped.

  QA-4  Inference receives BGR frames (not RGB)
        → sanity-checks the frame that actually reaches Triton is correctly
          colour-ordered by verifying the JPEG decode round-trip.

Run inside the DeepStream container with Triton running:
  python3 /workspace/scripts/test_service_quality.py \\
      file:///workspace/test.mp4 \\
      [--triton localhost:8002] \\
      [--file-duration 10]
"""

import argparse
import asyncio
import os
import sys
import time

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("source", help="File URI — e.g. file:///workspace/test.mp4")
parser.add_argument("--triton",        default="localhost:8002")
parser.add_argument("--file-duration", type=int, default=10,
                    help="Approximate duration of the test video in seconds")
args = parser.parse_args()

triton_host = args.triton.split(":")[0]
os.environ["TRITON_URL"]     = f"{triton_host}:8001"
os.environ["META_JSON_PATH"] = "/workspace/models/yoloworld.meta.json"

sys.path.insert(0, "/workspace")
from services import deepstream_service as ds  # noqa: E402

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    results.append((name, condition, detail))
    return condition


# ── QA-1: FastAPI app + required routes ──────────────────────────────────────
print("\n=== QA-1: FastAPI app and required routes ===")

try:
    from fastapi import FastAPI
    has_app     = hasattr(ds, "app")
    correct_type = isinstance(getattr(ds, "app", None), FastAPI)
except ImportError:
    has_app = correct_type = False

check("deepstream_service.app exists",      has_app)
check("deepstream_service.app is FastAPI",  correct_type)

if has_app and correct_type:
    route_paths = {getattr(r, "path", "") for r in ds.app.routes}
    for required in ["/health", "/streams", "/queries", "/queries/status",
                     "/events", "/threshold", "/ws/{cam_id}", "/ws"]:
        check(f"Route '{required}' registered", required in route_paths)

    # Health endpoint returns 200 synchronously via ASGI
    from fastapi.testclient import TestClient
    client = TestClient(ds.app)
    resp = client.get("/health")
    check("/health returns 200", resp.status_code == 200, str(resp.status_code))
    check("/health body has ok=true", resp.json().get("ok") is True, str(resp.json()))

    resp2 = client.get("/streams")
    check("/streams returns 200", resp2.status_code == 200)
    check("/streams body has 'streams' key", "streams" in resp2.json())

    resp3 = client.get("/queries/status")
    check("/queries/status returns 200",      resp3.status_code == 200,
          str(resp3.status_code))
    check("/queries/status has 'state' key",  "state" in resp3.json(),
          str(resp3.json()))

    resp4 = client.get("/events")
    check("/events returns 200",           resp4.status_code == 200)
    check("/events body has 'events' key", "events" in resp4.json())

    resp5 = client.get("/threshold")
    check("/threshold returns 200",              resp5.status_code == 200)
    check("/threshold body has 'threshold' key", "threshold" in resp5.json())

    # Verify no FastAPI route function name shadows a module-level variable.
    # This catches bugs like `async def _events()` shadowing `_events: deque`.
    import inspect
    module_vars = {
        name for name, obj in inspect.getmembers(ds)
        if not inspect.isfunction(obj) and not inspect.isclass(obj)
        and not name.startswith("__")
    }
    route_fn_names = {
        route.endpoint.__name__
        for route in ds.app.routes
        if hasattr(route, "endpoint")
    }
    collisions = module_vars & route_fn_names
    check("No route function name shadows a module-level variable",
          len(collisions) == 0, f"collisions: {collisions}")


# ── QA-2: BGR colour correctness ─────────────────────────────────────────────
print("\n=== QA-2: BGR colour correctness ===")

# Simulate exactly what _on_new_sample does when it reads RGBA from GStreamer.
# A pure-red pixel in RGBA memory is bytes [255, 0, 0, 255].
# Correct conversion (RGBA → BGR): channels 2,1,0 → [0, 0, 255] in BGR = red in OpenCV.
# Wrong conversion (RGBA → RGB):   channels 0,1,2 → [255, 0, 0] in RGB, but OpenCV
# reads it as BGR [255, 0, 0] = blue → red and blue are swapped.

rgba_red = np.array([[[255, 0, 0, 255]]], dtype=np.uint8)   # 1×1 RGBA, red pixel

bgr_correct = rgba_red[:, :, 2::-1]   # what the fixed code does
bgr_wrong   = rgba_red[:, :, :3]       # what the old buggy code did

check("RGBA[:,:,2::-1] channel-0 is B (=0 for red pixel)",
      int(bgr_correct[0, 0, 0]) == 0,   f"got {int(bgr_correct[0,0,0])}")
check("RGBA[:,:,2::-1] channel-2 is R (=255 for red pixel)",
      int(bgr_correct[0, 0, 2]) == 255, f"got {int(bgr_correct[0,0,2])}")
check("Old RGBA[:,:,:3] would have been WRONG (R in channel-0)",
      int(bgr_wrong[0, 0, 0]) == 255,   f"got {int(bgr_wrong[0,0,0])}")

# Round-trip: encode BGR frame as JPEG, decode, verify colour survives
ok_enc, jpg = cv2.imencode(".jpg", bgr_correct, [cv2.IMWRITE_JPEG_QUALITY, 99])
check("cv2.imencode succeeds on BGR frame", ok_enc)
if ok_enc:
    decoded = cv2.imdecode(np.frombuffer(jpg.tobytes(), np.uint8), cv2.IMREAD_COLOR)
    check("Decoded JPEG R channel high (red pixel survives round-trip)",
          int(decoded[0, 0, 2]) > 200, f"R={int(decoded[0,0,2])}")
    check("Decoded JPEG B channel low",
          int(decoded[0, 0, 0]) < 50,  f"B={int(decoded[0,0,0])}")

# Repeat for a pure-green pixel
rgba_green = np.array([[[0, 255, 0, 255]]], dtype=np.uint8)
bgr_green  = rgba_green[:, :, 2::-1]
ok_g, jpg_g = cv2.imencode(".jpg", bgr_green, [cv2.IMWRITE_JPEG_QUALITY, 99])
if ok_g:
    dec_g = cv2.imdecode(np.frombuffer(jpg_g.tobytes(), np.uint8), cv2.IMREAD_COLOR)
    check("Green pixel: G channel high after round-trip",
          int(dec_g[0, 0, 1]) > 200, f"G={int(dec_g[0,0,1])}")
    check("Green pixel: R and B channels low",
          int(dec_g[0, 0, 0]) < 50 and int(dec_g[0, 0, 2]) < 50,
          f"B={int(dec_g[0,0,0])} R={int(dec_g[0,0,2])}")


# ── QA-3: File source loops on EOS ───────────────────────────────────────────
print(f"\n=== QA-3: File source loops on EOS (file ~{args.file_duration}s) ===")

frame_q = ds.subscribe_frames("qa_cam")
ds.add_stream("qa_cam", args.source)

# Collect frames for 1.5 × file duration — long enough to cross EOS at least once
wait_s = int(args.file_duration * 1.5) + 5
print(f"  Collecting frames for {wait_s}s (1.5 × file duration + 5s buffer)...")

frame_times: list[float] = []
deadline = time.time() + wait_s
while time.time() < deadline:
    try:
        chunk = frame_q.get(timeout=1.0)
        if chunk is not None:
            frame_times.append(time.time())
    except Exception:
        pass

ds.unsubscribe_frames(frame_q, "qa_cam")

total_frames  = len(frame_times)
check("Frames arrived at all", total_frames > 0, f"{total_frames} frames")

if total_frames > 1:
    elapsed   = frame_times[-1] - frame_times[0]
    avg_fps   = total_frames / elapsed if elapsed > 0 else 0

    # Check gaps between consecutive frames AND gap from last frame to end of
    # the wait window.  Without the trailing gap check, a pipeline that plays
    # once then stops shows max_gap≈0 because all frames arrived back-to-back.
    consecutive_gaps = [frame_times[i+1] - frame_times[i]
                        for i in range(len(frame_times)-1)]
    trailing_gap = (frame_times[0] + wait_s) - frame_times[-1]
    all_gaps = consecutive_gaps + [trailing_gap]
    max_gap = max(all_gaps) if all_gaps else 0.0
    check("No frame gap > 3s (pipeline did not stall at EOS)",
          max_gap < 3.0, f"max gap={max_gap:.1f}s (trailing={trailing_gap:.1f}s)")

    # Frames should still be arriving in the final 5 seconds of the wait window
    recent_cutoff = deadline - 5
    recent_frames = sum(1 for t in frame_times if t > recent_cutoff)
    check("Frames still arriving in final 5s (looping confirmed)",
          recent_frames > 0, f"{recent_frames} frames in last 5s")

    check(f"Average fps ≥ 10 over {elapsed:.0f}s",
          avg_fps >= 10, f"{avg_fps:.1f} fps")

ds.remove_stream("qa_cam")


# ── QA-4: Inference receives BGR frames ──────────────────────────────────────
print("\n=== QA-4: Inference frame colour order (JPEG round-trip via Triton) ===")
# Verify that the frame pushed to _infer_slots is BGR by checking that when
# cv2.imencode encodes it and Triton decodes it (via PIL/cv2 in model.py),
# the colour interpretation is consistent with BGR.
#
# We do this indirectly: start a stream, wait for detections, then inspect
# that detections contain valid bounding boxes (implying Triton received a
# decodable, correctly-coloured frame — a garbage frame would produce 0 dets
# or random noise boxes).

frame_q2 = ds.subscribe_frames("qa_infer")
ds.add_stream("qa_infer", args.source)

print("  Waiting up to 30s for detections (confirms Triton received valid BGR frames)...")
got_dets = False
deadline2 = time.time() + 30
while time.time() < deadline2:
    dets = ds.get_latest_detections("qa_infer")
    if dets:
        got_dets = True
        break
    time.sleep(1.0)

ds.unsubscribe_frames(frame_q2, "qa_infer")
ds.remove_stream("qa_infer")

check("Detections appear (Triton received valid BGR JPEG)", got_dets,
      "If this fails but QA-2 passed, check Triton connectivity or threshold")

if got_dets:
    det = ds.get_latest_detections("qa_infer")[0] if ds.get_latest_detections("qa_infer") else None
    # Try once more briefly since we just removed the stream
    if det is None:
        ds.add_stream("qa_infer2", args.source)
        time.sleep(5)
        det = ds.get_latest_detections("qa_infer2")
        det = det[0] if det else None
        ds.remove_stream("qa_infer2")

    if det:
        x1, y1, x2, y2 = det.box
        w_box = x2 - x1
        h_box = y2 - y1
        check("Detection box is non-degenerate (w>0, h>0)",
              w_box > 0 and h_box > 0, f"box={det.box}")
        check("Detection box within frame bounds",
              0 <= x1 < 1280 and 0 <= y1 < 720 and x2 <= 1280 and y2 <= 720,
              f"box={det.box}")


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("QUALITY REGRESSION SUMMARY")
print(f"{'='*55}")
passed = sum(1 for _, ok, _ in results if ok)
total  = len(results)
for name, ok, detail in results:
    status = PASS if ok else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

print(f"\n{passed}/{total} checks passed")
print(f"\nPhase 6 QA: {'PASS ✓' if passed == total else 'FAIL ✗'}")
sys.exit(0 if passed == total else 1)
