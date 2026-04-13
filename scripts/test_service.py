"""
Phase 6 — deepstream_service standalone test.

Tests the service API without FastAPI. Exercises:
  1. Query management (add/remove, status, meta.json persistence)
  2. Stream add → frames flow → subscribers receive JPEG
  3. Inference fires → detections appear
  4. Events appear within 60s
  5. Query hot-swap (add query, wait for status=ready, verify in get_queries)
  6. remove_stream cleans up state

Run inside the DeepStream container with Triton running:
  python3 /workspace/scripts/test_service.py \\
      file:///workspace/test.mp4 \\
      [--triton localhost:8002] \\
      [--duration 60]
"""

import argparse
import asyncio
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Point service at correct Triton URL before importing
parser = argparse.ArgumentParser()
parser.add_argument("source", help="RTSP URL or file URI")
parser.add_argument("--triton",   default="localhost:8002",
                    help="Triton HTTP host:port")
parser.add_argument("--duration", type=int, default=60)
args = parser.parse_args()

# Inject env vars before service import so module-level constants pick them up
triton_host = args.triton.split(":")[0]
os.environ["TRITON_URL"]     = f"{triton_host}:8001"
os.environ["META_JSON_PATH"] = "/workspace/models/yoloworld.meta.json"

# Add project root to path
sys.path.insert(0, "/workspace")

from services import deepstream_service as ds  # noqa: E402 (import after env setup)


PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    results.append((name, condition, detail))
    return condition


# ── Test 1: Module import + initial queries ───────────────────────────────────
print("\n=== Test 1: Module import + initial query load ===")
queries = ds.get_queries()
check("get_queries() returns a list", isinstance(queries, list))
check("Initial queries non-empty", len(queries) > 0, str(queries))
check("get_threshold() is float", isinstance(ds.get_threshold(), float))
check("get_query_status() state=ready",
      ds.get_query_status()["state"] == "ready")
check("get_streams() is empty", ds.get_streams() == {})
check("get_latest_detections() is list",
      isinstance(ds.get_latest_detections(), list))
check("get_events() is list", isinstance(ds.get_events(), list))


# ── Test 2: add_query / remove_query ─────────────────────────────────────────
print("\n=== Test 2: add_query / remove_query ===")
initial_count = len(ds.get_queries())

added = ds.add_query("__test_query__")
check("add_query returns True for new query", added)
added_dup = ds.add_query("__test_query__")
check("add_query returns False for duplicate", not added_dup)

# Wait for query update worker to finish
deadline = time.time() + 20
while time.time() < deadline:
    if ds.get_query_status()["state"] == "ready":
        break
    time.sleep(0.5)
check("Query status returns to ready", ds.get_query_status()["state"] == "ready")

q_list = ds.get_queries()
check("Query list contains new query", "__test_query__" in q_list, str(q_list))

# Remove it
idx = q_list.index("__test_query__")
removed = ds.remove_query(idx)
check("remove_query returns True", removed)
check("remove_query(999) returns False", not ds.remove_query(999))

deadline = time.time() + 20
while time.time() < deadline:
    if ds.get_query_status()["state"] == "ready":
        break
    time.sleep(0.5)
check("Query status returns to ready after remove",
      ds.get_query_status()["state"] == "ready")
check("Query list back to initial count",
      len(ds.get_queries()) == initial_count,
      f"{len(ds.get_queries())} vs {initial_count}")


# ── Test 3: set_threshold / set_debug_overlay ─────────────────────────────────
print("\n=== Test 3: Threshold and overlay setters ===")
ds.set_threshold(0.5)
check("set_threshold(0.5) persists", ds.get_threshold() == 0.5)
ds.set_threshold(1.5)   # out-of-range → clamp
check("set_threshold(1.5) clamped to 1.0", ds.get_threshold() == 1.0)
ds.set_threshold(0.3)   # reset
ds.set_debug_overlay(True)
check("set_debug_overlay persists", ds.get_debug_overlay() is True)
ds.set_debug_overlay(False)
ds.set_pad_factor(0.5)  # no-op
check("get_pad_factor() always 0.0", ds.get_pad_factor() == 0.0)
ds.push_frame(None)     # no-op, should not raise
check("push_frame(None) is a no-op", True)


# ── Test 4: add_stream → frames → subscriber ─────────────────────────────────
print(f"\n=== Test 4: add_stream({args.source!r}) → frames flow ===")
frame_q = ds.subscribe_frames("cam0")

ds.add_stream("cam0", args.source)
check("get_streams() contains cam0", "cam0" in ds.get_streams())
check("get_stream_url() returns uri",
      ds.get_stream_url() == args.source)

print("  Waiting up to 15s for first frame...")
first_frame = None
deadline = time.time() + 15
while time.time() < deadline:
    try:
        chunk = frame_q.get(timeout=1.0)
        if chunk is not None:
            first_frame = chunk
            break
    except Exception:
        pass
check("Subscriber received first JPEG frame",
      first_frame is not None,
      f"{len(first_frame)} bytes" if first_frame else "timeout")

# Collect frames for 5 more seconds
frame_count = 1 if first_frame else 0
t0 = time.time()
while time.time() - t0 < 5:
    try:
        chunk = frame_q.get(timeout=1.0)
        if chunk is not None:
            frame_count += 1
    except Exception:
        pass
ds.unsubscribe_frames(frame_q, "cam0")

fps_5s = frame_count / 5
check("Frame rate ≥ 10fps over 5s", fps_5s >= 10, f"{fps_5s:.1f}fps")


# ── Test 5: detections appear ─────────────────────────────────────────────────
print("\n=== Test 5: Detections appear within 30s ===")
print("  Waiting up to 30s for detections...")
got_dets = False
deadline = time.time() + 30
while time.time() < deadline:
    dets = ds.get_latest_detections("cam0")
    if dets:
        got_dets = True
        print(f"  Got {len(dets)} detection(s): "
              f"{[d.label for d in dets[:5]]}")
        break
    time.sleep(1)
check("Detections appear for cam0", got_dets)

if got_dets:
    det = ds.get_latest_detections("cam0")[0]
    check("Detection has track_id", isinstance(det.track_id, int))
    check("Detection has box (4-tuple)", len(det.box) == 4)
    check("Detection has cam_id=cam0", det.cam_id == "cam0")
    check("Detection has label", isinstance(det.label, str) and len(det.label) > 0)


# ── Test 6: Events appear ─────────────────────────────────────────────────────
print(f"\n=== Test 6: Events appear (waiting up to {args.duration - 40}s) ===")
remaining = max(10, args.duration - int(time.time() - (deadline - 30)) - 5)
print(f"  Waiting up to {remaining}s...")
got_events = False
deadline = time.time() + remaining
while time.time() < deadline:
    events = ds.get_events(max_age_hours=1.0)
    if events:
        got_events = True
        print(f"  Got {len(events)} event(s): {events[0]['query']} "
              f"conf={events[0]['confidence']}")
        break
    time.sleep(2)
check("Events appear in event log", got_events,
      "Events require a detection above threshold — may need longer video")

if got_events:
    ev = ds.get_events()[0]
    check("Event has timestamp", "timestamp" in ev)
    check("Event has cam_id", "cam_id" in ev)
    check("Event has image_b64", len(ev.get("image_b64", "")) > 0)


# ── Test 7: ws_frame_generator (async) ───────────────────────────────────────
print("\n=== Test 7: ws_frame_generator yields JPEG frames ===")

async def _test_ws_gen():
    frames = []
    gen = ds.ws_frame_generator("cam0")
    try:
        async for chunk in gen:
            frames.append(chunk)
            if len(frames) >= 5:
                break
    except Exception as exc:
        return False, str(exc)
    return len(frames) >= 5, f"{len(frames)} frames"

ok_ws, detail_ws = asyncio.run(_test_ws_gen())
check("ws_frame_generator yields ≥5 frames", ok_ws, detail_ws)


# ── Test 8: remove_stream cleans up ──────────────────────────────────────────
print("\n=== Test 8: remove_stream cleans up ===")
ds.remove_stream("cam0")
time.sleep(1)
check("get_streams() is empty after remove", ds.get_streams() == {})
check("get_stream_url() is None after remove", ds.get_stream_url() is None)
dets_after = ds.get_latest_detections("cam0")
check("Detections cleared for removed cam", dets_after == [])


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("SUMMARY")
print(f"{'='*55}")
passed = sum(1 for _, ok, _ in results if ok)
total  = len(results)
for name, ok, detail in results:
    status = PASS if ok else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

print(f"\n{passed}/{total} checks passed")
print(f"\nPhase 6: {'PASS ✓' if passed == total else 'FAIL ✗'}")
sys.exit(0 if passed == total else 1)
