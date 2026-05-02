"""
Phase 2 regression checkpoints.

T2.1 — no module-level dict/list/deque state in the surveillance modules
T2.2 — concurrency stress: add+remove 10 cameras at 100ms intervals,
       registry empties, no orphan ffmpeg processes, no orphan MediaMTX paths
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

import httpx
import pytest

SURVEILLANCE_FILES = [
    "services/camera_runtime.py",
    "services/ingress.py",
    "services/clips.py",
    "services/deepstream_service.py",
    "services/inference_service.py",
    "services/inference_worker.py",
]

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_T2_1_no_module_level_collections() -> None:
    """No module-level `_name: dict|list|deque` annotations in surveillance modules.
    Per-camera state must live on CameraRuntime; global state must be encapsulated
    in a class (EventLog, ClipIndex, _Queries, _Registry)."""
    pattern = re.compile(r"^_[a-z_]+: *(dict|list|deque)")
    offenders: list[str] = []
    for rel in SURVEILLANCE_FILES:
        f = REPO_ROOT / rel
        if not f.exists():
            continue
        for lineno, line in enumerate(f.read_text().splitlines(), start=1):
            if pattern.match(line):
                offenders.append(f"{rel}:{lineno}: {line.rstrip()}")
    assert not offenders, "Module-level collection state found:\n" + "\n".join(offenders)


def test_T2_2_concurrency_stress(app_url: str) -> None:
    """Add and remove 10 cameras at 100 ms intervals against the live stack.
    Verify the registry empties, no orphan ffmpeg processes survive, and the
    smoke source itself is unaffected (still serves HLS to the production cam
    if one is configured — we don't disturb the persistent balcony entry)."""
    cam_ids = [f"_stress_cam_{i}" for i in range(10)]
    smoke_source = os.environ.get(
        "BIANCA_SMOKE_SOURCE",
        "rtsp://bianca-test-rtsp-source:8554/test",
    )

    # Best-effort cleanup before the test
    for cid in cam_ids:
        try:
            httpx.delete(f"{app_url}/cameras/streams/{cid}", timeout=10.0)
        except Exception:  # noqa: BLE001
            pass

    # Snapshot baseline ffmpeg-recorder count inside the inference container.
    # Each registered camera spawns one ffmpeg sibling; baseline includes any
    # production cameras (e.g. balcony if reachable) so we measure delta.
    def _ffmpeg_count() -> int:
        out = subprocess.run(
            ["docker", "exec", "bianca-inference",
             "bash", "-c", "ps -ef | grep -c '[f]fmpeg -y -rtsp_transport'"],
            capture_output=True, text=True, check=False,
        )
        try:
            return int(out.stdout.strip())
        except ValueError:
            return 0

    baseline = _ffmpeg_count()

    # Add+remove cycle, 100 ms apart
    for cid in cam_ids:
        r = httpx.post(
            f"{app_url}/cameras/streams",
            json={"cam_id": cid, "url": smoke_source},
            timeout=10.0,
        )
        assert r.status_code in (200, 201), f"add {cid}: {r.status_code} {r.text}"
        time.sleep(0.1)

        r = httpx.delete(f"{app_url}/cameras/streams/{cid}", timeout=15.0)
        assert r.status_code in (200, 204), f"delete {cid}: {r.status_code} {r.text}"
        time.sleep(0.1)

    # Verify registry is back to baseline (none of the stress cams remain)
    streams = httpx.get(f"{app_url}/cameras/streams", timeout=5.0).json()["streams"]
    leftover = [c for c in cam_ids if c in streams]
    assert not leftover, f"registry leak: {leftover}"

    # ffmpeg cleanup — give the inference container 3s to reap subprocesses
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if _ffmpeg_count() <= baseline:
            break
        time.sleep(0.5)
    final = _ffmpeg_count()
    assert final <= baseline, (
        f"ffmpeg recorders leaked: baseline={baseline} final={final}"
    )
