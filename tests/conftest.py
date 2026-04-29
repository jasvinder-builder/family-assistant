"""
Phase 0 regression harness — shared fixtures.

Talks to the running docker compose stack via its public HTTP API on
http://localhost:8000.  Does not import any production module; treats the
stack as a black box.

Pre-condition: `docker compose up -d --wait` has already been run.  The
fixture below verifies that and skips the whole session loudly if not.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Iterator

import httpx
import pytest

APP_URL = os.environ.get("BIANCA_APP_URL", "http://localhost:8000")
DEEPSTREAM_URL = os.environ.get("BIANCA_DEEPSTREAM_URL", "http://localhost:8090")

# A stable cam_id used by the smoke tests.  Chosen so it does not collide with
# any human-curated camera in cameras.json.
SMOKE_CAM_ID = "_smoke_test_cam"

# Default test source: MediaMTX container that loops test.mp4 (see
# docker-compose.test.yml).  Overridable via env so a CI or dev environment
# can point at a real camera if desired.
SMOKE_SOURCE = os.environ.get(
    "BIANCA_SMOKE_SOURCE",
    "rtsp://bianca-test-rtsp-source:8554/test",
)


# ── Session-scoped: stack is healthy ──────────────────────────────────────────


@pytest.fixture(scope="session")
def app_url() -> str:
    return APP_URL


@pytest.fixture(scope="session", autouse=True)
def _stack_up() -> Iterator[None]:
    """Verify the stack is reachable; otherwise skip everything with a clear msg."""
    deadline = time.monotonic() + 5
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{APP_URL}/", timeout=2.0)
            if r.status_code == 200:
                yield
                return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(0.3)
    pytest.skip(
        f"bianca-app not reachable at {APP_URL}: {last_err}\n"
        "Run `docker compose up -d --wait` first."
    )


# ── Per-test: ensure SMOKE_CAM_ID is absent before and after ──────────────────


def _delete_stream(cam_id: str) -> None:
    """Best-effort teardown.  Ignores 404 and timeouts.

    30 s timeout because dynamic-container teardown (Savant RTSP adapter +
    sink) sometimes overruns 10 s under Docker pressure.  This is a known
    pre-Phase-1 cost; Phase 1 (MediaMTX) eliminates it.
    """
    try:
        httpx.delete(f"{APP_URL}/cameras/streams/{cam_id}", timeout=30.0)
    except Exception:
        pass


def _list_streams() -> dict:
    r = httpx.get(f"{APP_URL}/cameras/streams", timeout=5.0)
    r.raise_for_status()
    return r.json().get("streams", {})


@pytest.fixture
def clean_smoke_cam() -> Iterator[str]:
    """Yield SMOKE_CAM_ID.  Ensures it is removed both before and after the test
    so tests cannot influence each other or the running balcony camera."""
    _delete_stream(SMOKE_CAM_ID)
    # Wait until gone before yielding
    deadline = time.monotonic() + 5
    while SMOKE_CAM_ID in _list_streams() and time.monotonic() < deadline:
        time.sleep(0.2)
    yield SMOKE_CAM_ID
    _delete_stream(SMOKE_CAM_ID)


@pytest.fixture
def smoke_source() -> str:
    return SMOKE_SOURCE


# ── Helpers shared between tests ──────────────────────────────────────────────


def docker_running_containers(name_substring: str) -> list[str]:
    """Return names of running containers whose name contains the substring."""
    out = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return [n for n in out.stdout.split() if name_substring in n]
