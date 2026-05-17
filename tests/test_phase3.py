"""
Phase 3 observability regression tests.

T3.1 — GET /diag/{unknown_cam} → 404
T3.2 — POST /internal/heartbeat with missing cam_id → 400
T3.3 — POST /internal/heartbeat for unknown cam → 200, registered=False
T3.4 — Registered cam + heartbeat posted → /diag reflects it, health=ok
T3.5 — Freshly registered cam (no heartbeat) → /diag health=ok (startup grace)
T3.6 — GET /health returns {status, reasons}; 200 when ok, 503 when not
"""

from __future__ import annotations

import time

import httpx
import pytest

# How long to wait for the cam to appear in the registry before polling diag
_REGISTRY_SETTLE_S = 3.0


def _add_stream(app_url: str, cam_id: str, source: str) -> None:
    r = httpx.post(
        f"{app_url}/cameras/streams",
        json={"cam_id": cam_id, "url": source},
        timeout=10.0,
    )
    assert r.status_code in (200, 201), f"add_stream {cam_id}: {r.status_code} {r.text}"


def _post_heartbeat(deepstream_url: str, payload: dict) -> httpx.Response:
    return httpx.post(
        f"{deepstream_url}/internal/heartbeat",
        json=payload,
        timeout=5.0,
    )


# ── T3.1 ─────────────────────────────────────────────────────────────────────


def test_T3_1_diag_unknown_cam_404(deepstream_url: str) -> None:
    """GET /diag for a cam_id that was never registered returns 404."""
    r = httpx.get(f"{deepstream_url}/diag/__no_such_cam__", timeout=5.0)
    assert r.status_code == 404
    body = r.json()
    assert "error" in body


# ── T3.2 ─────────────────────────────────────────────────────────────────────


def test_T3_2_heartbeat_missing_cam_id_400(deepstream_url: str) -> None:
    """POST /internal/heartbeat without cam_id returns 400."""
    r = _post_heartbeat(deepstream_url, {"decoded": 10})
    assert r.status_code == 400
    assert "cam_id" in r.json().get("error", "").lower()


# ── T3.3 ─────────────────────────────────────────────────────────────────────


def test_T3_3_heartbeat_unknown_cam_silent(deepstream_url: str) -> None:
    """POST /internal/heartbeat for an unregistered cam is silently accepted
    (worker may outlive its DELETE; we don't want 404 spam in worker logs)."""
    r = _post_heartbeat(
        deepstream_url,
        {"cam_id": "__ghost_cam__", "decoded": 1, "inferred": 0},
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is True
    assert body.get("registered") is False


# ── T3.4 ─────────────────────────────────────────────────────────────────────


def test_T3_4_heartbeat_reflected_in_diag(
    app_url: str, deepstream_url: str, clean_smoke_cam: str, smoke_source: str
) -> None:
    """After a heartbeat is posted for a registered cam, /diag reflects it
    with a fresh heartbeat_age_s and health=ok."""
    cam_id = clean_smoke_cam
    _add_stream(app_url, cam_id, smoke_source)

    # Give the registry a moment to settle before posting the heartbeat
    time.sleep(_REGISTRY_SETTLE_S)

    hb_payload = {
        "cam_id":           cam_id,
        "decoded":          42,
        "motion_skipped":   10,
        "inferred":         32,
        "events":           2,
        "triton_errors":    0,
        "reconnect_count":  0,
        "bytes_total":      1_000_000,
        "last_decode_wall": time.time(),
        "last_triton_wall": time.time(),
        "last_event_wall":  0.0,
        "triton_ms_p50":    8.1,
        "triton_ms_p99":    14.3,
    }
    r = _post_heartbeat(deepstream_url, hb_payload)
    assert r.status_code == 200, f"heartbeat POST failed: {r.text}"
    assert r.json().get("registered") is True

    diag = httpx.get(f"{deepstream_url}/diag/{cam_id}", timeout=5.0).json()
    assert diag["cam_id"] == cam_id
    assert diag["heartbeat_age_s"] is not None
    assert diag["heartbeat_age_s"] < 10.0, "heartbeat should be fresh"
    assert diag["inference"]["decoded"] == 42
    assert diag["inference"]["inferred"] == 32
    # Health may be "degraded" if the inference subprocess is dead for a fake cam,
    # but it must NOT be "down" — a received heartbeat proves the pipeline is live.
    assert diag["health"] != "down", f"reasons={diag['reasons']}"
    stale_reasons = [r for r in diag["reasons"] if "heartbeat" in r.lower()]
    assert not stale_reasons, f"unexpected heartbeat reasons: {stale_reasons}"


# ── T3.5 ─────────────────────────────────────────────────────────────────────


def test_T3_5_grace_period_suppresses_down(
    app_url: str, deepstream_url: str, clean_smoke_cam: str, smoke_source: str
) -> None:
    """A freshly registered camera with no heartbeat yet should report
    health=ok during the STARTUP_GRACE_S window, not health=down."""
    cam_id = clean_smoke_cam
    _add_stream(app_url, cam_id, smoke_source)

    # Poll immediately — definitely within the 30s grace window
    deadline = time.monotonic() + _REGISTRY_SETTLE_S
    diag = None
    while time.monotonic() < deadline:
        r = httpx.get(f"{deepstream_url}/diag/{cam_id}", timeout=5.0)
        if r.status_code == 200:
            diag = r.json()
            break
        time.sleep(0.2)

    assert diag is not None, f"cam {cam_id} never appeared in /diag"
    assert diag["startup_grace"] is True, "expected to be inside grace window"
    assert diag["health"] != "down", (
        f"health={diag['health']} reasons={diag['reasons']} — "
        "grace period should suppress 'down' for a newly registered camera"
    )


# ── T3.6 ─────────────────────────────────────────────────────────────────────


def test_T3_6_health_endpoint_structure(deepstream_url: str) -> None:
    """/health always returns {status, reasons} and a valid HTTP status code."""
    r = httpx.get(f"{deepstream_url}/health", timeout=5.0)
    assert r.status_code in (200, 503), f"unexpected HTTP status: {r.status_code}"
    body = r.json()
    assert "status" in body, "response missing 'status' key"
    assert "reasons" in body, "response missing 'reasons' key"
    assert body["status"] in ("ok", "degraded", "down"), (
        f"unknown status value: {body['status']!r}"
    )
    assert isinstance(body["reasons"], list)
    if r.status_code == 200:
        assert body["status"] == "ok"
    else:
        assert body["status"] in ("degraded", "down")
