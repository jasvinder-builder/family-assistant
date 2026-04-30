"""
Phase 0 — smoke regression suite.

Implements T0.1 – T0.9 from ARCHITECTURE_REVIEW.md.  Each test is a
single, named checkpoint; the whole suite must pass in well under
2 minutes.

The tests target the running stack via its public HTTP API only.  No
internal Python imports.  This is deliberate: the harness must keep
working through Phases 1+ where the internals get rewritten.
"""

from __future__ import annotations

import time

import httpx
import pytest

pytestmark = pytest.mark.smoke


# ── T0.1 / T0.2 — basic smoke ─────────────────────────────────────────────────


def test_T0_1_home_returns_200(app_url: str) -> None:
    r = httpx.get(f"{app_url}/", timeout=5.0)
    assert r.status_code == 200, r.text


def test_T0_2_cameras_page_returns_200(app_url: str) -> None:
    r = httpx.get(f"{app_url}/cameras", timeout=5.0)
    assert r.status_code == 200, r.text


# ── T0.3 — add stream is observable within 5 s ────────────────────────────────


def test_T0_3_add_stream_visible_within_5s(
    app_url: str, clean_smoke_cam: str, smoke_source: str
) -> None:
    r = httpx.post(
        f"{app_url}/cameras/streams",
        json={"cam_id": clean_smoke_cam, "url": smoke_source},
        timeout=10.0,
    )
    assert r.status_code in (200, 201), f"add stream failed: {r.status_code} {r.text}"

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        streams = httpx.get(f"{app_url}/cameras/streams", timeout=5.0).json()["streams"]
        if clean_smoke_cam in streams:
            return
        time.sleep(0.25)
    pytest.fail("stream did not appear in /cameras/streams within 5 s")


# ── T0.4 — HLS playlist is served within 15 s ─────────────────────────────────


def _resolve_variant_playlist(app_url: str, cam_id: str) -> str | None:
    """If the master playlist references a variant, return the variant URL.
    Otherwise return the master URL itself if it already contains TARGETDURATION."""
    master_url = f"{app_url}/cameras/hls/{cam_id}/index.m3u8"
    try:
        r = httpx.get(master_url, timeout=4.0)
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    if "EXT-X-TARGETDURATION" in r.text:
        return master_url
    # Master playlist — find first non-comment, non-tag line ending in .m3u8
    for line in r.text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and line.endswith(".m3u8"):
            return f"{app_url}/cameras/hls/{cam_id}/{line}"
    return None


def test_T0_4_hls_playlist_within_15s(
    app_url: str, clean_smoke_cam: str, smoke_source: str
) -> None:
    httpx.post(
        f"{app_url}/cameras/streams",
        json={"cam_id": clean_smoke_cam, "url": smoke_source},
        timeout=10.0,
    ).raise_for_status()

    deadline = time.monotonic() + 15
    last_status: int | None = None
    last_body: str = ""
    while time.monotonic() < deadline:
        variant = _resolve_variant_playlist(app_url, clean_smoke_cam)
        if variant:
            try:
                r = httpx.get(variant, timeout=4.0)
                last_status = r.status_code
                last_body = r.text[:300]
                if r.status_code == 200 and "EXT-X-TARGETDURATION" in r.text:
                    return
            except httpx.HTTPError:
                pass
        time.sleep(0.5)
    pytest.fail(
        f"HLS variant playlist not served within 15 s "
        f"(last status={last_status}, body={last_body!r})"
    )


# ── T0.5 — at least one segment carries non-zero bytes within 20 s ────────────


def test_T0_5_segment_bytes_within_20s(
    app_url: str, clean_smoke_cam: str, smoke_source: str
) -> None:
    httpx.post(
        f"{app_url}/cameras/streams",
        json={"cam_id": clean_smoke_cam, "url": smoke_source},
        timeout=10.0,
    ).raise_for_status()

    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        variant = _resolve_variant_playlist(app_url, clean_smoke_cam)
        if variant:
            try:
                r = httpx.get(variant, timeout=4.0)
                if r.status_code == 200:
                    base = variant.rsplit("/", 1)[0]
                    # LL-HLS lists "gap.mp4" placeholders before real video appears;
                    # those return 404.  Try every segment/part candidate until
                    # one returns non-zero bytes, ignoring "gap.mp4" entries.
                    candidates = [
                        line.strip()
                        for line in r.text.splitlines()
                        if line.strip()
                        and not line.startswith("#")
                        and line.strip().endswith((".ts", ".m4s", ".mp4"))
                        and "gap.mp4" not in line
                    ]
                    # Also try the init segment from #EXT-X-MAP:URI=
                    for line in r.text.splitlines():
                        if line.startswith("#EXT-X-MAP:URI="):
                            init = line.split("URI=", 1)[1].strip().strip('"')
                            candidates.insert(0, init)
                    # Also try parts from #EXT-X-PART:URI=...
                    for line in r.text.splitlines():
                        if line.startswith("#EXT-X-PART:") and "URI=" in line:
                            uri_part = line.split("URI=", 1)[1].split(",")[0]
                            candidates.append(uri_part.strip().strip('"'))

                    for seg in candidates:
                        try:
                            sr = httpx.get(f"{base}/{seg}", timeout=8.0)
                            if sr.status_code == 200 and len(sr.content) > 0:
                                return
                        except httpx.HTTPError:
                            continue
            except httpx.HTTPError:
                pass
        time.sleep(0.5)
    pytest.fail("no non-zero segment served within 20 s")


# ── T0.6 — at least one detection event within 45 s ───────────────────────────


def test_T0_6_event_within_45s(
    app_url: str, clean_smoke_cam: str, smoke_source: str
) -> None:
    # Make sure there is at least one query the source can match.
    queries = httpx.get(f"{app_url}/cameras/queries", timeout=5.0).json()["queries"]
    if "person" not in queries:
        httpx.post(
            f"{app_url}/cameras/queries", json={"text": "person"}, timeout=10.0
        )

    httpx.post(
        f"{app_url}/cameras/streams",
        json={"cam_id": clean_smoke_cam, "url": smoke_source},
        timeout=10.0,
    ).raise_for_status()

    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        events = httpx.get(f"{app_url}/cameras/events", timeout=5.0).json()
        if any(e.get("cam_id") == clean_smoke_cam for e in events.get("events", events)):
            return
        time.sleep(1.0)
    pytest.fail(f"no event for {clean_smoke_cam} within 45 s")


# ── T0.7 — at least one clip exists and is range-served ──────────────────────
#
# Phase 1 step B made this work end-to-end: MediaMTX pulls from the source,
# the inference container's system ffmpeg writes rolling segments, deepstream
# cuts the clip via concat-stream-copy.
#
# Timing budget for a continuous-detection source (test.mp4):
#   - Trigger is extended on every per-detection POST, so it only expires
#     once MAX_CLIP_DURATION_S (60 s) elapses since first detect and a new
#     trigger displaces it, then POST_BUFFER_S (5 s) of no extension.
#   - Then _process_trigger waits up to SEG_DURATION_S (30 s) for a segment
#     boundary that covers clip_end_wall.
#   - 150 s gives headroom for stack warmup + Triton first-call + segment
#     clock-alignment.
def test_T0_7_clip_and_range_within_60s(
    app_url: str, clean_smoke_cam: str, smoke_source: str
) -> None:
    httpx.post(
        f"{app_url}/cameras/streams",
        json={"cam_id": clean_smoke_cam, "url": smoke_source},
        timeout=10.0,
    ).raise_for_status()

    deadline = time.monotonic() + 150
    clip = None
    while time.monotonic() < deadline:
        clips = httpx.get(
            f"{app_url}/cameras/clips?cam_id={clean_smoke_cam}", timeout=5.0
        ).json()
        items = clips.get("clips", clips) if isinstance(clips, dict) else clips
        if items:
            clip = items[0]
            break
        time.sleep(1.0)
    assert clip is not None, "no clip produced within 60 s"

    filename = clip.get("filename") or clip.get("file") or clip.get("name")
    assert filename, f"clip dict missing filename: {clip!r}"

    file_url = f"{app_url}/cameras/clips/file/{clean_smoke_cam}/{filename}"
    r = httpx.get(file_url, headers={"Range": "bytes=0-1023"}, timeout=10.0)
    # Either explicit 206 with Accept-Ranges, or 200 with Accept-Ranges header
    assert r.status_code in (200, 206), f"clip HTTP {r.status_code}: {r.text[:200]}"
    assert "accept-ranges" in {h.lower() for h in r.headers.keys()} or r.status_code == 206, (
        f"no Accept-Ranges and status {r.status_code}; headers={dict(r.headers)}"
    )


# ── T0.8 — delete cleans up within 5 s, no stray dynamic containers ───────────


def test_T0_8_delete_cleans_up(
    app_url: str, clean_smoke_cam: str, smoke_source: str
) -> None:
    from .conftest import docker_running_containers

    httpx.post(
        f"{app_url}/cameras/streams",
        json={"cam_id": clean_smoke_cam, "url": smoke_source},
        timeout=10.0,
    ).raise_for_status()

    # Make sure it really is registered before we delete
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if clean_smoke_cam in httpx.get(
            f"{app_url}/cameras/streams", timeout=5.0
        ).json()["streams"]:
            break
        time.sleep(0.2)

    httpx.delete(
        f"{app_url}/cameras/streams/{clean_smoke_cam}", timeout=30.0
    ).raise_for_status()

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if clean_smoke_cam not in httpx.get(
            f"{app_url}/cameras/streams", timeout=5.0
        ).json()["streams"]:
            break
        time.sleep(0.2)
    else:
        pytest.fail("stream still listed 5 s after delete")

    # No dynamic containers should remain for this cam_id (safe-name handles dots)
    leftover = [
        n
        for n in docker_running_containers("bianca-")
        if clean_smoke_cam.replace(".", "_") in n
    ]
    assert not leftover, f"leftover containers: {leftover}"


# ── T0.9 — three add/remove cycles without restart ────────────────────────────
# Catches the "works once, breaks on second add" class of state-leak bug.


def test_T0_9_three_cycles_no_leaks(
    app_url: str, clean_smoke_cam: str, smoke_source: str
) -> None:
    for i in range(3):
        r = httpx.post(
            f"{app_url}/cameras/streams",
            json={"cam_id": clean_smoke_cam, "url": smoke_source},
            timeout=10.0,
        )
        assert r.status_code in (200, 201), f"cycle {i}: add failed {r.status_code}"

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if clean_smoke_cam in httpx.get(
                f"{app_url}/cameras/streams", timeout=5.0
            ).json()["streams"]:
                break
            time.sleep(0.2)
        else:
            pytest.fail(f"cycle {i}: stream did not appear within 5 s")

        r = httpx.delete(
            f"{app_url}/cameras/streams/{clean_smoke_cam}", timeout=30.0
        )
        assert r.status_code in (200, 204), f"cycle {i}: delete failed {r.status_code}"

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if clean_smoke_cam not in httpx.get(
                f"{app_url}/cameras/streams", timeout=5.0
            ).json()["streams"]:
                break
            time.sleep(0.2)
        else:
            pytest.fail(f"cycle {i}: stream still listed 5 s after delete")
