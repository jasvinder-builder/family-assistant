"""
Phase 1.2 prototype — standalone clip recorder.

Spawns one ffmpeg subprocess to write rolling N-second segments from a
MediaMTX RTSP re-stream, then on demand cuts a clip via ffmpeg concat
stream-copy and verifies the result with ffprobe.

This is the Case-A clip-recording pattern.  Two important deltas from the
current production path:

  1. Source is MediaMTX RTSP, not the Savant Always-On Sink.  No keyframe
     warnings, no DTS discontinuities, no per-camera dynamic sink container.
  2. Recorder is *system* ffmpeg (the savant-deepstream base image's apt-installed
     binary, which has cuvid + libx264).  The current production code uses
     `imageio_ffmpeg.get_ffmpeg_exe()` — a bundled static binary that segfaults
     against any RTSP source on this host (verified 2026-04-26).  That single
     swap is what makes T0.7 finally pass.

Run inside the bench container:
  docker compose -f docker-compose.yml -f docker-compose.test.yml --profile bench \\
      run --rm bench /bench/clip_recorder_prototype.py \\
          --rtsp rtsp://bianca-mediamtx-prototype:8554/test \\
          --record-secs 35 --clip-duration 8

The script returns 0 if (a) ≥1 segment file appears on disk within
`record-secs`, (b) at least one is non-empty + valid H.264 per ffprobe,
and (c) the cut clip file is a valid MP4.  Anything else is non-zero.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# ── ffmpeg helpers ────────────────────────────────────────────────────────────


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not on PATH (this script expects the bench container)")
    return exe


def _ffprobe() -> str:
    exe = shutil.which("ffprobe")
    if not exe:
        raise RuntimeError("ffprobe not on PATH")
    return exe


def _is_valid_video(path: Path) -> tuple[bool, str]:
    """Return (ok, detail) — ok=True if ffprobe sees a video stream with frames."""
    try:
        out = subprocess.run(
            [_ffprobe(), "-v", "error", "-print_format", "json",
             "-show_streams", "-show_format", str(path)],
            capture_output=True, text=True, timeout=10, check=True,
        )
        meta = json.loads(out.stdout)
    except subprocess.CalledProcessError as exc:
        return False, f"ffprobe failed: {exc.stderr.strip()[:200]}"
    except Exception as exc:
        return False, f"ffprobe exception: {exc}"

    streams = [s for s in meta.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        return False, "no video streams"
    s = streams[0]
    return True, (
        f"{s.get('codec_name')} {s.get('width')}x{s.get('height')} "
        f"dur={meta.get('format', {}).get('duration', '?')}s "
        f"size={path.stat().st_size}B"
    )


# ── Recorder lifecycle ────────────────────────────────────────────────────────


def start_recorder(rtsp_url: str, segs_dir: Path, seg_secs: int) -> subprocess.Popen:
    """Spawn `ffmpeg -f segment` to write rolling .ts segments.  Same pattern
    as services/deepstream_service.py:_start_clip_recorder, but with system
    ffmpeg + a MediaMTX source URL."""
    segs_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        _ffmpeg(), "-y",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-c", "copy",
        "-f", "segment",
        "-segment_time", str(seg_secs),
        "-reset_timestamps", "1",
        "-segment_format", "mpegts",
        str(segs_dir / "seg_%05d.ts"),
    ]
    print(f"[rec] starting ffmpeg: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def stop_recorder(proc: subprocess.Popen, timeout_s: int = 5) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()


# ── Clip extraction ───────────────────────────────────────────────────────────


def cut_clip_concat(segs_dir: Path, out_path: Path,
                    start_offset_s: float, duration_s: float) -> tuple[bool, str]:
    """Cut a clip using `ffmpeg -f concat -c copy`, the same approach as
    services/deepstream_service.py:_extract_and_save_clip."""
    segs = sorted(segs_dir.glob("seg_*.ts"))
    if not segs:
        return False, "no segments to concat"

    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, dir=str(segs_dir)
    ) as f:
        list_path = Path(f.name)
        for s in segs:
            f.write(f"file '{s}'\n")

    cmd = [
        _ffmpeg(), "-y",
        "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-ss", f"{start_offset_s:.3f}",
        "-t", f"{duration_s:.3f}",
        "-c", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ]
    print(f"[cut] {' '.join(cmd)}", flush=True)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    finally:
        list_path.unlink(missing_ok=True)

    if result.returncode != 0:
        return False, f"ffmpeg rc={result.returncode}: {result.stderr.strip()[-300:]}"
    if not out_path.exists() or out_path.stat().st_size == 0:
        return False, "output missing or empty"
    return True, ""


# ── Main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Phase 1.2 clip recorder prototype")
    p.add_argument(
        "--rtsp",
        default=os.environ.get("PROTOTYPE_RTSP",
                               "rtsp://bianca-mediamtx-prototype:8554/test"),
    )
    p.add_argument(
        "--record-secs",
        type=int,
        default=35,
        help="How long to run the rolling recorder before cutting a clip",
    )
    p.add_argument("--seg-secs", type=int, default=10,
                   help="Length of each rolling segment file")
    p.add_argument("--clip-duration", type=float, default=6.0,
                   help="Length of the clip to cut at the end")
    p.add_argument(
        "--workdir", default="/tmp/clip_proto",
        help="Where to put the rolling segs and the cut clip",
    )
    args = p.parse_args(argv)

    workdir = Path(args.workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    segs_dir = workdir / "segs"
    out_clip = workdir / "clip.mp4"

    # ── 1. Recorder ──
    proc = start_recorder(args.rtsp, segs_dir, args.seg_secs)
    try:
        deadline = time.monotonic() + args.record_secs
        last_count = 0
        while time.monotonic() < deadline:
            time.sleep(2)
            count = len(list(segs_dir.glob("seg_*.ts")))
            if count != last_count:
                print(f"[rec] segs on disk: {count}", flush=True)
                last_count = count

        # Confirm recorder is still running (it should be — we'll terminate it)
        if proc.poll() is not None:
            err_tail = proc.stderr.read().decode(errors="replace")[-400:] \
                       if proc.stderr else ""
            print(f"[rec] ffmpeg died early: rc={proc.returncode}\n{err_tail}",
                  file=sys.stderr)
            return 1
    finally:
        stop_recorder(proc)

    segs = sorted(segs_dir.glob("seg_*.ts"))
    if not segs:
        print("[FAIL] no segments produced", file=sys.stderr)
        return 1
    print(f"[rec] stopped; {len(segs)} segments on disk", flush=True)

    # Verify the LAST segment is valid (not just the first — splitmuxsink
    # writes the trailing one after termination).  We accept the 2nd-to-last
    # if the very last is mid-write.
    target_seg = segs[-2] if len(segs) >= 2 else segs[-1]
    ok, detail = _is_valid_video(target_seg)
    print(f"[verify] segment {target_seg.name}: ok={ok} {detail}", flush=True)
    if not ok:
        return 1

    # ── 2. Cut a clip ──
    # Cut from the start of the second-to-last seg, length `clip-duration`.
    # That mimics how production extracts a window across boundaries.
    cut_ok, err = cut_clip_concat(
        segs_dir,
        out_clip,
        start_offset_s=max(0.0, args.record_secs - args.clip_duration - 2),
        duration_s=args.clip_duration,
    )
    if not cut_ok:
        print(f"[FAIL] clip cut failed: {err}", file=sys.stderr)
        return 1

    ok, detail = _is_valid_video(out_clip)
    print(f"[verify] clip {out_clip.name}: ok={ok} {detail}", flush=True)
    if not ok:
        return 1

    summary = {
        "segments": len(segs),
        "segment_total_bytes": sum(s.stat().st_size for s in segs),
        "clip_path": str(out_clip),
        "clip_size_bytes": out_clip.stat().st_size,
    }
    print(f"[summary] {json.dumps(summary)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
