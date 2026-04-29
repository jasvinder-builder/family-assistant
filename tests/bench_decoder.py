"""
Phase 0 — decoder benchmark.

Compares two decoder paths a Case-A inference worker could use:

  1. PyAV with CUDA hwaccel  — ~30 lines of code, single Python module
  2. GStreamer + nvv4l2decoder — ~150 lines of GStreamer boilerplate

For each path we report (against the same input, same loop count):
  - frames decoded
  - wall-clock duration
  - decode rate (fps)
  - per-frame latency p50 / p95
  - process CPU% (averaged)
  - process GPU memory (peak MB)
  - GPU utilisation (peak %, averaged %)

Run inside the bench container (see docker-compose.test.yml):
  docker compose -f docker-compose.yml -f docker-compose.test.yml \\
      run --rm bench python3 /bench/bench_decoder.py /bench/test.mp4

The output is intentionally compact and deterministic so we can capture
it in WORKLOG.md as the input to the Phase 1 decoder pick.

This is NOT a pytest test — it's a long-running benchmark.  pytest
collects it only via `pytest -m bench tests/bench_decoder.py` and a
matching @pytest.mark.bench at module level (kept minimal so the script
remains directly invocable).
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field

# psutil + pynvml are runtime dependencies of this script only — they live
# in Dockerfile.bench, not in any production image.
import psutil  # type: ignore[import-not-found]

try:
    import pynvml  # type: ignore[import-not-found]

    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False


# ── Shared sampler ───────────────────────────────────────────────────────────


@dataclass
class _Sampler:
    """Background thread sampling CPU% and GPU stats every `interval` seconds.
    Use as a context manager around the section being measured."""

    interval: float = 0.25
    cpu_samples: list[float] = field(default_factory=list)
    gpu_util_samples: list[int] = field(default_factory=list)
    gpu_mem_peak_mb: int = 0
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def __enter__(self) -> "_Sampler":
        proc = psutil.Process()
        proc.cpu_percent(None)  # prime: first call always returns 0.0

        def _loop() -> None:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0) if _NVML_OK else None
            while not self._stop.is_set():
                self.cpu_samples.append(proc.cpu_percent(None))
                if handle is not None:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.gpu_util_samples.append(util.gpu)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self.gpu_mem_peak_mb = max(
                        self.gpu_mem_peak_mb, info.used // (1024 * 1024)
                    )
                time.sleep(self.interval)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)


# ── PyAV CUDA decoder ────────────────────────────────────────────────────────


def bench_pyav(path: str, loops: int, mode: str) -> tuple[int, float, list[float]]:
    """Decode `path` `loops` times via PyAV.

    mode='sw'    — pure libavcodec software decode (multithreaded).
    mode='cuvid' — NVDEC via the h264_cuvid decoder (true GPU offload).
    Returns (total_frames, wall_seconds, per_frame_ms).
    """
    import av  # noqa: PLC0415 (deliberately lazy)

    per_frame_ms: list[float] = []
    total_frames = 0
    t_start = time.perf_counter()

    for _ in range(loops):
        container = av.open(path)
        stream = container.streams.video[0]
        try:
            if mode == "cuvid":
                # Bypass PyAV's auto-selected decoder and demux into a manually
                # created h264_cuvid context.  This is the canonical pattern
                # for hwaccel decoders that PyAV won't pick automatically.
                ctx = av.codec.CodecContext.create("h264_cuvid", "r")
                if stream.codec_context.extradata:
                    ctx.extradata = stream.codec_context.extradata
                last_t = time.perf_counter()
                for packet in container.demux(stream):
                    for _frame in ctx.decode(packet):
                        now = time.perf_counter()
                        per_frame_ms.append((now - last_t) * 1000.0)
                        last_t = now
                        total_frames += 1
            else:
                stream.thread_type = "AUTO"
                last_t = time.perf_counter()
                for _frame in container.decode(stream):
                    now = time.perf_counter()
                    per_frame_ms.append((now - last_t) * 1000.0)
                    last_t = now
                    total_frames += 1
        finally:
            container.close()

    return total_frames, time.perf_counter() - t_start, per_frame_ms


# ── GStreamer NVDEC decoder ──────────────────────────────────────────────────


def bench_gst(path: str, loops: int) -> tuple[int, float, list[float]]:
    """Decode `path` `loops` times via filesrc → qtdemux → h264parse →
    nvv4l2decoder → fakesink.  Counts buffers via a pad probe.
    Returns (total_frames, wall_seconds, per_frame_ms)."""
    import gi  # noqa: PLC0415

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst, GLib  # type: ignore[attr-defined]

    Gst.init(None)

    per_frame_ms: list[float] = []
    total_frames = 0

    t_start_total = time.perf_counter()

    for _ in range(loops):
        pipeline = Gst.parse_launch(
            f"filesrc location={path} ! qtdemux ! h264parse ! "
            f"nvv4l2decoder ! fakesink sync=false async=false name=sink"
        )

        loop_state = {"last_t": 0.0, "frames": 0}
        sink = pipeline.get_by_name("sink")

        def _probe(_pad, _info):
            now = time.perf_counter()
            if loop_state["last_t"] != 0.0:
                per_frame_ms.append((now - loop_state["last_t"]) * 1000.0)
            loop_state["last_t"] = now
            loop_state["frames"] += 1
            return Gst.PadProbeReturn.OK

        sink_pad = sink.get_static_pad("sink")
        sink_pad.add_probe(Gst.PadProbeType.BUFFER, _probe)

        loop = GLib.MainLoop()

        def _on_message(_bus, message):
            t = message.type
            if t == Gst.MessageType.EOS:
                loop.quit()
            elif t == Gst.MessageType.ERROR:
                err, _dbg = message.parse_error()
                print(f"[GST ERROR] {err}", file=sys.stderr)
                loop.quit()

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", _on_message)

        pipeline.set_state(Gst.State.PLAYING)
        try:
            loop.run()
        finally:
            pipeline.set_state(Gst.State.NULL)
            bus.remove_signal_watch()
        total_frames += loop_state["frames"]

    return total_frames, time.perf_counter() - t_start_total, per_frame_ms


# ── Reporting ────────────────────────────────────────────────────────────────


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


def _report(name: str, frames: int, secs: float, per_ms: list[float], smp: _Sampler) -> dict:
    fps = frames / secs if secs > 0 else 0.0
    return {
        "decoder": name,
        "frames": frames,
        "seconds": round(secs, 2),
        "fps": round(fps, 1),
        "p50_ms": round(_percentile(per_ms, 50), 2),
        "p95_ms": round(_percentile(per_ms, 95), 2),
        "cpu_avg_pct": round(
            statistics.mean(smp.cpu_samples) if smp.cpu_samples else 0.0, 1
        ),
        "gpu_util_avg_pct": round(
            statistics.mean(smp.gpu_util_samples) if smp.gpu_util_samples else 0.0, 1
        ),
        "gpu_util_peak_pct": max(smp.gpu_util_samples) if smp.gpu_util_samples else 0,
        "gpu_mem_peak_mb": smp.gpu_mem_peak_mb,
    }


def _print_table(rows: list[dict]) -> None:
    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("  ".join("─" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Phase 0 decoder benchmark")
    p.add_argument("video", help="Path to a local .mp4 to decode")
    p.add_argument(
        "--loops",
        type=int,
        default=10,
        help="How many times to decode the file end-to-end (default: 10)",
    )
    p.add_argument(
        "--only",
        choices=["pyav-sw", "pyav-cuvid", "gst", "all"],
        default="all",
        help="Run only one decoder (useful for isolating failures)",
    )
    args = p.parse_args(argv)

    if not os.path.isfile(args.video):
        print(f"error: not a file: {args.video}", file=sys.stderr)
        return 2

    rows: list[dict] = []

    if args.only in ("pyav-sw", "all"):
        print(f"[bench] PyAV software × {args.loops} loops on {args.video} …", flush=True)
        try:
            with _Sampler() as smp:
                f, s, per_ms = bench_pyav(args.video, args.loops, mode="sw")
            rows.append(_report("pyav-sw", f, s, per_ms, smp))
        except Exception as exc:
            print(f"[bench] PyAV-sw failed: {exc}", file=sys.stderr)

    if args.only in ("pyav-cuvid", "all"):
        print(f"[bench] PyAV h264_cuvid × {args.loops} loops on {args.video} …", flush=True)
        try:
            with _Sampler() as smp:
                f, s, per_ms = bench_pyav(args.video, args.loops, mode="cuvid")
            rows.append(_report("pyav-cuvid", f, s, per_ms, smp))
        except Exception as exc:
            print(f"[bench] PyAV-cuvid failed: {exc}", file=sys.stderr)

    if args.only in ("gst", "all"):
        print(
            f"[bench] GStreamer nvv4l2decoder × {args.loops} loops on {args.video} …",
            flush=True,
        )
        try:
            with _Sampler() as smp:
                f, s, per_ms = bench_gst(args.video, args.loops)
            rows.append(_report("gst-nvdec", f, s, per_ms, smp))
        except Exception as exc:
            print(f"[bench] GStreamer failed: {exc}", file=sys.stderr)

    if not rows:
        return 1

    print()
    _print_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
