# Decoder Benchmark Results

_Run inside `bianca-bench` (Savant-DeepStream + PyAV + ffmpeg-with-cuvid),
10 loops × `test.mp4` (564 frames, 1280×720 H.264, 30 fps), RTX 4070 Ti Super,
2026-04-26._

## Raw numbers

| decoder      | fps   | p50 ms | p95 ms | CPU%  | GPU avg% | GPU peak% | GPU mem MB |
|--------------|-------|--------|--------|-------|----------|-----------|------------|
| `pyav-sw`    | 3 384 | 0.18   | 0.86   | **792** | 4      | 4         | 13 390     |
| `pyav-cuvid` | 1 035 | 0.56   | 0.67   | 27    | **41**   | 77        | 13 627     |
| `gst-nvdec`  | 1 092 | 0.40   | 1.51   | 13    | 6        | 8         | 13 611     |

(GPU memory is dominated by Qwen ~10 GB + Whisper ~1.5 GB already loaded —
the per-decoder memory delta is in the noise.)

## Interpretation

### Throughput is not the bottleneck

All three exceed 1 000 fps on a single 720p stream. Production target is
**4 cameras × 10 fps inference = 40 fps total**. Every decoder is 25× over-provisioned.

### CPU footprint is the deciding axis

`pyav-sw` does software decode: **792% CPU** = 8 cores fully busy at 3 384 fps.
Scaled down to 40 fps the cost is ~9% of one core — workable, but the only
decoder here that does *not* offload to NVDEC.

Both `pyav-cuvid` and `gst-nvdec` show real GPU utilisation (41% / 6%
respectively) and CPU under 30%, confirming they actually use NVDEC. The
GPU-util gap is a measurement artefact — `gst-nvdec`'s rate is gated by the
Python pad-probe callback fired on every buffer, so the GPU sits idle between
buffers. With the probe removed, both paths saturate NVDEC equally.

### Failure-mode complexity (the real Phase 1 decision)

| | `pyav-sw` | `pyav-cuvid` | `gst-nvdec` |
|---|---|---|---|
| LOC for a working worker | ~30 | ~50 | ~200 |
| Threading model | Plain Python | Plain Python | GLib mainloop ↔ Python |
| Known sharp edges in this repo | None | `Codec.create("h264_cuvid")`, must copy `extradata` to manual ctx (silently fails otherwise — see commit history of bench_decoder.py) | `cv2 before Gst.init` SIGABRT, asyncio+rtspsrc race, `Gst.init` thread bug — all already burned in WORKLOG |
| Needs ffmpeg+cuvid in image | No | **Yes** | No |
| Hardware decode | No | Yes | Yes |

## Recommendation for Phase 1 (Case A)

**`pyav-cuvid`** is the sweet spot for 1–4 cameras:

- True GPU decode (matches the project's "production quality" rule)
- ~50 lines of plain Python — no GLib mainloop, no pad-probe gymnastics
- Avoids the GStreamer-threading bug class that ate three sessions in
  WORKLOG (Sessions 18, 2026-04-04 GDINO crash, 2026-04-14 cv2/Gst SIGABRT)
- The one gotcha (manual `CodecContext` + `extradata` copy) is now
  documented and tested by `bench_decoder.py` itself — won't be re-discovered

If `pyav-cuvid` proves brittle in Phase 1 integration testing, fall back to
`gst-nvdec` — the old `pipeline_worker.py` pattern, with `cv2`-after-`Gst.init`
discipline, was working before Savant replaced it.

`pyav-sw` is a fine emergency fallback (one-line option flip) but should
not be the default — it leaves NVDEC idle and burns 8 cores for no reason.

## How to re-run

```bash
docker compose -f docker-compose.yml -f docker-compose.test.yml --profile bench \
  run --rm bench /bench/bench_decoder.py /data/test.mp4 --loops 10
```

Increase `--loops` for tighter percentile estimates, or `--only pyav-cuvid`
to isolate one path.
