# Architecture Review & Phased Improvement Plan
_Reviewer: Opus 4.7 · last revised 2026-04-25_
_Scope: the real-time surveillance pipeline (cameras, inference, clips, browser display). Voice/LLM/games stack is not in scope — it is small, stable, and well-factored._

---

## 1. Diagnosis

### 1.1 What works
- Voice, games, dashboard, reminders are small, testable, and independent. Keep that architecture as-is.
- Inference correctness is solid: YOLO-World TRT + ROI crop + `_SimpleTracker` + motion gate is the right shape (~8 ms per frame). The TRT re-export / `engine_queries` handshake is genuinely tricky and you got it right.
- Clip recording is well-factored: `imageio-ffmpeg` bundled binary, `FileResponse` with `Accept-Ranges`, pre/post ring buffer, `MIN_CLIP_DETECTIONS` gate.
- Hard bugs are documented well in WORKLOG.md (DTS discontinuities, LL-HLS `_HLS_msn`/`_HLS_part`, bind-mount EBUSY, `cv2 before Gst.init` SIGABRT). The WORKLOG is genuinely useful.

### 1.2 What's wrong
The surveillance pipeline has accidentally over-grown for one camera on one GPU:

- **9 containers (7 static + 2 dynamic per camera)** to display one feed and run one TRT engine.
- **5 network hops** between camera bytes and browser pixels: `Camera → go2rtc → bianca-rtsp-{cam_id} → ZMQ → savant-module → ZMQ → bianca-sink-{cam_id} → LL-HLS → deepstream proxy → app proxy → browser`.
- **3 hops** between camera and detector. Each link has been a bug at least once and several are re-entrant.
- **`services/deepstream_service.py` is 1,598 lines** with 13 module-level `dict[cam_id, …]` globals, 6 locks, 5 background threads, and a `_rebuild()` that restarts everything on any change.
- **Cross-process orchestration for query changes** (`main.py._run_reexport`) is a bespoke distributed transaction with ~90 s of user-visible LLM downtime and no recovery path if a step fails mid-way.
- **The frame and display paths are fused.** Savant stalls → both die. RTSP adapter timeout → both die. go2rtc stream drop → both die. Coupling without payoff.
- **The iteration pattern has been: add a layer to fix a symptom.** Across 19 WORKLOG sessions the video path went PyAV → GStreamer NVDEC → PyAV → Triton-isolation → GDINO Triton (bug) → FastAPI GDINO → YOLO-World TRT → `pipeline_worker.py` subprocess → binary stdout protocol → ROI tee → Savant framework → go2rtc re-added. Each fix was real; none removed a layer.

### 1.3 Why this happened
1. **"Production quality" was interpreted as "maximum hardware acceleration,"** which pulled in DeepStream, Savant, NVMM everywhere, `nvjpegenc`, dynamic per-camera containers. All of those have sharp edges that only matter at 10+ cameras.
2. **No end-to-end regression harness.** Each fix was verified against whatever was failing _right now_, not against the whole known-good behaviour set. That's the classic spaghetti-generator.

---

## 2. Confirmed Decisions

1. **Scope: decision record, not implementation commit.** Both Case A (≤4 cam) and Case B (10+ cam) are documented. Phase 0 (regression harness), Phase 2 (state consolidation), Phase 3 (observability), and Phase 4 (resilience) are shared and can start immediately. Phase 1 (ingress/decoder swap) is case-specific and waits on a case selection.
2. **Decoder for Case A: decide after Phase 0 on real numbers.** Phase 0's harness benchmarks both PyAV CUDA and GStreamer NVDEC side-by-side on the same `test.mp4`. Adds ~½ day to Phase 0.
3. **TRT re-export: keep the 90 s LLM pause.** No queuing, no scheduling. Phase 4's `test_reexport_triton_down` test (kill Triton mid-reexport, assert state returns to `ready` with Qwen reloaded) is the guardrail.

---

## 3. Design Principles (apply to both cases)

1. **Independent failure domains.** Display and inference must not share a process or buffer. Inference crash → video stays up. Display stall → clips keep recording.
2. **One authoritative RTSP hub.** Exactly one component opens the camera. Everything else pulls from the hub. All DTS/RTCP/audio-drop hacks live in one place.
3. **State lives in objects, not module globals.** One `CameraRuntime` per camera, one lock per camera. `deepstream_service.py` becomes a thin registry + router.
4. **Observability first.** `/diag/{cam_id}` reports last-frame/last-inference/last-event/last-clip timestamps + counters. Without this, debugging stays archaeology.
5. **Every failure has a named test.** Future regressions start with "add a test that reproduces it, make it pass."
6. **No silent hardware downgrades** (per CLAUDE.md). Any move off NVDEC/NVMM/nvjpegenc is called out and requires explicit approval.

---

## 4. Case A — ≤4 cameras (Recommended default)

### 4.1 Target stack

| Container | Role | Replaces |
|---|---|---|
| `mediamtx` | RTSP ingest + normalization, LL-HLS :888, WebRTC :8889, re-stream on :8554 | `go2rtc` + `savant-module` + `bianca-sink-{cam_id}` + `bianca-rtsp-{cam_id}` |
| `inference` | One Python process; `CameraRuntime` per camera; PyAV CUDA **or** GStreamer NVDEC (decided after Phase 0); Triton client; clip cutter | `savant-module` pyfunc + `deepstream_service` pipeline bits |
| `triton` | YOLO-World TRT + management sidecar | unchanged |
| `whisper`, `ollama`, `app` | unchanged | unchanged |

**5 static containers, 0 dynamic.** One RTSP connection per camera. Two consumers inside MediaMTX (browser + inference). Clips via ffmpeg stream-copy from MediaMTX's re-stream, same pattern as today.

### 4.2 Frame path

```
Camera ──RTSP──► mediamtx ──┬──► LL-HLS/WebRTC ──► browser
                            └──► RTSP re-stream ──► inference (decode → Triton → tracker → events)
                                                        │
                                                        └──► clip trigger → ffmpeg stream-copy from mediamtx → .mp4
```

### 4.3 Trade-offs to confirm before Phase 1
1. **Drop Savant entirely.** Savant is designed for batched multi-source `nvstreammux → nvinfer` pipelines with 10+ cameras. With 1–4 cameras the pyfunc → HTTP POST bridge is re-implementing what plain GStreamer does directly.
2. **go2rtc → MediaMTX.** MediaMTX's LL-HLS is the reference implementation. The `_HLS_msn/_HLS_part` blocking-request issue (hit 2026-04-21) disappears.
3. **Decoder choice deferred to Phase 0 benchmark.** PyAV CUDA (~30 lines) vs GStreamer NVDEC (~200 lines, the old `pipeline_worker.py` pattern). Decision on numbers, not vibes.

---

## 5. Case B — 10+ cameras (if you expect to scale)

### 5.1 Why the current direction is right but mis-applied
At 10+ cameras the batched GPU path becomes the dominant cost optimization: one `nvstreammux` batch → one `nvinfer` call per batch → one `nvtracker` call → per-source demux. This is what DeepStream was built for. The current stack _has_ Savant/DeepStream but isn't using it this way — it's using Savant as a frame-ripper that JSON-POSTs JPEGs one at a time to an external Triton over HTTP. That gives you:

- No batching → N× inference cost
- JPEG encode/decode round-trip per frame → extra latency and CPU
- HTTP overhead per frame → fine at 4 cameras × 10 fps, bad at 20 × 10

### 5.2 Target stack

| Container | Role | Replaces |
|---|---|---|
| `mediamtx` | RTSP ingest hub (N paths) + browser-facing LL-HLS/WebRTC | `go2rtc` + dynamic sink containers |
| `inference-pipeline` | One Savant module (correctly configured): `rtspsrc × N → nvstreammux batch=N → nvinfer (YOLO-World TRT in-pipeline) → nvtracker → pyfunc (events + clip triggers)`. No per-frame HTTP POST. | pyfunc HTTP POST + external Triton for the hot path |
| `triton-mgmt` | TRT re-export sidecar only | smaller, no inference server |
| `clip-cutter` | Reads `nvtracker` output + ffmpeg stream-copy from mediamtx | `_clip_manager` in deepstream_service |
| `whisper`, `ollama`, `app` | unchanged | unchanged |

**6 static containers, 0 dynamic.**

### 5.3 Frame path

```
Camera × N ──RTSP──► mediamtx ──┬──► LL-HLS/WebRTC ──► browser
                                └──► RTSP re-stream × N
                                         │
                                         ▼
   rtspsrc × N → nvstreammux(batch=N) → nvinfer(YOLO-World TRT) → nvtracker → pyfunc
                                                                                │
                                                                                ├─► events
                                                                                └─► clip triggers → ffmpeg stream-copy from mediamtx → .mp4
```

### 5.4 Numbers to validate before committing to Case B
- `nvinfer` TRT batch throughput at batch=4, 8, 16 with YOLO-World M at 720p. Measure — don't assume.
- VRAM at batch=16 may need 1–2 GB (vs 0.3 GB today). With Qwen ~10 GB + Whisper ~1.5 GB, headroom is ~4 GB — tight but workable.
- `nvtracker` config (NvDCF) needs explicit tuning.

### 5.5 Trade-offs to confirm before Phase 1
1. **Commit to Savant/DeepStream framework properly** — `nvstreammux + nvinfer` in-pipeline. Opposite of "drop Savant"; lean into it.
2. **Drop external-Triton inference hot path.** TRT engine lives inside `nvinfer`. External Triton becomes only the TRT re-export mechanism.
3. **Accept that Case B is more code than Case A.** The payoff only shows at 10+ cameras. Don't pick Case B "just in case."

---

## 6. Shared Phases (both cases)

Test-gated, incremental. Each phase has a single definition of done.

### Phase 0 — Regression harness (blocks everything else)
**Why:** without this, every subsequent phase is "iterate until the current symptom is gone" — the same loop that produced the current state.

**Build:**
- `tests/test_smoke.py` + `tests/conftest.py` + `tests/fixtures/`. Pytest + httpx, no internal mocking. Uses the existing `test.mp4`.
- One `docker compose up -d --wait` at session start; tear down at end.
- **Decoder benchmark harness:** `tests/bench_decoder.py` runs `test.mp4` through (a) PyAV CUDA and (b) GStreamer NVDEC, reports decode ms/frame p50/p99, CPU%, GPU watts. Lives outside the smoke suite (`pytest -m bench`) so it doesn't run on every CI loop.

**Checkpoints (all must pass in <2 min):**

| # | Test | Passing criterion |
|---|---|---|
| T0.1 | `GET /` returns 200 | smoke |
| T0.2 | `GET /cameras` returns 200 | smoke |
| T0.3 | Add `file://` camera via `POST /cameras/streams`, poll, assert present within 5 s | add works |
| T0.4 | `GET /cameras/hls/{cam_id}/index.m3u8` returns 200 with `EXT-X-TARGETDURATION` within 15 s | display path up |
| T0.5 | `GET /cameras/hls/{cam_id}/*.ts` or `*.m4s` returns non-zero body within 20 s | segments flowing |
| T0.6 | With `queries=["person"]`, within 45 s of T0.3, `GET /cameras/events?cam_id=X` returns ≥1 event | inference up |
| T0.7 | Within 60 s of T0.3, `GET /cameras/clips?cam_id=X` returns ≥1 clip; clip file GET returns 200 with `Accept-Ranges: bytes` | clips end-to-end |
| T0.8 | `DELETE /cameras/streams/{cam_id}`; within 5 s stream gone; no stray containers (`docker ps \| grep bianca-rtsp` empty) | clean teardown |
| T0.9 | Repeat T0.3 → T0.8 three times without restarting stack | no leaks / stale state |

**Done when:** harness is green against current HEAD. **No non-test code changes in this phase.**

### Phase 1 — Ingress/decoder swap (diverges per case)

**Case A (MediaMTX + small Python worker):**
- Add `mediamtx` to compose. Mount `mediamtx.yml` with paths generated from `cameras.json` at startup.
- Delete `go2rtc`, `savant-module`, dynamic `bianca-rtsp-*` / `bianca-sink-*` from compose and from `deepstream_service.py`. Keep deletions in one "deprecated" commit so revert is one git click.
- `deepstream_service` talks to MediaMTX HTTP control API (`/v3/paths/add`, `/v3/paths/remove`); no Docker SDK.
- Inference worker pulls from `rtsp://mediamtx:8554/{cam_id}`; rate-limits to `INFER_FPS`; calls Triton over HTTP.
- `/cameras/hls/*` proxy in `app` → `mediamtx:888` directly (drop the deepstream hop).

**Case B (MediaMTX + in-pipeline Savant):**
- Add `mediamtx` (same as Case A) for display + normalization.
- Rewrite `pipeline/module.yml` to `rtspsrc × N → nvstreammux batch=N → nvinfer (yoloworld.engine) → nvtracker → pyfunc (event emitter)`.
- Drop external-Triton inference path. Keep only `triton-mgmt` for TRT re-export.
- N sources wired statically from `cameras.json`; adding a camera rewrites module.yml and restarts savant-module. Still no dynamic containers.

**Checkpoints (both cases):**
- All T0 tests still pass, unchanged.
- **T1.1** — `docker restart bianca-mediamtx`; pipeline recovers within 20 s with no manual action.
- **T1.2** — `docker ps` shows exactly 5 (Case A) or 6 (Case B) containers. No `bianca-rtsp-*`, no `bianca-sink-*`.
- **T1.3** — display latency (HLS "now" vs ffprobe on the re-stream) ≤ 3 s.
- **Case B only — T1.4** — with N=4, batched `nvinfer` throughput ≥ N × Case A single-stream throughput. If not, Case B doesn't earn its complexity.

### Phase 2 — State consolidation (identical for both cases)
**Why:** 13 module-level dicts + 6 locks is the reason every refactor introduces new concurrency bugs.

**Build:**
- `class CameraRuntime` with: `cam_id, rtsp_url, roi, tracker, motion_prev, last_event_ts, clip_sessions, seg_ring, current_seg, detections, lock`. One lock per camera. Pick one concurrency model (threading _or_ asyncio).
- `_registry: dict[cam_id, CameraRuntime]` replaces every module-level `dict[cam_id, …]`.
- Background loops become methods or stateless functions taking a `CameraRuntime`.
- **Delete `_rebuild()`.** Add/remove/ROI-change touch one `CameraRuntime`; no global pipeline restart.
- Split the file:
  - `services/camera_runtime.py` — dataclass + per-camera lifecycle
  - `services/inference.py` — motion gate + Triton call + tracker + event fire
  - `services/clips.py` — segment watcher + clip cutter + index
  - `services/ingress.py` — MediaMTX integration
  - `services/deepstream_service.py` — ~100-line FastAPI router

**Checkpoints:**
- All T0 + T1 tests still pass. Refactor is invisible from outside.
- **T2.1** — `grep -E '^_[a-z_]+: *(dict|list|deque)' services/*.py` returns nothing.
- **T2.2** — concurrency stress: add/remove 10 cameras at 100 ms intervals; assert no orphan threads, no orphan ffmpeg processes, registry empties.

**Done when:** each new file <400 lines; T2.1/T2.2 green.

### Phase 3 — Observability
**Build:**
- `GET /diag/{cam_id}` JSON: `{ingress: {last_frame_ts, bytes_total, reconnect_count}, inference: {last_triton_ts, last_event_ts, triton_ms_p50, triton_ms_p99, motion_gate_skip_pct}, clips: {active_sessions, last_clip_ts, clips_total}, health: "ok"|"degraded"|"down", reasons: [...]}`.
- Structured logs: every line tagged `cam_id=X subsystem=Y`.
- `GET /health` returns 200 `"ok"` or 503 `"degraded: <reason>"`. Wire into docker healthcheck.
- Optional: `/metrics` in Prometheus format.

**Checkpoints:**
- **T3.1** — 1 camera × 60 s: `/diag` shows `last_frame_ts` within 2 s, `last_triton_ts` within 2 s, `motion_gate_skip_pct > 50` on a static file.
- **T3.2** — block MediaMTX for 5 s via network namespace; `/diag` shows `reconnect_count > 0`, recovery within 15 s, `/health` reports degraded then ok.

**Done when:** which subsystem is sick is visible from `/diag` without reading logs or code.

### Phase 4 — Resilience hardening
**Why:** stops the spaghetti loop. Each fix = one named test + the code to make it pass.

| Test | What it breaks | Expected behavior |
|---|---|---|
| `test_camera_unplugged` | Kill camera (or MediaMTX path) for 30 s, restore | Reconnect within 15 s; `/diag` reflects outage |
| `test_triton_restart_midstream` | `docker restart bianca-triton` mid-inference | Inference pauses gracefully; resumes within 15 s; no worker thread dies |
| `test_clip_dir_full` | Fill `clips/` to 99 % | Inference continues; clip writes fail cleanly with logs; oldest clips pruned |
| `test_reexport_triton_down` | Kill Triton mid-`_run_reexport` | State machine recovers to `ready` with Qwen reloaded; never stays `updating` forever |
| `test_roi_change_no_restart` | Change ROI 10× in 10 s | No pipeline rebuild, no frame drops, no orphan threads |

**Done when:** all five pass. Each test is a permanent regression test; CI blocks merges that break them.

### Phase 5 (optional) — Sub-second display via WebRTC
**When:** only if post-Phase-1 LL-HLS latency (~2–3 s) isn't tight enough.

**Build:** enable MediaMTX WebRTC; swap `cameras.html` from `hls.js` to WebRTC WHEP; fall back to HLS on connect failure.

**Checkpoints:**
- **T5.1** — end-to-end latency ≤ 500 ms via flashing-pattern test + `requestVideoFrameCallback`.
- **T5.2** — fallback to HLS works when port 8889 is firewalled (corporate-NAT case).

---

## 7. Bugs to Kill During the Phases
Fold into the matching phase above; not worth their own phase.

1. **Double HLS proxy** (`app` → `deepstream` → `mediamtx`). Post-Phase 1: `app` proxies directly to `mediamtx`.
2. **`cameras.json` backward-compat branch** (`_load_cameras` handles string OR dict values). Delete the string branch.
3. **`/cameras/ws/{cam_id}` and `_ds_ws_proxy`** — surviving MJPEG-over-WS artefact. With LL-HLS display, unused.
4. **Duplicate `SEG_DURATION_S`** in `services/deepstream_service.py` (lines 69 and 77).
5. **`_reexport_state` + `_reexport_running` globals** in `main.py` — race between check and set. Wrap in `class ReExportOrchestrator` with one lock.
6. **Inference rate limiting split** between pyfunc (`INFER_FPS`) and deepstream (`_last_infer`). Post-Phase 1: one place.

---

## 8. Decision Matrix

| Question | Case A (≤4 cam) | Case B (10+ cam) |
|---|---|---|
| Savant framework | Drop | Keep (properly) |
| go2rtc | Drop (→ MediaMTX) | Drop (→ MediaMTX) |
| External Triton (inference hot path) | Keep (HTTP is fine at this scale) | Drop (use in-pipeline `nvinfer`) |
| TRT re-export sidecar | Keep | Keep |
| Decoder in inference worker | PyAV CUDA or GStreamer NVDEC (Phase 0 benchmark) | `nvstreammux` batched, in DeepStream |
| Dynamic per-camera containers | Drop | Drop |
| Static container count | 5 | 6 |
| Implementation effort | ~9 days | ~14 days |

**If unsure, start Case A.** It's the floor — every decision in Case A is also valid in Case B. If you later outgrow Case A, only Phase 1 is redone for Case B; Phases 0/2/3/4 carry over unchanged.

---

## 9. Critical Files That Will Change

| File | Case A | Case B |
|---|---|---|
| `docker-compose.yml` | +mediamtx; −go2rtc, savant-module | +mediamtx; −go2rtc, savant-module; refactor savant-module into real inference pipeline |
| `services/deepstream_service.py` | Split into 5 files; delete Docker SDK adapter mgmt; delete `_rebuild` | Same; also drop HTTP inference path |
| `pipeline/module.yml` | Delete | Rewrite as real nvinfer pipeline |
| `pipeline/pyfunc.py` | Delete | Rewrite — emit events, not HTTP-POST JPEGs |
| `main.py` | Simplify `_run_reexport`; drop WS proxy; direct HLS proxy | Same |
| `templates/cameras.html` | Drop WS code; HLS → MediaMTX | Same; optionally WebRTC in Phase 5 |
| `cameras.json` | Keep | Keep |
| `go2rtc.yaml` | Delete | Delete |
| `Dockerfile.savant` | Delete | Keep; update for in-pipeline inference |
| `tests/` | New directory — Phase 0 builds it out | Same |

---

## 10. Existing Code to Reuse (do not reinvent)

These already work and should survive the refactor verbatim or with minimal edits:

- `services/deepstream_service.py:92` — `_SimpleTracker` IoU tracker. Keep for Case A. (Case B replaces with `nvtracker`.)
- `services/deepstream_service.py:657` onward — `_trigger_clip_on_detection`, `_extract_and_save_clip`, `_register_clip`, clip ring buffer. Move to `services/clips.py`, unchanged logic.
- `services/deepstream_service.py` motion gate (around `_inference_loop`) — `cv2.absdiff` on 320×180 gray frame. Keep as-is in Case A; disable in Case B (nvinfer doesn't skip frames cleanly and batched inference is cheap enough).
- `main.py:171` — `_triton_reload()`. Keep for both cases.
- `triton_models/management.py` (TRT re-export sidecar). Keep unchanged in both cases.
- `main.py:184` — `_run_reexport` orchestration. Wrap in a class in Phase 2; logic unchanged.
- `scripts/test_service.py` and `scripts/test_service_quality.py` — 71 existing checks. Phase 0 subsumes these; migrate what applies, delete what's stale.

---

## 11. Verification

After each phase:

```bash
# Full test sweep (Phase 0 harness plus any new checkpoints)
docker compose down -v && docker compose up -d --wait
pytest tests/ -v

# Static checks (Phase 2 onwards)
ruff check services/ main.py
grep -rE '^_[a-z_]+: *(dict|list|deque)' services/ || echo "no module-level state"

# Observability sanity (Phase 3 onwards)
curl -s localhost:8000/diag/balcony | jq .
curl -s localhost:8000/health

# Manual — browser
# http://localhost:8000/cameras  — add balcony camera, watch video + detections + clips
```

End-of-Phase-4 success criterion: **someone who has never read this repo's WORKLOG can debug a pipeline outage using only `/diag/{cam_id}`, `/health`, and the named tests in `tests/`.**
