# Family Assistant — Worklog

## Progress Log

### 2026-03-29 — Session 1
- Defined project scope: todos, events, research, WhatsApp only, single markdown file
- Completed full system design (folder structure, flows, schema, decisions)
- Created WORKLOG.md

### 2026-03-29 — Session 2
- Scaffolded full project: all folders, `__init__.py`, `requirements.txt`
- Implemented all services: `qwen.py`, `markdown_service.py`, `tavily_service.py`, `twilio_service.py`
- Implemented all handlers: `call_handler.py`, `intent_handler.py`, `todo_handler.py`, `event_handler.py`, `research_handler.py`, `response_handler.py`
- Wrote all 4 prompts in `prompts/`
- Wired everything in `main.py`
- Created `.venv` with `python3 -m venv .venv`, confirmed all imports clean
- Ready for: fill `.env`, run with ngrok, test end-to-end

### 2026-03-31 — Session 3
- Replaced tap-to-record + manual `AnalyserNode` silence detection with `@ricky0123/vad-web` (Silero VAD via ONNX Runtime Web)
- VAD runs entirely on the client device (phone/tablet CPU via WASM) — no audio sent to server until speech detected
- Removed mic button from `talk.html` and `hangman.html`; voice is always listening after page load
- Added animated ring indicator showing loading / listening / speech-detected / processing states
- VAD paused during TTS playback to prevent Bianca's voice feeding back into Whisper
- Hangman: VAD stays active after game over so players can say "new game" without tapping
- Updated `/transcribe` endpoint in `main.py` to detect audio format from filename (`.wav` or `.webm`) instead of hardcoding `.webm`; VAD delivers Float32Array encoded to WAV client-side
- CDN versions pinned: `@ricky0123/vad-web@0.0.30`, `onnxruntime-web@1.22.0` (must match — 1.22.0 is the ORT version bundled inside vad-web 0.0.30)

### 2026-03-31 — Session 4
- Added **Times Tables** game (`/games/multiply`): random A×B questions (1–9), VAD captures spoken answer, spoken number parser handles digits and English words up to 81, score tracking, Fresh Start button
- Added **Tell the Time** game (`/games/clock`): 4-option multiple choice, 4 SVG analogue clock faces drawn in pure JS (no images/libraries), distractors guaranteed visually distinct, VAD captures A/B/C/D, tapping a clock also accepted, Bianca speaks human-friendly time ("half past 3"), score tracking
- Hangman: pre-reveal hint letters at game start — 0 hints for ≤4-letter words, +1 per letter above 4, capped at 3; `_hint_count()` added to `hangman_service.py`
- Home page: games section moved to top, assistant cards moved to bottom; hero header made compact (1.5rem, reduced padding); Games/Assistant section labels added
- All new games follow same VAD + speech-unlock pattern as Talk to Bianca and Hangman

### 2026-03-31 — Session 5
- Added **Knowledge Quiz** game (`/games/quiz`):
  - Setup screen: 8 subject tiles (Science, Geography, Music, Arts, Sports, Maths, Nature, Space) + grade buttons 1–8
  - "Let's Go!" tap unlocks speechSynthesis and triggers Qwen question generation
  - Loading screen with cycling messages while Qwen runs (~15–25s)
  - New prompt `prompts/quiz_generate.txt`: enforces kid-safe content, grade-appropriate difficulty, JSON-only output
  - New `qwen.generate_quiz(subject, grade)` with `_extract_json_array()` helper; validates each question, requires ≥5 valid questions
  - New `POST /games/quiz/generate` endpoint with grade validation (1–8)
  - Quiz screen: progress bar, 4 option cards (tap or say A/B/C/D via VAD), random funny correct/wrong phrases (8 variants each)
  - Final score screen with performance-scaled message; Play Again (reshuffled) or New Quiz
  - Fixed missing `import qwen` in main.py (was only importing `_chat`)
- Home page: Knowledge Quiz card added to Games section

### 2026-03-31 — Session 6
- Improved voice answer detection across all three kids' games:
  - `minSpeechMs` raised to 600ms in quiz.html, clock.html, multiply.html — prevents accidental triggering on brief noises or background sounds
  - Whisper confidence check added (threshold 0.45): low-confidence transcripts silently discarded; VAD resumes listening instead of acting on noise
  - **quiz.html** — replaced `parseOption()` with `parseAnswer(transcript, options)`:
    1. Option text matching first: if any option text appears in the transcript, match it directly (e.g. "I think Paris" → selects Paris option)
    2. Filler stripping: removes "I think", "the answer is", "I choose", "I pick", "letter", "option", etc.
    3. Spoken letter names: ay→A, bee→B, see/cee→C, dee→D
    4. Conservative "A" guard: bare "a" only accepted when minimal other content (avoids false matches on sentences containing "a")
  - **clock.html** — `parseOption()` updated with same filler stripping and conservative "A" guard
  - **multiply.html** — confidence check added; `parseSpokenNumber()` already robust

### 2026-04-01 — Session 7
- **Root cause fix: VAD capturing TTS audio in games** (diagnosed from logs)
  - quiz.html: `nextQuestion()` was calling `micVAD.start()` immediately after `showQuestion()` (which calls `speak()` → pauses VAD), re-enabling VAD while TTS was still playing. Whisper was then transcribing the full 10–16s question as user speech. Fixed by removing `micVAD.start()` from `nextQuestion()` — `speak()`'s `onend` callback already calls `resumeListening()`.
  - clock.html: `initVAD()` is async but wasn't awaited in `startGame()`, so the first question's TTS started before VAD was initialised; VAD came up mid-TTS. Fixed by `await initVAD()`.
  - All three games: added audio duration guard in `onSpeechEnd` — reject clips > 5s (quiz), > 4s (clock), > 3s (multiply). TTS echo is always long; real answers are always short.
- **Qwen fallback for quiz answer matching** — new `POST /games/resolve-answer` endpoint; if client-side `parseAnswer()` returns null, Qwen resolves the transcript against the 4 options. Handles natural language like "I think it's the blue whale". Fast path stays local; Qwen only activates when needed.
- **Fixed multiply game parsing wrong answer** — `parseSpokenNumber` was taking the first digit match. "It is 8 times 9. 72." → returned 8 instead of 72 (Whisper hallucinated the question into the transcript). Fixed to take the last digit/word match.
- **Quiz generation failures diagnosed and fixed:**
  - Qwen returning `"correct": "2"` (string) or `"correct": "B"` (letter) instead of integer — strict `isinstance(correct, int)` check dropped every question → "Too few valid questions: 0". Fixed with coercion for string digits and letter answers.
  - Qwen omitting surrounding `[ ]` brackets — outputting bare comma-separated objects. Fixed `_extract_json_array()` with a fallback that wraps first-`{` to last-`}` in brackets. Also tightened pattern 1 to require at least one dict (prevents matching option sub-arrays like `["A","B","C","D"]`).
  - Added logging of raw Qwen output when validation fails, making future failures diagnosable without guessing.
  - Rewrote `quiz_generate.txt` prompt with a concrete 2-question array example and explicit `"correct" must be an integer, never a string or letter"` rule — addresses both failure modes at source.
  - `format:"json"` Ollama parameter tried and reverted — caused Qwen to return only 1 question instead of 10.
- **Replaced Maths with US Geography and World Geography** — Maths generates ambiguous questions (e.g. both 29 and 31 are prime in the same question). Geography facts are unambiguous, single-answer, and Qwen handles them reliably.
- **Confidence threshold lowered to 0.30 for clock and quiz** — single-letter answers (A/B/C/D) consistently score 0.33–0.44 confidence; the 0.45 threshold was silently rejecting real answers. Duration guard handles TTS echo; 0.30 still blocks pure silence/noise.
- **Key learning — VAD + speechSynthesis interaction:** always `await initVAD()` before first TTS call; never call `micVAD.start()` after a `speak()` call — `speak()` owns the resume cycle via its `onend` callback.
- **Key learning — Whisper confidence on short answers:** single letters score 0.33–0.44 by design; a high confidence threshold blocks legitimate short answers. Duration guard is a better primary filter than confidence for these games.
- **Key learning — Qwen JSON output:** never rely on `format:"json"` for array output from Qwen (breaks generation length). Use a concrete example in the prompt + code-level coercion as defence-in-depth.

### 2026-04-01 — Session 8
- **Root cause: `speechSynthesis.onend` unreliable on mobile browsers** — when TTS finishes but `onend` silently never fires, `resumeListening()` is never called, VAD stays paused, ring shows "ready" but nothing is actually listening. This was the main cause of "stuck in listening state" in clock and quiz games.
- **Fix: TTS watchdog timer** added to `speak()` in clock.html and quiz.html — estimates TTS duration (`text.length / 15` seconds) + 3s buffer, then force-calls the resume logic if `onend` hasn't fired. Clears itself if `onend` fires normally.
- **Fix: "Bianca is speaking" ring state** — light blue ring + "Bianca is speaking..." status text while TTS plays, so user knows not to speak yet. Previously ring showed "ready" during TTS which was misleading.
- **Fix: show heard transcript** — after Whisper transcribes, show `Heard: "..."` below the ring so user can see exactly what was detected. Cleared when next listening cycle begins. Gives users immediate feedback to self-correct ("I heard: see you later" → user understands why and tries "letter C").
- **Key learning — production voice robustness:** production apps don't have smarter VAD — they treat every async operation as potentially failing silently. Rule: *if a state can be entered, a timer or fallback must be able to exit it*, not just an event that might never fire. Apply this to any future async state transitions.
- **Key learning — `speechSynthesis.onend` on mobile:** cannot be relied on. Always pair with a watchdog timer. Estimate duration as `(text.length / 15) * 1000 + 3000ms`.

### 2026-04-02 — Session 9
- **Fixed `minSpeechMs` in clock.html and quiz.html** — lowered from 600ms to 250ms. Single-letter answers ("D") are ~150ms of actual speech; 600ms threshold silently discarded them before VAD ever fired. Duration guards (>4s clock, >5s quiz) already handle TTS echo, so minSpeechMs does not need to be high.
- **Fixed Hangman game broken state** — three issues combined to make the game freeze or mis-behave:
  1. `speak()` had no watchdog timer — if `speechSynthesis.onend` never fired (unreliable on mobile), `resumeListening()` was never called and the game stayed in processing state permanently. Fixed with same watchdog pattern as clock/quiz: `Math.ceil(text.length/15)*1000 + 3000ms`.
  2. No duration guard in `onSpeechEnd` — TTS audio echo (always >4s) was being sent to Whisper as guesses. Added `if (audio.length / 16000 > 4) return;`.
  3. No confidence check — low-quality transcripts from noise were submitted as guesses. Added `(data.confidence ?? 1) >= 0.30` gate.
- All three games (clock, quiz, hangman) now consistent: minSpeechMs 250ms, duration guard, confidence 0.30, watchdog timer in speak().

### 2026-04-02 — Session 10
- **Restructured home page layout** — replaced two-section (Games / Assistant) layout with 4 large navigation cards: Games, Family Dashboard, Talk to Bianca, Cameras. Cleaner top-level navigation.
- **Added Games hub page** (`GET /games`) — `templates/games.html` collects all game cards (Hangman, Times Tables, Tell the Time, Knowledge Quiz) in one place. Home now links to `/games` rather than individual game routes.
- **Added Cameras page** (`GET /cameras`) — Stage 1: RTSP URL input + live MJPEG stream viewer + placeholder for future AI event detection.
  - `services/camera_service.py`: stores the active RTSP URL in memory; `mjpeg_generator()` reads frames via OpenCV in a background thread, encodes JPEG at quality 70, yields `multipart/x-mixed-replace` chunks to the async generator.
  - `POST /cameras/set-stream`: saves RTSP URL (must start with `rtsp://`); empty URL disconnects.
  - `GET /cameras/stream`: streams MJPEG via FastAPI `StreamingResponse`.
  - Browser displays stream in a plain `<img>` tag — no plugin needed.
  - Default test stream pre-filled: `rtsp://test.rtsp.stream/people`.
- **Added `opencv-python-headless`** to `requirements.txt` and installed in venv (4.13.0.92). Headless variant used — no X11/Qt dependencies needed on a server.
- **Updated ARCHITECTURE.md** — new routes, camera service, updated component map and browser interface flow.

### 2026-04-02 — Session 11
- **Implemented AI scene analysis pipeline** on the cameras page — YOLOv8 + ByteTrack + CLIP, all running locally on GPU.
  - `services/scene_service.py`: independent background thread opens the RTSP stream at 3 fps, runs YOLOv8n person detection + ByteTrack tracking, crops each person bounding box, computes CLIP cosine similarity against user-defined natural-language queries, and logs events when similarity ≥ 0.25. Deduplicates by (track_id, query_index) with a 30-second recheck window.
  - Models loaded lazily on first `start_analysis()` call — no impact on startup when cameras page is unused.
  - Analysis loop is separate from the MJPEG stream thread — live video unaffected by inference.
- **Scene query management** — users define arbitrary natural-language queries (e.g. "person in red clothing", "small child"). Queries are global (shared across cameras for now). Stored in memory.
  - `GET /cameras/queries` — list current queries
  - `POST /cameras/queries` — add query
  - `DELETE /cameras/queries/{index}` — remove query
- **Event log** — rolling 1-hour window, served via `GET /cameras/events`. Frontend polls every 5 seconds and renders events most-recent-first with timestamp, query text, and confidence %.
- **cameras.html updated** — query management UI (add/remove badges) + live event log replace the "under construction" placeholder.
- **Added packages:** `torch 2.6.0+cu124`, `torchvision 0.21.0+cu124`, `ultralytics` (YOLOv8 + ByteTrack), `transformers` (CLIP ViT-B/32), `Pillow`. All verified on CUDA.
- **VRAM budget (post-AI):** Qwen 14b ~10GB + Whisper ~1.5GB + YOLOv8n ~0.05GB + CLIP ViT-B/32 ~0.6GB ≈ 12.15GB — fits comfortably in 16GB.

### 2026-04-02 — Session 12
- **Fixed debug overlay bounding boxes not tracking with moving persons** — root cause: two independent `VideoCapture` instances (one in camera_service for MJPEG display, one in scene_service for AI analysis) reading the same file at different frame positions. Boxes were stamped from the analysis thread's position onto the display thread's frame → appeared frozen/misaligned.
  - **Fix: single-reader frame sharing.** Removed the separate VideoCapture from `scene_service`. Camera_service's `_reader()` now calls `scene_service.push_frame(frame)` on every decoded frame. Scene service stores the latest frame in `_shared_frame` (protected by a lock + `threading.Event`). Analysis loop waits for the event and samples from the shared frame at 3fps — bounding boxes now perfectly align with what's shown in the live view.
- **Fixed debug overlay not showing when no queries are defined** — analysis loop skipped YOLO entirely when `queries` was empty, so `_latest_detections` was never populated. Fixed by also checking the debug overlay flag: YOLO runs whenever `queries` OR `debug_overlay` is active.
  - Debug overlay flag is now mirrored into `scene_service` via `scene_service.set_debug_overlay()` when `POST /cameras/debug-overlay` is called.
- **Added CLIP match image thumbnails to event log** — when CLIP similarity ≥ threshold, the person crop is encoded as a base64 JPEG and stored on `CameraEvent.image_b64`. The `/cameras/events` API returns it; the event log renders a 56×80px thumbnail alongside each event row.
- **Added threshold and debug-overlay routes to ARCHITECTURE.md**

### 2026-04-02 — Session 4 (continued) — CLIP/YOLO detection quality improvements

Six improvements applied to `services/scene_service.py` to reduce false positives, missed detections, and noisy one-frame events:

**YOLO improvements:**
- Upgraded model from `yolov8n` (nano) to `yolov8s` (small) — better recall on partially occluded and distant objects at the cost of slightly more VRAM (~100MB vs ~50MB)
- Lowered detection confidence threshold from default 0.25 to 0.15 to catch more borderline detections
- Increased analysis FPS from 3 to 5 — more frames analysed per second reduces the chance of missing a fast-moving subject

**CLIP improvements:**
- Upgraded model from `clip-vit-base-patch32` to `clip-vit-large-patch14` — 3× more parameters, sharper, more discriminative embeddings (~1.7GB vs ~600MB VRAM)
- Added prompt template ensembling (from the original CLIP paper): each query is expanded into 4 templates (`"{}"`, `"a photo of {}"`, `"a picture of {}"`, `"an image of {}"`) whose embeddings are averaged and re-normalized; produces more robust text representations than a bare label
- Added module-level `_text_emb_cache` (CPU tensors keyed by query string) — text embeddings are static and computed once per query, eliminating redundant GPU work on every frame
- Added multi-frame voting: a `score_buffer` deque accumulates the last 3 per-frame similarity scores per `(track_id, query_idx)` pair; an event only fires when the rolling mean meets the threshold — eliminates false positives from single-frame noise
- Added minimum crop size guard: CLIP is skipped for bounding boxes with crop area < 3000 px² (upscaling a 40×60 px crop to 224×224 produces blurry, unreliable embeddings)

### 2026-04-02 — Session 16
- **Expanded YOLO detection to vehicles and animals** — previously only class 0 (person). Now detects: bicycle (1), car (2), motorcycle (3), bus (5), truck (7), bird (14), cat (15), dog (16). All feed into CLIP matching and the event log. Debug overlay now shows the class label (e.g. `car #3`) alongside track ID.
- **Added configurable crop padding** (`_crop_padding`, default 0.3) — before CLIP inference the YOLO bounding box is expanded by `pad_factor × box_size` on each side (clamped to frame boundaries). Gives CLIP more surrounding context for relational queries like "person with stroller" or "dog near car".
  - New `GET/POST /cameras/pad` endpoints to read/set the factor at runtime.
  - Pad factor slider (0.0–1.0, step 0.05) added to Scene Queries card in cameras.html.
- **Detection dataclass extended** — added `label: str` field to `Detection`; populated from `_CLASS_LABELS` map keyed by YOLO class ID.

### 2026-04-03 — Session 5 — GStreamer NVDEC hardware decode (Phase 1)

**What was built:**
- Replaced PyAV software decode in `camera_service.py` with a GStreamer NVDEC pipeline (`nvv4l2decoder`) as the primary backend
- PyAV retained as an automatic fallback when GStreamer or the nvv4l2decoder/nvcodec plugin is not available
- Backend is probed once at startup via `Gst.Registry` plugin lookup and cached; no runtime switching overhead

**Key technical decisions:**
- Used `appsink emit-signals=false` + `pull-sample` (blocking) rather than the `new-sample` signal + GLib main loop — avoids GLib thread complexity in a Python threading context; one less moving part
- `nvvidconv` outputs `video/x-raw,format=BGRx` — 4 bytes/pixel, trivially sliced to BGR with `arr[:, :, :3]` (no colour conversion needed)
- For local files (non-RTSP), used `decodebin` so GStreamer auto-selects nvv4l2decoder when available; handles mp4/mkv/etc. without format-specific parsing
- Display cap kept at 25fps (same as before) — appsink is configured `max-buffers=2 drop=true` so decode never backs up regardless of downstream speed

**Benefits vs. PyAV:**
- Zero CPU H.264/H.265 decode — NVDEC handles it entirely
- Frees ~1–2 CPU cores previously spent on software decode
- Frames arrive already in system RAM from NVDEC DMA; next step (Phase 3/DALI) will keep them on GPU entirely

### 2026-04-03 — Session 6 — Game improvements + GDINO ONNX investigation

**What was built:**

*Game improvements:*
- Bulls & Cows: difficulty selector (2/3/4 digits, Grades 1–3/4–6/7+), always 10 guesses
- Word Ladder: escalating hints — 1st hint names the position, 2nd reveals the letter, 3rd gives the full next word; hint level resets on a valid step
- Twenty Questions: raised minimum question floor to 10 + code guard so Qwen can't guess early

*GDINO ONNX/TRT investigation (Step 4):*
- Investigated all available export paths for GDINO Tiny on PyTorch 2.6 + transformers 5.5
- All paths blocked: trace ONNX (data-dependent Python loops), dynamo ONNX (unsupported ops), torch.compile (transformers 5.x output_capturing/dynamo bug), SDPA (not implemented for GDINO in transformers 5.5)
- `scene_service.py` already loads GDINO in fp16 with `torch.autocast` — already at the optimum reachable point
- At 5fps (200ms budget) with ~83ms inference, inference is not the bottleneck anyway

**Benchmark (RTX 4070 Ti Super):** fp32=113ms → fp16 autocast=83ms (1.36×)

### 2026-04-03 — Session 13
- **Switched tunnel from ngrok to Cloudflare Tunnel** — ngrok free tier hit bandwidth limits. Installed `cloudflared` (v2026.3.0) as a replacement.
  - Quick tunnel started with `cloudflared tunnel --url http://localhost:8000` — no account required, no bandwidth limits on free tier.
  - HTTPS provided automatically (required for browser mic access and Twilio webhooks).
  - Twilio voice webhook updated to the new `trycloudflare.com` URL.
  - Note: quick tunnel URLs change on each restart. A named tunnel (requires free Cloudflare account + `cloudflared tunnel login`) gives a stable persistent URL if needed in future.
- **Added quiz subjects: India, Cricket, Cooking**
  - India: festivals, cities, landmarks, famous personalities, history, culture, sports achievements
  - Cricket: rules, players, ICC tournaments, cricket terminology, records, famous grounds
  - Cooking: ingredients, cooking methods, kitchen tools, famous dishes by country, food science, cooking terms
  - Tightened subject-boundary rules in `quiz_generate.txt` for all subjects — each now has an explicit "stay within this subject only" guard to prevent cross-subject bleed

### 2026-04-03 — Session 14
- **Switched camera stream from MJPEG to WebSocket** — MJPEG (`multipart/x-mixed-replace`) is buffered and silently dropped by Cloudflare Tunnel, making the camera page blank when accessed remotely.
  - `camera_service`: replaced per-connection `VideoCapture` reader threads with a **singleton broadcaster**. One `_reader_loop` thread reads frames and pushes JPEG bytes to all registered subscriber queues via `_broadcast()`. Both MJPEG and WebSocket consumers use `subscribe_frames()` / `unsubscribe_frames()`.
  - `main.py`: added `GET /cameras/ws` WebSocket endpoint — accepts connection, streams raw JPEG bytes from `ws_frame_generator()`, handles `WebSocketDisconnect` cleanly.
  - `cameras.html`: replaced `<img>` with `<canvas>`; JS opens `wss://` (or `ws://` on plain HTTP) on connect, draws incoming JPEG blobs via `createImageBitmap()` + canvas 2D context. Multiple simultaneous browser clients now share a single reader thread.
  - MJPEG endpoint kept as local fallback.
- **Updated CLAUDE.md** — documentation rule expanded to cover ARCHITECTURE.md, WORKLOG.md, and README.md explicitly with trigger conditions for each. Dev workflow updated from ngrok to cloudflared.

### 2026-04-03 — Session 15
- **Diagnosed and fixed blank WebSocket camera stream** — WebSocket connected (`[WS] open`) but no frames arrived. Root cause discovered via server-side logging: `camera_service` had no logger, so the reader thread was silently failing. Added `logging.getLogger` to camera_service, plus log lines in `_reader_loop` (start, open failure, first 3 frames, crash, stop) and the WebSocket endpoint. Error surfaced: `failed to open stream /home/test.mp4` — video file was at `/home/jasvinder/test.mp4`, not `/home/test.mp4`. Fixed by using correct path; stream now works through Cloudflare Tunnel.
- **Added debug console logging to cameras.html** — logs WebSocket URL, open/close/error events, and first 3 frame sizes to browser console for future diagnostics.

### 2026-04-03 — Session 17
- **Added three new voice games for kids:** Bulls and Cows, Word Ladder, and 20 Questions.
- **Bulls and Cows** (`/games/bulls-cows`): Pure logic game — computer picks a secret 4-digit number (all unique, non-zero first); kid says digits aloud ("one two three four"); `parse_spoken_number()` maps tokens word-by-word (handles homophones: to→2, for→4, ate→8, digit chars). Server returns bulls/cows count; 10 attempts to win.
- **Word Ladder** (`/games/word-ladder`): Qwen generates a start+target word pair (4-letter, common kid vocabulary); BFS validates a path exists in the filtered system dictionary (`/usr/share/dict/words` → lowercase alpha-only 3-5 letters, frozenset built at import). Per-step validation: same length, exactly 1 letter different, word in set. BFS-based hint tells which letter position to change. 5 wrong attempts. Hardcoded fallback pairs if Qwen pair fails BFS. Vertical chain visualiser with changed letter underlined.
- **20 Questions** (`/games/twenty-questions`): Qwen asks yes/no questions to guess what the kid is thinking. Full Ollama `/api/chat` multi-turn conversation (messages list preserved in game state). 20-question limit; `force_twenty_questions_guess()` appends override message when limit reached. `_normalize_answer()` matches yes/no/maybe from natural speech. 4-phase UI: thinking → playing → guessing → finished. VAD only active during playing and guessing phases; paused during TTS (6s duration guard for longer questions).
- **Qwen additions:** `ask_twenty_questions(messages)`, `force_twenty_questions_guess(messages)`, `generate_word_ladder()` — all use 30s timeout. Non-JSON Qwen responses in 20Q handled gracefully (treated as question with warning log).
- **New routes:** 3 GET page routes + 8 POST game action routes added to main.py.
- **games.html:** 3 new cards (teal/green/purple) added to the existing 4-game grid.

### 2026-04-04 — Session 18
**Root cause of crash:** `Aborted (core dumped)` immediately after GDINO Tiny loaded on CUDA. Cause: GStreamer `Gst.init()` + PyTorch CUDA context initialisation on different threads → SIGABRT. Manifested every time the cameras page was opened.

**Fix — Triton Inference Server (Docker, Path A):**
- Moved all GDINO inference (PyTorch, transformers) out of the main process into a Triton Python backend running in a separate Docker container
- Main app process now has **zero PyTorch/CUDA** — no CUDA context to conflict with GStreamer
- Communication: `scene_service.py` → Triton gRPC (port 8001) via `tritonclient[grpc]`

**New files:**
- `triton_models/gdino/config.pbtxt` — Triton model config (Python backend, gRPC I/O: IMAGE bytes + QUERIES JSON + thresholds in, BOXES/SCORES/LABELS out)
- `triton_models/gdino/1/model.py` — Python backend; loads GDINO Tiny fp16; pre-resizes frames to 800px; returns full-res boxes via `target_sizes=[(orig_h, orig_w)]`
- `Dockerfile.triton` — `nvcr.io/nvidia/tritonserver:24.12-py3` + transformers + Pillow + opencv; gRPC-only on port 8001

**Changes:**
- `services/scene_service.py` — stripped all torch/transformers/PIL imports; replaced model loading + inference block with `_connect_triton()` + `_triton_infer()` gRPC calls; analysis loop is now CPU-only (ByteTrack still in process)
- `services/camera_service.py` — re-enabled GStreamer path (safe now; CUDA conflict eliminated)
- `requirements.txt` — removed torch/transformers/Pillow; added `tritonclient[grpc]`

**Key technical decisions:**
- `target_sizes=[(orig_h, orig_w)]` in model.py: GDINO outputs normalised [0,1] boxes; passing original dims directly converts them to full-res pixel coords even though inference ran on a downscaled frame — works because aspect ratio is preserved
- Triton Python backend chosen over TRT: GDINO Tiny's data-dependent Python loops in the text encoder block all ONNX/TRT export paths. Python backend has zero conversion friction and same model quality.
- HF cache mounted as Docker volume (`~/.cache/huggingface`) — avoids re-downloading ~900MB weights on each container start

**GStreamer status:** Disabled (PyAV used). Three crash attempts documented:
1. GDINO PyTorch + GStreamer nvcodec both init CUDA on different threads → SIGABRT (fixed: GDINO moved to Triton)
2. CTranslate2 (Whisper) CUDA context + GStreamer nvcodec scan on reader thread → SIGABRT (attempted fix: eager Gst.init at import time)
3. `Gst.init(None)` called inside reader thread function → SIGSEGV (GStreamer threading violation)

### 2026-04-11 — Session 19 — Docker Compose + GDINO FastAPI (Triton Python backend bypass)

**What was built:**

*Docker Compose containerisation (4-container stack):*
- Split the single bare-metal process into 4 Docker containers: `bianca-whisper`, `bianca-triton` (GDINO), `bianca-ollama`, `bianca-app`
- All containers share GPU via Nvidia Container Toolkit; `bianca-app` has **no GPU reservation** (enforced CUDA-free constraint)
- Bridge network `bianca-net` — containers reach each other by service name
- `depends_on: condition: service_healthy` — app waits for all three AI services before starting
- `~/.cache/huggingface` bind-mounted into whisper + triton; `ollama-models` named volume for Qwen weights
- `family-data` and `app-logs` named volumes survive app rebuilds
- New files: `Dockerfile.whisper`, `Dockerfile.app`, `Dockerfile.triton`, `docker-compose.yml`, `scripts/ollama-entrypoint.sh`

*Whisper microservice (Dockerfile.whisper):*
- `faster-whisper` wrapped in a FastAPI REST service (`services/whisper_server.py`)
- POST `/transcribe` accepts audio bytes; GET `/health` for healthcheck
- Loads `large-v3` fp16 at startup; responses are JSON `{transcript, confidence, language}`

*Triton Python backend — investigated and abandoned:*
- Implemented Grounding DINO Tiny as a Triton Python backend (`triton_models/gdino/1/model.py`)
- **Root cause of failure:** Triton 24.04 Python backend has a shared-memory IPC broadcast bug — when C++ Triton core writes tensor data to the shared-memory channel for the Python stub subprocess, **all N tensor elements are overwritten with the first element's value**. Pattern confirmed: `received[i] == received[0]` for all i, regardless of dtype (UINT8, FP32), shape, or transport (gRPC/HTTP).
- Attempted workarounds: UINT8 encoding, FP32 encoding, null stripping, HTTP transport, `rstrip(b"\x00")` — all failed because the bug is pre-data (in the IPC write, not serialisation).

*GDINO FastAPI service (replacement):*
- Created `triton_models/gdino_server.py` — standalone FastAPI/uvicorn app that loads and runs GDINO directly
- Runs in the same "triton" Docker container on port 8082 (bind-mounted from `./triton_models/`)
- POST `/infer`: multipart JPEG + JSON queries → JSON `{boxes, scores, labels}` — no Triton IPC involved
- GET `/health`: used by Docker Compose healthcheck and `scene_service` wait loop
- Dockerfile.triton CMD changed from `tritonserver ...` to `uvicorn gdino_server:app ...`

*scene_service.py refactor:*
- Removed all `tritonclient` imports and gRPC connection logic
- `_connect_triton()` now polls GET `/health` via `httpx.get` (waits up to 60s)
- `_triton_infer()` now sends multipart POST via `httpx.Client`, parses JSON response
- `_SimpleTracker` (pure-numpy IoU) replaces ByteTrack/supervision — eliminates pybind11/matplotlib dependency

*requirements.app.txt:*
- Removed `tritonclient[http]`; added comment noting GDINO moved to plain FastAPI

**Key technical decisions:**
- FastAPI over Triton Python backend: zero conversion friction, same model quality, same GPU
- Plain HTTP (httpx) over gRPC: simpler client, no protobuf, easier debugging with curl
- `_SimpleTracker` over supervision/ByteTrack: 0 native deps; sufficient for family-assistant use case
- App container intentionally CUDA-free: if a CUDA dep sneaks in, inference calls fail loudly

**Verified working:**
- Standalone test from app container confirmed GDINO service returns correct empty detections on black test image: `STANDALONE TEST PASSED`
- Full stack started with Docker Compose; whisper/triton/ollama pass healthchecks; app reaches healthy

---

### 2026-04-12 — DeepStream + YOLO-World Pipeline (Phases 0–4)

**Goal:** Replace GDINO FastAPI + PyAV software decode with DeepStream NVDEC hardware decode +
YOLO-World TRT inference via Triton. Target: <10ms inference, 4+ cameras, natural language queries.

#### What was built

**Phase 0 — Environment validation**
- Confirmed DeepStream 7.0 (`nvcr.io/nvidia/deepstream:7.0-gc-triton-devel`) runs on this host with GPU access
- ultralytics + CLIP fork installed in project venv
- YOLO-World nano sanity check passed

**Phase 1 — YOLO-World benchmark**
- `scripts/benchmark_yoloworld.py` — benchmarks YOLO-World PyTorch fp16 on GPU
- Key gotcha: use `half=True, device="cuda"` in predict kwargs, NOT `model.model.half()` (causes crash)
- yolov8m-worldv2 measured on this GPU

**Phase 2 — TRT export**
- `scripts/export_yoloworld_trt.py` — exports to `models/yoloworld.engine` + `models/yoloworld.meta.json`
- Selected yolov8m-worldv2 for better recall; baked queries: person, car, dog, bicycle, cat
- Engine size ~80MB; export time ~90s

**Phase 3 — DeepStream single-camera capture (PASS ✓)**
- **564 frames decoded from file, shape (720,1280,3), 48fps**
- Key gotcha: `nvurisrcbin` uses dynamic pads — `Gst.parse_launch()` doesn't work; must use
  explicit `Gst.ElementFactory.make()` + `pad-added` callback to link to nvstreammux

**Phase 3b — Triton YOLO-World backend (PASS ✓)**
- `triton_models/yoloworld/1/model.py` — Python backend loads TRT engine, runs inference
- `triton_models/yoloworld/config.pbtxt` — IMAGE (UINT8 flat JPEG), THRESHOLD (FP32) in;
  BOXES (FP32), SCORES (FP32), LABEL_IDS (FP32 class indices) out
- Query management: queries stored in `models/yoloworld.meta.json`; change queries by updating
  JSON + calling Triton unload/load API (no per-request QUERIES tensor)
- **14ms median inference at 2496×1664 (expect ~5-8ms at 1280×720); 16 detections on real image**
- Using `tritonclient.http` (not gRPC) — gRPC TYPE_STRING crashes Python backend

**Phase 4 — DeepStream → Triton inference integration (PASS ✓)**
- Leaky queue (max 1 frame) decouples NVDEC (48fps) from inference thread
- **8.1ms median inference, p99 9.9ms, 48.9fps decode, 426 inference frames in 11.5s**

#### Key technical decisions

| Decision | Rationale |
|---|---|
| No per-request QUERIES tensor | tritonserver 24.09 UINT8 variable-length tensor corruption bug; queries in meta.json + Triton reload is cleaner anyway |
| LABEL_IDS (FP32) not LABELS (STRING) | TYPE_STRING crashes pybind11 in Triton Python backend; FP32 class indices map to queries list client-side |
| tritonclient.http not gRPC | gRPC TYPE_STRING deserialization crashes Triton Python backend subprocess |
| tritonserver:25.03-py3 | Fixes widespread variable-length UINT8 tensor corruption vs 24.09 |
| YOLO-World M (not S) | Better recall at marginal VRAM cost |
| Explicit GStreamer element creation | nvurisrcbin dynamic pads are incompatible with parse_launch shorthand |
| Leaky queue (maxsize=1) | Decoder never waits on inference; always runs at latest frame |

#### Bugs encountered and fixed

- **tritonserver 24.09 UINT8 corruption**: all bytes in variable-length tensor set to first byte value. Fix: upgraded to tritonserver 25.03
- **tensorrt Python bindings missing** in 25.03: `tensorrt-cu12 tensorrt-lean-cu12 tensorrt-dispatch-cu12` from `pypi.nvidia.com` must be added to Dockerfile.triton
- **PyTorch CUDA version mismatch**: 25.03 uses CUDA 12.8; must use `--index-url .../cu128`
- **`__pycache__` caching old model.py**: Triton runs cached .pyc; `rm -rf triton_models/yoloworld/1/__pycache__` before restart
- **deepstream:7.1-devel doesn't exist**: use `7.0-gc-triton-devel`
- **libGL.so.1 missing**: ultralytics pulls non-headless opencv; fix: uninstall + reinstall headless variant

#### Updated files

- `Dockerfile.triton` — upgraded to tritonserver:25.03-py3, added TRT Python bindings, cu128 PyTorch
- `Dockerfile.deepstream` — new, based on deepstream:7.0-gc-triton-devel
- `docker-compose.yml` — added triton service, added deepstream service, updated app service
- `triton_models/yoloworld/config.pbtxt` — final tensor interface
- `triton_models/yoloworld/1/model.py` — TRT backend
- `scripts/benchmark_yoloworld.py` — Phase 1 benchmark
- `scripts/export_yoloworld_trt.py` — Phase 2 TRT export

---

### 2026-04-12 — Session 19 — DeepStream pipeline Phases 5–8 (multi-camera service + UI)

**What was built:**

**Phase 5 — Multi-camera DeepStream pipeline (PASS ✓)**
- Topology: `nvurisrcbin × N → nvstreammux(batch=N) → nvmultistreamtiler(1×N) → nvvideoconvert → appsink`
- appsink receives one tiled frame (W*N × H); callback slices by column index for per-camera frames
- Result: both cameras 39fps decode, cam0 6.1ms median inference, cam1 8.1ms median — PASS ✓

**Phase 6 — `deepstream_service.py` — service layer (PASS ✓)**
- `services/deepstream_service.py` — single module replacing both `camera_service.py` and `scene_service.py`
- All 37/37 checks pass in `scripts/test_service.py`
- Key design points:
  - Backward-compatible API: `set_stream_url`, `get_stream_url`, `start_analysis`, `mjpeg_generator`, `push_frame` all preserved
  - `add_query` updates `_queries` in-memory immediately (optimistic); background worker only handles meta.json + Triton reload
  - EOS bus watch seeks to position 0 for file source looping
  - `_rebuild(new_streams)` handles hot add/remove of cameras without app restart
  - `CameraEvent.cam_id` field added for multi-camera event attribution

**Phase 7 — `main.py` wiring**
- Replaced `from services import camera_service, scene_service` with `deepstream_service` aliased to both names
- Added new REST endpoints:
  - `GET /cameras/queries/status` — returns `{state, eta_s}` for re-export progress
  - `POST /cameras/streams` — add camera `{cam_id, url}`
  - `DELETE /cameras/streams/{cam_id}` — remove camera
  - `GET /cameras/streams` — list active streams
  - `WS /cameras/ws/{cam_id}` — per-camera WebSocket frame stream

**Phase 8 — Multi-camera UI (`cameras.html`)**
- Replaced single stream URL input with "Add Camera" form (cam_id + URL)
- Dynamic camera grid: 1 cam = full width, 2 cams = 2-col, 3+ cams = 3-col responsive grid
- Each tile: per-camera WebSocket `/cameras/ws/{cam_id}`, label overlay, remove button
- Query status indicator: "⏳ Updating query engine…" polls `/cameras/queries/status` while state=updating
- Event log: added camera ID column + filter dropdown by cam_id

#### Key technical decisions

| Decision | Rationale |
|---|---|
| nvmultistreamtiler instead of nvstreamdemux | nvstreamdemux dynamic pads not linked before data flows → immediate EOS; tiler outputs one flat frame that is trivially sliced by column |
| Single GStreamer Pipeline for all sources | Two Gst.Pipeline objects in same process cause DeepStream context conflict; all cameras share one pipeline |
| Optimistic query update | Tests showed `add_query` returning False for duplicate when background worker hadn't finished yet; immediate in-memory update fixes this |
| EOS bus watch seek-to-0 | File sources hit EOS and stop; bus watch loops them by seeking to position 0 on EOS |
| `_rebuild` for hot camera add/remove | Stops inference + pipeline, clears removed camera state (trackers, detections), restarts with new stream set |

#### Bugs encountered and fixed

- **Two Gst.Pipeline objects crash**: DeepStream CUDA context conflict. Fix: single shared pipeline.
- **nvvideoconvert batch transform failed** with batch-size=2: Can't convert batched NVMM to system memory. Fix: insert nvmultistreamtiler before nvvideoconvert.
- **nvstreamdemux immediate EOS**: Dynamic pads not linked when data flows. Fix: use nvmultistreamtiler.
- **`file://workspace/...` (2 slashes)**: nvurisrcbin fails. Fix: `file:///workspace/...` (3 slashes).
- **`add_query` duplicate check failing**: `_queries` not updated until worker completes. Fix: update list inside lock before spawning worker.
- **Test 7 ws_frame_generator hanging**: File source hit EOS during 55s test wait. Fix: restart stream before test + `asyncio.wait_for(10s)` + EOS bus watch.

#### Updated files

- `services/deepstream_service.py` — new service replacing camera_service + scene_service
- `scripts/test_service.py` — Phase 6 standalone service test (37/37 PASS)
- `main.py` — Phase 7 wiring: new imports + 5 new camera endpoints
- `templates/cameras.html` — Phase 8 multi-camera grid UI
- `ARCHITECTURE.md` — updated camera pipeline diagram, REST API table, query change flow

---

### 2026-04-12 — TRT query re-export orchestration + QA improvements

**What was built:**

- **Fixed TRT query change bug** — changing queries was remapping labels but not re-encoding text
  embeddings; the TRT engine still detected old class patterns with new labels. Root cause: TRT bakes
  CLIP text embeddings at export time; swapping `meta.json` only changes the label string mapping.
  Fix: added `engine_queries` field to meta.json to track what's baked into the TRT engine; queries
  can now only be changed correctly via full TRT re-export.

- **Management sidecar** (`triton_models/management.py`) — FastAPI app running on port 8004 alongside
  tritonserver. Handles TRT re-export requests (`POST /reexport`, `GET /reexport/status`). Runs
  ultralytics `YOLOWorld.export(format="engine")` in a background thread; atomically swaps engine
  file and updates `meta.json` on completion. Startup script `scripts/triton_start.sh` launches
  both tritonserver and the management sidecar.

- **App-level re-export orchestration** — `main.py` owns the full cross-service lifecycle. When
  queries change: (1) deepstream updates `_queries` + meta.json immediately; (2) app pauses Ollama
  (`keep_alive=0`) to free ~10GB VRAM; (3) app calls triton management sidecar to start TRT build;
  (4) app polls until done; (5) app reloads Triton; (6) app resumes Ollama. deepstream_service has
  no knowledge of Ollama — clean service boundary.

- **Qwen-dependent routes blocked during re-export** — `/chat`, `/voice/transcription`,
  `/games/quiz/generate`, `/games/word-ladder/new`, `/games/twenty-questions/start`,
  `/games/twenty-questions/answer`, `/games/resolve-answer` return 503 with eta when
  `ollama_paused=True`. Voice routes return TwiML saying to call back in a couple of minutes.

- **Phase 6 QA regression tests** (`scripts/test_service_quality.py`) — 34 checks covering:
  - QA-1: FastAPI app existence, required routes, no route/variable name collisions
  - QA-2: BGR colour correctness (RGBA→BGR round-trip via JPEG)
  - QA-3: File source EOS looping (trailing gap check catches pipeline stall)
  - QA-4: Detections appear via Triton (confirms correct frame delivery)

**Key bugs found and fixed:**
| Bug | Root cause | Fix |
|---|---|---|
| Wrong colours in browser | `[:,:,:3]` gives RGB not BGR; OpenCV expects BGR | `[:,:,2::-1]` for RGBA→BGR |
| EOS never loops (file sources) | `bus.add_watch()` requires GLib main loop; uvicorn never starts one | `timed_pop_filtered()` poll thread → rebuild pipeline on EOS |
| Route function shadows module variable | `async def _events()` shadowed `_events: deque` → `TypeError: 'function' object is not iterable` | Renamed route functions (`_events_route`, `_queries_status_route`) |
| Inference disabled silently | `tritonclient.http` not installed; daemon thread dies on ImportError with no output | Added `tritonclient[grpc,http]` to Dockerfile.deepstream + explicit try/except in `_inference_loop` |
| Detections with wrong labels after query change | TRT engine has fixed text embeddings; label remapping without re-export gives wrong results | Full TRT re-export on every query change |

**Updated files:**
- `main.py` — re-export orchestration, Qwen pause/resume, LLM availability guards on Qwen routes
- `services/deepstream_service.py` — removed all Ollama/Triton orchestration; `add_query`/`remove_query` only update `_queries` + meta.json; added `/queries/commit` endpoint
- `triton_models/management.py` — new management sidecar (port 8004)
- `scripts/triton_start.sh` — new startup script for triton container
- `Dockerfile.triton` — added fastapi/uvicorn, COPY triton_start.sh, changed CMD
- `docker-compose.yml` — exposed port 8004 for management sidecar
- `scripts/export_yoloworld_trt.py` — saves `engine_queries` field to meta.json
- `scripts/test_service_quality.py` — new Phase 6 QA regression tests (34 checks)
- `ARCHITECTURE.md` — restructured: tech stack and Docker sections moved to top; HTTP vs gRPC decision documented; TRT engine_queries / query orchestration decisions added

---

### 2026-04-14 — RTSP pipeline debugging + fix

**Goal:** Get `rtsp://mediamtx:8554/cam1` streaming into the camera UI with GPU decode (nvv4l2decoder).

**Root cause found:** `import cv2` before `Gst.init()` causes SIGABRT. OpenCV (CUDA build) initialises NVIDIA codec libraries in an order that conflicts with DeepStream's own CUDA init triggered by Gst.init(). Produced `std::runtime_error("Unable to read configuration")` immediately after `pipeline.set_state(PLAYING)`.

Diagnosed via progressive isolation tests — everything passed individually (rtspsrc, decode chain, appsink callback) until cv2 was isolated as the trigger.

**Fix + refactor:** Removed cv2 and numpy from pipeline_worker.py entirely. Switched from a shared nvstreammux+nvmultistreamtiler topology to per-camera independent chains with GStreamer-native JPEG encoding:

```
rtspsrc (latency=0) → rtph264/5depay → h264/5parse → nvv4l2decoder
        → nvvideoconvert → capsfilter(NVMM I420) → nvjpegenc → appsink
```

`nvjpegenc` (hardware) keeps the frame in NVMM GPU memory through encode — no GPU→CPU copy until final JPEG bytes land in appsink. The `new-sample` callback does `bytes(minfo.data)` and writes to stdout — no numpy, no cv2.

**Performance improvements applied:**
- `rtspsrc latency=0` (was 100ms) — removes jitter buffer, cuts E2E latency on LAN
- `nvjpegenc` instead of `jpegenc` (CPU) — JPEG encode stays in NVMM, faster
- Removed `nvstreammux` + `nvmultistreamtiler` — not needed for display; simpler pipeline

**Other findings:**
- nvurisrcbin (rank 256) must be bypassed for RTSP in DS7.0 — use rtspsrc directly
- asyncio + rtspsrc in the same process → SIGABRT — fixed by subprocess isolation
- nvstreammux/nvmultistreamtiler only needed when feeding nvinfer (batching requirement)

### 2026-04-14 — Codebase cleanup

Removed dead code accumulated since the GDINO → YOLO-World / camera_service → deepstream_service migration:

**Deleted:**
- `services/camera_service.py` — replaced by `deepstream_service.py`
- `services/scene_service.py` — replaced by `deepstream_service.py`
- `triton_models/gdino/` — GDINO fully replaced by YOLO-World TRT
- `triton_models/gdino_server.py` — old GDINO FastAPI server
- `scripts/export_gdino_onnx.py` — GDINO export script
- `scripts/test_deepstream_capture.py`, `test_deepstream_inference.py`, `test_deepstream_multicam.py` — phase 3/4/5 dev scripts (stale patterns: cv2 before Gst.init, old mux topology)
- `improvement_ideas.md` — 231-line doc about PyAV→GStreamer migration and GDINO optimisations, fully shipped

**Other fixes:**
- `.gitignore` — added `*.onnx` (109MB files were about to be committed)
- `README.md` — removed gdino_server reference, added pipeline_worker.py to project structure

---

## Open Questions / Future Ideas
- Add a "complete todo" voice command ("mark buy groceries as done")
- Scheduled reminders: outbound WhatsApp at event time
- Multi-language support (Qwen handles this well)
- family.md sync via Dropbox/git for backup
- Presentation generation (python-pptx) and WhatsApp as document
