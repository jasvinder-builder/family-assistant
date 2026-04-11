# Family Assistant — Worklog

## Project Overview
A voice-based family productivity assistant. Family members call a Twilio number, speak commands, and the assistant handles todos, events, and research requests. Research results are sent via WhatsApp. Qwen runs locally via Ollama. No email, no database — just markdown files.

---

## Status: Live and being tested locally

### Agent name: Bianca (female voice — Polly.Aditi)

---

## Design

### Folder Structure
```
family-assistant/
├── main.py                        # FastAPI app, route definitions
├── config.py                      # Settings, env loading, phone→name mapping
├── family.md                      # Shared storage (todos + events)
├── .env                           # API keys (never commit)
├── .env.example                   # Template
├── requirements.txt
│
├── handlers/
│   ├── call_handler.py            # Twilio webhook entry, resolves caller
│   ├── intent_handler.py          # Routes intents to sub-handlers
│   ├── todo_handler.py            # Add/query todos
│   ├── event_handler.py           # Add/query events
│   ├── research_handler.py        # Async research + WhatsApp dispatch
│   └── response_handler.py        # TwiML builder helpers
│
├── services/
│   ├── qwen.py                    # Ollama REST wrapper
│   ├── twilio_service.py          # TwiML + WhatsApp sender
│   ├── tavily_service.py          # Web/job/image search
│   └── markdown_service.py        # Read/write family.md with filelock
│
├── models/
│   └── schemas.py                 # Pydantic models: Intent, TodoItem, EventItem
│
└── prompts/
    ├── intent_classify.txt
    ├── todo_extract.txt
    ├── event_extract.txt
    └── research_synthesize.txt
```

### Tech Stack
| Component | Tool |
|---|---|
| Voice calls | Twilio (inbound) |
| Speech-to-text | Twilio `<Gather input="speech">` |
| Text-to-speech | Twilio `<Say>` |
| LLM | Qwen via Ollama (local) |
| Web search | Tavily API |
| Outbound messaging | Twilio WhatsApp API |
| Storage | Markdown file (`family.md`) |
| Backend | Python + FastAPI |
| File locking | `filelock` library |
| Datetime parsing | Qwen (inject today's date into prompt) |

### Intent Types
| Intent | Example trigger |
|---|---|
| `add_todo` | "add a todo to buy milk" |
| `add_event` | "add dentist appointment next Tuesday at 2pm" |
| `query_tasks` | "what's on my list", "what are my todos" |
| `query_events` | "what events do I have this week" |
| `research_web` | "look up", "find info about" |
| `research_jobs` | "find biotech jobs near Newark" |
| `research_images` | "find Kuromi cake pictures" |
| `unknown` | fallback with help message |

### family.md Schema
```markdown
# Family Assistant

## Todos

- [ ] Buy groceries | due: 2026-04-03 | added_by: Alice | added_at: 2026-03-29T10:15:00
- [x] Renew passport | due: none | added_by: Bob | added_at: 2026-03-20T09:00:00 | completed_at: 2026-03-25T12:00:00

## Events

- 2026-04-10T14:00:00 | Dentist appointment | added_by: Alice | added_at: 2026-03-29T10:20:00
- 2026-05-01T00:00:00 | Family vacation starts | added_by: Bob | added_at: 2026-03-01T08:00:00
```

### Call Flow Summary

**Add todo / event:**
```
Call → Twilio STT → POST /voice/transcription
→ Qwen: classify intent
→ Qwen: extract structured data
→ Write to family.md (filelock)
→ TwiML voice confirmation → hang up
```

**Query tasks / events:**
```
Call → STT → classify
→ Read family.md
→ Qwen: summarize for voice
→ TwiML speaks summary → hang up
```

**Research (jobs, web, images):**
```
Call → STT → classify
→ TwiML: "I'll send results to your WhatsApp shortly" → hang up immediately
→ [background task] Tavily search → Qwen synthesize → WhatsApp message sent
```

**Unknown caller:**
```
Call → not in PHONE_TO_NAME → TwiML: "Not registered. Goodbye." → hang up
```

### Key Design Decisions
- **Twilio Gather for STT** (not Whisper) — simpler, synchronous, no audio download needed
- **Two Qwen calls** per request — one for classification (fast, small), one for extraction (focused prompt). More reliable than one combined call.
- **FastAPI BackgroundTasks** for async research — no Redis/Celery needed at family scale
- **filelock** for markdown writes — concurrent writes are extremely rare at family scale
- **No confirmation step** — family members can manually edit family.md if something is wrong
- **Caller identified by phone number** — Twilio passes `From` field, mapped to name in `.env`
- **Qwen model sizes** — use `qwen2.5:7b` for classification (speed), `qwen2.5:14b` for synthesis (quality)

### Environment Variables (.env)
```
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_PHONE_NUMBER=
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
TAVILY_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:14b
FAMILY_MD_PATH=./family.md
PHONE_TO_NAME={"+"447911123456":"Alice","+447911987654":"Bob"}
```

---

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

### Next Steps
- [x] Scaffold folder structure and stub files
- [x] Implement `config.py` and `.env.example`
- [x] Implement `services/markdown_service.py` + `family.md` initial file
- [x] Implement `services/qwen.py` (Ollama wrapper)
- [x] Write all prompts (intent_classify, todo_extract, event_extract, research_synthesize)
- [x] Implement `handlers/call_handler.py` + `handlers/response_handler.py`
- [x] Implement `handlers/intent_handler.py`
- [x] Implement `handlers/todo_handler.py` + `handlers/event_handler.py`
- [x] Implement `services/tavily_service.py`
- [x] Implement `handlers/research_handler.py`
- [x] Implement `services/twilio_service.py` (WhatsApp)
- [x] Write `main.py` and wire everything together
- [x] Verify all imports clean with venv
- [ ] Copy `.env.example` to `.env` and fill in credentials
- [ ] Test locally with ngrok + Twilio dev number
- [ ] Test each intent type end-to-end

### 2026-04-03 — Session 17
- **Added three new voice games for kids:** Bulls and Cows, Word Ladder, and 20 Questions.
- **Bulls and Cows** (`/games/bulls-cows`): Pure logic game — computer picks a secret 4-digit number (all unique, non-zero first); kid says digits aloud ("one two three four"); `parse_spoken_number()` maps tokens word-by-word (handles homophones: to→2, for→4, ate→8, digit chars). Server returns bulls/cows count; 10 attempts to win.
- **Word Ladder** (`/games/word-ladder`): Qwen generates a start+target word pair (4-letter, common kid vocabulary); BFS validates a path exists in the filtered system dictionary (`/usr/share/dict/words` → lowercase alpha-only 3-5 letters, frozenset built at import). Per-step validation: same length, exactly 1 letter different, word in set. BFS-based hint tells which letter position to change. 5 wrong attempts. Hardcoded fallback pairs if Qwen pair fails BFS. Vertical chain visualiser with changed letter underlined.
- **20 Questions** (`/games/twenty-questions`): Qwen asks yes/no questions to guess what the kid is thinking. Full Ollama `/api/chat` multi-turn conversation (messages list preserved in game state). 20-question limit; `force_twenty_questions_guess()` appends override message when limit reached. `_normalize_answer()` matches yes/no/maybe from natural speech. 4-phase UI: thinking → playing → guessing → finished. VAD only active during playing and guessing phases; paused during TTS (6s duration guard for longer questions).
- **Qwen additions:** `ask_twenty_questions(messages)`, `force_twenty_questions_guess(messages)`, `generate_word_ladder()` — all use 30s timeout. Non-JSON Qwen responses in 20Q handled gracefully (treated as question with warning log).
- **New routes:** 3 GET page routes + 8 POST game action routes added to main.py.
- **games.html:** 3 new cards (teal/green/purple) added to the existing 4-game grid.

### 2026-04-02 — Session 16
- **Expanded YOLO detection to vehicles and animals** — previously only class 0 (person). Now detects: bicycle (1), car (2), motorcycle (3), bus (5), truck (7), bird (14), cat (15), dog (16). All feed into CLIP matching and the event log. Debug overlay now shows the class label (e.g. `car #3`) alongside track ID.
- **Added configurable crop padding** (`_crop_padding`, default 0.3) — before CLIP inference the YOLO bounding box is expanded by `pad_factor × box_size` on each side (clamped to frame boundaries). Gives CLIP more surrounding context for relational queries like "person with stroller" or "dog near car".
  - New `GET/POST /cameras/pad` endpoints to read/set the factor at runtime.
  - Pad factor slider (0.0–1.0, step 0.05) added to Scene Queries card in cameras.html.
- **Detection dataclass extended** — added `label: str` field to `Detection`; populated from `_CLASS_LABELS` map keyed by YOLO class ID.

### 2026-04-03 — Session 15
- **Diagnosed and fixed blank WebSocket camera stream** — WebSocket connected (`[WS] open`) but no frames arrived. Root cause discovered via server-side logging: `camera_service` had no logger, so the reader thread was silently failing. Added `logging.getLogger` to camera_service, plus log lines in `_reader_loop` (start, open failure, first 3 frames, crash, stop) and the WebSocket endpoint. Error surfaced: `failed to open stream /home/test.mp4` — video file was at `/home/jasvinder/test.mp4`, not `/home/test.mp4`. Fixed by using correct path; stream now works through Cloudflare Tunnel.
- **Added debug console logging to cameras.html** — logs WebSocket URL, open/close/error events, and first 3 frame sizes to browser console for future diagnostics.

### 2026-04-03 — Session 14
- **Switched camera stream from MJPEG to WebSocket** — MJPEG (`multipart/x-mixed-replace`) is buffered and silently dropped by Cloudflare Tunnel, making the camera page blank when accessed remotely.
  - `camera_service`: replaced per-connection `VideoCapture` reader threads with a **singleton broadcaster**. One `_reader_loop` thread reads frames and pushes JPEG bytes to all registered subscriber queues via `_broadcast()`. Both MJPEG and WebSocket consumers use `subscribe_frames()` / `unsubscribe_frames()`.
  - `main.py`: added `GET /cameras/ws` WebSocket endpoint — accepts connection, streams raw JPEG bytes from `ws_frame_generator()`, handles `WebSocketDisconnect` cleanly.
  - `cameras.html`: replaced `<img>` with `<canvas>`; JS opens `wss://` (or `ws://` on plain HTTP) on connect, draws incoming JPEG blobs via `createImageBitmap()` + canvas 2D context. Multiple simultaneous browser clients now share a single reader thread.
  - MJPEG endpoint kept as local fallback.
- **Updated CLAUDE.md** — documentation rule expanded to cover ARCHITECTURE.md, WORKLOG.md, and README.md explicitly with trigger conditions for each. Dev workflow updated from ngrok to cloudflared.

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
  - Tightened subject-boundary rules in `quiz_generate.txt` for all subjects — each now has an explicit "stay within this subject only" guard to prevent cross-subject bleed (e.g. Arts generating music questions)

### 2026-04-02 — Session 12
- **Fixed debug overlay bounding boxes not tracking with moving persons** — root cause: two independent `VideoCapture` instances (one in camera_service for MJPEG display, one in scene_service for AI analysis) reading the same file at different frame positions. Boxes were stamped from the analysis thread's position onto the display thread's frame → appeared frozen/misaligned.
  - **Fix: single-reader frame sharing.** Removed the separate VideoCapture from `scene_service`. Camera_service's `_reader()` now calls `scene_service.push_frame(frame)` on every decoded frame. Scene service stores the latest frame in `_shared_frame` (protected by a lock + `threading.Event`). Analysis loop waits for the event and samples from the shared frame at 3fps — bounding boxes now perfectly align with what's shown in the live view.
- **Fixed debug overlay not showing when no queries are defined** — analysis loop skipped YOLO entirely when `queries` was empty, so `_latest_detections` was never populated. Fixed by also checking the debug overlay flag: YOLO runs whenever `queries` OR `debug_overlay` is active.
  - Debug overlay flag is now mirrored into `scene_service` via `scene_service.set_debug_overlay()` when `POST /cameras/debug-overlay` is called.
- **Added CLIP match image thumbnails to event log** — when CLIP similarity ≥ threshold, the person crop is encoded as a base64 JPEG and stored on `CameraEvent.image_b64`. The `/cameras/events` API returns it; the event log renders a 56×80px thumbnail alongside each event row.
- **Added threshold and debug-overlay routes to ARCHITECTURE.md**

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

### 2026-04-02 — Session 9
- **Fixed `minSpeechMs` in clock.html and quiz.html** — lowered from 600ms to 250ms. Single-letter answers ("D") are ~150ms of actual speech; 600ms threshold silently discarded them before VAD ever fired. Duration guards (>4s clock, >5s quiz) already handle TTS echo, so minSpeechMs does not need to be high.
- **Fixed Hangman game broken state** — three issues combined to make the game freeze or mis-behave:
  1. `speak()` had no watchdog timer — if `speechSynthesis.onend` never fired (unreliable on mobile), `resumeListening()` was never called and the game stayed in processing state permanently. Fixed with same watchdog pattern as clock/quiz: `Math.ceil(text.length/15)*1000 + 3000ms`.
  2. No duration guard in `onSpeechEnd` — TTS audio echo (always >4s) was being sent to Whisper as guesses. Added `if (audio.length / 16000 > 4) return;`.
  3. No confidence check — low-quality transcripts from noise were submitted as guesses. Added `(data.confidence ?? 1) >= 0.30` gate.
- All three games (clock, quiz, hangman) now consistent: minSpeechMs 250ms, duration guard, confidence 0.30, watchdog timer in speak().

### 2026-04-01 — Session 8
- **Root cause: `speechSynthesis.onend` unreliable on mobile browsers** — when TTS finishes but `onend` silently never fires, `resumeListening()` is never called, VAD stays paused, ring shows "ready" but nothing is actually listening. This was the main cause of "stuck in listening state" in clock and quiz games.
- **Fix: TTS watchdog timer** added to `speak()` in clock.html and quiz.html — estimates TTS duration (`text.length / 15` seconds) + 3s buffer, then force-calls the resume logic if `onend` hasn't fired. Clears itself if `onend` fires normally.
- **Fix: "Bianca is speaking" ring state** — light blue ring + "Bianca is speaking..." status text while TTS plays, so user knows not to speak yet. Previously ring showed "ready" during TTS which was misleading.
- **Fix: show heard transcript** — after Whisper transcribes, show `Heard: "..."` below the ring so user can see exactly what was detected. Cleared when next listening cycle begins. Gives users immediate feedback to self-correct ("I heard: see you later" → user understands why and tries "letter C").
- **Key learning — production voice robustness:** production apps don't have smarter VAD — they treat every async operation as potentially failing silently. Rule: *if a state can be entered, a timer or fallback must be able to exit it*, not just an event that might never fire. Apply this to any future async state transitions.
- **Key learning — `speechSynthesis.onend` on mobile:** cannot be relied on. Always pair with a watchdog timer. Estimate duration as `(text.length / 15) * 1000 + 3000ms`.

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

### 2026-03-31 — Session 4
- Added **Times Tables** game (`/games/multiply`): random A×B questions (1–9), VAD captures spoken answer, spoken number parser handles digits and English words up to 81, score tracking, Fresh Start button
- Added **Tell the Time** game (`/games/clock`): 4-option multiple choice, 4 SVG analogue clock faces drawn in pure JS (no images/libraries), distractors guaranteed visually distinct, VAD captures A/B/C/D, tapping a clock also accepted, Bianca speaks human-friendly time ("half past 3"), score tracking
- Hangman: pre-reveal hint letters at game start — 0 hints for ≤4-letter words, +1 per letter above 4, capped at 3; `_hint_count()` added to `hangman_service.py`
- Home page: games section moved to top, assistant cards moved to bottom; hero header made compact (1.5rem, reduced padding); Games/Assistant section labels added
- All new games follow same VAD + speech-unlock pattern as Talk to Bianca and Hangman

### 2026-03-31 — Session 3
- Replaced tap-to-record + manual `AnalyserNode` silence detection with `@ricky0123/vad-web` (Silero VAD via ONNX Runtime Web)
- VAD runs entirely on the client device (phone/tablet CPU via WASM) — no audio sent to server until speech detected
- Removed mic button from `talk.html` and `hangman.html`; voice is always listening after page load
- Added animated ring indicator showing loading / listening / speech-detected / processing states
- VAD paused during TTS playback to prevent Bianca's voice feeding back into Whisper
- Hangman: VAD stays active after game over so players can say "new game" without tapping
- Updated `/transcribe` endpoint in `main.py` to detect audio format from filename (`.wav` or `.webm`) instead of hardcoding `.webm`; VAD delivers Float32Array encoded to WAV client-side
- CDN versions pinned: `@ricky0123/vad-web@0.0.30`, `onnxruntime-web@1.22.0` (must match — 1.22.0 is the ORT version bundled inside vad-web 0.0.30)

---

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

---

---

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

**Updated:** `improvement_ideas.md` with Steps 7 (DALI) and 8 (DeepStream), plus a phased roadmap table

---

### 2026-04-04 — Session 18 — Triton Python backend for GDINO (camera crash fix)

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
- Triton Python backend chosen over TRT: GDINO Tiny's data-dependent Python loops in the text encoder block all ONNX/TRT export paths (investigated in Session 6/improvement_ideas.md Step 4). Python backend has zero conversion friction and same model quality. TRT upgrade can follow if GDINO gets ONNX support or model is replaced with YOLOv8-World.
- HF cache mounted as Docker volume (`~/.cache/huggingface`) — avoids re-downloading ~900MB weights on each container start

**To start Triton:**
```bash
docker build -f Dockerfile.triton -t family-assistant-triton .
docker run --gpus all -d -p 8001:8001 -v ${PWD}/triton_models:/models -v ${HOME}/.cache/huggingface:/root/.cache/huggingface --name triton-gdino family-assistant-triton
```

**GStreamer status:** Disabled (PyAV used). Three crash attempts documented:
1. GDINO PyTorch + GStreamer nvcodec both init CUDA on different threads → SIGABRT (fixed: GDINO moved to Triton)
2. CTranslate2 (Whisper) CUDA context + GStreamer nvcodec scan on reader thread → SIGABRT (attempted fix: eager Gst.init at import time)
3. `Gst.init(None)` called inside reader thread function → SIGSEGV (GStreamer threading violation)

**Next session fix for GStreamer:** Remove `Gst.init(None)` from `_reader_loop_gst()` (already called at import). If nvcodec CUDA scan still conflicts with CTranslate2, add `os.environ["GST_PLUGIN_FEATURE_RANK"] = "nvh264dec:0,nvh265dec:0"` before `Gst.init()` call.

---

### 2026-04-03 — Session 6 — Game improvements + GDINO ONNX investigation

**What was built:**

*Game improvements (committed in session 5):*
- Bulls & Cows: difficulty selector (2/3/4 digits, Grades 1–3/4–6/7+), always 10 guesses
- Word Ladder: escalating hints — 1st hint names the position, 2nd reveals the letter, 3rd gives the full next word; hint level resets on a valid step
- Twenty Questions: raised minimum question floor to 10 + code guard so Qwen can't guess early

*GDINO ONNX/TRT investigation (Step 4):*
- Investigated all available export paths for GDINO Tiny on PyTorch 2.6 + transformers 5.5
- All paths blocked: trace ONNX (data-dependent Python loops), dynamo ONNX (unsupported ops), torch.compile (transformers 5.x output_capturing/dynamo bug), SDPA (not implemented for GDINO in transformers 5.5)
- `scene_service.py` already loads GDINO in fp16 with `torch.autocast` — already at the optimum reachable point
- At 5fps (200ms budget) with ~83ms inference, inference is not the bottleneck anyway
- Documented in `improvement_ideas.md` (Step 4 table) and `scripts/export_gdino_onnx.py`

**Benchmark (RTX 4070 Ti Super):** fp32=113ms → fp16 autocast=83ms (1.36×)

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
- Diagnostic: added `print(f"[diag] elem[{i}]={v}")` in model.py; confirmed `first_byte & 0x3F` repeated N times for UINT8; `first_float` repeated for FP32.

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
- `sync=true` on appsink for local file GStreamer sources (correct playback speed)

**Verified working:**
- Standalone test from app container confirmed GDINO service returns correct empty detections on black test image: `STANDALONE TEST PASSED`
- Full stack started with Docker Compose; whisper/triton/ollama pass healthchecks; app reaches healthy

---

## Open Questions / Future Ideas
- Add a "complete todo" voice command ("mark buy groceries as done")
- Scheduled reminders: outbound WhatsApp at event time
- Multi-language support (Qwen handles this well)
- family.md sync via Dropbox/git for backup
- Presentation generation (python-pptx) and WhatsApp as document
