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

## Open Questions / Future Ideas
- Add a "complete todo" voice command ("mark buy groceries as done")
- Scheduled reminders: outbound WhatsApp at event time
- Multi-language support (Qwen handles this well)
- family.md sync via Dropbox/git for backup
- Presentation generation (python-pptx) and WhatsApp as document
