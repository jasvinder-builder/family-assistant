# Bianca — Family Assistant: Architecture

## Overview

Bianca is a family productivity assistant with two interfaces: phone calls (via Twilio) and a browser interface (any device on the local network). She handles todos, events, research, and games. Research results are delivered via WhatsApp. Proactive event reminders are sent automatically. All AI runs locally on GPU — no cloud AI services.

---

## System Architecture

```
                    ┌─────────────────┐        ┌──────────────────────┐
                    │  CALLER'S PHONE │        │  BROWSER (phone /    │
                    │  (PSTN call)    │        │  Portal / tablet)    │
                    └────────┬────────┘        └──────────┬───────────┘
                             │                            │ HTTPS (ngrok)
                             ▼                            │ MediaRecorder audio
                    ┌────────────────┐                    │ JSON responses
                    │  TWILIO CLOUD  │                    │
                    │  inbound call  │                    │
                    │  TTS playback  │                    │
                    └────────┬───────┘                    │
                             │ webhooks (HTTP POST)       │
                             ▼                            ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     FASTAPI APPLICATION  (uvicorn :8000)               │
│                                                                        │
│  Phone routes:                    Browser routes:                      │
│  POST /voice/incoming             GET  /                               │
│  POST /voice/transcription        GET  /talk                           │
│  POST /voice/research-choice      GET  /dashboard                      │
│  POST /voice/research-whatsapp-   POST /dashboard/add-todo             │
│       choice                      POST /dashboard/complete-todo        │
│  GET  /voice/answer/{sid}         POST /dashboard/delete-todo          │
│                                   POST /dashboard/add-event            │
│                                   POST /dashboard/delete-event         │
│                                   POST /transcribe  (Whisper STT)      │
│                                   POST /chat        (text→JSON)        │
│                                   GET  /games/hangman                  │
│                                   POST /games/hangman/new              │
│                                   POST /games/hangman/guess            │
│                                   GET  /health                         │
└───────────────┬────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                  LOCAL MACHINE  (RTX 4070 Ti Super, 16GB VRAM)        │
│                                                                       │
│  ┌──────────────────────┐    ┌─────────────────────────────────────┐  │
│  │  faster-whisper      │    │  Ollama  (REST on :11434)           │  │
│  │  large-v3 / CUDA     │    │  Qwen 2.5:14b  Q4_K_M              │  │
│  │  int8_float16        │    │  ~9-10GB VRAM                       │  │
│  │  ~1.5GB VRAM         │    │  Warmed up at startup               │  │
│  │  Loaded at startup   │    └─────────────────────────────────────┘  │
│  └──────────────────────┘                                             │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  family.md  (pipe-delimited markdown, filelock protected)        │ │
│  │  ## Todos    — [ ] / [x] items with due date, added_by, etc     │ │
│  │  ## Events   — ISO datetime | title | added_by                  │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  APScheduler — scans events every 30min                         │ │
│  │  Sends WhatsApp reminders at 24h and 4h before each event       │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  logs/app.log  (daily rotation, 7 days retention)               │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────┐       ┌──────────────────────────────────┐
│  TAVILY API (cloud)       │       │  FAMILY WHATSAPP                 │
│  Web + image search       │       │  Research results, reminders,    │
│  (research intents only)  │       │  WhatsApp-choice deliveries      │
└───────────────────────────┘       └──────────────────────────────────┘
```
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     LOCAL MACHINE  (RTX 4070 Ti Super, 16GB VRAM)   │
│                                                                      │
│  ┌─────────────────────┐    ┌──────────────────────────────────────┐ │
│  │  faster-whisper     │    │  Ollama                              │ │
│  │  large-v3           │    │  (separate process, REST on :11434)  │ │
│  │  CUDA               │    │                                      │ │
│  │  int8_float16       │    │  ┌────────────────────────────────┐  │ │
│  │  ~1.5GB VRAM        │    │  │  Qwen 2.5:14b  Q4_K_M         │  │ │
│  │                     │    │  │  ~9-10GB VRAM                  │  │ │
│  │  Loaded at startup  │    │  │  Warmed up at startup          │  │ │
│  │  ~1-2s per call     │    │  └────────────────────────────────┘  │ │
│  └─────────────────────┘    └──────────────────────────────────────┘ │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │  family.md  (pipe-delimited markdown, filelock protected)     │   │
│  │  ## Todos    — [ ] / [x] items with due date, added_by, etc  │   │
│  │  ## Events   — ISO datetime | title | added_by               │   │
│  └───────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────┐
│     TAVILY API (cloud)       │
│  Web search, image search    │
│  (used only for research     │
│   intents)                   │
└──────────────────────────────┘
```

---

## Call Flow Detail

### Flow 1 — Greeting & Speech Capture

```
Phone call arrives at Twilio
        │
        ▼
POST /voice/incoming
  call_handler.handle_incoming()
        │
        ├─ Phone number in PHONE_TO_NAME? ──No──▶ TwiML: "Not registered. Goodbye." → hangup
        │
        └─ Yes ──▶ TwiML: <Say> "Hi {name}, this is Bianca..."
                          <Record maxLength=15 action=/voice/transcription>
                   Twilio speaks greeting, records caller, POSTs RecordingUrl
```

### Flow 2 — Transcription & Intent Routing (Filler + Async)

The transcription route returns a filler phrase **immediately** and computes the answer in parallel, eliminating perceived silence.

```
POST /voice/transcription  (RecordingUrl, RecordingSid, From)
  call_handler.handle_transcription()
        │
        ├─ Create Session (session_store) keyed by RecordingSid
        │   Session holds: asyncio.Event + result slot
        │
        ├─ asyncio.create_task(_compute_answer(...))  ← runs in background
        │
        └─ Return immediately:
           TwiML: <Say> "Sure, let me check that..." </Say>
                  <Redirect GET /voice/answer/{sid}>

                          ┌─────────────────────────────────┐
                          │  _compute_answer() (background) │
                          │                                 │
                          │  httpx download recording       │
                          │         ↓                       │
                          │  asyncio.to_thread(             │
                          │    whisper_service.transcribe)  │
                          │  confidence = exp(avg_logprob)  │
                          │         ↓                       │
                          │  confidence < 0.3 or < 2 words? │
                          │    → "Could you repeat?" result │
                          │         ↓                       │
                          │  asyncio.to_thread(             │
                          │    qwen.classify_intent)        │
                          │         ↓                       │
                          │  asyncio.to_thread(             │
                          │    intent_handler.route)        │
                          │         ↓                       │
                          │  session.result = TwiML         │
                          │  session.event.set()            │
                          └─────────────────────────────────┘

GET /voice/answer/{sid}                 ← Twilio calls this after filler plays
  await session.event (timeout=30s)
        │
        ├─ timeout ──▶ "Sorry, that took too long." → hangup
        └─ result ready ──▶ return TwiML answer → continue conversation

  intent_handler.route() returns:
        ├─ confidence < 0.6 or unknown ──▶ help message + <Record> (loop)
        ├─ goodbye ──────────────────────▶ farewell + hangup
        ├─ add_todo ─────────────────────▶ [Flow 3a]
        ├─ complete_todo ────────────────▶ [Flow 3b]
        ├─ add_event ────────────────────▶ [Flow 3c]
        ├─ query_tasks ──────────────────▶ [Flow 3d]
        ├─ query_events ─────────────────▶ [Flow 3d]
        └─ research / research_images ───▶ [Flow 3e]
```

**New component:** `services/session_store.py` — thin dict-based store mapping `RecordingSid → Session(event, result)`.

### Flow 3a — Add Todo

```
qwen.extract_todo(transcript)
  Qwen — prompts/todo_extract.txt
  Returns: { text, due (ISO date or null) }
        │
        ▼
markdown_service.append_todo(item)
  Sanitize text (strip | and \n)
  FileLock → write to family.md
        │
        ▼
TwiML: <Say> "Done. I've added '{text}'..." + <Record> (loop)
```

### Flow 3b — Complete Todo

```
markdown_service.read_todos()  →  pending todos list
        │
        ▼
qwen.match_todo(transcript, pending_todos)
  Qwen — prompts/todo_match.txt  (fuzzy semantic match)
  Returns: matched todo text or null
        │
        ├─ no match ──▶ "I couldn't find that. Be more specific?" + <Record>
        │
        ▼
markdown_service.complete_todo(matched_text)
  FileLock → rewrite - [ ] as - [x] + completed_at timestamp
        │
        ▼
TwiML: <Say> "Done. I've marked '{text}' as complete." + <Record> (loop)
```

### Flow 3c — Add Event

```
qwen.extract_event(transcript)
  Qwen — prompts/event_extract.txt
  Resolves relative dates ("next Friday") to absolute ISO datetime
  Returns: { title, event_datetime, human_readable }
        │
        ▼
markdown_service.append_event(item)
  FileLock → append to ## Events in family.md
        │
        ▼
TwiML: <Say> "Done. Added '{title}' on {human_readable}." + <Record> (loop)
```

### Flow 3d — Query Todos / Events

```
markdown_service.read_todos() or read_events()
  Parses all items from family.md (past + future for events)
        │
        ▼
qwen.answer_family_query(transcript, items, item_type)
  Qwen — prompts/family_query.txt
  Knows today's date and time
  Answers the specific question (not just a list dump)
  e.g. "when is the birthday?" → finds event, says "14 days from now"
        │
        ▼
TwiML: <Say> {answer} + <Record> (loop)
```

### Flow 3e — Research

```
intent.query present and ≥ 2 words?
        │
        ├─ No ──▶ "Could you give me more detail?" + <Record>
        │
        ▼
research_images intent?
        ├─ Yes ──▶ asyncio.create_task(_deliver_images) → Tavily → WhatsApp
        │           TwiML: "I'll send those images to your WhatsApp." + <Record>
        │
        ▼  (research intent)
qwen.quick_answer(query)   [asyncio.to_thread]
  Can Qwen answer from knowledge alone?
        │
        ├─ Yes ──▶ TwiML: <Say> answer + <Record> (loop)
        │
        └─ No ──▶ asyncio.create_task(_do_research(query))
                       _do_research returns (voice_text, whatsapp_text):
                         Tavily search → asyncio.gather(
                           qwen.voice_summarize_research,  ← 2-3 spoken sentences
                           qwen.synthesize_research,       ← full WhatsApp detail
                         )

                  await asyncio.wait_for(shield(task), timeout=10s)
                        │
                        ├─ Done in time ──▶ store whatsapp_text in _pending_whatsapp[wid]
                        │                   TwiML: <Say> voice_text (2-3 sentences)
                        │                          "Want the full details on WhatsApp?"
                        │                          <Record action=/voice/research-whatsapp-choice/{wid}>
                        │
                        └─ Timeout ──▶ store task in _pending[rid]
                                       TwiML: "Still researching... say 'wait' or 'WhatsApp'"
                                              <Record action=/voice/research-choice/{rid}>

POST /voice/research-choice/{rid}
  Download audio → Whisper → keyword match ("wait" / "WhatsApp")
        │
        ├─ "WhatsApp" ──▶ asyncio.create_task(_deliver_whatsapp_when_done)
        │                  TwiML: "I'll send to your WhatsApp shortly." + <Record>
        │
        └─ "wait" ──▶ await shield(task), timeout=15s
                            │
                            ├─ Done ──▶ store whatsapp_text in _pending_whatsapp[wid]
                            │           TwiML: <Say> voice_text + "Want full details on WhatsApp?"
                            │                  <Record action=/voice/research-whatsapp-choice/{wid}>
                            └─ Timeout ──▶ asyncio.create_task(_deliver_whatsapp_when_done)
                                           TwiML: "Taking longer, sending to WhatsApp." + <Record>

POST /voice/research-whatsapp-choice/{wid}
  Download audio → Whisper → keyword match ("yes" / "no")
        │
        ├─ "yes/sure/send" ──▶ twilio_service.send_whatsapp(full detail text)
        │                       TwiML: "Sent! Anything else?" + <Record>
        │
        └─ "no" ──▶ TwiML: "No problem. Anything else?" + <Record>
```

---

## Proactive Reminders

A background scheduler (`APScheduler AsyncIOScheduler`) starts at app startup and scans `family.md` every 30 minutes for upcoming events.

```
App startup (lifespan)
        │
        ▼
reminder_service.start()
  Schedules _scan_and_schedule() to run immediately + every 30 min
        │
        ▼
_scan_and_schedule()
  markdown_service.read_events(after=now, before=now+48h)
        │
        ▼
  For each event in window:
    For each offset in [24h, 4h]:
      remind_at = event_datetime - offset
      if remind_at <= now → skip (already past)
      job_id = f"reminder_{event_datetime}_{offset_minutes}"
      if job already exists in scheduler → skip (deduplication)
      scheduler.add_job(DateTrigger(run_date=remind_at), _send_reminder)

_send_reminder(title, label, event_dt)
  Formats message: "Reminder: *{title}* is in {label} ({human time})."
  Sends WhatsApp to every number in PHONE_TO_NAME
```

**Deduplication:** APScheduler job IDs are deterministic (`reminder_{iso_datetime}_{minutes}`). The 30-minute scan simply skips any job ID that already exists. One-off jobs are removed by APScheduler after they fire, so there is no risk of double-sending within a single server run.

**Restart behaviour:** The in-memory job store is cleared on restart. The scanner runs immediately on startup and reschedules any reminders whose `remind_at` is still in the future. Reminders already sent (whose `remind_at` is in the past) are naturally skipped by the `remind_at <= now` guard.

---

## Web Dashboard Flow

```
Browser → GET /dashboard
        markdown_service.read_all_data() → todos + events
        Annotate events with is_past, sort (pending/upcoming first)
        Resolve family_names from PHONE_TO_NAME for name dropdowns
        Jinja2 renders templates/dashboard.html
        Bootstrap 5 two-column layout, auto-refreshes every 30s

Dashboard is fully editable — all mutations are JSON POSTs, reload on success:

  POST /dashboard/add-todo        {text, due?, added_by}  → append_todo()
  POST /dashboard/complete-todo   {text}                  → complete_todo()
  POST /dashboard/delete-todo     {text}                  → delete_todo()

  POST /dashboard/add-event       {title, event_datetime, added_by}  → append_event()
  POST /dashboard/delete-event    {title, event_datetime}            → delete_event()

Edit (todo or event) is handled client-side:
  delete-old → add-new (two sequential API calls, single page reload)
```

---

## Browser Interface Flow

Family members open the browser interface on any device on the local network. The Talk and Hangman pages require HTTPS (use the ngrok URL) because `MediaRecorder` mic access requires a secure context.

```
Browser (Portal / phone / tablet)
        │
        ├─ GET /          → home.html  (card grid: Talk, Dashboard, Hangman)
        │
        ├─ GET /talk       → talk.html
        │   User taps mic → MediaRecorder captures audio
        │   User taps stop (or 2s silence auto-stop via AnalyserNode)
        │        │
        │        ▼
        │   POST /transcribe  (audio blob, webm/opus)
        │     asyncio.to_thread(whisper_service.transcribe, bytes, ".webm")
        │     → {transcript, confidence}
        │        │
        │        ▼
        │   POST /chat  {transcript, caller_name}
        │     chat_handler.handle_chat()
        │       classify_intent → route to handler
        │       research → Tavily + parallel Qwen summaries
        │       returns {speech, display, intent}
        │        │
        │        ▼
        │   speechSynthesis.speak(speech)     ← browser reads aloud
        │   Show display text in Full Answer panel (research only)
        │
        ├─ GET /dashboard  → dashboard.html
        │   Editable todos and events (add / complete / edit / delete)
        │   All mutations via JSON POST, reload on success
        │   Auto-refreshes every 30s
        │
        └─ GET /games/hangman → hangman.html
            POST /games/hangman/new    → new HangmanGame (random word)
            POST /games/hangman/guess  {session_id, guess}
              hangman_service.guess()
              Accepts: "letter A", "word elephant", bare single letter
              Strips punctuation + spoken prefixes before matching
              → {display_word, wrong_letters, figure, speech, won, lost}
            speechSynthesis.speak(speech)  ← reads result aloud
            Mic re-enabled only after speech finishes (onend callback)
```

---

## Component Map

```
family-assistant/
│
├── main.py                    FastAPI app, all routes, startup warmup, log setup
├── config.py                  Pydantic settings, .env loading, phone→name map
├── family.md                  Shared storage: todos + events (pipe-delimited markdown)
├── logs/                      Daily rotating logs (app.log, 7 days retention)
│
├── handlers/
│   ├── call_handler.py        Twilio webhooks → download audio → Whisper → classify
│   ├── chat_handler.py        Browser /chat → classify → route → return {speech, display}
│   ├── intent_handler.py      Routes IntentResult to correct sub-handler (phone path)
│   ├── todo_handler.py        add_todo, query_todos, complete_todo
│   ├── event_handler.py       add_event, query_events
│   ├── research_handler.py    quick_answer → Tavily → parallel voice+WhatsApp summaries
│   └── response_handler.py    TwiML builders: voice_gather, voice_say_then_gather, etc.
│
├── services/
│   ├── whisper_service.py     faster-whisper large-v3 CUDA, suffix param for webm/wav
│   ├── qwen.py                Ollama REST wrapper, all LLM calls, JSON extraction
│   ├── markdown_service.py    Read/write/parse/delete family.md with FileLock
│   ├── session_store.py       In-memory sessions (RecordingSid → asyncio.Event + result)
│   ├── tavily_service.py      Tavily web + image search with retry
│   ├── twilio_service.py      WhatsApp message + image sender
│   ├── reminder_service.py    APScheduler — 24h/4h WhatsApp reminders for events
│   └── hangman_service.py     Hangman game state, word list, guess logic
│
├── models/
│   └── schemas.py             Pydantic models: IntentResult, TodoItem, EventItem
│
├── prompts/
│   ├── intent_classify.txt    Few-shot intent classifier (9 intents)
│   ├── todo_extract.txt       Extract todo text + due date from transcript
│   ├── todo_match.txt         Fuzzy-match transcript to existing todo
│   ├── event_extract.txt      Extract event title + datetime from transcript
│   ├── family_query.txt       Answer natural language questions about todos/events
│   ├── quick_answer.txt       Decide if Qwen can answer from knowledge vs web search
│   ├── research_voice.txt     2-3 sentence spoken summary of search results
│   └── research_synthesize.txt  Full Markdown summary for WhatsApp / browser display
│
└── templates/
    ├── home.html              Landing page — Bootstrap card grid
    ├── talk.html              Browser voice interface — MediaRecorder + Whisper STT
    ├── hangman.html           Voice hangman game — silence detection, prefix enforcement
    └── dashboard.html         Editable family dashboard — Bootstrap 5, vanilla JS
```

---

## Technology Stack

| Layer | Technology | Why |
|---|---|---|
| Phone calls | Twilio (inbound PSTN) | Handles carrier complexity, webhooks, TTS |
| Phone TTS | AWS Polly via Twilio (`Polly.Joanna`) | Natural voice, no extra integration |
| Browser mic | `MediaRecorder` API | Works on all modern browsers, no plugins |
| Browser TTS | `speechSynthesis` API | Built-in, no server round-trip |
| Speech-to-text | faster-whisper `large-v3` on CUDA | Free, accurate, used for both phone and browser |
| LLM | Qwen 2.5:14b via Ollama | Strong reasoning, runs fully locally |
| Web search | Tavily API | Clean results API, image search support |
| Messaging | Twilio WhatsApp API | Async research + reminder delivery |
| Storage | Markdown file (`family.md`) | Human-readable, editable, no DB setup |
| File locking | `filelock` | Prevents concurrent write corruption |
| Scheduler | APScheduler `AsyncIOScheduler` | Proactive reminders without Celery/Redis |
| Backend | FastAPI + uvicorn | Async, fast, minimal boilerplate |
| Templates | Jinja2 + Bootstrap 5 | No build step, zero JS framework needed |
| Logging | `TimedRotatingFileHandler` | Daily log files, 7-day retention |
| Tunnel (dev) | ngrok | Twilio webhooks + HTTPS for browser mic |

---

## GPU Memory Layout (RTX 4070 Ti Super — 16GB)

```
┌──────────────────────────────────────────────────┐
│                  16 GB VRAM                      │
│                                                  │
│  ┌────────────────────────┐                      │
│  │  Qwen 2.5:14b Q4_K_M  │  ~9-10 GB            │
│  │  (Ollama, always hot)  │                      │
│  └────────────────────────┘                      │
│  ┌──────────────┐                                │
│  │  Whisper     │  ~1.5 GB                       │
│  │  large-v3    │  (always loaded, not active    │
│  │  int8_float16│   during Qwen inference)       │
│  └──────────────┘                                │
│  ┌──────────────────────────────┐                │
│  │  Free headroom  ~4-5 GB     │                │
│  └──────────────────────────────┘                │
└──────────────────────────────────────────────────┘

Note: Whisper and Qwen never run at the same time —
Whisper transcribes first, then Qwen processes.
```

---

## Data Schema (family.md)

```markdown
# Family Assistant

## Todos

- [ ] Buy groceries | due: 2026-04-03 | added_by: Alice | added_at: 2026-03-29T10:15:00
- [x] Renew passport | due: none | added_by: Bob | added_at: 2026-03-20T09:00:00 | completed_at: 2026-03-25T12:00:00

## Events

- 2026-04-10T14:00:00 | Dentist appointment | added_by: Alice | added_at: 2026-03-29T10:20:00
- 2026-05-01T00:00:00 | Family vacation starts | added_by: Bob | added_at: 2026-03-01T08:00:00
```

**Rules:**
- Todos: `- [ ]` pending, `- [x]` complete. Fields pipe-delimited. Text must not contain `|` (sanitized on write).
- Events: ISO datetime first field, then title + metadata. Sorted by insertion order.
- File is protected by `filelock` — one writer at a time, 5-second timeout.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `<Record>` + Whisper instead of `<Gather input="speech">` | Free, more accurate STT especially for names and natural speech |
| Two Qwen calls per request (classify → extract) | More reliable than one combined call; smaller focused prompts |
| `quick_answer` check before Tavily | Avoids paid API call for simple knowledge questions |
| FastAPI `BackgroundTasks` for research | No Redis/Celery needed; delivers WhatsApp while call is already ended |
| Markdown file instead of database | Human-readable, manually editable, zero infrastructure |
| `filelock` for writes | Prevents corruption from rare concurrent calls; family-scale volume is fine |
| Whisper loaded at startup | Eliminates cold-start delay on first call |
| Qwen warmed up at startup via dummy `_chat("hi")` | Ollama lazy-loads model; warmup ensures GPU is ready |
| Few-shot examples in intent classifier | Significantly reduces misclassifications vs zero-shot |
