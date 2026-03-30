# Bianca — Family Assistant: Architecture

## Overview

Bianca is a voice-based family productivity assistant. Family members call a Twilio phone number, speak naturally, and Bianca handles todos, events, research, and general questions. Research results are delivered via WhatsApp. A web dashboard provides a visual view of the family's shared data.

All AI runs locally on GPU. No cloud AI services are used.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CALLER'S PHONE                             │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ PSTN call
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         TWILIO CLOUD                                │
│                                                                     │
│   1. Receives inbound call                                          │
│   2. Sends webhook → POST /voice/incoming                           │
│   3. Plays TwiML response (Bianca's greeting via Polly.Joanna TTS)  │
│   4. Records caller's speech (up to 15s)                            │
│   5. Sends webhook → POST /voice/transcription  (RecordingUrl)      │
│   6. Plays TwiML response (Bianca's answer via Polly.Joanna TTS)    │
│   7. Records next question → back to step 5 (loop)                 │
│   8. On goodbye intent → plays farewell → hangs up                 │
└──────────┬───────────────────────────────────────┬──────────────────┘
           │ webhooks (HTTP POST)                  │ WhatsApp API
           ▼                                       ▼
┌──────────────────────────────┐      ┌────────────────────────────────┐
│     FASTAPI APPLICATION      │      │      CALLER'S WHATSAPP         │
│     (uvicorn, port 8000)     │      │  (research results delivered   │
│     exposed via ngrok        │      │   as async background task)    │
│                              │      └────────────────────────────────┘
│  Routes:                     │
│  POST /voice/incoming        │
│  POST /voice/transcription   │
│  POST /voice/research-       │
│       choice/{rid}           │
│  POST /voice/research-       │
│       whatsapp-choice/{wid}  │
│  GET  /voice/answer/{sid}    │
│  GET  /dashboard             │
│  POST /dashboard/complete-   │
│       todo                   │
│  GET  /health                │
└──────────┬───────────────────┘
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

## Web Dashboard Flow

```
Browser → GET /dashboard
                │
                ▼
        markdown_service.read_all_data()
        Reads family.md → todos + events
                │
                ▼
        Annotate events with is_past (compare event_datetime to now)
        Sort: todos (pending first), events (upcoming first, past last)
                │
                ▼
        Jinja2 renders templates/dashboard.html
        Bootstrap 5 — two-column layout
        Auto-refreshes every 30s via JS setTimeout

Browser → POST /dashboard/complete-todo  { "text": "buy milk" }
                │
                ▼
        markdown_service.complete_todo(text)
        FileLock → rewrite - [ ] as - [x] in family.md
```

---

## Component Map

```
family-assistant/
│
├── main.py                    FastAPI app, routes, startup warmup
├── config.py                  Pydantic settings, .env loading, phone→name map
├── family.md                  Shared storage: todos + events (pipe-delimited markdown)
│
├── handlers/
│   ├── call_handler.py        Twilio webhooks → download audio → Whisper → classify
│   ├── intent_handler.py      Routes IntentResult to correct sub-handler
│   ├── todo_handler.py        add_todo, query_todos, complete_todo
│   ├── event_handler.py       add_event, query_events
│   ├── research_handler.py    quick_answer check → Tavily search → WhatsApp dispatch
│   └── response_handler.py    TwiML builders: voice_gather, voice_say_then_gather, voice_say_hangup
│
├── services/
│   ├── whisper_service.py     faster-whisper large-v3 on CUDA (loaded at startup)
│   ├── qwen.py                Ollama REST wrapper, all LLM calls, JSON extraction
│   ├── markdown_service.py    Read/write/parse family.md with FileLock
│   ├── session_store.py       In-memory sessions (RecordingSid → asyncio.Event + result)
│   ├── tavily_service.py      Tavily web + image search
│   └── twilio_service.py      WhatsApp message + image sender
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
│   ├── quick_answer.txt       Decide if Qwen can answer from knowledge vs needs search
│   └── research_synthesize.txt  Summarize Tavily results for WhatsApp
│
└── templates/
    └── dashboard.html         Bootstrap 5 dashboard, vanilla JS, auto-refresh
```

---

## Technology Stack

| Layer | Technology | Why |
|---|---|---|
| Phone calls | Twilio (inbound PSTN) | Handles carrier complexity, webhooks, TTS |
| Text-to-speech | AWS Polly via Twilio (`Polly.Joanna`) | Natural voice, no extra integration |
| Speech-to-text | faster-whisper `large-v3` on CUDA | Free, best open-source STT accuracy |
| LLM | Qwen 2.5:14b via Ollama | Strong multilingual reasoning, runs locally |
| Web search | Tavily API | Clean search results API, image support |
| Messaging | Twilio WhatsApp API | Delivers research results asynchronously |
| Storage | Markdown file (`family.md`) | Human-readable, editable, no DB setup |
| File locking | `filelock` | Prevents concurrent write corruption |
| Backend | FastAPI + uvicorn | Async, fast, minimal boilerplate |
| Dashboard | Jinja2 + Bootstrap 5 | No build step, zero JS dependencies |
| Tunnel (dev) | ngrok | Exposes local server to Twilio webhooks |

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
