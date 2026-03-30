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

---

## Open Questions / Future Ideas
- Add a "complete todo" voice command ("mark buy groceries as done")
- Scheduled reminders: outbound WhatsApp at event time
- Multi-language support (Qwen handles this well)
- family.md sync via Dropbox/git for backup
- Presentation generation (python-pptx) and WhatsApp as document
