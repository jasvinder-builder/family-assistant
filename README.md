# Bianca — Voice Family Assistant

A voice-based AI assistant for families. Call a phone number, speak naturally, and Bianca manages your shared todos, events, and answers research questions. Results can be delivered to WhatsApp. Everything runs locally — no cloud AI.

**Full architecture and call flow:** see [`ARCHITECTURE.md`](ARCHITECTURE.md)

---

## What it does

- **Add todos** — "Add a todo to buy milk"
- **Complete todos** — "Mark buy milk as done"
- **Add events** — "Add dentist appointment next Tuesday at 2pm"
- **Query family data** — "What events do we have this week?", "What's on my list?"
- **Research** — Answers from Qwen's knowledge, or falls back to web search (Tavily). Short spoken summary on the call, full detail sent to WhatsApp on request.
- **Image search** — Results sent to WhatsApp
- **Web dashboard** — View and complete todos/events at `http://localhost:8000/dashboard`

---

## Prerequisites

### Accounts and API keys

| Service | Purpose | Free tier |
|---|---|---|
| [Twilio](https://www.twilio.com) | Inbound voice calls, WhatsApp messaging | Yes (trial number) |
| [Tavily](https://tavily.com) | Web and image search | Yes |

**Twilio setup:**
1. Create an account and buy a phone number with Voice capability
2. Note your Account SID and Auth Token from the Console dashboard
3. For WhatsApp: join the [Twilio Sandbox for WhatsApp](https://console.twilio.com/us1/develop/sms/try-it-out/whatsapp-learn) and note the sandbox number

### Local software

- **Python 3.11+**
- **[Ollama](https://ollama.com)** — runs the Qwen LLM locally
- **[ngrok](https://ngrok.com)** — exposes your local server to Twilio's webhooks
- **NVIDIA GPU** (recommended) — Whisper large-v3 and Qwen 2.5:14b together need ~12GB VRAM. CPU-only works but is significantly slower.
  - If using CPU: set `WHISPER_MODEL_SIZE=base` in `.env` for faster startup

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/jasvinder-builder/family-assistant.git
cd family-assistant
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Pull the Qwen model via Ollama

```bash
ollama pull qwen2.5:14b
```

The model (~9GB) downloads once and is cached locally. Start Ollama before running the app:

```bash
ollama serve
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxx
TWILIO_PHONE_NUMBER=+15005550006          # your Twilio number
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886 # Twilio sandbox number

TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxx

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:14b

FAMILY_MD_PATH=./family.md
WHISPER_MODEL_SIZE=large-v3              # use "base" for CPU-only machines

# JSON mapping of E.164 phone numbers to names
PHONE_TO_NAME={"+447911123456": "Alice", "+447911987654": "Bob"}
```

The `PHONE_TO_NAME` map controls who can use the assistant. Only numbers listed here will be answered; all others hear "Not registered" and are disconnected.

### 4. Create your family data file

```bash
cp family.md.example family.md
```

Bianca reads from and writes to this file. It is excluded from git — your family's data stays local.

### 5. Start the server

```bash
uvicorn main:app --port 8000
```

On first startup, Whisper and Qwen both load into memory (30–60 seconds). Subsequent calls are fast.

### 6. Expose to Twilio with ngrok

```bash
ngrok http 8000
```

Copy the `https://` forwarding URL (e.g. `https://abc123.ngrok.io`).

In the [Twilio Console](https://console.twilio.com), go to your phone number's Voice configuration and set:

- **A call comes in** → Webhook → `https://abc123.ngrok.io/voice/incoming`
- Method: `HTTP POST`

---

## Usage

Call your Twilio number. Bianca greets you by name (from `PHONE_TO_NAME`) and listens for a command.

**Dashboard:** open `http://localhost:8000/dashboard` in a browser to see todos and events, and mark todos as done.

---

## Project structure

```
family-assistant/
├── main.py                   # FastAPI app and route definitions
├── config.py                 # Settings, env loading, phone→name mapping
├── family.md                 # Shared storage — created from family.md.example (not in git)
├── .env                      # Your credentials (not in git)
├── .env.example              # Template
├── family.md.example         # Template for family.md
├── ARCHITECTURE.md           # Full technical architecture and call flow diagrams
│
├── handlers/
│   ├── call_handler.py       # Twilio webhook entry, async filler+compute pattern
│   ├── intent_handler.py     # Routes intents to sub-handlers
│   ├── todo_handler.py       # Add and complete todos
│   ├── event_handler.py      # Add events
│   ├── research_handler.py   # Web research, voice summary, WhatsApp delivery
│   └── response_handler.py   # TwiML builder helpers
│
├── services/
│   ├── qwen.py               # Ollama REST wrapper and prompt runners
│   ├── whisper_service.py    # faster-whisper STT (loaded at startup)
│   ├── twilio_service.py     # TwiML and WhatsApp sender
│   ├── tavily_service.py     # Web and image search
│   ├── markdown_service.py   # Read/write family.md with filelock
│   └── session_store.py      # In-memory session state for async call flow
│
├── models/
│   └── schemas.py            # Pydantic models: IntentResult, TodoItem, EventItem
│
├── prompts/                  # LLM prompt templates
│   ├── intent_classify.txt
│   ├── todo_extract.txt
│   ├── todo_match.txt
│   ├── event_extract.txt
│   ├── family_query.txt
│   ├── quick_answer.txt
│   ├── research_voice.txt
│   └── research_synthesize.txt
│
└── templates/
    └── dashboard.html        # Bootstrap 5 web dashboard
```
