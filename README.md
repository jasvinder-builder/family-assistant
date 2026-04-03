# Bianca — Voice Family Assistant

A voice-based AI assistant for families. Call a phone number, speak naturally, or open the browser interface — Bianca manages todos, events, and answers research questions. All AI runs locally on GPU. No cloud AI services.

**Full architecture and call flow:** see [`ARCHITECTURE.md`](ARCHITECTURE.md)

---

## UI

```
┌─────────────────────────────────────────────────────┐
│  👋 Hi, I'm Bianca                                  │
│  Your family assistant — voice, todos, events,      │
│  and games                                          │
│                                                     │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────┐ │
│  │   🎙️          │ │   📋          │ │   🎮      │ │
│  │ Talk to Bianca│ │   Family      │ │  Hangman  │ │
│  │               │ │  Dashboard    │ │           │ │
│  │ Ask questions,│ │ View & manage │ │ Guess the │ │
│  │ add todos and │ │ todos and     │ │ word with │ │
│  │ events, search│ │ events        │ │ your voice│ │
│  └───────────────┘ └───────────────┘ └───────────┘ │
└─────────────────────────────────────────────────────┘

Talk to Bianca (/talk)          Family Dashboard (/dashboard)
┌──────────────────────────┐    ┌──────────────────────────────┐
│ You are: [Jasvinder ▾]   │    │ Todos          │ Events       │
│                          │    │ ─────────────────────────────│
│ Jasvinder                │    │ □ Buy groceries│ ✦ Dentist   │
│ ╔══════════════════════╗ │    │   Due Apr 3    │   Fri Apr 10│
│ ║ What's the weather   ║ │    │ □ Call plumber │ ✦ Gymnastics│
│ ║ in Sunnyvale?        ║ │    │ ──────────────────────────── │
│ ╚══════════════════════╝ │    │ [✓][✏️][🗑]    │   [✏️][🗑]  │
│                          │    │                │             │
│ Bianca                   │    │          [+ Add]│       [+Add]│
│ ╔══════════════════════╗ │    └──────────────────────────────┘
│ ║ It's partly cloudy   ║ │
│ ║ and around 62°F...   ║ │    Hangman (/games/hangman)
│ ╚══════════════════════╝ │    ┌──────────────────────────────┐
│  ┌─ Full Answer ───────┐ │    │   +---+                      │
│  │ Sunnyvale, CA today │ │    │   |   |   _ _ _ _ _ _ _ _   │
│  │ shows partly cloudy │ │    │   O   |                      │
│  │ skies with a high...│ │    │   |   |   Wrong: X Z         │
│  └─────────────────────┘ │    │  /|   |                      │
│                          │    │       |   💬 No X. 4 guesses  │
│  Auto-stops after 2s     │    │ say "letter A" or "word X"   │
│  Just speak — no tap     │    │ Just speak — no tap          │
└──────────────────────────┘    └──────────────────────────────┘
```

---

## What it does

**Via phone call (Twilio):**
- **Add todos** — "Add a todo to buy milk"
- **Complete todos** — "Mark buy milk as done"
- **Add events** — "Add dentist appointment next Tuesday at 2pm"
- **Query family data** — "What events do we have this week?"
- **Research** — Short spoken answer, full details sent to WhatsApp on request
- **Image search** — Results sent to WhatsApp

**Via browser (any device on your network):**
- **Talk to Bianca** — same features as phone; mic auto-detects your voice (no tap needed) using Silero VAD running locally in the browser
- **Family Dashboard** — view, add, edit, and delete todos and events
- **Hangman** — voice-controlled word game for kids; just speak your guess, no tap required
- **Times Tables** — voice multiplication practice; Bianca asks, kid answers out loud
- **Tell the Time** — 4-option clock reading game with pure SVG clock faces
- **Knowledge Quiz** — pick a subject and grade; Qwen generates 10 tailored questions on the fly
- **Cameras** — live view of any RTSP stream or local video file streamed via WebSocket (works through Cloudflare Tunnel); YOLOv8 + ByteTrack detects and tracks persons, vehicles (bicycle, car, motorcycle, bus, truck), and animals (bird, cat, dog); CLIP matches them against user-defined natural-language queries (e.g. "small child", "person in red", "cat on sofa"); configurable crop padding expands the bounding box to give CLIP surrounding context for relational queries; matched events are logged with a thumbnail crop

**Proactive:**
- **Event reminders** — WhatsApp reminders sent to all family members 24h and 4h before events

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
- **[cloudflared](https://github.com/cloudflare/cloudflared)** — Cloudflare Tunnel; exposes your local server to Twilio webhooks and provides HTTPS for browser mic access. Free with no bandwidth limits.
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
TWILIO_PHONE_NUMBER=+15005550006           # your Twilio number
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886 # Twilio sandbox number

TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxx

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:14b

FAMILY_MD_PATH=./family.md
WHISPER_MODEL_SIZE=large-v3               # use "base" for CPU-only machines

# JSON mapping of E.164 phone numbers to names
PHONE_TO_NAME={"+447911123456": "Alice", "+447911987654": "Bob"}
```

`PHONE_TO_NAME` controls who can call Bianca. Only registered numbers are answered.

### 4. Create your family data file

```bash
cp family.md.example family.md
```

Bianca reads from and writes to this file. It is excluded from git — your family's data stays local.

### 5. Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` makes the server reachable from other devices on your network (phones, Portal, etc.).

On first startup Whisper and Qwen load into GPU memory (30–60 seconds). Subsequent requests are fast.

### 6. Expose to Twilio and get HTTPS with Cloudflare Tunnel

```bash
cloudflared tunnel --url http://localhost:8000
```

Copy the `https://` forwarding URL printed in the output (e.g. `https://some-name.trycloudflare.com`).

- **Twilio:** set your phone number's Voice webhook to `https://some-name.trycloudflare.com/voice/incoming` (POST)
- **Browser mic:** the Talk and Hangman pages require HTTPS — use the cloudflared URL, not the local IP
- **Note:** quick tunnel URLs change on each restart. For a stable persistent URL, log in with `cloudflared tunnel login` and create a named tunnel.

---

## Usage

### Phone
Call your Twilio number. Bianca greets you by name and listens. Speak naturally — she handles the rest. Research results too long for voice are sent to your WhatsApp.

### Browser (same WiFi network)
Open on any phone, tablet, or smart display on your network:

| Page | Local (HTTP) | HTTPS required |
|---|---|---|
| Home | `http://<your-ip>:8000/` | No |
| Dashboard | `http://<your-ip>:8000/dashboard` | No |
| Talk to Bianca | `https://<cloudflared-url>/talk` | **Yes** |
| Hangman | `https://<cloudflared-url>/games/hangman` | **Yes** |

Find your local IP: `ip addr show | grep "inet " | grep -v 127.0.0.1`

### Hangman voice commands
- `"letter A"` — guess a letter
- `"word elephant"` — guess the whole word
- `"new game"` — start a fresh game

---

## Project structure

```
family-assistant/
├── main.py                   # FastAPI app, all routes, startup warmup, logging
├── config.py                 # Settings, .env loading, phone→name mapping
├── family.md                 # Shared storage (not in git — copy from family.md.example)
├── .env                      # Credentials (not in git)
├── .env.example              # Template
├── family.md.example         # Template for family.md
├── ARCHITECTURE.md           # Full technical architecture and call flow diagrams
├── logs/                     # Daily rotating log files (not in git)
│
├── handlers/
│   ├── call_handler.py       # Twilio webhooks, async filler+compute pattern
│   ├── chat_handler.py       # Browser /chat endpoint — text in, structured JSON out
│   ├── intent_handler.py     # Routes intents to sub-handlers (phone path)
│   ├── todo_handler.py       # Add and complete todos
│   ├── event_handler.py      # Add events
│   ├── research_handler.py   # Web research, voice summary, WhatsApp delivery
│   └── response_handler.py   # TwiML builder helpers
│
├── services/
│   ├── qwen.py               # Ollama REST wrapper and all LLM prompt runners
│   ├── whisper_service.py    # faster-whisper STT, loaded at startup on GPU
│   ├── twilio_service.py     # TwiML and WhatsApp sender
│   ├── tavily_service.py     # Web and image search with retry
│   ├── markdown_service.py   # Read/write/parse family.md with filelock
│   ├── session_store.py      # In-memory sessions for async phone call flow
│   ├── reminder_service.py   # APScheduler — proactive WhatsApp event reminders
│   ├── hangman_service.py    # Hangman game logic and in-memory game state
│   ├── camera_service.py     # RTSP/file VideoCapture → MJPEG generator; shares frames with scene_service
│   └── scene_service.py      # YOLOv8n + ByteTrack + CLIP — person/vehicle/animal detection, tracking, query matching, event log
│
├── models/
│   └── schemas.py            # Pydantic models: IntentResult, TodoItem, EventItem
│
├── prompts/                  # LLM prompt templates (plain text, injected at runtime)
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
    ├── home.html             # Landing page — card grid linking to all features
    ├── talk.html             # Browser voice interface (MediaRecorder + Whisper)
    ├── hangman.html          # Voice hangman game
    ├── dashboard.html        # Family todos and events — fully editable
    ├── games.html            # Games hub (Hangman, Times Tables, Clock, Quiz)
    └── cameras.html          # RTSP live view + AI scene detection UI
```
