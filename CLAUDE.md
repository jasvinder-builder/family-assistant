# Bianca — Family Assistant: Claude Instructions

## Project Objective

Bianca is a voice-based AI family assistant. Family members call a Twilio phone number, speak naturally, and Bianca handles todos, events, research, and general questions. Research results are delivered via WhatsApp. A web dashboard shows the family's shared data. All AI (Whisper STT + Qwen LLM) runs locally on GPU — no cloud AI services.

## Architecture Rule

**Always update `ARCHITECTURE.md` after any change that affects:**
- Call flow or routing logic
- New or modified API endpoints
- New services or handlers
- Changes to how models are used (STT, LLM)
- Storage schema changes
- New external integrations

Update the relevant section(s) only — do not rewrite the whole file unless the change is sweeping.

## Key Facts

- **Voice:** Twilio inbound calls → `<Record>` → POST RecordingUrl to `/voice/transcription`
- **STT:** faster-whisper `large-v3` on CUDA, loaded at startup
- **LLM:** Qwen 2.5:14b via Ollama (localhost:11434), warmed up at startup
- **Storage:** `family.md` — pipe-delimited markdown, protected by `filelock`
- **Dashboard:** `GET /dashboard` — Bootstrap 5, Jinja2, auto-refreshes every 30s
- **GPU:** RTX 4070 Ti Super (16GB VRAM) — Whisper ~1.5GB + Qwen ~10GB, ~4.5GB free
- **Filler UX:** `/voice/transcription` returns filler phrase immediately, computation runs async, Twilio redirects to `/voice/answer/{sid}` for the real answer

## Dev Workflow

```bash
# Start
source .venv/bin/activate
uvicorn main:app --port 8000

# Expose to Twilio
ngrok http 8000
# Set Twilio webhook to: https://<ngrok-url>/voice/incoming

# Dashboard
http://localhost:8000/dashboard
```

## Environment Variables (.env)

```
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_PHONE_NUMBER=
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
TAVILY_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:14b
FAMILY_MD_PATH=./family.md
PHONE_TO_NAME={"+"447911123456":"Alice"}
WHISPER_MODEL_SIZE=large-v3
```
