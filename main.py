import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, Form, Response, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import httpx
from handlers.call_handler import handle_incoming, handle_transcription
from handlers.research_handler import handle_research_choice, handle_research_whatsapp_choice
from handlers.response_handler import voice_say_hangup
from services import whisper_service
from config import settings as app_settings
from services import markdown_service, session_store
from services.qwen import _chat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up Ollama so Qwen is loaded into GPU before the first call
    logger.info("Warming up Qwen via Ollama...")
    try:
        _chat("hi")
        logger.info("Qwen warm-up done.")
    except Exception:
        logger.warning("Qwen warm-up failed — Ollama may not be running yet.")
    yield


app = FastAPI(title="Family Assistant", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


@app.post("/voice/incoming")
async def voice_incoming(From: str = Form(default="")):
    twiml = handle_incoming(From=From)
    return Response(content=twiml, media_type="application/xml")


@app.post("/voice/transcription")
async def voice_transcription(
    From: str = Form(default=""),
    RecordingUrl: str = Form(default=""),
    RecordingSid: str = Form(default=""),
):
    twiml = handle_transcription(From=From, RecordingUrl=RecordingUrl, RecordingSid=RecordingSid)
    return Response(content=twiml, media_type="application/xml")


@app.post("/voice/research-choice/{rid}")
async def voice_research_choice(
    rid: str,
    RecordingUrl: str = Form(default=""),
):
    transcript = ""
    if RecordingUrl:
        try:
            async with httpx.AsyncClient() as client:
                audio_resp = await client.get(
                    RecordingUrl,
                    auth=(app_settings.twilio_account_sid, app_settings.twilio_auth_token),
                    timeout=10,
                    follow_redirects=True,
                )
            transcript, _ = await asyncio.to_thread(whisper_service.transcribe, audio_resp.content)
        except Exception:
            logger.warning("Could not transcribe research choice recording for rid=%s", rid)

    twiml = await handle_research_choice(rid, transcript)
    return Response(content=twiml, media_type="application/xml")


@app.post("/voice/research-whatsapp-choice/{wid}")
async def voice_research_whatsapp_choice(
    wid: str,
    RecordingUrl: str = Form(default=""),
):
    transcript = ""
    if RecordingUrl:
        try:
            async with httpx.AsyncClient() as client:
                audio_resp = await client.get(
                    RecordingUrl,
                    auth=(app_settings.twilio_account_sid, app_settings.twilio_auth_token),
                    timeout=10,
                    follow_redirects=True,
                )
            transcript, _ = await asyncio.to_thread(whisper_service.transcribe, audio_resp.content)
        except Exception:
            logger.warning("Could not transcribe WhatsApp choice recording for wid=%s", wid)

    twiml = await handle_research_whatsapp_choice(wid, transcript)
    return Response(content=twiml, media_type="application/xml")


@app.get("/voice/answer/{sid}")
async def voice_answer(sid: str):
    session = session_store.get(sid)
    if not session:
        twiml = voice_say_hangup("Sorry, something went wrong. Please call again.")
        return Response(content=twiml, media_type="application/xml")
    try:
        await asyncio.wait_for(session.event.wait(), timeout=30)
        twiml = session.result or voice_say_hangup("Sorry, I couldn't get an answer. Please try again.")
    except asyncio.TimeoutError:
        logger.warning("Answer timeout for session %s", sid)
        twiml = voice_say_hangup("Sorry, that took too long. Please call again.")
    finally:
        session_store.delete(sid)
    return Response(content=twiml, media_type="application/xml")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    data = markdown_service.read_all_data()
    now = datetime.now()

    todos = sorted(data["todos"], key=lambda t: t.completed)  # pending first

    events = []
    for e in data["events"]:
        try:
            dt = datetime.fromisoformat(e.event_datetime)
        except ValueError:
            dt = now
        events.append({"title": e.title, "human_readable": e.human_readable,
                        "added_by": e.added_by, "is_past": dt < now})
    events.sort(key=lambda e: (e["is_past"], e["human_readable"]))

    pending_count = sum(1 for t in todos if not t.completed)
    upcoming_count = sum(1 for e in events if not e["is_past"])

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "todos": todos,
            "events": events,
            "pending_count": pending_count,
            "upcoming_count": upcoming_count,
        },
    )


@app.post("/dashboard/complete-todo")
async def dashboard_complete_todo(payload: dict):
    text = payload.get("text", "")
    if not text:
        return JSONResponse({"error": "missing text"}, status_code=400)
    success = markdown_service.complete_todo(text)
    if success:
        return JSONResponse({"ok": True})
    return JSONResponse({"error": "not found"}, status_code=404)


@app.get("/health")
async def health():
    return {"status": "ok"}
