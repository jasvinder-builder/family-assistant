import asyncio
import json
import logging
import logging.handlers
import os
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, Form, Response, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import httpx
from handlers.call_handler import handle_incoming, handle_transcription
from handlers.research_handler import handle_research_choice, handle_research_whatsapp_choice
from handlers.response_handler import voice_say_hangup
from handlers import chat_handler
from services import whisper_service, reminder_service
from services import hangman_service
from services import camera_service
from config import settings as app_settings
from services import markdown_service, session_store
from services import qwen
from services.qwen import _chat
from models.schemas import TodoItem, EventItem

os.makedirs("logs", exist_ok=True)
_log_fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s")

_file_handler = logging.handlers.TimedRotatingFileHandler(
    "logs/app.log",
    when="midnight",
    interval=1,
    backupCount=7,
    encoding="utf-8",
)
_file_handler.setFormatter(_log_fmt)
_file_handler.suffix = "%Y-%m-%d"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(), _file_handler],
)
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

    reminder_service.start()
    yield
    reminder_service.stop()


app = FastAPI(title="Family Assistant", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="home.html", context={})


@app.get("/talk", response_class=HTMLResponse)
async def talk(request: Request):
    try:
        family_names = sorted(set(json.loads(app_settings.phone_to_name).values()))
    except Exception:
        family_names = []
    return templates.TemplateResponse(request=request, name="talk.html", context={"family_names": family_names})


@app.get("/games", response_class=HTMLResponse)
async def games_hub(request: Request):
    return templates.TemplateResponse(request=request, name="games.html", context={})


@app.get("/cameras", response_class=HTMLResponse)
async def cameras_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="cameras.html",
        context={"current_url": camera_service.get_stream_url()},
    )


@app.post("/cameras/set-stream")
async def cameras_set_stream(payload: dict):
    url = payload.get("url", "").strip()
    if url and not url.startswith("rtsp://"):
        return JSONResponse({"error": "URL must start with rtsp://"}, status_code=400)
    camera_service.set_stream_url(url)
    return JSONResponse({"ok": True})


@app.get("/cameras/stream")
async def cameras_stream():
    url = camera_service.get_stream_url()
    if not url:
        return JSONResponse({"error": "No stream configured"}, status_code=404)
    return StreamingResponse(
        camera_service.mjpeg_generator(url),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Receive browser audio blob (webm or wav), transcribe via local Whisper."""
    audio_bytes = await audio.read()
    filename = audio.filename or "recording.webm"
    suffix = os.path.splitext(filename)[1].lower()
    if suffix not in (".webm", ".wav", ".mp3", ".ogg", ".m4a"):
        suffix = ".webm"
    transcript, confidence = await asyncio.to_thread(
        whisper_service.transcribe, audio_bytes, suffix
    )
    return JSONResponse({"transcript": transcript, "confidence": round(confidence, 2)})


@app.post("/chat")
async def chat(payload: dict):
    transcript = payload.get("transcript", "").strip()
    caller_name = payload.get("caller_name", "Family").strip()
    result = await chat_handler.handle_chat(transcript, caller_name)
    return JSONResponse(result)


@app.get("/games/hangman", response_class=HTMLResponse)
async def hangman_page(request: Request):
    return templates.TemplateResponse(request=request, name="hangman.html", context={})


@app.get("/games/multiply", response_class=HTMLResponse)
async def multiply_page(request: Request):
    return templates.TemplateResponse(request=request, name="multiply.html", context={})


@app.get("/games/clock", response_class=HTMLResponse)
async def clock_page(request: Request):
    return templates.TemplateResponse(request=request, name="clock.html", context={})


@app.get("/games/quiz", response_class=HTMLResponse)
async def quiz_page(request: Request):
    return templates.TemplateResponse(request=request, name="quiz.html", context={})


@app.post("/games/resolve-answer")
async def resolve_answer(payload: dict):
    transcript = payload.get("transcript", "").strip()
    options = payload.get("options", [])
    if not transcript or len(options) != 4:
        return JSONResponse({"index": None})
    try:
        idx = await asyncio.to_thread(qwen.resolve_answer, transcript, options)
        return JSONResponse({"index": idx})
    except Exception as e:
        logger.error("resolve_answer failed: %s", e)
        return JSONResponse({"index": None})


@app.post("/games/quiz/generate")
async def quiz_generate(payload: dict):
    subject = payload.get("subject", "").strip()
    grade = payload.get("grade")
    if not subject or grade is None:
        return JSONResponse({"error": "subject and grade are required"}, status_code=400)
    try:
        grade = int(grade)
        if not 1 <= grade <= 8:
            raise ValueError
    except (TypeError, ValueError):
        return JSONResponse({"error": "grade must be 1–8"}, status_code=400)
    try:
        questions = await asyncio.to_thread(qwen.generate_quiz, subject, grade)
        return JSONResponse({"questions": questions})
    except Exception as e:
        logger.error("Quiz generation failed: %s", e)
        return JSONResponse({"error": "Could not generate quiz. Please try again."}, status_code=500)


@app.post("/games/hangman/new")
async def hangman_new():
    game = hangman_service.new_game()
    intro = f"New game! The word has {len(game.word)} letters. Guess a letter!"
    return JSONResponse(game.to_dict(speech=intro))


@app.post("/games/hangman/guess")
async def hangman_guess(payload: dict):
    session_id = payload.get("session_id", "")
    guess = payload.get("guess", "").strip()
    result = hangman_service.guess(session_id, guess)
    return JSONResponse(result)


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
        events.append({
            "title": e.title,
            "event_datetime": e.event_datetime,
            "human_readable": e.human_readable,
            "added_by": e.added_by,
            "is_past": dt < now,
        })
    events.sort(key=lambda e: (e["is_past"], e["human_readable"]))

    pending_count = sum(1 for t in todos if not t.completed)
    upcoming_count = sum(1 for e in events if not e["is_past"])

    try:
        family_names = sorted(set(json.loads(app_settings.phone_to_name).values()))
    except Exception:
        family_names = []

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "todos": todos,
            "events": events,
            "pending_count": pending_count,
            "upcoming_count": upcoming_count,
            "family_names": family_names,
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


@app.post("/dashboard/add-todo")
async def dashboard_add_todo(payload: dict):
    text = payload.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "missing text"}, status_code=400)
    item = TodoItem(
        text=text,
        due=payload.get("due") or None,
        added_by=payload.get("added_by", "Family").strip(),
        added_at=datetime.now().isoformat(timespec="seconds"),
    )
    markdown_service.append_todo(item)
    return JSONResponse({"ok": True})


@app.post("/dashboard/delete-todo")
async def dashboard_delete_todo(payload: dict):
    text = payload.get("text", "")
    if not text:
        return JSONResponse({"error": "missing text"}, status_code=400)
    success = markdown_service.delete_todo(text)
    if success:
        return JSONResponse({"ok": True})
    return JSONResponse({"error": "not found"}, status_code=404)


@app.post("/dashboard/add-event")
async def dashboard_add_event(payload: dict):
    title = payload.get("title", "").strip()
    event_datetime = payload.get("event_datetime", "").strip()
    if not title or not event_datetime:
        return JSONResponse({"error": "missing fields"}, status_code=400)
    try:
        dt = datetime.fromisoformat(event_datetime)
    except ValueError:
        return JSONResponse({"error": "invalid datetime"}, status_code=400)
    human_readable = f"{dt.strftime('%A %B')} {dt.day} at {dt.strftime('%I:%M %p').lstrip('0')}"
    item = EventItem(
        title=title,
        event_datetime=event_datetime,
        human_readable=human_readable,
        added_by=payload.get("added_by", "Family").strip(),
        added_at=datetime.now().isoformat(timespec="seconds"),
    )
    markdown_service.append_event(item)
    return JSONResponse({"ok": True})


@app.post("/dashboard/delete-event")
async def dashboard_delete_event(payload: dict):
    title = payload.get("title", "")
    event_datetime = payload.get("event_datetime", "")
    if not title or not event_datetime:
        return JSONResponse({"error": "missing fields"}, status_code=400)
    success = markdown_service.delete_event(title, event_datetime)
    if success:
        return JSONResponse({"ok": True})
    return JSONResponse({"error": "not found"}, status_code=404)


@app.get("/health")
async def health():
    return {"status": "ok"}
