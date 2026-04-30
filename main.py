import asyncio
import json
import logging
import logging.handlers
import os
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, Form, Response, Request, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pathlib import Path
from fastapi.templating import Jinja2Templates
import httpx
from handlers.call_handler import handle_incoming, handle_transcription
from handlers.research_handler import handle_research_choice, handle_research_whatsapp_choice
from handlers.response_handler import voice_say_hangup
from handlers import chat_handler
from services import whisper_service, reminder_service
from services import hangman_service, bulls_cows_service, word_ladder_service, twenty_questions_service
from config import settings as app_settings
from services import markdown_service, session_store
from services import qwen
from services.qwen import _chat
from models.schemas import TodoItem, EventItem

DEEPSTREAM_URL  = os.environ.get("DEEPSTREAM_URL",  "http://localhost:8090").rstrip("/")
MEDIAMTX_HLS_URL = os.environ.get("MEDIAMTX_HLS_URL", "http://localhost:8888").rstrip("/")
_triton_host    = os.environ.get("TRITON_URL",      "localhost:8001").split(":")[0]
TRITON_HTTP_URL = f"http://{_triton_host}:8002"
TRITON_MGMT_URL = f"http://{_triton_host}:8004"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL",    "qwen2.5:14b")

# ── Re-export state (owned here — app orchestrates cross-service lifecycle) ────
# Qwen-dependent routes read ollama_paused directly; no HTTP hop needed.
_reexport_state: dict = {"state": "ready", "eta_s": 0, "ollama_paused": False}
_reexport_running = False   # guard against concurrent re-exports

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
    return templates.TemplateResponse(request=request, name="cameras.html", context={})


async def _ds_proxy(method: str, path: str, payload: dict | None = None):
    """Proxy an HTTP request to the deepstream service."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        url = f"{DEEPSTREAM_URL}{path}"
        if method == "GET":
            r = await client.get(url)
        elif method == "DELETE":
            r = await client.delete(url)
        else:
            r = await client.request(method, url, json=payload)
    return JSONResponse(r.json(), status_code=r.status_code)


# ── Re-export orchestration ───────────────────────────────────────────────────

async def _pause_ollama() -> None:
    """Evict Qwen from GPU to free ~10 GB VRAM for TRT workspace."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            await c.post(f"{OLLAMA_BASE_URL}/api/generate",
                         json={"model": OLLAMA_MODEL, "keep_alive": 0})
        logger.info("Ollama: Qwen unloaded from GPU")
    except Exception as exc:
        logger.warning("Could not unload Ollama (proceeding anyway): %s", exc)


async def _resume_ollama() -> None:
    """Reload Qwen and warm it up so the first post-export request is fast."""
    try:
        async with httpx.AsyncClient(timeout=120.0) as c:
            await c.post(f"{OLLAMA_BASE_URL}/api/generate",
                         json={"model": OLLAMA_MODEL, "prompt": "hi",
                               "keep_alive": -1, "stream": False})
        logger.info("Ollama: Qwen reloaded and warmed up")
    except Exception as exc:
        logger.warning("Could not reload Ollama: %s — will load on next request", exc)


async def _triton_reload() -> None:
    """Unload then load the yoloworld model so Triton picks up the new TRT engine."""
    base = f"{TRITON_HTTP_URL}/v2/repository/models/yoloworld"
    for action in ("unload", "load"):
        try:
            async with httpx.AsyncClient(timeout=30.0) as c:
                await c.post(f"{base}/{action}", content=b"{}")
            logger.info("Triton: yoloworld %s OK", action)
        except Exception as exc:
            logger.warning("Triton %s failed: %s", action, exc)
        await asyncio.sleep(1)


async def _run_reexport(queries: list[str]) -> None:
    """Full re-export lifecycle: pause Qwen → TRT build → reload Triton → resume Qwen."""
    global _reexport_running
    try:
        _reexport_state.update({"state": "updating", "eta_s": 90, "ollama_paused": False})

        # 1. Free VRAM
        logger.info("Re-export: unloading Qwen to free VRAM (queries=%s)", queries)
        await _pause_ollama()
        _reexport_state["ollama_paused"] = True

        # 2. Kick off TRT build on triton management sidecar
        try:
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.post(f"{TRITON_MGMT_URL}/reexport", json={"queries": queries})
                r.raise_for_status()
            logger.info("Re-export: TRT build started")
        except Exception as exc:
            logger.error("Re-export: failed to start TRT build: %s", exc)
            await _resume_ollama()
            _reexport_state.update({"state": "ready", "eta_s": 0, "ollama_paused": False})
            return

        # 3. Poll until done (up to 180 s)
        deadline = asyncio.get_event_loop().time() + 180
        export_ok = False
        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(3)
            try:
                async with httpx.AsyncClient(timeout=5.0) as c:
                    st = (await c.get(f"{TRITON_MGMT_URL}/reexport/status")).json()
            except Exception:
                continue
            _reexport_state["eta_s"] = max(0, int(deadline - asyncio.get_event_loop().time()))
            if st.get("state") == "done":
                logger.info("Re-export: TRT build complete")
                export_ok = True
                break
            if st.get("state") == "error":
                logger.error("Re-export: TRT build failed: %s", st.get("error"))
                break
        else:
            logger.warning("Re-export: TRT build timed out after 180 s")

        # 4. Reload Triton so it picks up the new engine
        if export_ok:
            await _triton_reload()
            # Tell deepstream to sync its in-memory label mapping
            await _ds_proxy("POST", "/queries/commit", {"queries": queries})

        # 5. Reload Qwen regardless of export outcome
        logger.info("Re-export: reloading Qwen into GPU")
        await _resume_ollama()

    except Exception as exc:
        logger.error("Re-export: unexpected error: %s", exc)
        await _resume_ollama()   # always resume Qwen
    finally:
        _reexport_state.update({"state": "ready", "eta_s": 0, "ollama_paused": False})
        _reexport_running = False
        logger.info("Re-export: done (queries=%s)", queries)


@app.post("/cameras/set-stream")
async def cameras_set_stream(payload: dict):
    return await _ds_proxy("POST", "/set-stream", payload)


@app.post("/cameras/debug-overlay")
async def cameras_debug_overlay(payload: dict):
    return await _ds_proxy("POST", "/debug-overlay", payload)


@app.get("/cameras/queries")
async def cameras_get_queries():
    return await _ds_proxy("GET", "/queries")


async def _get_current_queries() -> list[str]:
    """Fetch the current query list from deepstream."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{DEEPSTREAM_URL}/queries")
            return r.json().get("queries", [])
    except Exception:
        return []


@app.post("/cameras/queries")
async def cameras_add_query(payload: dict):
    global _reexport_running
    # Step 1: update deepstream's in-memory queries + meta.json immediately
    r = await _ds_proxy("POST", "/queries", payload)
    if r.status_code != 200:
        return r
    queries = await _get_current_queries()
    if not _reexport_running:
        _reexport_running = True
        asyncio.create_task(_run_reexport(queries))
    return JSONResponse({"ok": True, "queries": queries,
                         "state": "updating", "eta_s": 90})


@app.delete("/cameras/queries/{index}")
async def cameras_remove_query(index: int):
    global _reexport_running
    r = await _ds_proxy("DELETE", f"/queries/{index}")
    if r.status_code != 200:
        return r
    queries = await _get_current_queries()
    if not _reexport_running:
        _reexport_running = True
        asyncio.create_task(_run_reexport(queries))
    return JSONResponse({"ok": True, "queries": queries,
                         "state": "updating", "eta_s": 90})


@app.get("/cameras/events")
async def cameras_get_events():
    return await _ds_proxy("GET", "/events")


@app.post("/cameras/threshold")
async def cameras_set_threshold(payload: dict):
    return await _ds_proxy("POST", "/threshold", payload)


@app.get("/cameras/threshold")
async def cameras_get_threshold():
    return await _ds_proxy("GET", "/threshold")


@app.post("/cameras/pad")
async def cameras_set_pad(payload: dict):
    return JSONResponse({"ok": True, "pad": 0.0})


@app.get("/cameras/pad")
async def cameras_get_pad():
    return JSONResponse({"pad": 0.0})


@app.get("/cameras/queries/status")
async def cameras_query_status():
    return JSONResponse(_reexport_state)


@app.post("/cameras/streams")
async def cameras_add_stream(payload: dict):
    return await _ds_proxy("POST", "/streams", payload)


@app.delete("/cameras/streams/{cam_id}")
async def cameras_remove_stream(cam_id: str):
    return await _ds_proxy("DELETE", f"/streams/{cam_id}")


@app.get("/cameras/streams")
async def cameras_list_streams():
    return await _ds_proxy("GET", "/streams")


@app.patch("/cameras/streams/{cam_id}/roi")
async def cameras_set_roi(cam_id: str, payload: dict):
    return await _ds_proxy("PATCH", f"/streams/{cam_id}/roi", payload)


@app.get("/cameras/clips")
async def cameras_list_clips(cam_id: str | None = None):
    qs = f"?cam_id={cam_id}" if cam_id else ""
    return await _ds_proxy("GET", f"/clips{qs}")


CLIPS_DIR = Path(os.environ.get("CLIPS_DIR", "/app/clips"))
DEEPSTREAM_URL = os.environ.get("DEEPSTREAM_URL", "http://deepstream:8090")


@app.get("/cameras/hls/{cam_id}/{path:path}")
async def cameras_hls_proxy(request: Request, cam_id: str, path: str):
    """Stream-proxy LL-HLS segments/playlists from MediaMTX directly.
    Phase 1 step B: drop the deepstream → sink-container hop entirely; MediaMTX
    serves LL-HLS at :8888/{cam_id}/* natively."""
    if ".." in cam_id or "/" in cam_id:
        return JSONResponse({"error": "invalid cam_id"}, status_code=400)
    target = f"{MEDIAMTX_HLS_URL}/{cam_id}/{path}"
    # LL-HLS blocking playlist requests carry _HLS_msn/_HLS_part query params — must forward.
    params = dict(request.query_params)
    client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=30.0))
    try:
        r = await client.send(client.build_request("GET", target, params=params), stream=True)
    except Exception:
        await client.aclose()
        return JSONResponse({"error": "stream unavailable"}, status_code=503)

    async def _stream():
        try:
            async for chunk in r.aiter_bytes(65536):
                yield chunk
        finally:
            await r.aclose()
            await client.aclose()

    from fastapi.responses import StreamingResponse
    passthrough = {k: v for k, v in r.headers.items()
                   if k.lower() not in ("content-encoding", "transfer-encoding", "content-length")}
    return StreamingResponse(
        _stream(), status_code=r.status_code,
        media_type=r.headers.get("content-type", "application/octet-stream"),
        headers=passthrough,
    )


@app.get("/cameras/clips/file/{cam_id}/{filename}")
async def cameras_serve_clip(cam_id: str, filename: str):
    if "/" in cam_id or "/" in filename or ".." in cam_id or ".." in filename:
        return JSONResponse({"error": "invalid path"}, status_code=400)
    path = CLIPS_DIR / cam_id / filename
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(path), media_type="video/mp4")



# ── LLM availability guard ────────────────────────────────────────────────────

def _llm_busy_json():
    eta = _reexport_state.get("eta_s", 0)
    msg = f"Bianca is updating her vision system and will be back in about {eta} seconds. Please try again shortly."
    return JSONResponse({"error": msg, "llm_busy": True}, status_code=503)

def _llm_busy_twiml():
    return Response(
        content=voice_say_hangup(
            "Bianca is updating her vision system. Please call back in a couple of minutes."
        ),
        media_type="application/xml",
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
    if _reexport_state.get("ollama_paused"):
        return _llm_busy_json()
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
    if _reexport_state.get("ollama_paused"):
        return _llm_busy_json()
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
    if _reexport_state.get("ollama_paused"):
        return _llm_busy_json()
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


# ── Bulls and Cows ────────────────────────────────────────────────────────────

@app.get("/games/bulls-cows", response_class=HTMLResponse)
async def bulls_cows_page(request: Request):
    return templates.TemplateResponse(request=request, name="bulls_cows.html", context={})


@app.post("/games/bulls-cows/new")
async def bulls_cows_new(payload: dict = {}):
    digits = int(payload.get("digits", 4))
    game   = bulls_cows_service.new_game(digits)
    digit_word = ["", "one", "two", "three", "four"][game.digits]
    speech = (
        f"I'm thinking of a {game.digits}-digit number — "
        f"all digits are different. Say {digit_word} digits to guess!"
    )
    return JSONResponse(game.to_dict(speech=speech))


@app.post("/games/bulls-cows/guess")
async def bulls_cows_guess(payload: dict):
    session_id = payload.get("session_id", "")
    guess_text = payload.get("guess", "").strip()
    result = bulls_cows_service.guess(session_id, guess_text)
    return JSONResponse(result)


# ── Word Ladder ───────────────────────────────────────────────────────────────

@app.get("/games/word-ladder", response_class=HTMLResponse)
async def word_ladder_page(request: Request):
    return templates.TemplateResponse(request=request, name="word_ladder.html", context={})


@app.post("/games/word-ladder/new")
async def word_ladder_new():
    if _reexport_state.get("ollama_paused"):
        return _llm_busy_json()
    start = target = None
    try:
        pair = await asyncio.to_thread(qwen.generate_word_ladder)
        start = pair.get("start", "").lower().strip()
        target = pair.get("target", "").lower().strip()
    except Exception as e:
        logger.warning("word_ladder/new: Qwen pair generation failed: %s", e)
    try:
        game = await asyncio.to_thread(word_ladder_service.new_game, start, target)
    except RuntimeError as e:
        logger.error("word_ladder/new: %s", e)
        return JSONResponse({"error": "Could not generate puzzle. Please try again."}, status_code=500)
    s = game.start_word.upper()
    t = game.target_word.upper()
    speech = f"Word Ladder! Change {s} one letter at a time to reach {t}. Each step must be a real word. Say your first word!"
    return JSONResponse(game.to_dict(speech=speech))


@app.post("/games/word-ladder/step")
async def word_ladder_step(payload: dict):
    session_id = payload.get("session_id", "")
    word = payload.get("word", "").strip()
    result = word_ladder_service.step(session_id, word)
    return JSONResponse(result)


@app.post("/games/word-ladder/hint")
async def word_ladder_hint(payload: dict):
    session_id = payload.get("session_id", "")
    result = word_ladder_service.hint(session_id)
    return JSONResponse(result)


# ── 20 Questions ──────────────────────────────────────────────────────────────

@app.get("/games/twenty-questions", response_class=HTMLResponse)
async def twenty_questions_page(request: Request):
    return templates.TemplateResponse(request=request, name="twenty_questions.html", context={})


@app.post("/games/twenty-questions/new")
async def twenty_questions_new():
    game = twenty_questions_service.new_game()
    return JSONResponse(game.to_dict(
        speech="Think of an animal, food, object, or place. Don't tell me! Tap 'I'm ready' when you've got one."
    ))


@app.post("/games/twenty-questions/start")
async def twenty_questions_start(payload: dict):
    if _reexport_state.get("ollama_paused"):
        return _llm_busy_json()
    session_id = payload.get("session_id", "")
    result = await asyncio.to_thread(twenty_questions_service.start_questions, session_id)
    return JSONResponse(result)


@app.post("/games/twenty-questions/answer")
async def twenty_questions_answer(payload: dict):
    if _reexport_state.get("ollama_paused"):
        return _llm_busy_json()
    session_id = payload.get("session_id", "")
    answer_text = payload.get("answer", "").strip()
    result = await asyncio.to_thread(twenty_questions_service.answer, session_id, answer_text)
    return JSONResponse(result)


@app.post("/games/twenty-questions/confirm")
async def twenty_questions_confirm(payload: dict):
    session_id = payload.get("session_id", "")
    answer_text = payload.get("answer", "").strip()
    result = twenty_questions_service.confirm(session_id, answer_text)
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
    if _reexport_state.get("ollama_paused"):
        return _llm_busy_twiml()
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
