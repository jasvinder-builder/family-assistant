import asyncio
import logging
import httpx
from fastapi import Form
from config import settings
from services import whisper_service, qwen, session_store
from handlers import intent_handler, research_handler
from handlers.response_handler import voice_gather, voice_say_hangup, voice_say_then_gather, voice_filler_redirect

logger = logging.getLogger(__name__)


def handle_incoming(From: str = Form(default="")) -> str:
    """Initial webhook when someone calls. Returns TwiML to gather speech."""
    caller_name = settings.get_caller_name(From)
    if not caller_name:
        logger.warning("Unknown caller: %s", From)
        return voice_say_hangup("Sorry, this number isn't registered with the family assistant. Goodbye.")
    return voice_gather(f"Hi {caller_name}, this is Bianca, your family assistant. How can I help you?")


def handle_transcription(
    From: str = Form(default=""),
    RecordingUrl: str = Form(default=""),
    RecordingSid: str = Form(default=""),
) -> str:
    """Webhook called by Twilio with the recording. Returns filler TwiML immediately
    and launches async computation in the background."""
    caller_name = settings.get_caller_name(From)
    if not caller_name:
        return voice_say_hangup("Sorry, this number isn't registered. Goodbye.")

    if not RecordingUrl:
        return voice_say_then_gather("Sorry, I didn't catch that. Could you repeat?")

    # Create session, launch computation, return filler immediately
    session = session_store.create(RecordingSid)
    asyncio.create_task(_compute_answer(RecordingSid, RecordingUrl, From, caller_name, session))
    return voice_filler_redirect(RecordingSid)


async def _compute_answer(
    sid: str,
    recording_url: str,
    from_number: str,
    caller_name: str,
    session: session_store.Session,
) -> None:
    """Download audio, transcribe, classify and handle intent. Stores result in session."""
    try:
        # Download recording from Twilio
        async with httpx.AsyncClient() as client:
            audio_resp = await client.get(
                recording_url,
                auth=(settings.twilio_account_sid, settings.twilio_auth_token),
                timeout=10,
                follow_redirects=True,
            )
        audio_resp.raise_for_status()

        # Transcribe with Whisper (runs in thread to avoid blocking event loop)
        transcript, confidence = await asyncio.to_thread(
            whisper_service.transcribe, audio_resp.content
        )
        logger.info("Transcript from %s (%.2f): %s", caller_name, confidence, transcript)

        if not transcript or len(transcript.split()) < 2 or confidence < 0.3:
            session.result = voice_say_then_gather("Sorry, I didn't catch that clearly. Could you repeat?")
            session.event.set()
            return

        # Classify intent
        intent = await asyncio.to_thread(qwen.classify_intent, transcript, caller_name)

        # Research is async-native (quick timeout logic) — call directly
        # Everything else runs in a thread (sync Qwen calls)
        if intent.intent in ("research", "research_images"):
            result = await research_handler.handle_research(intent, from_number)
        else:
            result = await asyncio.to_thread(
                intent_handler.route,
                intent=intent,
                transcript=transcript,
                caller_name=caller_name,
                caller_number=from_number,
            )
        session.result = result

    except Exception:
        logger.exception("Failed to compute answer for recording %s", sid)
        session.result = voice_say_then_gather("Sorry, something went wrong. Please try again.")
    finally:
        session.event.set()
