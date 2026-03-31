import asyncio
import logging
import time
from models.schemas import IntentResult
from services import qwen, tavily_service, twilio_service
from handlers.response_handler import (
    voice_say_then_gather,
    voice_say_hangup,
    voice_research_pending,
    voice_research_answer,
)

logger = logging.getLogger(__name__)

# If research finishes within this many seconds → answer on voice
QUICK_TIMEOUT = 10

# Pending tasks waiting for user choice: rid → {task, caller_number}
_pending: dict[str, dict] = {}

# Pending WhatsApp deliveries: wid → {whatsapp_text, caller_number}
_pending_whatsapp: dict[str, dict] = {}


async def handle_research(intent: IntentResult, caller_number: str) -> str:
    if not intent.query or len(intent.query.split()) < 2:
        return voice_say_then_gather("Could you give me a bit more detail on what you'd like to know?")

    # Images always go to WhatsApp — can't send images via voice
    if intent.intent == "research_images":
        asyncio.create_task(_deliver_images(intent.query, caller_number))
        return voice_say_then_gather(
            "I'll find those images and send them to your WhatsApp shortly."
        )

    # Try quick answer from Qwen's own knowledge first (fast, no Tavily needed)
    can_answer, answer = await asyncio.to_thread(qwen.quick_answer, intent.query)
    if can_answer and answer:
        logger.info("Quick knowledge answer for: %s", intent.query)
        return voice_say_then_gather(answer)

    # Start full web research (returns (voice_text, whatsapp_text) tuple)
    task = asyncio.create_task(_do_research(intent.query))

    try:
        voice_text, whatsapp_text = await asyncio.wait_for(asyncio.shield(task), timeout=QUICK_TIMEOUT)
        logger.info("Research completed quickly for: %s", intent.query)
        wid = f"{caller_number}_{int(time.time())}"
        _pending_whatsapp[wid] = {"whatsapp_text": whatsapp_text, "caller_number": caller_number}
        return voice_research_answer(voice_text, wid)
    except asyncio.TimeoutError:
        rid = f"{caller_number}_{int(time.time())}"
        _pending[rid] = {"task": task, "caller_number": caller_number}
        logger.info("Research taking long, asking user preference (rid=%s)", rid)
        return voice_research_pending(rid)
    except Exception:
        logger.exception("Research failed for query: %s", intent.query)
        return voice_say_then_gather(
            "Sorry, I couldn't reach the search service just now. Please try again in a moment."
        )


async def handle_research_choice(rid: str, transcript: str) -> str:
    """Called after user says 'wait' or 'WhatsApp' during a slow research."""
    pending = _pending.pop(rid, None)
    if not pending:
        return voice_say_then_gather("Sorry, I lost track of that search. Could you ask again?")

    task: asyncio.Task = pending["task"]
    caller_number: str = pending["caller_number"]

    wants_whatsapp = any(
        w in transcript.lower()
        for w in ["whatsapp", "send", "message", "no", "that's fine", "ok"]
    )

    if wants_whatsapp:
        asyncio.create_task(_deliver_whatsapp_when_done(task, caller_number))
        return voice_say_then_gather("Sure, I'll send the results to your WhatsApp shortly.")

    # User wants to wait — give it 15 more seconds
    try:
        voice_text, whatsapp_text = await asyncio.wait_for(asyncio.shield(task), timeout=15)
        wid = f"{caller_number}_{int(time.time())}"
        _pending_whatsapp[wid] = {"whatsapp_text": whatsapp_text, "caller_number": caller_number}
        return voice_research_answer(voice_text, wid)
    except asyncio.TimeoutError:
        asyncio.create_task(_deliver_whatsapp_when_done(task, caller_number))
        return voice_say_then_gather(
            "That's taking longer than expected. I'll send it to your WhatsApp when it's ready."
        )


async def handle_research_whatsapp_choice(wid: str, transcript: str) -> str:
    """Called after Bianca asks 'Want full details on WhatsApp?' — user says yes or no."""
    pending = _pending_whatsapp.pop(wid, None)
    if not pending:
        return voice_say_then_gather("Anything else I can help with?")

    wants_whatsapp = any(
        w in transcript.lower()
        for w in ["yes", "yeah", "sure", "please", "send", "whatsapp", "yep", "ok"]
    )

    if wants_whatsapp:
        try:
            twilio_service.send_whatsapp(pending["caller_number"], pending["whatsapp_text"])
        except Exception:
            logger.exception("Failed to send WhatsApp research detail for wid=%s", wid)
        return voice_say_then_gather("Sent! Anything else?")

    return voice_say_then_gather("No problem. Anything else I can help with?")


async def _do_research(query: str) -> tuple[str, str]:
    """Run Tavily search + parallel Qwen summaries. Returns (voice_text, whatsapp_text)."""
    results = await asyncio.to_thread(tavily_service.search_web, query)
    if not results:
        msg = f"Sorry, I couldn't find anything useful for: {query}"
        return msg, msg

    voice_text, whatsapp_text = await asyncio.gather(
        asyncio.to_thread(qwen.voice_summarize_research, query, results),
        asyncio.to_thread(qwen.synthesize_research, query, results),
    )
    return voice_text, whatsapp_text


async def _deliver_whatsapp_when_done(task: asyncio.Task, caller_number: str) -> None:
    """Wait for the research task to finish then send full result via WhatsApp."""
    try:
        _voice_text, whatsapp_text = await task
        twilio_service.send_whatsapp(caller_number, whatsapp_text)
    except Exception:
        logger.exception("Failed to deliver WhatsApp research result")
        twilio_service.send_whatsapp(caller_number, "Sorry, the search failed. Please try calling again.")


async def _deliver_images(query: str, caller_number: str) -> None:
    try:
        image_urls = await asyncio.to_thread(tavily_service.search_images, query)
        if not image_urls:
            twilio_service.send_whatsapp(caller_number, f"Sorry, couldn't find images for: {query}")
            return
        twilio_service.send_whatsapp(caller_number, f"*Images: {query}*")
        for url in image_urls[:5]:
            twilio_service.send_whatsapp_image(caller_number, url)
    except Exception:
        logger.exception("Failed to deliver images for query: %s", query)
        twilio_service.send_whatsapp(caller_number, "Sorry, the image search failed.")
