import asyncio
import logging

from services import qwen, markdown_service, tavily_service
from models.schemas import IntentResult

logger = logging.getLogger(__name__)


async def handle_chat(transcript: str, caller_name: str = "Family") -> dict:
    """
    Process a plain-text transcript from the browser interface.
    Returns {speech, display, intent}
      - speech:  short text read aloud via speechSynthesis
      - display: full text shown on page (longer for research)
      - intent:  classified intent string
    """
    if not transcript or len(transcript.strip()) < 2:
        return _r("I didn't catch that. Could you try again?")

    try:
        intent = await asyncio.to_thread(qwen.classify_intent, transcript, caller_name)
    except Exception:
        logger.exception("Intent classification failed in chat")
        return _r("Sorry, something went wrong. Please try again.")

    logger.info("Chat intent from %s: %s (%.2f)", caller_name, intent.intent, intent.confidence)

    if intent.confidence < 0.3 or intent.intent == "unknown":
        return _r(
            "I'm not sure I understood that. Try asking about your todos, events, or a research question.",
            intent="unknown",
        )

    if intent.intent == "goodbye":
        return _r(f"Goodbye {caller_name}! Take care.", intent="goodbye")

    if intent.intent == "add_todo":
        return await _add_todo(transcript, caller_name)

    if intent.intent == "complete_todo":
        return await _complete_todo(transcript)

    if intent.intent == "add_event":
        return await _add_event(transcript, caller_name)

    if intent.intent == "query_tasks":
        return await _query(transcript, "todos")

    if intent.intent == "query_events":
        return await _query(transcript, "events")

    if intent.intent.startswith("research") and intent.intent != "research_images":
        return await _research(intent)

    if intent.intent == "research_images":
        return _r(
            "Image search isn't supported in the browser interface. Try asking on the phone instead.",
            intent="research_images",
        )

    return _r("I'm not sure how to help with that.", intent="unknown")


# ── Intent handlers ───────────────────────────────────────────────────────────

async def _add_todo(transcript: str, caller_name: str) -> dict:
    try:
        item = await asyncio.to_thread(qwen.extract_todo, transcript, caller_name)
        markdown_service.append_todo(item)
        msg = f"Done! I've added '{item.text}' to the todo list."
        if item.due:
            msg += f" Due {item.due}."
        return _r(msg, intent="add_todo")
    except Exception:
        logger.exception("Failed to add todo from chat")
        return _r("Sorry, I had trouble understanding that todo. Could you rephrase?")


async def _complete_todo(transcript: str) -> dict:
    try:
        todos = markdown_service.read_todos()
        pending = [t for t in todos if not t.completed]
        if not pending:
            return _r("There are no pending todos to complete.", intent="complete_todo")
        matched = await asyncio.to_thread(qwen.match_todo, transcript, pending)
        if not matched:
            return _r("I couldn't find that on the list. Could you be more specific?", intent="complete_todo")
        markdown_service.complete_todo(matched)
        return _r(f"Done! I've marked '{matched}' as complete.", intent="complete_todo")
    except Exception:
        logger.exception("Failed to complete todo from chat")
        return _r("Sorry, something went wrong. Please try again.")


async def _add_event(transcript: str, caller_name: str) -> dict:
    try:
        item = await asyncio.to_thread(qwen.extract_event, transcript, caller_name)
        markdown_service.append_event(item)
        return _r(f"Done! I've added '{item.title}' on {item.human_readable}.", intent="add_event")
    except Exception:
        logger.exception("Failed to add event from chat")
        return _r("Sorry, I had trouble understanding that event. Could you rephrase?")


async def _query(transcript: str, item_type: str) -> dict:
    try:
        items = markdown_service.read_todos() if item_type == "todos" else markdown_service.read_events()
        answer = await asyncio.to_thread(qwen.answer_family_query, transcript, items, item_type)
        return _r(answer, intent=f"query_{item_type}")
    except Exception:
        logger.exception("Failed to handle %s query from chat", item_type)
        return _r("Sorry, I couldn't retrieve that information.")


async def _research(intent: IntentResult) -> dict:
    if not intent.query or len(intent.query.split()) < 2:
        return _r("Could you give me a bit more detail on what you'd like to know?", intent="research_web")

    # Try Qwen's own knowledge first (fast)
    try:
        can_answer, answer = await asyncio.to_thread(qwen.quick_answer, intent.query)
        if can_answer and answer:
            return _r(answer, intent="research_web")
    except Exception:
        pass

    # Full web research — parallel voice summary + full detail
    try:
        results = await asyncio.to_thread(tavily_service.search_web, intent.query)
        if not results:
            return _r(f"Sorry, I couldn't find anything useful for: {intent.query}", intent="research_web")

        speech, display = await asyncio.gather(
            asyncio.to_thread(qwen.voice_summarize_research, intent.query, results),
            asyncio.to_thread(qwen.synthesize_research, intent.query, results),
        )
        return {"speech": speech, "display": display, "intent": "research_web"}
    except Exception:
        logger.exception("Research failed for chat query: %s", intent.query)
        return _r("Sorry, I couldn't reach the search service. Please try again.")


def _r(speech: str, display: str | None = None, intent: str = "unknown") -> dict:
    return {"speech": speech, "display": display or speech, "intent": intent}
