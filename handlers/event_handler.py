import logging
from services import qwen, markdown_service
from handlers.response_handler import voice_say_then_gather

logger = logging.getLogger(__name__)


def add_event(transcript: str, caller_name: str) -> str:
    try:
        item = qwen.extract_event(transcript, added_by=caller_name)
        markdown_service.append_event(item)
        return voice_say_then_gather(f"Done. I've added '{item.title}' on {item.human_readable}.")
    except Exception:
        logger.exception("Failed to add event from transcript: %r", transcript)
        return voice_say_then_gather("Sorry, I had trouble understanding that. Could you try rephrasing?")


def query_events(caller_name: str, transcript: str) -> str:
    events = markdown_service.read_events()
    summary = qwen.answer_family_query(transcript, events, "Events")
    return voice_say_then_gather(summary)
