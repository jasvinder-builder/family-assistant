import logging
from services import qwen, markdown_service
from handlers.response_handler import voice_say_then_gather

logger = logging.getLogger(__name__)


def add_todo(transcript: str, caller_name: str) -> str:
    try:
        item = qwen.extract_todo(transcript, added_by=caller_name)
        markdown_service.append_todo(item)
        due_msg = f", due {item.due}" if item.due else ""
        return voice_say_then_gather(f"Done. I've added '{item.text}'{due_msg} to the list.")
    except Exception:
        logger.exception("Failed to add todo from transcript: %r", transcript)
        return voice_say_then_gather("Sorry, I had trouble understanding that. Could you try rephrasing?")


def query_todos(caller_name: str, transcript: str) -> str:
    todos = markdown_service.read_todos()
    summary = qwen.answer_family_query(transcript, todos, "Todos")
    return voice_say_then_gather(summary)


def complete_todo(transcript: str, caller_name: str) -> str:
    try:
        pending = [t for t in markdown_service.read_todos() if not t.completed]
        if not pending:
            return voice_say_then_gather("You don't have any pending todos to mark as done.")
        matched_text = qwen.match_todo(transcript, pending)
        if not matched_text:
            return voice_say_then_gather("I couldn't find that on your list. Can you be more specific?")
        success = markdown_service.complete_todo(matched_text)
        if success:
            return voice_say_then_gather(f"Done. I've marked '{matched_text}' as complete.")
        return voice_say_then_gather("I couldn't find that on your list. Can you be more specific?")
    except Exception:
        logger.exception("Failed to complete todo from transcript: %r", transcript)
        return voice_say_then_gather("Sorry, something went wrong. Please try again.")
