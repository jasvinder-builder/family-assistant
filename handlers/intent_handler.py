import logging
from models.schemas import IntentResult
from handlers import todo_handler, event_handler
from handlers.response_handler import voice_say_hangup, voice_say_then_gather

logger = logging.getLogger(__name__)

HELP_MSG = (
    "Sorry, I didn't understand that. You can: add a todo, add an event, "
    "ask what's on your list, check upcoming events, or ask me to search for something."
)


def route(
    intent: IntentResult,
    transcript: str,
    caller_name: str,
    caller_number: str,
) -> str:
    logger.info("Intent: %s (confidence=%.2f) for %s", intent.intent, intent.confidence, caller_name)

    if intent.intent == "goodbye":
        return voice_say_hangup(f"Great talking with you, {caller_name}! Take care. Bye!")

    if intent.confidence < 0.6 or intent.intent == "unknown":
        return voice_say_then_gather(HELP_MSG)

    if intent.intent == "add_todo":
        return todo_handler.add_todo(transcript, caller_name)

    if intent.intent == "complete_todo":
        return todo_handler.complete_todo(transcript, caller_name)

    if intent.intent == "add_event":
        return event_handler.add_event(transcript, caller_name)

    if intent.intent == "query_tasks":
        return todo_handler.query_todos(caller_name, transcript)

    if intent.intent == "query_events":
        return event_handler.query_events(caller_name, transcript)

    return voice_say_then_gather(HELP_MSG)
