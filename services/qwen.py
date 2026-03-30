import json
from datetime import datetime
from pathlib import Path

import httpx

from config import settings
from models.schemas import IntentResult, TodoItem, EventItem

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()


def _today_context() -> dict:
    now = datetime.now()
    return {
        "today": now.strftime("%Y-%m-%d"),
        "day_of_week": now.strftime("%A"),
        "current_time": now.strftime("%H:%M"),
    }


def _extract_json(text: str) -> dict:
    """Extract first valid JSON object from LLM response, tolerating surrounding prose."""
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char == '{':
            try:
                obj, _ = decoder.raw_decode(text, i)
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No JSON found in response: {text!r}")


def _chat(prompt: str) -> str:
    response = httpx.post(
        f"{settings.ollama_base_url}/api/chat",
        json={
            "model": settings.ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def classify_intent(transcript: str, caller_name: str) -> IntentResult:
    ctx = _today_context()
    prompt = _load_prompt("intent_classify.txt").format(
        caller_name=caller_name,
        transcript=transcript,
        **ctx,
    )
    raw = _chat(prompt)
    data = _extract_json(raw)
    return IntentResult(**data)


def extract_todo(transcript: str, added_by: str) -> TodoItem:
    ctx = _today_context()
    prompt = _load_prompt("todo_extract.txt").format(transcript=transcript, **ctx)
    raw = _chat(prompt)
    data = _extract_json(raw)
    return TodoItem(
        text=data["text"],
        due=data.get("due"),
        added_by=added_by,
        added_at=datetime.now().isoformat(timespec="seconds"),
    )


def extract_event(transcript: str, added_by: str) -> EventItem:
    ctx = _today_context()
    prompt = _load_prompt("event_extract.txt").format(transcript=transcript, **ctx)
    raw = _chat(prompt)
    data = _extract_json(raw)
    return EventItem(
        title=data["title"],
        event_datetime=data["event_datetime"],
        human_readable=data["human_readable"],
        added_by=added_by,
        added_at=datetime.now().isoformat(timespec="seconds"),
    )


def match_todo(transcript: str, todos: list[TodoItem]) -> str | None:
    """Fuzzy-match the caller's words to the closest pending todo. Returns matched text or None."""
    if not todos:
        return None
    todo_list = "\n".join(f"- {t.text}" for t in todos)
    prompt = _load_prompt("todo_match.txt").format(
        transcript=transcript,
        todo_list=todo_list,
    )
    raw = _chat(prompt)
    data = _extract_json(raw)
    matched = data.get("matched")
    return matched if matched else None


def answer_family_query(transcript: str, items: list, item_type: str) -> str:
    ctx = _today_context()
    items_json = json.dumps(
        [i.model_dump() for i in items],
        default=str,
        indent=2,
    )
    prompt = _load_prompt("family_query.txt").format(
        transcript=transcript,
        item_type=item_type,
        items_json=items_json,
        **ctx,
    )
    return _chat(prompt)


def quick_answer(query: str) -> tuple[bool, str | None]:
    """Try to answer from Qwen's own knowledge. Returns (can_answer, answer_text)."""
    ctx = _today_context()
    prompt = _load_prompt("quick_answer.txt").format(query=query, **ctx)
    raw = _chat(prompt)
    data = _extract_json(raw)
    return bool(data.get("can_answer")), data.get("answer")


def voice_summarize_research(query: str, results: list) -> str:
    """Short 2-3 sentence spoken summary of research results."""
    results_text = json.dumps(results, indent=2)
    prompt = _load_prompt("research_voice.txt").format(
        query=query,
        results=results_text,
    )
    return _chat(prompt)


def synthesize_research(query: str, results: list) -> str:
    ctx = _today_context()
    results_text = json.dumps(results, indent=2)
    prompt = _load_prompt("research_synthesize.txt").format(
        query=query,
        results=results_text,
        today=ctx["today"],
    )
    return _chat(prompt)
