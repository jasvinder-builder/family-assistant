import json
import logging
from datetime import datetime
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

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


def _extract_json_array(text: str) -> list:
    """Extract first valid JSON array from LLM response, tolerating surrounding prose."""
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char == '[':
            try:
                arr, _ = decoder.raw_decode(text, i)
                if isinstance(arr, list):
                    return arr
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No JSON array found in response: {text!r}")


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


def generate_quiz(subject: str, grade: int) -> list[dict]:
    """Generate 10 quiz questions for the given subject and grade level."""
    age_map = {1: "6", 2: "7", 3: "8", 4: "9", 5: "10", 6: "11", 7: "12", 8: "13"}
    age = age_map.get(grade, str(grade + 5))
    prompt = _load_prompt("quiz_generate.txt").format(
        subject=subject,
        grade=grade,
        age_range=f"{age} years old",
    )
    response = httpx.post(
        f"{settings.ollama_base_url}/api/chat",
        json={
            "model": settings.ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    raw = response.json()["message"]["content"]
    questions = _extract_json_array(raw)
    validated = []
    letter_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    for q in questions:
        if not isinstance(q, dict):
            continue
        if not isinstance(q.get("question"), str):
            continue
        opts = q.get("options")
        if not isinstance(opts, list) or len(opts) != 4:
            continue
        # Coerce "correct" — Qwen sometimes returns "2" (string) or "B" (letter) instead of 2 (int)
        raw_correct = q.get("correct")
        correct_idx = None
        if isinstance(raw_correct, int):
            correct_idx = raw_correct
        elif isinstance(raw_correct, str):
            if raw_correct.strip() in letter_to_idx:
                correct_idx = letter_to_idx[raw_correct.strip()]
            elif raw_correct.strip().isdigit():
                correct_idx = int(raw_correct.strip())
        if correct_idx is None or not (0 <= correct_idx <= 3):
            logger.warning("quiz: dropping question with invalid correct=%r: %s", raw_correct, q.get("question", "")[:60])
            continue
        validated.append({
            "question": q["question"],
            "options": [str(o) for o in opts],
            "correct": correct_idx,
        })
    if len(validated) < 5:
        logger.error("quiz: too few valid questions (%d). Raw output:\n%s", len(validated), raw[:500])
        raise ValueError(f"Too few valid questions: {len(validated)}")
    return validated[:10]


def resolve_answer(transcript: str, options: list[str]) -> int | None:
    """Use Qwen to match a spoken transcript to one of 4 quiz options. Returns 0–3 or None."""
    labels = ['A', 'B', 'C', 'D']
    options_text = '\n'.join(f"{labels[i]}: {opt}" for i, opt in enumerate(options))
    prompt = (
        f"Answer options:\n{options_text}\n\n"
        f"The person said: \"{transcript}\"\n\n"
        f"Which option (A, B, C, or D) did they choose? "
        f"Reply with a single letter only — no punctuation, no explanation."
    )
    raw = _chat(prompt).strip().upper()
    idx_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    return idx_map.get(raw[0]) if raw else None


def synthesize_research(query: str, results: list) -> str:
    ctx = _today_context()
    results_text = json.dumps(results, indent=2)
    prompt = _load_prompt("research_synthesize.txt").format(
        query=query,
        results=results_text,
        today=ctx["today"],
    )
    return _chat(prompt)
