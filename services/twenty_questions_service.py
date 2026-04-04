import uuid
from dataclasses import dataclass, field

MAX_QUESTIONS = 20
MIN_QUESTIONS_BEFORE_GUESS = 10  # Qwen must ask at least this many before guessing

_SYSTEM_PROMPT = """\
You are playing 20 Questions with a child. The child has thought of something secret.
It could be an animal, a food, an object, or a place — nothing scary or inappropriate.

Your job: guess what it is by asking yes/no questions.

Rules:
- Ask ONE short, simple yes/no question per turn
- Keep all questions friendly and age-appropriate for children aged 6-12
- Build logically on previous answers to narrow down the answer
- Do NOT guess before you have asked at least 10 questions — you need enough information first
- Only make a final guess when you are highly confident after 10+ questions, or by question 18 at the latest
- Do NOT ask for descriptions, hints, or 'does it start with X' — yes/no questions only

You MUST respond with valid JSON only — absolutely no other text before or after:
{"type": "question", "content": "Is it an animal?"}
OR when making your final guess:
{"type": "guess", "content": "Is it a cat?"}
"""

_YES_WORDS = {
    "yes", "yeah", "yep", "yup", "yay", "correct", "right", "true",
    "absolutely", "definitely", "sure", "of course", "uh huh", "mhm",
    "yes it is", "yes it does", "that's right", "that is right",
}
_NO_WORDS = {
    "no", "nope", "nah", "wrong", "incorrect", "false", "noway",
    "no it isn't", "no it doesn't", "no it's not", "definitely not",
    "not really", "no way",
}
_MAYBE_WORDS = {
    "maybe", "sort of", "kind of", "kinda", "not sure", "unsure",
    "i don't know", "i dont know", "depends", "not really sure",
}

_games: dict[str, "TwentyQGame"] = {}


def _normalize_answer(text: str) -> str:
    """Normalize spoken yes/no/maybe to one of: 'yes', 'no', 'maybe'."""
    t = text.lower().strip().rstrip(".,!?")
    for w in _YES_WORDS:
        if w in t:
            return "yes"
    for w in _NO_WORDS:
        if w in t:
            return "no"
    for w in _MAYBE_WORDS:
        if w in t:
            return "maybe"
    # If first word is yes/no, trust it
    first = t.split()[0] if t.split() else ""
    if first in ("yes", "yeah", "yep", "yup"):
        return "yes"
    if first in ("no", "nope", "nah"):
        return "no"
    return t[:60]  # pass through as-is if unrecognised — Qwen can handle it


@dataclass
class TwentyQGame:
    session_id:     str
    messages:       list = field(default_factory=list)   # Ollama /api/chat messages list
    question_count: int = 0
    phase:          str = "thinking"  # thinking | playing | guessing | finished
    last_question:  str = ""
    final_guess:    str = ""
    correct:        bool | None = None
    qa_history:     list = field(default_factory=list)   # [(question, answer), ...]

    def to_dict(self, speech: str = "") -> dict:
        return {
            "session_id":     self.session_id,
            "phase":          self.phase,
            "question_count": self.question_count,
            "max_questions":  MAX_QUESTIONS,
            "last_question":  self.last_question,
            "final_guess":    self.final_guess,
            "correct":        self.correct,
            "qa_history":     self.qa_history,
            "speech":         speech,
        }


def new_game() -> "TwentyQGame":
    sid = str(uuid.uuid4())
    game = TwentyQGame(
        session_id=sid,
        messages=[{"role": "system", "content": _SYSTEM_PROMPT}],
    )
    _games[sid] = game
    return game


def get_game(session_id: str) -> "TwentyQGame | None":
    return _games.get(session_id)


def start_questions(session_id: str) -> dict:
    """Kid has thought of something — get the first question from Qwen."""
    from services import qwen

    game = _games.get(session_id)
    if not game:
        game = new_game()

    game.messages.append({
        "role": "user",
        "content": "I've thought of something. Please ask me your first yes/no question.",
    })

    try:
        result = qwen.ask_twenty_questions(game.messages)
    except Exception:
        result = {"type": "question", "content": "Is it an animal?"}

    question = result.get("content", "Is it an animal?")
    game.messages.append({"role": "assistant", "content": str(result)})
    game.question_count = 1
    game.last_question = question
    game.phase = "playing"

    return game.to_dict(speech=question)


def answer(session_id: str, text: str) -> dict:
    """Process kid's yes/no answer and get next question or final guess from Qwen."""
    from services import qwen

    game = _games.get(session_id)
    if not game:
        return {"speech": "Game not found. Please start a new game.", "session_id": ""}

    if game.phase == "guessing":
        return game.to_dict(speech=f"{game.final_guess} — was I right? Say yes or no!")

    if game.phase != "playing":
        return game.to_dict(speech="")

    normalized = _normalize_answer(text)

    # Record Q&A pair in history
    if game.last_question:
        game.qa_history.append((game.last_question, normalized))

    game.messages.append({"role": "user", "content": normalized})

    # Force a guess if we've hit the question limit
    if game.question_count >= MAX_QUESTIONS:
        game.phase = "guessing"
        try:
            result = qwen.force_twenty_questions_guess(game.messages)
        except Exception:
            result = {"type": "guess", "content": "I'm not sure — I give up! What was it?"}
        guess_text = result.get("content", "I give up! What was it?")
        game.final_guess = guess_text
        game.messages.append({"role": "assistant", "content": str(result)})
        return game.to_dict(speech=f"{guess_text} Am I right?")

    # Ask Qwen for next turn
    try:
        result = qwen.ask_twenty_questions(game.messages)
    except Exception:
        result = {"type": "question", "content": "Does it have legs?"}

    msg_type = result.get("type", "question")
    content  = result.get("content", "Hmm, let me think...")
    game.messages.append({"role": "assistant", "content": str(result)})

    # Guard: don't allow a guess before the minimum question threshold,
    # regardless of what Qwen decided — treat it as a question instead.
    if msg_type == "guess" and game.question_count < MIN_QUESTIONS_BEFORE_GUESS:
        msg_type = "question"

    if msg_type == "guess":
        game.phase = "guessing"
        game.final_guess = content
        return game.to_dict(speech=f"{content} Am I right?")

    game.question_count += 1
    game.last_question = content
    return game.to_dict(speech=content)


def confirm(session_id: str, text: str) -> dict:
    """Kid confirms whether Qwen's guess is correct."""
    game = _games.get(session_id)
    if not game:
        return {"speech": "Game not found.", "session_id": ""}

    game.phase = "finished"
    normalized = _normalize_answer(text)

    if normalized == "yes":
        game.correct = True
        return game.to_dict(
            speech="Yes! I got it! You're a great thinker! Say 'new game' to play again."
        )
    else:
        game.correct = False
        return game.to_dict(
            speech="Aww, you stumped me! Well done! What was it? Say 'new game' to play again."
        )
