import random
import uuid
from dataclasses import dataclass, field

MAX_ATTEMPTS = 10

# Map spoken words / homophones / digit strings to digit values
_WORD_TO_DIGIT: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    # common Whisper homophones
    "won": 1, "to": 2, "too": 2, "for": 4, "ate": 8, "nein": 9,
    # digit characters (e.g. "2 4 1 3" → each token is a digit string)
    **{str(i): i for i in range(10)},
}

_games: dict[str, "BullsCowsGame"] = {}


@dataclass
class BullsCowsGame:
    session_id: str
    secret: str                          # 4-char digit string, e.g. "2413"
    attempts:   list[str] = field(default_factory=list)
    bulls_list: list[int] = field(default_factory=list)
    cows_list:  list[int] = field(default_factory=list)
    won:  bool = False
    lost: bool = False

    def to_dict(self, speech: str = "") -> dict:
        return {
            "session_id":    self.session_id,
            "attempts":      self.attempts,
            "bulls_list":    self.bulls_list,
            "cows_list":     self.cows_list,
            "attempt_count": len(self.attempts),
            "max_attempts":  MAX_ATTEMPTS,
            "won":           self.won,
            "lost":          self.lost,
            "secret":        self.secret if (self.won or self.lost) else "",
            "speech":        speech,
        }


def _make_secret() -> str:
    digits = list(range(10))
    random.shuffle(digits)
    while digits[0] == 0:
        random.shuffle(digits)
    return "".join(map(str, digits[:4]))


def _score(secret: str, guess: str) -> tuple[int, int]:
    bulls = sum(s == g for s, g in zip(secret, guess))
    cows  = sum(g in secret for g in guess) - bulls
    return bulls, cows


def parse_spoken_number(text: str) -> str | None:
    """Convert spoken text to a 4-digit string, or None if invalid.

    Accepts:
    - Word sequences: "two four one three" → "2413"
    - Digit strings:  "2 4 1 3"           → "2413"
    - Homophones:     "to for won ate"     → "2418"

    Returns None if the result isn't exactly 4 digits.
    """
    tokens = text.lower().strip().replace(",", " ").split()
    digits = []
    for tok in tokens:
        tok = tok.strip(".,!?;:")
        if tok in _WORD_TO_DIGIT:
            digits.append(_WORD_TO_DIGIT[tok])
        else:
            return None  # unrecognised token — reject whole input
    if len(digits) != 4:
        return None
    return "".join(map(str, digits))


def new_game() -> "BullsCowsGame":
    sid = str(uuid.uuid4())
    game = BullsCowsGame(session_id=sid, secret=_make_secret())
    _games[sid] = game
    return game


def get_game(session_id: str) -> "BullsCowsGame | None":
    return _games.get(session_id)


def guess(session_id: str, text: str) -> dict:
    game = _games.get(session_id)
    if not game:
        g = new_game()
        return g.to_dict(speech=(
            "Starting a new game! I'm thinking of a 4-digit number — "
            "all digits are different. Say each digit separately to guess."
        ))

    if game.won or game.lost:
        return game.to_dict(speech="The game is over! Say 'new game' to play again.")

    text = text.strip().lower().rstrip(".,!?")

    if any(w in text for w in ["new game", "restart", "again", "start over", "reset"]):
        g = new_game()
        _games[session_id] = g
        _games[g.session_id] = g
        return g.to_dict(speech="New game! I'm thinking of a 4-digit number with all different digits.")

    number = parse_spoken_number(text)
    if number is None:
        return game.to_dict(speech=(
            "I didn't catch that as four digits. "
            "Say each digit separately — for example: one, two, three, four."
        ))

    if len(set(number)) != 4:
        return game.to_dict(speech="All four digits must be different from each other. Try again!")

    bulls, cows = _score(game.secret, number)
    game.attempts.append(number)
    game.bulls_list.append(bulls)
    game.cows_list.append(cows)

    if bulls == 4:
        game.won = True
        n = len(game.attempts)
        attempt_word = "attempt" if n == 1 else "attempts"
        return game.to_dict(speech=f"Amazing! {' '.join(number)} is correct! You cracked it in {n} {attempt_word}!")

    remaining = MAX_ATTEMPTS - len(game.attempts)
    if remaining <= 0:
        game.lost = True
        secret_spoken = " ".join(game.secret)
        return game.to_dict(speech=f"Out of guesses! The secret number was {secret_spoken}. Say 'new game' to try again!")

    bull_word = "bull" if bulls == 1 else "bulls"
    cow_word  = "cow"  if cows  == 1 else "cows"
    left_word = "guess" if remaining == 1 else "guesses"
    return game.to_dict(speech=f"{bulls} {bull_word}, {cows} {cow_word}. {remaining} {left_word} left.")
