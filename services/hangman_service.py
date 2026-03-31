import random
import uuid
from dataclasses import dataclass, field

WORDS = [
    # easy
    "cat", "dog", "sun", "hat", "cup", "red", "big", "run", "fly", "jump",
    "fish", "bird", "frog", "star", "moon", "book", "rain", "snow", "tree", "fire",
    # medium
    "apple", "tiger", "happy", "house", "water", "music", "snake", "cloud",
    "pizza", "magic", "robot", "beach", "clock", "dream", "globe", "horse",
    "lemon", "ocean", "plant", "queen", "river", "storm", "towel", "wheel",
    # harder
    "elephant", "butterfly", "adventure", "chocolate", "rainbow", "dinosaur",
    "treasure", "umbrella", "mountain", "penguin", "volcano", "keyboard",
    "strawberry", "crocodile", "telescope", "fireworks",
]

STAGES = [
    # 0 wrong
    ("  +---+\n"
     "  |   |\n"
     "      |\n"
     "      |\n"
     "      |\n"
     "      |\n"
     "========="),
    # 1
    ("  +---+\n"
     "  |   |\n"
     "  O   |\n"
     "      |\n"
     "      |\n"
     "      |\n"
     "========="),
    # 2
    ("  +---+\n"
     "  |   |\n"
     "  O   |\n"
     "  |   |\n"
     "      |\n"
     "      |\n"
     "========="),
    # 3
    ("  +---+\n"
     "  |   |\n"
     "  O   |\n"
     " /|   |\n"
     "      |\n"
     "      |\n"
     "========="),
    # 4
    ("  +---+\n"
     "  |   |\n"
     "  O   |\n"
     " /|\\  |\n"
     "      |\n"
     "      |\n"
     "========="),
    # 5
    ("  +---+\n"
     "  |   |\n"
     "  O   |\n"
     " /|\\  |\n"
     " /    |\n"
     "      |\n"
     "========="),
    # 6 — dead
    ("  +---+\n"
     "  |   |\n"
     "  O   |\n"
     " /|\\  |\n"
     " / \\  |\n"
     "      |\n"
     "========="),
]

MAX_WRONG = 6

_games: dict[str, "HangmanGame"] = {}


@dataclass
class HangmanGame:
    session_id: str
    word: str
    guessed: set = field(default_factory=set)
    wrong_count: int = 0

    @property
    def display_word(self) -> str:
        return " ".join(c.upper() if c in self.guessed else "_" for c in self.word)

    @property
    def wrong_letters(self) -> list[str]:
        return sorted(c.upper() for c in self.guessed if c not in self.word)

    @property
    def won(self) -> bool:
        return all(c in self.guessed for c in self.word)

    @property
    def lost(self) -> bool:
        return self.wrong_count >= MAX_WRONG

    @property
    def figure(self) -> str:
        return STAGES[min(self.wrong_count, MAX_WRONG)]

    def to_dict(self, speech: str = "") -> dict:
        return {
            "session_id": self.session_id,
            "display_word": self.display_word,
            "wrong_letters": self.wrong_letters,
            "wrong_count": self.wrong_count,
            "max_wrong": MAX_WRONG,
            "figure": self.figure,
            "won": self.won,
            "lost": self.lost,
            "word": self.word.upper() if (self.won or self.lost) else "",
            "speech": speech,
        }


def new_game() -> HangmanGame:
    sid = str(uuid.uuid4())
    game = HangmanGame(session_id=sid, word=random.choice(WORDS))
    _games[sid] = game
    return game


def get_game(session_id: str) -> HangmanGame | None:
    return _games.get(session_id)


HINT = "Say 'letter A' to guess a letter, or 'word elephant' to guess the whole word."


def guess(session_id: str, text: str) -> dict:
    """Process a guess (letter or whole word). Returns game state dict with speech."""
    game = _games.get(session_id)
    if not game:
        g = new_game()
        return g.to_dict(speech="I couldn't find your game — starting a new one! " + _intro(g))

    if game.won or game.lost:
        return game.to_dict(speech="The game is already over! Say 'new game' to play again.")

    text = text.strip().lower().rstrip(".,!?")

    # New game request
    if any(w in text for w in ["new game", "restart", "again", "reset", "start over"]):
        g = new_game()
        _games[session_id] = g
        _games[g.session_id] = g
        return g.to_dict(speech="New game! " + _intro(g))

    # Require explicit "letter X" or "word XXXX" prefix to filter noise
    if text.startswith("letter "):
        letter = text[7:].strip().rstrip(".,!?")
        if len(letter) == 1 and letter.isalpha():
            return _guess_letter(game, letter)
        return game.to_dict(speech=f"Say 'letter' followed by a single letter. For example: letter A.")

    if text.startswith("word "):
        word = text[5:].strip().rstrip(".,!?")
        if word:
            return _guess_word(game, word)
        return game.to_dict(speech="Say 'word' followed by your guess. For example: word elephant.")

    # Single bare letter also accepted (e.g. just "A")
    if len(text) == 1 and text.isalpha():
        return _guess_letter(game, text)

    return game.to_dict(speech=HINT)


def _guess_letter(game: HangmanGame, letter: str) -> dict:
    if letter in game.guessed:
        return game.to_dict(speech=f"You already tried {letter.upper()}.")
    game.guessed.add(letter)
    if letter in game.word:
        count = game.word.count(letter)
        plural = "times" if count > 1 else "time"
        if game.won:
            return game.to_dict(speech=f"Yes! {letter.upper()} appears {count} {plural}. You got it — the word was {game.word.upper()}!")
        return game.to_dict(speech=f"Yes! {letter.upper()} appears {count} {plural}.")
    else:
        game.wrong_count += 1
        if game.lost:
            return game.to_dict(speech=f"No {letter.upper()} — out of guesses! The word was {game.word.upper()}. Say 'new game' to try again!")
        remaining = MAX_WRONG - game.wrong_count
        return game.to_dict(speech=f"No {letter.upper()}. {remaining} {'guess' if remaining == 1 else 'guesses'} left.")


def _guess_word(game: HangmanGame, word: str) -> dict:
    if word == game.word:
        game.guessed.update(game.word)
        return game.to_dict(speech=f"Amazing! You guessed it — the word was {game.word.upper()}!")
    game.wrong_count = min(game.wrong_count + 1, MAX_WRONG)
    if game.lost:
        return game.to_dict(speech=f"Wrong word, and out of guesses! The word was {game.word.upper()}. Say 'new game' to try again!")
    remaining = MAX_WRONG - game.wrong_count
    return game.to_dict(speech=f"Nope, not {word.upper()}. {remaining} {'guess' if remaining == 1 else 'guesses'} left.")


def _intro(game: HangmanGame) -> str:
    length = len(game.word)
    return f"The word has {length} letters. Guess a letter!"
