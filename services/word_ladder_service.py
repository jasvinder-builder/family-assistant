import logging
import random
import uuid
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MAX_WRONG_STEPS = 5

# ── Word set ───────────────────────────────────────────────────────────────────

# Minimal bundled fallback used when /usr/share/dict/words is absent
_MINIMAL_WORDS = frozenset([
    # 3-letter
    "cat", "cot", "dot", "dog", "fog", "fig", "fit", "bit", "big", "bag",
    "bat", "hat", "hot", "hog", "hip", "hit", "cap", "cup", "pup",
    "sun", "run", "fun", "fan", "can", "pan", "ban", "man", "men",
    "hen", "ten", "den", "pen", "pet", "pit", "pig", "pin", "tin", "tan",
    "tap", "top", "tip", "dip", "dim", "him", "his", "hit", "sit", "set",
    "wet", "bet", "bed", "fed", "red", "rod", "rob", "rib", "rid", "bid",
    # 4-letter
    "love", "live", "like", "bike", "hike", "hide", "hire", "fire", "fine",
    "line", "lane", "land", "hand", "band", "sand", "sane", "same", "game",
    "gate", "late", "lake", "cake", "cave", "cave", "gave", "have", "hare",
    "bare", "bore", "more", "mare", "dare", "date", "fate", "face", "lace",
    "race", "rice", "dice", "mice", "nice", "nine", "pine", "pint", "hint",
    "mint", "mist", "fist", "fish", "dish", "wish", "wash", "cash", "gash",
    "lash", "dash", "bash", "base", "case", "care", "core", "code", "mode",
    "made", "make", "rake", "rate", "mate", "mane", "male", "tale", "tall",
    "ball", "call", "fall", "hall", "wall", "bell", "sell", "tell", "well",
    "will", "hill", "fill", "mill", "bill", "bull", "pull", "full", "fall",
    "play", "clay", "clan", "plan", "flat", "flag", "flab", "slab", "slam",
    "slap", "snap", "snag", "snob", "knob", "know", "snow", "slow", "blow",
    "blue", "glue", "clue", "clew", "chew", "chow", "show", "shop", "ship",
    "chip", "chap", "chat", "that", "than", "thin", "this", "with", "wish",
    "bear", "beer", "deer", "deep", "seep", "seed", "feed", "feel", "reel",
    "real", "read", "bead", "bean", "lean", "leap", "heap", "heat", "meat",
    "meal", "seal", "deal", "dear", "fear", "gear", "hear", "near", "wear",
    "pear", "tear", "year", "yarn", "barn", "born", "corn", "cord", "word",
    "ward", "warm", "worm", "worn", "torn", "horn", "hone", "bone", "cone",
    "done", "dune", "tune", "tone", "zone", "gone", "lone", "lone", "love",
])


def _load_word_set() -> frozenset[str]:
    """Load common 3-5 letter English words from system dictionary."""
    try:
        with open("/usr/share/dict/words") as f:
            return frozenset(
                w for line in f
                if (w := line.strip().lower()) and w.isalpha() and 3 <= len(w) <= 5
            )
    except FileNotFoundError:
        logger.warning(
            "word_ladder: /usr/share/dict/words not found — "
            "using bundled minimal word set. Install 'wamerican' for best results."
        )
        return _MINIMAL_WORDS


_WORD_SET: frozenset[str] = _load_word_set()

# Pre-group by word length for BFS efficiency
_WORDS_BY_LEN: dict[int, list[str]] = {}
for _w in _WORD_SET:
    _WORDS_BY_LEN.setdefault(len(_w), []).append(_w)

# Fallback puzzle pairs (pre-verified to have short BFS paths in common word sets)
_FALLBACK_PAIRS: list[tuple[str, str]] = [
    # 4-letter, 2 steps
    ("love", "like"),   # love→live→like
    ("bear", "deer"),   # bear→beer→deer
    ("ball", "tale"),   # ball→tall→tale
    ("hand", "lane"),   # hand→land→lane
    ("play", "clan"),   # play→clay→clan
    ("ship", "shop"),   # 1 step (easy)
    # 3-letter, 2-3 steps
    ("cat",  "dog"),    # cat→cot→dot→dog
    ("hat",  "cap"),    # hat→cat→cap
    ("big",  "fit"),    # big→bit→fit
    ("hot",  "fog"),    # hot→hog→fog
]

_games: dict[str, "WordLadderGame"] = {}


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class WordLadderGame:
    session_id:   str
    start_word:   str
    target_word:  str
    current_word: str
    path: list[str] = field(default_factory=list)   # words visited, includes start
    wrong_attempts: int = 0
    won:  bool = False
    lost: bool = False

    def to_dict(self, speech: str = "") -> dict:
        return {
            "session_id":     self.session_id,
            "start_word":     self.start_word,
            "target_word":    self.target_word,
            "current_word":   self.current_word,
            "path":           self.path,
            "wrong_attempts": self.wrong_attempts,
            "max_wrong":      MAX_WRONG_STEPS,
            "won":            self.won,
            "lost":           self.lost,
            "speech":         speech,
        }


# ── Game logic ─────────────────────────────────────────────────────────────────

def _one_letter_apart(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    return sum(x != y for x, y in zip(a, b)) == 1


def _bfs_path(start: str, target: str) -> list[str] | None:
    """Return shortest word ladder path from start to target, or None."""
    if start == target:
        return [start]
    if len(start) != len(target):
        return None
    if start not in _WORD_SET or target not in _WORD_SET:
        return None
    candidates = _WORDS_BY_LEN.get(len(start), [])
    queue: deque[list[str]] = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        for w in candidates:
            if w in visited:
                continue
            if _one_letter_apart(path[-1], w):
                new_path = path + [w]
                if w == target:
                    return new_path
                visited.add(w)
                queue.append(new_path)
    return None


def new_game(start: str | None = None, target: str | None = None) -> "WordLadderGame":
    """Create a new game. start/target from Qwen; falls back to hardcoded pairs."""
    sid = str(uuid.uuid4())

    # Try Qwen-suggested pair first
    if start and target:
        s, t = start.lower().strip(), target.lower().strip()
        if (s in _WORD_SET and t in _WORD_SET
                and len(s) == len(t) and s != t
                and _bfs_path(s, t) is not None):
            game = WordLadderGame(
                session_id=sid, start_word=s, target_word=t,
                current_word=s, path=[s],
            )
            _games[sid] = game
            return game
        logger.warning("word_ladder: Qwen pair (%r→%r) failed BFS validation — using fallback", start, target)

    # Try hardcoded fallback pairs
    pairs = list(_FALLBACK_PAIRS)
    random.shuffle(pairs)
    for s, t in pairs:
        if s in _WORD_SET and t in _WORD_SET and _bfs_path(s, t) is not None:
            game = WordLadderGame(
                session_id=sid, start_word=s, target_word=t,
                current_word=s, path=[s],
            )
            _games[sid] = game
            return game

    raise RuntimeError("word_ladder: no valid puzzle found — word set may be empty")


def get_game(session_id: str) -> "WordLadderGame | None":
    return _games.get(session_id)


def step(session_id: str, word: str) -> dict:
    game = _games.get(session_id)
    if not game:
        return {"speech": "Game not found. Please start a new game.", "session_id": ""}

    if game.won or game.lost:
        return game.to_dict(speech="The game is over! Say 'new game' to play again.")

    word = word.strip().lower().rstrip(".,!?")

    if any(w in word for w in ["new game", "give up", "quit", "start over"]):
        return game.to_dict(
            speech=f"No worries! The target was {game.target_word.upper()}. Say 'new game' to try again!"
        )

    if not word.isalpha():
        return game.to_dict(speech="Say a single word — letters only.")

    if len(word) != len(game.current_word):
        return game.to_dict(
            speech=f"Your word must have {len(game.current_word)} letters, same as {game.current_word.upper()}."
        )

    if word == game.current_word:
        return game.to_dict(speech="That's the same word! You need to change exactly one letter.")

    if not _one_letter_apart(game.current_word, word):
        game.wrong_attempts += 1
        if game.wrong_attempts >= MAX_WRONG_STEPS:
            game.lost = True
            return game.to_dict(
                speech=f"{word.upper()} differs by more than one letter. Too many wrong attempts! "
                       f"The target was {game.target_word.upper()}. Say 'new game' to try again!"
            )
        remaining = MAX_WRONG_STEPS - game.wrong_attempts
        return game.to_dict(
            speech=f"{word.upper()} differs by more than one letter from {game.current_word.upper()}. "
                   f"{remaining} wrong {'attempt' if remaining == 1 else 'attempts'} left."
        )

    if word not in _WORD_SET:
        game.wrong_attempts += 1
        if game.wrong_attempts >= MAX_WRONG_STEPS:
            game.lost = True
            return game.to_dict(
                speech=f"Sorry, {word.upper()} isn't in my word list. Too many wrong attempts! "
                       f"The target was {game.target_word.upper()}."
            )
        remaining = MAX_WRONG_STEPS - game.wrong_attempts
        return game.to_dict(
            speech=f"Sorry, {word.upper()} isn't a word I know. "
                   f"{remaining} wrong {'attempt' if remaining == 1 else 'attempts'} left."
        )

    # Valid step!
    game.current_word = word
    game.path.append(word)

    if word == game.target_word:
        game.won = True
        steps = len(game.path) - 1
        step_word = "step" if steps == 1 else "steps"
        return game.to_dict(speech=f"Yes! You reached {game.target_word.upper()} in {steps} {step_word}! Brilliant!")

    return game.to_dict(
        speech=f"Nice! {game.current_word.upper()}. Keep going — change one letter to get to {game.target_word.upper()}."
    )


def hint(session_id: str) -> dict:
    """BFS-based hint: tells which letter position to change next."""
    game = _games.get(session_id)
    if not game:
        return {"speech": "Game not found."}

    if game.won or game.lost:
        return game.to_dict(speech="The game is over!")

    path = _bfs_path(game.current_word, game.target_word)
    if path and len(path) >= 2:
        next_word = path[1]
        ordinals = ["first", "second", "third", "fourth", "fifth"]
        for i, (a, b) in enumerate(zip(game.current_word, next_word)):
            if a != b:
                pos = ordinals[i] if i < len(ordinals) else f"position {i + 1}"
                return game.to_dict(speech=f"Try changing the {pos} letter of {game.current_word.upper()}.")

    return game.to_dict(
        speech=f"Try to think of a word one letter away from {game.current_word.upper()} "
               f"that gets you closer to {game.target_word.upper()}."
    )
