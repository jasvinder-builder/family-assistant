import asyncio
from dataclasses import dataclass, field


@dataclass
class Session:
    event: asyncio.Event = field(default_factory=asyncio.Event)
    result: str | None = None


_sessions: dict[str, Session] = {}


def create(sid: str) -> Session:
    session = Session()
    _sessions[sid] = session
    return session


def get(sid: str) -> Session | None:
    return _sessions.get(sid)


def delete(sid: str) -> None:
    _sessions.pop(sid, None)
