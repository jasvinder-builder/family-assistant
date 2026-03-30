from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class IntentResult(BaseModel):
    intent: str  # add_todo, add_event, query_tasks, query_events, research_web, research_jobs, research_images, unknown
    subtype: Optional[str] = None
    query: Optional[str] = None  # for research intents
    confidence: float = 1.0


class TodoItem(BaseModel):
    text: str
    due: Optional[str] = None  # ISO date string or None
    added_by: str
    added_at: str  # ISO datetime string
    completed: bool = False
    completed_at: Optional[str] = None


class EventItem(BaseModel):
    title: str
    event_datetime: str  # ISO datetime string
    human_readable: str  # e.g. "Tuesday April 10th at 2pm" — read back to caller
    added_by: str
    added_at: str
