from datetime import datetime
from filelock import FileLock
from config import settings
from models.schemas import TodoItem, EventItem


LOCK_TIMEOUT = 5  # seconds


def _lock_path() -> str:
    return settings.family_md_path + ".lock"


def _read_raw() -> str:
    try:
        with open(settings.family_md_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return "# Family Assistant\n\n## Todos\n\n## Events\n"


def _write_raw(content: str) -> None:
    with open(settings.family_md_path, "w") as f:
        f.write(content)


def _sanitize(text: str) -> str:
    """Remove characters that break the pipe-delimited markdown format."""
    return text.replace("|", "-").replace("\n", " ").strip()


def append_todo(item: TodoItem) -> None:
    due_str = item.due if item.due else "none"
    text = _sanitize(item.text)
    line = f"- [ ] {text} | due: {due_str} | added_by: {item.added_by} | added_at: {item.added_at}\n"
    with FileLock(_lock_path(), timeout=LOCK_TIMEOUT):
        content = _read_raw()
        if "## Events" in content:
            content = content.replace("## Events", line + "\n## Events")
        else:
            content = content.rstrip() + "\n" + line + "\n"
        _write_raw(content)


def append_event(item: EventItem) -> None:
    title = _sanitize(item.title)
    line = f"- {item.event_datetime} | {title} | added_by: {item.added_by} | added_at: {item.added_at}\n"
    with FileLock(_lock_path(), timeout=LOCK_TIMEOUT):
        content = _read_raw()
        content = content.rstrip() + "\n" + line + "\n"
        _write_raw(content)


def complete_todo(todo_text: str) -> bool:
    """Mark a todo as complete by matching its text. Returns True if found and updated."""
    now = datetime.now().isoformat(timespec="seconds")
    with FileLock(_lock_path(), timeout=LOCK_TIMEOUT):
        content = _read_raw()
        lines = content.splitlines(keepends=True)
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith("- [ ]"):
                continue
            parts = [p.strip() for p in stripped[5:].split("|")]
            if parts and parts[0].lower() == todo_text.lower():
                lines[i] = line.replace("- [ ]", "- [x]", 1).rstrip("\n") + f" | completed_at: {now}\n"
                _write_raw("".join(lines))
                return True
    return False


def read_todos() -> list[TodoItem]:
    content = _read_raw()
    todos = []
    in_todos = False
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if line == "## Todos":
            in_todos = True
            continue
        if line.startswith("## ") and in_todos:
            break
        if not in_todos:
            continue
        completed = line.startswith("- [x]")
        if not (line.startswith("- [ ]") or completed):
            continue
        line = line[5:].strip()
        parts = [p.strip() for p in line.split("|")]
        fields = {"text": parts[0], "completed": completed}
        for part in parts[1:]:
            if ": " in part:
                k, v = part.split(": ", 1)
                fields[k.strip()] = v.strip()
        todos.append(TodoItem(
            text=fields.get("text", ""),
            due=fields.get("due") if fields.get("due") != "none" else None,
            added_by=fields.get("added_by", ""),
            added_at=fields.get("added_at", ""),
            completed=completed,
            completed_at=fields.get("completed_at"),
        ))
    return todos


def read_events(after: datetime | None = None, before: datetime | None = None) -> list[EventItem]:
    content = _read_raw()
    events = []
    in_events = False
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if line == "## Events":
            in_events = True
            continue
        if line.startswith("## ") and in_events:
            break
        if not in_events or not line.startswith("- 20"):
            continue
        line = line[2:].strip()
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        event_dt_str = parts[0].strip()
        fields = {"title": parts[1].strip() if len(parts) > 1 else ""}
        for part in parts[2:]:
            if ": " in part:
                k, v = part.split(": ", 1)
                fields[k.strip()] = v.strip()
        try:
            event_dt = datetime.fromisoformat(event_dt_str)
        except ValueError:
            continue
        if after and event_dt < after:
            continue
        if before and event_dt > before:
            continue
        events.append(EventItem(
            title=fields.get("title", ""),
            event_datetime=event_dt_str,
            human_readable=f"{event_dt.strftime('%A %B')} {event_dt.day} at {event_dt.strftime('%I:%M %p')}",
            added_by=fields.get("added_by", ""),
            added_at=fields.get("added_at", ""),
        ))
    return events


def delete_todo(text: str) -> bool:
    """Remove a todo line matching text (pending or completed). Returns True if found."""
    with FileLock(_lock_path(), timeout=LOCK_TIMEOUT):
        content = _read_raw()
        lines = content.splitlines(keepends=True)
        new_lines = []
        found = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("- [ ]") or stripped.startswith("- [x]"):
                parts = [p.strip() for p in stripped[5:].split("|")]
                if parts and parts[0].lower() == text.lower():
                    found = True
                    continue
            new_lines.append(line)
        if found:
            _write_raw("".join(new_lines))
        return found


def delete_event(title: str, event_datetime: str) -> bool:
    """Remove an event line matching title and datetime. Returns True if found."""
    with FileLock(_lock_path(), timeout=LOCK_TIMEOUT):
        content = _read_raw()
        lines = content.splitlines(keepends=True)
        new_lines = []
        found = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(f"- {event_datetime}"):
                parts = [p.strip() for p in stripped[2:].split("|")]
                if len(parts) >= 2 and parts[1].strip().lower() == title.lower():
                    found = True
                    continue
            new_lines.append(line)
        if found:
            _write_raw("".join(new_lines))
        return found


def read_all_data() -> dict:
    """Return all todos and events for the dashboard."""
    return {
        "todos": read_todos(),
        "events": read_events(),
    }
