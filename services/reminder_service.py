import json
import logging
from datetime import datetime, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from config import settings
from services import markdown_service, twilio_service

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()

SCAN_INTERVAL_MINUTES = 30
REMINDER_OFFSETS = [
    (24 * 60, "24 hours"),
    (4 * 60, "4 hours"),
]


def start() -> None:
    scheduler.add_job(
        _scan_and_schedule,
        IntervalTrigger(minutes=SCAN_INTERVAL_MINUTES),
        id="event_scanner",
        replace_existing=True,
        next_run_time=datetime.now(),  # run immediately on startup
    )
    scheduler.start()
    logger.info("Reminder scheduler started (scan every %d min).", SCAN_INTERVAL_MINUTES)


def stop() -> None:
    scheduler.shutdown()
    logger.info("Reminder scheduler stopped.")


async def _scan_and_schedule() -> None:
    now = datetime.now()
    window_end = now + timedelta(hours=48)
    events = markdown_service.read_events(after=now, before=window_end)

    new_jobs = 0
    for event in events:
        try:
            event_dt = datetime.fromisoformat(event.event_datetime)
        except ValueError:
            continue

        for offset_minutes, label in REMINDER_OFFSETS:
            remind_at = event_dt - timedelta(minutes=offset_minutes)
            if remind_at <= now:
                continue  # window already passed

            job_id = f"reminder_{event.event_datetime}_{offset_minutes}".replace(":", "-")
            if scheduler.get_job(job_id):
                continue  # already scheduled

            scheduler.add_job(
                _send_reminder,
                DateTrigger(run_date=remind_at),
                id=job_id,
                kwargs={"title": event.title, "label": label, "event_dt": event_dt},
            )
            new_jobs += 1
            logger.info(
                "Scheduled %s reminder for '%s' at %s", label, event.title, remind_at.strftime("%Y-%m-%d %H:%M")
            )

    logger.info("Reminder scan complete: %d new reminder(s) scheduled.", new_jobs)


async def _send_reminder(title: str, label: str, event_dt: datetime) -> None:
    time_str = f"{event_dt.strftime('%A, %B')} {event_dt.day} at {event_dt.strftime('%I:%M %p').lstrip('0')}"
    message = f"Reminder: *{title}* is in {label} ({time_str})."

    try:
        mapping = json.loads(settings.phone_to_name)
    except Exception:
        logger.error("Could not parse PHONE_TO_NAME — no reminders sent.")
        return

    for phone in mapping:
        try:
            twilio_service.send_whatsapp(phone, message)
            logger.info("Sent %s reminder for '%s' to %s", label, title, phone)
        except Exception:
            logger.exception("Failed to send reminder to %s", phone)
