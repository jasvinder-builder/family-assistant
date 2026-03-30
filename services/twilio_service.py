from twilio.rest import Client
from config import settings

_client = None


def _get_client() -> Client:
    global _client
    if _client is None:
        _client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
    return _client


def send_whatsapp(to_number: str, message: str) -> None:
    """Send a WhatsApp text message to a phone number (E.164 format)."""
    _get_client().messages.create(
        from_=settings.twilio_whatsapp_from,
        to=f"whatsapp:{to_number}",
        body=message,
    )


def send_whatsapp_image(to_number: str, image_url: str, caption: str = "") -> None:
    """Send a WhatsApp image message."""
    _get_client().messages.create(
        from_=settings.twilio_whatsapp_from,
        to=f"whatsapp:{to_number}",
        media_url=[image_url],
        body=caption,
    )
