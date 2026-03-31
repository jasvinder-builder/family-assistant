import random
from twilio.twiml.voice_response import VoiceResponse, Record

VOICE = "Polly.Joanna"
RECORD_MAX_LENGTH = 15  # seconds — enough for any reasonable command

_FILLERS = [
    "Sure, let me check that for you...",
    "One moment, looking into that...",
    "Give me just a second...",
    "Let me think about that...",
    "On it, just a moment...",
]


def voice_filler_redirect(sid: str) -> str:
    """Immediately return a filler phrase then redirect to the answer endpoint."""
    response = VoiceResponse()
    response.say(random.choice(_FILLERS), voice=VOICE)
    response.redirect(f"/voice/answer/{sid}", method="GET")
    return str(response)


def voice_gather(prompt: str, action: str = "/voice/transcription", timeout: int = 4) -> str:
    """Speak a prompt then open the mic for recording (sent to Whisper for STT)."""
    response = VoiceResponse()
    response.say(prompt, voice=VOICE)
    response.record(
        action=action,
        max_length=RECORD_MAX_LENGTH,
        play_beep=False,
        timeout=timeout,
    )
    # Fallback if no recording received
    response.say("I didn't hear anything. Please call back and try again.", voice=VOICE)
    return str(response)


def voice_say_then_gather(
    text: str,
    follow_up: str = "Anything else?",
    action: str = "/voice/transcription",
) -> str:
    """Speak an answer then reopen the mic for the next question."""
    response = VoiceResponse()
    response.say(f"{text} ... {follow_up}", voice=VOICE)
    response.record(
        action=action,
        max_length=RECORD_MAX_LENGTH,
        play_beep=False,
        timeout=3,
    )
    # Fallback if caller goes silent
    response.say("I didn't hear anything. Take care!", voice=VOICE)
    response.hangup()
    return str(response)


def voice_research_answer(voice_text: str, wid: str) -> str:
    """Speak a short research summary and offer full details on WhatsApp."""
    return voice_say_then_gather(
        voice_text,
        follow_up="Want the full details sent to your WhatsApp?",
        action=f"/voice/research-whatsapp-choice/{wid}",
    )


def voice_research_pending(rid: str) -> str:
    """Ask caller to wait or get results on WhatsApp while research runs."""
    response = VoiceResponse()
    response.say(
        "I'm still pulling that together. Say 'wait' to hold on a bit longer, "
        "or say 'WhatsApp' to get the results sent there.",
        voice=VOICE,
    )
    response.record(
        action=f"/voice/research-choice/{rid}",
        max_length=5,
        play_beep=False,
        timeout=3,
    )
    # Fallback if no speech
    response.say("I didn't catch that — I'll send the results to your WhatsApp.", voice=VOICE)
    response.hangup()
    return str(response)


def voice_say_hangup(text: str) -> str:
    """Speak text then hang up."""
    response = VoiceResponse()
    response.say(text, voice=VOICE)
    response.hangup()
    return str(response)
