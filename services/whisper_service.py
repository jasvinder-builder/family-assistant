"""
Whisper STT client — delegates to the whisper microservice over HTTP.

The whisper container (Dockerfile.whisper) owns faster-whisper and CTranslate2.
This module is a thin httpx wrapper so the main app process has zero CUDA deps.
"""

import logging
import httpx
from config import settings

logger = logging.getLogger(__name__)


def transcribe(audio_bytes: bytes, suffix: str = ".webm") -> tuple[str, float]:
    """Send audio bytes to the whisper service. Returns (transcript, confidence).

    Called via asyncio.to_thread() in main.py — httpx.post is synchronous by design.
    """
    response = httpx.post(
        f"{settings.whisper_url}/transcribe",
        files={"audio": (f"recording{suffix}", audio_bytes, "application/octet-stream")},
        data={"suffix": suffix},
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()
    transcript = data.get("transcript", "")
    confidence = float(data.get("confidence", 0.0))
    logger.info("Whisper (%.2f): %s", confidence, transcript)
    return transcript, confidence
