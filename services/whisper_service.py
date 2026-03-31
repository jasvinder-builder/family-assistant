import logging
import math
import tempfile
import os
from faster_whisper import WhisperModel
from config import settings

logger = logging.getLogger(__name__)

# Model loaded eagerly at import time so it's ready before the first call.
# large-v3 on GPU with int8_float16: ~1.5GB VRAM, ~1-2s inference.
logger.info("Loading Whisper model '%s' on CUDA...", settings.whisper_model_size)
_model = WhisperModel(
    settings.whisper_model_size,
    device="cuda",
    compute_type="int8_float16",
)
logger.info("Whisper model loaded.")


def _get_model() -> WhisperModel:
    return _model


def transcribe(audio_bytes: bytes, suffix: str = ".wav") -> tuple[str, float]:
    """Transcribe audio bytes. Returns (transcript_text, avg_confidence).
    Pass suffix='.webm' for browser MediaRecorder output (webm/opus)."""
    model = _get_model()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        segments, info = model.transcribe(tmp_path, beam_size=5, language="en")
        words = []
        logprobs = []
        for segment in segments:
            words.append(segment.text.strip())
            logprobs.append(segment.avg_logprob)

        transcript = " ".join(words).strip()
        # avg_logprob is negative (log probability); convert to 0-1 with exp()
        avg_logprob = sum(logprobs) / len(logprobs) if logprobs else -1.0
        avg_confidence = math.exp(avg_logprob)
        logger.info("Whisper transcript (%.2f): %s", avg_confidence, transcript)
        return transcript, avg_confidence
    finally:
        os.unlink(tmp_path)
