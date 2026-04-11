"""
Whisper microservice — faster-whisper large-v3 on CUDA.

POST /transcribe  multipart: audio (file) + suffix (form field, default .webm)
                  → {"transcript": "...", "confidence": 0.92}

GET  /health      → {"status": "ok", "model": "large-v3"}  (503 until model ready)
"""

import logging
import math
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
ALLOWED_SUFFIXES = {".webm", ".wav", ".mp3", ".ogg", ".m4a", ".mp4", ".flac"}

_model: WhisperModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    logger.info("Loading Whisper model '%s' on CUDA...", WHISPER_MODEL_SIZE)
    _model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="int8_float16")
    logger.info("Whisper model loaded.")
    yield
    _model = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ok", "model": WHISPER_MODEL_SIZE}


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    suffix: str = Form(default=".webm"),
):
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported suffix: {suffix}")
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    audio_bytes = await audio.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        segments, _ = _model.transcribe(tmp_path, beam_size=5, language="en")
        words, logprobs = [], []
        for segment in segments:
            words.append(segment.text.strip())
            logprobs.append(segment.avg_logprob)

        transcript = " ".join(words).strip()
        avg_logprob = sum(logprobs) / len(logprobs) if logprobs else -1.0
        confidence = round(math.exp(avg_logprob), 4)
        logger.info("Transcribed (%.2f): %s", confidence, transcript)
        return {"transcript": transcript, "confidence": confidence}
    finally:
        os.unlink(tmp_path)
