"""
TTS Server — FastAPI

POST /synthesize   → PCM16 8 kHz binary (voice-sip compatible)
GET  /health       → JSON status
GET  /voices       → list cloned voices
GET  /emotions     → list emotion profiles + parameters

Emotion tokens supported inline in `text`:
    [breath] [breath:deep] [laugh:soft] [laugh:full]
    [sigh] [hmm] [pause:N] [emotion:X]

See emotion.py for full reference.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from pydantic import BaseModel, Field

from audio_utils import silence_frames
from config import settings
from emotion import EMOTION_PROFILES, AudioEvent, TextSegment, parse
from model import VoiceModel

# ── global model ──────────────────────────────────────────────────────────────

_model = VoiceModel(device=settings.device, voices_dir=settings.voices_dir)

# ── auth ──────────────────────────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False)


def _check_auth(creds: HTTPAuthorizationCredentials | None = Security(_bearer)):
    if not settings.api_token:
        return  # auth disabled
    if creds is None or creds.credentials != settings.api_token:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_: FastAPI):
    await _model.load()
    yield


app = FastAPI(title="AI Voice TTS", version="1.0.0", lifespan=lifespan)

# ── schemas ───────────────────────────────────────────────────────────────────


class SynthRequest(BaseModel):
    text: str = Field(..., description="Text with optional [emotion:X] / [laugh:soft] tokens")
    voice_id: str = Field(default="ahana")
    emotion: str  = Field(default="neutral", description="Default emotion for segments with no [emotion:X] token")


# ── routes ────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {
        "ok": _model.ready,
        "device": settings.device,
        "voices": _model.list_voices(),
        "emotions": list(EMOTION_PROFILES.keys()),
    }


@app.get("/voices")
async def voices(_=Depends(_check_auth)):
    return {"voices": _model.list_voices()}


@app.get("/emotions")
async def emotions():
    return {"emotions": EMOTION_PROFILES}


@app.post("/synthesize", dependencies=[Depends(_check_auth)])
async def synthesize(req: SynthRequest):
    """
    Synthesize `text` to PCM16 mono 8 kHz binary.

    Returns raw bytes (application/octet-stream) — 320-byte frames,
    ready for mod_audio_stream / voice-sip.
    """
    if not _model.ready:
        raise HTTPException(503, "Model not loaded yet")

    if req.voice_id not in _model.list_voices():
        raise HTTPException(404, f"Voice '{req.voice_id}' not found. Available: {_model.list_voices()}")

    segments = parse(req.text, default_emotion=req.emotion)
    all_frames: list[bytes] = []

    for seg in segments:
        if isinstance(seg, AudioEvent):
            if seg.kind == "pause":
                all_frames.extend(silence_frames(seg.duration_ms))
            else:
                all_frames.extend(_model.get_event(seg.kind, req.voice_id))

        elif isinstance(seg, TextSegment) and seg.text.strip():
            frames = await _model.synthesize(
                text=seg.text,
                voice_id=req.voice_id,
                exaggeration=seg.exaggeration,
                cfg_weight=seg.cfg_weight,
            )
            all_frames.extend(frames)

    if not all_frames:
        return Response(content=b"", media_type="application/octet-stream")

    return Response(
        content=b"".join(all_frames),
        media_type="application/octet-stream",
        headers={"X-Frame-Count": str(len(all_frames))},
    )


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, log_level="info")
