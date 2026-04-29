"""
TTS Server — FastAPI

POST /synthesize          → PCM16 8 kHz binary (voice-sip / phone calls)
POST /synthesize/preview  → WAV file 22 kHz (high-quality testing)
GET  /health              → JSON status
GET  /voices              → list cloned voices
GET  /emotions            → list emotion profiles + parameters

Emotion tokens supported inline in `text`:
    [breath] [breath:deep] [laugh:soft] [laugh:full]
    [sigh] [sigh:deep] [hmm] [hmm:long] [um] [uh] [oh]
    [pause:N]  [emotion:X]

See emotion.py for full reference.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from pydantic import BaseModel, Field

from audio_utils import HQ_SAMPLE_RATE, build_wav_bytes, silence_frames
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


app = FastAPI(title="AI Voice TTS", version="2.0.0", lifespan=lifespan)

# ── schemas ───────────────────────────────────────────────────────────────────


class SynthRequest(BaseModel):
    text: str  = Field(..., description="Text with optional [emotion:X] / [laugh:soft] tokens")
    voice_id: str  = Field(default="ahana")
    emotion: str   = Field(default="warm", description="Default emotion for untagged segments")


# ── helpers ───────────────────────────────────────────────────────────────────

# Tiny gap inserted before each speech segment so audio events (breath, laugh)
# don't click directly into speech — sounds more natural at phone quality.
_SEGMENT_GAP_MS = 15


async def _render_segments(
    req: SynthRequest,
    hq: bool = False,
) -> tuple[list[bytes], list[np.ndarray]]:
    """
    Parse and render all segments from req.text.

    Returns:
        pstn_frames  — 8 kHz PCM frames (for /synthesize)
        hq_chunks    — 22 kHz PCM arrays (for /synthesize/preview)
    Only one of these is populated depending on the `hq` flag.
    """
    segments = parse(req.text, default_emotion=req.emotion)
    pstn_frames: list[bytes]      = []
    hq_chunks:   list[np.ndarray] = []

    for seg in segments:

        if isinstance(seg, AudioEvent):
            if seg.kind == "pause":
                if hq:
                    # Insert silence at 22 kHz
                    n = int(HQ_SAMPLE_RATE * seg.duration_ms / 1000)
                    hq_chunks.append(np.zeros(n, dtype=np.int16))
                else:
                    pstn_frames.extend(silence_frames(seg.duration_ms))
            else:
                if hq:
                    # Audio events were pre-built at 8 kHz; skip them in HQ
                    # preview (just add a short silence as placeholder).
                    n = int(HQ_SAMPLE_RATE * 0.20)   # ~200 ms placeholder
                    hq_chunks.append(np.zeros(n, dtype=np.int16))
                else:
                    pstn_frames.extend(_model.get_event(seg.kind, req.voice_id))

        elif isinstance(seg, TextSegment) and seg.text.strip():
            # Small natural gap before each speech chunk
            if hq:
                if hq_chunks:
                    n = int(HQ_SAMPLE_RATE * _SEGMENT_GAP_MS / 1000)
                    hq_chunks.append(np.zeros(n, dtype=np.int16))
                chunk = await _model.synthesize_hq(
                    text=seg.text,
                    voice_id=req.voice_id,
                    exaggeration=seg.exaggeration,
                    cfg_weight=seg.cfg_weight,
                )
                if chunk.size:
                    hq_chunks.append(chunk)
            else:
                if pstn_frames:
                    pstn_frames.extend(silence_frames(_SEGMENT_GAP_MS))
                frames = await _model.synthesize(
                    text=seg.text,
                    voice_id=req.voice_id,
                    exaggeration=seg.exaggeration,
                    cfg_weight=seg.cfg_weight,
                )
                pstn_frames.extend(frames)

    return pstn_frames, hq_chunks


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
    Synthesize text to raw PCM16 mono 8 kHz (PSTN / phone quality).

    Returns binary application/octet-stream — 320-byte frames,
    ready for mod_audio_stream / voice-sip.
    """
    if not _model.ready:
        raise HTTPException(503, "Model not loaded yet")
    if req.voice_id not in _model.list_voices():
        raise HTTPException(404, f"Voice '{req.voice_id}' not found. Available: {_model.list_voices()}")

    pstn_frames, _ = await _render_segments(req, hq=False)

    if not pstn_frames:
        return Response(content=b"", media_type="application/octet-stream")

    return Response(
        content=b"".join(pstn_frames),
        media_type="application/octet-stream",
        headers={"X-Frame-Count": str(len(pstn_frames))},
    )


@app.post("/synthesize/preview", dependencies=[Depends(_check_auth)])
async def synthesize_preview(req: SynthRequest):
    """
    Synthesize text to a 22 kHz WAV file (high-quality, for testing).

    Use this endpoint to evaluate voice quality without the 8 kHz
    telephone downgrade. Download and play in any audio player.
    Returns audio/wav binary.
    """
    if not _model.ready:
        raise HTTPException(503, "Model not loaded yet")
    if req.voice_id not in _model.list_voices():
        raise HTTPException(404, f"Voice '{req.voice_id}' not found. Available: {_model.list_voices()}")

    _, hq_chunks = await _render_segments(req, hq=True)

    if not hq_chunks:
        return Response(content=b"", media_type="audio/wav")

    combined = np.concatenate(hq_chunks).astype(np.int16)
    wav_bytes = build_wav_bytes(combined, HQ_SAMPLE_RATE)

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'attachment; filename="preview.wav"',
            "X-Sample-Rate": str(HQ_SAMPLE_RATE),
        },
    )


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, log_level="info")
