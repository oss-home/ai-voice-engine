"""
STT Server — FastAPI + WebSocket

Implements the Deepgram Live Transcription WebSocket protocol so
voice-sip's existing stt.py needs zero changes — just point
DEEPGRAM_URL to this server.

Protocol:
  Client → binary PCM16 8 kHz audio frames (any chunk size)
  Server → JSON transcripts:
      {
        "type": "Results",
        "is_final": true | false,
        "speech_final": true | false,
        "channel": {
          "alternatives": [{"transcript": "...", "confidence": 0.95}]
        }
      }

Lifecycle:
  1. Accept WebSocket at /v1/listen
  2. Buffer incoming audio; run Silero VAD on each 32 ms chunk
  3. While VAD says "speech": accumulate buffer, emit partials every
     PARTIAL_INTERVAL_MS using a quick run of Whisper on what we have
  4. When VAD says silence for SILENCE_MS: run full Whisper on buffer,
     emit is_final=true, clear buffer
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from config import settings
from transcriber import Transcriber
from vad import CHUNK_BYTES, SileroVAD

log = logging.getLogger(__name__)

# ── globals ───────────────────────────────────────────────────────────────────

_transcriber = Transcriber()
_vad         = SileroVAD()


@asynccontextmanager
async def lifespan(_: FastAPI):
    _vad.load()
    _transcriber.load()
    yield


app = FastAPI(title="AI Voice STT", version="1.0.0", lifespan=lifespan)


# ── helpers ───────────────────────────────────────────────────────────────────

def _dg_message(transcript: str, is_final: bool, confidence: float = 0.95) -> str:
    """Build a Deepgram-compatible JSON transcript message."""
    return json.dumps({
        "type": "Results",
        "is_final": is_final,
        "speech_final": is_final,
        "channel": {
            "alternatives": [
                {"transcript": transcript, "confidence": confidence}
            ]
        },
    })


# ── health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "ok": _transcriber.ready,
        "model": settings.model_size,
        "device": settings.device,
        "language": settings.language,
    }


# ── WebSocket STT ─────────────────────────────────────────────────────────────

@app.websocket("/v1/listen")
async def stt_ws(ws: WebSocket):
    """
    Deepgram-compatible streaming STT endpoint.

    voice-sip connects here instead of wss://api.deepgram.com/v1/listen.
    The JSON transcript format is identical — no changes needed in stt.py.
    """
    await ws.accept()
    log.info("STT WebSocket connected")

    audio_buffer     = bytearray()
    raw_chunk        = bytearray()      # accumulates until CHUNK_BYTES
    silent_count     = 0
    in_speech        = False
    last_partial_ts  = time.monotonic()

    partial_interval = settings.partial_interval_ms / 1000.0  # seconds

    try:
        async for data in ws.iter_bytes():
            if not data:
                continue

            # ── accumulate into 32 ms VAD chunks ─────────────────────────
            raw_chunk.extend(data)

            while len(raw_chunk) >= CHUNK_BYTES:
                chunk = bytes(raw_chunk[:CHUNK_BYTES])
                raw_chunk = raw_chunk[CHUNK_BYTES:]

                is_speech = _vad.is_speech(chunk)

                if is_speech:
                    silent_count = 0
                    in_speech    = True
                    audio_buffer.extend(chunk)

                    # ── emit partial every PARTIAL_INTERVAL_MS ─────────
                    now = time.monotonic()
                    if (now - last_partial_ts) >= partial_interval and len(audio_buffer) > CHUNK_BYTES * 4:
                        try:
                            partial_text, conf = await _transcriber.transcribe(bytes(audio_buffer))
                            if partial_text:
                                await ws.send_text(_dg_message(partial_text, is_final=False, confidence=conf))
                                last_partial_ts = now
                        except Exception as exc:
                            log.warning("partial transcribe error: %s", exc)

                else:
                    # silence chunk
                    if in_speech:
                        silent_count += 1
                        audio_buffer.extend(chunk)  # include the silence tail

                        if silent_count >= _vad.silence_chunks:
                            # ── utterance ended — run full Whisper ────────
                            if len(audio_buffer) > CHUNK_BYTES * 2:
                                try:
                                    text, conf = await _transcriber.transcribe(bytes(audio_buffer))
                                    if text:
                                        log.info("FINAL: %s (conf=%.2f)", text, conf)
                                        await ws.send_text(_dg_message(text, is_final=True, confidence=conf))
                                except Exception as exc:
                                    log.error("transcribe error: %s", exc)

                            audio_buffer.clear()
                            in_speech       = False
                            silent_count    = 0
                            last_partial_ts = time.monotonic()

    except WebSocketDisconnect:
        log.info("STT WebSocket disconnected")
    except Exception as exc:
        log.error("STT WebSocket error: %s", exc)
    finally:
        # Flush any remaining audio as a final transcript
        if audio_buffer and len(audio_buffer) > CHUNK_BYTES:
            try:
                text, conf = await _transcriber.transcribe(bytes(audio_buffer))
                if text:
                    await ws.send_text(_dg_message(text, is_final=True, confidence=conf))
            except Exception:
                pass
        log.info("STT session ended")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, log_level="info")
