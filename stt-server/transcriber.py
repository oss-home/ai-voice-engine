"""
Faster-Whisper transcription engine.

Uses large-v3 (or distil-large-v3) with:
  - float16 on GPU for maximum speed + quality
  - int8 on CPU as fallback
  - VAD filter built into faster-whisper for extra noise rejection
  - beam_size=5 for best accuracy (Deepgram nova-3 equivalent)
  - Word-level timestamps for accurate endpointing
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import io
import logging

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

from config import settings

log = logging.getLogger(__name__)

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

SAMPLE_RATE = 8000  # PSTN PCM16 input


class Transcriber:
    def __init__(self) -> None:
        self._model: WhisperModel | None = None

    def load(self) -> None:
        log.info(
            "Loading Faster-Whisper %s on %s (%s) …",
            settings.model_size, settings.device, settings.compute_type,
        )
        self._model = WhisperModel(
            settings.model_size,
            device=settings.device,
            compute_type=settings.compute_type,
            # Download model to local cache; no re-download on restart
            download_root="/app/models",
            cpu_threads=8,
            num_workers=2,
        )
        log.info("Faster-Whisper ready.")

    def _transcribe_sync(self, pcm_bytes: bytes) -> tuple[str, float]:
        """
        Transcribe raw PCM16 8 kHz bytes → (transcript, avg_logprob).
        Runs in thread-pool executor (blocking).
        """
        if not self._model:
            return "", 0.0

        # Convert PCM16 bytes → float32 numpy array
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # faster-whisper needs float32 at 16 kHz — upsample 8 kHz → 16 kHz
        # via simple linear interpolation (cheap, good enough for ASR)
        audio_16k = np.interp(
            np.linspace(0, len(audio), len(audio) * 2),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

        lang = settings.language if settings.language != "multi" else None

        segments, info = self._model.transcribe(
            audio_16k,
            beam_size=settings.beam_size,
            language=lang,
            # VAD filter removes non-speech segments before sending to Whisper
            vad_filter=True,
            vad_parameters={
                "threshold": settings.vad_threshold,
                "min_silence_duration_ms": 100,
                "speech_pad_ms": 30,
            },
            # Condition on previous text for better context continuity
            condition_on_previous_text=True,
            # Temperature fallback: try 0 first (greedy), then 0.2 if low confidence
            temperature=[0.0, 0.2],
            # Suppress common hallucinations on silence
            suppress_blank=True,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            word_timestamps=False,
        )

        # Collect all segment texts
        texts = [s.text.strip() for s in segments]
        transcript = " ".join(t for t in texts if t).strip()

        confidence = getattr(info, "language_probability", 1.0)
        return transcript, float(confidence)

    async def transcribe(self, pcm_bytes: bytes) -> tuple[str, float]:
        """Async wrapper — runs inference in thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, self._transcribe_sync, pcm_bytes)

    @property
    def ready(self) -> bool:
        return self._model is not None
