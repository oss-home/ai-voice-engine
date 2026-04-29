"""
Silero VAD — voice activity detection.

Used to detect utterance boundaries in the incoming 8 kHz PCM stream
so we know exactly when to run Whisper (only when someone finished
speaking, not on every frame).

Silero VAD expects:
  - 16-bit signed PCM
  - 8 kHz or 16 kHz
  - Chunk size: 256 samples @ 8 kHz  OR  512 samples @ 16 kHz

We use 8 kHz chunks (256 samples = 32 ms per chunk).
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from config import settings

log = logging.getLogger(__name__)

CHUNK_SAMPLES = 256   # 32 ms @ 8 kHz
CHUNK_BYTES   = CHUNK_SAMPLES * 2  # int16


class SileroVAD:
    def __init__(self) -> None:
        self._model = None
        self._threshold = settings.vad_threshold
        # How many consecutive silent chunks = end of utterance
        self._silence_chunks = max(1, settings.silence_ms // 32)

    def load(self) -> None:
        log.info("Loading Silero VAD …")
        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self._model.eval()
        log.info("Silero VAD ready.")

    def is_speech(self, pcm_chunk: bytes) -> bool:
        """Return True if this 32 ms chunk contains speech."""
        if self._model is None:
            return True   # failsafe: assume speech if VAD not loaded
        arr = np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        with torch.no_grad():
            prob = self._model(tensor, 8000).item()
        return prob >= self._threshold

    @property
    def silence_chunks(self) -> int:
        return self._silence_chunks
