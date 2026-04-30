"""
F5-TTS voice model — replaces Kokoro-82M.

Why F5-TTS over Kokoro:
  - TRUE zero-shot voice cloning: conditions the entire generation on the
    reference waveform, not just timbre.  Kokoro only copies speaking style;
    F5-TTS preserves the reference speaker's ACCENT as well.
  - Ahana's ElevenLabs reference audio has Indian-English accent.
    F5-TTS output therefore speaks with Indian accent — Kokoro did not.
  - RTF ~0.12-0.18 on T4 (vs Kokoro 0.034) — acceptable because
      (a) calls have ~3% TTS duty cycle  → 300 calls ≈ 9 concurrent synths
      (b) greetings + fillers are pre-generated at startup (zero live latency)
      (c) streaming from LLM → first chunk plays before rest is ready

Voice loading (per voice directory under voices_dir):
  1. voices/{name}/reference.mp3  (or .wav/.m4a/.flac) — cloning source
  2. voices/{name}/reference.txt  — optional pre-transcription (faster startup)
     If missing, F5-TTS auto-transcribes via its built-in ASR on first call.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from pathlib import Path

import numpy as np
import torch
import torchaudio
from loguru import logger

from audio_utils import (
    HQ_SAMPLE_RATE,
    pcm_to_frames,
    resample_to_hq,
    resample_to_pstn,
    silence_frames,
)
from config import settings
from emotion import EMOTION_PROFILES

# ── thread pool ───────────────────────────────────────────────────────────────
# F5-TTS releases the GIL during CUDA ops; the thread pool keeps FastAPI
# non-blocking.  4 workers = 4 concurrent synthesis ops per GPU.
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=settings.max_workers)

# ── constants ─────────────────────────────────────────────────────────────────

# F5-TTS v1 outputs 24 kHz float32; we resample to 8 kHz PCM16 for telephony.
F5_SAMPLE_RATE = 24_000

# Number of ODE solver steps.  32 = good quality/speed balance on T4.
# Drop to 16 for ~2x speedup at minor quality cost; raise to 64 for studio.
NFE_STEP = 32

# Emotion → F5-TTS speed map (same formula as Kokoro era for API compat).
_EMOTION_SPEEDS: dict[str, float] = {
    name: max(0.80, min(1.25, 0.85 + p["exaggeration"] * 0.40))
    for name, p in EMOTION_PROFILES.items()
}
_EMOTION_SPEEDS.update({
    "empathy":      0.88,
    "gentle":       0.82,
    "sad":          0.80,
    "professional": 1.00,
    "urgent":       1.20,
    "enthusiastic": 1.22,
})

# Audio events pre-synthesised at startup so calls have zero event latency.
_AUDIO_EVENT_SPECS: dict[str, tuple[str, float]] = {
    "breath":      ("Mm.",          1.20),
    "breath_deep": ("Mmm.",         1.12),
    "hmm":         ("Hmm.",         1.00),
    "hmm_long":    ("Hmmm.",        0.92),
    "sigh":        ("Haah.",        0.85),
    "sigh_deep":   ("Haaaah.",      0.80),
    "laugh_soft":  ("Hehe.",        1.10),
    "laugh_full":  ("Ha ha ha ha!", 1.15),
    "um":          ("Um,",          0.95),
    "uh":          ("Uh,",          0.95),
    "oh":          ("Oh!",          1.10),
}


# ── VoiceModel ────────────────────────────────────────────────────────────────

class VoiceModel:
    """Thread-safe F5-TTS with Indian-accent voice cloning + audio-event cache."""

    def __init__(self, device: str, voices_dir: str) -> None:
        self.device     = device
        self.voices_dir = Path(voices_dir)
        self._tts       = None   # F5TTS instance
        # voice_id -> {"ref_file": str, "ref_text": str}
        self._voices: dict[str, dict] = {}
        # "voice_id:event_name" -> list[bytes]  (pre-built 8 kHz PCM16 frames)
        self._event_cache: dict[str, list[bytes]] = {}

    # ── startup ───────────────────────────────────────────────────────────────

    async def load(self) -> None:
        logger.info(f"Loading F5-TTS on {self.device} ...")
        loop      = asyncio.get_running_loop()
        self._tts = await loop.run_in_executor(_executor, self._load_model)
        logger.success("F5-TTS loaded.")

        self._discover_voices()

        for voice_id in list(self._voices):
            await self._prebuild_events(voice_id)

    def _load_model(self):
        """Download / load F5-TTS weights (blocking -- runs in thread pool)."""
        from f5_tts.api import F5TTS

        device = self.device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available -- falling back to CPU")
            device = "cpu"

        tts = F5TTS(
            model_type="F5TTS_v1_Base",
            device=device,
        )
        logger.info(f"  F5-TTS device: {device}")
        return tts

    # ── voice discovery ───────────────────────────────────────────────────────

    def _discover_voices(self) -> None:
        """
        Scan voices_dir.  For each sub-directory that has a reference audio file,
        register it as a cloning source.  No voice.pt needed -- F5-TTS conditions
        directly on the reference waveform at synthesis time.
        """
        if not self.voices_dir.exists():
            logger.warning(f"voices_dir not found: {self.voices_dir}")
            return

        for d in sorted(self.voices_dir.iterdir()):
            if not d.is_dir():
                continue
            voice_id = d.name

            for ext in ("mp3", "wav", "m4a", "flac", "ogg"):
                ref = d / f"reference.{ext}"
                if not ref.exists():
                    continue

                ref_text = self._load_ref_text(d)
                self._voices[voice_id] = {
                    "ref_file": str(ref),
                    "ref_text": ref_text,
                }
                preview = (
                    f"  ref_text='{ref_text[:40]}...'" if len(ref_text) > 40
                    else f"  ref_text='{ref_text}'" if ref_text
                    else "  (auto-transcribe on first call)"
                )
                logger.info(f"Voice ready: {voice_id}  <- {ref.name}{preview}")
                break
            else:
                logger.warning(f"Voice '{voice_id}': no reference audio found -- skipped")

        if not self._voices:
            logger.warning("No voices found -- TTS will return silence for all calls")

    @staticmethod
    def _load_ref_text(voice_dir: Path) -> str:
        """Load pre-computed reference transcription if it exists."""
        txt = voice_dir / "reference.txt"
        if txt.exists():
            return txt.read_text(encoding="utf-8").strip()
        return ""   # F5-TTS will auto-transcribe via its built-in ASR

    # ── audio-event pre-generation ────────────────────────────────────────────

    async def _prebuild_events(self, voice_id: str) -> None:
        logger.info(f"Pre-building {len(_AUDIO_EVENT_SPECS)} audio events for '{voice_id}' ...")
        loop = asyncio.get_running_loop()

        for name, (text, speed) in _AUDIO_EVENT_SPECS.items():
            try:
                frames = await loop.run_in_executor(
                    _executor, self._synth_raw, text, voice_id, speed
                )
                self._event_cache[f"{voice_id}:{name}"] = frames
                self._event_cache[name] = frames   # default alias
                logger.info(f"  ok {name:15s}  ({len(frames)} frames)")
            except Exception as exc:
                logger.warning(f"  fail {name}: {exc}")

        logger.success(f"Audio events ready for '{voice_id}'.")

    # ── synthesis ─────────────────────────────────────────────────────────────

    def _synth_raw(self, text: str, voice_id: str, speed: float) -> list[bytes]:
        """Synthesise text -> list of 320-byte 8 kHz PCM16 frames (telephony)."""
        voice = self._voices.get(voice_id) or next(iter(self._voices.values()), None)
        if voice is None:
            logger.warning(f"No voice registered for '{voice_id}' -- returning silence")
            return silence_frames(300)

        try:
            audio_arr, sr, _ = self._tts.infer(
                ref_file           = voice["ref_file"],
                ref_text           = voice["ref_text"],
                gen_text           = text,
                speed              = speed,
                nfe_step           = NFE_STEP,
                cfg_strength       = 2.0,
                sway_sampling_coef = -1.0,
                remove_silence     = True,
            )
        except Exception as exc:
            logger.error(f"F5-TTS infer failed for voice '{voice_id}': {exc}")
            return silence_frames(300)

        # audio_arr: numpy float32 at sr Hz
        wav_tensor = torch.from_numpy(np.array(audio_arr, dtype=np.float32)).unsqueeze(0)
        pcm        = resample_to_pstn(wav_tensor, orig_freq=sr)
        return pcm_to_frames(pcm)

    def _synth_raw_hq(self, text: str, voice_id: str, speed: float) -> np.ndarray:
        """Synthesise text -> 22 kHz PCM16 numpy array (HQ preview / download)."""
        voice = self._voices.get(voice_id) or next(iter(self._voices.values()), None)
        if voice is None:
            return np.array([], dtype=np.int16)

        try:
            audio_arr, sr, _ = self._tts.infer(
                ref_file           = voice["ref_file"],
                ref_text           = voice["ref_text"],
                gen_text           = text,
                speed              = speed,
                nfe_step           = NFE_STEP,
                cfg_strength       = 2.0,
                sway_sampling_coef = -1.0,
                remove_silence     = True,
            )
        except Exception as exc:
            logger.error(f"F5-TTS infer (HQ) failed for voice '{voice_id}': {exc}")
            return np.array([], dtype=np.int16)

        wav_tensor = torch.from_numpy(np.array(audio_arr, dtype=np.float32)).unsqueeze(0)
        return resample_to_hq(wav_tensor, orig_freq=sr, target_sr=HQ_SAMPLE_RATE)

    def _params_to_speed(self, exaggeration: float, cfg_weight: float) -> float:
        """Map Chatterbox/Kokoro (exaggeration, cfg_weight) -> F5-TTS speed."""
        speed = 0.85 + float(exaggeration) * 0.40
        return max(0.80, min(1.25, speed))

    # ── public API (same interface as old Kokoro VoiceModel) ──────────────────

    async def synthesize(
        self,
        text:         str,
        voice_id:     str,
        exaggeration: float = 0.60,
        cfg_weight:   float = 0.50,
    ) -> list[bytes]:
        """Async synthesis -> 8 kHz PCM16 frames for phone calls."""
        if not text.strip():
            return []
        speed = self._params_to_speed(exaggeration, cfg_weight)
        loop  = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor, self._synth_raw, text, voice_id, speed
        )

    async def synthesize_hq(
        self,
        text:         str,
        voice_id:     str,
        exaggeration: float = 0.60,
        cfg_weight:   float = 0.50,
    ) -> np.ndarray:
        """Async synthesis -> 22 kHz PCM16 numpy array for preview/testing."""
        if not text.strip():
            return np.array([], dtype=np.int16)
        speed = self._params_to_speed(exaggeration, cfg_weight)
        loop  = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor, self._synth_raw_hq, text, voice_id, speed
        )

    def get_event(self, kind: str, voice_id: str) -> list[bytes]:
        """Return pre-generated 8 kHz PCM frames for an audio event."""
        return (
            self._event_cache.get(f"{voice_id}:{kind}")
            or self._event_cache.get(kind)
            or silence_frames(150)
        )

    def list_voices(self) -> list[str]:
        return list(self._voices.keys())

    @property
    def ready(self) -> bool:
        return self._tts is not None
