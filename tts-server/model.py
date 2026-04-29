"""
Chatterbox TTS model wrapper.

Responsibilities:
  - Load Chatterbox once at startup (GPU / CPU)
  - Auto-trim reference audio to the most consistent 25-second segment
  - Manage per-voice reference audio paths
  - Pre-generate all audio events (breath, laugh, sigh, um, oh…) in the
    cloned voice so live calls have zero warm-up latency
  - Synthesize text with per-segment emotion parameters
  - Provide high-quality (22 kHz) synthesis for preview/testing
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
    build_wav_bytes,
    pcm_to_frames,
    resample_to_hq,
    resample_to_pstn,
    silence_frames,
)
from emotion import EMOTION_PROFILES

# Thread pool — Chatterbox inference is CPU/GPU bound, not async-native.
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Ideal reference duration for voice cloning (seconds).
_REF_TARGET_SECS = 25
_REF_SKIP_SECS   = 20   # skip the first N seconds (often intro / silence)


class VoiceModel:
    """Thread-safe Chatterbox TTS with full emotion + audio-event support."""

    def __init__(self, device: str, voices_dir: str) -> None:
        self.device = device
        self.voices_dir = Path(voices_dir)
        self._model = None
        self._voice_paths: dict[str, str]        = {}   # voice_id → trimmed wav path
        self._event_cache: dict[str, list[bytes]] = {}  # "voice:event" → PCM frames

    # ── startup ──────────────────────────────────────────────────────────────

    async def load(self) -> None:
        logger.info(f"Loading Chatterbox on {self.device} …")
        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(_executor, self._load_model)
        logger.success("Chatterbox loaded.")

        self._discover_voices()

        for voice_id in list(self._voice_paths):
            await self._prebuild_events(voice_id)

    def _load_model(self):
        from chatterbox.tts import ChatterboxTTS
        return ChatterboxTTS.from_pretrained(device=self.device)

    # ── reference audio handling ─────────────────────────────────────────────

    def _discover_voices(self) -> None:
        """Find voices and prepare a trimmed reference clip for each."""
        for d in self.voices_dir.iterdir():
            if not d.is_dir():
                continue
            for ext in ("mp3", "wav", "m4a", "flac", "ogg"):
                ref = d / f"reference.{ext}"
                if ref.exists():
                    trimmed = self._prepare_reference(d.name, str(ref))
                    self._voice_paths[d.name] = trimmed
                    logger.info(f"Voice ready: {d.name}  (ref: {trimmed})")
                    break

    def _prepare_reference(self, voice_id: str, src_path: str) -> str:
        """
        Load reference audio, skip the first _REF_SKIP_SECS, then find
        the most CONSISTENT 25-second window (RMS closest to median).
        Consistent speech = best voice cloning; avoids loud/dramatic sections
        that introduce hallucination noise in Chatterbox.
        Saves a 16 kHz mono WAV to /tmp (always writable).
        """
        out_path = f"/tmp/{voice_id}_reference_trimmed.wav"

        try:
            wav, sr = torchaudio.load(src_path)

            # Convert to mono
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            # Resample to 16 kHz (Chatterbox native)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                wav = resampler(wav)
                sr = 16000

            total_secs = wav.shape[-1] / sr
            logger.info(f"  [{voice_id}] reference audio: {total_secs:.1f}s total")

            skip  = int(min(_REF_SKIP_SECS, total_secs * 0.15) * sr)
            want  = int(_REF_TARGET_SECS * sr)
            avail = wav.shape[-1] - skip

            if avail <= 0:
                trimmed = wav
            elif avail <= want:
                trimmed = wav[:, skip:]
            else:
                # Sample every 0.5 s and record RMS for each candidate window
                step = sr // 2
                windows: list[tuple[int, float]] = []
                for s in range(skip, wav.shape[-1] - want, step):
                    chunk = wav[:, s : s + want]
                    rms   = float(chunk.pow(2).mean().sqrt())
                    windows.append((s, rms))

                if windows:
                    # Pick the window whose RMS is closest to the median —
                    # this selects the most average/steady speech segment.
                    sorted_rms = sorted(w[1] for w in windows)
                    median_rms = sorted_rms[len(sorted_rms) // 2]
                    best_start = min(windows, key=lambda w: abs(w[1] - median_rms))[0]
                else:
                    best_start = skip

                trimmed = wav[:, best_start : best_start + want]
                logger.info(
                    f"  [{voice_id}] reference window: "
                    f"{best_start/sr:.1f}s – {(best_start+want)/sr:.1f}s"
                )

            # Normalize to -20 dBFS for consistent voice cloning
            rms = trimmed.pow(2).mean().sqrt().clamp(min=1e-8)
            target_rms = 10 ** (-20 / 20)
            trimmed = trimmed * (target_rms / rms)
            trimmed = trimmed.clamp(-1.0, 1.0)

            torchaudio.save(out_path, trimmed, sr)
            logger.info(
                f"  [{voice_id}] trimmed reference: "
                f"{trimmed.shape[-1]/sr:.1f}s → {out_path}"
            )
            return out_path

        except Exception as exc:
            logger.warning(
                f"  [{voice_id}] reference prep failed ({exc}) — using original"
            )
            return src_path

    # ── audio-event pre-generation ───────────────────────────────────────────

    # (text_prompt, exaggeration, cfg_weight)
    # NOTE: Keep exaggeration ≤ 1.0 — higher values cause hallucination noise.
    _AUDIO_EVENT_SPECS: dict[str, tuple[str, float, float]] = {
        "breath":       ("Mm.",           0.08, 0.92),  # short inhale
        "breath_deep":  ("Mmm.",          0.12, 0.90),  # deep inhale
        "hmm":          ("Hmm.",          0.40, 0.65),  # quick thinking
        "hmm_long":     ("Hmmm.",         0.45, 0.62),  # longer thinking
        "sigh":         ("Haah.",         0.18, 0.85),  # gentle sigh
        "sigh_deep":    ("Haaaah.",       0.22, 0.82),  # deep emotional sigh
        "laugh_soft":   ("Hehe.",         0.80, 0.40),  # soft giggle
        "laugh_full":   ("Ha ha ha ha!",  1.00, 0.30),  # genuine laugh
        "um":           ("Um,",           0.38, 0.70),  # hesitation um
        "uh":           ("Uh,",           0.35, 0.72),  # hesitation uh
        "oh":           ("Oh!",           0.75, 0.42),  # realisation / surprise
    }

    async def _prebuild_events(self, voice_id: str) -> None:
        """Pre-generate all audio events in the cloned voice at startup."""
        logger.info(
            f"Pre-building {len(self._AUDIO_EVENT_SPECS)} audio events for '{voice_id}' …"
        )
        ref  = self._voice_paths[voice_id]
        loop = asyncio.get_running_loop()

        for name, (text, ex, cfg) in self._AUDIO_EVENT_SPECS.items():
            try:
                frames = await loop.run_in_executor(
                    _executor, self._synth_raw, text, ref, ex, cfg
                )
                self._event_cache[f"{voice_id}:{name}"] = frames
                self._event_cache[name] = frames   # default alias
                logger.info(f"  ✓ {name:15s}  ({len(frames)} frames)")
            except Exception as exc:
                logger.warning(f"  ✗ {name}: {exc}")

        logger.success(f"Audio events ready for '{voice_id}'.")

    # ── synthesis ────────────────────────────────────────────────────────────

    def _synth_raw(
        self,
        text: str,
        ref_audio: str,
        exaggeration: float,
        cfg_weight: float,
    ) -> list[bytes]:
        """Synthesize text and return 8 kHz PCM16 frames (PSTN / phone quality)."""
        wav: torch.Tensor = self._model.generate(
            text=text,
            audio_prompt_path=ref_audio,
            exaggeration=float(exaggeration),
            cfg_weight=float(cfg_weight),
        )
        pcm = resample_to_pstn(wav)
        return pcm_to_frames(pcm)

    def _synth_raw_hq(
        self,
        text: str,
        ref_audio: str,
        exaggeration: float,
        cfg_weight: float,
    ) -> np.ndarray:
        """Synthesize text and return 22 kHz PCM16 (high quality, for preview)."""
        wav: torch.Tensor = self._model.generate(
            text=text,
            audio_prompt_path=ref_audio,
            exaggeration=float(exaggeration),
            cfg_weight=float(cfg_weight),
        )
        return resample_to_hq(wav, target_sr=HQ_SAMPLE_RATE)

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        exaggeration: float = 0.60,
        cfg_weight: float   = 0.50,
    ) -> list[bytes]:
        """Async synthesis → 8 kHz PCM16 frames for phone calls."""
        if not text.strip():
            return []
        ref = self._voice_paths.get(voice_id)
        if not ref:
            raise ValueError(f"Voice not found: {voice_id!r}")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor, self._synth_raw, text, ref, exaggeration, cfg_weight
        )

    async def synthesize_hq(
        self,
        text: str,
        voice_id: str,
        exaggeration: float = 0.60,
        cfg_weight: float   = 0.50,
    ) -> np.ndarray:
        """Async synthesis → 22 kHz PCM16 numpy array for preview/testing."""
        if not text.strip():
            return np.array([], dtype=np.int16)
        ref = self._voice_paths.get(voice_id)
        if not ref:
            raise ValueError(f"Voice not found: {voice_id!r}")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor, self._synth_raw_hq, text, ref, exaggeration, cfg_weight
        )

    def get_event(self, kind: str, voice_id: str) -> list[bytes]:
        """Return pre-generated frames for an audio event, or silence."""
        return (
            self._event_cache.get(f"{voice_id}:{kind}")
            or self._event_cache.get(kind)
            or silence_frames(150)
        )

    def list_voices(self) -> list[str]:
        return list(self._voice_paths.keys())

    @property
    def ready(self) -> bool:
        return self._model is not None
