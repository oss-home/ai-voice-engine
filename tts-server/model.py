"""
Chatterbox TTS model wrapper.

Responsibilities:
  - Load Chatterbox once at startup
  - Manage per-voice reference audio paths
  - Pre-generate all audio events (breath, laugh, sigh…) in Ahana's cloned
    voice at startup so the first live call has zero warm-up latency
  - Synthesize text with per-segment emotion parameters
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from pathlib import Path

import torch
from loguru import logger

from audio_utils import pcm_to_frames, resample_to_pstn, silence_frames
from emotion import EMOTION_PROFILES

# Thread pool — Chatterbox inference is CPU/GPU bound, not async-native.
# We run it in an executor so FastAPI's event loop stays unblocked.
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


class VoiceModel:
    """Thread-safe Chatterbox TTS with emotion + audio-event support."""

    def __init__(self, device: str, voices_dir: str) -> None:
        self.device = device
        self.voices_dir = Path(voices_dir)
        self._model = None                          # ChatterboxTTS (loaded async)
        self._voice_paths: dict[str, str] = {}      # voice_id → file path
        self._event_cache: dict[str, list[bytes]] = {}  # "voice:event" → frames

    # ── startup ──────────────────────────────────────────────────────────────

    async def load(self) -> None:
        logger.info(f"Loading Chatterbox on {self.device} …")
        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(_executor, self._load_model)
        logger.success("Chatterbox loaded.")

        self._discover_voices()

        if self._voice_paths:
            first = next(iter(self._voice_paths))
            await self._prebuild_events(first)

    def _load_model(self):
        from chatterbox.tts import ChatterboxTTS  # heavy import — keep deferred
        return ChatterboxTTS.from_pretrained(device=self.device)

    def _discover_voices(self) -> None:
        for d in self.voices_dir.iterdir():
            if not d.is_dir():
                continue
            for ext in ("mp3", "wav", "m4a", "flac", "ogg"):
                ref = d / f"reference.{ext}"
                if ref.exists():
                    self._voice_paths[d.name] = str(ref)
                    logger.info(f"Voice registered: {d.name}  ({ref})")
                    break

    # ── audio-event pre-generation ───────────────────────────────────────────

    async def _prebuild_events(self, voice_id: str) -> None:
        """
        Generate every audio event in the cloned voice so live calls
        never pay synthesis latency for a breath or laugh.

        We generate them with the model itself — the laugh / sigh / breath
        texts produce surprisingly natural paralinguistic sounds when
        Chatterbox is given high exaggeration in the cloned voice.
        """
        logger.info(f"Pre-building audio events for voice '{voice_id}' …")
        ref = self._voice_paths[voice_id]

        # (text_to_say, exaggeration, cfg_weight)
        event_specs: dict[str, tuple[str, float, float]] = {
            "breath":       ("mm,",             0.10, 0.90),
            "breath_deep":  ("mmm,",            0.12, 0.90),
            "hmm":          ("hmm.",            0.30, 0.80),
            "sigh":         ("hmmmm.",          0.20, 0.85),
            "laugh_soft":   ("hehe.",           1.60, 0.25),
            "laugh_full":   ("ha ha ha ha!",    1.80, 0.20),
        }

        loop = asyncio.get_running_loop()
        for name, (text, ex, cfg) in event_specs.items():
            try:
                frames = await loop.run_in_executor(
                    _executor, self._synth_raw, text, ref, ex, cfg
                )
                self._event_cache[f"{voice_id}:{name}"] = frames
                self._event_cache[name] = frames   # default alias
                logger.info(f"  {name}: {len(frames)} frames")
            except Exception as exc:
                logger.warning(f"  {name}: FAILED — {exc}")

        logger.success(f"Audio events ready for '{voice_id}'.")

    # ── synthesis ────────────────────────────────────────────────────────────

    def _synth_raw(
        self,
        text: str,
        ref_audio: str,
        exaggeration: float,
        cfg_weight: float,
    ) -> list[bytes]:
        wav: torch.Tensor = self._model.generate(
            text=text,
            audio_prompt_path=ref_audio,
            exaggeration=float(exaggeration),
            cfg_weight=float(cfg_weight),
        )
        pcm = resample_to_pstn(wav)
        return pcm_to_frames(pcm)

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
    ) -> list[bytes]:
        if not text.strip():
            return []
        ref = self._voice_paths.get(voice_id)
        if not ref:
            raise ValueError(f"Voice not found: {voice_id!r}")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor, self._synth_raw, text, ref, exaggeration, cfg_weight
        )

    def get_event(self, kind: str, voice_id: str) -> list[bytes]:
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
