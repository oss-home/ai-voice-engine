"""
PCM conversion, frame utilities, and audio post-processing.

Pipeline for every synthesized segment:
  Kokoro 24 kHz tensor / numpy array
    → resample to 8 kHz (PSTN) or 22 kHz (preview)
    → normalize loudness to -18 dBFS
    → apply 6 ms fade-in / fade-out  (eliminates click artifacts)
    → split into 320-byte frames

Note: resample_to_pstn / resample_to_hq accept EITHER a torch.Tensor OR a
numpy.ndarray so callers don't have to convert.  Kokoro outputs numpy float32;
older code may pass torch tensors — both work.
"""
from __future__ import annotations

import struct
from typing import Union

import numpy as np
import torch
import torchaudio

from config import settings

# 20 ms × 8000 Hz × 2 bytes = 320 bytes per frame
FRAME_BYTES = int(settings.output_sample_rate * settings.frame_ms / 1000) * 2

# High-quality sample rate used by the /synthesize/preview endpoint
HQ_SAMPLE_RATE = 22050

# Loudness target for all output (telephony-safe)
_TARGET_DBFS = -18.0

# Type alias accepted by the resamplers
AudioInput = Union[torch.Tensor, np.ndarray]


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_tensor(wav: AudioInput) -> torch.Tensor:
    """
    Convert numpy array or torch tensor → 2-D float32 torch tensor (1, N).
    Values must be in [-1.0, 1.0].
    """
    if isinstance(wav, np.ndarray):
        t = torch.from_numpy(wav.astype(np.float32))
    else:
        t = wav.float()
    if t.dim() == 1:
        t = t.unsqueeze(0)           # (N,) → (1, N)
    if t.shape[0] > 1:
        t = t.mean(dim=0, keepdim=True)   # stereo → mono
    return t


def apply_fade(pcm: np.ndarray, fade_ms: float = 6.0, sr: int | None = None) -> np.ndarray:
    """
    Apply a linear fade-in and fade-out to eliminate click artifacts at
    segment boundaries when audio chunks are concatenated.
    """
    if sr is None:
        sr = settings.output_sample_rate
    fade_n = max(1, int(sr * fade_ms / 1000))
    if len(pcm) < 2 * fade_n:
        return pcm
    out = pcm.astype(np.float32)
    out[:fade_n]  *= np.linspace(0.0, 1.0, fade_n)
    out[-fade_n:] *= np.linspace(1.0, 0.0, fade_n)
    return out.astype(np.int16)


def normalize_pcm(pcm: np.ndarray, target_dbfs: float = _TARGET_DBFS) -> np.ndarray:
    """
    Normalize PCM16 to a consistent loudness level.
    Keeps all audio events and speech segments at the same perceived volume.
    """
    f = pcm.astype(np.float32) / 32767.0
    rms = float(np.sqrt(np.mean(f ** 2)))
    if rms < 1e-8:
        return pcm
    target = 10 ** (target_dbfs / 20.0)
    f = np.clip(f * (target / rms), -1.0, 1.0)
    return (f * 32767).astype(np.int16)


def trim_silence(pcm: np.ndarray, threshold: float = 0.005, sr: int | None = None) -> np.ndarray:
    """
    Remove leading and trailing silence (below threshold) from a PCM16 array.
    Keeps at least 8 ms of audio on each side for natural breathing room.
    """
    if sr is None:
        sr = settings.output_sample_rate
    min_keep = max(1, int(sr * 0.008))  # 8 ms
    f = pcm.astype(np.float32) / 32767.0
    mask = np.abs(f) > threshold
    if not mask.any():
        return pcm
    first = max(0, np.argmax(mask) - min_keep)
    last  = min(len(pcm), len(mask) - np.argmax(mask[::-1]) + min_keep)
    return pcm[first:last]


# ── resamplers ────────────────────────────────────────────────────────────────

def resample_to_pstn(
    wav: AudioInput,
    orig_freq: int | None = None,
) -> np.ndarray:
    """
    Kokoro/any 24 kHz float audio → 8 kHz PCM16 mono.

    wav:       torch.Tensor (1, N) or numpy.ndarray (N,) — float32, range [-1, 1]
    orig_freq: source sample rate (default: settings.model_sample_rate = 24000)

    Applies loudness normalisation + fade for clean concatenation.
    """
    if orig_freq is None:
        orig_freq = settings.model_sample_rate

    t = _to_tensor(wav)
    resampled = torchaudio.functional.resample(
        t,
        orig_freq=orig_freq,
        new_freq=settings.output_sample_rate,
    )
    arr = np.clip(resampled.squeeze().cpu().numpy(), -1.0, 1.0)
    pcm = (arr * 32767).astype(np.int16)
    pcm = normalize_pcm(pcm)
    pcm = trim_silence(pcm)
    pcm = apply_fade(pcm, fade_ms=6.0, sr=settings.output_sample_rate)
    return pcm


def resample_to_hq(
    wav: AudioInput,
    orig_freq: int | None = None,
    target_sr: int = HQ_SAMPLE_RATE,
) -> np.ndarray:
    """
    Kokoro/any 24 kHz float audio → 22 kHz PCM16 mono for preview/testing.

    wav:       torch.Tensor (1, N) or numpy.ndarray (N,) — float32, range [-1, 1]
    orig_freq: source sample rate (default: settings.model_sample_rate = 24000)
    """
    if orig_freq is None:
        orig_freq = settings.model_sample_rate

    t = _to_tensor(wav)
    resampled = torchaudio.functional.resample(t, orig_freq=orig_freq, new_freq=target_sr)
    arr = np.clip(resampled.squeeze().cpu().numpy(), -1.0, 1.0)
    pcm = (arr * 32767).astype(np.int16)
    pcm = normalize_pcm(pcm)
    pcm = apply_fade(pcm, fade_ms=6.0, sr=target_sr)
    return pcm


# ── frame utilities ───────────────────────────────────────────────────────────

def pcm_to_frames(pcm: np.ndarray) -> list[bytes]:
    """Split PCM16 into 320-byte frames (20 ms each), zero-padding the last."""
    raw = pcm.tobytes()
    frames: list[bytes] = []
    for i in range(0, len(raw), FRAME_BYTES):
        chunk = raw[i : i + FRAME_BYTES]
        if len(chunk) < FRAME_BYTES:
            chunk += b"\x00" * (FRAME_BYTES - len(chunk))
        frames.append(chunk)
    return frames


def silence_frames(duration_ms: int) -> list[bytes]:
    """Return silence as a list of 320-byte PCM16 frames."""
    n = int(settings.output_sample_rate * duration_ms / 1000)
    return pcm_to_frames(np.zeros(n, dtype=np.int16))


def build_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Pack a PCM16 mono numpy array into a complete WAV file (in memory)."""
    data = pcm.tobytes()
    ds   = len(data)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", ds + 36, b"WAVE",
        b"fmt ", 16,
        1,                   # PCM
        1,                   # mono
        sample_rate,
        sample_rate * 2,     # byte rate (sr × channels × bps/8)
        2,                   # block align
        16,                  # bits per sample
        b"data", ds,
    )
    return header + data
