"""PCM conversion and frame utilities."""
import numpy as np
import torch
import torchaudio

from config import settings

FRAME_BYTES = int(settings.output_sample_rate * settings.frame_ms / 1000) * 2  # 320


def resample_to_pstn(wav: torch.Tensor) -> np.ndarray:
    """Resample Chatterbox output (24 kHz) → 8 kHz PCM16 numpy array."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)   # stereo → mono
    resampled = torchaudio.functional.resample(
        wav, orig_freq=settings.model_sample_rate, new_freq=settings.output_sample_rate
    )
    arr = resampled.squeeze().cpu().numpy()
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767).astype(np.int16)


def pcm_to_frames(pcm: np.ndarray) -> list[bytes]:
    """Split PCM16 array into FRAME_BYTES (320-byte) chunks, padding last."""
    raw = pcm.tobytes()
    frames = []
    for i in range(0, len(raw), FRAME_BYTES):
        chunk = raw[i : i + FRAME_BYTES]
        if len(chunk) < FRAME_BYTES:
            chunk += b"\x00" * (FRAME_BYTES - len(chunk))
        frames.append(chunk)
    return frames


def silence_frames(duration_ms: int) -> list[bytes]:
    n = int(settings.output_sample_rate * duration_ms / 1000)
    return pcm_to_frames(np.zeros(n, dtype=np.int16))
