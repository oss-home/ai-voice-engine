#!/usr/bin/env python3
"""
Create a Kokoro voicepack (.pt) from reference audio.

Run ONCE per voice to generate voices/{name}/voice.pt.
The TTS server loads this at startup — no more cloning delay.

Usage:
    # Inside the container (recommended):
    docker exec -it ai-voice-tts python scripts/create_voice.py --voice ahana

    # Or directly on the host (if dependencies installed):
    python scripts/create_voice.py --voice ahana --voices-dir ~/ai-voice-engine/voices

The script tries several extraction strategies depending on which Kokoro
version is installed.  It prints the method used and saves the result.

After running, restart the TTS container so it picks up the new voice.pt:
    docker compose restart tts-server
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchaudio


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a Kokoro voicepack from reference audio.")
    p.add_argument("--voice",      required=True, help="Voice directory name (e.g. ahana)")
    p.add_argument("--voices-dir", default="/app/voices",
                   help="Path to voices directory (default: /app/voices)")
    p.add_argument("--device",     default="cuda",
                   help="'cuda' or 'cpu' (default: cuda)")
    p.add_argument("--overwrite",  action="store_true",
                   help="Overwrite existing voice.pt if present")
    return p.parse_args()


# ── reference prep ────────────────────────────────────────────────────────────

def load_reference(ref_path: str, target_sr: int = 24000) -> torch.Tensor:
    """Load reference audio, convert to mono 24 kHz, trim to best 25 s window."""
    print(f"Loading reference: {ref_path}")
    wav, sr = torchaudio.load(ref_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr  = target_sr

    total = wav.shape[-1] / sr
    print(f"  Duration: {total:.1f} s  SR: {sr} Hz")

    # Skip first 15 % (often intro) then find the most consistent 25 s window
    skip   = int(min(20, total * 0.15) * sr)
    want   = int(25 * sr)
    avail  = wav.shape[-1] - skip

    if avail <= 0:
        return wav
    if avail <= want:
        return wav[:, skip:]

    step = sr // 2
    windows: list[tuple[int, float]] = []
    for s in range(skip, wav.shape[-1] - want, step):
        chunk = wav[:, s : s + want]
        rms   = float(chunk.pow(2).mean().sqrt())
        windows.append((s, rms))

    sorted_rms = sorted(w[1] for w in windows)
    median_rms = sorted_rms[len(sorted_rms) // 2]
    best_start = min(windows, key=lambda w: abs(w[1] - median_rms))[0]

    print(f"  Best window: {best_start/sr:.1f}s – {(best_start+want)/sr:.1f}s")
    return wav[:, best_start : best_start + want]


# ── style extraction ──────────────────────────────────────────────────────────

def extract_style(wav: torch.Tensor, pipeline, device: torch.device) -> torch.Tensor:
    """
    Extract the Kokoro style embedding from reference audio.
    Tries multiple API paths — works across Kokoro versions.
    """
    SR = 24000
    wav = wav.to(device)
    model = pipeline.model

    # Approach 1: pipeline.create_voice  (Kokoro ≥ 0.9.4)
    if hasattr(pipeline, "create_voice"):
        try:
            pack = pipeline.create_voice(wav, sr=SR)
            print("  Method: pipeline.create_voice  ✓")
            return pack.cpu()
        except Exception as e:
            print(f"  pipeline.create_voice failed: {e}")

    # Approach 2: model.style_encoder (StyleTTS2 internals)
    if hasattr(model, "style_encoder"):
        try:
            mel  = _log_mel(wav.squeeze(0), SR, device)
            pack = model.style_encoder(mel.unsqueeze(0))
            print("  Method: model.style_encoder  ✓")
            return pack.cpu()
        except Exception as e:
            print(f"  model.style_encoder failed: {e}")

    # Approach 3: model.ref_enc
    if hasattr(model, "ref_enc"):
        try:
            mel  = _log_mel(wav.squeeze(0), SR, device)
            pack = model.ref_enc(mel.unsqueeze(0))
            print("  Method: model.ref_enc  ✓")
            return pack.cpu()
        except Exception as e:
            print(f"  model.ref_enc failed: {e}")

    # Approach 4: voice mixing — blend warm voices as approximation
    print("  No direct cloning API found — using voice mixing as approximation.")
    print("  NOTE: This is NOT a clone of the reference audio — it's a preset blend.")
    try:
        # Mix af_heart (warm) + af_bella (clear) for a pleasant female voice
        from kokoro import KPipeline
        voices_pt = {}
        for name in ("af_heart", "af_bella"):
            try:
                v = pipeline.load_voice(name)
                if v is not None:
                    voices_pt[name] = v
            except Exception:
                pass
        if voices_pt:
            tensors = list(voices_pt.values())
            mixed   = torch.stack(tensors).mean(dim=0)
            print(f"  Method: voice mixing ({list(voices_pt)})  ✓")
            return mixed.cpu()
    except Exception as e:
        print(f"  voice mixing failed: {e}")

    raise RuntimeError(
        "Could not extract a voice embedding.  "
        "Try upgrading kokoro: pip install -U kokoro"
    )


def _log_mel(
    wav: torch.Tensor,
    sr:  int,
    device: torch.device,
    n_fft:  int = 1024,
    n_mels: int = 80,
    hop:    int = 256,
) -> torch.Tensor:
    """Log-mel spectrogram compatible with StyleTTS2 style encoder."""
    wav = wav.to(device).float()
    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=1.0,
    ).to(device)
    mel = mel_tf(wav.unsqueeze(0)).squeeze(0)
    return torch.log(mel.clamp(min=1e-5))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    voices_dir = Path(args.voices_dir)
    voice_dir  = voices_dir / args.voice
    out_path   = voice_dir / "voice.pt"

    if out_path.exists() and not args.overwrite:
        print(f"voice.pt already exists: {out_path}")
        print("Use --overwrite to regenerate.")
        sys.exit(0)

    # Find reference audio
    ref_path = None
    for ext in ("mp3", "wav", "m4a", "flac", "ogg"):
        candidate = voice_dir / f"reference.{ext}"
        if candidate.exists():
            ref_path = str(candidate)
            break

    if ref_path is None:
        print(f"ERROR: No reference audio found in {voice_dir}")
        print("Expected: reference.mp3 (or .wav/.m4a/.flac/.ogg)")
        sys.exit(1)

    # Load Kokoro pipeline
    print("Loading Kokoro-82M …")
    from kokoro import KPipeline
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=str(device))
    print("Kokoro loaded.")

    # Load and prep reference audio
    wav = load_reference(ref_path)

    # Extract style embedding
    print("Extracting voice style …")
    with torch.no_grad():
        voicepack = extract_style(wav, pipeline, device)

    # Save
    voice_dir.mkdir(parents=True, exist_ok=True)
    torch.save(voicepack, str(out_path))
    print(f"\n✓ Voicepack saved → {out_path}  (shape: {voicepack.shape})")
    print("\nRestart the TTS server to use the new voice:")
    print("  docker compose restart tts-server")


if __name__ == "__main__":
    main()
