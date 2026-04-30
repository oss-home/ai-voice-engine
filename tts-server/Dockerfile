FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# System dependencies
# espeak-ng  — required by some F5-TTS phonemizers
# ffmpeg     — audio format conversion in librosa/soundfile/torchaudio
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (F5-TTS + audio stack)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# torchcodec is pulled in by f5-tts but is incompatible with PyTorch 2.5.1 on T4
# (libavutil shared libs mismatch). Remove it so torchaudio falls back to soundfile.
RUN pip uninstall -y torchcodec || true

# Pre-download F5-TTS + Vocos weights at build time so container starts instantly.
RUN python3 -c "from f5_tts.api import F5TTS; F5TTS(device='cpu'); print('F5-TTS weights cached.')"

# Application source
COPY *.py ./

# Voice cloning utility (run once outside container to pre-build voice.pt)
RUN mkdir -p /app/scripts
COPY scripts/ ./scripts/

# Voices directory — mounted as a read-only volume at runtime
RUN mkdir -p /app/voices

EXPOSE 8002

CMD ["python", "main.py"]
