FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# System dependencies
# espeak-ng  — required by misaki (Kokoro's English phonemizer)
# ffmpeg     — audio format conversion in librosa/soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (Kokoro + audio stack)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application source
COPY *.py ./

# Voice cloning utility (run once outside container to pre-build voice.pt)
RUN mkdir -p /app/scripts
COPY scripts/ ./scripts/

# Voices directory — mounted as a read-only volume at runtime
RUN mkdir -p /app/voices

EXPOSE 8002

CMD ["python", "main.py"]
