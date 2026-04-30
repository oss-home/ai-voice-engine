from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8002
    device: str = "cuda"          # "cuda" | "cpu"
    voices_dir: str = "/app/voices"
    default_voice: str = "ahana"

    # Audio pipeline
    model_sample_rate: int = 24000   # Kokoro native output rate (same as Chatterbox)
    output_sample_rate: int = 8000   # PSTN / voice-sip
    frame_ms: int = 20               # 20 ms → 320 bytes at 8 kHz PCM16

    # Concurrency — number of thread-pool workers for GPU synthesis.
    # Kokoro is 23× faster than Chatterbox so 4 workers handles ~60–80 concurrent
    # phone calls on a single T4.  Increase if you add more GPUs / RAM.
    max_workers: int = 4

    # Auth
    api_token: str = ""              # Bearer token — empty = no auth

    class Config:
        env_prefix = "TTS_"
        env_file = ".env"


settings = Settings()
