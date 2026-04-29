from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8002
    device: str = "cuda"          # "cuda" | "cpu"
    voices_dir: str = "/app/voices"
    default_voice: str = "ahana"

    # Audio
    model_sample_rate: int = 24000   # Chatterbox native output
    output_sample_rate: int = 8000   # PSTN / voice-sip
    frame_ms: int = 20               # 20 ms → 320 bytes at 8 kHz PCM16

    # Auth
    api_token: str = ""              # Bearer token — empty = no auth

    class Config:
        env_prefix = "TTS_"
        env_file = ".env"


settings = Settings()
