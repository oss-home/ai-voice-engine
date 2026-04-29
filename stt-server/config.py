from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8001
    device: str = "cuda"           # "cuda" | "cpu"
    model_size: str = "large-v3"   # "large-v3" | "distil-large-v3" | "medium"
    compute_type: str = "float16"  # "float16" | "int8" (int8 for CPU)
    language: str = "en"           # "en" | "hi" | "multi"
    beam_size: int = 5
    vad_threshold: float = 0.5
    silence_ms: int = 300          # ms of silence → end of utterance
    partial_interval_ms: int = 400 # ms between partial transcript emissions

    # Auth
    api_token: str = ""            # Bearer token — empty = no auth

    class Config:
        env_prefix = "STT_"
        env_file = ".env"


settings = Settings()
