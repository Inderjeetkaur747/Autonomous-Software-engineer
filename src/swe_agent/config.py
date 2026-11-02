"""Central settings. Every other module reads config from here."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Values come from environment variables or the .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unknown variables in .env
    )

    # --- LLM (Groq) ---
    groq_api_key: SecretStr = SecretStr("")
    groq_model: str = "llama-3.3-70b-versatile"  # check Groq console for current names

    # --- GitHub (optional, only needed for private repos / higher rate limits) ---
    github_token: SecretStr = SecretStr("")

    # --- Vector DB (Qdrant) ---
    qdrant_path: Path = Path("data/qdrant")  # local embedded mode
    qdrant_url: str = ""  # set later if you run the Qdrant Docker server

    # --- Embeddings (local) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- Agent safety limits ---
    workspace_dir: Path = Path("workspaces")
    max_repair_attempts: int = Field(default=3, ge=0, le=10)
    command_timeout_seconds: int = Field(default=120, ge=1, le=900)

    # --- Logging ---
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Create settings once and reuse them (cached)."""
    return Settings()