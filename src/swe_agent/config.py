"""Central settings. Every other module reads config from here."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

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
    groq_model: str = "openai/gpt-oss-20b"  # check Groq console for current names
    llm_timeout_seconds: int = Field(default=60, ge=5, le=300)
    llm_max_retries: int = Field(default=3, ge=1, le=6)
    llm_temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    llm_max_output_tokens: int = Field(default=2048, ge=256, le=8192)

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

        # --- GitHub / cloning ---
    github_api_url: str = "https://api.github.com"
    github_timeout_seconds: int = Field(default=15, ge=3, le=120)
    clone_timeout_seconds: int = Field(default=180, ge=10, le=900)
    max_repo_size_mb: int = Field(default=200, ge=1, le=2000)

        # --- Scanning / chunking ---
    max_file_kb: int = Field(default=256, ge=1, le=2048)
    max_scan_files: int = Field(default=3000, ge=1, le=50000)
    chunk_max_lines: int = Field(default=60, ge=10, le=300)
    chunk_overlap_lines: int = Field(default=10, ge=0, le=50)

        # --- Embeddings / vector search ---
    embedding_backend: Literal["sentence-transformers", "hashing"] = "sentence-transformers"
    embedding_batch_size: int = Field(default=32, ge=1, le=256)
    embedding_max_seq_length: int = Field(default=512, ge=32, le=1024)
    default_top_k: int = Field(default=8, ge=1, le=50)

        # --- Hybrid retrieval / reranking ---
    retrieval_candidates: int = Field(default=30, ge=5, le=200)  # per retriever
    rrf_k: int = Field(default=60, ge=1, le=200)
    reranker_backend: Literal["cross-encoder", "overlap", "none"] = "cross-encoder"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = Field(default=20, ge=1, le=100)  # how many go to the reranker
    
    # --- Logging ---
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Create settings once and reuse them (cached)."""
    return Settings()