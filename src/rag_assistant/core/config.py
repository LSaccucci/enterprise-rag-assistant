
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Enterprise RAG Assistant"
    app_version: str = "0.1.0"
    debug: bool = False

    # Local storage
    data_dir: Path = PROJECT_ROOT / "data"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b"
    embedding_model: str = "nomic-embed-text"

    # Retrieval
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    chunk_size: int = Field(default=700, ge=100, le=5000)
    chunk_overlap: int = Field(default=100, ge=0, le=1000)


@lru_cache
def get_settings() -> Settings:
    """Return a cached application settings instance."""
    return Settings()
