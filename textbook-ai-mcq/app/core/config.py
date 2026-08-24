"""Application configuration loaded from environment variables / .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "textbook-ai-mcq"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # PostgreSQL / pgvector — interface only in Phase 0, not connected yet.
    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/textbook_mcq"

    # LLM access — reserved for later phases, never used at import time.
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = ""
    LLM_MODEL: str = ""

    # Vector search — reserved for later phases.
    EMBEDDING_MODEL: str = ""

    # Parser / storage locations (relative to project root unless absolute)
    UPLOADS_DIR: str = "uploads"
    ARTIFACTS_DIR: str = "data"

    # Hierarchical PDF parser knobs (see app/services/parser/config.py)
    PARSER_MAX_HEADING_LEVELS: int = 4
    PARSER_HEADER_FOOTER_BAND: float = 0.09
    PARSER_REPEAT_RATIO: float = 0.30
    PARSER_CHUNK_TARGET_CHARS: int = 600
    PARSER_CHUNK_MAX_CHARS: int = 1200
    PARSER_CHUNK_OVERLAP_SENTENCES: int = 1


@lru_cache
def get_settings() -> Settings:
    return Settings()
