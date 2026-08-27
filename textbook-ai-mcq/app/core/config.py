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

    # Embedding access — OpenAI-compatible endpoint (default: Zhipu bigmodel).
    # "hash" selects the deterministic offline provider used by tests/CI.
    EMBEDDING_BASE_URL: str = "https://open.bigmodel.cn/api/paas/v4"
    EMBEDDING_API_KEY: str = ""  # falls back to LLM_API_KEY when empty
    EMBEDDING_MODEL: str = "embedding-3"
    EMBEDDING_DIM: int = 1024
    EMBEDDING_BATCH_SIZE: int = 64

    # Hybrid retrieval knobs (see app/services/retrieval/config.py)
    RETRIEVAL_DENSE_TOP_K: int = 20
    RETRIEVAL_SPARSE_TOP_K: int = 20
    RETRIEVAL_RRF_K: int = 60
    RETRIEVAL_MIN_CHUNK_CHARS: int = 10  # drop front-matter/noise chunks at index time

    # Evidence gate knobs (see app/services/evidence/config.py)
    EVIDENCE_COVERAGE_THRESHOLD: float = 0.8  # coverage_score must reach this to pass
    EVIDENCE_TERM_WEIGHT: float = 0.75  # weight of query-term coverage in the heuristic score
    EVIDENCE_SPARSE_WEIGHT: float = 0.15  # weight of the BM25 strength signal
    EVIDENCE_HIT_WEIGHT: float = 0.1  # weight of the multi-hit depth signal
    EVIDENCE_SPARSE_SATURATE: float = 12.0  # BM25 top score mapped to full signal
    EVIDENCE_HIT_FLOOR: float = 2.0  # a hit only counts toward depth above this BM25 score
    EVIDENCE_WINDOW: int = 5  # hits cited in the coverage report
    EVIDENCE_COVERAGE_POOL: int = 8  # deeper pool inspected for term coverage (>= window)

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

    # Paper figure semantics knobs (see app/services/paper_semantics/config.py)
    PAPER_SEMANTICS_MAX_EVIDENCE_PER_FIGURE: int = 12  # cap on the per-figure evidence bundle
    PAPER_SEMANTICS_MAX_METHODS_PARAGRAPHS: int = 4  # top-N Methods paragraphs kept as supporting evidence
    PAPER_SEMANTICS_CAPTION_MAX_CHARS: int = 400  # longer caption-start paragraphs are not captions


@lru_cache
def get_settings() -> Settings:
    return Settings()
