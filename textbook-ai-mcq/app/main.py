"""Application entry point for textbook-ai-mcq."""

from fastapi import FastAPI

from app.core.config import get_settings
from app.routers import documents, health

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "基于生物学教材的长文本选择题生成系统 "
        "(Long-text MCQ generation from biology textbooks). "
        "Phase 1: hierarchical PDF parser + document structure + chunks."
    ),
)

app.include_router(health.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
