"""Application entry point for textbook-ai-mcq."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings
from app.routers import documents, health, paper_questions

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
app.include_router(paper_questions.router, prefix="/api/v1")

# Paper Question Review page (vanilla JS, no frontend framework).
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/review", include_in_schema=False)
    def review_page():
        return FileResponse(str(_STATIC_DIR / "review.html"), media_type="text/html")
