"""Figure-aware retrieval adapter over the existing Phase 2 engine.

The Phase 2 RetrievalEngine is used strictly read-only: we load it from
``data/index/{doc_id}`` (when that index exists) and turn per-figure queries
("Figure 2 <caption terms>") into supplemental supporting evidence. When no
index exists the adapter simply reports unavailable and the pipeline runs on
structural evidence alone. No Phase 2 code is modified or re-implemented.

Chunk→paragraph mapping comes from the Phase 2 records.jsonl so retrieved
evidence deduplicates against paragraph-level evidence by paragraph id.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from app.schemas.paper_semantics import PaperEvidence

from .config import PaperSemanticsConfig
from .figure_reference import FigureReference
from .patterns import word_tokens

try:  # pragma: no cover - trivial import guard, retrieval always present in practice
    from app.services.retrieval import RetrievalEngine
except ImportError:  # pragma: no cover
    RetrievalEngine = None  # type: ignore[assignment]


class FigureAwareRetriever:
    """Wraps a loaded RetrievalEngine to pull extra evidence per figure."""

    def __init__(self, engine: "RetrievalEngine", config: Optional[PaperSemanticsConfig] = None) -> None:
        self.engine = engine
        self.config = config or PaperSemanticsConfig()
        self._paragraph_map: dict[str, list[str]] = {}

    @classmethod
    def try_load(
        cls,
        doc_id: str,
        artifacts_root: str | Path = "data",
        config: Optional[PaperSemanticsConfig] = None,
        embedder=None,
    ) -> Optional["FigureAwareRetriever"]:
        """Load the engine if an index exists; return None otherwise (graceful skip)."""
        if RetrievalEngine is None:
            return None
        index_dir = Path(artifacts_root) / "index" / doc_id
        if not (index_dir / "manifest.json").exists():
            return None
        try:
            engine = RetrievalEngine.load(doc_id, artifacts_root, embedder=embedder)
        except Exception:
            return None
        retriever = cls(engine, config)
        retriever._load_paragraph_map(index_dir / "records.jsonl")
        return retriever

    def _load_paragraph_map(self, records_path: Path) -> None:
        if not records_path.exists():
            return
        with records_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._paragraph_map[record.get("chunk_id", "")] = record.get("paragraph_ids", [])

    def build_query(self, ref: FigureReference) -> str:
        terms = word_tokens(ref.caption_text)[:8]
        return " ".join([ref.figure_id, *terms]) if terms else ref.figure_id

    def collect(self, ref: FigureReference, covered_paragraph_ids: set[str]) -> list[PaperEvidence]:
        """Retrieve supporting evidence not already covered by paragraph-level collection."""
        result = self.engine.retrieve(self.build_query(ref), mode="hybrid", top_k=self.config.retrieval_top_k)
        collected: list[PaperEvidence] = []
        for hit in result.hits:
            paragraph_ids = self._paragraph_map.get(hit.chunk_id, [])
            if set(paragraph_ids) & covered_paragraph_ids:
                continue  # already collected structurally — keep provenance at paragraph level
            if any(e.chunk_id == hit.chunk_id for e in collected):
                continue
            collected.append(
                PaperEvidence(
                    figure_id=ref.figure_id,
                    text=hit.text,
                    role="supporting",
                    section_type="other",
                    breadcrumb=list(hit.breadcrumb),
                    paragraph_id=paragraph_ids[0] if paragraph_ids else "",
                    page_no=hit.pages[0] if hit.pages else 0,
                    chunk_id=hit.chunk_id,
                )
            )
        return collected
