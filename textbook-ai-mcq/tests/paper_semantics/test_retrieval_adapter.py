"""Figure-aware retrieval adapter over the Phase 2 engine (read-only reuse)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.schemas.document import Chunk
from app.schemas.paper_semantics import FigureReference
from app.services.retrieval import HashEmbeddingProvider, RetrievalConfig, build_index
from app.services.paper_semantics import FigureAwareRetriever, PaperSemanticsConfig

DOC_ID = "paper-mini"

CHUNK_TEXTS = [
    ("c1", "Treatment A significantly increased expression of gene X compared with control."),
    ("c2", "Cells were divided into control and treatment groups in the experiment."),
    ("c3", "Photosynthesis converts light energy into chemical energy in chloroplasts."),
]


def _write_chunks(root: Path) -> None:
    directory = root / "chunks" / DOC_ID
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk_id, text in CHUNK_TEXTS:
            handle.write(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    breadcrumb=["Results"],
                    pages=[1],
                    char_count=len(text),
                    paragraph_ids=[f"p-{chunk_id}"],
                ).model_dump_json()
                + "\n"
            )


@pytest.fixture(scope="module")
def indexed_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("paper-retrieval")
    _write_chunks(root)
    build_index(DOC_ID, root, embedder=HashEmbeddingProvider(dim=16), config=RetrievalConfig(min_chunk_chars=0))
    return root


REF = FigureReference(
    figure_id="Figure 2",
    kind="figure",
    number=2,
    caption_text="Relative expression of gene X in control and treatment groups.",
)


class TestFigureAwareRetriever:
    def test_unavailable_without_index(self, tmp_path):
        assert FigureAwareRetriever.try_load(DOC_ID, tmp_path, PaperSemanticsConfig()) is None

    def test_collect_maps_hits_to_supporting_evidence(self, indexed_root):
        retriever = FigureAwareRetriever.try_load(DOC_ID, indexed_root, PaperSemanticsConfig())
        assert retriever is not None
        collected = retriever.collect(REF, covered_paragraph_ids=set())
        assert collected, "expected gene-X chunks to be retrieved for a gene-X caption"
        for evidence in collected:
            assert evidence.role == "supporting"
            assert evidence.chunk_id
            assert evidence.figure_id == "Figure 2"
        texts = " ".join(e.text for e in collected).lower()
        assert "gene x" in texts  # retrieval is anchored on the figure's subject

    def test_covered_paragraphs_are_deduplicated(self, indexed_root):
        retriever = FigureAwareRetriever.try_load(DOC_ID, indexed_root, PaperSemanticsConfig())
        assert retriever is not None
        first = retriever.collect(REF, covered_paragraph_ids=set())
        covered = {e.paragraph_id for e in first if e.paragraph_id}
        assert covered  # chunks carry paragraph ids from Phase 1
        second = retriever.collect(REF, covered_paragraph_ids=covered)
        assert second == []

    def test_build_query_uses_caption_terms(self, indexed_root):
        retriever = FigureAwareRetriever.try_load(DOC_ID, indexed_root, PaperSemanticsConfig())
        query = retriever.build_query(REF)
        assert query.startswith("Figure 2")
        assert "expression" in query
