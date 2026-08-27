"""Per-figure evidence collection with roles, semantic types and provenance.

Two dimensions are recorded on every PaperEvidence:

- ``role`` — the *source* dimension: caption / direct (Results) / supporting
  (Methods) / interpretation (Discussion);
- ``evidence_type`` — the *semantic* dimension: direct_observation /
  experimental_design / statistical_result / author_interpretation / mixed,
  so future question blueprints can impose rules like "conclusion questions
  require direct_observation evidence".

Priority order (per project spec): caption > Results > Methods > Discussion.
Statistics notes inside captions ("Data were analyzed by ...; ****P < 0.0001")
are retained as sentence-level statistical_result evidence units instead of
being dropped. Evidence ids are assigned by :number_evidence once a bundle is
final, keeping ids stable and deterministic.
"""

from __future__ import annotations

from app.schemas.paper_semantics import FigureTextBlock, PaperEvidence

from .config import PaperSemanticsConfig
from .experiment_model import detect_direction
from .figure_reference import FigureReference
from .patterns import (
    CONTROL_TERM_RE,
    P_VALUE_RE,
    SIGNIFICANT_RE,
    STATS_NOTE_RE,
    TREATMENT_SPAN_RE,
    split_sentences_en,
    word_tokens,
)
from .sections import ParagraphRecord

# A Methods paragraph must share at least this many content words with the
# caption/direct anchor to be pulled in — otherwise multi-experiment papers
# would mix unrelated protocols into every figure.
MIN_METHODS_OVERLAP = 2


def infer_evidence_type(role: str, text: str) -> str:
    """Map (source role, text) onto the semantic evidence type, deterministically."""
    if role == "direct":
        return "direct_observation"
    if role == "supporting":
        return "experimental_design"
    if role == "interpretation":
        return "author_interpretation"
    if role == "caption":
        # captions mix design and results; type by content with stats notes last
        if STATS_NOTE_RE.match(text.strip()):
            return "statistical_result"
        for sentence in split_sentences_en(text):
            if STATS_NOTE_RE.match(sentence):
                continue
            if detect_direction(sentence) != "unspecified" or SIGNIFICANT_RE.search(sentence):
                return "direct_observation"
        if TREATMENT_SPAN_RE.search(text) or CONTROL_TERM_RE.search(text) or P_VALUE_RE.search(text):
            return "experimental_design"
        return "experimental_design"
    return "mixed"


def _evidence(paragraph: ParagraphRecord, figure_id: str, role: str, text: str | None = None, panel_id: str = "") -> PaperEvidence:
    text = paragraph.text if text is None else text
    return PaperEvidence(
        figure_id=figure_id,
        text=text,
        role=role,  # type: ignore[arg-type]
        evidence_type=infer_evidence_type(role, text),  # type: ignore[arg-type]
        section_type=paragraph.section_type,
        section_title=paragraph.section_title,
        breadcrumb=list(paragraph.breadcrumb),
        paragraph_id=paragraph.paragraph_id,
        page_no=paragraph.page_no,
        panel_id=panel_id,
    )


def _stat_note_evidences(evidence: PaperEvidence) -> list[PaperEvidence]:
    """Sentence-level statistical_result units from a caption/direct paragraph."""
    notes: list[PaperEvidence] = []
    if evidence.role not in ("caption", "direct"):
        return notes
    for sentence in split_sentences_en(evidence.text):
        if not STATS_NOTE_RE.match(sentence) or not P_VALUE_RE.search(sentence):
            continue
        notes.append(
            PaperEvidence(
                figure_id=evidence.figure_id,
                text=sentence,
                role=evidence.role,
                evidence_type="statistical_result",
                section_type=evidence.section_type,
                section_title=evidence.section_title,
                breadcrumb=list(evidence.breadcrumb),
                paragraph_id=evidence.paragraph_id,
                page_no=evidence.page_no,
                panel_id=evidence.panel_id,
                assignment=evidence.assignment,
            )
        )
    return notes


def _methods_relevance(
    methods: list[ParagraphRecord], anchor_text: str
) -> list[tuple[int, ParagraphRecord]]:
    """Score Methods paragraphs by content-word overlap with caption + direct evidence.

    Returns (overlap, paragraph) pairs sorted by descending overlap then document
    order; only paragraphs with overlap > 0 are useful anchors.
    """
    anchor_tokens = set(word_tokens(anchor_text))
    scored = []
    for paragraph in methods:
        overlap = len(anchor_tokens & set(word_tokens(paragraph.text)))
        scored.append((overlap, paragraph.order, paragraph))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [(overlap, paragraph) for overlap, _, paragraph in scored]


def collect_evidence(
    ref: FigureReference,
    paragraphs: list[ParagraphRecord],
    config: PaperSemanticsConfig,
    text_block: FigureTextBlock | None = None,
) -> list[PaperEvidence]:
    """Build the evidence bundle for one figure, in priority order.

    Direct evidence comes from the L1 text block when available: anchors
    (paragraphs citing this figure) plus inherited continuation paragraphs.
    Continuation evidence is recorded with assignment="continuation" for
    downstream reading/question use; semantic extraction (observations,
    groups, variables) deliberately ignores it this round so the figure-level
    baseline stays frozen.
    """
    by_id = {p.paragraph_id: p for p in paragraphs}
    evidences: list[PaperEvidence] = []
    collected_paragraph_ids: set[str] = set()

    def add(paragraph: ParagraphRecord, role: str, assignment: str = "") -> None:
        if len(evidences) >= config.max_evidence_per_figure:
            return
        if paragraph.paragraph_id in collected_paragraph_ids:
            return
        collected_paragraph_ids.add(paragraph.paragraph_id)
        evidence = _evidence(paragraph, ref.figure_id, role)
        evidence.assignment = assignment  # type: ignore[assignment]
        evidences.append(evidence)

    # 1. Caption — the strongest single piece of evidence. Uses the caption
    #    text recorded by the reference extractor, which is sliced from the
    #    paragraph when the caption is embedded mid-paragraph (two-column PDFs).
    if ref.caption_paragraph_id and ref.caption_paragraph_id in by_id:
        paragraph = by_id[ref.caption_paragraph_id]
        evidence = _evidence(paragraph, ref.figure_id, "caption")
        evidence.text = ref.caption_text or paragraph.text
        if len(evidences) < config.max_evidence_per_figure:
            evidences.append(evidence)

    # 2. Direct evidence. The L1 text block drives the order when present:
    #    anchors (explicit citations, assignment=anchor) first, then inherited
    #    continuation paragraphs (assignment=continuation). Paragraphs citing
    #    this figure outside its block (e.g. shared multi-figure paragraphs)
    #    are still anchors by definition.
    if text_block is not None:
        for paragraph_id in text_block.anchor_paragraph_ids:
            paragraph = by_id.get(paragraph_id)
            if paragraph is None or paragraph.section_type in ("methods", "discussion"):
                continue
            add(paragraph, "direct", assignment="anchor")
        # Continuation paragraphs are stored context only (excluded from semantic
        # extraction) — cap them so they never crowd Methods/Discussion evidence
        # out of the bundle.
        stored_continuations = 0
        for paragraph_id in text_block.continuation_paragraph_ids:
            if stored_continuations >= config.max_continuation_evidence:
                break
            paragraph = by_id.get(paragraph_id)
            if paragraph is None or paragraph.section_type in ("methods", "discussion"):
                continue
            before = len(evidences)
            add(paragraph, "direct", assignment="continuation")
            stored_continuations += len(evidences) - before
    for paragraph_id in ref.mention_paragraph_ids:
        paragraph = by_id.get(paragraph_id)
        if paragraph is None or paragraph.section_type in ("methods", "discussion"):
            continue
        add(paragraph, "direct", assignment="anchor")

    # 3. Methods paragraphs (supporting evidence), ranked by overlap with what
    #    we already know about the figure so multi-experiment papers do not
    #    pull in unrelated protocols.
    anchor = " ".join(e.text for e in evidences)
    methods = [p for p in paragraphs if p.section_type == "methods"]
    for overlap, paragraph in _methods_relevance(methods, anchor)[: config.max_methods_paragraphs]:
        if overlap >= MIN_METHODS_OVERLAP:
            add(paragraph, "supporting")

    # 4. Discussion paragraphs that mention the figure (interpretation).
    for paragraph_id in ref.mention_paragraph_ids:
        paragraph = by_id.get(paragraph_id)
        if paragraph is not None and paragraph.section_type == "discussion":
            add(paragraph, "interpretation")

    # 5. Statistics notes as sentence-level statistical_result units, inserted
    #    right after their source paragraph so ids stay deterministic.
    augmented: list[PaperEvidence] = []
    for evidence in evidences:
        augmented.append(evidence)
        for note in _stat_note_evidences(evidence):
            if len(augmented) < config.max_evidence_per_figure:
                augmented.append(note)
    return augmented


def collect_panel_evidence(
    ref: FigureReference, label: str, paragraphs: list[ParagraphRecord], config: PaperSemanticsConfig
) -> list[PaperEvidence]:
    """Evidence bundle for one panel: its caption chunk + paragraphs citing that panel.

    Methods/Discussion remain shared at the figure level and are deliberately
    not duplicated into panel bundles — a panel's semantics rest on what its
    own caption says and what the text says *about that panel*.
    """
    by_id = {p.paragraph_id: p for p in paragraphs}
    panel_id = f"{ref.number}{label}"
    evidences: list[PaperEvidence] = []

    chunk = ref.panel_texts.get(label, "")
    if chunk and ref.caption_paragraph_id in by_id:
        paragraph = by_id[ref.caption_paragraph_id]
        evidences.append(_evidence(paragraph, ref.figure_id, "caption", text=chunk, panel_id=panel_id))

    for paragraph_id in ref.panel_mention_paragraph_ids.get(label, []):
        paragraph = by_id.get(paragraph_id)
        if paragraph is None or paragraph.section_type in ("methods", "discussion"):
            continue
        if any(e.paragraph_id == paragraph.paragraph_id for e in evidences):
            continue
        if len(evidences) >= config.max_evidence_per_figure:
            break
        evidences.append(_evidence(paragraph, ref.figure_id, "direct", panel_id=panel_id))

    augmented: list[PaperEvidence] = []
    for evidence in evidences:
        augmented.append(evidence)
        for note in _stat_note_evidences(evidence):
            if len(augmented) < config.max_evidence_per_figure:
                augmented.append(note)
    return augmented


def number_evidence(evidences: list[PaperEvidence], start: int = 1, id_prefix: str = "ev_") -> list[PaperEvidence]:
    """Assign sequential evidence ids in list order (deterministic)."""
    for index, evidence in enumerate(evidences, start=start):
        evidence.evidence_id = f"{id_prefix}{index:03d}"
    return evidences
