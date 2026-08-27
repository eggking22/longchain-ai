"""Panel-level semantic reconstruction (Figure 2 → 2A / 2B / ...).

Each panel gets an independent evidence bundle (its caption chunk + paragraphs
citing that panel), its own experiment reconstruction and its own gate verdict,
reusing the figure-level machinery unchanged. The figure-level status is never
auto-upgraded from panel verdicts — panels are a finer-grained, independent
view of the same evidence.
"""

from __future__ import annotations

from app.schemas.paper_semantics import FigureReference, PanelSemantic

from .config import PaperSemanticsConfig
from .conclusion import build_conclusions
from .evidence_collector import collect_panel_evidence, number_evidence
from .experiment_model import build_experiment
from .figure_reference import figure_key
from .gate import SemanticEvidenceGate
from .sections import ParagraphRecord


def _title_from_chunk(chunk: str, max_chars: int = 120) -> str:
    first = chunk.split(". ", 1)[0].strip()
    return first[:max_chars]


def reconstruct_panel(
    ref: FigureReference,
    label: str,
    paragraphs: list[ParagraphRecord],
    config: PaperSemanticsConfig,
    gate: SemanticEvidenceGate | None = None,
) -> PanelSemantic:
    """Reconstruct one panel of a figure from panel-scoped evidence."""
    gate = gate or SemanticEvidenceGate()
    panel_id = f"{ref.number}{label}"
    evidences = number_evidence(
        collect_panel_evidence(ref, label, paragraphs, config),
        id_prefix=f"ev_{figure_key(ref)}{label}_",
    )

    panel_view = FigureReference(
        figure_id=f"{ref.figure_id}{label.upper()}",
        kind=ref.kind,
        number=ref.number,
        caption_text=ref.panel_texts.get(label, ""),
    )
    experiment = build_experiment(ref, evidences, id_suffix=label)
    experiment.conclusions = build_conclusions(experiment)
    verdict = gate.evaluate(panel_view, experiment, evidences)

    return PanelSemantic(
        panel_id=panel_id,
        label=label,
        title=_title_from_chunk(ref.panel_texts.get(label, "")),
        experiment=experiment,
        evidence=evidences,
        reconstruction_status=verdict.status,
        missing_information=verdict.missing_information,
        confidence=verdict.confidence,
        method="deterministic",
    )


def panel_labels(ref: FigureReference) -> list[str]:
    """Deterministic panel label set: caption-defined panels plus panels cited in text."""
    labels = set(ref.panel_texts) | set(ref.panel_mention_paragraph_ids)
    return sorted(labels)
