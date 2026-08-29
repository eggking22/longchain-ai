"""LLM semantic normalizer: strict JSON, evidence binding, rejection rules.

All HTTP is mocked with httpx.MockTransport (same seam as Phase 3's
LlmEvidenceEvaluator tests) — the suite stays fully offline.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.schemas.paper_semantics import (
    ExperimentModel,
    FigureReference,
    LlmNormalizationVerdict,
    Observation,
    PaperEvidence,
)
from app.services.paper_semantics import LlmSemanticNormalizer, PaperSemanticsConfig

BASE = "https://llm.test/v1"
EVIDENCE = [
    PaperEvidence(
        evidence_id="ev_001",
        figure_id="Figure 2",
        text="Figure 2. Relative expression of gene X in control and treatment groups.",
        role="caption",
    ),
    PaperEvidence(
        evidence_id="ev_002",
        figure_id="Figure 2",
        text="Treatment A significantly increased expression of gene X compared with control.",
        role="direct",
    ),
]
DRAFT = ExperimentModel(
    experiment_id="exp_f02",
    research_question="Does Treatment A affect gene X expression?",
    independent_variables=["Treatment A"],
    dependent_variables=["gene X expression"],
    experimental_groups=["Treatment A"],
    control_groups=["control"],
    intervention="Treatment A",
    observations=[
        Observation(
            statement="Treatment A significantly increased expression of gene X compared with control.",
            direction="increase",
            significance="significant",
            evidence_ids=["ev_002"],
        )
    ],
)
REF = FigureReference(figure_id="Figure 2", kind="figure", number=2, caption_text=EVIDENCE[0].text)
VALID_IDS = {"ev_001", "ev_002"}
EVIDENCE_TEXT = "\n".join(e.text for e in EVIDENCE)


def _verdict_payload(**overrides) -> dict:
    payload = {
        "research_question": "Does Treatment A affect gene X expression in cultured cells?",
        "hypothesis": "",
        "subjects": ["cells"],
        "independent_variables": ["Treatment A"],
        "dependent_variables": ["gene X expression"],
        "experimental_groups": ["Treatment A"],
        "control_groups": ["control"],
        "intervention": "Treatment A",
        "measurements": ["gene X expression"],
        "conclusions": [
            {
                "statement": "Treatment A increases gene X expression.",
                "relationship_type": "causal",
                "evidence_ids": ["ev_002"],
            }
        ],
    }
    payload.update(overrides)
    return payload


def _ok_handler(payload: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["temperature"] == 0.0
        assert body["messages"][0]["role"] == "system"
        user_prompt = body["messages"][1]["content"]
        assert "ev_001" in user_prompt and "ev_002" in user_prompt  # evidence is in the prompt
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(payload)}}]})

    return handler


def _normalizer(handler) -> LlmSemanticNormalizer:
    return LlmSemanticNormalizer(
        base_url=BASE,
        api_key="test-key",
        model="test-model",
        config=PaperSemanticsConfig(),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


class TestNormalize:
    def test_happy_path(self):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(
                200, json={"choices": [{"message": {"content": json.dumps(_verdict_payload())}}]}
            )

        verdict = _normalizer(handler).normalize(REF, EVIDENCE, DRAFT)
        assert verdict.research_question.endswith("in cultured cells?")
        assert verdict.conclusions[0].evidence_ids == ["ev_002"]
        assert len(calls) == 1
        assert calls[0].headers["Authorization"] == "Bearer test-key"

    def test_fenced_json_is_parsed(self):
        def handler(request: httpx.Request) -> httpx.Response:
            fenced = f"```json\n{json.dumps(_verdict_payload())}\n```"
            return httpx.Response(200, json={"choices": [{"message": {"content": fenced}}]})

        verdict = _normalizer(handler).normalize(REF, EVIDENCE, DRAFT)
        assert verdict.intervention == "Treatment A"

    def test_malformed_json_reasks_then_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})

        with pytest.raises(Exception):
            _normalizer(handler).normalize(REF, EVIDENCE, DRAFT)


class TestApplyValidation:
    def test_accepted_verdict_merges(self):
        verdict = LlmNormalizationVerdict.model_validate(_verdict_payload())
        merged, rejection = _normalizer(_ok_handler(_verdict_payload())).apply(
            DRAFT, verdict, VALID_IDS, EVIDENCE_TEXT
        )
        assert rejection is None
        assert merged.research_question.endswith("in cultured cells?")
        assert any(c.statement == "Treatment A increases gene X expression." for c in merged.conclusions)

    def test_hallucinated_evidence_id_is_rejected(self):
        payload = _verdict_payload(
            conclusions=[
                {
                    "statement": "Treatment A increases gene X expression.",
                    "relationship_type": "causal",
                    "evidence_ids": ["ev_999"],
                }
            ]
        )
        verdict = LlmNormalizationVerdict.model_validate(payload)
        merged, rejection = _normalizer(_ok_handler(payload)).apply(DRAFT, verdict, VALID_IDS, EVIDENCE_TEXT)
        assert rejection is not None and "ev_999" in rejection
        assert merged is DRAFT  # deterministic draft untouched

    def test_uncited_conclusion_is_rejected(self):
        payload = _verdict_payload(
            conclusions=[
                {"statement": "Treatment A increases gene X expression.", "relationship_type": "causal",
                 "evidence_ids": []}
            ]
        )
        verdict = LlmNormalizationVerdict.model_validate(payload)
        _, rejection = _normalizer(_ok_handler(payload)).apply(DRAFT, verdict, VALID_IDS, EVIDENCE_TEXT)
        assert rejection is not None and "citation" in rejection

    def test_fabricated_number_is_rejected(self):
        payload = _verdict_payload(
            conclusions=[
                {"statement": "Treatment A increases gene X expression by 2.37-fold.",
                 "relationship_type": "causal", "evidence_ids": ["ev_002"]}
            ]
        )
        verdict = LlmNormalizationVerdict.model_validate(payload)
        _, rejection = _normalizer(_ok_handler(payload)).apply(DRAFT, verdict, VALID_IDS, EVIDENCE_TEXT)
        assert rejection is not None and "numbers" in rejection

    def test_fabricated_p_value_is_rejected(self):
        payload = _verdict_payload(
            conclusions=[
                {"statement": "Treatment A increases gene X expression (p < 0.01).",
                 "relationship_type": "causal", "evidence_ids": ["ev_002"]}
            ]
        )
        verdict = LlmNormalizationVerdict.model_validate(payload)
        _, rejection = _normalizer(_ok_handler(payload)).apply(DRAFT, verdict, VALID_IDS, EVIDENCE_TEXT)
        assert rejection is not None

    def test_unsupported_direction_is_rejected(self):
        payload = _verdict_payload(
            conclusions=[
                {"statement": "Treatment A decreases gene X expression.",
                 "relationship_type": "causal", "evidence_ids": ["ev_002"]}
            ]
        )
        verdict = LlmNormalizationVerdict.model_validate(payload)
        _, rejection = _normalizer(_ok_handler(payload)).apply(DRAFT, verdict, VALID_IDS, EVIDENCE_TEXT)
        assert rejection is not None and "direction" in rejection

    def test_number_present_in_evidence_passes(self):
        evidence = [
            PaperEvidence(evidence_id="ev_001", figure_id="Figure 2", text="Figure 2. Results.", role="caption"),
            PaperEvidence(
                evidence_id="ev_002",
                figure_id="Figure 2",
                text="Treatment A increased expression of gene X by 2.37-fold (p < 0.05).",
                role="direct",
            ),
        ]
        payload = _verdict_payload(
            conclusions=[
                {"statement": "Treatment A increases gene X expression by 2.37-fold.",
                 "relationship_type": "causal", "evidence_ids": ["ev_002"]}
            ]
        )
        verdict = LlmNormalizationVerdict.model_validate(payload)
        _, rejection = _normalizer(_ok_handler(payload)).apply(
            DRAFT, verdict, {"ev_001", "ev_002"}, "\n".join(e.text for e in evidence)
        )
        assert rejection is None


class TestPipelineIntegration:
    def test_pipeline_uses_llm_and_marks_method(self, tmp_path):
        from app.services.paper_semantics import reconstruct_figures

        from .conftest import build_paper_tree, write_document_artifact

        write_document_artifact(build_paper_tree(), tmp_path, "paper-llm")

        def handler(request: httpx.Request) -> httpx.Response:
            content = request.content.decode("utf-8")
            if "Figure: Figure 2" in content:
                payload = _verdict_payload()
                # the real corpus numbers evidence with figure-scoped ids (ev_f02_###)
                payload["conclusions"][0]["evidence_ids"] = ["ev_f02_002"]
                body = json.dumps(payload)
            else:  # other figures: empty no-op verdict
                body = json.dumps(LlmNormalizationVerdict().model_dump())
            return httpx.Response(200, json={"choices": [{"message": {"content": body}}]})

        normalizer = _normalizer(handler)
        report = reconstruct_figures("paper-llm", tmp_path, normalizer=normalizer)
        figure2 = next(f for f in report.figures if f.figure_id == "Figure 2")
        assert figure2.method == "deterministic+llm"
        assert any(
            c.statement == "Treatment A increases gene X expression." for c in figure2.experiment.conclusions
        )
        figure3 = next(f for f in report.figures if f.figure_id == "Figure 3")
        assert figure3.method == "deterministic"  # INSUFFICIENT figures skip the LLM

    def test_pipeline_survives_llm_error(self, tmp_path):
        from app.services.paper_semantics import reconstruct_figures

        from .conftest import build_paper_tree, write_document_artifact

        write_document_artifact(build_paper_tree(), tmp_path, "paper-err")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        normalizer = _normalizer(handler)
        report = reconstruct_figures("paper-err", tmp_path, normalizer=normalizer)
        figure2 = next(f for f in report.figures if f.figure_id == "Figure 2")
        assert figure2.method == "deterministic"
        assert "llm_error" in figure2.detail
        assert figure2.reconstruction_status == "SUFFICIENT"  # deterministic result stands

    def test_novel_llm_conclusion_persists(self, tmp_path):
        """An LLM conclusion NOT already in the draft must survive persistence.

        Regression: LlmConclusion lacks interpretation_ids, so appending it raw
        crashed build_figures_document (`AttributeError`); it must land as a
        regular Conclusion with empty interpretation links.
        """
        import json as _json
        from pathlib import Path

        from app.services.paper_semantics import reconstruct_figures

        from .conftest import build_paper_tree, write_document_artifact

        write_document_artifact(build_paper_tree(), tmp_path, "paper-novel")

        def handler(request: httpx.Request) -> httpx.Response:
            content = request.content.decode("utf-8")
            if "Figure: Figure 2" in content:
                payload = _verdict_payload(
                    conclusions=[
                        {  # novel wording → not deduplicated against the draft
                            "statement": "Treatment A reproducibly increases gene X expression.",
                            "relationship_type": "causal",
                            "evidence_ids": ["ev_f02_002"],
                        }
                    ]
                )
                body = _json.dumps(payload)
            else:
                body = _json.dumps(LlmNormalizationVerdict().model_dump())
            return httpx.Response(200, json={"choices": [{"message": {"content": body}}]})

        report = reconstruct_figures("paper-novel", tmp_path, normalizer=_normalizer(handler))
        figure2 = next(f for f in report.figures if f.figure_id == "Figure 2")
        assert figure2.method == "deterministic+llm"
        assert any(
            c.statement == "Treatment A reproducibly increases gene X expression."
            and c.interpretation_ids == []
            for c in figure2.experiment.conclusions
        )
        figures_doc = _json.loads(
            (Path(tmp_path) / "paper_semantics" / "paper-novel" / "figures.json").read_text(encoding="utf-8")
        )
        entry = next(f for f in figures_doc["figures"] if f["figure_id"] == "Figure 2")
        assert any(
            c["statement"] == "Treatment A reproducibly increases gene X expression."
            and c["interpretations"] == []
            for c in entry["conclusions"]
        )
