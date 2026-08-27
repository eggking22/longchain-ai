"""Optional LLM semantic normalization with strict evidence binding.

Reuses the access pattern of Phase 3's LlmEvidenceEvaluator (raw httpx POST to
an OpenAI-compatible /chat/completions, temperature 0, strict JSON with one
re-ask, injectable httpx.Client for offline tests). It is a *patcher* over the
deterministic draft, never a replacement:

- the prompt only contains the figure's collected evidence (id + text) and the
  draft slots;
- every normalized conclusion must cite evidence_ids that exist in the bundle;
- no numeric value / p-value may appear in the output unless it literally
  occurs in the evidence texts;
- any violation rejects the whole verdict and the deterministic result stands.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import httpx

from app.schemas.paper_semantics import (
    ExperimentModel,
    FigureReference,
    LlmNormalizationVerdict,
    PaperEvidence,
)

from .config import PaperSemanticsConfig
from .patterns import DECREASE_RE, INCREASE_RE, P_VALUE_RE

DEFAULT_LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

SYSTEM_PROMPT = """\
You normalize a deterministic draft of the experiment semantics behind one \
scientific figure. You receive the draft fields and the numbered evidence \
texts the draft was derived from.

Rules:
- Use ONLY facts present in the evidence texts. Do not add knowledge.
- Fill or rephrase empty/misphrased draft fields (research question, groups, \
variables, measurements) so they read naturally.
- Every conclusion must cite the evidence_ids (ev_###) that support it; \
conclusions without valid citations will be rejected.
- Never invent numeric values, fold changes or p-values.
- Do not upgrade association/correlation into causation.
- Keep the exact relationship_type vocabulary: causal, correlation, \
association, inhibition, activation, knockout, overexpression, \
dose_response, time_dependent, unspecified.

Answer with strict JSON only (no prose, no markdown fences) matching:
{"research_question": str, "hypothesis": str, "subjects": [str], \
"independent_variables": [str], "dependent_variables": [str], \
"experimental_groups": [str], "control_groups": [str], "intervention": str, \
"measurements": [str], "conclusions": [{"statement": str, \
"relationship_type": str, "evidence_ids": [str]}]}
"""

_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)%?\b")


class LlmNormalizationError(RuntimeError):
    """The normalizer could not produce a usable verdict (network, JSON, ...)."""


class LlmSemanticNormalizer:
    """LLM patcher over the deterministic ExperimentModel draft."""

    name = "llm"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        config: Optional[PaperSemanticsConfig] = None,
        timeout: float = 60.0,
        max_attempts: int = 2,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.config = config or PaperSemanticsConfig()
        self.timeout = timeout
        self.max_attempts = max_attempts
        self._client = client

    # --- public API -----------------------------------------------------------

    def normalize(
        self, ref: FigureReference, evidences: list[PaperEvidence], draft: ExperimentModel
    ) -> LlmNormalizationVerdict:
        payload = self._build_user_prompt(ref, evidences, draft)
        for _ in range(self.max_attempts):
            content = self._chat(payload)
            verdict = self._parse_verdict(content)
            if verdict is not None:
                return verdict
            payload += "\n\nYour previous answer was not valid JSON. Answer again with strict JSON only."
        raise LlmNormalizationError("LLM normalizer failed to return strict JSON")

    def apply(
        self,
        draft: ExperimentModel,
        verdict: LlmNormalizationVerdict,
        valid_evidence_ids: set[str],
        evidence_text: str,
    ) -> tuple[ExperimentModel, Optional[str]]:
        """Merge a verdict onto the draft; returns (model, rejection_reason).

        rejection_reason is None when the verdict was accepted.
        """
        reason = self._validate(verdict, valid_evidence_ids, evidence_text)
        if reason is not None:
            return draft, reason

        merged = draft.model_copy(deep=True)
        merged.research_question = verdict.research_question or merged.research_question
        merged.hypothesis = verdict.hypothesis or merged.hypothesis
        merged.subjects = verdict.subjects or merged.subjects
        merged.independent_variables = verdict.independent_variables or merged.independent_variables
        merged.dependent_variables = verdict.dependent_variables or merged.dependent_variables
        merged.experimental_groups = verdict.experimental_groups or merged.experimental_groups
        merged.control_groups = verdict.control_groups or merged.control_groups
        merged.intervention = verdict.intervention or merged.intervention
        merged.measurements = verdict.measurements or merged.measurements
        merged.conclusions = [
            c for c in draft.conclusions
        ]  # deterministic conclusions stay; LLM may append its own
        for conclusion in verdict.conclusions:
            if not any(c.statement == conclusion.statement for c in merged.conclusions):
                merged.conclusions.append(conclusion)
        return merged, None

    # --- validation -------------------------------------------------------------

    def _validate(
        self, verdict: LlmNormalizationVerdict, valid_evidence_ids: set[str], evidence_text: str
    ) -> Optional[str]:
        for conclusion in verdict.conclusions:
            unknown = [eid for eid in conclusion.evidence_ids if eid not in valid_evidence_ids]
            if unknown:
                return f"conclusion cites unknown evidence ids: {unknown}"
            if not conclusion.evidence_ids:
                return "conclusion without evidence citation"
            if self._numbers_not_in_evidence(conclusion.statement, evidence_text):
                return f"conclusion contains numbers absent from evidence: {conclusion.statement!r}"
            if self._direction_not_in_evidence(conclusion.statement, evidence_text):
                return f"conclusion direction not supported by evidence: {conclusion.statement!r}"
        for field in (
            verdict.research_question,
            verdict.hypothesis,
            verdict.intervention,
            *verdict.dependent_variables,
            *verdict.independent_variables,
        ):
            if field and self._numbers_not_in_evidence(field, evidence_text):
                return f"normalized field contains numbers absent from evidence: {field!r}"
        return None

    @staticmethod
    def _numbers_not_in_evidence(text: str, evidence_text: str) -> bool:
        for number in _NUMBER_RE.findall(text):
            if number not in evidence_text:
                return True
        for p_value in P_VALUE_RE.findall(text):
            if p_value.replace(" ", "") not in evidence_text.replace(" ", ""):
                return True
        return False

    @staticmethod
    def _direction_not_in_evidence(text: str, evidence_text: str) -> bool:
        has_increase = INCREASE_RE.search(text) is not None
        has_decrease = DECREASE_RE.search(text) is not None
        if has_increase and not INCREASE_RE.search(evidence_text):
            return True
        if has_decrease and not DECREASE_RE.search(evidence_text):
            return True
        return False

    # --- prompt / transport --------------------------------------------------------

    def _build_user_prompt(
        self, ref: FigureReference, evidences: list[PaperEvidence], draft: ExperimentModel
    ) -> str:
        lines = [f"Figure: {ref.figure_id}", "", "Evidence:"]
        for evidence in evidences:
            lines.append(f"[{evidence.evidence_id}] ({evidence.role}, {evidence.section_type}) {evidence.text}")
        lines += [
            "",
            "Deterministic draft:",
            json.dumps(
                {
                    "research_question": draft.research_question,
                    "hypothesis": draft.hypothesis,
                    "subjects": draft.subjects,
                    "independent_variables": draft.independent_variables,
                    "dependent_variables": draft.dependent_variables,
                    "experimental_groups": draft.experimental_groups,
                    "control_groups": draft.control_groups,
                    "intervention": draft.intervention,
                    "measurements": draft.measurements,
                    "observations": [
                        {
                            "statement": o.statement,
                            "direction": o.direction,
                            "significance": o.significance,
                            "relationship_type": o.relationship_type,
                        }
                        for o in draft.observations
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "Return the strict JSON now.",
        ]
        return "\n".join(lines)

    def _parse_verdict(self, content: str) -> Optional[LlmNormalizationVerdict]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", cleaned).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None
        try:
            return LlmNormalizationVerdict.model_validate(data)
        except Exception:
            return None

    def _chat(self, payload: str) -> str:
        client = self._client or httpx.Client(timeout=self.timeout)
        own_client = self._client is None
        try:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "temperature": 0.0,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": payload},
                    ],
                },
            )
            response.raise_for_status()
            body = response.json()
            return body["choices"][0]["message"]["content"]
        except httpx.HTTPError as exc:
            raise LlmNormalizationError(f"LLM request failed: {exc}") from exc
        except (KeyError, IndexError, ValueError) as exc:
            raise LlmNormalizationError(f"LLM response malformed: {exc}") from exc
        finally:
            if own_client:
                client.close()
