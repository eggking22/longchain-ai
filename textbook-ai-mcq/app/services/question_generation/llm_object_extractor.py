"""Optional LLM object extraction with a deterministic verbatim-span gate.

Reuses the access pattern of Phase 4's LlmSemanticNormalizer (raw httpx POST to
an OpenAI-compatible /chat/completions, temperature 0, strict JSON with one
re-ask, injectable httpx.Client for offline tests). It only ever *patches* the
kind-label fallback of a DATA statement whose evidence sentence could not be
quoted:

- the prompt only contains the value, its kind, and the evidence texts;
- the returned object phrase must be a verbatim contiguous span of one of the
  evidence texts (after the same whitespace normalization the statement
  builder applies) and at most 80 characters;
- any violation returns None and the deterministic fallback stands.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import httpx

from app.schemas.question_draft import LlmObjectExtraction

from .perturbations import normalize_typography

DEFAULT_LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

SYSTEM_PROMPT = """\
You extract the measured object for one reported value from scientific \
evidence texts.

Rules:
- The object is a short noun phrase answering "the {kind} of WHAT?" — e.g. the \
treatment/reagent a concentration belongs to, the comparison a p-value belongs \
to, the quantity a percentage is measured over.
- Copy the object VERBATIM as a contiguous substring of one evidence text. \
Do not rephrase, translate, abbreviate or invent words.
- At most 80 characters, no leading/trailing punctuation.
- If no clear object is present in the texts, return an empty string.

Answer with strict JSON only (no prose, no markdown fences) matching:
{"object": str}
"""

_MAX_OBJECT_CHARS = 80
_STRIP_PUNCT = " \t\n\r,;.:"


class LlmObjectExtractionError(RuntimeError):
    """The extractor could not produce a usable answer (network, JSON, ...)."""


class LlmObjectExtractor:
    """LLM patcher over the DATA kind-label fallback; never a replacement."""

    name = "llm"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        max_attempts: int = 2,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_attempts = max_attempts
        self._client = client

    # --- public API -----------------------------------------------------------

    def extract(self, value: str, kind: str, texts: list[str]) -> Optional[str]:
        """Return a validated object phrase, or None when rejected/unavailable."""
        payload = self._build_user_prompt(value, kind, texts)
        for _ in range(self.max_attempts):
            content = self._chat(payload)
            verdict = self._parse_verdict(content)
            if verdict is not None:
                return self._validated(verdict, texts)
            payload += "\n\nYour previous answer was not valid JSON. Answer again with strict JSON only."
        raise LlmObjectExtractionError("LLM object extractor failed to return strict JSON")

    # --- validation -------------------------------------------------------------

    @staticmethod
    def _validated(verdict: LlmObjectExtraction, texts: list[str]) -> Optional[str]:
        phrase = verdict.object_phrase.strip().strip(_STRIP_PUNCT)
        if not phrase or len(phrase) > _MAX_OBJECT_CHARS:
            return None
        for text in texts:
            if phrase in normalize_typography(text):
                return phrase
        return None  # not a verbatim span of any evidence text → reject

    # --- prompt / transport --------------------------------------------------------

    @staticmethod
    def _build_user_prompt(value: str, kind: str, texts: list[str]) -> str:
        lines = [f"Reported value: {value}", f"Kind: {kind}", "", "Evidence texts:"]
        lines += [f"- {text}" for text in texts if text]
        lines += ["", "Return the strict JSON now."]
        return "\n".join(lines)

    @staticmethod
    def _parse_verdict(content: str) -> Optional[LlmObjectExtraction]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", cleaned).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None
        if isinstance(data, dict) and "object" in data and "object_phrase" not in data:
            data = {"object_phrase": data["object"]}  # the prompt asks for {"object": str}
        try:
            return LlmObjectExtraction.model_validate(data)
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
            raise LlmObjectExtractionError(f"LLM request failed: {exc}") from exc
        except (KeyError, IndexError, ValueError) as exc:
            raise LlmObjectExtractionError(f"LLM response malformed: {exc}") from exc
        finally:
            if own_client:
                client.close()
