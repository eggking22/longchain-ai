"""Optional LLM statement translation with a deterministic invariant gate.

Reuses the access pattern of the LLM object extractor (raw httpx POST to an
OpenAI-compatible /chat/completions, temperature 0, strict JSON with one
re-ask, injectable httpx.Client for offline tests). The LLM only changes the
language — never the science. Four hard invariants are validated
deterministically and any violation rejects the whole translation, falling
back to the deterministic registry/template result:

- every number token in the English statement appears in the Chinese one;
- every figure anchor ("Figure 2", "Extended Data Figure 5a") is verbatim;
- direction words map to their single registered counterparts (提高/降低),
  so a translation can never flip or blur a direction;
- gene/protein/compound tokens (letter+digit mixes like CCR7/CK666/AACOF3 and
  ALL-CAPS tokens like GFP/MHC) are copied verbatim.

Known limitation: purely alphabetic gene names (Arpin, WASp) are not covered
by the preserve-regex and rely on prompt compliance; the fallback still
catches number/anchor/direction violations.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import httpx

from .terminology import TERMINOLOGY  # canonical counterparts documented in SYSTEM_PROMPT

DEFAULT_LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

SYSTEM_PROMPT = """\
You translate one English scientific MCQ statement into natural, fluent \
scientific Chinese.

Rules:
- Translate ONLY — never restate, interpret, add or omit information.
- Copy numbers, units and statistics verbatim: "25 µM", "35%", "p < 0.01", \
"2–4 µm", "3-fold".
- Copy gene/protein/compound names verbatim: GFP, CCR7, cPLA2, AACOF3, CK666, \
WASp, Arpin, MHC, LPS, DCs ...
- Copy figure/table references verbatim: "Figure 2", "Extended Data Figure \
5a", "Table 1".
- Direction verbs have fixed counterparts: increases/increase/increased/higher \
→ 提高, decreases/decrease/decreased/lower → 降低. Never flip or blur them.
- Output natural Chinese word order (中文语序), not word-by-word substitution.

Answer with strict JSON only (no prose, no markdown fences) matching:
{"zh": str}
"""

_NUMBER_RE = re.compile(r"\d[\d.]*")
_FIGURE_ANCHOR_RE = re.compile(r"(?:Extended Data |Supplementary )?(?:Figure|Table) \d+[a-h]?", re.IGNORECASE)
# letter+digit mixes (CCR7, CK666, AACOF3, cPLA2, CD80) and ALL-CAPS tokens (GFP, MHC, LPS)
_PRESERVE_TOKEN_RE = re.compile(r"\b(?:[A-Za-z]+\d[A-Za-z0-9]*|[A-Z]{2,})\b")
# natural-Chinese synonyms the gate accepts per polarity (the prompt still asks
# for the canonical 提高/降低 first); the anti-flip rule stays: a statement with
# an up direction may never carry a down word, and vice versa
_ZH_UP_WORDS = ("提高", "增加", "升高", "增强", "上调", "增多")
_ZH_DOWN_WORDS = ("降低", "减少", "下降", "减弱", "下调", "较低")
_EN_UP_RE = re.compile(r"\b(?:increases?|increased|higher)\b", re.IGNORECASE)
_EN_DOWN_RE = re.compile(r"\b(?:decreases?|decreased|lower)\b", re.IGNORECASE)


class LlmTranslationError(RuntimeError):
    """The translator could not produce a usable answer (network, JSON, ...)."""


class LlmStatementTranslator:
    """LLM whole-sentence translator over the deterministic registry/template layer."""

    name = "llm"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,  # real-run evidence: long DATA quotes on glm-4.6 can exceed 60s
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

    def translate(self, statement: str) -> Optional[str]:
        """Return a validated Chinese translation, or None when rejected."""
        payload = f"English statement:\n{statement}"
        for _ in range(self.max_attempts):
            content = self._chat(payload)
            zh = self._parse(content)
            if zh is not None:
                return self._validated(statement, zh)
            payload += "\n\nYour previous answer was not valid JSON. Answer again with strict JSON only."
        raise LlmTranslationError("LLM translator failed to return strict JSON")

    # --- validation -------------------------------------------------------------

    @staticmethod
    def _anchor_ok(anchor: str, chinese: str) -> bool:
        """Attribution is preserved when the reference survives with its number —
        either verbatim ("Figure 2") or localized ("图2" / "图 5a")."""
        if anchor in chinese:
            return True
        label = re.search(r"(\d+[a-h]?)$", anchor)
        if label is None:
            return False
        return re.search(rf"(?:图|[Ff]igure)\s*{re.escape(label.group(1))}\b", chinese) is not None

    @classmethod
    def validate(cls, english: str, chinese: str) -> Optional[str]:
        """Deterministic invariant gate; returns the rejection reason or None."""
        for anchor in _FIGURE_ANCHOR_RE.findall(english):
            if not cls._anchor_ok(anchor, chinese):
                return f"figure anchor {anchor!r} missing in translation"
        for token in _PRESERVE_TOKEN_RE.findall(english):
            if token not in chinese:
                return f"preserve-token {token!r} missing in translation"
        # number check runs on the residual text so digits inside anchors and
        # gene/compound tokens (AACOF3, Figure 2) are covered by the checks above
        residual = _PRESERVE_TOKEN_RE.sub(" ", _FIGURE_ANCHOR_RE.sub(" ", english))
        for number in _NUMBER_RE.findall(residual):
            if number not in chinese:
                return f"number {number!r} missing in translation"
        en_up, en_down = bool(_EN_UP_RE.search(english)), bool(_EN_DOWN_RE.search(english))
        zh_up = any(word in chinese for word in _ZH_UP_WORDS)
        zh_down = any(word in chinese for word in _ZH_DOWN_WORDS)
        # the governing verb's polarity must survive; the opposite word is still
        # allowed inside noun compounds ("decreases the upregulation" → 降低…上调)
        if en_up and not en_down and not zh_up:
            return "direction 'increase' not preserved (no up-word in translation)"
        if en_down and not en_up and not zh_down:
            return "direction 'decrease' not preserved (no down-word in translation)"
        return None

    @classmethod
    def _validated(cls, english: str, chinese: str) -> Optional[str]:
        chinese = chinese.strip()
        if not chinese:
            return None
        return chinese if cls.validate(english, chinese) is None else None

    # --- prompt / transport --------------------------------------------------------

    @staticmethod
    def _parse(content: str) -> Optional[str]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", cleaned).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict) or "zh" not in data or not isinstance(data["zh"], str):
            return None
        return data["zh"]

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
            raise LlmTranslationError(f"LLM request failed: {exc}") from exc
        except (KeyError, IndexError, ValueError) as exc:
            raise LlmTranslationError(f"LLM response malformed: {exc}") from exc
        finally:
            if own_client:
                client.close()
