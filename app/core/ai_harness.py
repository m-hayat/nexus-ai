"""
Nexus-AI Platform — AI Safety Harness.

Ki:
- CitationValidator: parse [Source N], check bounds
- FaithfulnessScorer: LLM-as-judge faithfulness 0.0-1.0
- HarnessResult: combined verdict
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from langchain_core.documents import Document

from app.config import Settings, get_settings
from app.core.llm_factory import get_llm
from app.core.prompts import FAITHFULNESS_PROMPT

logger = logging.getLogger(__name__)

# Regex — [Source 1], [Source 12], etc.
CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)


# ═══════════════════════════════════════════════════════════════
# RESULT DATACLASS
# ═══════════════════════════════════════════════════════════════

@dataclass
class HarnessResult:
    """Combined validation result."""
    valid_citations: list[dict] = field(default_factory=list)
    invalid_citations: list[dict] = field(default_factory=list)
    faithfulness_score: float | None = None
    validation_passed: bool = False
    failure_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "valid_citations": self.valid_citations,
            "invalid_citations": self.invalid_citations,
            "faithfulness_score": self.faithfulness_score,
            "validation_passed": self.validation_passed,
            "failure_reason": self.failure_reason,
        }


# ═══════════════════════════════════════════════════════════════
# CITATION VALIDATOR
# ═══════════════════════════════════════════════════════════════

class CitationValidator:
    """
    Parse + validate [Source N] citations in an answer.

    Checks:
    1. Bounds: N must be 1 <= N <= len(retrieved_docs)
    2. Presence: answer has at least 1 citation (unless fallback/refusal)
    """

    def __init__(self, min_citations: int = 1):
        self.min_citations = min_citations

    def extract_citations(self, answer: str) -> list[int]:
        """
        Ki: Answer theke [Source N] extract kore, unique sorted list return.
        Kno: Duplicate citation dedupe — same source bar bar cite korle ekbar.
        """
        if not answer:
            return []
        matches = CITATION_PATTERN.findall(answer)
        return sorted({int(m) for m in matches})

    def validate(
        self,
        answer: str,
        retrieved_docs: list[Document],
    ) -> tuple[list[dict], list[dict]]:
        """
        Ki: Valid ar invalid citation er list return kore.

        Valid: source_id bounds e ache
        Invalid: source_id bounds er baire ba negative

        How:
          - Extract sob [Source N]
          - Prottek ta bounds check kori
          - Valid gulo metadata soho return
        """
        cited_ids = self.extract_citations(answer)
        max_source = len(retrieved_docs)

        valid: list[dict] = []
        invalid: list[dict] = []

        for source_id in cited_ids:
            if 1 <= source_id <= max_source:
                doc = retrieved_docs[source_id - 1]
                snippet = doc.page_content.strip()[:300]
                valid.append({
                    "source_id": source_id,
                    "snippet": snippet,
                    "score": 1.0,
                })
            else:
                invalid.append({
                    "source_id": source_id,
                    "reason": f"out_of_bounds (max {max_source})",
                })

        return valid, invalid

    def has_minimum_citations(self, answer: str) -> bool:
        """Answer e minimum number of citation ache kina."""
        return len(self.extract_citations(answer)) >= self.min_citations


# ═══════════════════════════════════════════════════════════════
# FAITHFULNESS SCORER (LLM-as-judge)
# ═══════════════════════════════════════════════════════════════

class FaithfulnessScorer:
    """
    LLM-as-judge faithfulness scoring.

    Ki: Answer context er sathe faithful kina 0.0-1.0 score.
    Kno: Bounds check shudhu citation number verify kore, kintu claim
         context e ache kina verify kore na. Ei class oita kore.
    How: LLM ke structured JSON output chai, parse kori, fallback ache.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def score(self, answer: str, context: str) -> float | None:
        """
        Ki: Faithfulness score return kore 0.0-1.0.
        Kno: LLM-as-judge with JSON output. Robust fallback.
        How: Prompt → LLM → JSON parse → score extract.

        Returns None jodi score compute korte na pare.
        """
        if not answer or not context:
            return None

        try:
            llm = get_llm()
            prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
            response = llm.invoke(prompt)
            raw = response.content if hasattr(response, "content") else str(response)
            return self._parse_score(raw) # pyright: ignore[reportArgumentType]
        except Exception as exc:
            logger.exception("Faithfulness scoring failed: %s", exc)
            return None

    def _parse_score(self, raw: str) -> float | None:
        """
        Ki: LLM output theke score extract kore.
        Kno: Small models sometimes JSON ke markdown e wrap kore.
             Multiple fallback strategy.
        """
        if not raw:
            return None

        # Strip markdown code fences
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Remove ```json ... ``` wrapper
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)

        # Try direct JSON parse
        try:
            data = json.loads(cleaned)
            score = float(data.get("score"))
            return max(0.0, min(1.0, score))  # clamp 0-1
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Fallback — regex for {"score": X} or score: X
        match = re.search(r'"?score"?\s*[:=]\s*([\d.]+)', cleaned, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            except ValueError:
                pass

        logger.warning("Could not parse faithfulness score from: %r", raw[:200])
        return None


# ═══════════════════════════════════════════════════════════════
# COMBINED HARNESS
# ═══════════════════════════════════════════════════════════════

class SafetyHarness:
    """
    Combined validator + scorer.

    Usage:
        harness = SafetyHarness()
        result = harness.validate(answer, docs, context)
        if not result.validation_passed:
            # fallback response
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.citation_validator = CitationValidator(min_citations=1)
        self.faithfulness_scorer = FaithfulnessScorer(self.settings)
        self.min_faithfulness = self.settings.citation_min_faithfulness

    def validate(
        self,
        answer: str,
        retrieved_docs: list[Document],
        context: str,
    ) -> HarnessResult:
        """
        Ki: Full validation pipeline.
        How:
          1. Citation bounds check
          2. Faithfulness scoring (LLM-as-judge)
          3. Combined verdict

        Validation passes if:
          - No invalid citations AND
          - At least 1 valid citation AND
          - Faithfulness >= threshold (or scoring failed but citations valid)
        """
        result = HarnessResult()

        # ── Step 1: Citation bounds ──
        valid, invalid = self.citation_validator.validate(answer, retrieved_docs)
        result.valid_citations = valid
        result.invalid_citations = invalid

        if invalid:
            result.failure_reason = (
                f"{len(invalid)} invalid citation(s) detected: "
                f"{[c['source_id'] for c in invalid]}"
            )
            logger.warning("Harness: %s", result.failure_reason)
            result.validation_passed = False
            return result

        if not valid:
            # Refusal response ("I don't have...") er jonno citation nei
            if self._is_refusal(answer):
                result.validation_passed = True
                result.failure_reason = None
                return result

            result.failure_reason = "no_citations_in_answer"
            logger.warning("Harness: %s", result.failure_reason)
            result.validation_passed = False
            return result

        # ── Step 2: Faithfulness score ──
        score = self.faithfulness_scorer.score(answer, context)
        result.faithfulness_score = score

        if score is None:
            # Scoring fail — citations valid, so allow with warning
            logger.warning("Harness: faithfulness scoring failed, allowing with caution")
            result.validation_passed = True
            result.failure_reason = "faithfulness_scoring_failed"
            return result

        # ── Step 3: Combined verdict ──
        if score < self.min_faithfulness:
            result.failure_reason = (
                f"faithfulness {score:.2f} below threshold {self.min_faithfulness}"
            )
            logger.warning("Harness: %s", result.failure_reason)
            result.validation_passed = False
            return result

        result.validation_passed = True
        return result

    def _is_refusal(self, answer: str) -> bool:
        """
        Ki: Answer ekta refusal ("I don't have that information") kina.
        Kno: Refusal response e citation thake na — ei case allow korte hobe.
        """
        if not answer:
            return False
        refusal_markers = [
            "i don't have that information",
            "i do not have that information",
            "i don't know",
            "not in the betsson policy",
            "no information available",
        ]
        lowered = answer.lower()
        return any(marker in lowered for marker in refusal_markers)


# ═══════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════

_harness: SafetyHarness | None = None


def get_harness() -> SafetyHarness:
    """Singleton safety harness."""
    global _harness
    if _harness is None:
        _harness = SafetyHarness()
    return _harness
