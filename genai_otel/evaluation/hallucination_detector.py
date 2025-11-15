"""Hallucination Detection for GenAI applications.

This module provides hallucination detection capabilities to identify potentially
false or unsupported claims in LLM responses.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import HallucinationConfig

logger = logging.getLogger(__name__)


@dataclass
class HallucinationResult:
    """Result of hallucination detection.

    Attributes:
        has_hallucination: Whether potential hallucination was detected
        hallucination_score: Overall hallucination risk score (0.0-1.0)
        hallucination_indicators: List of indicators found
        factual_claim_count: Count of factual claims detected
        unsupported_claims: Claims that appear unsupported
        hedge_words_count: Count of hedge words (indicating uncertainty)
        citation_count: Count of citations found
        original_text: Original input text
        context_text: Optional context text for fact-checking
    """

    has_hallucination: bool
    hallucination_score: float = 0.0
    hallucination_indicators: List[str] = field(default_factory=list)
    factual_claim_count: int = 0
    unsupported_claims: List[str] = field(default_factory=list)
    hedge_words_count: int = 0
    citation_count: int = 0
    original_text: Optional[str] = None
    context_text: Optional[str] = None


class HallucinationDetector:
    """Hallucination detector using pattern-based heuristics.

    This detector identifies potential hallucinations through:
    - Specific factual claim patterns (dates, numbers, names)
    - Hedge word detection (may, might, possibly)
    - Citation presence
    - Contradiction detection with provided context
    - Confidence marker detection

    Note: This is a heuristic-based detector. For production use cases requiring
    high accuracy, consider integrating dedicated fact-checking services or models.
    """

    # Patterns for factual claims
    FACTUAL_CLAIM_PATTERNS = {
        "specific_dates": [
            r"\b(?:in|on|during)\s+\d{4}\b",  # "in 2020"
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        ],
        "specific_numbers": [
            r"\$?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|trillion|thousand)\b",  # $45.7 billion, 328 million
            r"\b(?:exactly|precisely|approximately)\s+\d+(?:,\d{3})*(?:\.\d+)?\b",
            r"\b\d{1,3}(?:,\d{3})+\b",  # 1,234 employees
            r"\b\d+(?:\.\d+)?%",  # 99.9% satisfaction
        ],
        "specific_names": [
            r"\b(?:according\s+to|says|stated|claimed)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b",
            r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:said|wrote|published|discovered)\b",
        ],
        "statistics": [
            r"\b\d+(?:\.\d+)?%\s+of\b",
            r"\b(?:studies?|research|data)\s+(?:shows?|indicates?|suggests?)\s+that\b",
        ],
    }

    # Hedge words indicating uncertainty
    HEDGE_WORDS = [
        "may",
        "might",
        "possibly",
        "probably",
        "perhaps",
        "likely",
        "could",
        "would",
        "should",
        "seems",
        "appears",
        "suggests",
        "indicates",
        "potentially",
        "allegedly",
        "reportedly",
    ]

    # High-confidence markers (lack of these may indicate hallucination)
    CONFIDENCE_MARKERS = {
        "citations": [
            r"\[(?:\d+|[a-zA-Z]+)\]",  # [1] or [a]
            r"\((?:Source|Ref|Citation):",
            r"https?://\S+",  # URLs
            r"\([A-Z][a-z]+(?:\s+et\s+al\.)?,\s+\d{4}\)",  # (Smith, 2020) or (Jones et al., 2021)
            r"¹|²|³|⁴|⁵|⁶|⁷|⁸|⁹",  # Superscript numbers for footnotes
        ],
        "attribution": [
            r"according\s+to",
            r"based\s+on",
            r"as\s+(?:per|stated|mentioned)\s+(?:in|by)",
        ],
    }

    # Absolute statement markers
    ABSOLUTE_WORDS = [
        "absolutely",
        "certainly",
        "definitely",
        "always",
        "never",
        "all",
        "none",
        "every",
    ]

    def __init__(self, config: HallucinationConfig):
        """Initialize hallucination detector.

        Args:
            config: Hallucination detection configuration
        """
        self.config = config

    def is_available(self) -> bool:
        """Check if hallucination detector is available.

        Returns:
            bool: Always True (pattern-based detection always available)
        """
        return True

    def detect(self, text: str, context: Optional[str] = None) -> HallucinationResult:
        """Detect potential hallucinations in text.

        Args:
            text: Text to analyze (typically a response)
            context: Optional context text to check against

        Returns:
            HallucinationResult: Detection results
        """
        if not self.config.enabled:
            return HallucinationResult(
                has_hallucination=False, original_text=text, context_text=context
            )

        try:
            indicators = []
            unsupported_claims = []

            # Count factual claims
            factual_claims_count = self._count_factual_claims(text)

            # Count hedge words (only if enabled in config)
            hedge_count = self._count_hedge_words(text) if self.config.check_hedging else 0

            # Count citations/attributions (only if enabled in config)
            citation_count = self._count_citations(text) if self.config.check_citations else 0

            # Calculate hallucination score based on heuristics
            hallucination_score = self._calculate_hallucination_score(
                text, factual_claims_count, hedge_count, citation_count, context
            )

            # Identify indicators
            if factual_claims_count > 0 and citation_count == 0:
                indicators.append("specific_claims_without_citations")
                unsupported_claims.extend(self._extract_factual_claims(text)[:3])

            if hedge_count > 3:
                indicators.append("high_uncertainty_language")

            # Check for absolute statements without citations
            if self._has_absolute_statements(text) and citation_count == 0:
                indicators.append("absolute_statements")

            # Check for internal contradictions within the text
            if self._check_internal_contradiction(text):
                indicators.append("context_contradiction")
                hallucination_score = min(hallucination_score + 0.3, 1.0)

            # Check for contradictions with external context
            if context and self._check_context_contradiction(text, context):
                if "context_contradiction" not in indicators:
                    indicators.append("context_contradiction")
                hallucination_score = min(hallucination_score + 0.3, 1.0)

            has_hallucination = hallucination_score >= self.config.threshold

            return HallucinationResult(
                has_hallucination=has_hallucination,
                hallucination_score=hallucination_score,
                hallucination_indicators=indicators,
                factual_claim_count=factual_claims_count,
                unsupported_claims=unsupported_claims,
                hedge_words_count=hedge_count,
                citation_count=citation_count,
                original_text=text,
                context_text=context,
            )

        except Exception as e:
            logger.error("Error detecting hallucinations: %s", e, exc_info=True)
            return HallucinationResult(
                has_hallucination=False, original_text=text, context_text=context
            )

    def _count_factual_claims(self, text: str) -> int:
        """Count factual claims in text.

        Args:
            text: Text to analyze

        Returns:
            int: Number of factual claims found
        """
        count = 0
        for claim_type, patterns in self.FACTUAL_CLAIM_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                count += len(matches)
        return count

    def _extract_factual_claims(self, text: str) -> List[str]:
        """Extract factual claims from text.

        Args:
            text: Text to analyze

        Returns:
            List[str]: List of factual claims found
        """
        claims = []
        for claim_type, patterns in self.FACTUAL_CLAIM_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Get sentence containing the match
                    sentence_pattern = r"[^.!?]*" + re.escape(match.group()) + r"[^.!?]*[.!?]"
                    sentence_match = re.search(sentence_pattern, text, re.IGNORECASE)
                    if sentence_match:
                        claims.append(sentence_match.group().strip())
        return claims

    def _count_hedge_words(self, text: str) -> int:
        """Count hedge words indicating uncertainty.

        Args:
            text: Text to analyze

        Returns:
            int: Number of hedge words found
        """
        text_lower = text.lower()
        count = 0
        for hedge_word in self.HEDGE_WORDS:
            # Use word boundaries to avoid partial matches
            pattern = r"\b" + re.escape(hedge_word) + r"\b"
            count += len(re.findall(pattern, text_lower))
        return count

    def _count_citations(self, text: str) -> int:
        """Count citations and attributions in text.

        Args:
            text: Text to analyze

        Returns:
            int: Number of citations found
        """
        count = 0
        for citation_type, patterns in self.CONFIDENCE_MARKERS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                count += len(matches)
        return count

    def _calculate_hallucination_score(
        self,
        text: str,
        factual_claims_count: int,
        hedge_count: int,
        citation_count: int,
        context: Optional[str] = None,
    ) -> float:
        """Calculate overall hallucination risk score.

        Args:
            text: Text to analyze
            factual_claims_count: Number of factual claims
            hedge_count: Number of hedge words
            citation_count: Number of citations
            context: Optional context text

        Returns:
            float: Hallucination score (0.0-1.0)
        """
        score = 0.0

        # High factual claims without citations increases score
        if factual_claims_count > 0:
            citation_ratio = citation_count / factual_claims_count
            if citation_ratio == 0:
                score += 0.4  # No citations for factual claims
            elif citation_ratio < 0.3:
                score += 0.2  # Few citations for factual claims

        # High hedge word count increases score
        if len(text.split()) > 0:
            hedge_density = hedge_count / len(text.split())
            if hedge_density > 0.1:  # More than 10% hedge words
                score += 0.3

        # Very specific claims without attribution
        specific_patterns = [
            r"\b(?:exactly|precisely)\s+\d+",
            r"\bthe\s+fact\s+(?:is|that)\b",
            r"\b(?:definitely|certainly|absolutely)\b",
        ]
        for pattern in specific_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.1

        return min(score, 1.0)

    def _has_absolute_statements(self, text: str) -> bool:
        """Check if text contains absolute statements.

        Args:
            text: Text to analyze

        Returns:
            bool: True if absolute statements found
        """
        text_lower = text.lower()
        for absolute_word in self.ABSOLUTE_WORDS:
            pattern = r"\b" + re.escape(absolute_word) + r"\b"
            if re.search(pattern, text_lower):
                return True
        return False

    def _check_internal_contradiction(self, text: str) -> bool:
        """Check if text contradicts itself internally.

        This detects cases where the same subject has different values mentioned.
        For example: "The population is 300 million. However, the population is 400 million."

        Args:
            text: Text to check

        Returns:
            bool: True if potential internal contradiction detected
        """
        # Split into sentences
        sentences = re.split(r"[.!?]+", text)
        if len(sentences) < 2:
            return False

        # Look for contradictory numbers for the same subject
        # Pattern: extract subject + number pairs
        number_pattern = r"(\w+)\s+(?:is|are|was|were|reached?)\s+(\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|thousand))?)"

        subject_values = {}
        for sentence in sentences:
            matches = re.findall(number_pattern, sentence.lower())
            for subject, value in matches:
                if subject in subject_values:
                    # Same subject mentioned with different value
                    if subject_values[subject] != value:
                        return True
                else:
                    subject_values[subject] = value

        # Look for contradictory statements with "however", "but", "although"
        contradiction_markers = [r"\bhowever\b", r"\bbut\b", r"\balthough\b", r"\bactually\b"]
        has_contradiction_marker = any(
            re.search(marker, text.lower()) for marker in contradiction_markers
        )

        if has_contradiction_marker and len(sentences) >= 2:
            # Check if similar topics discussed with different numbers
            numbers_in_text = re.findall(
                r"\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|thousand))?", text
            )
            if len(numbers_in_text) >= 2:
                # Multiple different numbers with contradiction marker suggests contradiction
                unique_numbers = set(numbers_in_text)
                if len(unique_numbers) >= 2:
                    return True

        return False

    def _check_context_contradiction(self, text: str, context: str) -> bool:
        """Check if response contradicts provided context.

        This is a simple heuristic check. For production use, consider
        using dedicated NLI (Natural Language Inference) models.

        Args:
            text: Text to check
            context: Context to check against

        Returns:
            bool: True if potential contradiction detected
        """
        # Extract key terms from context
        context_lower = context.lower()
        text_lower = text.lower()

        # Check for number contradictions with same subject
        # Extract subject + number pairs from both texts
        number_pattern = r"(\w+)\s+(?:is|are|was|were|reached?)\s+(?:exactly|approximately|about)?\s*(\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|thousand))?)"

        context_numbers = {}
        text_numbers = {}

        for match in re.finditer(number_pattern, context_lower):
            subject, number = match.groups()
            context_numbers[subject] = number.strip()

        for match in re.finditer(number_pattern, text_lower):
            subject, number = match.groups()
            text_numbers[subject] = number.strip()

        # Check if same subject has different numbers
        for subject in context_numbers:
            if subject in text_numbers and context_numbers[subject] != text_numbers[subject]:
                return True

        # Simple negation detection
        negation_patterns = [
            r"\bnot\b",
            r"\bno\b",
            r"\bnever\b",
            r"\bn't\b",
            r"\bimpossible\b",
            r"\bfalse\b",
        ]

        context_has_negation = any(
            re.search(pattern, context_lower) for pattern in negation_patterns
        )
        text_has_negation = any(re.search(pattern, text_lower) for pattern in negation_patterns)

        # Very simple contradiction check: if context says "not X" and text says "X"
        # This is a naive approach - consider using NLI models for better accuracy
        if context_has_negation != text_has_negation:
            # Extract nouns/entities and check if they appear in both
            context_words = set(re.findall(r"\b[a-z]{4,}\b", context_lower))
            text_words = set(re.findall(r"\b[a-z]{4,}\b", text_lower))
            common_words = context_words & text_words

            if len(common_words) > 2:  # Some overlap in content
                return True

        return False

    def analyze_batch(
        self, texts: List[str], contexts: Optional[List[str]] = None
    ) -> List[HallucinationResult]:
        """Analyze multiple texts for hallucinations.

        Args:
            texts: List of texts to analyze
            contexts: Optional list of context texts (same length as texts)

        Returns:
            List[HallucinationResult]: Detection results for each text
        """
        results = []
        for i, text in enumerate(texts):
            context = contexts[i] if contexts and i < len(contexts) else None
            results.append(self.detect(text, context))
        return results

    def get_statistics(self, results: List[HallucinationResult]) -> Dict[str, Any]:
        """Get statistics from multiple detection results.

        Args:
            results: List of detection results

        Returns:
            Dict[str, Any]: Statistics including hallucination rates
        """
        total_hallucinations = sum(1 for r in results if r.has_hallucination)

        # Aggregate indicator counts
        indicator_counts: Dict[str, int] = {}
        for result in results:
            for indicator in result.hallucination_indicators:
                indicator_counts[indicator] = indicator_counts.get(indicator, 0) + 1

        # Calculate average scores
        avg_score = sum(r.hallucination_score for r in results) / len(results) if results else 0.0
        avg_hedge_count = (
            sum(r.hedge_words_count for r in results) / len(results) if results else 0.0
        )
        avg_citation_count = (
            sum(r.citation_count for r in results) / len(results) if results else 0.0
        )

        return {
            "total_responses_analyzed": len(results),
            "hallucination_count": total_hallucinations,
            "hallucination_rate": total_hallucinations / len(results) if results else 0.0,
            "indicator_counts": indicator_counts,
            "average_hallucination_score": avg_score,
            "average_hedge_words": avg_hedge_count,
            "average_citations": avg_citation_count,
            "most_common_indicator": (
                max(indicator_counts.items(), key=lambda x: x[1])[0] if indicator_counts else None
            ),
        }
