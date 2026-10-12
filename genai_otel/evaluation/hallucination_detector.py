"""Hallucination Detection for GenAI applications.

This module provides hallucination detection capabilities to identify potentially
false or unsupported claims in LLM responses.
"""

import bisect
import logging
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from .config import DEFAULT_CACHE_SIZE, HallucinationConfig
from .result_cache import ResultCache

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"[.!?]")
_WORD_CHARACTER = re.compile(r"\w")

# A subject, a verb and a number: "revenue was 45.7 million".
_NUMBER = r"(\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|thousand))?)"
_SUBJECT_AND_VERB = r"(\w+)\s+(?:is|are|was|were|reached?)\s+"
# The same with an adverb allowed before the number: "was about 45.7 million".
# The space after the adverb belongs to the adverb. Written as
# "\s+(?:adverb)?\s*", a run of blank space could be split between the two in
# as many ways as it has characters, and every one of them was tried.
_ADVERB = r"(?:(?:exactly|approximately|about)\s*)?"
# Each expression twice: as it is, and held to the start of a word. See
# HallucinationDetector._find_pairs for why.
_SUBJECT_NUMBER = re.compile(_SUBJECT_AND_VERB + _NUMBER)
_SUBJECT_NUMBER_AT_WORD = re.compile(r"(?<!\w)" + _SUBJECT_AND_VERB + _NUMBER)
_SUBJECT_NUMBER_IN_CONTEXT = re.compile(_SUBJECT_AND_VERB + _ADVERB + _NUMBER)
_SUBJECT_NUMBER_IN_CONTEXT_AT_WORD = re.compile(r"(?<!\w)" + _SUBJECT_AND_VERB + _ADVERB + _NUMBER)


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
            # $45.7 billion, 328 million. A number is read from where it starts:
            # not from inside a run of digits, "(?<!\d)", and not from a group
            # of three that the number before the comma leads into,
            # "(?!(?<=\d,)\d{3}(?!\d))". Either would be read again as part of
            # the number it belongs to, so a long number, or a long list of
            # them, took time in proportion to the square of its length.
            r"\$?(?<!\d)(?!(?<=\d,)\d{3}(?!\d))\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|trillion|thousand)\b",
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
        self._cache = ResultCache(getattr(config, "cache_size", DEFAULT_CACHE_SIZE))

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
            key = self._cache_key(text, context)
            cached = self._cache.get(key)
            if cached is not None:
                return replace(cached, original_text=text, context_text=context)

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
                unsupported_claims.extend(self._extract_factual_claims(text, limit=3))

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

            result = HallucinationResult(
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
            # Stored without the texts: the entry outlives the request it came from.
            self._cache.put(key, replace(result, original_text=None, context_text=None))
            return result

        except Exception as e:
            # Not remembered: the detector must be asked again once it can answer.
            logger.error("Error detecting hallucinations: %s", e, exc_info=True)
            return HallucinationResult(
                has_hallucination=False, original_text=text, context_text=context
            )

    def _cache_key(self, text: str, context: Optional[str]):
        """Key for a result: both texts and every setting the result depends on."""
        if not self._cache.enabled:
            return None
        context_key = self._cache.key(context) if context else ()
        if context_key is None:
            return None
        try:
            return self._cache.key(
                text,
                context_key,
                self.config.threshold,
                bool(self.config.check_citations),
                bool(self.config.check_hedging),
            )
        except TypeError:
            return None

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

    def _extract_factual_claims(self, text: str, limit: Optional[int] = None) -> List[str]:
        """Extract factual claims from text.

        Each claim is reported as the first sentence of the text that contains
        its wording.

        Args:
            text: Text to analyze
            limit: Stop once this many claims have been found. None for all.

        Returns:
            List[str]: List of factual claims found
        """
        claims: List[str] = []
        ends = [match.start() for match in _SENTENCE_END.finditer(text)]
        if not ends:
            # No sentence ends anywhere, so there is none to report a claim as.
            return claims
        # One search per wording, not one per place it occurs: a figure quoted
        # in every paragraph is looked up once.
        sentences: Dict[str, Optional[str]] = {}
        lowered = text.lower() if text.isascii() else None
        for claim_type, patterns in self.FACTUAL_CLAIM_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    claim = match.group()
                    if claim not in sentences:
                        sentences[claim] = self._sentence_with(text, lowered, claim, ends)
                    sentence = sentences[claim]
                    if sentence is None:
                        continue
                    claims.append(sentence)
                    if limit is not None and len(claims) >= limit:
                        return claims
        return claims

    @staticmethod
    def _sentence_with(
        text: str, lowered: Optional[str], claim: str, ends: List[int]
    ) -> Optional[str]:
        """The first sentence of ``text`` that contains ``claim``, whatever its case.

        What ``[^.!?]*<claim>[^.!?]*[.!?]`` finds, worked out from where
        sentences end (``ends``) instead of by trying that expression from every
        position of the text, which on a long text without full stops did not
        finish. None when no sentence that contains the claim comes to an end.
        """
        last_end = ends[-1]
        length = len(claim)
        # Where the claim first occurs. It has to be over by the last sentence
        # end to be inside a sentence at all.
        if lowered is not None:
            # Plain text: looking for the lower-cased wording finds the same
            # places as ignoring case does, and is much the faster way.
            wording = claim.lower()
            claim_start = lowered.find(wording, 0, last_end)

            def occurs_at(start: int) -> bool:
                return lowered.startswith(wording, start)

        else:
            occurrence = re.compile(re.escape(claim), re.IGNORECASE)
            found = occurrence.search(text, 0, last_end)
            claim_start = found.start() if found else -1

            def occurs_at(start: int) -> bool:
                return occurrence.match(text, start) is not None

        if claim_start < 0:
            return None
        claim_end = claim_start + length

        # The sentence begins after the last sentence end before the claim.
        before = bisect.bisect_left(ends, claim_start)
        sentence_start = ends[before - 1] + 1 if before else 0

        # A claim can contain a full stop itself ("12.5%"). The expression took
        # the last place the claim occurs before the first sentence end; with a
        # full stop inside the claim that may be a later, overlapping one.
        first_end = ends[before]
        if first_end < claim_end:
            for start in range(first_end, claim_start, -1):
                if start + length <= last_end and occurs_at(start):
                    claim_end = start + length
                    break

        sentence_end = ends[bisect.bisect_left(ends, claim_end)]
        return text[sentence_start : sentence_end + 1].strip()

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

    @staticmethod
    def _find_pairs(anywhere, at_word_start, text: str) -> List[Tuple[str, str]]:
        """What ``anywhere.findall(text)`` finds, in time proportional to the text.

        The expression begins with a word, the subject. Tried from every
        position, a failed attempt at the start of a word is repeated from each
        of its other letters, so one long word (a base64 blob, a long number)
        took time in proportion to the square of its length. ``at_word_start``
        is the same expression held to the start of a word. A subject can begin
        inside a word in one case only, straight after the number of the pair
        before it ("is 5abc is 7"), and that one place is tried as it is.
        """
        pairs = []
        position = 0
        while position <= len(text):
            match = None
            if (
                0 < position < len(text)
                and _WORD_CHARACTER.match(text, position - 1)
                and _WORD_CHARACTER.match(text, position)
            ):
                match = anywhere.match(text, position)
            if match is None:
                match = at_word_start.search(text, position)
            if match is None:
                break
            pairs.append(match.groups())
            position = match.end()
        return pairs

    def _subject_numbers(self, text: str) -> List[Tuple[str, str]]:
        """Every "<subject> is <number>" in ``text``, as (subject, number)."""
        return self._find_pairs(_SUBJECT_NUMBER, _SUBJECT_NUMBER_AT_WORD, text)

    def _subject_numbers_in_context(self, text: str) -> List[Tuple[str, str]]:
        """Every "<subject> is about <number>" in ``text``, as (subject, number)."""
        return self._find_pairs(
            _SUBJECT_NUMBER_IN_CONTEXT, _SUBJECT_NUMBER_IN_CONTEXT_AT_WORD, text
        )

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
        subject_values = {}
        for sentence in sentences:
            matches = self._subject_numbers(sentence.lower())
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
        context_numbers = {}
        text_numbers = {}

        for subject, number in self._subject_numbers_in_context(context_lower):
            context_numbers[subject] = number.strip()

        for subject, number in self._subject_numbers_in_context(text_lower):
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
