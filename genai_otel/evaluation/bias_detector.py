"""Bias Detection for GenAI applications.

This module provides bias detection capabilities using pattern-based and
optional ML-based approaches to identify demographic and other biases
in prompts and responses.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .config import BiasConfig

logger = logging.getLogger(__name__)


@dataclass
class BiasDetectionResult:
    """Result of bias detection.

    Attributes:
        has_bias: Whether bias was detected above threshold
        bias_scores: Bias scores by type
        max_score: Maximum bias score across all types
        detected_biases: List of bias types detected
        patterns_matched: Specific patterns that triggered detection
        original_text: Original input text
    """

    has_bias: bool
    bias_scores: Dict[str, float] = field(default_factory=dict)
    max_score: float = 0.0
    detected_biases: List[str] = field(default_factory=list)
    patterns_matched: Dict[str, List[str]] = field(default_factory=dict)
    original_text: Optional[str] = None


class BiasDetector:
    """Bias detector using pattern-based and optional ML approaches.

    This detector identifies various types of bias including:
    - Gender bias
    - Racial/ethnic bias
    - Religious bias
    - Age bias
    - Disability bias
    - Sexual orientation bias
    - Political bias

    Requirements:
        Optional: pip install fairlearn scikit-learn
    """

    # Pattern definitions for different bias types
    BIAS_PATTERNS = {
        "gender": {
            "patterns": [
                r"\b(women|woman|female|girls?|she|her)\s+(?:are|is|always|never|can't|cannot|shouldn't)\b",
                r"\b(men|man|male|boys?|he|him)\s+(?:are|is|always|never|can't|cannot|shouldn't)\b",
                r"\b(women|woman|female|girls?),?\s+(?:they're?|she's)\s+(?:always|never|too|so)\b",  # "women, they're always"
                r"\b(men|man|male|boys?),?\s+(?:they're?|he's)\s+(?:always|never|too|so)\b",
                r"\bfor\s+(his|her)\s+gender\b",
                r"\b(manly|womanly|girly|boyish)\s+(?:behavior|traits?|characteristic)",
                r"(?:cry|act|behave|think)\s+like\s+(?:a\s+)?(?:man|woman|boys?|girls?)",
                r"\b(?:real|proper|typical)\s+(?:man|woman|boy|girl)",
            ],
            "keywords": ["sexist", "misogyny", "misandry", "gender stereotype"],
        },
        "race": {
            "patterns": [
                r"\b(?:all|most|typical)?\s*(?:black|white|asian|hispanic|latino|arab)\s+people\s+(?:are|tend\s+to|always|never)\b",
                r"\brace\s+(?:card|baiting)\b",
                r"\b(?:act|sound|look)\s+(?:white|black|asian|hispanic)\b",
                r"\b(?:being|that's|stop\s+being|don't\s+be)\s+racist\b",  # Pattern for "racist" usage
            ],
            "keywords": [
                "racial slur"
            ],  # Removed "racist" keyword to avoid false political matches
        },
        "ethnicity": {
            "patterns": [
                r"\b(?:all|most|typical)\s+\w+(?:ese|ian|ish)\s+people\b",
                r"\bforeigner",
                r"\b(?:speak|accent|look)\s+like\s+(?:an?\s+)?\w+(?:ese|ian|ish)\b",
            ],
            "keywords": ["xenophobic", "ethnic stereotype", "ethnicity"],
        },
        "religion": {
            "patterns": [
                r"\b(?:muslims?|christians?|jews?|hindus?|buddhists?|atheists?)\s+(?:are|always|never|tend\s+to)\b",
                r"\bsharia\s+law\b",
                r"\breligious\s+extremis",
            ],
            "keywords": ["islamophobic", "anti-semitic", "religious bias", "religious stereotype"],
        },
        "age": {
            "patterns": [
                r"\b(?:old|elderly|senior)\s+people\s+(?:are|can't|cannot|shouldn't)\b",
                r"\b(?:millennials?|gen\s*z|zoomers?|boomers?)\s+(?:are|always|never)\b",
                r"\b(?:young)\s+people\s+(?:are|always|never)\b",
                r"\btoo\s+(?:old|young)\s+(?:to|for)\b",
                r"\b(?:act|look)\s+(?:your|their)\s+age\b",
            ],
            "keywords": ["ageist", "ageism", "age discrimination", "age stereotype"],
        },
        "disability": {
            "patterns": [
                r"\b(?:disabled|handicapped|crippled|retarded)\s+people\s+(?:are|can't|cannot)\b",
                r"\bwheelchair\s+bound\b",
                r"\bsuffer(?:s|ing)\s+from\s+(?:autism|disability)\b",
                r"\b(?:special|differently)\s+(?:needs|abled)\b",
            ],
            "keywords": ["ableist", "ableism", "disability discrimination"],
        },
        "sexual_orientation": {
            "patterns": [
                r"\b(?:gay|lesbian|homosexual|bisexual|transgender|lgbt)\s+people\s+(?:are|always|never)\b",
                r"\b(?:gay|lesbian|homosexual|bisexual|transgender|lgbt)\s+(?:agenda|lifestyle)\b",
                r"\bchoose\s+to\s+be\s+(?:gay|homosexual|transgender)\b",
                r"\b(?:real|normal|natural)\s+(?:man|woman|gender)\b",
                r"\b(?:he-she|it|tranny)\b",
            ],
            "keywords": [
                "homophobic",
                "transphobic",
                "lgbtq discrimination",
                "sexual orientation bias",
            ],
        },
        "political": {
            "patterns": [
                r"\b(?:liberals?|conservatives?|democrats?|republicans?)\s+(?:are|were)\s+(?:all\s+)?(?:\w+)",  # "Conservatives are all racists"
                r"\b(?:all|most|typical)\s+(?:liberals?|conservatives?|democrats?|republicans?)\s+(?:are|always|never)\b",
                r"\b(?:libtard|conservatard|trumptard)\b",
                r"\b(?:left|right)\s+wing\s+(?:nut|extremist|radical)\b",
                r"\b(?:fake|biased|lying)\s+(?:media|news)\b",
            ],
            "keywords": [],  # Removed broad keywords
        },
    }

    def __init__(self, config: BiasConfig):
        """Initialize bias detector.

        Args:
            config: Bias detection configuration
        """
        self.config = config
        self._fairlearn_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if optional ML libraries are available."""
        if self.config.use_fairlearn:
            try:
                import fairlearn  # noqa: F401
                import sklearn  # noqa: F401

                self._fairlearn_available = True
                logger.info("Fairlearn ML-based bias detection available")
            except ImportError as e:
                logger.warning(
                    "Fairlearn not available: %s. Install with: pip install fairlearn scikit-learn",
                    e,
                )
                self._fairlearn_available = False

    def is_available(self) -> bool:
        """Check if bias detector is available.

        Returns:
            bool: Always True (pattern-based detection always available)
        """
        return True  # Pattern-based detection is always available

    def detect(self, text: str) -> BiasDetectionResult:
        """Detect bias in text.

        Args:
            text: Text to analyze

        Returns:
            BiasDetectionResult: Detection results
        """
        if not self.config.enabled:
            return BiasDetectionResult(has_bias=False, original_text=text)

        try:
            # Perform pattern-based detection
            bias_scores = {}
            patterns_matched = {}

            for bias_type in self.config.bias_types:
                if bias_type not in self.BIAS_PATTERNS:
                    continue

                score, matched = self._check_bias_type(text, bias_type)
                bias_scores[bias_type] = score
                if matched:
                    patterns_matched[bias_type] = matched

            # Determine which biases exceed threshold
            detected_biases = [
                bias_type
                for bias_type, score in bias_scores.items()
                if score >= self.config.threshold
            ]

            has_bias = len(detected_biases) > 0
            max_score = max(bias_scores.values(), default=0.0)

            return BiasDetectionResult(
                has_bias=has_bias,
                bias_scores=bias_scores,
                max_score=max_score,
                detected_biases=detected_biases,
                patterns_matched=patterns_matched,
                original_text=text,
            )

        except Exception as e:
            logger.error("Error detecting bias: %s", e, exc_info=True)
            return BiasDetectionResult(has_bias=False, original_text=text)

    def _check_bias_type(self, text: str, bias_type: str) -> tuple[float, List[str]]:
        """Check for a specific type of bias.

        Args:
            text: Text to analyze
            bias_type: Type of bias to check

        Returns:
            tuple: (score, matched_patterns)
        """
        patterns_config = self.BIAS_PATTERNS.get(bias_type, {})
        patterns = patterns_config.get("patterns", [])
        keywords = patterns_config.get("keywords", [])

        matched = []
        text_lower = text.lower()

        # Handle leetspeak/character substitutions to detect obfuscated bias
        # @ -> a, $ -> s, 3 -> e, 1 -> i, 0 -> o
        deobfuscated_text = text
        substitutions = {"@": "a", "$": "s", "3": "e", "1": "i", "0": "o"}
        for char, replacement in substitutions.items():
            deobfuscated_text = deobfuscated_text.replace(char, replacement)

        # Normalize text by removing remaining special characters but keeping spaces
        # This helps patterns match even with # etc.
        normalized_text = re.sub(r"[^\w\s]", " ", deobfuscated_text)

        # Check regex patterns on original, deobfuscated, and normalized text
        for pattern in patterns:
            # Try on original text first
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if not matches:
                # Try on deobfuscated text
                matches = list(re.finditer(pattern, deobfuscated_text, re.IGNORECASE))
            if not matches:
                # Try on normalized text
                matches = list(re.finditer(pattern, normalized_text, re.IGNORECASE))
            for match in matches:
                matched.append(match.group())

        # Check keywords - but only as standalone words to avoid false positives
        for keyword in keywords:
            # Use word boundary matching to avoid substring matches
            keyword_pattern = r"\b" + re.escape(keyword) + r"\b"
            if re.search(keyword_pattern, text_lower):
                matched.append(keyword)

        # Calculate score based on matches
        if not matched:
            return 0.0, []

        # Score calculation:
        # - Base score of 0.3 for any match
        # - Additional 0.1 per unique match, capped at 0.9
        base_score = 0.3
        match_score = min(len(set(matched)) * 0.1, 0.6)
        total_score = min(base_score + match_score, 0.9)

        return total_score, matched

    def analyze_batch(self, texts: List[str]) -> List[BiasDetectionResult]:
        """Analyze multiple texts for bias.

        Args:
            texts: List of texts to analyze

        Returns:
            List[BiasDetectionResult]: Detection results for each text
        """
        results = []
        for text in texts:
            results.append(self.detect(text))
        return results

    def get_statistics(self, results: List[BiasDetectionResult]) -> Dict[str, Any]:
        """Get statistics from multiple detection results.

        Args:
            results: List of detection results

        Returns:
            Dict[str, Any]: Statistics including bias rates, type distribution
        """
        total_biased = sum(1 for r in results if r.has_bias)

        # Aggregate bias type counts
        bias_type_counts: Dict[str, int] = {}
        for result in results:
            for bias_type in result.detected_biases:
                bias_type_counts[bias_type] = bias_type_counts.get(bias_type, 0) + 1

        # Calculate average scores by type
        avg_scores: Dict[str, float] = {}
        for bias_type in self.config.bias_types:
            scores = [
                r.bias_scores.get(bias_type, 0.0) for r in results if r.bias_scores.get(bias_type)
            ]
            avg_scores[bias_type] = sum(scores) / len(scores) if scores else 0.0

        # Calculate max scores by type
        max_scores: Dict[str, float] = {}
        for bias_type in self.config.bias_types:
            scores = [
                r.bias_scores.get(bias_type, 0.0) for r in results if r.bias_scores.get(bias_type)
            ]
            max_scores[bias_type] = max(scores, default=0.0)

        return {
            "total_texts_analyzed": len(results),
            "biased_texts_count": total_biased,
            "bias_rate": total_biased / len(results) if results else 0.0,
            "bias_type_counts": bias_type_counts,
            "average_scores": avg_scores,
            "max_scores": max_scores,
            "most_common_bias": (
                max(bias_type_counts.items(), key=lambda x: x[1])[0] if bias_type_counts else None
            ),
        }

    def get_sensitive_attributes(self) -> Set[str]:
        """Get configured sensitive attributes for bias checking.

        Returns:
            Set[str]: Sensitive attributes from config
        """
        if self.config.sensitive_attributes:
            return set(self.config.sensitive_attributes)

        # Default sensitive attributes
        return {
            "gender",
            "race",
            "ethnicity",
            "age",
            "religion",
            "disability",
            "sexual_orientation",
        }
