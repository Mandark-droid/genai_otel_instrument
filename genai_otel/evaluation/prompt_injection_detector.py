"""Prompt Injection Detection for GenAI applications.

This module provides prompt injection detection capabilities using pattern-based
approaches to identify attempts to manipulate or bypass LLM instructions.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import PromptInjectionConfig

logger = logging.getLogger(__name__)


@dataclass
class PromptInjectionResult:
    """Result of prompt injection detection.

    Attributes:
        is_injection: Whether prompt injection was detected above threshold
        injection_score: Overall injection risk score (0.0-1.0)
        injection_types: List of injection types detected
        patterns_matched: Specific patterns that triggered detection
        original_text: Original input text
        blocked: Whether the request was blocked
    """

    is_injection: bool
    injection_score: float = 0.0
    injection_types: List[str] = field(default_factory=list)
    patterns_matched: Dict[str, List[str]] = field(default_factory=dict)
    original_text: Optional[str] = None
    blocked: bool = False


class PromptInjectionDetector:
    """Prompt injection detector using pattern-based approaches.

    This detector identifies various types of prompt injection attacks:
    - Instruction override attempts
    - Role-playing attacks
    - Jailbreak attempts
    - Context switching
    - Encoding/obfuscation attacks
    - System prompt extraction

    All detection is pattern-based and does not require external dependencies.
    """

    # Pattern definitions for different injection types
    INJECTION_PATTERNS = {
        "instruction_override": {
            "patterns": [
                r"ignore\s+(?:all\s+)?(?:(?:previous|above|the|your)\s+)*(?:instructions?|prompts?|directions?|rules?)",
                r"disregard\s+(?:all\s+)?(?:(?:previous|above|the|your)\s+)*(?:instructions?|prompts?|directions?)",
                r"forget\s+(?:all\s+)?(?:(?:previous|above|the)\s+)*(?:instructions?|prompts?|directions?)",
                r"new\s+instructions?:\s*",
                r"instead,?\s+(?:you\s+)?(?:will|must|should)\s+",
                r"override\s+(?:your|the)\s+(?:instructions?|programming|rules)",
            ],
        },
        "role_playing": {
            "patterns": [
                r"(?:pretend|act|behave)\s+(?:like|as\s+if)\s+you\s+(?:are|were|have|can|could|must|should)",
                r"(?:pretend|act|behave)\s+(?:you\s+)?(?:are|were)",
                r"you\s+are\s+now\s+(?:a|an)\s+[\w\s]+",
                r"from\s+now\s+on,?\s+you\s+(?:are|will\s+be)",
                r"roleplay\s+as",
                r"imagine\s+you\s+are\s+(?:a|an)\s+\w+",
            ],
        },
        "jailbreak": {
            "patterns": [
                r"DAN\s+mode",
                r"developer\s+mode",
                r"evil\s+mode",
                r"opposite\s+mode",
                r"jailbreak",
                r"you\s+(?:now\s+)?have\s+no\s+(?:restrictions?|limitations?|ethics?|guidelines?)",  # More specific: "you have no restrictions"
                r"without\s+any\s+(?:restrictions?|limitations?|ethics?|guidelines?|filters?)",
                r"bypass\s+(?:your|the)\s+(?:safety|ethical|content)\s+(?:filters?|guidelines?)",
            ],
        },
        "context_switching": {
            "patterns": [
                r"---+\s*(?:end|stop|ignore)",
                r"```+\s*(?:end|stop)",
                r"</(?:system|instruction|prompt)>",
                r"\[(?:END|STOP|IGNORE)\s+(?:SYSTEM|INSTRUCTION|PROMPT)\]",
                r"<\|(?:end|stop)(?:of)?(?:system|instruction|prompt)\|>",
            ],
        },
        "system_extraction": {
            "patterns": [
                r"(?:show|reveal|display|print|output)\s+(?:me\s+)?(?:your|the)\s+(?:system\s+)?(?:prompt|instructions?)",
                r"what\s+(?:are|were)\s+your\s+(?:original\s+)?instructions?",
                r"repeat\s+(?:your|the)\s+(?:system\s+)?(?:prompt|instructions?)",
                r"tell\s+me\s+your\s+(?:system\s+)?(?:prompt|instructions?)",
            ],
        },
        "encoding_obfuscation": {
            "patterns": [
                r"(?:base64|rot13|hex|unicode)\s+(?:decode|encoded?|version)",
                r"\\x[0-9a-fA-F]{2}",  # Hex encoding
                r"&#x?[0-9a-fA-F]+;",  # HTML entities
                r"\{[0-9]+\}",  # Format string injection
            ],
        },
    }

    def __init__(self, config: PromptInjectionConfig):
        """Initialize prompt injection detector.

        Args:
            config: Prompt injection detection configuration
        """
        self.config = config

    def is_available(self) -> bool:
        """Check if prompt injection detector is available.

        Returns:
            bool: Always True (pattern-based detection always available)
        """
        return True

    def detect(self, text: str) -> PromptInjectionResult:
        """Detect prompt injection attempts in text.

        Args:
            text: Text to analyze (typically a user prompt)

        Returns:
            PromptInjectionResult: Detection results
        """
        if not self.config.enabled:
            return PromptInjectionResult(is_injection=False, original_text=text)

        try:
            # Perform pattern-based detection
            injection_scores = {}
            patterns_matched = {}

            for injection_type in self.INJECTION_PATTERNS:
                score, matched = self._check_injection_type(text, injection_type)
                injection_scores[injection_type] = score
                if matched:
                    patterns_matched[injection_type] = matched

            # Calculate overall injection score (max of all types)
            injection_score = max(injection_scores.values(), default=0.0)

            # Determine which injection types exceed threshold
            injection_types = [
                inj_type
                for inj_type, score in injection_scores.items()
                if score >= self.config.threshold
            ]

            is_injection = len(injection_types) > 0
            blocked = self.config.block_on_detection and is_injection

            return PromptInjectionResult(
                is_injection=is_injection,
                injection_score=injection_score,
                injection_types=injection_types,
                patterns_matched=patterns_matched,
                original_text=text,
                blocked=blocked,
            )

        except Exception as e:
            logger.error("Error detecting prompt injection: %s", e, exc_info=True)
            return PromptInjectionResult(is_injection=False, original_text=text)

    def _check_injection_type(self, text: str, injection_type: str) -> tuple[float, List[str]]:
        """Check for a specific type of prompt injection.

        Args:
            text: Text to analyze
            injection_type: Type of injection to check

        Returns:
            tuple: (score, matched_patterns)
        """
        patterns_config = self.INJECTION_PATTERNS.get(injection_type, {})
        patterns = patterns_config.get("patterns", [])

        matched = []
        text_lower = text.lower()

        # Check regex patterns
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                matched.append(match.group())

        # Calculate score based on matches
        if not matched:
            return 0.0, []

        # Score calculation:
        # - Base score of 0.5 for any match (injection attempts are high risk)
        # - Additional 0.1 per unique match, capped at 1.0
        base_score = 0.5
        match_score = min(len(set(matched)) * 0.1, 0.5)
        total_score = min(base_score + match_score, 1.0)

        return total_score, matched

    def analyze_batch(self, texts: List[str]) -> List[PromptInjectionResult]:
        """Analyze multiple texts for prompt injection.

        Args:
            texts: List of texts to analyze

        Returns:
            List[PromptInjectionResult]: Detection results for each text
        """
        results = []
        for text in texts:
            results.append(self.detect(text))
        return results

    def get_statistics(self, results: List[PromptInjectionResult]) -> Dict[str, Any]:
        """Get statistics from multiple detection results.

        Args:
            results: List of detection results

        Returns:
            Dict[str, Any]: Statistics including injection rates, type distribution
        """
        total_injections = sum(1 for r in results if r.is_injection)

        # Aggregate injection type counts
        injection_type_counts: Dict[str, int] = {}
        for result in results:
            for inj_type in result.injection_types:
                injection_type_counts[inj_type] = injection_type_counts.get(inj_type, 0) + 1

        # Calculate average scores
        avg_score = sum(r.injection_score for r in results) / len(results) if results else 0.0

        # Calculate max score
        max_score = max((r.injection_score for r in results), default=0.0)

        return {
            "total_prompts_analyzed": len(results),
            "injection_attempts_count": total_injections,
            "injection_rate": total_injections / len(results) if results else 0.0,
            "injection_type_counts": injection_type_counts,
            "average_score": avg_score,
            "max_score": max_score,
            "most_common_injection": (
                max(injection_type_counts.items(), key=lambda x: x[1])[0]
                if injection_type_counts
                else None
            ),
        }
