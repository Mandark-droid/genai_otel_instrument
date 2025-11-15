"""Restricted Topics Detection for GenAI applications.

This module provides topic classification and detection capabilities to identify
and optionally block sensitive or inappropriate topics in prompts and responses.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .config import RestrictedTopicsConfig

logger = logging.getLogger(__name__)


@dataclass
class RestrictedTopicsResult:
    """Result of restricted topics detection.

    Attributes:
        has_restricted_topic: Whether a restricted topic was detected
        detected_topics: List of restricted topics found
        topic_scores: Confidence scores by topic
        max_score: Maximum confidence score across all topics
        patterns_matched: Specific patterns that triggered detection
        original_text: Original input text
        blocked: Whether the request was blocked
    """

    has_restricted_topic: bool
    detected_topics: List[str] = field(default_factory=list)
    topic_scores: Dict[str, float] = field(default_factory=dict)
    max_score: float = 0.0
    patterns_matched: Dict[str, List[str]] = field(default_factory=dict)
    original_text: Optional[str] = None
    blocked: bool = False


class RestrictedTopicsDetector:
    """Restricted topics detector using pattern-based classification.

    This detector identifies sensitive or inappropriate topics that may need
    to be monitored or blocked:
    - Medical advice
    - Legal advice
    - Financial advice
    - Violence and self-harm
    - Illegal activities
    - Adult content
    - Personal information requests
    - Political manipulation

    All detection is pattern-based and does not require external dependencies.
    """

    # Pattern definitions for different restricted topics
    TOPIC_PATTERNS = {
        "medical_advice": {
            "patterns": [
                r"should\s+I\s+(?:take|use|stop|avoid)\s+\w+",  # "should I take aspirin"
                r"(?:can|should)\s+I\s+(?:take|use)\s+(?:this|that|\w+)\s+for\s+(?:my|this|a)",  # "can I take aspirin for my headache"
                r"what\s+(?:drug|medicine|medication|treatment)\s+(?:should|can)\s+I\s+(?:take|use)",
                r"is\s+(?:it|this|\w+)\s+(?:safe|dangerous|ok|okay)\s+to\s+(?:take|use|consume)",
                r"(?:can|should)\s+you\s+(?:diagnose|recommend|prescribe)",
                r"(?:diagnose|treat)\s+my\s+(?:symptoms?|illness|condition|disease)",
            ],
            "keywords": [],  # Remove overly broad keywords
        },
        "legal_advice": {
            "patterns": [
                r"should\s+I\s+(?:sue|file\s+a\s+lawsuit|take\s+legal\s+action|hire\s+(?:a|an)\s+(?:lawyer|attorney))",
                r"(?:can|should)\s+I\s+(?:sue|take\s+legal\s+action|file\s+charges)",
                r"is\s+(?:it|this)\s+(?:legal|illegal)\s+(?:to|for|if|\?|or)",  # Added "or" and "?" for "legal or illegal?"
                r"what\s+(?:are\s+my\s+)?legal\s+(?:rights|options)",
                r"(?:can|should)\s+I\s+(?:file|take)\s+(?:a\s+)?(?:lawsuit|legal\s+action)",
                r"(?:need|want|seeking|looking\s+for)\s+legal\s+advice",  # "need legal advice"
                r"legal\s+advice\s+(?:needed|wanted|required)",  # "legal advice needed"
            ],
            "keywords": [],  # Removed broad keywords to avoid false positives
        },
        "financial_advice": {
            "patterns": [
                r"should\s+I\s+(?:invest|buy|sell|trade)\s+(?:in\s+)?(?:stocks?|crypto|bitcoin|forex)",
                r"(?:you|we)\s+should\s+(?:definitely|absolutely)?\s*(?:buy|sell|invest\s+in|trade)\s+(?:these\s+)?(?:stocks?|crypto|bitcoin)",  # "you should buy these stocks"
                r"(?:best|good|top)\s+(?:stocks?|investments?|crypto)\s+to\s+(?:buy|invest|purchase)",
                r"(?:best|good|top)\s+investment\s+strategies",  # "best investment strategies"
                r"(?:what|which)\s+(?:stocks?|crypto|investments?)\s+should\s+(?:I|we)\s+(?:buy|invest\s+in)",
                r"(?:should|can)\s+(?:I|we)\s+(?:invest\s+in|buy|sell|trade)",
                r"financial\s+advice\s+(?:on|for|about)",
            ],
            "keywords": [],  # Removed broad keywords to avoid false positives
        },
        "violence": {
            "patterns": [
                r"\b(?:kill|murder|assault|attack|weapon|bomb|explosive|terrorism)\b",
                r"how\s+to\s+(?:kill|murder|harm|hurt|attack)",
                r"(?:make|build|create)\s+(?:a\s+)?(?:bomb|weapon|explosive)",
                r"ways\s+to\s+(?:kill|murder|harm|hurt)",
            ],
            "keywords": ["violence", "harm", "weapon creation"],
        },
        "self_harm": {
            "patterns": [
                r"\b(?:suicide|self-harm|kill\s+myself|end\s+my\s+life)\b",
                r"how\s+to\s+(?:commit\s+suicide|kill\s+myself|end\s+my\s+life)",
                r"ways\s+to\s+(?:commit\s+suicide|kill\s+myself|die)",
                r"I\s+want\s+to\s+(?:die|kill\s+myself|end\s+it\s+all)",
            ],
            "keywords": ["suicide", "self-harm", "suicidal ideation"],
        },
        "illegal_activities": {
            "patterns": [
                r"how\s+to\s+(?:hack|crack|break\s+into|steal)",
                r"(?:make|create|produce)\s+(?:illegal\s+)?(?:drugs|narcotics)",
                r"how\s+to\s+(?:launder\s+money|evade\s+taxes|commit\s+fraud)",
                r"ways\s+to\s+(?:steal|rob|break\s+the\s+law)",
            ],
            "keywords": ["illegal activity", "crime", "hacking", "fraud"],
        },
        "adult_content": {
            "patterns": [
                r"\b(?:porn|pornography|xxx|nsfw|nude|naked)\b",
                r"(?:explicit|sexual)\s+(?:content|material|images?)",
                r"how\s+to\s+(?:find|access|watch)\s+(?:porn|adult\s+content)",
            ],
            "keywords": ["adult content", "pornography", "explicit material"],
        },
        "personal_information": {
            "patterns": [
                r"(?:give|provide|tell)\s+me\s+(?:your|the)\s+(?:password|credit\s+card|ssn)",
                r"what\s+is\s+(?:your|the)\s+(?:password|pin|code|key)",
                r"share\s+(?:your|the)\s+(?:login|credentials|password)",
            ],
            "keywords": ["password request", "credential theft", "phishing"],
        },
        "political_manipulation": {
            "patterns": [
                r"how\s+to\s+(?:manipulate|influence)\s+(?:voters|elections?|polls?)",
                r"create\s+(?:fake|misleading)\s+(?:news|information|propaganda)",
                r"spread\s+(?:misinformation|disinformation|propaganda)",
            ],
            "keywords": ["election manipulation", "propaganda", "misinformation"],
        },
    }

    def __init__(self, config: RestrictedTopicsConfig):
        """Initialize restricted topics detector.

        Args:
            config: Restricted topics detection configuration
        """
        self.config = config

    def is_available(self) -> bool:
        """Check if restricted topics detector is available.

        Returns:
            bool: Always True (pattern-based detection always available)
        """
        return True

    def detect(self, text: str) -> RestrictedTopicsResult:
        """Detect restricted topics in text.

        Args:
            text: Text to analyze

        Returns:
            RestrictedTopicsResult: Detection results
        """
        if not self.config.enabled:
            return RestrictedTopicsResult(has_restricted_topic=False, original_text=text)

        try:
            # Perform pattern-based detection
            topic_scores = {}
            patterns_matched = {}

            # Get topics to check (either configured topics or all)
            topics_to_check = (
                self.config.restricted_topics
                if self.config.restricted_topics
                else list(self.TOPIC_PATTERNS.keys())
            )

            for topic in topics_to_check:
                if topic not in self.TOPIC_PATTERNS:
                    logger.warning("Unknown topic: %s", topic)
                    continue

                score, matched = self._check_topic(text, topic)
                topic_scores[topic] = score
                if matched:
                    patterns_matched[topic] = matched

            # Determine which topics exceed threshold
            detected_topics = [
                topic for topic, score in topic_scores.items() if score >= self.config.threshold
            ]

            has_restricted_topic = len(detected_topics) > 0
            max_score = max(topic_scores.values(), default=0.0)
            blocked = self.config.block_on_detection and has_restricted_topic

            return RestrictedTopicsResult(
                has_restricted_topic=has_restricted_topic,
                detected_topics=detected_topics,
                topic_scores=topic_scores,
                max_score=max_score,
                patterns_matched=patterns_matched,
                original_text=text,
                blocked=blocked,
            )

        except Exception as e:
            logger.error("Error detecting restricted topics: %s", e, exc_info=True)
            return RestrictedTopicsResult(has_restricted_topic=False, original_text=text)

    def _check_topic(self, text: str, topic: str) -> tuple[float, List[str]]:
        """Check for a specific restricted topic.

        Args:
            text: Text to analyze
            topic: Topic to check

        Returns:
            tuple: (score, matched_patterns)
        """
        patterns_config = self.TOPIC_PATTERNS.get(topic, {})
        patterns = patterns_config.get("patterns", [])
        keywords = patterns_config.get("keywords", [])

        matched = []
        text_lower = text.lower()

        # Check regex patterns
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                matched.append(match.group())

        # Check keywords
        for keyword in keywords:
            if keyword in text_lower:
                matched.append(keyword)

        # Calculate score based on matches
        if not matched:
            return 0.0, []

        # Score calculation:
        # - Base score of 0.4 for any match
        # - Additional 0.1 per unique match, capped at 1.0
        base_score = 0.4
        match_score = min(len(set(matched)) * 0.1, 0.6)
        total_score = min(base_score + match_score, 1.0)

        return total_score, matched

    def analyze_batch(self, texts: List[str]) -> List[RestrictedTopicsResult]:
        """Analyze multiple texts for restricted topics.

        Args:
            texts: List of texts to analyze

        Returns:
            List[RestrictedTopicsResult]: Detection results for each text
        """
        results = []
        for text in texts:
            results.append(self.detect(text))
        return results

    def get_statistics(self, results: List[RestrictedTopicsResult]) -> Dict[str, Any]:
        """Get statistics from multiple detection results.

        Args:
            results: List of detection results

        Returns:
            Dict[str, Any]: Statistics including topic distribution
        """
        total_restricted = sum(1 for r in results if r.has_restricted_topic)

        # Aggregate topic counts
        topic_counts: Dict[str, int] = {}
        for result in results:
            for topic in result.detected_topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1

        # Calculate average scores
        avg_score = sum(r.max_score for r in results) / len(results) if results else 0.0

        # Calculate max score
        max_score = max((r.max_score for r in results), default=0.0)

        return {
            "total_texts_analyzed": len(results),
            "restricted_topics_count": total_restricted,
            "restricted_rate": total_restricted / len(results) if results else 0.0,
            "topic_counts": topic_counts,
            "average_score": avg_score,
            "max_score": max_score,
            "most_common_topic": (
                max(topic_counts.items(), key=lambda x: x[1])[0] if topic_counts else None
            ),
        }

    def get_available_topics(self) -> Set[str]:
        """Get list of available topic classifications.

        Returns:
            Set[str]: Available topic classifications
        """
        return set(self.TOPIC_PATTERNS.keys())
