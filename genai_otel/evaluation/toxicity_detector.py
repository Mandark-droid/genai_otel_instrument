"""Toxicity Detection using Google Perspective API and Detoxify.

This module provides toxicity detection capabilities using:
1. Google Perspective API (optional, requires API key)
2. Detoxify local model (fallback/alternative)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import ToxicityConfig

logger = logging.getLogger(__name__)

# Try to import optional dependencies at module level for testability
try:
    from detoxify import Detoxify
except ImportError:
    Detoxify = None

try:
    from googleapiclient import discovery
except ImportError:
    discovery = None


@dataclass
class ToxicityDetectionResult:
    """Result of toxicity detection.

    Attributes:
        is_toxic: Whether toxicity was detected above threshold
        scores: Toxicity scores by category
        max_score: Maximum toxicity score across all categories
        toxic_categories: List of categories that exceeded threshold
        original_text: Original input text
        blocked: Whether the text was blocked due to toxicity
    """

    is_toxic: bool
    scores: Dict[str, float] = field(default_factory=dict)
    max_score: float = 0.0
    toxic_categories: List[str] = field(default_factory=list)
    original_text: Optional[str] = None
    blocked: bool = False


class ToxicityDetector:
    """Toxicity detector using Perspective API and/or Detoxify.

    This detector can use:
    - Google Perspective API for production-grade detection
    - Detoxify local model for offline/fallback detection

    Requirements:
        Perspective API: pip install google-api-python-client
        Detoxify: pip install detoxify
    """

    def __init__(self, config: ToxicityConfig):
        """Initialize toxicity detector.

        Args:
            config: Toxicity detection configuration
        """
        self.config = config
        self._perspective_client = None
        self._detoxify_model = None
        self._perspective_available = False
        self._detoxify_available = False
        self._check_availability()

    def _check_availability(self):
        """Check which toxicity detection methods are available."""
        # Check Perspective API
        if self.config.use_perspective_api and self.config.perspective_api_key:
            try:
                if discovery is None:
                    raise ImportError("googleapiclient not installed")

                self._perspective_client = discovery.build(
                    "commentanalyzer",
                    "v1alpha1",
                    developerKey=self.config.perspective_api_key,
                    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                    static_discovery=False,
                )
                self._perspective_available = True
                logger.info("Google Perspective API initialized successfully")
            except ImportError as e:
                logger.warning(
                    "Perspective API not available: %s. Install with: pip install google-api-python-client",
                    e,
                )
                self._perspective_available = False
            except Exception as e:
                logger.error("Failed to initialize Perspective API: %s", e)
                self._perspective_available = False

        # Check Detoxify
        if self.config.use_local_model:
            try:
                if Detoxify is None:
                    raise ImportError("detoxify not installed")

                # Load the model (using "original" model by default)
                self._detoxify_model = Detoxify("original")
                self._detoxify_available = True
                logger.info("Detoxify model loaded successfully")
            except ImportError as e:
                logger.warning(
                    "Detoxify not available: %s. Install with: pip install detoxify",
                    e,
                )
                self._detoxify_available = False
            except Exception as e:
                logger.error("Failed to load Detoxify model: %s", e)
                self._detoxify_available = False

        if not self._perspective_available and not self._detoxify_available:
            logger.warning(
                "No toxicity detection method available. Install either:\n"
                "  - Perspective API: pip install google-api-python-client\n"
                "  - Detoxify: pip install detoxify"
            )

    def is_available(self) -> bool:
        """Check if toxicity detector is available.

        Returns:
            bool: True if at least one detection method is available
        """
        return self._perspective_available or self._detoxify_available

    def detect(self, text: str) -> ToxicityDetectionResult:
        """Detect toxicity in text.

        Args:
            text: Text to analyze

        Returns:
            ToxicityDetectionResult: Detection results
        """
        if not self.config.enabled:
            return ToxicityDetectionResult(is_toxic=False, original_text=text)

        if not self.is_available():
            logger.warning("No toxicity detection method available")
            return ToxicityDetectionResult(is_toxic=False, original_text=text)

        try:
            # Try Perspective API first if configured and available
            if self.config.use_perspective_api and self._perspective_available:
                return self._detect_with_perspective(text)
            # Fall back to Detoxify
            elif self._detoxify_available:
                return self._detect_with_detoxify(text)
            else:
                return ToxicityDetectionResult(is_toxic=False, original_text=text)

        except Exception as e:
            logger.error("Error detecting toxicity: %s", e, exc_info=True)
            return ToxicityDetectionResult(is_toxic=False, original_text=text)

    def _detect_with_perspective(self, text: str) -> ToxicityDetectionResult:
        """Detect toxicity using Google Perspective API.

        Args:
            text: Text to analyze

        Returns:
            ToxicityDetectionResult: Detection results
        """
        try:
            # Build the request
            analyze_request = {
                "comment": {"text": text},
                "requestedAttributes": {},
            }

            # Map our categories to Perspective API attributes
            category_mapping = {
                "toxicity": "TOXICITY",
                "severe_toxicity": "SEVERE_TOXICITY",
                "identity_attack": "IDENTITY_ATTACK",
                "insult": "INSULT",
                "profanity": "PROFANITY",
                "threat": "THREAT",
            }

            # Add requested categories
            for category in self.config.categories:
                if category in category_mapping:
                    analyze_request["requestedAttributes"][category_mapping[category]] = {}

            # Make the API call
            response = self._perspective_client.comments().analyze(body=analyze_request).execute()

            # Parse scores
            scores = {}
            for category in self.config.categories:
                if category in category_mapping:
                    api_attr = category_mapping[category]
                    if api_attr in response.get("attributeScores", {}):
                        score_data = response["attributeScores"][api_attr]
                        scores[category] = score_data["summaryScore"]["value"]

            # Determine if toxic
            max_score = max(scores.values(), default=0.0)
            toxic_categories = [
                cat for cat, score in scores.items() if score >= self.config.threshold
            ]
            is_toxic = len(toxic_categories) > 0

            # Check if should block
            blocked = self.config.block_on_detection and is_toxic

            return ToxicityDetectionResult(
                is_toxic=is_toxic,
                scores=scores,
                max_score=max_score,
                toxic_categories=toxic_categories,
                original_text=text,
                blocked=blocked,
            )

        except Exception as e:
            logger.error("Perspective API error: %s", e, exc_info=True)
            # Fall back to Detoxify if available
            if self._detoxify_available:
                logger.info("Falling back to Detoxify")
                return self._detect_with_detoxify(text)
            return ToxicityDetectionResult(is_toxic=False, original_text=text)

    def _detect_with_detoxify(self, text: str) -> ToxicityDetectionResult:
        """Detect toxicity using Detoxify local model.

        Args:
            text: Text to analyze

        Returns:
            ToxicityDetectionResult: Detection results
        """
        try:
            # Get predictions
            predictions = self._detoxify_model.predict(text)

            # Map Detoxify outputs to our categories
            # Detoxify outputs: toxicity, severe_toxicity, obscene, threat, insult, identity_attack
            category_mapping = {
                "toxicity": "toxicity",
                "severe_toxicity": "severe_toxicity",
                "identity_attack": "identity_attack",
                "insult": "insult",
                "profanity": "obscene",  # Map obscene to profanity
                "threat": "threat",
            }

            scores = {}
            for our_cat, detoxify_cat in category_mapping.items():
                if our_cat in self.config.categories and detoxify_cat in predictions:
                    scores[our_cat] = float(predictions[detoxify_cat])

            # Determine if toxic
            max_score = max(scores.values(), default=0.0)
            toxic_categories = [
                cat for cat, score in scores.items() if score >= self.config.threshold
            ]
            is_toxic = len(toxic_categories) > 0

            # Check if should block
            blocked = self.config.block_on_detection and is_toxic

            return ToxicityDetectionResult(
                is_toxic=is_toxic,
                scores=scores,
                max_score=max_score,
                toxic_categories=toxic_categories,
                original_text=text,
                blocked=blocked,
            )

        except Exception as e:
            logger.error("Detoxify error: %s", e, exc_info=True)
            return ToxicityDetectionResult(is_toxic=False, original_text=text)

    def analyze_batch(self, texts: List[str]) -> List[ToxicityDetectionResult]:
        """Analyze multiple texts for toxicity.

        Args:
            texts: List of texts to analyze

        Returns:
            List[ToxicityDetectionResult]: Detection results for each text
        """
        # Detoxify supports batch processing
        if self._detoxify_available and not self.config.use_perspective_api:
            try:
                return self._batch_detect_with_detoxify(texts)
            except Exception as e:
                logger.error("Batch detection error: %s", e)

        # Fall back to sequential processing
        results = []
        for text in texts:
            results.append(self.detect(text))
        return results

    def _batch_detect_with_detoxify(self, texts: List[str]) -> List[ToxicityDetectionResult]:
        """Batch detect toxicity using Detoxify.

        Args:
            texts: List of texts to analyze

        Returns:
            List[ToxicityDetectionResult]: Detection results
        """
        try:
            # Get batch predictions
            predictions = self._detoxify_model.predict(texts)

            results = []
            for i, text in enumerate(texts):
                # Extract scores for this text
                category_mapping = {
                    "toxicity": "toxicity",
                    "severe_toxicity": "severe_toxicity",
                    "identity_attack": "identity_attack",
                    "insult": "insult",
                    "profanity": "obscene",
                    "threat": "threat",
                }

                scores = {}
                for our_cat, detoxify_cat in category_mapping.items():
                    if our_cat in self.config.categories and detoxify_cat in predictions:
                        scores[our_cat] = float(predictions[detoxify_cat][i])

                # Determine if toxic
                max_score = max(scores.values(), default=0.0)
                toxic_categories = [
                    cat for cat, score in scores.items() if score >= self.config.threshold
                ]
                is_toxic = len(toxic_categories) > 0
                blocked = self.config.block_on_detection and is_toxic

                results.append(
                    ToxicityDetectionResult(
                        is_toxic=is_toxic,
                        scores=scores,
                        max_score=max_score,
                        toxic_categories=toxic_categories,
                        original_text=text,
                        blocked=blocked,
                    )
                )

            return results

        except Exception as e:
            logger.error("Batch Detoxify error: %s", e, exc_info=True)
            return [ToxicityDetectionResult(is_toxic=False, original_text=text) for text in texts]

    def get_statistics(self, results: List[ToxicityDetectionResult]) -> Dict[str, Any]:
        """Get statistics from multiple detection results.

        Args:
            results: List of detection results

        Returns:
            Dict[str, Any]: Statistics including toxicity rate, category distribution
        """
        total_toxic = sum(1 for r in results if r.is_toxic)

        # Aggregate category counts
        category_counts: Dict[str, int] = {}
        for result in results:
            for category in result.toxic_categories:
                category_counts[category] = category_counts.get(category, 0) + 1

        # Calculate average scores
        avg_scores: Dict[str, float] = {}
        for category in self.config.categories:
            scores = [r.scores.get(category, 0.0) for r in results if r.scores.get(category)]
            avg_scores[category] = sum(scores) / len(scores) if scores else 0.0

        # Calculate max scores seen
        max_scores: Dict[str, float] = {}
        for category in self.config.categories:
            scores = [r.scores.get(category, 0.0) for r in results if r.scores.get(category)]
            max_scores[category] = max(scores, default=0.0)

        return {
            "total_texts_analyzed": len(results),
            "toxic_texts_count": total_toxic,
            "toxicity_rate": total_toxic / len(results) if results else 0.0,
            "category_counts": category_counts,
            "average_scores": avg_scores,
            "max_scores": max_scores,
            "detection_method": (
                "perspective_api"
                if self._perspective_available and self.config.use_perspective_api
                else "detoxify"
            ),
        }
