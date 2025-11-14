"""PII Detection using Microsoft Presidio.

This module provides PII (Personally Identifiable Information) detection and redaction
capabilities using Microsoft Presidio library.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import PIIConfig, PIIEntityType, PIIMode

logger = logging.getLogger(__name__)


@dataclass
class PIIDetectionResult:
    """Result of PII detection.

    Attributes:
        has_pii: Whether PII was detected
        entities: List of detected PII entities
        entity_counts: Count of entities by type
        redacted_text: Text with PII redacted (if redaction enabled)
        original_text: Original input text
        score: Overall PII detection confidence score
        blocked: Whether the text was blocked due to PII
    """

    has_pii: bool
    entities: List[Dict[str, Any]] = field(default_factory=list)
    entity_counts: Dict[str, int] = field(default_factory=dict)
    redacted_text: Optional[str] = None
    original_text: Optional[str] = None
    score: float = 0.0
    blocked: bool = False


class PIIDetector:
    """PII detector using Microsoft Presidio.

    This detector uses Presidio's analyzer and anonymizer to detect and redact
    personally identifiable information from text.

    Requirements:
        pip install presidio-analyzer presidio-anonymizer spacy
        python -m spacy download en_core_web_lg
    """

    def __init__(self, config: PIIConfig):
        """Initialize PII detector.

        Args:
            config: PII detection configuration
        """
        self.config = config
        self._analyzer = None
        self._anonymizer = None
        self._presidio_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if Presidio is available."""
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
            self._presidio_available = True
            logger.info("Presidio PII detection initialized successfully")
        except ImportError as e:
            logger.warning("Presidio not available, PII detection will be limited: %s", e)
            logger.info(
                "Install with: pip install presidio-analyzer presidio-anonymizer spacy && python -m spacy download en_core_web_lg"
            )
            self._presidio_available = False
        except Exception as e:
            logger.error("Failed to initialize Presidio: %s", e)
            self._presidio_available = False

    def is_available(self) -> bool:
        """Check if PII detector is available.

        Returns:
            bool: True if Presidio is available
        """
        return self._presidio_available

    def detect(self, text: str, language: str = "en") -> PIIDetectionResult:
        """Detect PII in text.

        Args:
            text: Text to analyze
            language: Language code (default: "en")

        Returns:
            PIIDetectionResult: Detection results
        """
        if not self.config.enabled:
            return PIIDetectionResult(has_pii=False, original_text=text)

        if not self._presidio_available:
            logger.warning("Presidio not available, using pattern-based detection")
            return self._fallback_detection(text)

        try:
            # Convert entity types to Presidio format
            entity_types = [entity.value for entity in self.config.entity_types]

            # Analyze text
            results = self._analyzer.analyze(
                text=text,
                language=language,
                entities=entity_types,
                score_threshold=self.config.threshold,
                allow_list=self.config.allow_list,
            )

            # Process results
            entities = []
            entity_counts: Dict[str, int] = {}

            for result in results:
                entity = {
                    "type": result.entity_type,
                    "start": result.start,
                    "end": result.end,
                    "score": result.score,
                    "text": text[result.start : result.end],
                }
                entities.append(entity)

                # Count by type
                entity_type = result.entity_type
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

            has_pii = len(entities) > 0

            # Calculate overall score (max confidence)
            score = max([e["score"] for e in entities], default=0.0)

            # Redact if mode is REDACT
            redacted_text = None
            if self.config.mode == PIIMode.REDACT and has_pii:
                redacted_text = self._redact_pii(text, results)

            # Block if mode is BLOCK
            blocked = self.config.mode == PIIMode.BLOCK and has_pii

            return PIIDetectionResult(
                has_pii=has_pii,
                entities=entities,
                entity_counts=entity_counts,
                redacted_text=redacted_text,
                original_text=text,
                score=score,
                blocked=blocked,
            )

        except Exception as e:
            logger.error("Error detecting PII: %s", e, exc_info=True)
            return PIIDetectionResult(has_pii=False, original_text=text)

    def _redact_pii(self, text: str, analyzer_results) -> str:
        """Redact PII from text.

        Args:
            text: Original text
            analyzer_results: Presidio analyzer results

        Returns:
            str: Text with PII redacted
        """
        try:
            from presidio_anonymizer.entities import OperatorConfig

            # Create anonymization config
            operators = {
                entity_type.value: OperatorConfig(
                    "replace", {"new_value": self.config.redaction_char * 8}
                )
                for entity_type in self.config.entity_types
            }

            # Anonymize
            anonymized = self._anonymizer.anonymize(
                text=text, analyzer_results=analyzer_results, operators=operators
            )

            return anonymized.text

        except Exception as e:
            logger.error("Error redacting PII: %s", e)
            return text

    def _fallback_detection(self, text: str) -> PIIDetectionResult:
        """Fallback pattern-based PII detection.

        This is used when Presidio is not available. It uses simple regex patterns
        for common PII types.

        Args:
            text: Text to analyze

        Returns:
            PIIDetectionResult: Detection results
        """
        entities = []
        entity_counts: Dict[str, int] = {}

        # Email pattern
        if PIIEntityType.EMAIL_ADDRESS in self.config.entity_types:
            email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            for match in re.finditer(email_pattern, text):
                entities.append(
                    {
                        "type": "EMAIL_ADDRESS",
                        "start": match.start(),
                        "end": match.end(),
                        "score": 0.9,
                        "text": match.group(),
                    }
                )
                entity_counts["EMAIL_ADDRESS"] = entity_counts.get("EMAIL_ADDRESS", 0) + 1

        # Phone pattern (US)
        if PIIEntityType.PHONE_NUMBER in self.config.entity_types:
            phone_pattern = r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
            for match in re.finditer(phone_pattern, text):
                entities.append(
                    {
                        "type": "PHONE_NUMBER",
                        "start": match.start(),
                        "end": match.end(),
                        "score": 0.8,
                        "text": match.group(),
                    }
                )
                entity_counts["PHONE_NUMBER"] = entity_counts.get("PHONE_NUMBER", 0) + 1

        # Credit card pattern
        if PIIEntityType.CREDIT_CARD in self.config.entity_types:
            cc_pattern = r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
            for match in re.finditer(cc_pattern, text):
                entities.append(
                    {
                        "type": "CREDIT_CARD",
                        "start": match.start(),
                        "end": match.end(),
                        "score": 0.85,
                        "text": match.group(),
                    }
                )
                entity_counts["CREDIT_CARD"] = entity_counts.get("CREDIT_CARD", 0) + 1

        # SSN pattern
        if PIIEntityType.US_SSN in self.config.entity_types:
            ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
            for match in re.finditer(ssn_pattern, text):
                entities.append(
                    {
                        "type": "US_SSN",
                        "start": match.start(),
                        "end": match.end(),
                        "score": 0.95,
                        "text": match.group(),
                    }
                )
                entity_counts["US_SSN"] = entity_counts.get("US_SSN", 0) + 1

        # IP Address pattern
        if PIIEntityType.IP_ADDRESS in self.config.entity_types:
            ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
            for match in re.finditer(ip_pattern, text):
                entities.append(
                    {
                        "type": "IP_ADDRESS",
                        "start": match.start(),
                        "end": match.end(),
                        "score": 0.9,
                        "text": match.group(),
                    }
                )
                entity_counts["IP_ADDRESS"] = entity_counts.get("IP_ADDRESS", 0) + 1

        has_pii = len(entities) > 0
        score = max([e["score"] for e in entities], default=0.0)

        # Simple redaction for fallback
        redacted_text = None
        if self.config.mode == PIIMode.REDACT and has_pii:
            redacted_text = text
            for entity in sorted(entities, key=lambda x: x["start"], reverse=True):
                start, end = entity["start"], entity["end"]
                replacement = self.config.redaction_char * (end - start)
                redacted_text = redacted_text[:start] + replacement + redacted_text[end:]

        blocked = self.config.mode == PIIMode.BLOCK and has_pii

        return PIIDetectionResult(
            has_pii=has_pii,
            entities=entities,
            entity_counts=entity_counts,
            redacted_text=redacted_text,
            original_text=text,
            score=score,
            blocked=blocked,
        )

    def analyze_batch(self, texts: List[str], language: str = "en") -> List[PIIDetectionResult]:
        """Analyze multiple texts for PII.

        Args:
            texts: List of texts to analyze
            language: Language code (default: "en")

        Returns:
            List[PIIDetectionResult]: Detection results for each text
        """
        results = []
        for text in texts:
            results.append(self.detect(text, language))
        return results

    def get_statistics(self, results: List[PIIDetectionResult]) -> Dict[str, Any]:
        """Get statistics from multiple detection results.

        Args:
            results: List of detection results

        Returns:
            Dict[str, Any]: Statistics including total PII count, entity type distribution
        """
        total_detections = sum(1 for r in results if r.has_pii)
        total_entities = sum(len(r.entities) for r in results)

        # Aggregate entity counts
        entity_type_counts: Dict[str, int] = {}
        for result in results:
            for entity_type, count in result.entity_counts.items():
                entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + count

        # Calculate average score
        scores = [r.score for r in results if r.has_pii]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "total_texts_analyzed": len(results),
            "texts_with_pii": total_detections,
            "total_entities_detected": total_entities,
            "entity_type_distribution": entity_type_counts,
            "average_confidence_score": avg_score,
            "detection_rate": total_detections / len(results) if results else 0.0,
        }
