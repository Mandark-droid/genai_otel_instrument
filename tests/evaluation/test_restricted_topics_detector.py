"""Tests for RestrictedTopicsDetector."""

import pytest

from genai_otel.evaluation.config import RestrictedTopicsConfig
from genai_otel.evaluation.restricted_topics_detector import (
    RestrictedTopicsDetector,
    RestrictedTopicsResult,
)


class TestRestrictedTopicsConfig:
    """Tests for RestrictedTopicsConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = RestrictedTopicsConfig()
        assert config.enabled is False
        assert config.threshold == 0.5
        assert config.block_on_detection is False
        assert config.restricted_topics is None

    def test_custom_config(self):
        """Test custom configuration."""
        config = RestrictedTopicsConfig(
            enabled=True,
            threshold=0.7,
            block_on_detection=True,
            restricted_topics=["medical_advice", "legal_advice"],
        )
        assert config.enabled is True
        assert config.threshold == 0.7
        assert config.block_on_detection is True
        assert len(config.restricted_topics) == 2


class TestRestrictedTopicsDetector:
    """Tests for RestrictedTopicsDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        config = RestrictedTopicsConfig(enabled=True)
        detector = RestrictedTopicsDetector(config)
        assert detector.config == config
        assert detector.is_available() is True

    def test_disabled_detection(self):
        """Test that detection returns no topic when disabled."""
        config = RestrictedTopicsConfig(enabled=False)
        detector = RestrictedTopicsDetector(config)

        result = detector.detect("Should I take this medication?")
        assert result.has_restricted_topic is False
        assert result.max_score == 0.0

    def test_medical_advice_detection(self):
        """Test medical advice detection."""
        config = RestrictedTopicsConfig(enabled=True, threshold=0.4)
        detector = RestrictedTopicsDetector(config)

        test_cases = [
            "Should I take aspirin for my headache?",
            "What medication should I use for this illness?",
            "Can you diagnose my symptoms?",
        ]

        for text in test_cases:
            result = detector.detect(text)
            assert result.has_restricted_topic is True, f"Failed to detect: {text}"
            assert "medical_advice" in result.detected_topics

    def test_legal_advice_detection(self):
        """Test legal advice detection."""
        config = RestrictedTopicsConfig(enabled=True, threshold=0.4)
        detector = RestrictedTopicsDetector(config)

        test_cases = [
            "Should I sue my employer?",
            "Is this legal or illegal?",
            "Can I file a lawsuit?",
        ]

        for text in test_cases:
            result = detector.detect(text)
            assert result.has_restricted_topic is True, f"Failed to detect: {text}"
            assert "legal_advice" in result.detected_topics

    def test_financial_advice_detection(self):
        """Test financial advice detection."""
        config = RestrictedTopicsConfig(enabled=True, threshold=0.4)
        detector = RestrictedTopicsDetector(config)

        test_cases = [
            "What stocks should I invest in?",
            "Should I buy Bitcoin now?",
            "Best investment strategies for 2024?",
        ]

        for text in test_cases:
            result = detector.detect(text)
            assert result.has_restricted_topic is True, f"Failed to detect: {text}"
            assert "financial_advice" in result.detected_topics

    def test_violence_detection(self):
        """Test violence content detection."""
        config = RestrictedTopicsConfig(enabled=True, threshold=0.4)
        detector = RestrictedTopicsDetector(config)

        test_cases = [
            "How to make a bomb?",
            "Ways to hurt someone",
        ]

        for text in test_cases:
            result = detector.detect(text)
            assert result.has_restricted_topic is True, f"Failed to detect: {text}"
            assert "violence" in result.detected_topics

    def test_safe_topics(self):
        """Test that safe topics are not flagged."""
        config = RestrictedTopicsConfig(enabled=True, threshold=0.4)
        detector = RestrictedTopicsDetector(config)

        safe_texts = [
            "What are the symptoms of the flu?",
            "Tell me about the legal system",
            "How does the stock market work?",
        ]

        for text in safe_texts:
            result = detector.detect(text)
            assert result.has_restricted_topic is False, f"False positive: {text}"

    def test_specific_topics_only(self):
        """Test detection with specific topics enabled."""
        config = RestrictedTopicsConfig(
            enabled=True,
            threshold=0.4,
            restricted_topics=["medical_advice", "legal_advice"],
        )
        detector = RestrictedTopicsDetector(config)

        # Medical should be detected
        result = detector.detect("Should I take this medication?")
        assert result.has_restricted_topic is True
        assert "medical_advice" in result.detected_topics

        # Financial should NOT be detected (not in restricted list)
        result = detector.detect("What stocks to buy?")
        # Financial advice is not in the restricted topics list
        assert "financial_advice" not in result.detected_topics

    def test_blocking_mode(self):
        """Test blocking mode."""
        config = RestrictedTopicsConfig(enabled=True, threshold=0.4, block_on_detection=True)
        detector = RestrictedTopicsDetector(config)

        result = detector.detect("Should I take this medication?")
        assert result.has_restricted_topic is True
        assert result.blocked is True

    def test_batch_processing(self):
        """Test batch processing."""
        config = RestrictedTopicsConfig(enabled=True, threshold=0.4)
        detector = RestrictedTopicsDetector(config)

        texts = [
            "Should I take aspirin?",
            "What's the weather today?",
            "Should I sue my employer?",
            "How to cook pasta?",
        ]

        results = detector.analyze_batch(texts)
        assert len(results) == 4
        assert results[0].has_restricted_topic is True
        assert results[1].has_restricted_topic is False
        assert results[2].has_restricted_topic is True
        assert results[3].has_restricted_topic is False

    def test_statistics_generation(self):
        """Test statistics generation."""
        config = RestrictedTopicsConfig(enabled=True, threshold=0.4)
        detector = RestrictedTopicsDetector(config)

        texts = [
            "Should I take medication?",
            "Normal question",
            "Legal advice needed",
            "Another normal question",
        ]

        results = detector.analyze_batch(texts)
        stats = detector.get_statistics(results)

        assert stats["total_texts_analyzed"] == 4
        assert stats["restricted_topics_count"] >= 2
        assert "topic_counts" in stats

    def test_get_available_topics(self):
        """Test get_available_topics method."""
        config = RestrictedTopicsConfig()
        detector = RestrictedTopicsDetector(config)

        topics = detector.get_available_topics()
        assert isinstance(topics, set)
        assert "medical_advice" in topics
        assert "legal_advice" in topics
        assert "financial_advice" in topics
        assert len(topics) >= 9

    def test_empty_text(self):
        """Test handling of empty text."""
        config = RestrictedTopicsConfig(enabled=True)
        detector = RestrictedTopicsDetector(config)

        result = detector.detect("")
        assert result.has_restricted_topic is False

    def test_result_dataclass(self):
        """Test RestrictedTopicsResult dataclass."""
        result = RestrictedTopicsResult(
            has_restricted_topic=True,
            detected_topics=["medical_advice"],
            topic_scores={"medical_advice": 0.8},
            max_score=0.8,
            patterns_matched={"medical_advice": ["medication"]},
            original_text="Should I take medication?",
            blocked=True,
        )

        assert result.has_restricted_topic is True
        assert "medical_advice" in result.detected_topics
        assert result.max_score == 0.8
        assert result.blocked is True
