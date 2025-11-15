"""Tests for HallucinationDetector."""

import pytest

from genai_otel.evaluation.config import HallucinationConfig
from genai_otel.evaluation.hallucination_detector import HallucinationDetector, HallucinationResult


class TestHallucinationConfig:
    """Tests for HallucinationConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = HallucinationConfig()
        assert config.enabled is False
        assert config.threshold == 0.7
        assert config.check_citations is True
        assert config.check_hedging is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = HallucinationConfig(
            enabled=True,
            threshold=0.5,
            check_citations=False,
            check_hedging=False,
        )
        assert config.enabled is True
        assert config.threshold == 0.5
        assert config.check_citations is False
        assert config.check_hedging is False


class TestHallucinationDetector:
    """Tests for HallucinationDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        config = HallucinationConfig(enabled=True)
        detector = HallucinationDetector(config)
        assert detector.config == config
        assert detector.is_available() is True

    def test_disabled_detection(self):
        """Test that detection returns no hallucination when disabled."""
        config = HallucinationConfig(enabled=False)
        detector = HallucinationDetector(config)

        result = detector.detect("This is definitely true without any evidence.")
        assert result.has_hallucination is False
        assert result.hallucination_score == 0.0

    def test_well_cited_response(self):
        """Test well-cited response with low hallucination risk."""
        config = HallucinationConfig(enabled=True, threshold=0.5)
        detector = HallucinationDetector(config)

        text = "According to the 2020 census [1], the population was 328 million. Studies show [2][3] that this represents a 7.4% increase."
        result = detector.detect(text)

        assert result.citation_count >= 3
        assert result.hallucination_score < 0.5
        assert result.has_hallucination is False

    def test_unsupported_claims(self):
        """Test unsupported factual claims."""
        config = HallucinationConfig(enabled=True, threshold=0.5)
        detector = HallucinationDetector(config)

        test_cases = [
            "In 2025, the population will definitely be exactly 350 million people.",
            "The CEO announced on January 15, 2024 that profits reached $500 million.",
            "Studies show that 99.9% of experts agree on this specific point.",
        ]

        for text in test_cases:
            result = detector.detect(text)
            assert result.factual_claim_count > 0, f"Failed to detect claims in: {text}"
            # High claims with no citations should increase score
            if result.citation_count == 0:
                assert result.hallucination_score > 0, f"Expected score > 0 for: {text}"

    def test_hedge_words_detection(self):
        """Test hedge word detection."""
        config = HallucinationConfig(enabled=True, threshold=0.5)
        detector = HallucinationDetector(config)

        text = "It might be possibly around 300 million, perhaps more or less. It seems likely that maybe this could be approximately correct."
        result = detector.detect(text)

        assert result.hedge_words_count >= 5
        # Excessive hedging increases hallucination risk
        assert result.hallucination_score > 0

    def test_specific_dates_detection(self):
        """Test specific date claim detection."""
        config = HallucinationConfig(enabled=True, threshold=0.5)
        detector = HallucinationDetector(config)

        text = "On January 15, 2024, the event occurred. In 2025, another milestone was reached during March 2025."
        result = detector.detect(text)

        assert result.factual_claim_count >= 2

    def test_specific_numbers_detection(self):
        """Test specific number claim detection."""
        config = HallucinationConfig(enabled=True, threshold=0.5)
        detector = HallucinationDetector(config)

        text = "The population is 328 million. Revenue reached $45.7 billion, with 1,234 employees and 99.9% satisfaction rate."
        result = detector.detect(text)

        assert result.factual_claim_count >= 3

    def test_citation_formats(self):
        """Test various citation format detection."""
        config = HallucinationConfig(enabled=True, threshold=0.5)
        detector = HallucinationDetector(config)

        test_cases = [
            ("Numbered citations [1][2][3]", 3),
            ("Source citations (Smith, 2020) and (Jones et al., 2021)", 2),
            ("Inline citations: see reference¹ and footnote²", 2),
            ("According to source [A] and report [B]", 2),
        ]

        for text, expected_min_citations in test_cases:
            result = detector.detect(text)
            assert (
                result.citation_count >= expected_min_citations
            ), f"Failed to detect citations in: {text}"

    def test_absolute_statements(self):
        """Test absolute statement detection."""
        config = HallucinationConfig(enabled=True, threshold=0.5)
        detector = HallucinationDetector(config)

        text = "This is absolutely certain. It is definitely true. All experts always agree. This never fails."
        result = detector.detect(text)

        # Absolute statements without citations should increase score
        assert "absolute_statements" in result.hallucination_indicators

    def test_low_risk_response(self):
        """Test response with low hallucination risk."""
        config = HallucinationConfig(enabled=True, threshold=0.5)
        detector = HallucinationDetector(config)

        text = "Generally speaking, many experts believe that diverse approaches can be effective. This tends to vary by context."
        result = detector.detect(text)

        # Should have low score due to hedging and lack of specific claims
        assert result.has_hallucination is False

    def test_high_risk_response(self):
        """Test response with high hallucination risk."""
        config = HallucinationConfig(enabled=True, threshold=0.4)
        detector = HallucinationDetector(config)

        text = "On January 1, 2024, exactly 5.7 million people attended. The CEO definitely stated revenues hit $999 billion. All 10,000 employees unanimously agreed."
        result = detector.detect(text)

        # Many specific claims without citations
        assert result.factual_claim_count > 3
        assert result.citation_count == 0
        assert result.has_hallucination is True

    def test_mixed_quality_response(self):
        """Test response with mixed quality signals."""
        config = HallucinationConfig(enabled=True, threshold=0.5)
        detector = HallucinationDetector(config)

        text = "According to studies [1], approximately 60% of users may prefer this approach. The data suggests around 2.5 million participants, though exact figures vary [2]."
        result = detector.detect(text)

        # Has both citations and hedge words - should be moderate
        assert result.citation_count >= 2
        assert result.hedge_words_count >= 2

    def test_context_contradiction_detection(self):
        """Test context contradiction detection."""
        config = HallucinationConfig(enabled=True, threshold=0.5)
        detector = HallucinationDetector(config)

        text = "The population is 300 million. However, the population is 400 million. But actually it's 350 million."
        result = detector.detect(text, context=None)

        # Should detect contradictory statements
        assert "context_contradiction" in result.hallucination_indicators

    def test_batch_processing(self):
        """Test batch processing."""
        config = HallucinationConfig(enabled=True, threshold=0.5)
        detector = HallucinationDetector(config)

        texts = [
            "According to research [1][2], this is well-documented.",
            "It might possibly be true, perhaps.",
            "On January 1, 2024, exactly 5 million people attended without any source.",
            "This is a general statement without specifics.",
        ]

        results = detector.analyze_batch(texts)
        assert len(results) == 4
        # First should have low score (citations)
        assert results[0].hallucination_score < results[2].hallucination_score
        # Third should have high score (specific claims, no citations)
        assert results[2].factual_claim_count > 0

    def test_statistics_generation(self):
        """Test statistics generation."""
        config = HallucinationConfig(enabled=True, threshold=0.5)
        detector = HallucinationDetector(config)

        texts = [
            "Well cited research [1][2][3].",
            "Unsupported claim from January 2024.",
            "Another claim from 2025 with no source.",
            "General observation.",
        ]

        results = detector.analyze_batch(texts)
        stats = detector.get_statistics(results)

        assert stats["total_responses_analyzed"] == 4
        assert "hallucination_count" in stats
        assert "average_hallucination_score" in stats
        assert "average_citations" in stats
        assert "average_hedge_words" in stats

    def test_empty_text(self):
        """Test handling of empty text."""
        config = HallucinationConfig(enabled=True)
        detector = HallucinationDetector(config)

        result = detector.detect("")
        assert result.has_hallucination is False
        assert result.hallucination_score == 0.0

    def test_threshold_filtering(self):
        """Test that threshold properly filters results."""
        config_low = HallucinationConfig(enabled=True, threshold=0.3)
        config_high = HallucinationConfig(enabled=True, threshold=0.9)

        detector_low = HallucinationDetector(config_low)
        detector_high = HallucinationDetector(config_high)

        text = "In 2024, exactly 5 million people did something."

        result_low = detector_low.detect(text)
        result_high = detector_high.detect(text)

        # Scores should be the same
        assert result_low.hallucination_score == result_high.hallucination_score
        # But detection depends on threshold
        if result_low.hallucination_score >= 0.3:
            assert result_low.has_hallucination is True
        if result_high.hallucination_score < 0.9:
            assert result_high.has_hallucination is False

    def test_check_citations_disabled(self):
        """Test with citation checking disabled."""
        config = HallucinationConfig(enabled=True, threshold=0.5, check_citations=False)
        detector = HallucinationDetector(config)

        text = "According to research [1][2][3], this is documented."
        result = detector.detect(text)

        # Citations should not be counted
        assert result.citation_count == 0

    def test_check_hedging_disabled(self):
        """Test with hedging checking disabled."""
        config = HallucinationConfig(enabled=True, threshold=0.5, check_hedging=False)
        detector = HallucinationDetector(config)

        text = "It might possibly be perhaps around maybe approximately correct."
        result = detector.detect(text)

        # Hedge words should not be counted
        assert result.hedge_words_count == 0

    def test_result_dataclass(self):
        """Test HallucinationResult dataclass."""
        result = HallucinationResult(
            has_hallucination=True,
            hallucination_score=0.8,
            factual_claim_count=5,
            citation_count=1,
            hedge_words_count=2,
            hallucination_indicators=["absolute_statements", "unsupported_claims"],
            original_text="This is definitely true.",
        )

        assert result.has_hallucination is True
        assert result.hallucination_score == 0.8
        assert result.factual_claim_count == 5
        assert result.citation_count == 1
        assert result.hedge_words_count == 2
        assert len(result.hallucination_indicators) == 2

    def test_with_context(self):
        """Test detection with context provided."""
        config = HallucinationConfig(enabled=True, threshold=0.5)
        detector = HallucinationDetector(config)

        context = "The population is approximately 300 million."
        text = "The population is exactly 500 million."

        result = detector.detect(text, context=context)

        # Should detect potential contradiction
        assert "context_contradiction" in result.hallucination_indicators
