"""Tests for BiasDetector."""

import pytest

from genai_otel.evaluation.bias_detector import BiasDetectionResult, BiasDetector
from genai_otel.evaluation.config import BiasConfig


class TestBiasConfig:
    """Tests for BiasConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = BiasConfig()
        assert config.enabled is False
        assert config.threshold == 0.5
        assert config.block_on_detection is False
        assert "gender" in config.bias_types
        assert "race" in config.bias_types
        assert len(config.bias_types) == 8

    def test_custom_config(self):
        """Test custom configuration."""
        config = BiasConfig(
            enabled=True,
            threshold=0.7,
            block_on_detection=True,
            bias_types=["gender", "race"],
        )
        assert config.enabled is True
        assert config.threshold == 0.7
        assert config.block_on_detection is True
        assert len(config.bias_types) == 2


class TestBiasDetector:
    """Tests for BiasDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        config = BiasConfig(enabled=True)
        detector = BiasDetector(config)
        assert detector.config == config
        assert detector.is_available() is True

    def test_disabled_detection(self):
        """Test that detection returns no bias when disabled."""
        config = BiasConfig(enabled=False)
        detector = BiasDetector(config)

        result = detector.detect("Women are always emotional")
        assert result.has_bias is False
        assert result.max_score == 0.0

    def test_gender_bias_detection(self):
        """Test gender bias detection."""
        config = BiasConfig(enabled=True, threshold=0.3)
        detector = BiasDetector(config)

        # Test positive cases
        test_cases = [
            "Women are always emotional and irrational",
            "Men are never good at multitasking",
            "She can't do that because of her gender",
            "Boys shouldn't cry like girls",
        ]

        for text in test_cases:
            result = detector.detect(text)
            assert result.has_bias is True, f"Failed to detect bias in: {text}"
            assert "gender" in result.detected_biases
            assert result.max_score >= 0.3

        # Test negative case
        result = detector.detect("The meeting will include both men and women")
        assert result.has_bias is False

    def test_race_bias_detection(self):
        """Test race bias detection."""
        config = BiasConfig(enabled=True, threshold=0.3)
        detector = BiasDetector(config)

        # Test positive cases
        test_cases = [
            "All Asian people are good at math",
            "Black people are always late",
            "White people tend to be privileged",
        ]

        for text in test_cases:
            result = detector.detect(text)
            assert result.has_bias is True, f"Failed to detect bias in: {text}"
            assert "race" in result.detected_biases
            assert result.max_score >= 0.3

        # Test negative case
        result = detector.detect("The team includes people of various racial backgrounds")
        assert result.has_bias is False

    def test_age_bias_detection(self):
        """Test age bias detection."""
        config = BiasConfig(enabled=True, threshold=0.3)
        detector = BiasDetector(config)

        # Test positive cases
        test_cases = [
            "Old people can't learn new technology",
            "Young people are always irresponsible",
            "Millennials are lazy and entitled",
            "You're too old to do this job",
        ]

        for text in test_cases:
            result = detector.detect(text)
            assert result.has_bias is True, f"Failed to detect bias in: {text}"
            assert "age" in result.detected_biases
            assert result.max_score >= 0.3

    def test_religion_bias_detection(self):
        """Test religion bias detection."""
        config = BiasConfig(enabled=True, threshold=0.3)
        detector = BiasDetector(config)

        # Test positive cases
        test_cases = [
            "All Muslims are terrorists",
            "Christians always try to convert people",
            "Jews are greedy with money",
        ]

        for text in test_cases:
            result = detector.detect(text)
            assert result.has_bias is True, f"Failed to detect bias in: {text}"
            assert "religion" in result.detected_biases
            assert result.max_score >= 0.3

    def test_disability_bias_detection(self):
        """Test disability bias detection."""
        config = BiasConfig(enabled=True, threshold=0.3)
        detector = BiasDetector(config)

        # Test positive cases
        test_cases = [
            "Disabled people can't work effectively",
            "He is wheelchair bound and helpless",
            "She suffers from autism",
        ]

        for text in test_cases:
            result = detector.detect(text)
            assert result.has_bias is True, f"Failed to detect bias in: {text}"
            assert "disability" in result.detected_biases
            assert result.max_score >= 0.3

    def test_sexual_orientation_bias_detection(self):
        """Test sexual orientation bias detection."""
        config = BiasConfig(enabled=True, threshold=0.3)
        detector = BiasDetector(config)

        # Test positive cases
        test_cases = [
            "The gay agenda is corrupting our children",
            "People choose to be homosexual",
            "Transgender people are confused",
        ]

        for text in test_cases:
            result = detector.detect(text)
            assert result.has_bias is True, f"Failed to detect bias in: {text}"
            assert "sexual_orientation" in result.detected_biases
            assert result.max_score >= 0.3

    def test_political_bias_detection(self):
        """Test political bias detection."""
        config = BiasConfig(enabled=True, threshold=0.3)
        detector = BiasDetector(config)

        # Test positive cases
        test_cases = [
            "All liberals are snowflakes",
            "Conservatives are all racists",
            "The media is fake news",
        ]

        for text in test_cases:
            result = detector.detect(text)
            assert result.has_bias is True, f"Failed to detect bias in: {text}"
            assert "political" in result.detected_biases
            assert result.max_score >= 0.3

    def test_ethnicity_bias_detection(self):
        """Test ethnicity bias detection."""
        config = BiasConfig(enabled=True, threshold=0.3)
        detector = BiasDetector(config)

        # Test positive cases
        test_cases = [
            "All Japanese people are workaholics",
            "Those foreigners are taking our jobs",
        ]

        for text in test_cases:
            result = detector.detect(text)
            assert result.has_bias is True, f"Failed to detect bias in: {text}"
            assert "ethnicity" in result.detected_biases
            assert result.max_score >= 0.3

    def test_threshold_filtering(self):
        """Test that threshold properly filters results."""
        config_low = BiasConfig(enabled=True, threshold=0.3)
        config_high = BiasConfig(enabled=True, threshold=0.8)

        detector_low = BiasDetector(config_low)
        detector_high = BiasDetector(config_high)

        text = "Women are always emotional"

        result_low = detector_low.detect(text)
        result_high = detector_high.detect(text)

        # Low threshold should detect bias
        assert result_low.has_bias is True

        # High threshold might not detect bias (depends on score calculation)
        # But both should have the same max_score
        assert result_low.max_score == result_high.max_score

    def test_multiple_bias_types(self):
        """Test detection of multiple bias types in same text."""
        config = BiasConfig(enabled=True, threshold=0.3)
        detector = BiasDetector(config)

        # Text with both gender and age bias
        text = "Women are too old to learn programming after 40"

        result = detector.detect(text)
        assert result.has_bias is True
        # Should detect at least one bias type
        assert len(result.detected_biases) >= 1
        assert result.max_score >= 0.3

    def test_patterns_matched(self):
        """Test that matched patterns are returned."""
        config = BiasConfig(enabled=True, threshold=0.3)
        detector = BiasDetector(config)

        text = "Women are always emotional"

        result = detector.detect(text)
        assert result.has_bias is True
        assert len(result.patterns_matched) > 0
        assert "gender" in result.patterns_matched

    def test_original_text_preservation(self):
        """Test that original text is preserved in result."""
        config = BiasConfig(enabled=True)
        detector = BiasDetector(config)

        text = "Women are always emotional"
        result = detector.detect(text)

        assert result.original_text == text

    def test_batch_processing(self):
        """Test batch processing of multiple texts."""
        config = BiasConfig(enabled=True, threshold=0.3)
        detector = BiasDetector(config)

        texts = [
            "Women are always emotional",
            "This is neutral text",
            "Old people can't use technology",
            "Normal conversation here",
        ]

        results = detector.analyze_batch(texts)

        assert len(results) == 4
        assert results[0].has_bias is True  # Gender bias
        assert results[1].has_bias is False  # Neutral
        assert results[2].has_bias is True  # Age bias
        assert results[3].has_bias is False  # Neutral

    def test_statistics_generation(self):
        """Test statistics generation from multiple results."""
        config = BiasConfig(enabled=True, threshold=0.3)
        detector = BiasDetector(config)

        texts = [
            "Women are always emotional",
            "Men never cry",
            "This is neutral text",
            "Old people can't learn",
        ]

        results = detector.analyze_batch(texts)
        stats = detector.get_statistics(results)

        assert stats["total_texts_analyzed"] == 4
        assert stats["biased_texts_count"] == 3
        assert stats["bias_rate"] == 0.75
        assert "gender" in stats["bias_type_counts"]
        assert "age" in stats["bias_type_counts"]
        assert stats["most_common_bias"] is not None

    def test_case_insensitive_detection(self):
        """Test that detection is case insensitive."""
        config = BiasConfig(enabled=True, threshold=0.3)
        detector = BiasDetector(config)

        texts = [
            "WOMEN ARE ALWAYS EMOTIONAL",
            "women are always emotional",
            "Women Are Always Emotional",
        ]

        for text in texts:
            result = detector.detect(text)
            assert result.has_bias is True, f"Failed for: {text}"

    def test_score_calculation(self):
        """Test that scores are calculated correctly."""
        config = BiasConfig(enabled=True, threshold=0.0)
        detector = BiasDetector(config)

        # Single match should give base score + small increment
        result_single = detector.detect("Women are always emotional")

        # Multiple matches should give higher score
        result_multiple = detector.detect(
            "Women are always emotional and never logical. "
            "Girls can't do math and shouldn't try."
        )

        assert result_single.max_score > 0
        assert result_multiple.max_score >= result_single.max_score

    def test_empty_text(self):
        """Test handling of empty text."""
        config = BiasConfig(enabled=True)
        detector = BiasDetector(config)

        result = detector.detect("")
        assert result.has_bias is False
        assert result.max_score == 0.0

    def test_whitespace_only_text(self):
        """Test handling of whitespace-only text."""
        config = BiasConfig(enabled=True)
        detector = BiasDetector(config)

        result = detector.detect("   \n\t  ")
        assert result.has_bias is False
        assert result.max_score == 0.0

    def test_specific_bias_types_only(self):
        """Test detection with specific bias types enabled."""
        config = BiasConfig(
            enabled=True,
            threshold=0.3,
            bias_types=["gender", "age"],  # Only these two
        )
        detector = BiasDetector(config)

        # Gender bias - should be detected
        result_gender = detector.detect("Women are always emotional")
        assert result_gender.has_bias is True
        assert "gender" in result_gender.detected_biases

        # Age bias - should be detected
        result_age = detector.detect("Old people can't learn")
        assert result_age.has_bias is True
        assert "age" in result_age.detected_biases

        # Race bias - should NOT be detected (not in enabled types)
        result_race = detector.detect("All Asian people are good at math")
        # The score might be calculated, but it won't be in detected_biases
        # because "race" is not in bias_types config
        assert "race" not in result_race.bias_scores

    def test_sensitive_attributes(self):
        """Test get_sensitive_attributes method."""
        config = BiasConfig()
        detector = BiasDetector(config)

        attributes = detector.get_sensitive_attributes()
        assert "gender" in attributes
        assert "race" in attributes
        assert "age" in attributes
        assert len(attributes) >= 7  # Default attributes

    def test_custom_sensitive_attributes(self):
        """Test custom sensitive attributes."""
        config = BiasConfig(sensitive_attributes=["gender", "age"])
        detector = BiasDetector(config)

        attributes = detector.get_sensitive_attributes()
        assert len(attributes) == 2
        assert "gender" in attributes
        assert "age" in attributes

    def test_keyword_matching(self):
        """Test that keywords are matched correctly."""
        config = BiasConfig(enabled=True, threshold=0.3)
        detector = BiasDetector(config)

        # Text containing bias keywords
        texts = [
            "That's a sexist comment",
            "Stop being racist",
            "This is ageist discrimination",
        ]

        for text in texts:
            result = detector.detect(text)
            assert result.has_bias is True, f"Failed to detect keyword in: {text}"
            assert result.max_score >= 0.3

    def test_bias_result_dataclass(self):
        """Test BiasDetectionResult dataclass."""
        result = BiasDetectionResult(
            has_bias=True,
            bias_scores={"gender": 0.8, "age": 0.5},
            max_score=0.8,
            detected_biases=["gender"],
            patterns_matched={"gender": ["women are always"]},
            original_text="Women are always emotional",
        )

        assert result.has_bias is True
        assert result.max_score == 0.8
        assert len(result.detected_biases) == 1
        assert "gender" in result.bias_scores
        assert result.original_text == "Women are always emotional"


class TestBiasDetectorEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_long_text(self):
        """Test handling of very long text."""
        config = BiasConfig(enabled=True, threshold=0.3)
        detector = BiasDetector(config)

        # Create a very long text with bias
        long_text = "Women are always emotional. " * 1000

        result = detector.detect(long_text)
        assert result.has_bias is True
        assert "gender" in result.detected_biases

    def test_special_characters(self):
        """Test handling of special characters."""
        config = BiasConfig(enabled=True, threshold=0.3)
        detector = BiasDetector(config)

        text = "Women @re #always $emotional! 😊"
        result = detector.detect(text)
        # Should still detect the bias pattern
        assert result.has_bias is True

    def test_unicode_text(self):
        """Test handling of unicode text."""
        config = BiasConfig(enabled=True, threshold=0.3)
        detector = BiasDetector(config)

        text = "Women are always emotional 女性情绪化"
        result = detector.detect(text)
        # Should detect the English bias pattern
        assert result.has_bias is True
        assert "gender" in result.detected_biases

    def test_mixed_case_keywords(self):
        """Test that mixed case keywords are matched."""
        config = BiasConfig(enabled=True, threshold=0.3)
        detector = BiasDetector(config)

        texts = [
            "That's SeXiSt",
            "Don't be RACIST",
            "This is AgeIsT",
        ]

        for text in texts:
            result = detector.detect(text)
            assert result.has_bias is True, f"Failed for: {text}"
