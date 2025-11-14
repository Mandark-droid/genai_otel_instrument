"""Tests for toxicity detection functionality."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from genai_otel.evaluation.config import ToxicityConfig
from genai_otel.evaluation.toxicity_detector import ToxicityDetectionResult, ToxicityDetector


class TestToxicityConfig:
    """Tests for ToxicityConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = ToxicityConfig()
        assert config.enabled is False
        assert config.threshold == 0.7
        assert not config.use_perspective_api
        assert config.use_local_model is True
        assert not config.block_on_detection
        assert "toxicity" in config.categories
        assert "severe_toxicity" in config.categories

    def test_perspective_api_requires_key(self):
        """Test Perspective API requires API key."""
        with pytest.raises(ValueError, match="perspective_api_key is required"):
            ToxicityConfig(use_perspective_api=True, perspective_api_key=None)

    def test_invalid_threshold_raises_error(self):
        """Test invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
            ToxicityConfig(threshold=1.5)

        with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
            ToxicityConfig(threshold=-0.1)

    def test_custom_categories(self):
        """Test custom toxicity categories."""
        custom_cats = {"toxicity", "insult"}
        config = ToxicityConfig(categories=custom_cats)
        assert config.categories == custom_cats


class TestToxicityDetector:
    """Tests for ToxicityDetector class."""

    def test_detector_initialization(self):
        """Test detector initializes correctly."""
        config = ToxicityConfig(enabled=True, use_local_model=False)
        detector = ToxicityDetector(config)
        assert detector.config == config

    def test_disabled_detector_returns_not_toxic(self):
        """Test disabled detector returns not toxic."""
        config = ToxicityConfig(enabled=False)
        detector = ToxicityDetector(config)

        result = detector.detect("This is toxic content")
        assert not result.is_toxic
        assert len(result.scores) == 0

    @patch("genai_otel.evaluation.toxicity_detector.ToxicityDetector._check_availability")
    def test_detector_without_backends_warns(self, mock_check):
        """Test detector warns when no backends available."""
        config = ToxicityConfig(enabled=True, use_local_model=True)
        detector = ToxicityDetector(config)
        detector._perspective_available = False
        detector._detoxify_available = False

        assert not detector.is_available()

    @patch("genai_otel.evaluation.toxicity_detector.Detoxify")
    @patch("genai_otel.evaluation.toxicity_detector.ToxicityDetector._check_availability")
    def test_detoxify_detection(self, mock_check, mock_detoxify_class):
        """Test toxicity detection with Detoxify."""
        # Mock Detoxify model
        mock_model = Mock()
        mock_model.predict.return_value = {
            "toxicity": 0.85,
            "severe_toxicity": 0.2,
            "obscene": 0.1,
            "threat": 0.05,
            "insult": 0.75,
            "identity_attack": 0.1,
        }
        mock_detoxify_class.return_value = mock_model

        config = ToxicityConfig(enabled=True, use_local_model=True, threshold=0.7)
        detector = ToxicityDetector(config)
        detector._detoxify_model = mock_model
        detector._detoxify_available = True
        detector._perspective_available = False

        text = "You are stupid and worthless"
        result = detector.detect(text)

        assert result.is_toxic
        assert result.max_score == 0.85
        assert "toxicity" in result.toxic_categories
        assert "insult" in result.toxic_categories
        assert result.scores["toxicity"] == 0.85
        assert result.scores["insult"] == 0.75

    @patch("genai_otel.evaluation.toxicity_detector.Detoxify")
    @patch("genai_otel.evaluation.toxicity_detector.ToxicityDetector._check_availability")
    def test_detoxify_clean_text(self, mock_check, mock_detoxify_class):
        """Test Detoxify with clean text."""
        mock_model = Mock()
        mock_model.predict.return_value = {
            "toxicity": 0.1,
            "severe_toxicity": 0.05,
            "obscene": 0.02,
            "threat": 0.01,
            "insult": 0.03,
            "identity_attack": 0.02,
        }
        mock_detoxify_class.return_value = mock_model

        config = ToxicityConfig(enabled=True, use_local_model=True, threshold=0.7)
        detector = ToxicityDetector(config)
        detector._detoxify_model = mock_model
        detector._detoxify_available = True
        detector._perspective_available = False

        text = "This is a nice and friendly message"
        result = detector.detect(text)

        assert not result.is_toxic
        assert result.max_score == 0.1
        assert len(result.toxic_categories) == 0

    @patch("genai_otel.evaluation.toxicity_detector.discovery")
    @patch("genai_otel.evaluation.toxicity_detector.ToxicityDetector._check_availability")
    def test_perspective_api_detection(self, mock_check, mock_discovery):
        """Test toxicity detection with Perspective API."""
        # Mock Perspective API
        mock_client = Mock()
        mock_response = {
            "attributeScores": {
                "TOXICITY": {"summaryScore": {"value": 0.9}},
                "SEVERE_TOXICITY": {"summaryScore": {"value": 0.3}},
                "INSULT": {"summaryScore": {"value": 0.85}},
                "PROFANITY": {"summaryScore": {"value": 0.2}},
                "THREAT": {"summaryScore": {"value": 0.1}},
                "IDENTITY_ATTACK": {"summaryScore": {"value": 0.15}},
            }
        }
        mock_client.comments().analyze().execute.return_value = mock_response
        mock_discovery.build.return_value = mock_client

        config = ToxicityConfig(
            enabled=True,
            use_perspective_api=True,
            perspective_api_key="test-key",
            threshold=0.7,
        )
        detector = ToxicityDetector(config)
        detector._perspective_client = mock_client
        detector._perspective_available = True
        detector._detoxify_available = False

        text = "You are a terrible person"
        result = detector.detect(text)

        assert result.is_toxic
        assert result.max_score == 0.9
        assert "toxicity" in result.toxic_categories
        assert "insult" in result.toxic_categories
        assert result.scores["toxicity"] == 0.9

    @patch("genai_otel.evaluation.toxicity_detector.Detoxify")
    @patch("genai_otel.evaluation.toxicity_detector.ToxicityDetector._check_availability")
    def test_block_mode(self, mock_check, mock_detoxify_class):
        """Test blocking mode marks content as blocked."""
        mock_model = Mock()
        mock_model.predict.return_value = {
            "toxicity": 0.95,
            "severe_toxicity": 0.8,
            "obscene": 0.7,
            "threat": 0.6,
            "insult": 0.9,
            "identity_attack": 0.5,
        }
        mock_detoxify_class.return_value = mock_model

        config = ToxicityConfig(
            enabled=True,
            use_local_model=True,
            threshold=0.7,
            block_on_detection=True,
        )
        detector = ToxicityDetector(config)
        detector._detoxify_model = mock_model
        detector._detoxify_available = True
        detector._perspective_available = False

        text = "Extremely toxic content"
        result = detector.detect(text)

        assert result.is_toxic
        assert result.blocked

    @patch("genai_otel.evaluation.toxicity_detector.Detoxify")
    @patch("genai_otel.evaluation.toxicity_detector.ToxicityDetector._check_availability")
    def test_threshold_filtering(self, mock_check, mock_detoxify_class):
        """Test threshold filtering."""
        mock_model = Mock()
        mock_model.predict.return_value = {
            "toxicity": 0.65,
            "severe_toxicity": 0.2,
            "obscene": 0.3,
            "threat": 0.1,
            "insult": 0.55,
            "identity_attack": 0.15,
        }
        mock_detoxify_class.return_value = mock_model

        # With threshold 0.7, should not detect as toxic
        config = ToxicityConfig(enabled=True, use_local_model=True, threshold=0.7)
        detector = ToxicityDetector(config)
        detector._detoxify_model = mock_model
        detector._detoxify_available = True
        detector._perspective_available = False

        result = detector.detect("Borderline content")

        assert not result.is_toxic
        assert result.max_score == 0.65

        # With threshold 0.5, should detect as toxic
        config2 = ToxicityConfig(enabled=True, use_local_model=True, threshold=0.5)
        detector2 = ToxicityDetector(config2)
        detector2._detoxify_model = mock_model
        detector2._detoxify_available = True
        detector2._perspective_available = False

        result2 = detector2.detect("Borderline content")

        assert result2.is_toxic
        assert "toxicity" in result2.toxic_categories

    @patch("genai_otel.evaluation.toxicity_detector.Detoxify")
    @patch("genai_otel.evaluation.toxicity_detector.ToxicityDetector._check_availability")
    def test_batch_analysis(self, mock_check, mock_detoxify_class):
        """Test batch text analysis."""
        mock_model = Mock()
        # Return arrays for batch processing
        mock_model.predict.return_value = {
            "toxicity": [0.9, 0.2, 0.85],
            "severe_toxicity": [0.3, 0.1, 0.4],
            "obscene": [0.2, 0.05, 0.3],
            "threat": [0.1, 0.02, 0.2],
            "insult": [0.85, 0.1, 0.75],
            "identity_attack": [0.15, 0.05, 0.2],
        }
        mock_detoxify_class.return_value = mock_model

        config = ToxicityConfig(enabled=True, use_local_model=True, threshold=0.7)
        detector = ToxicityDetector(config)
        detector._detoxify_model = mock_model
        detector._detoxify_available = True
        detector._perspective_available = False

        texts = [
            "Toxic message 1",
            "Clean message",
            "Toxic message 2",
        ]

        results = detector.analyze_batch(texts)

        assert len(results) == 3
        assert results[0].is_toxic
        assert not results[1].is_toxic
        assert results[2].is_toxic

    @patch("genai_otel.evaluation.toxicity_detector.Detoxify")
    @patch("genai_otel.evaluation.toxicity_detector.ToxicityDetector._check_availability")
    def test_statistics_generation(self, mock_check, mock_detoxify_class):
        """Test statistics from multiple results."""
        mock_model = Mock()
        mock_model.predict.return_value = {
            "toxicity": [0.9, 0.2, 0.85, 0.1],
            "severe_toxicity": [0.3, 0.1, 0.4, 0.05],
            "obscene": [0.2, 0.05, 0.3, 0.02],
            "threat": [0.1, 0.02, 0.2, 0.01],
            "insult": [0.85, 0.1, 0.75, 0.05],
            "identity_attack": [0.15, 0.05, 0.2, 0.03],
        }
        mock_detoxify_class.return_value = mock_model

        config = ToxicityConfig(enabled=True, use_local_model=True, threshold=0.7)
        detector = ToxicityDetector(config)
        detector._detoxify_model = mock_model
        detector._detoxify_available = True
        detector._perspective_available = False

        texts = ["Text 1", "Text 2", "Text 3", "Text 4"]
        results = detector.analyze_batch(texts)
        stats = detector.get_statistics(results)

        assert stats["total_texts_analyzed"] == 4
        assert stats["toxic_texts_count"] == 2  # First and third texts
        assert 0 < stats["toxicity_rate"] <= 1.0
        assert "toxicity" in stats["average_scores"]
        assert stats["detection_method"] == "detoxify"

    @patch("genai_otel.evaluation.toxicity_detector.Detoxify")
    @patch("genai_otel.evaluation.toxicity_detector.ToxicityDetector._check_availability")
    def test_category_selection(self, mock_check, mock_detoxify_class):
        """Test selecting specific toxicity categories."""
        mock_model = Mock()
        mock_model.predict.return_value = {
            "toxicity": 0.5,
            "severe_toxicity": 0.9,
            "obscene": 0.3,
            "threat": 0.85,
            "insult": 0.4,
            "identity_attack": 0.2,
        }
        mock_detoxify_class.return_value = mock_model

        # Only check for severe_toxicity and threat
        config = ToxicityConfig(
            enabled=True,
            use_local_model=True,
            categories={"severe_toxicity", "threat"},
            threshold=0.7,
        )
        detector = ToxicityDetector(config)
        detector._detoxify_model = mock_model
        detector._detoxify_available = True
        detector._perspective_available = False

        result = detector.detect("Content")

        assert result.is_toxic
        assert "severe_toxicity" in result.toxic_categories
        assert "threat" in result.toxic_categories
        # Should only have scores for requested categories
        assert "severe_toxicity" in result.scores
        assert "threat" in result.scores

    @patch("genai_otel.evaluation.toxicity_detector.Detoxify")
    @patch("genai_otel.evaluation.toxicity_detector.ToxicityDetector._check_availability")
    def test_empty_text(self, mock_check, mock_detoxify_class):
        """Test handling of empty text."""
        mock_model = Mock()
        mock_model.predict.return_value = {
            "toxicity": 0.0,
            "severe_toxicity": 0.0,
            "obscene": 0.0,
            "threat": 0.0,
            "insult": 0.0,
            "identity_attack": 0.0,
        }
        mock_detoxify_class.return_value = mock_model

        config = ToxicityConfig(enabled=True, use_local_model=True)
        detector = ToxicityDetector(config)
        detector._detoxify_model = mock_model
        detector._detoxify_available = True
        detector._perspective_available = False

        result = detector.detect("")

        assert not result.is_toxic

    @patch("genai_otel.evaluation.toxicity_detector.Detoxify")
    @patch("genai_otel.evaluation.toxicity_detector.ToxicityDetector._check_availability")
    def test_error_handling(self, mock_check, mock_detoxify_class):
        """Test error handling in detection."""
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Model error")
        mock_detoxify_class.return_value = mock_model

        config = ToxicityConfig(enabled=True, use_local_model=True)
        detector = ToxicityDetector(config)
        detector._detoxify_model = mock_model
        detector._detoxify_available = True
        detector._perspective_available = False

        # Should not raise, should return not toxic
        result = detector.detect("Test text")

        assert not result.is_toxic

    @patch("genai_otel.evaluation.toxicity_detector.discovery")
    @patch("genai_otel.evaluation.toxicity_detector.Detoxify")
    @patch("genai_otel.evaluation.toxicity_detector.ToxicityDetector._check_availability")
    def test_fallback_from_perspective_to_detoxify(
        self, mock_check, mock_detoxify_class, mock_discovery
    ):
        """Test fallback from Perspective API to Detoxify on error."""
        # Mock Perspective API to fail
        mock_perspective = Mock()
        mock_perspective.comments().analyze().execute.side_effect = Exception("API error")
        mock_discovery.build.return_value = mock_perspective

        # Mock Detoxify to succeed
        mock_detoxify_model = Mock()
        mock_detoxify_model.predict.return_value = {
            "toxicity": 0.85,
            "severe_toxicity": 0.2,
            "obscene": 0.1,
            "threat": 0.05,
            "insult": 0.75,
            "identity_attack": 0.1,
        }
        mock_detoxify_class.return_value = mock_detoxify_model

        config = ToxicityConfig(
            enabled=True,
            use_perspective_api=True,
            perspective_api_key="test-key",
            use_local_model=True,
            threshold=0.7,
        )
        detector = ToxicityDetector(config)
        detector._perspective_client = mock_perspective
        detector._perspective_available = True
        detector._detoxify_model = mock_detoxify_model
        detector._detoxify_available = True

        result = detector.detect("Test text")

        # Should fall back to Detoxify and succeed
        assert result.is_toxic
        assert result.scores["toxicity"] == 0.85

    @patch("genai_otel.evaluation.toxicity_detector.Detoxify")
    @patch("genai_otel.evaluation.toxicity_detector.ToxicityDetector._check_availability")
    def test_multiple_toxic_categories(self, mock_check, mock_detoxify_class):
        """Test detection with multiple toxic categories."""
        mock_model = Mock()
        mock_model.predict.return_value = {
            "toxicity": 0.92,
            "severe_toxicity": 0.85,
            "obscene": 0.88,
            "threat": 0.78,
            "insult": 0.95,
            "identity_attack": 0.72,
        }
        mock_detoxify_class.return_value = mock_model

        config = ToxicityConfig(enabled=True, use_local_model=True, threshold=0.7)
        detector = ToxicityDetector(config)
        detector._detoxify_model = mock_model
        detector._detoxify_available = True
        detector._perspective_available = False

        result = detector.detect("Very toxic content")

        assert result.is_toxic
        assert len(result.toxic_categories) >= 4
        assert "toxicity" in result.toxic_categories
        assert "insult" in result.toxic_categories
        assert "severe_toxicity" in result.toxic_categories

    def test_toxicity_detection_result_dataclass(self):
        """Test ToxicityDetectionResult dataclass."""
        result = ToxicityDetectionResult(
            is_toxic=True,
            scores={"toxicity": 0.9, "insult": 0.85},
            max_score=0.9,
            toxic_categories=["toxicity", "insult"],
            original_text="Test text",
            blocked=True,
        )

        assert result.is_toxic
        assert result.max_score == 0.9
        assert len(result.toxic_categories) == 2
        assert result.blocked
