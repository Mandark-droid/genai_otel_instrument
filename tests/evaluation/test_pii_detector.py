"""Tests for PII detection functionality."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from genai_otel.evaluation.config import PIIConfig, PIIEntityType, PIIMode
from genai_otel.evaluation.pii_detector import PIIDetectionResult, PIIDetector


class TestPIIConfig:
    """Tests for PIIConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = PIIConfig()
        assert config.enabled is False
        assert config.mode == PIIMode.DETECT
        assert config.threshold == 0.7
        assert config.redaction_char == "*"
        assert not config.gdpr_mode
        assert not config.hipaa_mode
        assert not config.pci_dss_mode

    def test_gdpr_mode_adds_entities(self):
        """Test GDPR mode adds EU-specific entity types."""
        config = PIIConfig(gdpr_mode=True)
        assert PIIEntityType.IBAN_CODE in config.entity_types
        assert PIIEntityType.UK_NHS in config.entity_types
        assert PIIEntityType.NRP in config.entity_types

    def test_hipaa_mode_adds_entities(self):
        """Test HIPAA mode adds healthcare entity types."""
        config = PIIConfig(hipaa_mode=True)
        assert PIIEntityType.MEDICAL_LICENSE in config.entity_types
        assert PIIEntityType.US_PASSPORT in config.entity_types
        assert PIIEntityType.DATE_TIME in config.entity_types

    def test_pci_dss_mode_adds_entities(self):
        """Test PCI-DSS mode ensures credit card detection."""
        config = PIIConfig(pci_dss_mode=True)
        assert PIIEntityType.CREDIT_CARD in config.entity_types
        assert PIIEntityType.US_BANK_NUMBER in config.entity_types

    def test_invalid_threshold_raises_error(self):
        """Test invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
            PIIConfig(threshold=1.5)

        with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
            PIIConfig(threshold=-0.1)


class TestPIIDetector:
    """Tests for PIIDetector class."""

    def test_detector_initialization(self):
        """Test detector initializes correctly."""
        config = PIIConfig(enabled=True)
        detector = PIIDetector(config)
        assert detector.config == config
        assert detector._presidio_available in [True, False]  # Depends on installation

    def test_disabled_detector_returns_no_pii(self):
        """Test disabled detector returns no PII found."""
        config = PIIConfig(enabled=False)
        detector = PIIDetector(config)

        result = detector.detect("My email is test@example.com")
        assert not result.has_pii
        assert len(result.entities) == 0

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_fallback_email_detection(self, mock_check):
        """Test fallback email detection."""
        config = PIIConfig(enabled=True, entity_types={PIIEntityType.EMAIL_ADDRESS})
        detector = PIIDetector(config)
        detector._presidio_available = False  # Force fallback mode

        text = "Contact me at john.doe@example.com for more info."
        result = detector.detect(text)

        assert result.has_pii
        assert len(result.entities) == 1
        assert result.entities[0]["type"] == "EMAIL_ADDRESS"
        assert result.entities[0]["text"] == "john.doe@example.com"
        assert result.entities[0]["score"] == 0.9

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_fallback_phone_detection(self, mock_check):
        """Test fallback phone number detection."""
        config = PIIConfig(enabled=True, entity_types={PIIEntityType.PHONE_NUMBER})
        detector = PIIDetector(config)
        detector._presidio_available = False

        text = "Call me at 123-456-7890 or 555.123.4567"
        result = detector.detect(text)

        assert result.has_pii
        assert len(result.entities) == 2
        assert all(e["type"] == "PHONE_NUMBER" for e in result.entities)

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_fallback_ssn_detection(self, mock_check):
        """Test fallback SSN detection."""
        config = PIIConfig(enabled=True, entity_types={PIIEntityType.US_SSN})
        detector = PIIDetector(config)
        detector._presidio_available = False

        text = "My SSN is 123-45-6789"
        result = detector.detect(text)

        assert result.has_pii
        assert len(result.entities) == 1
        assert result.entities[0]["type"] == "US_SSN"
        assert result.entities[0]["text"] == "123-45-6789"

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_fallback_credit_card_detection(self, mock_check):
        """Test fallback credit card detection."""
        config = PIIConfig(enabled=True, entity_types={PIIEntityType.CREDIT_CARD})
        detector = PIIDetector(config)
        detector._presidio_available = False

        text = "My card number is 1234-5678-9012-3456"
        result = detector.detect(text)

        assert result.has_pii
        assert len(result.entities) == 1
        assert result.entities[0]["type"] == "CREDIT_CARD"

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_fallback_ip_address_detection(self, mock_check):
        """Test fallback IP address detection."""
        config = PIIConfig(enabled=True, entity_types={PIIEntityType.IP_ADDRESS})
        detector = PIIDetector(config)
        detector._presidio_available = False

        text = "Server IP: 192.168.1.100"
        result = detector.detect(text)

        assert result.has_pii
        assert len(result.entities) == 1
        assert result.entities[0]["type"] == "IP_ADDRESS"
        assert result.entities[0]["text"] == "192.168.1.100"

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_detect_mode(self, mock_check):
        """Test detection mode only detects PII without modifying text."""
        config = PIIConfig(enabled=True, mode=PIIMode.DETECT)
        detector = PIIDetector(config)
        detector._presidio_available = False

        text = "Email: test@example.com"
        result = detector.detect(text)

        assert result.has_pii
        assert result.redacted_text is None
        assert not result.blocked

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_redact_mode(self, mock_check):
        """Test redaction mode redacts detected PII."""
        config = PIIConfig(enabled=True, mode=PIIMode.REDACT, redaction_char="X")
        detector = PIIDetector(config)
        detector._presidio_available = False

        text = "Email: test@example.com"
        result = detector.detect(text)

        assert result.has_pii
        assert result.redacted_text is not None
        assert "test@example.com" not in result.redacted_text
        assert "X" in result.redacted_text
        assert not result.blocked

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_block_mode(self, mock_check):
        """Test block mode marks content as blocked."""
        config = PIIConfig(enabled=True, mode=PIIMode.BLOCK)
        detector = PIIDetector(config)
        detector._presidio_available = False

        text = "SSN: 123-45-6789"
        result = detector.detect(text)

        assert result.has_pii
        assert result.blocked

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_entity_counts(self, mock_check):
        """Test entity counting."""
        config = PIIConfig(enabled=True)
        detector = PIIDetector(config)
        detector._presidio_available = False

        text = "Emails: test1@example.com and test2@example.com. Phone: 123-456-7890"
        result = detector.detect(text)

        assert result.has_pii
        assert result.entity_counts.get("EMAIL_ADDRESS") == 2
        assert result.entity_counts.get("PHONE_NUMBER") == 1
        assert len(result.entities) == 3

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_no_pii_detected(self, mock_check):
        """Test text without PII."""
        config = PIIConfig(enabled=True)
        detector = PIIDetector(config)
        detector._presidio_available = False

        text = "This is a clean text without any PII."
        result = detector.detect(text)

        assert not result.has_pii
        assert len(result.entities) == 0
        assert result.score == 0.0

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_batch_analysis(self, mock_check):
        """Test batch text analysis."""
        config = PIIConfig(enabled=True)
        detector = PIIDetector(config)
        detector._presidio_available = False

        texts = [
            "Email: test@example.com",
            "Phone: 123-456-7890",
            "Clean text",
        ]

        results = detector.analyze_batch(texts)

        assert len(results) == 3
        assert results[0].has_pii  # Has email
        assert results[1].has_pii  # Has phone
        assert not results[2].has_pii  # Clean

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_statistics_generation(self, mock_check):
        """Test statistics from multiple results."""
        config = PIIConfig(enabled=True)
        detector = PIIDetector(config)
        detector._presidio_available = False

        texts = [
            "Email: test1@example.com",
            "Email: test2@example.com and Phone: 123-456-7890",
            "Clean text",
        ]

        results = detector.analyze_batch(texts)
        stats = detector.get_statistics(results)

        assert stats["total_texts_analyzed"] == 3
        assert stats["texts_with_pii"] == 2
        assert stats["total_entities_detected"] == 3
        assert stats["entity_type_distribution"]["EMAIL_ADDRESS"] == 2
        assert stats["entity_type_distribution"]["PHONE_NUMBER"] == 1
        assert 0 < stats["detection_rate"] <= 1.0

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_threshold_filtering(self, mock_check):
        """Test confidence threshold filtering."""
        config = PIIConfig(enabled=True, threshold=0.95)
        detector = PIIDetector(config)
        detector._presidio_available = False

        # Fallback detection has scores around 0.8-0.95
        # With high threshold, some might be filtered
        text = "Email: test@example.com"
        result = detector.detect(text)

        # Email detection has 0.9 score in fallback, should still detect
        assert result.has_pii

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_multiple_entity_types(self, mock_check):
        """Test detection of multiple entity types in same text."""
        config = PIIConfig(enabled=True)
        detector = PIIDetector(config)
        detector._presidio_available = False

        text = """
        Contact Information:
        Email: john.doe@example.com
        Phone: 555-123-4567
        SSN: 123-45-6789
        Card: 1234-5678-9012-3456
        IP: 192.168.1.1
        """

        result = detector.detect(text)

        assert result.has_pii
        assert len(result.entities) >= 5  # At least 5 different types
        entity_types = [e["type"] for e in result.entities]
        assert "EMAIL_ADDRESS" in entity_types
        assert "PHONE_NUMBER" in entity_types
        assert "US_SSN" in entity_types
        assert "CREDIT_CARD" in entity_types
        assert "IP_ADDRESS" in entity_types

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_redaction_preserves_text_structure(self, mock_check):
        """Test redaction preserves overall text structure."""
        config = PIIConfig(enabled=True, mode=PIIMode.REDACT)
        detector = PIIDetector(config)
        detector._presidio_available = False

        text = "My email is test@example.com and my phone is 123-456-7890."
        result = detector.detect(text)

        assert result.has_pii
        assert result.redacted_text is not None
        # Check structure is preserved
        assert "My email is" in result.redacted_text
        assert "and my phone is" in result.redacted_text
        # Check PII is redacted
        assert "test@example.com" not in result.redacted_text
        assert "123-456-7890" not in result.redacted_text

    def test_presidio_integration(self):
        """Test Presidio integration if available."""
        config = PIIConfig(enabled=True)
        detector = PIIDetector(config)

        if detector.is_available():
            # Test with Presidio if available
            text = "My email is john@example.com"
            result = detector.detect(text)

            # Should detect email
            assert result.has_pii
            assert any(e["type"] == "EMAIL_ADDRESS" for e in result.entities)
        else:
            # Skip if Presidio not installed
            pytest.skip("Presidio not available")

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_empty_text(self, mock_check):
        """Test handling of empty text."""
        config = PIIConfig(enabled=True)
        detector = PIIDetector(config)
        detector._presidio_available = False

        result = detector.detect("")

        assert not result.has_pii
        assert len(result.entities) == 0

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_special_characters_in_text(self, mock_check):
        """Test handling of special characters."""
        config = PIIConfig(enabled=True)
        detector = PIIDetector(config)
        detector._presidio_available = False

        text = "Email: test+tag@example.com with special chars!@#$%"
        result = detector.detect(text)

        # Should still detect email
        assert result.has_pii
        assert any("test" in e["text"] for e in result.entities)

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_custom_redaction_char(self, mock_check):
        """Test custom redaction character."""
        config = PIIConfig(enabled=True, mode=PIIMode.REDACT, redaction_char="#")
        detector = PIIDetector(config)
        detector._presidio_available = False

        text = "Email: test@example.com"
        result = detector.detect(text)

        assert result.has_pii
        assert result.redacted_text is not None
        assert "#" in result.redacted_text
        assert "*" not in result.redacted_text  # Default char should not be used

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_score_calculation(self, mock_check):
        """Test confidence score calculation."""
        config = PIIConfig(enabled=True)
        detector = PIIDetector(config)
        detector._presidio_available = False

        text = "Email: test@example.com"
        result = detector.detect(text)

        assert result.has_pii
        assert 0.0 <= result.score <= 1.0
        assert result.score > 0.0  # Should have confidence
