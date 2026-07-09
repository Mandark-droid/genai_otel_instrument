"""Security-hardening tests for the evaluation subsystem.

Covers:
  - Perspective API egress gating (allow_external_egress)
  - Detoxify air-gapped runtime-download avoidance
  - Perspective developer-key never reaching logs
  - PII redacted-text bounding (content_max_length)
  - PII air-gapped offline env enforcement
"""

import os
from unittest.mock import Mock, patch

import pytest

from genai_otel.evaluation.config import PIIConfig, PIIMode, ToxicityConfig
from genai_otel.evaluation.pii_detector import PIIDetector
from genai_otel.evaluation.toxicity_detector import ToxicityDetector


class TestPerspectiveEgressGating:
    """The Perspective API must never be called when egress is disallowed."""

    @patch("genai_otel.evaluation.toxicity_detector.discovery")
    def test_perspective_not_initialized_when_egress_disabled(self, mock_discovery):
        config = ToxicityConfig(
            enabled=True,
            use_perspective_api=True,
            perspective_api_key="secret-key",
            use_local_model=False,
            allow_external_egress=False,
        )
        detector = ToxicityDetector(config)

        assert detector._perspective_available is False
        # The client must never be built (no egress).
        mock_discovery.build.assert_not_called()

    @patch("genai_otel.evaluation.toxicity_detector.discovery")
    def test_perspective_initialized_when_egress_allowed(self, mock_discovery):
        mock_discovery.build.return_value = Mock()
        config = ToxicityConfig(
            enabled=True,
            use_perspective_api=True,
            perspective_api_key="secret-key",
            use_local_model=False,
            allow_external_egress=True,
        )
        detector = ToxicityDetector(config)

        assert detector._perspective_available is True
        mock_discovery.build.assert_called_once()

    @patch("genai_otel.evaluation.toxicity_detector.ToxicityDetector._check_availability")
    def test_detect_never_routes_to_perspective_when_egress_disabled(self, mock_check):
        config = ToxicityConfig(
            enabled=True,
            use_perspective_api=True,
            perspective_api_key="secret-key",
            use_local_model=True,
            allow_external_egress=False,
        )
        detector = ToxicityDetector(config)
        # Force the (normally impossible) state where the client exists but
        # egress is off; detect() must still refuse to use it.
        detector._egress_allowed = False
        detector._perspective_available = True
        detector._detoxify_available = True
        detector._detect_with_perspective = Mock(
            side_effect=AssertionError("perspective must not be called")
        )
        detector._detect_with_detoxify = Mock(return_value="detoxify-result")

        result = detector.detect("some text")

        assert result == "detoxify-result"
        detector._detect_with_perspective.assert_not_called()

    def test_bank_profile_forces_perspective_off(self, monkeypatch):
        monkeypatch.setenv("GENAI_PROFILE", "bank")
        # Even asking for perspective should be neutralized by the profile, and
        # must not raise for a missing key because it is disabled first.
        config = ToxicityConfig(
            enabled=True,
            use_perspective_api=True,
            perspective_api_key=None,
        )
        assert config.use_perspective_api is False
        assert config.allow_external_egress is False
        assert config.air_gapped is True


class TestDetoxifyAirGap:
    """Detoxify must not download weights at runtime on air-gapped hosts."""

    @patch("genai_otel.evaluation.toxicity_detector.Detoxify")
    def test_airgapped_without_local_path_skips_download(self, mock_detoxify, monkeypatch):
        monkeypatch.delenv("GENAI_TOXICITY_DETOXIFY_PATH", raising=False)
        monkeypatch.delenv("GENAI_DETOXIFY_MODEL_PATH", raising=False)
        config = ToxicityConfig(
            enabled=True,
            use_perspective_api=False,
            use_local_model=True,
            air_gapped=True,
        )
        detector = ToxicityDetector(config)

        assert detector._detoxify_available is False
        # Must not attempt to construct the model (which would download).
        mock_detoxify.assert_not_called()

    @patch("genai_otel.evaluation.toxicity_detector.Detoxify")
    def test_airgapped_with_local_path_uses_checkpoint(self, mock_detoxify, monkeypatch):
        monkeypatch.setenv("GENAI_TOXICITY_DETOXIFY_PATH", "/models/detoxify.ckpt")
        config = ToxicityConfig(
            enabled=True,
            use_perspective_api=False,
            use_local_model=True,
            air_gapped=True,
        )
        detector = ToxicityDetector(config)

        assert detector._detoxify_available is True
        mock_detoxify.assert_called_once_with(checkpoint="/models/detoxify.ckpt")

    @patch("genai_otel.evaluation.toxicity_detector.Detoxify")
    def test_not_airgapped_uses_original(self, mock_detoxify, monkeypatch):
        monkeypatch.delenv("GENAI_TOXICITY_DETOXIFY_PATH", raising=False)
        monkeypatch.delenv("GENAI_DETOXIFY_MODEL_PATH", raising=False)
        config = ToxicityConfig(
            enabled=True,
            use_perspective_api=False,
            use_local_model=True,
            air_gapped=False,
        )
        detector = ToxicityDetector(config)

        assert detector._detoxify_available is True
        mock_detoxify.assert_called_once_with("original")


class TestPerspectiveKeyScrubbing:
    """The developer key must never appear in logs."""

    @patch("genai_otel.evaluation.toxicity_detector.ToxicityDetector._check_availability")
    def test_sanitize_removes_key_value(self, mock_check):
        config = ToxicityConfig(
            enabled=True,
            use_perspective_api=True,
            perspective_api_key="super-secret-key",
        )
        detector = ToxicityDetector(config)
        msg = "error at https://host/v1?key=super-secret-key&foo=bar"
        sanitized = detector._sanitize(msg)
        assert "super-secret-key" not in sanitized
        assert "***REDACTED***" in sanitized


class TestPIIRedactBounding:
    """Redacted PII text exported on spans must be bounded."""

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_redacted_text_is_capped(self, mock_check):
        config = PIIConfig(
            enabled=True,
            mode=PIIMode.REDACT,
            content_max_length=20,
        )
        detector = PIIDetector(config)
        detector._presidio_available = False  # use fallback path

        long_text = "My email is test@example.com. " + ("padding text " * 50)
        result = detector.detect(long_text)

        assert result.has_pii is True
        assert result.redacted_text is not None
        assert len(result.redacted_text) <= 20

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_redacted_text_unlimited_when_cap_zero(self, mock_check):
        config = PIIConfig(
            enabled=True,
            mode=PIIMode.REDACT,
            content_max_length=0,
        )
        detector = PIIDetector(config)
        detector._presidio_available = False

        long_text = "Email test@example.com " + ("word " * 100)
        result = detector.detect(long_text)

        assert result.redacted_text is not None
        # 0 => unlimited; the redacted copy keeps the full length.
        assert len(result.redacted_text) == len(long_text)


class TestPIIAirGap:
    """Air-gapped PII init forces the ML stack offline."""

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_airgapped_flag_read_from_config(self, mock_check):
        config = PIIConfig(enabled=True, air_gapped=True)
        detector = PIIDetector(config)
        assert detector._air_gapped is True

    @patch("presidio_analyzer.AnalyzerEngine", side_effect=Exception("boom"))
    def test_airgapped_sets_offline_env(self, mock_engine, monkeypatch):
        # Force the underlying HF/spaCy stack offline BEFORE any model resolve.
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        config = PIIConfig(enabled=True, air_gapped=True)
        # Runs the real _check_availability (presidio import patched to fail
        # fast, so no slow/networked model load happens).
        PIIDetector(config)
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
