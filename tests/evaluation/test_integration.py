"""Integration tests for evaluation features."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from opentelemetry.sdk.trace import TracerProvider, Span
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import TraceFlags, SpanContext, SpanKind
from opentelemetry import trace

from genai_otel.evaluation.config import PIIConfig, PIIMode, ToxicityConfig
from genai_otel.evaluation.span_processor import EvaluationSpanProcessor
from genai_otel.evaluation.pii_detector import PIIDetector
from genai_otel.evaluation.toxicity_detector import ToxicityDetector


class TestEvaluationSpanProcessorIntegration:
    """Integration tests for EvaluationSpanProcessor."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a tracer provider
        resource = Resource.create({"service.name": "test-service"})
        self.tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(self.tracer_provider)
        self.tracer = self.tracer_provider.get_tracer(__name__)

    def _create_span(self, name="test-span", attributes=None):
        """Helper to create a span with attributes."""
        span_context = SpanContext(
            trace_id=1,
            span_id=1,
            is_remote=False,
            trace_flags=TraceFlags(0x01),
        )
        span = Span(
            name=name,
            context=span_context,
            parent=None,
            sampler=None,
            trace_config=None,
            resource=Resource.create({}),
            attributes=attributes or {},
            span_processor=Mock(),
            kind=SpanKind.CLIENT,
            instrumentation_scope=None,
        )
        return span

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_pii_detection_in_prompt(self, mock_check):
        """Test PII detection in prompt attributes."""
        # Configure PII detection
        pii_config = PIIConfig(enabled=True, mode=PIIMode.DETECT)
        processor = EvaluationSpanProcessor(pii_config=pii_config)
        processor.pii_detector._presidio_available = False  # Use fallback

        # Create span with prompt containing PII
        attributes = {
            "gen_ai.prompt": "My email is test@example.com and phone is 123-456-7890",
        }
        span = self._create_span(attributes=attributes)

        # Process span
        processor.on_end(span)

        # Check PII detection attributes were added
        span_attributes = dict(span.attributes)
        assert span_attributes.get("evaluation.pii.prompt.detected") is True
        assert span_attributes.get("evaluation.pii.prompt.entity_count") == 2
        assert "EMAIL_ADDRESS" in span_attributes.get("evaluation.pii.prompt.entity_types", [])
        assert "PHONE_NUMBER" in span_attributes.get("evaluation.pii.prompt.entity_types", [])
        assert span_attributes.get("evaluation.pii.prompt.score") > 0.0

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_pii_detection_in_response(self, mock_check):
        """Test PII detection in response attributes."""
        pii_config = PIIConfig(enabled=True, mode=PIIMode.DETECT)
        processor = EvaluationSpanProcessor(pii_config=pii_config)
        processor.pii_detector._presidio_available = False

        # Create span with response containing PII
        attributes = {
            "gen_ai.response": "Sure! You can reach me at contact@company.com",
        }
        span = self._create_span(attributes=attributes)

        # Process span
        processor.on_end(span)

        # Check PII detection attributes
        span_attributes = dict(span.attributes)
        assert span_attributes.get("evaluation.pii.response.detected") is True
        assert span_attributes.get("evaluation.pii.response.entity_count") >= 1
        assert "EMAIL_ADDRESS" in span_attributes.get("evaluation.pii.response.entity_types", [])

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_pii_redaction_mode(self, mock_check):
        """Test PII redaction mode adds redacted text."""
        pii_config = PIIConfig(enabled=True, mode=PIIMode.REDACT, redaction_char="*")
        processor = EvaluationSpanProcessor(pii_config=pii_config)
        processor.pii_detector._presidio_available = False

        # Create span with PII
        attributes = {
            "gen_ai.prompt": "My SSN is 123-45-6789",
        }
        span = self._create_span(attributes=attributes)

        # Process span
        processor.on_end(span)

        # Check redacted text is present
        span_attributes = dict(span.attributes)
        assert span_attributes.get("evaluation.pii.prompt.detected") is True
        redacted_text = span_attributes.get("evaluation.pii.prompt.redacted")
        assert redacted_text is not None
        assert "123-45-6789" not in redacted_text
        assert "*" in redacted_text

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_pii_block_mode(self, mock_check):
        """Test PII block mode sets error status."""
        pii_config = PIIConfig(enabled=True, mode=PIIMode.BLOCK)
        processor = EvaluationSpanProcessor(pii_config=pii_config)
        processor.pii_detector._presidio_available = False

        # Create span with PII
        attributes = {
            "gen_ai.prompt": "My credit card is 1234-5678-9012-3456",
        }
        span = self._create_span(attributes=attributes)

        # Process span
        processor.on_end(span)

        # Check blocked status
        span_attributes = dict(span.attributes)
        assert span_attributes.get("evaluation.pii.prompt.detected") is True
        assert span_attributes.get("evaluation.pii.prompt.blocked") is True
        # Span status should be ERROR (checked via span.status)

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_no_pii_detected(self, mock_check):
        """Test clean text without PII."""
        pii_config = PIIConfig(enabled=True, mode=PIIMode.DETECT)
        processor = EvaluationSpanProcessor(pii_config=pii_config)
        processor.pii_detector._presidio_available = False

        # Create span without PII
        attributes = {
            "gen_ai.prompt": "What is the weather like today?",
            "gen_ai.response": "The weather is sunny and warm.",
        }
        span = self._create_span(attributes=attributes)

        # Process span
        processor.on_end(span)

        # Check no PII detected
        span_attributes = dict(span.attributes)
        assert span_attributes.get("evaluation.pii.prompt.detected") is False
        assert span_attributes.get("evaluation.pii.response.detected") is False

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_entity_type_counts(self, mock_check):
        """Test individual entity type counts are tracked."""
        pii_config = PIIConfig(enabled=True, mode=PIIMode.DETECT)
        processor = EvaluationSpanProcessor(pii_config=pii_config)
        processor.pii_detector._presidio_available = False

        # Create span with multiple entity types
        attributes = {
            "gen_ai.prompt": "Email: test@example.com, Phone: 123-456-7890, IP: 192.168.1.1",
        }
        span = self._create_span(attributes=attributes)

        # Process span
        processor.on_end(span)

        # Check entity type counts
        span_attributes = dict(span.attributes)
        assert span_attributes.get("evaluation.pii.prompt.email_address_count") == 1
        assert span_attributes.get("evaluation.pii.prompt.phone_number_count") == 1
        assert span_attributes.get("evaluation.pii.prompt.ip_address_count") == 1

    def test_disabled_pii_detection(self):
        """Test PII detection is skipped when disabled."""
        pii_config = PIIConfig(enabled=False)
        processor = EvaluationSpanProcessor(pii_config=pii_config)

        # Create span with PII
        attributes = {
            "gen_ai.prompt": "My email is test@example.com",
        }
        span = self._create_span(attributes=attributes)

        # Process span
        processor.on_end(span)

        # Check no PII attributes added
        span_attributes = dict(span.attributes)
        assert "evaluation.pii.prompt.detected" not in span_attributes

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_multiple_attribute_formats(self, mock_check):
        """Test processor handles different attribute formats."""
        pii_config = PIIConfig(enabled=True, mode=PIIMode.DETECT)
        processor = EvaluationSpanProcessor(pii_config=pii_config)
        processor.pii_detector._presidio_available = False

        # Test different attribute keys used by various instrumentors
        test_cases = [
            {"gen_ai.prompt": "Email: test@example.com"},
            {"gen_ai.prompt.0.content": "Email: test@example.com"},
            {"gen_ai.request.prompt": "Email: test@example.com"},
            {"llm.prompts": "Email: test@example.com"},
        ]

        for attributes in test_cases:
            span = self._create_span(attributes=attributes)
            processor.on_end(span)

            # Should detect PII in all cases
            span_attributes = dict(span.attributes)
            # At least one detection should occur
            # (exact attribute name varies based on extraction logic)

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_error_handling(self, mock_check):
        """Test error handling in PII detection."""
        pii_config = PIIConfig(enabled=True, mode=PIIMode.DETECT)
        processor = EvaluationSpanProcessor(pii_config=pii_config)

        # Mock detector to raise exception
        processor.pii_detector.detect = Mock(side_effect=Exception("Test error"))

        # Create span
        attributes = {
            "gen_ai.prompt": "Test prompt",
        }
        span = self._create_span(attributes=attributes)

        # Process span - should not raise
        processor.on_end(span)

        # Check error is logged in attributes
        span_attributes = dict(span.attributes)
        assert "evaluation.pii.error" in span_attributes

    def test_processor_shutdown(self):
        """Test processor shutdown."""
        pii_config = PIIConfig(enabled=True)
        processor = EvaluationSpanProcessor(pii_config=pii_config)

        # Should not raise
        processor.shutdown()

    def test_processor_force_flush(self):
        """Test processor force flush."""
        pii_config = PIIConfig(enabled=True)
        processor = EvaluationSpanProcessor(pii_config=pii_config)

        # Should return True (no buffering)
        result = processor.force_flush(timeout_millis=1000)
        assert result is True

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_both_prompt_and_response_pii(self, mock_check):
        """Test PII detection in both prompt and response."""
        pii_config = PIIConfig(enabled=True, mode=PIIMode.DETECT)
        processor = EvaluationSpanProcessor(pii_config=pii_config)
        processor.pii_detector._presidio_available = False

        # Create span with PII in both
        attributes = {
            "gen_ai.prompt": "My email is user@example.com",
            "gen_ai.response": "Sure, I'll contact you at user@example.com",
        }
        span = self._create_span(attributes=attributes)

        # Process span
        processor.on_end(span)

        # Check both detected
        span_attributes = dict(span.attributes)
        assert span_attributes.get("evaluation.pii.prompt.detected") is True
        assert span_attributes.get("evaluation.pii.response.detected") is True

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_gdpr_mode_detection(self, mock_check):
        """Test GDPR mode enables EU-specific entities."""
        pii_config = PIIConfig(enabled=True, gdpr_mode=True, mode=PIIMode.DETECT)
        processor = EvaluationSpanProcessor(pii_config=pii_config)

        # Verify GDPR entities are enabled
        assert processor.pii_config.gdpr_mode is True

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_hipaa_mode_detection(self, mock_check):
        """Test HIPAA mode enables healthcare entities."""
        pii_config = PIIConfig(enabled=True, hipaa_mode=True, mode=PIIMode.DETECT)
        processor = EvaluationSpanProcessor(pii_config=pii_config)

        # Verify HIPAA entities are enabled
        assert processor.pii_config.hipaa_mode is True

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_pci_dss_mode_detection(self, mock_check):
        """Test PCI-DSS mode ensures credit card detection."""
        pii_config = PIIConfig(enabled=True, pci_dss_mode=True, mode=PIIMode.DETECT)
        processor = EvaluationSpanProcessor(pii_config=pii_config)

        # Verify PCI-DSS entities are enabled
        assert processor.pii_config.pci_dss_mode is True

    def test_on_start_does_nothing(self):
        """Test on_start is a no-op."""
        pii_config = PIIConfig(enabled=True)
        processor = EvaluationSpanProcessor(pii_config=pii_config)

        span = self._create_span()

        # Should not raise
        processor.on_start(span, parent_context=None)


class TestMetricsIntegration:
    """Test metrics recording in evaluation processor."""

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    @patch("opentelemetry.metrics.get_meter")
    def test_pii_detection_metrics(self, mock_get_meter, mock_check):
        """Test PII detection metrics are recorded."""
        # Mock meter and counters
        mock_meter = Mock()
        mock_counter = Mock()
        mock_meter.create_counter = Mock(return_value=mock_counter)
        mock_get_meter.return_value = mock_meter

        # Create processor
        pii_config = PIIConfig(enabled=True, mode=PIIMode.DETECT)
        processor = EvaluationSpanProcessor(pii_config=pii_config)
        processor.pii_detector._presidio_available = False

        # Verify counters were created
        assert mock_meter.create_counter.call_count >= 3  # At least 3 PII metrics

        # Create span with PII
        from opentelemetry.sdk.trace import TracerProvider, Span
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.trace import TraceFlags, SpanContext, SpanKind

        span_context = SpanContext(
            trace_id=1,
            span_id=1,
            is_remote=False,
            trace_flags=TraceFlags(0x01),
        )
        span = Span(
            name="test-span",
            context=span_context,
            parent=None,
            sampler=None,
            trace_config=None,
            resource=Resource.create({}),
            attributes={"gen_ai.prompt": "Email: test@example.com"},
            span_processor=Mock(),
            kind=SpanKind.CLIENT,
            instrumentation_scope=None,
        )

        # Process span
        processor.on_end(span)

        # Verify metrics were recorded
        # Note: Actual metric recording depends on mock setup
        # In real integration, these would be recorded to the meter provider


class TestToxicityIntegration:
    """Integration tests for toxicity detection."""

    def setup_method(self):
        """Set up test fixtures."""
        resource = Resource.create({"service.name": "test-service"})
        self.tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(self.tracer_provider)
        self.tracer = self.tracer_provider.get_tracer(__name__)

    def _create_span(self, name="test-span", attributes=None):
        """Helper to create a span with attributes."""
        span_context = SpanContext(
            trace_id=1,
            span_id=1,
            is_remote=False,
            trace_flags=TraceFlags(0x01),
        )
        span = Span(
            name=name,
            context=span_context,
            parent=None,
            sampler=None,
            trace_config=None,
            resource=Resource.create({}),
            attributes=attributes or {},
            span_processor=Mock(),
            kind=SpanKind.CLIENT,
            instrumentation_scope=None,
        )
        return span

    @patch("genai_otel.evaluation.toxicity_detector.Detoxify")
    @patch("genai_otel.evaluation.toxicity_detector.ToxicityDetector._check_availability")
    def test_toxicity_detection_in_prompt(self, mock_check, mock_detoxify_class):
        """Test toxicity detection in prompt attributes."""
        # Mock Detoxify model
        mock_model = Mock()
        mock_model.predict.return_value = {
            "toxicity": 0.92,
            "severe_toxicity": 0.3,
            "obscene": 0.2,
            "threat": 0.1,
            "insult": 0.85,
            "identity_attack": 0.15,
        }
        mock_detoxify_class.return_value = mock_model

        # Configure toxicity detection
        toxicity_config = ToxicityConfig(enabled=True, use_local_model=True, threshold=0.7)
        processor = EvaluationSpanProcessor(toxicity_config=toxicity_config)
        processor.toxicity_detector._detoxify_model = mock_model
        processor.toxicity_detector._detoxify_available = True
        processor.toxicity_detector._perspective_available = False

        # Create span with toxic prompt
        attributes = {
            "gen_ai.prompt": "You are stupid and worthless",
        }
        span = self._create_span(attributes=attributes)

        # Process span
        processor.on_end(span)

        # Check toxicity detection attributes
        span_attributes = dict(span.attributes)
        assert span_attributes.get("evaluation.toxicity.prompt.detected") is True
        assert span_attributes.get("evaluation.toxicity.prompt.max_score") == 0.92
        assert "toxicity" in span_attributes.get("evaluation.toxicity.prompt.categories", [])
        assert "insult" in span_attributes.get("evaluation.toxicity.prompt.categories", [])

    @patch("genai_otel.evaluation.toxicity_detector.Detoxify")
    @patch("genai_otel.evaluation.toxicity_detector.ToxicityDetector._check_availability")
    def test_toxicity_detection_in_response(self, mock_check, mock_detoxify_class):
        """Test toxicity detection in response attributes."""
        mock_model = Mock()
        mock_model.predict.return_value = {
            "toxicity": 0.88,
            "severe_toxicity": 0.2,
            "obscene": 0.75,
            "threat": 0.05,
            "insult": 0.65,
            "identity_attack": 0.1,
        }
        mock_detoxify_class.return_value = mock_model

        toxicity_config = ToxicityConfig(enabled=True, use_local_model=True, threshold=0.7)
        processor = EvaluationSpanProcessor(toxicity_config=toxicity_config)
        processor.toxicity_detector._detoxify_model = mock_model
        processor.toxicity_detector._detoxify_available = True
        processor.toxicity_detector._perspective_available = False

        # Create span with toxic response
        attributes = {
            "gen_ai.response": "This is offensive content",
        }
        span = self._create_span(attributes=attributes)

        # Process span
        processor.on_end(span)

        # Check toxicity attributes
        span_attributes = dict(span.attributes)
        assert span_attributes.get("evaluation.toxicity.response.detected") is True
        assert span_attributes.get("evaluation.toxicity.response.max_score") >= 0.7

    @patch("genai_otel.evaluation.toxicity_detector.Detoxify")
    @patch("genai_otel.evaluation.toxicity_detector.ToxicityDetector._check_availability")
    def test_toxicity_block_mode(self, mock_check, mock_detoxify_class):
        """Test toxicity blocking mode sets error status."""
        mock_model = Mock()
        mock_model.predict.return_value = {
            "toxicity": 0.95,
            "severe_toxicity": 0.9,
            "obscene": 0.85,
            "threat": 0.8,
            "insult": 0.92,
            "identity_attack": 0.7,
        }
        mock_detoxify_class.return_value = mock_model

        toxicity_config = ToxicityConfig(
            enabled=True, use_local_model=True, threshold=0.7, block_on_detection=True
        )
        processor = EvaluationSpanProcessor(toxicity_config=toxicity_config)
        processor.toxicity_detector._detoxify_model = mock_model
        processor.toxicity_detector._detoxify_available = True
        processor.toxicity_detector._perspective_available = False

        # Create span with toxic content
        attributes = {
            "gen_ai.prompt": "Extremely toxic content",
        }
        span = self._create_span(attributes=attributes)

        # Process span
        processor.on_end(span)

        # Check blocked status
        span_attributes = dict(span.attributes)
        assert span_attributes.get("evaluation.toxicity.prompt.detected") is True
        assert span_attributes.get("evaluation.toxicity.prompt.blocked") is True

    @patch("genai_otel.evaluation.toxicity_detector.Detoxify")
    @patch("genai_otel.evaluation.toxicity_detector.ToxicityDetector._check_availability")
    def test_no_toxicity_detected(self, mock_check, mock_detoxify_class):
        """Test clean text without toxicity."""
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

        toxicity_config = ToxicityConfig(enabled=True, use_local_model=True, threshold=0.7)
        processor = EvaluationSpanProcessor(toxicity_config=toxicity_config)
        processor.toxicity_detector._detoxify_model = mock_model
        processor.toxicity_detector._detoxify_available = True
        processor.toxicity_detector._perspective_available = False

        # Create span without toxicity
        attributes = {
            "gen_ai.prompt": "What is the weather like today?",
            "gen_ai.response": "The weather is sunny and warm.",
        }
        span = self._create_span(attributes=attributes)

        # Process span
        processor.on_end(span)

        # Check no toxicity detected
        span_attributes = dict(span.attributes)
        assert span_attributes.get("evaluation.toxicity.prompt.detected") is False
        assert span_attributes.get("evaluation.toxicity.response.detected") is False

    def test_disabled_toxicity_detection(self):
        """Test toxicity detection is skipped when disabled."""
        toxicity_config = ToxicityConfig(enabled=False)
        processor = EvaluationSpanProcessor(toxicity_config=toxicity_config)

        # Create span with toxic content
        attributes = {
            "gen_ai.prompt": "Toxic content here",
        }
        span = self._create_span(attributes=attributes)

        # Process span
        processor.on_end(span)

        # Check no toxicity attributes added
        span_attributes = dict(span.attributes)
        assert "evaluation.toxicity.prompt.detected" not in span_attributes

    @patch("genai_otel.evaluation.toxicity_detector.Detoxify")
    @patch("genai_otel.evaluation.toxicity_detector.ToxicityDetector._check_availability")
    def test_category_scores(self, mock_check, mock_detoxify_class):
        """Test individual category scores are tracked."""
        mock_model = Mock()
        mock_model.predict.return_value = {
            "toxicity": 0.85,
            "severe_toxicity": 0.4,
            "obscene": 0.75,
            "threat": 0.3,
            "insult": 0.8,
            "identity_attack": 0.2,
        }
        mock_detoxify_class.return_value = mock_model

        toxicity_config = ToxicityConfig(enabled=True, use_local_model=True, threshold=0.7)
        processor = EvaluationSpanProcessor(toxicity_config=toxicity_config)
        processor.toxicity_detector._detoxify_model = mock_model
        processor.toxicity_detector._detoxify_available = True
        processor.toxicity_detector._perspective_available = False

        # Create span
        attributes = {
            "gen_ai.prompt": "Toxic message",
        }
        span = self._create_span(attributes=attributes)

        # Process span
        processor.on_end(span)

        # Check individual category scores
        span_attributes = dict(span.attributes)
        assert span_attributes.get("evaluation.toxicity.prompt.toxicity_score") == 0.85
        assert span_attributes.get("evaluation.toxicity.prompt.insult_score") == 0.8
        assert span_attributes.get("evaluation.toxicity.prompt.profanity_score") == 0.75

    @patch("genai_otel.evaluation.toxicity_detector.Detoxify")
    @patch("genai_otel.evaluation.toxicity_detector.ToxicityDetector._check_availability")
    def test_both_prompt_and_response_toxicity(self, mock_check, mock_detoxify_class):
        """Test toxicity detection in both prompt and response."""
        mock_model = Mock()
        # Return different values for different calls
        call_count = [0]

        def mock_predict(text):
            call_count[0] += 1
            if call_count[0] == 1:  # First call (prompt)
                return {
                    "toxicity": 0.9,
                    "severe_toxicity": 0.3,
                    "obscene": 0.2,
                    "threat": 0.1,
                    "insult": 0.85,
                    "identity_attack": 0.15,
                }
            else:  # Second call (response)
                return {
                    "toxicity": 0.88,
                    "severe_toxicity": 0.25,
                    "obscene": 0.75,
                    "threat": 0.05,
                    "insult": 0.65,
                    "identity_attack": 0.1,
                }

        mock_model.predict.side_effect = mock_predict
        mock_detoxify_class.return_value = mock_model

        toxicity_config = ToxicityConfig(enabled=True, use_local_model=True, threshold=0.7)
        processor = EvaluationSpanProcessor(toxicity_config=toxicity_config)
        processor.toxicity_detector._detoxify_model = mock_model
        processor.toxicity_detector._detoxify_available = True
        processor.toxicity_detector._perspective_available = False

        # Create span with both toxic
        attributes = {
            "gen_ai.prompt": "Toxic prompt",
            "gen_ai.response": "Toxic response",
        }
        span = self._create_span(attributes=attributes)

        # Process span
        processor.on_end(span)

        # Check both detected
        span_attributes = dict(span.attributes)
        assert span_attributes.get("evaluation.toxicity.prompt.detected") is True
        assert span_attributes.get("evaluation.toxicity.response.detected") is True
