"""Tests for response-content bounding and evaluation dedup guards.

Covers:
  - LiteLLM / Smolagents / MCP enrichment processors bound copied response text
  - The evaluation dedup guard (span processor + enriching exporter) skips spans
    that have already been evaluated.
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import Status, StatusCode

from genai_otel.evaluation_enriching_exporter import EvaluationEnrichingSpanExporter
from genai_otel.litellm_span_enrichment_processor import LiteLLMSpanEnrichmentProcessor
from genai_otel.mcp_span_enrichment_processor import MCPSpanEnrichmentProcessor
from genai_otel.smolagents_span_enrichment_processor import SmolagentsSpanEnrichmentProcessor


class TestResponseContentBounding(unittest.TestCase):
    """Response text copied onto gen_ai.response must be bounded."""

    def test_litellm_default_cap(self):
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("GENAI_CONTENT_MAX_LENGTH", None)
            processor = LiteLLMSpanEnrichmentProcessor()
        span = MagicMock()
        span.attributes = {"output.value": "x" * 50000}
        result = processor._extract_response_content(span)
        self.assertEqual(len(result), 10000)  # bounded default

    def test_litellm_custom_cap(self):
        with patch.dict("os.environ", {"GENAI_CONTENT_MAX_LENGTH": "42"}):
            processor = LiteLLMSpanEnrichmentProcessor()
        span = MagicMock()
        span.attributes = {"output.value": "y" * 5000}
        result = processor._extract_response_content(span)
        self.assertEqual(len(result), 42)

    def test_litellm_unlimited_cap(self):
        with patch.dict("os.environ", {"GENAI_CONTENT_MAX_LENGTH": "0"}):
            processor = LiteLLMSpanEnrichmentProcessor()
        span = MagicMock()
        span.attributes = {"output.value": "z" * 30000}
        result = processor._extract_response_content(span)
        self.assertEqual(len(result), 30000)  # 0 => unlimited

    def test_smolagents_default_cap(self):
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("GENAI_CONTENT_MAX_LENGTH", None)
            processor = SmolagentsSpanEnrichmentProcessor()
        span = MagicMock()
        span.attributes = {"output.value": "a" * 25000}
        result = processor._extract_response_content(span)
        self.assertEqual(len(result), 10000)

    def test_mcp_default_cap_tool_result(self):
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("GENAI_CONTENT_MAX_LENGTH", None)
            processor = MCPSpanEnrichmentProcessor()
        span = MagicMock()
        span.attributes = {"tool.result": "b" * 40000}
        result = processor._extract_response_content(span)
        self.assertEqual(len(result), 10000)

    def test_short_response_unchanged(self):
        processor = LiteLLMSpanEnrichmentProcessor()
        span = MagicMock()
        span.attributes = {"output.value": "short response"}
        result = processor._extract_response_content(span)
        self.assertEqual(result, "short response")


class TestEvaluationDedupGuard(unittest.TestCase):
    """The enriching exporter must skip already-evaluated spans."""

    def _make_processor(self):
        proc = MagicMock()
        proc.pii_config.enabled = True
        proc.toxicity_config.enabled = False
        proc.bias_config.enabled = False
        proc.prompt_injection_config.enabled = False
        proc.restricted_topics_config.enabled = False
        proc.hallucination_config.enabled = False
        proc.pii_detector = MagicMock()
        return proc

    def _span(self, attributes):
        return ReadableSpan(
            name="s",
            context=Mock(trace_id=1, span_id=2),
            kind=SpanKind.INTERNAL,
            parent=None,
            start_time=1,
            end_time=2,
            status=Status(StatusCode.OK),
            attributes=attributes,
            events=[],
            links=[],
            resource=Mock(),
            instrumentation_scope=Mock(),
        )

    def test_exporter_skips_when_completed_marker_present(self):
        proc = self._make_processor()
        exporter = EvaluationEnrichingSpanExporter(MagicMock(), proc)
        span = self._span({"gen_ai.request.first_message": "hi", "evaluation.completed": True})

        enriched = exporter._enrich_span(span)

        self.assertIs(enriched, span)
        proc.pii_detector.detect.assert_not_called()

    def test_exporter_marks_completed_after_enrichment(self):
        proc = self._make_processor()
        pii_result = MagicMock()
        pii_result.has_pii = False
        proc.pii_detector.detect.return_value = pii_result
        proc._extract_prompt.return_value = "hello"
        proc._extract_response.return_value = None

        exporter = EvaluationEnrichingSpanExporter(MagicMock(), proc)
        span = self._span({"gen_ai.request.first_message": "hello"})

        enriched = exporter._enrich_span(span)

        self.assertIsNot(enriched, span)
        self.assertTrue(enriched.attributes["evaluation.completed"])


class TestSpanProcessorDedupGuard(unittest.TestCase):
    """The evaluation span processor must skip already-evaluated spans."""

    @patch("genai_otel.evaluation.pii_detector.PIIDetector._check_availability")
    def test_on_end_skips_when_marker_present(self, mock_check):
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        from genai_otel.evaluation.config import PIIConfig, PIIMode
        from genai_otel.evaluation.span_processor import EvaluationSpanProcessor

        tracer_provider = TracerProvider(resource=Resource.create({"service.name": "t"}))
        trace.set_tracer_provider(tracer_provider)
        tracer = tracer_provider.get_tracer(__name__)

        processor = EvaluationSpanProcessor(pii_config=PIIConfig(enabled=True, mode=PIIMode.DETECT))
        processor.pii_detector._presidio_available = False
        processor.pii_detector.detect = Mock(
            side_effect=AssertionError("detector must not run on evaluated span")
        )

        span = tracer.start_span("s", kind=SpanKind.CLIENT)
        span.set_attribute("gen_ai.prompt", "My email is test@example.com")
        span.set_attribute("evaluation.completed", True)

        # Must not raise: the dedup guard skips before any detector runs.
        processor.on_end(span)
        processor.pii_detector.detect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
