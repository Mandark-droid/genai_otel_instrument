"""Tests for CostEnrichingSpanExporter."""

import unittest
from unittest.mock import MagicMock, Mock, patch

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import Status, StatusCode

from genai_otel.cost_calculator import CostCalculator
from genai_otel.cost_enriching_exporter import CostEnrichingSpanExporter


class TestCostEnrichingSpanExporter(unittest.TestCase):
    """Tests for CostEnrichingSpanExporter."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_exporter = MagicMock()
        self.mock_exporter.export.return_value = SpanExportResult.SUCCESS
        self.mock_calculator = MagicMock(spec=CostCalculator)

    def test_init_with_cost_calculator(self):
        """Test initialization with provided cost calculator."""
        exporter = CostEnrichingSpanExporter(self.mock_exporter, self.mock_calculator)

        self.assertEqual(exporter.wrapped_exporter, self.mock_exporter)
        self.assertEqual(exporter.cost_calculator, self.mock_calculator)

    def test_init_without_cost_calculator(self):
        """Test initialization creates default cost calculator."""
        with patch("genai_otel.cost_enriching_exporter.CostCalculator") as mock_calc_class:
            mock_instance = MagicMock()
            mock_calc_class.return_value = mock_instance

            exporter = CostEnrichingSpanExporter(self.mock_exporter)

            mock_calc_class.assert_called_once()
            self.assertEqual(exporter.cost_calculator, mock_instance)

    def test_export_success(self):
        """Test successful export of spans."""
        exporter = CostEnrichingSpanExporter(self.mock_exporter, self.mock_calculator)

        span1 = self._create_span(attributes={"gen_ai.request.model": "gpt-4"})
        span2 = self._create_span(attributes={"gen_ai.request.model": "gpt-3.5-turbo"})
        spans = [span1, span2]

        result = exporter.export(spans)

        self.assertEqual(result, SpanExportResult.SUCCESS)
        self.mock_exporter.export.assert_called_once()

    def test_export_failure(self):
        """Test export failure handling."""
        self.mock_exporter.export.side_effect = Exception("Export failed")
        exporter = CostEnrichingSpanExporter(self.mock_exporter, self.mock_calculator)

        span = self._create_span(attributes={"gen_ai.request.model": "gpt-4"})
        result = exporter.export([span])

        self.assertEqual(result, SpanExportResult.FAILURE)

    def test_enrich_span_without_attributes(self):
        """Test enriching span without attributes returns original span."""
        exporter = CostEnrichingSpanExporter(self.mock_exporter, self.mock_calculator)

        span = self._create_span(attributes=None)
        enriched = exporter._enrich_span(span)

        self.assertEqual(enriched, span)

    def test_enrich_span_without_model(self):
        """Test enriching span without model name returns original span."""
        exporter = CostEnrichingSpanExporter(self.mock_exporter, self.mock_calculator)

        span = self._create_span(attributes={"some_key": "some_value"})
        enriched = exporter._enrich_span(span)

        self.assertEqual(enriched, span)

    def test_enrich_span_with_existing_cost(self):
        """Test enriching span that already has cost attributes skips enrichment."""
        exporter = CostEnrichingSpanExporter(self.mock_exporter, self.mock_calculator)

        span = self._create_span(
            attributes={
                "gen_ai.request.model": "gpt-4",
                "gen_ai.usage.cost.total": 0.05,
                "gen_ai.usage.prompt_tokens": 100,
                "gen_ai.usage.completion_tokens": 50,
            }
        )
        enriched = exporter._enrich_span(span)

        self.assertEqual(enriched, span)
        # Calculator should not be called
        self.mock_calculator.calculate_granular_cost.assert_not_called()

    def test_enrich_span_without_tokens(self):
        """Test enriching span without token usage returns original span."""
        exporter = CostEnrichingSpanExporter(self.mock_exporter, self.mock_calculator)

        span = self._create_span(attributes={"gen_ai.request.model": "gpt-4"})
        enriched = exporter._enrich_span(span)

        self.assertEqual(enriched, span)

    def test_enrich_span_with_valid_data(self):
        """Test enriching span with valid model and token data."""
        self.mock_calculator.calculate_granular_cost.return_value = {
            "total": 0.05,
            "prompt": 0.03,
            "completion": 0.02,
        }

        exporter = CostEnrichingSpanExporter(self.mock_exporter, self.mock_calculator)

        span = self._create_span(
            attributes={
                "gen_ai.request.model": "gpt-4",
                "gen_ai.usage.prompt_tokens": 100,
                "gen_ai.usage.completion_tokens": 50,
            }
        )
        enriched = exporter._enrich_span(span)

        # Verify cost calculator was called
        self.mock_calculator.calculate_granular_cost.assert_called_once_with(
            model="gpt-4",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            call_type="chat",
        )

        # Verify enriched span has cost attributes
        self.assertIn("gen_ai.usage.cost.total", enriched.attributes)
        self.assertEqual(enriched.attributes["gen_ai.usage.cost.total"], 0.05)
        self.assertEqual(enriched.attributes["gen_ai.usage.cost.prompt"], 0.03)
        self.assertEqual(enriched.attributes["gen_ai.usage.cost.completion"], 0.02)

    def test_enrich_span_with_openinference_attributes(self):
        """Test enriching span with OpenInference attribute names."""
        self.mock_calculator.calculate_granular_cost.return_value = {
            "total": 0.03,
            "prompt": 0.02,
            "completion": 0.01,
        }

        exporter = CostEnrichingSpanExporter(self.mock_exporter, self.mock_calculator)

        span = self._create_span(
            attributes={
                "llm.model_name": "gpt-3.5-turbo",
                "llm.token_count.prompt": 75,
                "llm.token_count.completion": 25,
                "openinference.span.kind": "LLM",
            }
        )
        enriched = exporter._enrich_span(span)

        # Verify cost calculator was called with mapped attributes
        self.mock_calculator.calculate_granular_cost.assert_called_once_with(
            model="gpt-3.5-turbo",
            usage={"prompt_tokens": 75, "completion_tokens": 25, "total_tokens": 100},
            call_type="chat",
        )

        # Verify enriched span has cost attributes
        self.assertIn("gen_ai.usage.cost.total", enriched.attributes)
        self.assertEqual(enriched.attributes["gen_ai.usage.cost.total"], 0.03)

    def test_enrich_span_with_alternative_token_names(self):
        """Test enriching span with alternative token attribute names."""
        self.mock_calculator.calculate_granular_cost.return_value = {
            "total": 0.02,
            "prompt": 0.01,
            "completion": 0.01,
        }

        exporter = CostEnrichingSpanExporter(self.mock_exporter, self.mock_calculator)

        span = self._create_span(
            attributes={
                "gen_ai.request.model": "claude-3-sonnet",
                "gen_ai.usage.input_tokens": 50,
                "gen_ai.usage.output_tokens": 30,
            }
        )
        enriched = exporter._enrich_span(span)

        # Verify cost calculator was called with correct token counts
        self.mock_calculator.calculate_granular_cost.assert_called_once_with(
            model="claude-3-sonnet",
            usage={"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
            call_type="chat",
        )

    def test_enrich_span_with_embedding_call_type(self):
        """Test enriching span with embedding operation."""
        self.mock_calculator.calculate_granular_cost.return_value = {"total": 0.01}

        exporter = CostEnrichingSpanExporter(self.mock_exporter, self.mock_calculator)

        span = self._create_span(
            attributes={
                "embedding.model_name": "text-embedding-ada-002",
                "llm.token_count.prompt": 100,
                "llm.token_count.completion": 0,
                "gen_ai.operation.name": "embedding",
            }
        )
        enriched = exporter._enrich_span(span)

        # Verify cost calculator was called with embedding type
        self.mock_calculator.calculate_granular_cost.assert_called_once()
        call_args = self.mock_calculator.calculate_granular_cost.call_args
        self.assertEqual(call_args[1]["call_type"], "embedding")

    def test_enrich_span_call_type_mapping(self):
        """Test call type mapping for various operation names."""
        test_cases = [
            ("RETRIEVER", "embedding"),
            ("RERANKER", "embedding"),
            ("AGENT", "chat"),
            ("TOOL", "chat"),
            ("CHAIN", "chat"),
            ("TEXT_GENERATION", "chat"),
        ]

        for operation_name, expected_call_type in test_cases:
            self.mock_calculator.reset_mock()
            self.mock_calculator.calculate_granular_cost.return_value = {"total": 0.01}

            exporter = CostEnrichingSpanExporter(self.mock_exporter, self.mock_calculator)

            span = self._create_span(
                attributes={
                    "gen_ai.request.model": "gpt-4",
                    "gen_ai.usage.prompt_tokens": 10,
                    "gen_ai.usage.completion_tokens": 5,
                    "openinference.span.kind": operation_name,
                }
            )
            enriched = exporter._enrich_span(span)

            call_args = self.mock_calculator.calculate_granular_cost.call_args
            self.assertEqual(
                call_args[1]["call_type"],
                expected_call_type,
                f"Failed for operation: {operation_name}",
            )

    def test_enrich_span_with_zero_cost(self):
        """Test enriching span when calculator returns zero cost."""
        self.mock_calculator.calculate_granular_cost.return_value = {"total": 0.0}

        exporter = CostEnrichingSpanExporter(self.mock_exporter, self.mock_calculator)

        span = self._create_span(
            attributes={
                "gen_ai.request.model": "unknown-model",
                "gen_ai.usage.prompt_tokens": 10,
                "gen_ai.usage.completion_tokens": 5,
            }
        )
        enriched = exporter._enrich_span(span)

        # Should return original span when cost is zero
        self.assertEqual(enriched, span)

    def test_enrich_span_with_none_cost(self):
        """Test enriching span when calculator returns None."""
        self.mock_calculator.calculate_granular_cost.return_value = None

        exporter = CostEnrichingSpanExporter(self.mock_exporter, self.mock_calculator)

        span = self._create_span(
            attributes={
                "gen_ai.request.model": "gpt-4",
                "gen_ai.usage.prompt_tokens": 10,
                "gen_ai.usage.completion_tokens": 5,
            }
        )
        enriched = exporter._enrich_span(span)

        # Should return original span when cost is None
        self.assertEqual(enriched, span)

    def test_enrich_span_exception_handling(self):
        """Test that exceptions during enrichment are handled gracefully."""
        self.mock_calculator.calculate_granular_cost.side_effect = Exception("Calc error")

        exporter = CostEnrichingSpanExporter(self.mock_exporter, self.mock_calculator)

        span = self._create_span(
            attributes={
                "gen_ai.request.model": "gpt-4",
                "gen_ai.usage.prompt_tokens": 10,
                "gen_ai.usage.completion_tokens": 5,
            }
        )

        # Should return original span on exception
        enriched = exporter._enrich_span(span)
        self.assertEqual(enriched, span)

    def test_shutdown(self):
        """Test shutdown calls wrapped exporter shutdown."""
        exporter = CostEnrichingSpanExporter(self.mock_exporter, self.mock_calculator)

        exporter.shutdown()

        self.mock_exporter.shutdown.assert_called_once()

    def test_force_flush_success(self):
        """Test force flush calls wrapped exporter and returns result."""
        self.mock_exporter.force_flush.return_value = True
        exporter = CostEnrichingSpanExporter(self.mock_exporter, self.mock_calculator)

        result = exporter.force_flush(timeout_millis=5000)

        self.assertTrue(result)
        self.mock_exporter.force_flush.assert_called_once_with(5000)

    def test_force_flush_failure(self):
        """Test force flush returns False on failure."""
        self.mock_exporter.force_flush.return_value = False
        exporter = CostEnrichingSpanExporter(self.mock_exporter, self.mock_calculator)

        result = exporter.force_flush()

        self.assertFalse(result)
        self.mock_exporter.force_flush.assert_called_once_with(30000)

    def test_enrich_span_with_only_prompt_cost(self):
        """Test enriching span when only prompt cost is returned."""
        self.mock_calculator.calculate_granular_cost.return_value = {
            "total": 0.05,
            "prompt": 0.05,
        }

        exporter = CostEnrichingSpanExporter(self.mock_exporter, self.mock_calculator)

        span = self._create_span(
            attributes={
                "gen_ai.request.model": "gpt-4",
                "gen_ai.usage.prompt_tokens": 100,
                "gen_ai.usage.completion_tokens": 0,
            }
        )
        enriched = exporter._enrich_span(span)

        # Verify enriched span has total and prompt cost, but no completion cost
        self.assertEqual(enriched.attributes["gen_ai.usage.cost.total"], 0.05)
        self.assertEqual(enriched.attributes["gen_ai.usage.cost.prompt"], 0.05)
        self.assertNotIn("gen_ai.usage.cost.completion", enriched.attributes)

    def _create_span(
        self,
        name="test_span",
        attributes=None,
        status=None,
        span_kind=SpanKind.INTERNAL,
    ):
        """Helper to create a mock ReadableSpan for testing."""
        mock_context = Mock()
        mock_context.trace_id = 123456789
        mock_context.span_id = 987654321

        if status is None:
            status = Status(StatusCode.OK)

        return ReadableSpan(
            name=name,
            context=mock_context,
            kind=span_kind,
            parent=None,
            start_time=1000000000,
            end_time=1000001000,
            status=status,
            attributes=attributes,
            events=[],
            links=[],
            resource=Mock(),
            instrumentation_scope=Mock(),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
