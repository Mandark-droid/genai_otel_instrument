"""Tests for CostEnrichmentSpanProcessor."""

import unittest
from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace import Span

from genai_otel.cost_enrichment_processor import CostEnrichmentSpanProcessor


class TestCostEnrichmentSpanProcessor(unittest.TestCase):
    """Tests for CostEnrichmentSpanProcessor."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = CostEnrichmentSpanProcessor()

    def test_init(self):
        """Test processor initialization."""
        processor = CostEnrichmentSpanProcessor()
        self.assertIsNotNone(processor.cost_calculator)

    def test_init_with_custom_calculator(self):
        """Test processor initialization with custom cost calculator."""
        mock_calculator = MagicMock()
        processor = CostEnrichmentSpanProcessor(cost_calculator=mock_calculator)
        self.assertEqual(processor.cost_calculator, mock_calculator)

    def test_on_start_does_nothing(self):
        """Test that on_start doesn't raise errors."""
        mock_span = MagicMock()
        # Should not raise
        self.processor.on_start(mock_span)

    def test_on_end_no_attributes(self):
        """Test on_end with span that has no attributes."""
        mock_span = MagicMock(spec=Span)
        mock_span.attributes = None

        # Should not raise
        self.processor.on_end(mock_span)

    def test_on_end_no_model(self):
        """Test on_end with span that has no model attribute."""
        mock_span = MagicMock(spec=Span)
        mock_span.attributes = {"some_key": "some_value"}

        # Should not raise
        self.processor.on_end(mock_span)

    def test_on_end_no_tokens(self):
        """Test on_end with span that has model but no tokens."""
        mock_span = MagicMock(spec=Span)
        mock_span.attributes = {
            "gen_ai.request.model": "gpt-4",
            "gen_ai.usage.prompt_tokens": 0,
            "gen_ai.usage.completion_tokens": 0,
        }

        # Should not raise, but should not add cost
        self.processor.on_end(mock_span)
        mock_span.set_attribute.assert_not_called()

    def test_on_end_with_valid_tokens_new_convention(self):
        """Test on_end with valid tokens using new semantic conventions."""
        mock_span = MagicMock(spec=Span)
        mock_span.attributes = {
            "gen_ai.request.model": "gpt-4o",
            "gen_ai.usage.prompt_tokens": 100,
            "gen_ai.usage.completion_tokens": 200,
            "gen_ai.operation.name": "chat",
        }

        # Mock the cost calculator
        with patch.object(
            self.processor.cost_calculator,
            "calculate_granular_cost",
            return_value={
                "total": 0.0035,
                "prompt": 0.00005,
                "completion": 0.0003,
                "reasoning": 0.0,
                "cache_read": 0.0,
                "cache_write": 0.0,
            },
        ) as mock_calc:
            self.processor.on_end(mock_span)

            # Verify calculate_granular_cost was called with correct parameters
            mock_calc.assert_called_once_with(
                model="gpt-4o",
                usage={"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
                call_type="chat",
            )

            # Verify cost attributes were set
            mock_span.set_attribute.assert_any_call("gen_ai.usage.cost.total", 0.0035)
            mock_span.set_attribute.assert_any_call("gen_ai.usage.cost.prompt", 0.00005)
            mock_span.set_attribute.assert_any_call("gen_ai.usage.cost.completion", 0.0003)

    def test_on_end_with_valid_tokens_old_convention(self):
        """Test on_end with valid tokens using old semantic conventions."""
        mock_span = MagicMock(spec=Span)
        mock_span.attributes = {
            "gen_ai.request.model": "gpt-3.5-turbo",
            "gen_ai.usage.input_tokens": 50,
            "gen_ai.usage.output_tokens": 100,
            "gen_ai.operation.name": "completion",
        }

        # Mock the cost calculator
        with patch.object(
            self.processor.cost_calculator,
            "calculate_granular_cost",
            return_value={
                "total": 0.00015,
                "prompt": 0.000025,
                "completion": 0.00015,
                "reasoning": 0.0,
                "cache_read": 0.0,
                "cache_write": 0.0,
            },
        ) as mock_calc:
            self.processor.on_end(mock_span)

            # Verify calculate_granular_cost was called
            mock_calc.assert_called_once_with(
                model="gpt-3.5-turbo",
                usage={"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150},
                call_type="chat",
            )

            # Verify cost attribute was set
            mock_span.set_attribute.assert_any_call("gen_ai.usage.cost.total", 0.00015)

    def test_on_end_with_embedding_operation(self):
        """Test on_end with embedding operation."""
        mock_span = MagicMock(spec=Span)
        mock_span.attributes = {
            "gen_ai.request.model": "text-embedding-ada-002",
            "gen_ai.usage.prompt_tokens": 1000,
            "gen_ai.usage.completion_tokens": 0,
            "gen_ai.operation.name": "embedding",
        }

        # Mock the cost calculator
        with patch.object(
            self.processor.cost_calculator,
            "calculate_granular_cost",
            return_value={
                "total": 0.0001,
                "prompt": 0.0,
                "completion": 0.0,
                "reasoning": 0.0,
                "cache_read": 0.0,
                "cache_write": 0.0,
            },
        ) as mock_calc:
            self.processor.on_end(mock_span)

            # Verify calculate_granular_cost was called with embedding type
            mock_calc.assert_called_once_with(
                model="text-embedding-ada-002",
                usage={"prompt_tokens": 1000, "completion_tokens": 0, "total_tokens": 1000},
                call_type="embedding",
            )

    def test_on_end_operation_name_mapping(self):
        """Test that operation names are correctly mapped to call types."""
        test_cases = [
            ("chat", "chat"),
            ("completion", "chat"),
            ("embedding", "embedding"),
            ("embeddings", "embedding"),
            ("text_generation", "chat"),
            ("image_generation", "image"),
            ("audio", "audio"),
            ("unknown_operation", "chat"),  # Default fallback
        ]

        for operation_name, expected_call_type in test_cases:
            with self.subTest(operation=operation_name):
                mock_span = MagicMock(spec=Span)
                mock_span.attributes = {
                    "gen_ai.request.model": "test-model",
                    "gen_ai.usage.prompt_tokens": 10,
                    "gen_ai.usage.completion_tokens": 20,
                    "gen_ai.operation.name": operation_name,
                }

                with patch.object(
                    self.processor.cost_calculator,
                    "calculate_granular_cost",
                    return_value={
                        "total": 0.001,
                        "prompt": 0.0,
                        "completion": 0.0,
                        "reasoning": 0.0,
                        "cache_read": 0.0,
                        "cache_write": 0.0,
                    },
                ) as mock_calc:
                    self.processor.on_end(mock_span)

                    # Verify correct call_type was used
                    call_args = mock_calc.call_args
                    self.assertEqual(call_args[1]["call_type"], expected_call_type)

    def test_on_end_no_cost_calculated(self):
        """Test on_end when cost calculator returns zero cost."""
        mock_span = MagicMock(spec=Span)
        mock_span.attributes = {
            "gen_ai.request.model": "unknown-model",
            "gen_ai.usage.prompt_tokens": 100,
            "gen_ai.usage.completion_tokens": 200,
        }

        # Mock the cost calculator to return zero cost
        with patch.object(
            self.processor.cost_calculator,
            "calculate_granular_cost",
            return_value={
                "total": 0.0,
                "prompt": 0.0,
                "completion": 0.0,
                "reasoning": 0.0,
                "cache_read": 0.0,
                "cache_write": 0.0,
            },
        ):
            self.processor.on_end(mock_span)

            # Should not raise, should not set attributes when cost is 0
            mock_span.set_attribute.assert_not_called()

    def test_on_end_handles_exceptions(self):
        """Test that on_end handles exceptions gracefully."""
        mock_span = MagicMock(spec=Span)
        mock_span.name = "test-span"
        mock_span.attributes = {
            "gen_ai.request.model": "gpt-4",
            "gen_ai.usage.prompt_tokens": 100,
            "gen_ai.usage.completion_tokens": 200,
        }

        # Mock the cost calculator to raise an exception
        with patch.object(
            self.processor.cost_calculator,
            "calculate_granular_cost",
            side_effect=Exception("Test error"),
        ):
            # Should not raise
            self.processor.on_end(mock_span)

    def test_on_end_default_operation_name(self):
        """Test that missing operation name defaults to 'chat'."""
        mock_span = MagicMock(spec=Span)
        mock_span.attributes = {
            "gen_ai.request.model": "gpt-4",
            "gen_ai.usage.prompt_tokens": 100,
            "gen_ai.usage.completion_tokens": 200,
            # No gen_ai.operation.name
        }

        with patch.object(
            self.processor.cost_calculator,
            "calculate_granular_cost",
            return_value={
                "total": 0.001,
                "prompt": 0.0,
                "completion": 0.0,
                "reasoning": 0.0,
                "cache_read": 0.0,
                "cache_write": 0.0,
            },
        ) as mock_calc:
            self.processor.on_end(mock_span)

            # Verify default call_type is "chat"
            call_args = mock_calc.call_args
            self.assertEqual(call_args[1]["call_type"], "chat")

    def test_shutdown(self):
        """Test shutdown method."""
        # Should not raise
        self.processor.shutdown()

    def test_force_flush(self):
        """Test force_flush method."""
        result = self.processor.force_flush(timeout_millis=5000)
        self.assertTrue(result)

    def test_openinference_llm_span(self):
        """Test on_end with OpenInference LLM span attributes."""
        mock_span = MagicMock(spec=Span)
        mock_span.attributes = {
            "llm.model_name": "gpt-4",
            "llm.token_count.prompt": 100,
            "llm.token_count.completion": 200,
            "openinference.span.kind": "LLM",
        }

        with patch.object(
            self.processor.cost_calculator,
            "calculate_granular_cost",
            return_value={
                "total": 0.003,
                "prompt": 0.0003,
                "completion": 0.012,
                "reasoning": 0.0,
                "cache_read": 0.0,
                "cache_write": 0.0,
            },
        ) as mock_calc:
            self.processor.on_end(mock_span)

            # Verify calculate_granular_cost was called with correct parameters
            mock_calc.assert_called_once_with(
                model="gpt-4",
                usage={"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
                call_type="chat",  # LLM maps to chat
            )

            # Verify cost attributes were set
            mock_span.set_attribute.assert_any_call("gen_ai.usage.cost.total", 0.003)

    def test_openinference_embedding_span(self):
        """Test on_end with OpenInference EMBEDDING span attributes."""
        mock_span = MagicMock(spec=Span)
        mock_span.attributes = {
            "embedding.model_name": "text-embedding-ada-002",
            "llm.token_count.prompt": 500,
            "llm.token_count.completion": 0,
            "openinference.span.kind": "EMBEDDING",
        }

        with patch.object(
            self.processor.cost_calculator,
            "calculate_granular_cost",
            return_value={
                "total": 0.00005,
                "prompt": 0.0,
                "completion": 0.0,
                "reasoning": 0.0,
                "cache_read": 0.0,
                "cache_write": 0.0,
            },
        ) as mock_calc:
            self.processor.on_end(mock_span)

            # Verify calculate_granular_cost was called with embedding type
            mock_calc.assert_called_once_with(
                model="text-embedding-ada-002",
                usage={"prompt_tokens": 500, "completion_tokens": 0, "total_tokens": 500},
                call_type="embedding",
            )

    def test_openinference_missing_model(self):
        """Test that OpenInference span without model is skipped."""
        mock_span = MagicMock(spec=Span)
        mock_span.attributes = {
            "llm.token_count.prompt": 100,
            "llm.token_count.completion": 200,
            "openinference.span.kind": "LLM",
            # No llm.model_name
        }

        self.processor.on_end(mock_span)
        mock_span.set_attribute.assert_not_called()

    def test_openinference_missing_tokens(self):
        """Test that OpenInference span without tokens is skipped."""
        mock_span = MagicMock(spec=Span)
        mock_span.attributes = {
            "llm.model_name": "gpt-4",
            "openinference.span.kind": "LLM",
            # No token counts
        }

        self.processor.on_end(mock_span)
        mock_span.set_attribute.assert_not_called()

    def test_openinference_span_kinds(self):
        """Test that all OpenInference span.kind values map correctly."""
        span_kind_mappings = [
            ("LLM", "chat"),
            ("EMBEDDING", "embedding"),
            ("CHAIN", "chat"),
            ("RETRIEVER", "embedding"),
            ("RERANKER", "embedding"),
            ("TOOL", "chat"),
            ("AGENT", "chat"),
        ]

        for span_kind, expected_call_type in span_kind_mappings:
            with self.subTest(span_kind=span_kind):
                mock_span = MagicMock(spec=Span)
                mock_span.attributes = {
                    "llm.model_name": "test-model",
                    "llm.token_count.prompt": 10,
                    "llm.token_count.completion": 20,
                    "openinference.span.kind": span_kind,
                }

                with patch.object(
                    self.processor.cost_calculator,
                    "calculate_granular_cost",
                    return_value={
                        "total": 0.001,
                        "prompt": 0.0,
                        "completion": 0.0,
                        "reasoning": 0.0,
                        "cache_read": 0.0,
                        "cache_write": 0.0,
                    },
                ) as mock_calc:
                    self.processor.on_end(mock_span)

                    # Verify correct call_type was used
                    call_args = mock_calc.call_args
                    self.assertEqual(call_args[1]["call_type"], expected_call_type)

    @patch("genai_otel.cost_enrichment_processor.logger")
    def test_skip_when_cost_already_present(self, mock_logger):
        """Test that processor skips enrichment when cost attributes are already present."""
        mock_span = MagicMock(spec=Span)
        mock_span.name = "test.span"
        mock_span.attributes = {
            "gen_ai.request.model": "gpt-4",
            "gen_ai.usage.prompt_tokens": 100,
            "gen_ai.usage.completion_tokens": 50,
            "gen_ai.usage.cost.total": 0.005,  # Cost already present
            "gen_ai.usage.cost.prompt": 0.003,
            "gen_ai.usage.cost.completion": 0.002,
        }

        # Mock the cost calculator to track if it's called
        with patch.object(
            self.processor.cost_calculator,
            "calculate_granular_cost",
        ) as mock_calc:
            self.processor.on_end(mock_span)

            # Verify cost calculator was NOT called
            mock_calc.assert_not_called()

            # Verify debug log was emitted
            mock_logger.debug.assert_called_with(
                "Span 'test.span' already has cost attributes, skipping enrichment"
            )

            # Verify set_attribute was NOT called on span
            mock_span.set_attribute.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
