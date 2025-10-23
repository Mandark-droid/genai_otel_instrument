import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.togetherai_instrumentor import TogetherAIInstrumentor


class TestTogetherAIInstrumentor(unittest.TestCase):
    """Tests for TogetherAIInstrumentor"""

    def test_instrument_when_together_not_installed(self):
        """Test that instrument handles missing together gracefully."""
        with patch.dict("sys.modules", {"together": None}):
            instrumentor = TogetherAIInstrumentor()
            config = MagicMock()

            # Should not raise any exception
            instrumentor.instrument(config)

            # Config should be stored
            self.assertEqual(instrumentor.config, config)

    def test_instrument_with_together_installed(self):
        """Test that instrument wraps together.Complete.create when installed (legacy API)."""
        # Create mock together module with only Complete API (legacy)
        mock_together = MagicMock()
        original_create = MagicMock(return_value="completion result")
        mock_together.Complete.create = original_create
        # No Together class in legacy API
        del mock_together.Together

        with patch.dict("sys.modules", {"together": mock_together}):
            instrumentor = TogetherAIInstrumentor()
            config = MagicMock()

            # Create mock tracer and metrics
            mock_tracer = MagicMock()
            instrumentor.tracer = mock_tracer
            instrumentor.request_counter = MagicMock()
            instrumentor.token_counter = MagicMock()
            instrumentor.latency_histogram = MagicMock()
            instrumentor.cost_gauge = MagicMock()

            # Create mock span
            mock_span = MagicMock()
            mock_span_context = MagicMock()
            mock_span_context.__enter__ = MagicMock(return_value=mock_span)
            mock_span_context.__exit__ = MagicMock(return_value=None)
            mock_tracer.start_as_current_span.return_value = mock_span_context

            # Act
            instrumentor.instrument(config)

            # The create function should now be wrapped
            self.assertIsNotNone(mock_together.Complete.create)
            self.assertTrue(callable(mock_together.Complete.create))

            # Call the wrapped create function
            result = mock_together.Complete.create(
                model="mistralai/Mixtral-8x7B-v0.1", prompt="Test prompt"
            )

            # Assertions
            self.assertEqual(result, "completion result")
            # Verify original function was called
            original_create.assert_called_once_with(
                model="mistralai/Mixtral-8x7B-v0.1", prompt="Test prompt"
            )

    def test_wrapped_complete_without_model(self):
        """Test that wrapped complete handles call without model (uses 'unknown' as model)."""
        # Create mock together module
        mock_together = MagicMock()
        original_create = MagicMock(return_value="completion result")
        mock_together.Complete.create = original_create
        # No Together class in legacy API
        del mock_together.Together

        with patch.dict("sys.modules", {"together": mock_together}):
            instrumentor = TogetherAIInstrumentor()
            config = MagicMock()

            # Create mock tracer and metrics
            instrumentor.tracer = MagicMock()
            instrumentor.request_counter = MagicMock()
            instrumentor.token_counter = MagicMock()
            instrumentor.latency_histogram = MagicMock()
            instrumentor.cost_gauge = MagicMock()

            # Act
            instrumentor.instrument(config)

            # Call the wrapped create function without model kwarg
            result = mock_together.Complete.create(prompt="Test prompt")

            # Assertions
            self.assertEqual(result, "completion result")
            # Verify original function was called
            original_create.assert_called_once_with(prompt="Test prompt")

    def test_wrapped_complete_with_args_and_kwargs(self):
        """Test that wrapped complete handles both args and kwargs properly."""
        # Create mock together module
        mock_together = MagicMock()
        original_create = MagicMock(return_value="completion result")
        mock_together.Complete.create = original_create
        # No Together class in legacy API
        del mock_together.Together

        with patch.dict("sys.modules", {"together": mock_together}):
            instrumentor = TogetherAIInstrumentor()
            config = MagicMock()

            # Create mock tracer and metrics
            instrumentor.tracer = MagicMock()
            instrumentor.request_counter = MagicMock()
            instrumentor.token_counter = MagicMock()
            instrumentor.latency_histogram = MagicMock()
            instrumentor.cost_gauge = MagicMock()

            # Act
            instrumentor.instrument(config)

            # Call the wrapped create function with args and kwargs
            result = mock_together.Complete.create(
                model="test-model", prompt="Test prompt", max_tokens=100
            )

            # Assertions
            self.assertEqual(result, "completion result")

            # Verify original_create was called with the kwargs
            original_create.assert_called_once_with(
                model="test-model", prompt="Test prompt", max_tokens=100
            )

    def test_extract_usage(self):
        """Test that _extract_usage extracts token counts from Together AI response."""
        instrumentor = TogetherAIInstrumentor()

        # Test with None
        result = instrumentor._extract_usage(None)
        self.assertIsNone(result)

        # Test with response missing usage
        mock_response_no_usage = MagicMock(spec=[])
        result = instrumentor._extract_usage(mock_response_no_usage)
        self.assertIsNone(result)

        # Test with valid response containing usage (OpenAI-compatible format)
        mock_response = MagicMock()
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 25
        mock_response.usage.completion_tokens = 75
        mock_response.usage.total_tokens = 100

        result = instrumentor._extract_usage(mock_response)
        self.assertIsNotNone(result)
        self.assertEqual(result["prompt_tokens"], 25)
        self.assertEqual(result["completion_tokens"], 75)
        self.assertEqual(result["total_tokens"], 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
