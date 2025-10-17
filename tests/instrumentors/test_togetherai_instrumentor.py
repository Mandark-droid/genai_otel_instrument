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
        """Test that instrument wraps together.Complete.create when installed."""
        # Create mock together module
        mock_together = MagicMock()
        original_create = MagicMock(return_value="completion result")
        mock_together.Complete.create = original_create

        with patch.dict("sys.modules", {"together": mock_together}):
            instrumentor = TogetherAIInstrumentor()
            config = MagicMock()

            # Create mock tracer
            mock_tracer = MagicMock()
            instrumentor.tracer = mock_tracer

            # Create mock span
            mock_span = MagicMock()
            mock_span_context = MagicMock()
            mock_span_context.__enter__ = MagicMock(return_value=mock_span)
            mock_span_context.__exit__ = MagicMock(return_value=None)
            mock_tracer.start_as_current_span.return_value = mock_span_context

            # Create mock request counter
            mock_request_counter = MagicMock()
            instrumentor.request_counter = mock_request_counter

            # Act
            instrumentor.instrument(config)

            # The create function should now be wrapped
            self.assertNotEqual(mock_together.Complete.create, original_create)

            # Call the wrapped create function
            result = mock_together.Complete.create(
                model="mistralai/Mixtral-8x7B-v0.1", prompt="Test prompt"
            )

            # Assertions
            self.assertEqual(result, "completion result")

            # Verify tracing was called
            mock_tracer.start_as_current_span.assert_called_once_with("together.complete")

            # Verify span attributes were set
            mock_span.set_attribute.assert_any_call("gen_ai.system", "together")
            mock_span.set_attribute.assert_any_call(
                "gen_ai.request.model", "mistralai/Mixtral-8x7B-v0.1"
            )

            # Verify request counter was called
            mock_request_counter.add.assert_called_once_with(
                1, {"model": "mistralai/Mixtral-8x7B-v0.1", "provider": "together"}
            )

    def test_wrapped_complete_without_model(self):
        """Test that wrapped complete handles call without model (uses 'unknown' as model)."""
        # Create mock together module
        mock_together = MagicMock()
        original_create = MagicMock(return_value="completion result")
        mock_together.Complete.create = original_create

        with patch.dict("sys.modules", {"together": mock_together}):
            instrumentor = TogetherAIInstrumentor()
            config = MagicMock()

            # Create mock tracer
            mock_tracer = MagicMock()
            instrumentor.tracer = mock_tracer

            # Create mock span
            mock_span = MagicMock()
            mock_span_context = MagicMock()
            mock_span_context.__enter__ = MagicMock(return_value=mock_span)
            mock_span_context.__exit__ = MagicMock(return_value=None)
            mock_tracer.start_as_current_span.return_value = mock_span_context

            # Create mock request counter
            mock_request_counter = MagicMock()
            instrumentor.request_counter = mock_request_counter

            # Act
            instrumentor.instrument(config)

            # Call the wrapped create function without model kwarg
            result = mock_together.Complete.create(prompt="Test prompt")

            # Assertions
            self.assertEqual(result, "completion result")

            # Verify span attribute was set with "unknown" model
            mock_span.set_attribute.assert_any_call("gen_ai.request.model", "unknown")

            # Verify request counter was called with "unknown" model
            mock_request_counter.add.assert_called_once_with(
                1, {"model": "unknown", "provider": "together"}
            )

    def test_wrapped_complete_with_args_and_kwargs(self):
        """Test that wrapped complete handles both args and kwargs properly."""
        # Create mock together module
        mock_together = MagicMock()
        original_create = MagicMock(return_value="completion result")
        mock_together.Complete.create = original_create

        with patch.dict("sys.modules", {"together": mock_together}):
            instrumentor = TogetherAIInstrumentor()
            config = MagicMock()

            # Create mock tracer
            mock_tracer = MagicMock()
            instrumentor.tracer = mock_tracer

            # Create mock span
            mock_span = MagicMock()
            mock_span_context = MagicMock()
            mock_span_context.__enter__ = MagicMock(return_value=mock_span)
            mock_span_context.__exit__ = MagicMock(return_value=None)
            mock_tracer.start_as_current_span.return_value = mock_span_context

            # Create mock request counter
            mock_request_counter = MagicMock()
            instrumentor.request_counter = mock_request_counter

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
        """Test that _extract_usage returns None."""
        instrumentor = TogetherAIInstrumentor()
        result = instrumentor._extract_usage("any_result")

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
