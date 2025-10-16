import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.vertexai_instrumentor import VertexAIInstrumentor


class TestVertexAIInstrumentor(unittest.TestCase):
    """Tests for VertexAIInstrumentor"""

    def test_instrument_when_vertexai_not_installed(self):
        """Test that instrument handles missing vertexai gracefully."""
        with patch.dict("sys.modules", {"vertexai.preview.generative_models": None}):
            instrumentor = VertexAIInstrumentor()
            config = MagicMock()

            # Should not raise any exception
            instrumentor.instrument(config)

            # Config should be stored
            self.assertEqual(instrumentor.config, config)

    def test_instrument_with_vertexai_installed(self):
        """Test that instrument wraps GenerativeModel.generate_content when installed."""
        # Create mock vertexai module
        mock_vertexai = MagicMock()
        mock_generative_model_class = MagicMock()
        original_generate = MagicMock(return_value="generated content")
        mock_generative_model_class.generate_content = original_generate
        mock_vertexai.GenerativeModel = mock_generative_model_class

        with patch.dict("sys.modules", {"vertexai.preview.generative_models": mock_vertexai}):
            instrumentor = VertexAIInstrumentor()
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

            # The generate_content method should now be wrapped
            self.assertNotEqual(mock_vertexai.GenerativeModel.generate_content, original_generate)

            # Create a mock instance with _model_name attribute
            mock_instance = MagicMock()
            mock_instance._model_name = "gemini-pro"

            # Call the wrapped generate_content method
            result = mock_vertexai.GenerativeModel.generate_content(
                mock_instance, "Test prompt"
            )

            # Assertions
            self.assertEqual(result, "generated content")

            # Verify tracing was called
            mock_tracer.start_as_current_span.assert_called_once_with("vertexai.generate_content")

            # Verify span attributes were set
            mock_span.set_attribute.assert_any_call("gen_ai.system", "vertexai")
            mock_span.set_attribute.assert_any_call("gen_ai.request.model", "gemini-pro")

            # Verify request counter was called
            mock_request_counter.add.assert_called_once_with(
                1, {"model": "gemini-pro", "provider": "vertexai"}
            )

    def test_wrapped_generate_without_model_name(self):
        """Test that wrapped generate_content handles instance without _model_name (uses 'unknown')."""
        # Create mock vertexai module
        mock_vertexai = MagicMock()
        mock_generative_model_class = MagicMock()
        original_generate = MagicMock(return_value="generated content")
        mock_generative_model_class.generate_content = original_generate
        mock_vertexai.GenerativeModel = mock_generative_model_class

        with patch.dict("sys.modules", {"vertexai.preview.generative_models": mock_vertexai}):
            instrumentor = VertexAIInstrumentor()
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

            # Create a mock instance WITHOUT _model_name attribute
            mock_instance = MagicMock(spec=[])  # spec=[] means no attributes

            # Call the wrapped generate_content method
            result = mock_vertexai.GenerativeModel.generate_content(
                mock_instance, "Test prompt"
            )

            # Assertions
            self.assertEqual(result, "generated content")

            # Verify span attribute was set with "unknown" model
            mock_span.set_attribute.assert_any_call("gen_ai.request.model", "unknown")

            # Verify request counter was called with "unknown" model
            mock_request_counter.add.assert_called_once_with(
                1, {"model": "unknown", "provider": "vertexai"}
            )

    def test_wrapped_generate_with_args_and_kwargs(self):
        """Test that wrapped generate_content handles both args and kwargs properly."""
        # Create mock vertexai module
        mock_vertexai = MagicMock()
        mock_generative_model_class = MagicMock()
        original_generate = MagicMock(return_value="generated content")
        mock_generative_model_class.generate_content = original_generate
        mock_vertexai.GenerativeModel = mock_generative_model_class

        with patch.dict("sys.modules", {"vertexai.preview.generative_models": mock_vertexai}):
            instrumentor = VertexAIInstrumentor()
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

            # Create a mock instance with _model_name attribute
            mock_instance = MagicMock()
            mock_instance._model_name = "gemini-pro"

            # Call the wrapped generate_content method with args and kwargs
            result = mock_vertexai.GenerativeModel.generate_content(
                mock_instance, "Test prompt", temperature=0.7
            )

            # Assertions
            self.assertEqual(result, "generated content")

            # Verify original_generate was called with the instance, args, and kwargs
            original_generate.assert_called_once_with(mock_instance, "Test prompt", temperature=0.7)

    def test_extract_usage(self):
        """Test that _extract_usage returns None."""
        instrumentor = VertexAIInstrumentor()
        result = instrumentor._extract_usage("any_result")

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
