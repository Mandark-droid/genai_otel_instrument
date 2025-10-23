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

            # Create mock tracer and metrics
            instrumentor.tracer = MagicMock()
            instrumentor.request_counter = MagicMock()
            instrumentor.token_counter = MagicMock()
            instrumentor.latency_histogram = MagicMock()
            instrumentor.cost_gauge = MagicMock()

            # Act
            instrumentor.instrument(config)

            # The generate_content method should now be wrapped (callable)
            self.assertIsNotNone(mock_vertexai.GenerativeModel.generate_content)
            self.assertTrue(callable(mock_vertexai.GenerativeModel.generate_content))

            # Create a mock instance with _model_name attribute
            mock_instance = MagicMock()
            mock_instance._model_name = "gemini-pro"

            # Call the wrapped generate_content method
            result = mock_vertexai.GenerativeModel.generate_content(mock_instance, "Test prompt")

            # Assertions
            self.assertEqual(result, "generated content")
            # Verify original function was called
            original_generate.assert_called_once_with(mock_instance, "Test prompt")

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

            # Create mock tracer and metrics
            instrumentor.tracer = MagicMock()
            instrumentor.request_counter = MagicMock()
            instrumentor.token_counter = MagicMock()
            instrumentor.latency_histogram = MagicMock()
            instrumentor.cost_gauge = MagicMock()

            # Act
            instrumentor.instrument(config)

            # Create a mock instance WITHOUT _model_name attribute
            mock_instance = MagicMock(spec=[])  # spec=[] means no attributes

            # Call the wrapped generate_content method
            result = mock_vertexai.GenerativeModel.generate_content(mock_instance, "Test prompt")

            # Assertions
            self.assertEqual(result, "generated content")
            # Verify original function was called
            original_generate.assert_called_once_with(mock_instance, "Test prompt")

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
        """Test that _extract_usage extracts token counts from Vertex AI response."""
        instrumentor = VertexAIInstrumentor()

        # Test with None
        result = instrumentor._extract_usage(None)
        self.assertIsNone(result)

        # Test with response missing usage_metadata
        mock_response_no_meta = MagicMock(spec=[])
        result = instrumentor._extract_usage(mock_response_no_meta)
        self.assertIsNone(result)

        # Test with valid response containing usage_metadata (snake_case Python style)
        mock_response = MagicMock()
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 50
        mock_response.usage_metadata.candidates_token_count = 150
        mock_response.usage_metadata.total_token_count = 200

        result = instrumentor._extract_usage(mock_response)
        self.assertIsNotNone(result)
        self.assertEqual(result["prompt_tokens"], 50)
        self.assertEqual(result["completion_tokens"], 150)
        self.assertEqual(result["total_tokens"], 200)

        # Test with camelCase (REST API style)
        mock_response_camel = MagicMock()
        mock_response_camel.usage_metadata = MagicMock()
        mock_response_camel.usage_metadata.prompt_token_count = None
        mock_response_camel.usage_metadata.candidates_token_count = None
        mock_response_camel.usage_metadata.total_token_count = None
        mock_response_camel.usage_metadata.promptTokenCount = 30
        mock_response_camel.usage_metadata.candidatesTokenCount = 70
        mock_response_camel.usage_metadata.totalTokenCount = 100

        result = instrumentor._extract_usage(mock_response_camel)
        self.assertIsNotNone(result)
        self.assertEqual(result["prompt_tokens"], 30)
        self.assertEqual(result["completion_tokens"], 70)
        self.assertEqual(result["total_tokens"], 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
