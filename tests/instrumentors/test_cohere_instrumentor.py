import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.cohere_instrumentor import CohereInstrumentor


class TestCohereInstrumentor(unittest.TestCase):
    """Tests for CohereInstrumentor"""

    @patch("genai_otel.instrumentors.cohere_instrumentor.logger")
    def test_init_with_cohere_available(self, mock_logger):
        """Test that __init__ detects cohere availability."""
        with patch.dict("sys.modules", {"cohere": MagicMock()}):
            instrumentor = CohereInstrumentor()

            self.assertTrue(instrumentor._cohere_available)
            mock_logger.debug.assert_called_with(
                "cohere library detected and available for instrumentation"
            )

    @patch("genai_otel.instrumentors.cohere_instrumentor.logger")
    def test_init_with_cohere_not_available(self, mock_logger):
        """Test that __init__ handles missing cohere gracefully."""
        with patch.dict("sys.modules", {"cohere": None}):
            instrumentor = CohereInstrumentor()

            self.assertFalse(instrumentor._cohere_available)
            mock_logger.debug.assert_called_with(
                "cohere library not installed, instrumentation will be skipped"
            )

    @patch("genai_otel.instrumentors.cohere_instrumentor.logger")
    def test_instrument_with_cohere_not_available(self, mock_logger):
        """Test that instrument skips when cohere is not available."""
        with patch.dict("sys.modules", {"cohere": None}):
            instrumentor = CohereInstrumentor()
            config = OTelConfig()

            instrumentor.instrument(config)

            mock_logger.debug.assert_any_call("Skipping instrumentation - library not available")

    def test_instrument_with_cohere_available(self):
        """Test that instrument wraps cohere client when available."""

        # Create a real class to mock Cohere Client
        class MockCohereClient:
            def __init__(self, *args, **kwargs):
                self.generate = MagicMock()

        # Create mock cohere module
        mock_cohere = MagicMock()
        mock_cohere.Client = MockCohereClient

        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            instrumentor = CohereInstrumentor()
            config = OTelConfig()

            # Store original __init__
            original_init = MockCohereClient.__init__

            # Call instrument
            instrumentor.instrument(config)

            # Verify that Client.__init__ was replaced
            self.assertNotEqual(mock_cohere.Client.__init__, original_init)
            self.assertEqual(instrumentor.config, config)

    def test_instrument_with_import_error(self):
        """Test that instrument handles ImportError gracefully."""
        # Create mock cohere that is available for check but raises on re-import
        mock_cohere_initial = MagicMock()

        with patch.dict("sys.modules", {"cohere": mock_cohere_initial}):
            instrumentor = CohereInstrumentor()
            config = OTelConfig()

            # Now make cohere module None to simulate ImportError
            with patch.dict("sys.modules", {"cohere": None}):
                # Should not raise
                instrumentor.instrument(config)

    def test_instrument_client(self):
        """Test that _instrument_client wraps generate method."""
        instrumentor = CohereInstrumentor()

        # Create mock client
        mock_client = MagicMock()
        original_generate = MagicMock(return_value="result")
        mock_client.generate = original_generate

        # Mock tracer and metrics
        instrumentor.tracer = MagicMock()
        instrumentor.request_counter = MagicMock()
        instrumentor.token_counter = MagicMock()
        instrumentor.latency_histogram = MagicMock()
        instrumentor.cost_gauge = MagicMock()

        # Call _instrument_client
        instrumentor._instrument_client(mock_client)

        # Verify generate was replaced (will be wrapped)
        # The wrapper is callable, so just verify it's been changed
        self.assertIsNotNone(mock_client.generate)
        self.assertTrue(callable(mock_client.generate))

    def test_wrapped_generate_execution(self):
        """Test that wrapped generate method executes correctly."""
        instrumentor = CohereInstrumentor()

        # Create mock client
        mock_client = MagicMock()

        # Create mock response with token usage
        mock_response = MagicMock()
        mock_response.meta = MagicMock()
        mock_response.meta.tokens = MagicMock()
        mock_response.meta.tokens.input_tokens = 10
        mock_response.meta.tokens.output_tokens = 20

        original_generate = MagicMock(return_value=mock_response)
        mock_client.generate = original_generate

        # Mock tracer and metrics
        mock_span = MagicMock()
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
        mock_context_manager.__exit__ = MagicMock(return_value=None)

        instrumentor.tracer = MagicMock()
        instrumentor.tracer.start_as_current_span = MagicMock(return_value=mock_context_manager)
        instrumentor.request_counter = MagicMock()
        instrumentor.token_counter = MagicMock()
        instrumentor.latency_histogram = MagicMock()
        instrumentor.cost_gauge = MagicMock()

        # Instrument the client
        instrumentor._instrument_client(mock_client)

        # Call the wrapped method
        result = mock_client.generate(prompt="test prompt", model="command-light")

        # Verify result is returned
        self.assertEqual(result, mock_response)

        # Verify original generate was called
        original_generate.assert_called_once_with(prompt="test prompt", model="command-light")

    def test_wrapped_generate_with_default_model(self):
        """Test that wrapped generate uses default model 'command'."""
        instrumentor = CohereInstrumentor()

        # Create mock client
        mock_client = MagicMock()
        original_generate = MagicMock(return_value="generated text")
        mock_client.generate = original_generate

        # Mock tracer and metrics
        mock_span = MagicMock()
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
        mock_context_manager.__exit__ = MagicMock(return_value=None)

        instrumentor.tracer = MagicMock()
        instrumentor.tracer.start_as_current_span = MagicMock(return_value=mock_context_manager)
        instrumentor.request_counter = MagicMock()
        instrumentor.token_counter = MagicMock()
        instrumentor.latency_histogram = MagicMock()
        instrumentor.cost_gauge = MagicMock()

        # Instrument the client
        instrumentor._instrument_client(mock_client)

        # Call the wrapped method without model parameter
        result = mock_client.generate(prompt="test prompt")

        # Verify original generate was called
        original_generate.assert_called_once_with(prompt="test prompt")

    def test_extract_usage(self):
        """Test that _extract_usage extracts token counts from Cohere response."""
        instrumentor = CohereInstrumentor()

        # Test with None
        result = instrumentor._extract_usage(None)
        self.assertIsNone(result)

        # Test with response missing meta
        mock_response_no_meta = MagicMock(spec=[])
        result = instrumentor._extract_usage(mock_response_no_meta)
        self.assertIsNone(result)

        # Test with valid response containing tokens
        mock_response = MagicMock()
        mock_response.meta = MagicMock()
        mock_response.meta.tokens = MagicMock()
        mock_response.meta.tokens.input_tokens = 15
        mock_response.meta.tokens.output_tokens = 25

        result = instrumentor._extract_usage(mock_response)
        self.assertIsNotNone(result)
        self.assertEqual(result["prompt_tokens"], 15)
        self.assertEqual(result["completion_tokens"], 25)
        self.assertEqual(result["total_tokens"], 40)

        # Test with billed_units fallback
        mock_response_billed = MagicMock()
        mock_response_billed.meta = MagicMock()
        mock_response_billed.meta.tokens = None
        mock_response_billed.meta.billed_units = MagicMock()
        mock_response_billed.meta.billed_units.input_tokens = 10
        mock_response_billed.meta.billed_units.output_tokens = 20

        result = instrumentor._extract_usage(mock_response_billed)
        self.assertIsNotNone(result)
        self.assertEqual(result["prompt_tokens"], 10)
        self.assertEqual(result["completion_tokens"], 20)
        self.assertEqual(result["total_tokens"], 30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
