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

            mock_logger.debug.assert_any_call(
                "Skipping instrumentation - library not available"
            )

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

        # Call _instrument_client
        instrumentor._instrument_client(mock_client)

        # Verify generate was replaced
        self.assertNotEqual(mock_client.generate, original_generate)

    def test_wrapped_generate_execution(self):
        """Test that wrapped generate method executes correctly."""
        instrumentor = CohereInstrumentor()

        # Create mock client
        mock_client = MagicMock()
        original_generate = MagicMock(return_value="generated text")
        mock_client.generate = original_generate

        # Mock tracer and metrics
        mock_span = MagicMock()
        instrumentor.tracer = MagicMock()
        instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        instrumentor.request_counter = MagicMock()

        # Instrument the client
        instrumentor._instrument_client(mock_client)

        # Call the wrapped method
        result = mock_client.generate(prompt="test prompt", model="command-light")

        # Verify tracer was called
        instrumentor.tracer.start_as_current_span.assert_called_once_with("cohere.generate")

        # Verify span attributes were set
        mock_span.set_attribute.assert_any_call("gen_ai.system", "cohere")
        mock_span.set_attribute.assert_any_call("gen_ai.request.model", "command-light")

        # Verify metrics were recorded
        instrumentor.request_counter.add.assert_called_once_with(
            1, {"model": "command-light", "provider": "cohere"}
        )

        # Verify original generate was called
        original_generate.assert_called_once_with(prompt="test prompt", model="command-light")

        # Verify result
        self.assertEqual(result, "generated text")

    def test_wrapped_generate_with_default_model(self):
        """Test that wrapped generate uses default model 'command'."""
        instrumentor = CohereInstrumentor()

        # Create mock client
        mock_client = MagicMock()
        original_generate = MagicMock(return_value="generated text")
        mock_client.generate = original_generate

        # Mock tracer and metrics
        mock_span = MagicMock()
        instrumentor.tracer = MagicMock()
        instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        instrumentor.request_counter = MagicMock()

        # Instrument the client
        instrumentor._instrument_client(mock_client)

        # Call the wrapped method without model parameter
        result = mock_client.generate(prompt="test prompt")

        # Verify span attributes were set with default model
        mock_span.set_attribute.assert_any_call("gen_ai.request.model", "command")

        # Verify metrics were recorded with default model
        instrumentor.request_counter.add.assert_called_once_with(
            1, {"model": "command", "provider": "cohere"}
        )

    def test_extract_usage(self):
        """Test that _extract_usage returns None."""
        instrumentor = CohereInstrumentor()

        result = instrumentor._extract_usage(MagicMock())

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
