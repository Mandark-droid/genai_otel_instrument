"""Tests for MistralAI instrumentor (v1.0+ SDK)"""

import unittest
from unittest.mock import MagicMock, call, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.mistralai_instrumentor import MistralAIInstrumentor


class TestMistralAIInstrumentor(unittest.TestCase):
    """Tests for MistralAIInstrumentor (v1.0+ SDK)"""

    @patch("genai_otel.instrumentors.mistralai_instrumentor.logger")
    def test_instrument_with_mistralai_available(self, mock_logger):
        """Test that instrument works when MistralAI v1.0+ is available."""
        # Create mock mistralai module for v1.0+
        mock_mistralai = MagicMock()

        # Create a real class so we can check for instrumentation flag
        class MockMistral:
            pass

        mock_mistralai.Mistral = MockMistral

        # Mock Chat and Embeddings classes
        mock_chat = MagicMock()
        mock_chat.complete = MagicMock()
        mock_chat.stream = MagicMock()

        mock_embeddings = MagicMock()
        mock_embeddings.create = MagicMock()

        mock_mistralai.chat.Chat = mock_chat
        mock_mistralai.embeddings.Embeddings = mock_embeddings

        # Mock wrapt module since it's imported inside instrument()
        mock_wrapt = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "mistralai": mock_mistralai,
                "mistralai.chat": mock_mistralai.chat,
                "mistralai.embeddings": mock_mistralai.embeddings,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = MistralAIInstrumentor()
            config = OTelConfig()

            instrumentor.instrument(config)

            # Verify logger was called
            mock_logger.info.assert_called_with("MistralAI instrumentation enabled (v1.0+ SDK)")

    @patch("genai_otel.instrumentors.mistralai_instrumentor.logger")
    def test_instrument_with_mistralai_not_available(self, mock_logger):
        """Test that instrument handles missing MistralAI gracefully."""
        instrumentor = MistralAIInstrumentor()
        config = OTelConfig()

        # Mock import to fail
        with patch("builtins.__import__", side_effect=ImportError("No module named 'mistralai'")):
            instrumentor.instrument(config)

            mock_logger.warning.assert_called_with(
                "mistralai package not available, skipping instrumentation"
            )

    @patch("genai_otel.instrumentors.mistralai_instrumentor.logger")
    def test_instrument_with_exception(self, mock_logger):
        """Test that instrument handles exceptions during setup."""
        mock_mistralai = MagicMock()

        class MockMistral:
            pass

        # Make Mistral access raise an exception
        mock_mistralai.Mistral = MockMistral

        mock_wrapt = MagicMock()

        with patch.dict("sys.modules", {"mistralai": mock_mistralai, "wrapt": mock_wrapt}):
            instrumentor = MistralAIInstrumentor()
            config = OTelConfig()
            config.fail_on_error = False  # Should not raise

            # Make _wrap_mistral_methods raise an exception
            with patch.object(
                instrumentor, "_wrap_mistral_methods", side_effect=RuntimeError("Setup error")
            ):
                instrumentor.instrument(config)

            # Verify error was logged
            mock_logger.error.assert_called_once()
            assert "Failed to instrument mistralai" in str(mock_logger.error.call_args)

    def test_extract_usage_with_usage_object(self):
        """Test _extract_usage with a valid usage object."""
        instrumentor = MistralAIInstrumentor()

        # Create a mock result with usage
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150

        mock_result = MagicMock()
        mock_result.usage = mock_usage

        usage = instrumentor._extract_usage(mock_result)

        self.assertEqual(usage["prompt_tokens"], 100)
        self.assertEqual(usage["completion_tokens"], 50)
        self.assertEqual(usage["total_tokens"], 150)

    def test_extract_usage_without_usage_attribute(self):
        """Test _extract_usage when result has no usage attribute."""
        instrumentor = MistralAIInstrumentor()

        mock_result = MagicMock(spec=[])  # No usage attribute

        usage = instrumentor._extract_usage(mock_result)

        self.assertIsNone(usage)


if __name__ == "__main__":
    unittest.main()
