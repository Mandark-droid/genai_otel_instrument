"""Tests for OpenAI instrumentor."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.openai_instrumentor import OpenAIInstrumentor


class TestOpenAIInstrumentor:
    """Test cases for OpenAI instrumentor."""

    def test_init_without_openai(self):
        """Test initialization when OpenAI is not installed."""
        with patch("genai_otel.instrumentors.openai_instrumentor.logger") as mock_logger:
            with patch.dict("sys.modules", {"openai": None}):
                instrumentor = OpenAIInstrumentor()
                assert instrumentor._openai_available is False

    @patch("genai_otel.instrumentors.openai_instrumentor.logger")
    def test_instrument_without_openai(self, mock_logger):
        """Test instrumentation when OpenAI is not available."""
        instrumentor = OpenAIInstrumentor()
        instrumentor._openai_available = False

        config = OTelConfig()
        instrumentor.instrument(config)

        # Should log debug message and skip instrumentation
        assert not instrumentor._instrumented

    @pytest.mark.skipif(not pytest.importorskip("openai"), reason="OpenAI library not installed")
    def test_extract_openai_attributes(self):
        """Test extraction of attributes from OpenAI call."""
        instrumentor = OpenAIInstrumentor()

        kwargs = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        }

        attrs = instrumentor._extract_openai_attributes(None, [], kwargs)

        assert attrs["gen_ai.system"] == "openai"
        assert attrs["gen_ai.request.model"] == "gpt-4"
        assert attrs["gen_ai.request.message_count"] == 2
        assert "gen_ai.request.first_message" in attrs

    def test_extract_usage(self):
        """Test extraction of usage information."""
        instrumentor = OpenAIInstrumentor()

        # Mock response with usage
        mock_response = Mock()
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30

        usage = instrumentor._extract_usage(mock_response)

        assert usage is not None
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 20
        assert usage["total_tokens"] == 30

    def test_extract_usage_no_usage(self):
        """Test extraction when response has no usage."""
        instrumentor = OpenAIInstrumentor()

        mock_response = Mock(spec=[])
        usage = instrumentor._extract_usage(mock_response)

        assert usage is None

    def test_extract_attributes_empty_messages(self):
        """Test attribute extraction with empty messages."""
        instrumentor = OpenAIInstrumentor()

        kwargs = {"model": "gpt-3.5-turbo", "messages": []}

        attrs = instrumentor._extract_openai_attributes(None, [], kwargs)

        assert attrs["gen_ai.request.message_count"] == 0
        assert "gen_ai.request.first_message" not in attrs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
