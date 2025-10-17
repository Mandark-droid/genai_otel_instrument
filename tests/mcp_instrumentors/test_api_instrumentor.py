import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.mcp_instrumentors.api_instrumentor import APIInstrumentor


class TestAPIInstrumentor(unittest.TestCase):
    """Tests for APIInstrumentor"""

    def setUp(self):
        self.config = OTelConfig()

    def test_init(self):
        """Test that APIInstrumentor initializes correctly"""
        instrumentor = APIInstrumentor(self.config)
        self.assertEqual(instrumentor.config, self.config)

    @patch("genai_otel.mcp_instrumentors.api_instrumentor.logger")
    @patch("genai_otel.mcp_instrumentors.api_instrumentor.wrapt")
    def test_instrument_httpx_success(self, mock_wrapt, mock_logger):
        """Test successful instrumentation of httpx"""
        instrumentor = APIInstrumentor(self.config)
        instrumentor.instrument(self.config)

        # Verify httpx was wrapped
        mock_wrapt.wrap_function_wrapper.assert_called_once()
        mock_logger.info.assert_called_with("httpx library instrumented for API calls.")

    @patch("genai_otel.mcp_instrumentors.api_instrumentor.logger")
    @patch("genai_otel.mcp_instrumentors.api_instrumentor.wrapt")
    def test_instrument_httpx_import_error(self, mock_wrapt, mock_logger):
        """Test handling of httpx ImportError"""
        mock_wrapt.wrap_function_wrapper.side_effect = ImportError("No httpx module")

        instrumentor = APIInstrumentor(self.config)
        instrumentor.instrument(self.config)

        # Verify debug log for import error
        mock_logger.debug.assert_called_with("httpx library not found, skipping instrumentation.")

    @patch("genai_otel.mcp_instrumentors.api_instrumentor.logger")
    @patch("genai_otel.mcp_instrumentors.api_instrumentor.wrapt")
    def test_instrument_httpx_exception_fail_on_error_false(self, mock_wrapt, mock_logger):
        """Test handling of exceptions when fail_on_error is False"""
        mock_wrapt.wrap_function_wrapper.side_effect = RuntimeError("Wrapping failed")

        instrumentor = APIInstrumentor(self.config)
        self.config.fail_on_error = False
        instrumentor.instrument(self.config)

        # Verify error was logged
        mock_logger.error.assert_called()

    @patch("genai_otel.mcp_instrumentors.api_instrumentor.logger")
    @patch("genai_otel.mcp_instrumentors.api_instrumentor.wrapt")
    def test_instrument_httpx_exception_fail_on_error_true(self, mock_wrapt, mock_logger):
        """Test that exceptions are raised when fail_on_error is True"""
        mock_wrapt.wrap_function_wrapper.side_effect = RuntimeError("Wrapping failed")

        instrumentor = APIInstrumentor(self.config)
        self.config.fail_on_error = True

        with self.assertRaises(RuntimeError) as context:
            instrumentor.instrument(self.config)

        self.assertEqual(str(context.exception), "Wrapping failed")

    @patch("genai_otel.mcp_instrumentors.api_instrumentor.logger")
    def test_requests_library_warning(self, mock_logger):
        """Test that requests library warning is logged"""
        with patch("genai_otel.mcp_instrumentors.api_instrumentor.wrapt"):
            instrumentor = APIInstrumentor(self.config)
            instrumentor.instrument(self.config)

            # Verify warning about requests library
            mock_logger.warning.assert_called_with(
                "requests library instrumentation disabled to prevent OTLP exporter conflicts"
            )

    def test_wrap_api_call_with_url_in_kwargs(self):
        """Test _wrap_api_call with URL in kwargs"""
        instrumentor = APIInstrumentor(self.config)
        instrumentor.create_span_wrapper = MagicMock(return_value=MagicMock())

        wrapped = MagicMock()
        instance = None
        args = ()
        kwargs = {"method": "GET", "url": "https://api.openai.com/v1/chat"}

        instrumentor._wrap_api_call(wrapped, instance, args, kwargs)

        # Verify create_span_wrapper was called with correct span name
        call_args = instrumentor.create_span_wrapper.call_args
        self.assertEqual(call_args[1]["span_name"], "api.call.get.api.openai.com")

    def test_wrap_api_call_with_url_in_args(self):
        """Test _wrap_api_call with URL in args"""
        instrumentor = APIInstrumentor(self.config)
        instrumentor.create_span_wrapper = MagicMock(return_value=MagicMock())

        wrapped = MagicMock()
        instance = None
        args = ("POST", "https://api.anthropic.com/v1/messages")
        kwargs = {}

        instrumentor._wrap_api_call(wrapped, instance, args, kwargs)

        # Verify create_span_wrapper was called with correct span name
        call_args = instrumentor.create_span_wrapper.call_args
        self.assertEqual(call_args[1]["span_name"], "api.call.post.api.anthropic.com")

    def test_wrap_api_call_without_url(self):
        """Test _wrap_api_call without URL"""
        instrumentor = APIInstrumentor(self.config)
        instrumentor.create_span_wrapper = MagicMock(return_value=MagicMock())

        wrapped = MagicMock()
        instance = None
        args = ("GET",)
        kwargs = {}

        instrumentor._wrap_api_call(wrapped, instance, args, kwargs)

        # Verify create_span_wrapper was called with default span name
        call_args = instrumentor.create_span_wrapper.call_args
        self.assertEqual(call_args[1]["span_name"], "api.call.get")

    def test_wrap_api_call_with_invalid_url(self):
        """Test _wrap_api_call with invalid URL that fails parsing"""
        instrumentor = APIInstrumentor(self.config)
        instrumentor.create_span_wrapper = MagicMock(return_value=MagicMock())

        wrapped = MagicMock()
        instance = None
        args = ()
        kwargs = {"method": "GET", "url": "not a valid url"}

        # Should not raise, just use default span name
        instrumentor._wrap_api_call(wrapped, instance, args, kwargs)

        call_args = instrumentor.create_span_wrapper.call_args
        # Should have a span name even if URL parsing failed
        self.assertIsNotNone(call_args[1]["span_name"])

    def test_extract_api_attributes_with_openai_url(self):
        """Test _extract_api_attributes with OpenAI URL"""
        instrumentor = APIInstrumentor(self.config)

        args = ()
        kwargs = {"method": "POST", "url": "https://api.openai.com/v1/chat/completions"}

        attrs = instrumentor._extract_api_attributes(None, args, kwargs)

        self.assertEqual(attrs["net.peer.name"], "api.openai.com")
        self.assertEqual(attrs["url.full"], "https://api.openai.com/v1/chat/completions")
        self.assertEqual(attrs["http.method"], "POST")
        self.assertEqual(attrs["gen_ai.system"], "openai")

    def test_extract_api_attributes_with_anthropic_url(self):
        """Test _extract_api_attributes with Anthropic URL"""
        instrumentor = APIInstrumentor(self.config)

        args = ()
        kwargs = {"method": "POST", "url": "https://api.anthropic.com/v1/messages"}

        attrs = instrumentor._extract_api_attributes(None, args, kwargs)

        self.assertEqual(attrs["gen_ai.system"], "anthropic")
        self.assertEqual(attrs["net.peer.name"], "api.anthropic.com")

    def test_extract_api_attributes_with_google_url(self):
        """Test _extract_api_attributes with Google URL"""
        instrumentor = APIInstrumentor(self.config)

        args = ()
        kwargs = {"method": "POST", "url": "https://generativelanguage.google.com/v1/models"}

        attrs = instrumentor._extract_api_attributes(None, args, kwargs)

        self.assertEqual(attrs["gen_ai.system"], "google")

    def test_extract_api_attributes_with_method_in_args(self):
        """Test _extract_api_attributes with method in args"""
        instrumentor = APIInstrumentor(self.config)

        args = ("GET", "https://example.com/api")
        kwargs = {}

        attrs = instrumentor._extract_api_attributes(None, args, kwargs)

        self.assertEqual(attrs["http.method"], "GET")
        self.assertEqual(attrs["url.full"], "https://example.com/api")

    def test_extract_api_attributes_without_url(self):
        """Test _extract_api_attributes without URL"""
        instrumentor = APIInstrumentor(self.config)

        args = ("GET",)
        kwargs = {}

        attrs = instrumentor._extract_api_attributes(None, args, kwargs)

        # Should return empty dict or dict without url-related attributes
        self.assertNotIn("url.full", attrs)
        self.assertNotIn("net.peer.name", attrs)

    @patch("genai_otel.mcp_instrumentors.api_instrumentor.logger")
    def test_extract_api_attributes_with_invalid_url(self, mock_logger):
        """Test _extract_api_attributes with invalid URL"""
        instrumentor = APIInstrumentor(self.config)

        # Create a mock that will raise an exception during URL parsing
        with patch("genai_otel.mcp_instrumentors.api_instrumentor.urlparse") as mock_urlparse:
            mock_urlparse.side_effect = Exception("URL parsing failed")

            args = ()
            kwargs = {"method": "GET", "url": "invalid://url"}

            attrs = instrumentor._extract_api_attributes(None, args, kwargs)

            # Should log warning
            mock_logger.warning.assert_called()

    def test_extract_usage(self):
        """Test that _extract_usage returns None for API calls"""
        instrumentor = APIInstrumentor(self.config)

        result = MagicMock()
        usage = instrumentor._extract_usage(result)

        self.assertIsNone(usage)


if __name__ == "__main__":
    unittest.main(verbosity=2)
