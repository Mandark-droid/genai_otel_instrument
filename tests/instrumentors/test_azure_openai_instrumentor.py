import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.azure_openai_instrumentor import AzureOpenAIInstrumentor


class TestAzureOpenAIInstrumentor(unittest.TestCase):
    """Tests for AzureOpenAIInstrumentor"""

    def test_init_with_azure_openai_available(self):
        """Test that __init__ detects azure.ai.openai availability."""
        with patch.dict(
            "sys.modules",
            {"azure": MagicMock(), "azure.ai": MagicMock(), "azure.ai.openai": MagicMock()},
        ):
            instrumentor = AzureOpenAIInstrumentor()
            self.assertTrue(instrumentor._azure_openai_available)

    def test_init_with_azure_openai_not_available(self):
        """Test that __init__ handles missing azure.ai.openai gracefully."""
        with patch.dict("sys.modules", {"azure": None, "azure.ai": None, "azure.ai.openai": None}):
            instrumentor = AzureOpenAIInstrumentor()
            self.assertFalse(instrumentor._azure_openai_available)

    def test_instrument_with_azure_openai_not_available(self):
        """Test that instrument skips when azure.ai.openai is not available."""
        with patch.dict("sys.modules", {"azure": None, "azure.ai": None, "azure.ai.openai": None}):
            instrumentor = AzureOpenAIInstrumentor()
            config = OTelConfig()

            # Should not raise, just skip instrumentation
            instrumentor.instrument(config)

            # Config is always set at line 37 in the source, even if library is not available
            self.assertEqual(instrumentor.config, config)

    def test_instrument_with_azure_openai_available(self):
        """Test that instrument wraps OpenAIClient.complete when available."""

        # Create a real OpenAIClient class for testing
        class MockOpenAIClient:
            def complete(self, *args, **kwargs):
                return MagicMock()

        # Create mock azure.ai.openai module
        mock_azure_openai = MagicMock()
        mock_azure_openai.OpenAIClient = MockOpenAIClient

        with patch.dict(
            "sys.modules",
            {"azure": MagicMock(), "azure.ai": MagicMock(), "azure.ai.openai": mock_azure_openai},
        ):
            instrumentor = AzureOpenAIInstrumentor()
            config = OTelConfig()

            # Store original complete method
            original_complete = MockOpenAIClient.complete

            # Call instrument
            instrumentor.instrument(config)

            # Verify config was set
            self.assertEqual(instrumentor.config, config)

            # Verify complete method was replaced
            self.assertNotEqual(MockOpenAIClient.complete, original_complete)

    def test_instrument_with_import_error(self):
        """Test that instrument handles ImportError gracefully."""
        # Create a mock azure.ai.openai module
        mock_azure_openai = MagicMock()

        # Make the OpenAIClient import raise ImportError
        def import_side_effect(name, *args, **kwargs):
            if name == "azure.ai.openai":
                raise ImportError("Test import error")
            return MagicMock()

        with patch.dict(
            "sys.modules",
            {"azure": MagicMock(), "azure.ai": MagicMock(), "azure.ai.openai": mock_azure_openai},
        ):
            instrumentor = AzureOpenAIInstrumentor()
            config = OTelConfig()

            # Mock __import__ to raise ImportError for the from import statement
            with patch("builtins.__import__", side_effect=import_side_effect):
                # Should not raise, just pass silently (caught by except ImportError on line 58)
                instrumentor.instrument(config)

    def test_wrapped_complete_creates_span(self):
        """Test that the wrapped complete method creates a span and records metrics."""

        # Create a real OpenAIClient class for testing
        class MockOpenAIClient:
            def complete(self, *args, **kwargs):
                # Create mock result with usage
                result = MagicMock()
                result.usage = MagicMock()
                result.usage.prompt_tokens = 10
                result.usage.completion_tokens = 20
                result.usage.total_tokens = 30
                return result

        # Create mock azure.ai.openai module
        mock_azure_openai = MagicMock()
        mock_azure_openai.OpenAIClient = MockOpenAIClient

        with patch.dict(
            "sys.modules",
            {"azure": MagicMock(), "azure.ai": MagicMock(), "azure.ai.openai": mock_azure_openai},
        ):
            instrumentor = AzureOpenAIInstrumentor()
            config = OTelConfig()

            # Mock tracer and span
            mock_span = MagicMock()
            mock_tracer = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
            instrumentor.tracer = mock_tracer

            # Mock request_counter
            mock_counter = MagicMock()
            instrumentor.request_counter = mock_counter

            # Mock _record_result_metrics
            instrumentor._record_result_metrics = MagicMock()

            # Call instrument
            instrumentor.instrument(config)

            # Create client and call complete
            client = MockOpenAIClient()
            result = client.complete(model="gpt-4")

            # Verify span was created
            mock_tracer.start_as_current_span.assert_called_with("azure.openai.complete")

            # Verify span attributes were set
            mock_span.set_attribute.assert_any_call("gen_ai.system", "azure_openai")
            mock_span.set_attribute.assert_any_call("gen_ai.request.model", "gpt-4")

            # Verify request counter was incremented
            mock_counter.add.assert_called_once_with(
                1, {"model": "gpt-4", "provider": "azure_openai"}
            )

            # Verify _record_result_metrics was called
            instrumentor._record_result_metrics.assert_called_once()

    def test_wrapped_complete_with_unknown_model(self):
        """Test that wrapped complete handles missing model parameter."""

        # Create a real OpenAIClient class for testing
        class MockOpenAIClient:
            def complete(self, *args, **kwargs):
                result = MagicMock()
                result.usage = None
                return result

        # Create mock azure.ai.openai module
        mock_azure_openai = MagicMock()
        mock_azure_openai.OpenAIClient = MockOpenAIClient

        with patch.dict(
            "sys.modules",
            {"azure": MagicMock(), "azure.ai": MagicMock(), "azure.ai.openai": mock_azure_openai},
        ):
            instrumentor = AzureOpenAIInstrumentor()
            config = OTelConfig()

            # Mock tracer and span
            mock_span = MagicMock()
            mock_tracer = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
            instrumentor.tracer = mock_tracer

            # Mock request_counter
            mock_counter = MagicMock()
            instrumentor.request_counter = mock_counter

            # Mock _record_result_metrics
            instrumentor._record_result_metrics = MagicMock()

            # Call instrument
            instrumentor.instrument(config)

            # Create client and call complete without model
            client = MockOpenAIClient()
            result = client.complete()

            # Verify span attributes were set with "unknown" model
            mock_span.set_attribute.assert_any_call("gen_ai.request.model", "unknown")
            mock_counter.add.assert_called_once_with(
                1, {"model": "unknown", "provider": "azure_openai"}
            )

    def test_extract_usage_with_usage_object(self):
        """Test that _extract_usage extracts token counts from response."""
        with patch.dict(
            "sys.modules",
            {"azure": MagicMock(), "azure.ai": MagicMock(), "azure.ai.openai": MagicMock()},
        ):
            instrumentor = AzureOpenAIInstrumentor()

            # Create mock result with usage
            result = MagicMock()
            result.usage = MagicMock()
            result.usage.prompt_tokens = 15
            result.usage.completion_tokens = 25
            result.usage.total_tokens = 40

            usage = instrumentor._extract_usage(result)

            self.assertIsNotNone(usage)
            self.assertEqual(usage["prompt_tokens"], 15)
            self.assertEqual(usage["completion_tokens"], 25)
            self.assertEqual(usage["total_tokens"], 40)

    def test_extract_usage_without_usage_object(self):
        """Test that _extract_usage returns None when usage is missing."""
        with patch.dict(
            "sys.modules",
            {"azure": MagicMock(), "azure.ai": MagicMock(), "azure.ai.openai": MagicMock()},
        ):
            instrumentor = AzureOpenAIInstrumentor()

            # Create mock result without usage attribute
            result = MagicMock(spec=[])

            usage = instrumentor._extract_usage(result)

            self.assertIsNone(usage)

    def test_extract_usage_with_none_usage(self):
        """Test that _extract_usage returns None when usage is None."""
        with patch.dict(
            "sys.modules",
            {"azure": MagicMock(), "azure.ai": MagicMock(), "azure.ai.openai": MagicMock()},
        ):
            instrumentor = AzureOpenAIInstrumentor()

            # Create mock result with None usage
            result = MagicMock()
            result.usage = None

            usage = instrumentor._extract_usage(result)

            self.assertIsNone(usage)


if __name__ == "__main__":
    unittest.main(verbosity=2)
