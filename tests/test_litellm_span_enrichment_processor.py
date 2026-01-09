"""Tests for LiteLLM span enrichment processor."""

import json
import unittest
from unittest.mock import MagicMock, Mock

from genai_otel.litellm_span_enrichment_processor import LiteLLMSpanEnrichmentProcessor


class TestLiteLLMSpanEnrichmentProcessor(unittest.TestCase):
    """Tests for LiteLLMSpanEnrichmentProcessor"""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = LiteLLMSpanEnrichmentProcessor()

    def test_init(self):
        """Test processor initialization."""
        processor = LiteLLMSpanEnrichmentProcessor()
        self.assertIsNotNone(processor)

    def test_on_start_is_noop(self):
        """Test that on_start does nothing."""
        mock_span = MagicMock()
        # Should not raise
        self.processor.on_start(mock_span, None)

    def test_shutdown(self):
        """Test shutdown method."""
        # Should not raise
        self.processor.shutdown()

    def test_force_flush(self):
        """Test force_flush method."""
        result = self.processor.force_flush()
        self.assertTrue(result)

    def test_is_litellm_span_by_name(self):
        """Test detection of LiteLLM span by span name."""
        mock_span = MagicMock()
        mock_span.name = "litellm.completion"
        mock_span.attributes = {}

        result = self.processor._is_litellm_span(mock_span)
        self.assertTrue(result)

    def test_is_litellm_span_by_attributes(self):
        """Test detection of LiteLLM span by attributes."""
        mock_span = MagicMock()
        mock_span.name = "some.span"
        mock_span.attributes = {"llm.provider": "openai", "llm.model": "gpt-4"}

        result = self.processor._is_litellm_span(mock_span)
        self.assertTrue(result)

    def test_is_litellm_span_by_instrumentation_scope(self):
        """Test detection of LiteLLM span by instrumentation scope."""
        mock_span = MagicMock()
        mock_span.name = "completion"
        mock_span.attributes = {"gen_ai.system": "openai"}

        # Create mock instrumentation scope
        mock_scope = MagicMock()
        mock_scope.name = "openinference.instrumentation.litellm"
        mock_span.instrumentation_scope = mock_scope

        result = self.processor._is_litellm_span(mock_span)
        self.assertTrue(result)

    def test_is_not_litellm_span(self):
        """Test that non-LiteLLM spans are not detected."""
        mock_span = MagicMock()
        mock_span.name = "openai.chat.completion"
        mock_span.attributes = {"gen_ai.system": "openai"}
        mock_span.instrumentation_scope = None

        result = self.processor._is_litellm_span(mock_span)
        self.assertFalse(result)

    def test_extract_request_content_from_input_messages(self):
        """Test extraction of request content from llm.input_messages."""
        mock_span = MagicMock()
        messages = [{"role": "user", "content": "What is AI?"}]
        mock_span.attributes = {"llm.input_messages": json.dumps(messages)}

        result = self.processor._extract_request_content(mock_span)

        self.assertIsNotNone(result)
        self.assertIn("user", result)
        self.assertIn("What is AI?", result)

    def test_extract_request_content_from_input_value(self):
        """Test extraction of request content from input.value."""
        mock_span = MagicMock()
        mock_span.attributes = {"input.value": "Tell me about machine learning"}

        result = self.processor._extract_request_content(mock_span)

        self.assertIsNotNone(result)
        self.assertIn("user", result)
        self.assertIn("machine learning", result)

    def test_extract_request_content_from_prompts(self):
        """Test extraction of request content from llm.prompts."""
        mock_span = MagicMock()
        prompts = ["Explain quantum computing"]
        mock_span.attributes = {"llm.prompts": json.dumps(prompts)}

        result = self.processor._extract_request_content(mock_span)

        self.assertIsNotNone(result)
        self.assertIn("quantum computing", result)

    def test_extract_request_content_truncation(self):
        """Test that request content is truncated to 200 chars."""
        mock_span = MagicMock()
        long_content = "a" * 300
        messages = [{"role": "user", "content": long_content}]
        mock_span.attributes = {"llm.input_messages": json.dumps(messages)}

        result = self.processor._extract_request_content(mock_span)

        self.assertIsNotNone(result)
        # Should be truncated to 200 chars
        self.assertLessEqual(len(result), 200)

    def test_extract_request_content_no_messages(self):
        """Test extraction returns None when no messages found."""
        mock_span = MagicMock()
        mock_span.attributes = {}

        result = self.processor._extract_request_content(mock_span)

        self.assertIsNone(result)

    def test_extract_response_content_from_output_messages(self):
        """Test extraction of response content from llm.output_messages."""
        mock_span = MagicMock()
        messages = [{"message": {"content": "AI is artificial intelligence"}}]
        mock_span.attributes = {"llm.output_messages": json.dumps(messages)}

        result = self.processor._extract_response_content(mock_span)

        self.assertIsNotNone(result)
        self.assertEqual(result, "AI is artificial intelligence")

    def test_extract_response_content_from_output_messages_content_field(self):
        """Test extraction when content is at top level of message."""
        mock_span = MagicMock()
        messages = [{"content": "This is the response"}]
        mock_span.attributes = {"llm.output_messages": json.dumps(messages)}

        result = self.processor._extract_response_content(mock_span)

        self.assertIsNotNone(result)
        self.assertEqual(result, "This is the response")

    def test_extract_response_content_from_output_messages_string(self):
        """Test extraction when output message is a string."""
        mock_span = MagicMock()
        messages = ["Direct string response"]
        mock_span.attributes = {"llm.output_messages": json.dumps(messages)}

        result = self.processor._extract_response_content(mock_span)

        self.assertIsNotNone(result)
        self.assertEqual(result, "Direct string response")

    def test_extract_response_content_from_output_value(self):
        """Test extraction of response content from output.value."""
        mock_span = MagicMock()
        mock_span.attributes = {"output.value": "This is the output"}

        result = self.processor._extract_response_content(mock_span)

        self.assertIsNotNone(result)
        self.assertEqual(result, "This is the output")

    def test_extract_response_content_no_output(self):
        """Test extraction returns None when no output found."""
        mock_span = MagicMock()
        mock_span.attributes = {}

        result = self.processor._extract_response_content(mock_span)

        self.assertIsNone(result)

    def test_has_attribute_true(self):
        """Test checking if span has attribute."""
        mock_span = MagicMock()
        mock_span.attributes = {"gen_ai.response": "test"}

        result = self.processor._has_attribute(mock_span, "gen_ai.response")

        self.assertTrue(result)

    def test_has_attribute_false(self):
        """Test checking if span doesn't have attribute."""
        mock_span = MagicMock()
        mock_span.attributes = {}

        result = self.processor._has_attribute(mock_span, "gen_ai.response")

        self.assertFalse(result)

    def test_set_attribute_on_span(self):
        """Test setting attribute on Span instance."""
        from opentelemetry.sdk.trace import Span

        mock_span = MagicMock(spec=Span)
        mock_span.set_attribute = MagicMock()

        self.processor._set_attribute(mock_span, "test.key", "test.value")

        mock_span.set_attribute.assert_called_once_with("test.key", "test.value")

    def test_set_attribute_on_readable_span_with_private_attributes(self):
        """Test setting attribute on ReadableSpan with _attributes."""
        mock_span = MagicMock()
        mock_span._attributes = {}

        self.processor._set_attribute(mock_span, "test.key", "test.value")

        self.assertEqual(mock_span._attributes["test.key"], "test.value")

    def test_set_attribute_on_readable_span_with_public_attributes(self):
        """Test setting attribute on ReadableSpan with public attributes dict."""
        mock_span = MagicMock()
        # Remove _attributes to test fallback
        if hasattr(mock_span, "_attributes"):
            delattr(mock_span, "_attributes")
        mock_span.attributes = {}

        self.processor._set_attribute(mock_span, "test.key", "test.value")

        self.assertEqual(mock_span.attributes["test.key"], "test.value")

    def test_on_end_enriches_litellm_span(self):
        """Test that on_end enriches LiteLLM spans with evaluation attributes."""
        mock_span = MagicMock()
        mock_span.name = "litellm.completion"
        mock_span.attributes = {
            "llm.input_messages": json.dumps([{"role": "user", "content": "test prompt"}]),
            "llm.output_messages": json.dumps([{"content": "test response"}]),
        }
        mock_span._attributes = mock_span.attributes.copy()

        self.processor.on_end(mock_span)

        # Verify that evaluation attributes were added
        self.assertIn("gen_ai.request.first_message", mock_span._attributes)
        self.assertIn("gen_ai.response", mock_span._attributes)

    def test_on_end_skips_non_litellm_span(self):
        """Test that on_end skips non-LiteLLM spans."""
        mock_span = MagicMock()
        mock_span.name = "openai.chat.completion"
        mock_span.attributes = {
            "gen_ai.system": "openai",
            "llm.input_messages": json.dumps([{"role": "user", "content": "test"}]),
        }
        mock_span._attributes = mock_span.attributes.copy()
        mock_span.instrumentation_scope = None

        initial_attrs = mock_span._attributes.copy()
        self.processor.on_end(mock_span)

        # Attributes should not be modified
        self.assertNotIn("gen_ai.request.first_message", mock_span._attributes)

    def test_on_end_skips_if_attributes_already_present(self):
        """Test that on_end doesn't overwrite existing evaluation attributes."""
        mock_span = MagicMock()
        mock_span.name = "litellm.completion"
        mock_span.attributes = {
            "gen_ai.request.first_message": "existing request",
            "gen_ai.response": "existing response",
            "llm.input_messages": json.dumps([{"role": "user", "content": "new prompt"}]),
        }
        mock_span._attributes = mock_span.attributes.copy()

        self.processor.on_end(mock_span)

        # Should not overwrite existing attributes
        self.assertEqual(mock_span._attributes["gen_ai.request.first_message"], "existing request")
        self.assertEqual(mock_span._attributes["gen_ai.response"], "existing response")

    def test_on_end_handles_exception_gracefully(self):
        """Test that on_end handles exceptions gracefully."""
        mock_span = MagicMock()
        mock_span.name = "litellm.completion"
        # Create attributes that will cause an exception
        mock_span.attributes = {"llm.input_messages": "invalid json"}

        # Should not raise exception
        try:
            self.processor.on_end(mock_span)
        except Exception as e:
            self.fail(f"on_end raised exception: {e}")

    def test_integration_full_enrichment_flow(self):
        """Integration test for complete span enrichment flow."""
        # Create a realistic LiteLLM span
        mock_span = MagicMock()
        mock_span.name = "litellm.chat.completion"

        # Set up attributes as OpenInference would
        request_messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is machine learning?"},
        ]
        response_messages = [
            {
                "message": {
                    "role": "assistant",
                    "content": "Machine learning is a subset of artificial intelligence.",
                }
            }
        ]

        mock_span.attributes = {
            "llm.provider": "openai",
            "llm.model": "gpt-4",
            "llm.input_messages": json.dumps(request_messages),
            "llm.output_messages": json.dumps(response_messages),
        }
        mock_span._attributes = mock_span.attributes.copy()

        # Set up instrumentation scope
        mock_scope = MagicMock()
        mock_scope.name = "openinference.instrumentation.litellm"
        mock_span.instrumentation_scope = mock_scope

        # Process the span
        self.processor.on_end(mock_span)

        # Verify enrichment
        self.assertIn("gen_ai.request.first_message", mock_span._attributes)
        self.assertIn("gen_ai.response", mock_span._attributes)

        # Verify content accuracy
        request_attr = mock_span._attributes["gen_ai.request.first_message"]
        self.assertIn("system", request_attr)
        self.assertIn("helpful assistant", request_attr)

        response_attr = mock_span._attributes["gen_ai.response"]
        self.assertEqual(response_attr, "Machine learning is a subset of artificial intelligence.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
