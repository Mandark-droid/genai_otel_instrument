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
        """Test detection of LiteLLM span by attributes with OpenInference scope."""
        mock_span = MagicMock()
        mock_span.name = "some.span"
        mock_span.attributes = {"llm.provider": "openai", "llm.model": "gpt-4"}

        # Attributes alone are not enough - need scope name to avoid false positives
        mock_scope = MagicMock()
        mock_scope.name = "openinference.instrumentation.litellm"
        mock_span.instrumentation_scope = mock_scope

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
        """Test setting attribute on ReadableSpan via _attributes fallback.

        When set_attribute is not available (or raises), the processor
        falls back to writing directly to _attributes (BoundedAttributes).
        """
        mock_span = MagicMock()
        mock_span._attributes = {}
        # Make set_attribute raise to test fallback to _attributes
        mock_span.set_attribute.side_effect = AttributeError("not available")

        self.processor._set_attribute(mock_span, "test.key", "test.value")

        self.assertEqual(mock_span._attributes["test.key"], "test.value")

    def test_set_attribute_on_readable_span_with_public_attributes(self):
        """Test setting attribute via set_attribute when available."""
        mock_span = MagicMock()
        mock_span.set_attribute = MagicMock()

        self.processor._set_attribute(mock_span, "test.key", "test.value")

        mock_span.set_attribute.assert_called_once_with("test.key", "test.value")

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

        # Verify that evaluation attributes were set via set_attribute
        set_attr_calls = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}
        self.assertIn("gen_ai.request.first_message", set_attr_calls)
        self.assertIn("gen_ai.response", set_attr_calls)

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

        # Verify enrichment via set_attribute calls
        set_attr_calls = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}
        self.assertIn("gen_ai.request.first_message", set_attr_calls)
        self.assertIn("gen_ai.response", set_attr_calls)

        # Verify content accuracy
        request_attr = set_attr_calls["gen_ai.request.first_message"]
        self.assertIn("system", request_attr)
        self.assertIn("helpful assistant", request_attr)

        response_attr = set_attr_calls["gen_ai.response"]
        self.assertEqual(response_attr, "Machine learning is a subset of artificial intelligence.")

    def test_is_litellm_span_by_scope_name_primary(self):
        """Test that instrumentation scope name is the primary detection method."""
        mock_span = MagicMock()
        mock_span.name = "acompletion"  # Bare function name (no 'litellm' prefix)
        mock_span.attributes = {}  # No LLM-specific attributes

        # Set instrumentation scope to LiteLLM
        mock_scope = MagicMock()
        mock_scope.name = "openinference.instrumentation.litellm"
        mock_span.instrumentation_scope = mock_scope

        result = self.processor._is_litellm_span(mock_span)
        self.assertTrue(result)

    def test_is_litellm_span_by_bare_function_name(self):
        """Test detection of LiteLLM span by bare function name with OpenInference scope."""
        for span_name in ["acompletion", "completion", "atext_completion", "embedding"]:
            mock_span = MagicMock()
            mock_span.name = span_name
            mock_span.attributes = {}

            mock_scope = MagicMock()
            mock_scope.name = "openinference.instrumentation.litellm"
            mock_span.instrumentation_scope = mock_scope

            result = self.processor._is_litellm_span(mock_span)
            self.assertTrue(result, f"Failed to detect span with name: {span_name}")

    def test_is_litellm_span_by_model_name_attribute(self):
        """Test detection of LiteLLM span by llm.model_name (OpenInference convention)."""
        mock_span = MagicMock()
        mock_span.name = "acompletion"
        mock_span.attributes = {"llm.model_name": "openai/gpt-4o"}

        mock_scope = MagicMock()
        mock_scope.name = "openinference.instrumentation.litellm"
        mock_span.instrumentation_scope = mock_scope

        result = self.processor._is_litellm_span(mock_span)
        self.assertTrue(result)

    def test_is_not_litellm_span_bare_name_wrong_scope(self):
        """Test that bare function name with non-LiteLLM scope is not matched."""
        mock_span = MagicMock()
        mock_span.name = "acompletion"
        mock_span.attributes = {}

        mock_scope = MagicMock()
        mock_scope.name = "some.other.instrumentor"
        mock_span.instrumentation_scope = mock_scope

        result = self.processor._is_litellm_span(mock_span)
        self.assertFalse(result)

    def test_is_not_litellm_span_attributes_no_scope(self):
        """Test that LLM attributes alone without scope don't match."""
        mock_span = MagicMock()
        mock_span.name = "some.span"
        mock_span.attributes = {"llm.model_name": "gpt-4o"}
        mock_span.instrumentation_scope = None

        result = self.processor._is_litellm_span(mock_span)
        self.assertFalse(result)

    def test_extract_request_content_from_indexed_attributes(self):
        """Test extraction of request content from OpenInference indexed attributes."""
        mock_span = MagicMock()
        mock_span.attributes = {
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.content": "What is machine learning?",
            "llm.input_messages.1.message.role": "system",
            "llm.input_messages.1.message.content": "You are a helpful assistant",
        }

        result = self.processor._extract_request_content(mock_span)

        self.assertIsNotNone(result)
        self.assertIn("user", result)
        self.assertIn("What is machine learning?", result)

    def test_extract_request_content_from_input_value_json(self):
        """Test extraction of request content from input.value with JSON messages."""
        mock_span = MagicMock()
        mock_span.attributes = {
            "input.value": json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": "You are helpful"},
                        {"role": "user", "content": "Tell me about AI"},
                    ]
                }
            )
        }

        result = self.processor._extract_request_content(mock_span)

        self.assertIsNotNone(result)
        self.assertIn("user", result)
        self.assertIn("Tell me about AI", result)

    def test_extract_response_content_from_indexed_attributes(self):
        """Test extraction of response content from OpenInference indexed attributes."""
        mock_span = MagicMock()
        mock_span.attributes = {
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": "ML is a subset of AI.",
        }

        result = self.processor._extract_response_content(mock_span)

        self.assertIsNotNone(result)
        self.assertEqual(result, "ML is a subset of AI.")

    def test_on_end_enriches_openinference_litellm_span(self):
        """Test that on_end enriches a real OpenInference LiteLLM span (bare function name)."""
        mock_span = MagicMock()
        mock_span.name = "acompletion"

        # Set up attributes as OpenInference LiteLLM instrumentor v0.1.19 would
        mock_span.attributes = {
            "llm.model_name": "openai/gpt-4o",
            "openinference.span.kind": "LLM",
            "llm.token_count.prompt": 111,
            "llm.token_count.completion": 351,
            "llm.input_messages.0.message.role": "system",
            "llm.input_messages.0.message.content": "You are a senior loan underwriter",
            "llm.input_messages.1.message.role": "user",
            "llm.input_messages.1.message.content": "Provide a brief risk assessment",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": "Risk Assessment Note: Auto Loan",
            "input.value": '{"messages": [{"role": "system", "content": "You are a senior loan underwriter"}, {"role": "user", "content": "Provide a brief risk assessment"}]}',
            "output.value": "Risk Assessment Note: Auto Loan",
        }
        mock_span._attributes = dict(mock_span.attributes)

        # Set up instrumentation scope
        mock_scope = MagicMock()
        mock_scope.name = "openinference.instrumentation.litellm"
        mock_span.instrumentation_scope = mock_scope

        # Process the span
        self.processor.on_end(mock_span)

        # Verify enrichment via set_attribute calls
        set_attr_calls = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}
        self.assertIn("gen_ai.request.first_message", set_attr_calls)
        self.assertIn("gen_ai.response", set_attr_calls)

        # Verify response content
        response_attr = set_attr_calls["gen_ai.response"]
        self.assertEqual(response_attr, "Risk Assessment Note: Auto Loan")


if __name__ == "__main__":
    unittest.main(verbosity=2)
