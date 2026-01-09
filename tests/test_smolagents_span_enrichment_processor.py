"""Tests for Smolagents span enrichment processor."""

import json
import unittest
from unittest.mock import MagicMock

from genai_otel.smolagents_span_enrichment_processor import SmolagentsSpanEnrichmentProcessor


class TestSmolagentsSpanEnrichmentProcessor(unittest.TestCase):
    """Tests for SmolagentsSpanEnrichmentProcessor"""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = SmolagentsSpanEnrichmentProcessor()

    def test_init(self):
        """Test processor initialization."""
        processor = SmolagentsSpanEnrichmentProcessor()
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

    def test_is_smolagents_span_by_name_and_scope(self):
        """Test detection of Smolagents span by name and scope."""
        mock_span = MagicMock()
        mock_span.name = "agent.run"
        mock_span.attributes = {}

        # Create mock instrumentation scope
        mock_scope = MagicMock()
        mock_scope.name = "openinference.instrumentation.smolagents"
        mock_span.instrumentation_scope = mock_scope

        result = self.processor._is_smolagents_span(mock_span)
        self.assertTrue(result)

    def test_is_smolagents_span_by_agent_attributes(self):
        """Test detection of Smolagents span by agent attributes."""
        mock_span = MagicMock()
        mock_span.name = "some.span"
        mock_span.attributes = {"agent.name": "MyAgent", "agent.task": "research"}

        result = self.processor._is_smolagents_span(mock_span)
        self.assertTrue(result)

    def test_is_smolagents_span_by_scope_only(self):
        """Test detection of Smolagents span by instrumentation scope only."""
        mock_span = MagicMock()
        mock_span.name = "task"
        mock_span.attributes = {}

        # Create mock instrumentation scope
        mock_scope = MagicMock()
        mock_scope.name = "smolagents"
        mock_span.instrumentation_scope = mock_scope

        result = self.processor._is_smolagents_span(mock_span)
        self.assertTrue(result)

    def test_is_not_smolagents_span(self):
        """Test that non-Smolagents spans are not detected."""
        mock_span = MagicMock()
        mock_span.name = "openai.chat.completion"
        mock_span.attributes = {"gen_ai.system": "openai"}
        mock_span.instrumentation_scope = None

        result = self.processor._is_smolagents_span(mock_span)
        self.assertFalse(result)

    def test_extract_request_content_from_input_value(self):
        """Test extraction of request content from input.value."""
        mock_span = MagicMock()
        mock_span.attributes = {"input.value": "Research the latest AI trends"}

        result = self.processor._extract_request_content(mock_span)

        self.assertIsNotNone(result)
        self.assertIn("user", result)
        self.assertIn("AI trends", result)

    def test_extract_request_content_from_agent_input(self):
        """Test extraction of request content from agent.input."""
        mock_span = MagicMock()
        mock_span.attributes = {"agent.input": "Analyze this data"}

        result = self.processor._extract_request_content(mock_span)

        self.assertIsNotNone(result)
        self.assertIn("user", result)
        self.assertIn("Analyze this data", result)

    def test_extract_request_content_from_llm_messages(self):
        """Test extraction of request content from llm.input_messages."""
        mock_span = MagicMock()
        messages = [{"role": "user", "content": "What is AI?"}]
        mock_span.attributes = {"llm.input_messages": json.dumps(messages)}

        result = self.processor._extract_request_content(mock_span)

        self.assertIsNotNone(result)
        self.assertIn("user", result)
        self.assertIn("What is AI?", result)

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
        mock_span.attributes = {"input.value": long_content}

        result = self.processor._extract_request_content(mock_span)

        self.assertIsNotNone(result)
        # Should be truncated to 200 chars
        self.assertLessEqual(len(result), 200)

    def test_extract_request_content_no_input(self):
        """Test extraction returns None when no input found."""
        mock_span = MagicMock()
        mock_span.attributes = {}

        result = self.processor._extract_request_content(mock_span)

        self.assertIsNone(result)

    def test_extract_response_content_from_output_value(self):
        """Test extraction of response content from output.value."""
        mock_span = MagicMock()
        mock_span.attributes = {"output.value": "AI is artificial intelligence"}

        result = self.processor._extract_response_content(mock_span)

        self.assertIsNotNone(result)
        self.assertEqual(result, "AI is artificial intelligence")

    def test_extract_response_content_from_agent_output(self):
        """Test extraction of response content from agent.output."""
        mock_span = MagicMock()
        mock_span.attributes = {"agent.output": "Task completed successfully"}

        result = self.processor._extract_response_content(mock_span)

        self.assertIsNotNone(result)
        self.assertEqual(result, "Task completed successfully")

    def test_extract_response_content_from_llm_messages(self):
        """Test extraction of response content from llm.output_messages."""
        mock_span = MagicMock()
        messages = [{"message": {"content": "This is the agent response"}}]
        mock_span.attributes = {"llm.output_messages": json.dumps(messages)}

        result = self.processor._extract_response_content(mock_span)

        self.assertIsNotNone(result)
        self.assertEqual(result, "This is the agent response")

    def test_extract_response_content_from_llm_messages_content_field(self):
        """Test extraction when content is at top level of message."""
        mock_span = MagicMock()
        messages = [{"content": "Direct response content"}]
        mock_span.attributes = {"llm.output_messages": json.dumps(messages)}

        result = self.processor._extract_response_content(mock_span)

        self.assertIsNotNone(result)
        self.assertEqual(result, "Direct response content")

    def test_extract_response_content_from_llm_messages_string(self):
        """Test extraction when output message is a string."""
        mock_span = MagicMock()
        messages = ["String response"]
        mock_span.attributes = {"llm.output_messages": json.dumps(messages)}

        result = self.processor._extract_response_content(mock_span)

        self.assertIsNotNone(result)
        self.assertEqual(result, "String response")

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

    def test_on_end_enriches_smolagents_span(self):
        """Test that on_end enriches Smolagents spans with evaluation attributes."""
        mock_span = MagicMock()
        mock_span.name = "agent.run"
        mock_span.attributes = {
            "input.value": "Research AI safety",
            "output.value": "AI safety involves alignment, robustness, and transparency",
        }
        mock_span._attributes = mock_span.attributes.copy()

        # Create mock instrumentation scope
        mock_scope = MagicMock()
        mock_scope.name = "openinference.instrumentation.smolagents"
        mock_span.instrumentation_scope = mock_scope

        self.processor.on_end(mock_span)

        # Verify that evaluation attributes were added
        self.assertIn("gen_ai.request.first_message", mock_span._attributes)
        self.assertIn("gen_ai.response", mock_span._attributes)

    def test_on_end_skips_non_smolagents_span(self):
        """Test that on_end skips non-Smolagents spans."""
        mock_span = MagicMock()
        mock_span.name = "openai.chat.completion"
        mock_span.attributes = {
            "gen_ai.system": "openai",
            "input.value": "test",
        }
        mock_span._attributes = mock_span.attributes.copy()
        mock_span.instrumentation_scope = None

        self.processor.on_end(mock_span)

        # Attributes should not be modified
        self.assertNotIn("gen_ai.request.first_message", mock_span._attributes)

    def test_on_end_skips_if_attributes_already_present(self):
        """Test that on_end doesn't overwrite existing evaluation attributes."""
        mock_span = MagicMock()
        mock_span.name = "agent.tool"
        mock_span.attributes = {
            "gen_ai.request.first_message": "existing request",
            "gen_ai.response": "existing response",
            "input.value": "new input",
        }
        mock_span._attributes = mock_span.attributes.copy()

        # Create mock instrumentation scope
        mock_scope = MagicMock()
        mock_scope.name = "smolagents"
        mock_span.instrumentation_scope = mock_scope

        self.processor.on_end(mock_span)

        # Should not overwrite existing attributes
        self.assertEqual(mock_span._attributes["gen_ai.request.first_message"], "existing request")
        self.assertEqual(mock_span._attributes["gen_ai.response"], "existing response")

    def test_on_end_handles_exception_gracefully(self):
        """Test that on_end handles exceptions gracefully."""
        mock_span = MagicMock()
        mock_span.name = "agent.run"
        # Create attributes that will cause an exception
        mock_span.attributes = {"llm.input_messages": "invalid json"}

        # Create mock instrumentation scope
        mock_scope = MagicMock()
        mock_scope.name = "smolagents"
        mock_span.instrumentation_scope = mock_scope

        # Should not raise exception
        try:
            self.processor.on_end(mock_span)
        except Exception as e:
            self.fail(f"on_end raised exception: {e}")

    def test_integration_full_enrichment_flow(self):
        """Integration test for complete span enrichment flow."""
        # Create a realistic Smolagents span
        mock_span = MagicMock()
        mock_span.name = "agent.run.task"

        # Set up attributes as OpenInference would
        mock_span.attributes = {
            "agent.name": "ResearchAgent",
            "input.value": "Research the latest developments in quantum computing",
            "output.value": "Quantum computing has seen significant advances in error correction and qubit stability.",
        }
        mock_span._attributes = mock_span.attributes.copy()

        # Set up instrumentation scope
        mock_scope = MagicMock()
        mock_scope.name = "openinference.instrumentation.smolagents"
        mock_span.instrumentation_scope = mock_scope

        # Process the span
        self.processor.on_end(mock_span)

        # Verify enrichment
        self.assertIn("gen_ai.request.first_message", mock_span._attributes)
        self.assertIn("gen_ai.response", mock_span._attributes)

        # Verify content accuracy
        request_attr = mock_span._attributes["gen_ai.request.first_message"]
        self.assertIn("user", request_attr)
        self.assertIn("quantum computing", request_attr)

        response_attr = mock_span._attributes["gen_ai.response"]
        self.assertIn("Quantum computing", response_attr)
        self.assertIn("error correction", response_attr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
