"""Tests for MCP span enrichment processor."""

import json
import unittest
from unittest.mock import MagicMock

from genai_otel.mcp_span_enrichment_processor import MCPSpanEnrichmentProcessor


class TestMCPSpanEnrichmentProcessor(unittest.TestCase):
    """Tests for MCPSpanEnrichmentProcessor"""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = MCPSpanEnrichmentProcessor()

    def test_init(self):
        """Test processor initialization."""
        processor = MCPSpanEnrichmentProcessor()
        self.assertIsNotNone(processor)

    def test_on_start_is_noop(self):
        """Test that on_start does nothing."""
        mock_span = MagicMock()
        self.processor.on_start(mock_span, None)

    def test_shutdown(self):
        """Test shutdown method."""
        self.processor.shutdown()

    def test_force_flush(self):
        """Test force_flush method."""
        result = self.processor.force_flush()
        self.assertTrue(result)

    def test_is_mcp_span_by_scope(self):
        """Test detection of MCP span by instrumentation scope."""
        mock_span = MagicMock()
        mock_span.name = "tool.call"
        mock_span.attributes = {}
        mock_scope = MagicMock()
        mock_scope.name = "openinference.instrumentation.mcp"
        mock_span.instrumentation_scope = mock_scope

        result = self.processor._is_mcp_span(mock_span)
        self.assertTrue(result)

    def test_is_mcp_span_by_tool_attributes(self):
        """Test detection of MCP span by tool attributes."""
        mock_span = MagicMock()
        mock_span.name = "operation"
        mock_span.attributes = {"tool.name": "database_query", "tool.parameters": "{}"}

        result = self.processor._is_mcp_span(mock_span)
        self.assertTrue(result)

    def test_is_mcp_span_by_mcp_attributes(self):
        """Test detection of MCP span by mcp namespace attributes."""
        mock_span = MagicMock()
        mock_span.name = "operation"
        mock_span.attributes = {"mcp.tool": "redis_get", "mcp.request": "data"}

        result = self.processor._is_mcp_span(mock_span)
        self.assertTrue(result)

    def test_is_not_mcp_span(self):
        """Test that non-MCP spans are not detected."""
        mock_span = MagicMock()
        mock_span.name = "openai.chat.completion"
        mock_span.attributes = {"gen_ai.system": "openai"}
        mock_span.instrumentation_scope = None

        result = self.processor._is_mcp_span(mock_span)
        self.assertFalse(result)

    def test_extract_request_content_from_input_value(self):
        """Test extraction of request content from input.value."""
        mock_span = MagicMock()
        mock_span.attributes = {"input.value": "SELECT * FROM users"}

        result = self.processor._extract_request_content(mock_span)

        self.assertIsNotNone(result)
        self.assertIn("user", result)
        self.assertIn("SELECT", result)

    def test_extract_request_content_from_tool_parameters(self):
        """Test extraction of request content from tool.parameters."""
        mock_span = MagicMock()
        params = {"query": "test", "limit": 10}
        mock_span.attributes = {"tool.parameters": json.dumps(params)}

        result = self.processor._extract_request_content(mock_span)

        self.assertIsNotNone(result)
        self.assertIn("user", result)

    def test_extract_request_content_from_mcp_request(self):
        """Test extraction of request content from mcp.request."""
        mock_span = MagicMock()
        mock_span.attributes = {"mcp.request": "GET /api/data"}

        result = self.processor._extract_request_content(mock_span)

        self.assertIsNotNone(result)
        self.assertIn("GET /api/data", result)

    def test_extract_request_content_from_generic_attributes(self):
        """Test extraction from generic message/query attributes."""
        mock_span = MagicMock()
        mock_span.attributes = {"query": "search term"}

        result = self.processor._extract_request_content(mock_span)

        self.assertIsNotNone(result)
        self.assertIn("search term", result)

    def test_extract_request_content_truncation(self):
        """Test that request content is truncated to 200 chars."""
        mock_span = MagicMock()
        long_content = "a" * 300
        mock_span.attributes = {"input.value": long_content}

        result = self.processor._extract_request_content(mock_span)

        self.assertIsNotNone(result)
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
        mock_span.attributes = {"output.value": "Query returned 10 results"}

        result = self.processor._extract_response_content(mock_span)

        self.assertIsNotNone(result)
        self.assertEqual(result, "Query returned 10 results")

    def test_extract_response_content_from_tool_result_string(self):
        """Test extraction from tool.result as string."""
        mock_span = MagicMock()
        mock_span.attributes = {"tool.result": "Operation successful"}

        result = self.processor._extract_response_content(mock_span)

        self.assertIsNotNone(result)
        self.assertEqual(result, "Operation successful")

    def test_extract_response_content_from_tool_result_dict(self):
        """Test extraction from tool.result as dict."""
        mock_span = MagicMock()
        result_dict = {"status": "success", "data": [1, 2, 3]}
        mock_span.attributes = {"tool.result": result_dict}

        result = self.processor._extract_response_content(mock_span)

        self.assertIsNotNone(result)
        self.assertIn("success", result)

    def test_extract_response_content_from_mcp_response(self):
        """Test extraction from mcp.response."""
        mock_span = MagicMock()
        response_dict = {"value": "cached_data"}
        mock_span.attributes = {"mcp.response": response_dict}

        result = self.processor._extract_response_content(mock_span)

        self.assertIsNotNone(result)
        self.assertIn("cached_data", result)

    def test_extract_response_content_from_generic_attributes(self):
        """Test extraction from generic result/response attributes."""
        mock_span = MagicMock()
        mock_span.attributes = {"result": "operation completed"}

        result = self.processor._extract_response_content(mock_span)

        self.assertIsNotNone(result)
        self.assertEqual(result, "operation completed")

    def test_extract_response_content_no_output(self):
        """Test extraction returns None when no output found."""
        mock_span = MagicMock()
        mock_span.attributes = {}

        result = self.processor._extract_response_content(mock_span)

        self.assertIsNone(result)

    def test_on_end_enriches_mcp_span(self):
        """Test that on_end enriches MCP spans with evaluation attributes."""
        mock_span = MagicMock()
        mock_span.name = "tool.database.query"
        mock_span.attributes = {
            "tool.name": "postgres_query",
            "input.value": "SELECT * FROM products WHERE price > 100",
            "output.value": "Found 25 products matching criteria",
        }
        mock_span._attributes = mock_span.attributes.copy()

        mock_scope = MagicMock()
        mock_scope.name = "openinference.instrumentation.mcp"
        mock_span.instrumentation_scope = mock_scope

        self.processor.on_end(mock_span)

        self.assertIn("gen_ai.request.first_message", mock_span._attributes)
        self.assertIn("gen_ai.response", mock_span._attributes)

    def test_on_end_skips_non_mcp_span(self):
        """Test that on_end skips non-MCP spans."""
        mock_span = MagicMock()
        mock_span.name = "openai.chat.completion"
        mock_span.attributes = {"gen_ai.system": "openai"}
        mock_span._attributes = mock_span.attributes.copy()
        mock_span.instrumentation_scope = None

        self.processor.on_end(mock_span)

        self.assertNotIn("gen_ai.request.first_message", mock_span._attributes)

    def test_on_end_skips_if_attributes_already_present(self):
        """Test that on_end doesn't overwrite existing evaluation attributes."""
        mock_span = MagicMock()
        mock_span.name = "tool.call"
        mock_span.attributes = {
            "gen_ai.request.first_message": "existing request",
            "gen_ai.response": "existing response",
            "input.value": "new input",
        }
        mock_span._attributes = mock_span.attributes.copy()

        mock_scope = MagicMock()
        mock_scope.name = "mcp"
        mock_span.instrumentation_scope = mock_scope

        self.processor.on_end(mock_span)

        self.assertEqual(mock_span._attributes["gen_ai.request.first_message"], "existing request")
        self.assertEqual(mock_span._attributes["gen_ai.response"], "existing response")

    def test_integration_full_enrichment_flow(self):
        """Integration test for complete span enrichment flow."""
        mock_span = MagicMock()
        mock_span.name = "mcp.tool.redis.get"

        mock_span.attributes = {
            "tool.name": "redis_get",
            "input.value": "GET user:12345",
            "output.value": '{"name": "John", "email": "john@example.com"}',
        }
        mock_span._attributes = mock_span.attributes.copy()

        mock_scope = MagicMock()
        mock_scope.name = "openinference.instrumentation.mcp"
        mock_span.instrumentation_scope = mock_scope

        self.processor.on_end(mock_span)

        self.assertIn("gen_ai.request.first_message", mock_span._attributes)
        self.assertIn("gen_ai.response", mock_span._attributes)

        request_attr = mock_span._attributes["gen_ai.request.first_message"]
        self.assertIn("user", request_attr)
        self.assertIn("user:12345", request_attr)

        response_attr = mock_span._attributes["gen_ai.response"]
        self.assertIn("John", response_attr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
