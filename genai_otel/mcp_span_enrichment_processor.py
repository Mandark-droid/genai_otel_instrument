"""Span processor for enriching MCP (Model Context Protocol) spans with evaluation support.

This processor adds evaluation support to spans created by OpenInference's MCP
instrumentor by extracting and standardizing request/response content attributes.

Since we use OpenInference's MCPInstrumentor (external dependency), we cannot
directly modify how it captures spans. This processor runs as a post-processing step
to add the required attributes for evaluation metrics support.

MCP tools include: databases, vector DBs, caches (Redis), message queues (Kafka), APIs, etc.
"""

import json
import logging
from typing import Any, Dict, Optional

from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

logger = logging.getLogger(__name__)


class MCPSpanEnrichmentProcessor(SpanProcessor):
    """Span processor that enriches MCP tool spans with evaluation-compatible attributes.

    This processor:
    1. Detects spans from MCP tools (created by OpenInference instrumentor)
    2. Extracts request/response content from OpenInference attributes
    3. Adds standardized gen_ai.request.first_message and gen_ai.response attributes
    4. Enables automatic evaluation metrics via BaseInstrumentor._run_evaluation_checks()
    """

    def __init__(self):
        """Initialize the MCP span enrichment processor."""
        super().__init__()
        logger.debug("MCP span enrichment processor initialized")

    def on_start(self, span: Span, parent_context: Optional[otel_context.Context] = None) -> None:
        """Called when a span is started. No-op for this processor.

        Args:
            span: The span that was started.
            parent_context: The parent context.
        """
        pass  # We only enrich on span end

    def on_end(self, span: ReadableSpan) -> None:
        """Called when a span is ended. Enriches MCP spans with evaluation attributes.

        Args:
            span: The span that ended.
        """
        try:
            # Check if this is an MCP span
            if not self._is_mcp_span(span):
                return

            # Extract request and response content from OpenInference attributes
            request_content = self._extract_request_content(span)
            response_content = self._extract_response_content(span)

            # Add evaluation-compatible attributes if not already present
            if request_content and not self._has_attribute(span, "gen_ai.request.first_message"):
                self._set_attribute(span, "gen_ai.request.first_message", request_content)
                logger.debug("Added gen_ai.request.first_message to MCP span")

            if response_content and not self._has_attribute(span, "gen_ai.response"):
                self._set_attribute(span, "gen_ai.response", response_content)
                logger.debug("Added gen_ai.response to MCP span")

        except Exception as e:
            logger.debug("Failed to enrich MCP span: %s", e)

    def shutdown(self) -> None:
        """Called when the processor is shut down."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush. No-op for this processor.

        Args:
            timeout_millis: Timeout in milliseconds.

        Returns:
            bool: Always True for this processor.
        """
        return True

    def _is_mcp_span(self, span: ReadableSpan) -> bool:
        """Check if a span is from MCP instrumentation.

        Args:
            span: The span to check.

        Returns:
            bool: True if this is an MCP span.
        """
        # Check instrumentation scope for MCP
        instrumentation_scope = getattr(span, "instrumentation_scope", None)
        if instrumentation_scope:
            scope_name = getattr(instrumentation_scope, "name", "")
            if "mcp" in scope_name.lower() and "openinference" in scope_name.lower():
                return True

        # Check span name patterns used by OpenInference MCP instrumentor
        if span.name:
            span_name_lower = span.name.lower()
            # MCP tool calls typically have "tool" or specific tool names
            if "tool" in span_name_lower or "mcp" in span_name_lower:
                # Confirm with instrumentation scope
                if instrumentation_scope and "mcp" in scope_name.lower():
                    return True

        # Check for MCP-related attributes
        attributes = span.attributes or {}

        # MCP spans typically have tool-related attributes
        if any(key.startswith("tool.") for key in attributes.keys()):
            return True

        # Check for mcp namespace attributes
        if any(key.startswith("mcp.") for key in attributes.keys()):
            return True

        return False

    def _extract_request_content(self, span: ReadableSpan) -> Optional[str]:
        """Extract request content from OpenInference MCP span attributes.

        MCP tools may store requests in various formats:
        - input.value (tool input/query)
        - tool.parameters (tool invocation parameters)
        - mcp.request (MCP-specific request data)

        Args:
            span: The span to extract request content from.

        Returns:
            Optional[str]: The extracted request content in dict-string format, or None.
        """
        attributes = span.attributes or {}

        # Try to extract from input.value (most common for tool calls)
        if "input.value" in attributes:
            input_value = attributes["input.value"]
            if input_value:
                # Convert to dict-string format matching other instrumentors
                # Truncate to ensure final result is ~200 chars (accounting for dict structure)
                truncated_content = str(input_value)[:150]
                result = str({"role": "user", "content": truncated_content})
                return result[:200]

        # Try tool.parameters (structured tool input)
        if "tool.parameters" in attributes:
            try:
                parameters = attributes["tool.parameters"]
                if isinstance(parameters, str):
                    parameters = json.loads(parameters)

                if parameters:
                    # Convert parameters dict to string for content
                    params_str = json.dumps(parameters)[:150]
                    result = str({"role": "user", "content": params_str})
                    return result[:200]
            except (json.JSONDecodeError, TypeError) as e:
                logger.debug("Failed to parse tool.parameters: %s", e)

        # Try mcp.request
        if "mcp.request" in attributes:
            mcp_request = attributes["mcp.request"]
            if mcp_request:
                truncated_content = str(mcp_request)[:150]
                result = str({"role": "user", "content": truncated_content})
                return result[:200]

        # Try generic message/query attributes
        for key in ["message", "query", "request"]:
            if key in attributes:
                value = attributes[key]
                if value:
                    truncated_content = str(value)[:150]
                    result = str({"role": "user", "content": truncated_content})
                    return result[:200]

        return None

    def _extract_response_content(self, span: ReadableSpan) -> Optional[str]:
        """Extract response content from OpenInference MCP span attributes.

        MCP tools may store responses in various formats:
        - output.value (tool output/result)
        - tool.result (tool invocation result)
        - mcp.response (MCP-specific response data)

        Args:
            span: The span to extract response content from.

        Returns:
            Optional[str]: The extracted response content, or None.
        """
        attributes = span.attributes or {}

        # Try to extract from output.value (most common for tool results)
        if "output.value" in attributes:
            output_value = attributes["output.value"]
            if output_value:
                return str(output_value)

        # Try tool.result
        if "tool.result" in attributes:
            tool_result = attributes["tool.result"]
            if tool_result:
                # Handle both string and structured results
                if isinstance(tool_result, dict):
                    return json.dumps(tool_result)
                return str(tool_result)

        # Try mcp.response
        if "mcp.response" in attributes:
            mcp_response = attributes["mcp.response"]
            if mcp_response:
                if isinstance(mcp_response, dict):
                    return json.dumps(mcp_response)
                return str(mcp_response)

        # Try generic result/response attributes
        for key in ["result", "response", "output"]:
            if key in attributes:
                value = attributes[key]
                if value:
                    if isinstance(value, dict):
                        return json.dumps(value)
                    return str(value)

        return None

    def _has_attribute(self, span: ReadableSpan, key: str) -> bool:
        """Check if a span already has a specific attribute.

        Args:
            span: The span to check.
            key: The attribute key.

        Returns:
            bool: True if the attribute exists.
        """
        attributes = span.attributes or {}
        return key in attributes

    def _set_attribute(self, span: ReadableSpan, key: str, value: str) -> None:
        """Set an attribute on a span.

        Note: ReadableSpan attributes are immutable, but we can still set them
        during the on_end callback for export purposes.

        Args:
            span: The span to set the attribute on.
            key: The attribute key.
            value: The attribute value.
        """
        # Cast to Span to access set_attribute if still mutable
        if isinstance(span, Span):
            span.set_attribute(key, value)
        else:
            # For ReadableSpan, we need to modify the attributes dict directly
            # This works because on_end is called before the span is fully sealed
            if hasattr(span, "_attributes"):
                span._attributes[key] = value
            elif hasattr(span, "attributes") and isinstance(span.attributes, dict):
                span.attributes[key] = value
