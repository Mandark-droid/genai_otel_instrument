"""Span processor for enriching LiteLLM spans with evaluation support.

This processor adds evaluation support to spans created by OpenInference's LiteLLM
instrumentor by extracting and standardizing request/response content attributes.

Since we use OpenInference's LiteLLMInstrumentor (external dependency), we cannot
directly modify how it captures spans. This processor runs as a post-processing step
to add the required attributes for evaluation metrics support.
"""

import json
import logging
from typing import Any, Dict, Optional

from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

logger = logging.getLogger(__name__)


class LiteLLMSpanEnrichmentProcessor(SpanProcessor):
    """Span processor that enriches LiteLLM spans with evaluation-compatible attributes.

    This processor:
    1. Detects spans from LiteLLM (created by OpenInference instrumentor)
    2. Extracts request/response content from OpenInference attributes
    3. Adds standardized gen_ai.request.first_message and gen_ai.response attributes
    4. Enables automatic evaluation metrics via BaseInstrumentor._run_evaluation_checks()
    """

    def __init__(self):
        """Initialize the LiteLLM span enrichment processor."""
        super().__init__()
        logger.debug("LiteLLM span enrichment processor initialized")

    def on_start(self, span: Span, parent_context: Optional[otel_context.Context] = None) -> None:
        """Called when a span is started. No-op for this processor.

        Args:
            span: The span that was started.
            parent_context: The parent context.
        """
        pass  # We only enrich on span end

    def on_end(self, span: ReadableSpan) -> None:
        """Called when a span is ended. Enriches LiteLLM spans with evaluation attributes.

        Args:
            span: The span that ended.
        """
        try:
            # Check if this is a LiteLLM span
            if not self._is_litellm_span(span):
                return

            # Extract request and response content from OpenInference attributes
            request_content = self._extract_request_content(span)
            response_content = self._extract_response_content(span)

            # Add evaluation-compatible attributes if not already present
            if request_content and not self._has_attribute(span, "gen_ai.request.first_message"):
                self._set_attribute(span, "gen_ai.request.first_message", request_content)
                logger.debug("Added gen_ai.request.first_message to LiteLLM span")

            if response_content and not self._has_attribute(span, "gen_ai.response"):
                self._set_attribute(span, "gen_ai.response", response_content)
                logger.debug("Added gen_ai.response to LiteLLM span")

        except Exception as e:
            logger.debug("Failed to enrich LiteLLM span: %s", e)

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

    def _is_litellm_span(self, span: ReadableSpan) -> bool:
        """Check if a span is from LiteLLM instrumentation.

        Args:
            span: The span to check.

        Returns:
            bool: True if this is a LiteLLM span.
        """
        # Check span name patterns used by OpenInference LiteLLM instrumentor
        if span.name and "litellm" in span.name.lower():
            return True

        # Check for OpenInference semantic convention attributes
        attributes = span.attributes or {}

        # OpenInference uses 'llm.system' or similar attributes
        # LiteLLM spans typically have these patterns
        if "llm.provider" in attributes or "llm.model" in attributes:
            return True

        # Check for gen_ai.system attribute (may be set by OpenInference)
        if attributes.get("gen_ai.system") in ["litellm", "openai", "anthropic", "cohere"]:
            # Additional check: if it's already a known provider span, don't double-process
            # LiteLLM acts as a proxy, so its spans have litellm-specific markers
            instrumentation_scope = getattr(span, "instrumentation_scope", None)
            if instrumentation_scope:
                scope_name = getattr(instrumentation_scope, "name", "")
                if "litellm" in scope_name.lower():
                    return True

        return False

    def _extract_request_content(self, span: ReadableSpan) -> Optional[str]:
        """Extract request content from OpenInference span attributes.

        OpenInference may store messages in various formats:
        - llm.input_messages (JSON array of message objects)
        - input.value (raw input text)
        - llm.prompts (array of prompts)

        Args:
            span: The span to extract request content from.

        Returns:
            Optional[str]: The extracted request content in dict-string format, or None.
        """
        attributes = span.attributes or {}

        # Try to extract from llm.input_messages (most detailed format)
        if "llm.input_messages" in attributes:
            try:
                messages_json = attributes["llm.input_messages"]
                if isinstance(messages_json, str):
                    messages = json.loads(messages_json)
                else:
                    messages = messages_json

                if messages and isinstance(messages, list) and len(messages) > 0:
                    # Convert first message to dict-string format
                    first_message = messages[0]
                    return str(first_message)[:200]  # Truncate to 200 chars
            except (json.JSONDecodeError, TypeError, IndexError) as e:
                logger.debug("Failed to parse llm.input_messages: %s", e)

        # Try input.value (simple text input)
        if "input.value" in attributes:
            input_value = attributes["input.value"]
            if input_value:
                # Convert to dict-string format matching other instrumentors
                return str({"role": "user", "content": str(input_value)[:200]})

        # Try llm.prompts
        if "llm.prompts" in attributes:
            try:
                prompts = attributes["llm.prompts"]
                if isinstance(prompts, str):
                    prompts = json.loads(prompts)

                if prompts and isinstance(prompts, list) and len(prompts) > 0:
                    first_prompt = prompts[0]
                    return str({"role": "user", "content": str(first_prompt)[:200]})
            except (json.JSONDecodeError, TypeError, IndexError) as e:
                logger.debug("Failed to parse llm.prompts: %s", e)

        return None

    def _extract_response_content(self, span: ReadableSpan) -> Optional[str]:
        """Extract response content from OpenInference span attributes.

        OpenInference may store responses in various formats:
        - llm.output_messages (JSON array of message objects)
        - output.value (raw output text)

        Args:
            span: The span to extract response content from.

        Returns:
            Optional[str]: The extracted response content, or None.
        """
        attributes = span.attributes or {}

        # Try to extract from llm.output_messages (most detailed format)
        if "llm.output_messages" in attributes:
            try:
                messages_json = attributes["llm.output_messages"]
                if isinstance(messages_json, str):
                    messages = json.loads(messages_json)
                else:
                    messages = messages_json

                if messages and isinstance(messages, list) and len(messages) > 0:
                    # Extract content from first message
                    first_message = messages[0]
                    if isinstance(first_message, dict):
                        # Try 'message.content' or 'content' field
                        content = first_message.get("message", {}).get(
                            "content"
                        ) or first_message.get("content")
                        if content:
                            return str(content)
                    elif isinstance(first_message, str):
                        return first_message
            except (json.JSONDecodeError, TypeError, IndexError) as e:
                logger.debug("Failed to parse llm.output_messages: %s", e)

        # Try output.value (simple text output)
        if "output.value" in attributes:
            output_value = attributes["output.value"]
            if output_value:
                return str(output_value)

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
