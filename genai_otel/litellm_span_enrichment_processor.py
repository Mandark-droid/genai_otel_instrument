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

    # Known OpenInference LiteLLM span names (bare function names without prefix)
    _LITELLM_SPAN_NAMES = frozenset(
        {
            "acompletion",
            "completion",
            "atext_completion",
            "text_completion",
            "aembedding",
            "embedding",
            "aimage_generation",
            "image_generation",
        }
    )

    def _is_litellm_span(self, span: ReadableSpan) -> bool:
        """Check if a span is from LiteLLM instrumentation.

        Args:
            span: The span to check.

        Returns:
            bool: True if this is a LiteLLM span.
        """
        # PRIMARY CHECK: instrumentation scope name (most reliable for OpenInference spans)
        instrumentation_scope = getattr(span, "instrumentation_scope", None)
        scope_name = ""
        if instrumentation_scope:
            scope_name = getattr(instrumentation_scope, "name", "") or ""
            if "litellm" in scope_name.lower():
                return True

        # Check span name contains "litellm"
        if span.name and "litellm" in span.name.lower():
            return True

        # Check for OpenInference semantic convention attributes
        attributes = span.attributes or {}

        # OpenInference uses 'llm.provider', 'llm.model', or 'llm.model_name'
        if (
            "llm.provider" in attributes
            or "llm.model" in attributes
            or "llm.model_name" in attributes
        ):
            # Verify it's from a LiteLLM/OpenInference scope to avoid false positives
            if "litellm" in scope_name.lower() or "openinference" in scope_name.lower():
                return True

        # Check bare function name patterns from OpenInference LiteLLM instrumentor
        # e.g. span.name = "acompletion" (not "litellm.acompletion")
        if span.name in self._LITELLM_SPAN_NAMES:
            if "openinference" in scope_name.lower():
                return True

        # Check for gen_ai.system attribute with LiteLLM scope
        if attributes.get("gen_ai.system") in ["litellm", "openai", "anthropic", "cohere"]:
            if "litellm" in scope_name.lower():
                return True

        return False

    def _extract_request_content(self, span: ReadableSpan) -> Optional[str]:
        """Extract request content from OpenInference span attributes.

        OpenInference may store messages in various formats:
        - llm.input_messages (JSON array of message objects)
        - llm.input_messages.N.message.role / .content (indexed attributes)
        - input.value (raw input text or JSON)
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

        # Try OpenInference indexed attributes: llm.input_messages.0.message.content
        # This is the format used by OpenInference LiteLLM instrumentor v0.1.19+
        if "llm.input_messages.0.message.content" in attributes:
            role = attributes.get("llm.input_messages.0.message.role", "user")
            content = attributes["llm.input_messages.0.message.content"]
            if content:
                return str({"role": str(role), "content": str(content)[:200]})

        # Try input.value (simple text input or JSON)
        if "input.value" in attributes:
            input_value = attributes["input.value"]
            if input_value:
                # input.value may contain JSON with messages array
                if isinstance(input_value, str) and input_value.strip().startswith("{"):
                    try:
                        parsed = json.loads(input_value)
                        if isinstance(parsed, dict) and "messages" in parsed:
                            messages = parsed["messages"]
                            if messages and isinstance(messages, list):
                                # Find last user message
                                for msg in reversed(messages):
                                    if isinstance(msg, dict) and msg.get("role") == "user":
                                        content = msg.get("content", "")
                                        return str({"role": "user", "content": str(content)[:200]})
                                # Fallback to first message
                                first_msg = messages[0]
                                if isinstance(first_msg, dict):
                                    return str(first_msg)[:200]
                    except (json.JSONDecodeError, TypeError):
                        pass
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
        - llm.output_messages.N.message.content (indexed attributes)
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

        # Try OpenInference indexed attributes: llm.output_messages.0.message.content
        if "llm.output_messages.0.message.content" in attributes:
            content = attributes["llm.output_messages.0.message.content"]
            if content:
                return str(content)

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

        In on_end(), the span is a ReadableSpan. The .attributes property returns a
        MappingProxyType (immutable view), but the underlying _attributes is a
        BoundedAttributes object which supports item assignment. We access _attributes
        directly to enrich the span before it reaches the exporter.

        Args:
            span: The span to set the attribute on.
            key: The attribute key.
            value: The attribute value.
        """
        # Prefer set_attribute if available (mutable Span)
        if hasattr(span, "set_attribute") and callable(getattr(span, "set_attribute", None)):
            try:
                span.set_attribute(key, value)
                return
            except (AttributeError, RuntimeError):
                pass  # Fall through to _attributes approach

        # For ReadableSpan in on_end: _attributes is BoundedAttributes (mutable)
        if hasattr(span, "_attributes") and span._attributes is not None:
            try:
                span._attributes[key] = value
            except (TypeError, AttributeError) as e:
                logger.debug(
                    "Cannot set attribute '%s' on span '%s': %s",
                    key,
                    getattr(span, "name", "unknown"),
                    e,
                )
