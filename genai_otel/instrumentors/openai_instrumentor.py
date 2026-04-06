"""OpenTelemetry instrumentor for the OpenAI Python SDK.

This instrumentor automatically traces chat completion calls made using the
OpenAI SDK, capturing relevant attributes such as the model name, message count,
and token usage.
"""

import json
import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class OpenAIInstrumentor(BaseInstrumentor):
    """Instrumentor for OpenAI SDK"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._openai_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if OpenAI library is available."""
        try:
            import openai

            self._openai_available = True
            logger.debug("OpenAI library detected and available for instrumentation")
        except ImportError:
            logger.debug("OpenAI library not installed, instrumentation will be skipped")
            self._openai_available = False

    def instrument(self, config: OTelConfig):
        """Instrument OpenAI SDK if available.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._openai_available:
            logger.debug("Skipping OpenAI instrumentation - library not available")
            return

        self.config = config

        try:
            import openai
            import wrapt

            # Instrument sync OpenAI client initialization
            if hasattr(openai, "OpenAI"):
                original_init = openai.OpenAI.__init__

                def wrapped_init(wrapped, instance, args, kwargs):
                    result = wrapped(*args, **kwargs)
                    self._instrument_client(instance)
                    return result

                openai.OpenAI.__init__ = wrapt.FunctionWrapper(original_init, wrapped_init)
                self._instrumented = True
                logger.info("OpenAI instrumentation enabled")

            # Instrument async OpenAI client initialization
            if hasattr(openai, "AsyncOpenAI"):
                original_async_init = openai.AsyncOpenAI.__init__

                def wrapped_async_init(wrapped, instance, args, kwargs):
                    result = wrapped(*args, **kwargs)
                    self._instrument_async_client(instance)
                    return result

                openai.AsyncOpenAI.__init__ = wrapt.FunctionWrapper(
                    original_async_init, wrapped_async_init
                )
                self._instrumented = True
                logger.info("AsyncOpenAI instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument OpenAI: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _instrument_client(self, client):
        """Instrument OpenAI client methods.

        Args:
            client: The OpenAI client instance to instrument.
        """
        if (
            hasattr(client, "chat")
            and hasattr(client.chat, "completions")
            and hasattr(client.chat.completions, "create")
        ):
            original_create = client.chat.completions.create
            instrumented_create_method = self.create_span_wrapper(
                span_name="openai.chat.completion",
                extract_attributes=self._extract_openai_attributes,
            )(original_create)
            client.chat.completions.create = instrumented_create_method

    def _instrument_async_client(self, client):
        """Instrument AsyncOpenAI client methods.

        Args:
            client: The AsyncOpenAI client instance to instrument.
        """
        if (
            hasattr(client, "chat")
            and hasattr(client.chat, "completions")
            and hasattr(client.chat.completions, "create")
        ):
            original_create = client.chat.completions.create
            instrumented_create_method = self._create_async_span_wrapper(
                span_name="openai.chat.completion",
                extract_attributes=self._extract_openai_attributes,
            )(original_create)
            client.chat.completions.create = instrumented_create_method

    def _create_async_span_wrapper(self, span_name, extract_attributes=None):
        """Create an async wrapper that adds OpenTelemetry spans around async calls.

        Args:
            span_name: Name for the span.
            extract_attributes: Optional callable to extract span attributes.

        Returns:
            A decorator function for async methods.
        """
        import asyncio
        import time

        from opentelemetry import trace
        from opentelemetry.trace import SpanKind, Status, StatusCode

        instrumentor = self

        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                tracer = trace.get_tracer(__name__)
                with tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
                    if extract_attributes:
                        attrs = extract_attributes(None, args, kwargs)
                        for key, value in attrs.items():
                            span.set_attribute(key, value)

                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        duration = time.time() - start_time
                        span.set_attribute("gen_ai.latency", duration)

                        # Extract response attributes
                        response_attrs = instrumentor._extract_response_attributes(result)
                        for key, value in response_attrs.items():
                            span.set_attribute(key, value)

                        # Record metrics
                        instrumentor._record_result_metrics(span, result, start_time, kwargs)

                        # Add content events
                        instrumentor._add_content_events(span, result, kwargs)

                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise

            return async_wrapper

        return decorator

    def _extract_openai_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Extract attributes from OpenAI API call.

        Args:
            instance: The client instance.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])

        # Core attributes
        attrs["gen_ai.system"] = "openai"
        attrs["gen_ai.request.model"] = model
        attrs["gen_ai.operation.name"] = "chat"  # NEW: operation name
        attrs["gen_ai.request.message_count"] = len(messages)

        # Request parameters (NEW)
        if "temperature" in kwargs:
            attrs["gen_ai.request.temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            attrs["gen_ai.request.top_p"] = kwargs["top_p"]
        if "max_tokens" in kwargs:
            attrs["gen_ai.request.max_tokens"] = kwargs["max_tokens"]
        if "frequency_penalty" in kwargs:
            attrs["gen_ai.request.frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            attrs["gen_ai.request.presence_penalty"] = kwargs["presence_penalty"]

        # Tool/function definitions (Phase 3.1)
        if "tools" in kwargs:
            try:
                attrs["llm.tools"] = json.dumps(kwargs["tools"])
            except (TypeError, ValueError) as e:
                logger.debug("Failed to serialize tools: %s", e)

        first_message = self._build_first_message(messages)
        if first_message:
            attrs["gen_ai.request.first_message"] = first_message

        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from OpenAI response.

        Args:
            result: The API response object.

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        if hasattr(result, "usage") and result.usage:
            usage = result.usage
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }

            # Extract reasoning tokens for o1 models (Phase 3.2)
            if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                details = usage.completion_tokens_details
                usage_dict["completion_tokens_details"] = {
                    "reasoning_tokens": getattr(details, "reasoning_tokens", 0)
                }

            return usage_dict
        return None

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        """Extract response attributes from OpenAI response.

        Args:
            result: The API response object.

        Returns:
            Dict[str, Any]: Dictionary of response attributes.
        """
        attrs = {}

        # Response ID
        if hasattr(result, "id"):
            attrs["gen_ai.response.id"] = result.id

        # Response model (actual model used, may differ from request)
        if hasattr(result, "model"):
            attrs["gen_ai.response.model"] = result.model

        # Finish reasons
        if hasattr(result, "choices") and result.choices:
            finish_reasons = [
                choice.finish_reason
                for choice in result.choices
                if hasattr(choice, "finish_reason")
            ]
            if finish_reasons:
                attrs["gen_ai.response.finish_reasons"] = finish_reasons

            # Tool calls extraction (Phase 3.1)
            for choice_idx, choice in enumerate(result.choices):
                message = getattr(choice, "message", None)
                if message and hasattr(message, "tool_calls") and message.tool_calls:
                    for tc_idx, tool_call in enumerate(message.tool_calls):
                        prefix = f"llm.output_messages.{choice_idx}.message.tool_calls.{tc_idx}"
                        if hasattr(tool_call, "id"):
                            attrs[f"{prefix}.tool_call.id"] = tool_call.id
                        if hasattr(tool_call, "function"):
                            if hasattr(tool_call.function, "name"):
                                attrs[f"{prefix}.tool_call.function.name"] = tool_call.function.name
                            if hasattr(tool_call.function, "arguments"):
                                attrs[f"{prefix}.tool_call.function.arguments"] = (
                                    tool_call.function.arguments
                                )

        return attrs

    def _add_content_events(self, span, result, request_kwargs: dict):
        """Add prompt and completion content as span events and attributes.

        Args:
            span: The OpenTelemetry span.
            result: The API response object.
            request_kwargs: The original request kwargs.
        """
        # Add prompt content events
        messages = request_kwargs.get("messages", [])
        for idx, message in enumerate(messages):
            if isinstance(message, dict):
                role = message.get("role", "unknown")
                content = message.get("content", "")
                span.add_event(
                    f"gen_ai.prompt.{idx}",
                    attributes={"gen_ai.prompt.role": role, "gen_ai.prompt.content": str(content)},
                )

        # Add completion content events AND attributes (for evaluation processor)
        if hasattr(result, "choices") and result.choices:
            response_text = None
            for idx, choice in enumerate(result.choices):
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    content = choice.message.content
                    # Add as event for observability
                    span.add_event(
                        f"gen_ai.completion.{idx}",
                        attributes={
                            "gen_ai.completion.role": "assistant",
                            "gen_ai.completion.content": str(content),
                        },
                    )
                    # Capture first completion for evaluation
                    if idx == 0:
                        response_text = str(content)

            # Set as attribute for evaluation processor
            if response_text:
                span.set_attribute("gen_ai.response", response_text)

    def _extract_finish_reason(self, result) -> Optional[str]:
        """Extract finish reason from OpenAI response.

        Args:
            result: The OpenAI API response object.

        Returns:
            Optional[str]: The finish reason string or None if not available.
        """
        try:
            if hasattr(result, "choices") and result.choices:
                # Get the first finish_reason from the first choice
                first_choice = result.choices[0]
                if hasattr(first_choice, "finish_reason"):
                    return first_choice.finish_reason
        except Exception as e:
            logger.debug("Failed to extract finish_reason: %s", e)
        return None
