"""OpenTelemetry instrumentor for the OpenAI Python SDK.

This instrumentor automatically traces chat completion calls made using the
OpenAI SDK, capturing relevant attributes such as the model name, message count,
and token usage.
"""

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

            # Instrument OpenAI client initialization
            if hasattr(openai, "OpenAI"):
                original_init = openai.OpenAI.__init__

                def wrapped_init(wrapped, instance, args, kwargs):
                    result = wrapped(*args, **kwargs)
                    self._instrument_client(instance)
                    return result

                openai.OpenAI.__init__ = wrapt.FunctionWrapper(original_init, wrapped_init)
                self._instrumented = True
                logger.info("OpenAI instrumentation enabled")

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

        if messages:
            # Only capture first 200 chars to avoid sensitive data and span size issues
            first_message = str(messages[0])[:200]
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
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
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
                choice.finish_reason for choice in result.choices if hasattr(choice, "finish_reason")
            ]
            if finish_reasons:
                attrs["gen_ai.response.finish_reasons"] = finish_reasons

        return attrs

    def _add_content_events(self, span, result, request_kwargs: dict):
        """Add prompt and completion content as span events.

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

        # Add completion content events
        if hasattr(result, "choices") and result.choices:
            for idx, choice in enumerate(result.choices):
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    content = choice.message.content
                    span.add_event(
                        f"gen_ai.completion.{idx}",
                        attributes={
                            "gen_ai.completion.role": "assistant",
                            "gen_ai.completion.content": str(content),
                        },
                    )
