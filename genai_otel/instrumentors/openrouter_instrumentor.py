"""OpenTelemetry instrumentor for OpenRouter API.

OpenRouter provides a unified OpenAI-compatible API to access multiple LLM providers.
This instrumentor automatically traces chat completion calls made through OpenRouter,
capturing relevant attributes such as the model name, message count, and token usage.
"""

import json
import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class OpenRouterInstrumentor(BaseInstrumentor):
    """Instrumentor for OpenRouter API (OpenAI-compatible interface)"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._openrouter_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if OpenAI library is available (OpenRouter uses OpenAI client)."""
        try:
            import openai

            self._openrouter_available = True
            logger.debug("OpenAI library detected, OpenRouter instrumentation available")
        except ImportError:
            logger.debug("OpenAI library not installed, OpenRouter instrumentation will be skipped")
            self._openrouter_available = False

    def instrument(self, config: OTelConfig):
        """Instrument OpenRouter API calls if available.

        OpenRouter uses the OpenAI client library with a custom base_url.
        We detect OpenRouter usage by checking the base_url attribute.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._openrouter_available:
            logger.debug("Skipping OpenRouter instrumentation - library not available")
            return

        self.config = config

        try:
            import openai
            import wrapt

            # Instrument OpenAI client initialization to detect OpenRouter usage
            if hasattr(openai, "OpenAI"):
                original_init = openai.OpenAI.__init__

                def wrapped_init(wrapped, instance, args, kwargs):
                    result = wrapped(*args, **kwargs)
                    # Only instrument if this is an OpenRouter client
                    if self._is_openrouter_client(instance):
                        self._instrument_client(instance)
                        logger.debug("OpenRouter client detected and instrumented")
                    return result

                openai.OpenAI.__init__ = wrapt.FunctionWrapper(original_init, wrapped_init)
                self._instrumented = True
                logger.info("OpenRouter instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument OpenRouter: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _is_openrouter_client(self, client) -> bool:
        """Check if the OpenAI client is configured for OpenRouter.

        Args:
            client: The OpenAI client instance.

        Returns:
            bool: True if this is an OpenRouter client, False otherwise.
        """
        if hasattr(client, "base_url") and client.base_url:
            base_url = str(client.base_url).lower()
            return "openrouter.ai" in base_url
        return False

    def _instrument_client(self, client):
        """Instrument OpenRouter client methods.

        Args:
            client: The OpenRouter (OpenAI) client instance to instrument.
        """
        if (
            hasattr(client, "chat")
            and hasattr(client.chat, "completions")
            and hasattr(client.chat.completions, "create")
        ):
            original_create = client.chat.completions.create
            instrumented_create_method = self.create_span_wrapper(
                span_name="openrouter.chat.completion",
                extract_attributes=self._extract_openrouter_attributes,
            )(original_create)
            client.chat.completions.create = instrumented_create_method

    def _extract_openrouter_attributes(
        self, instance: Any, args: Any, kwargs: Any
    ) -> Dict[str, Any]:
        """Extract attributes from OpenRouter API call.

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
        attrs["gen_ai.system"] = "openrouter"
        attrs["gen_ai.request.model"] = model
        attrs["gen_ai.operation.name"] = "chat"
        attrs["gen_ai.request.message_count"] = len(messages)

        # Request parameters
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

        # OpenRouter-specific parameters
        if "provider" in kwargs:
            attrs["openrouter.provider"] = str(kwargs["provider"])
        if "route" in kwargs:
            attrs["openrouter.route"] = str(kwargs["route"])

        # Tool/function definitions
        if "tools" in kwargs:
            try:
                attrs["llm.tools"] = json.dumps(kwargs["tools"])
            except (TypeError, ValueError) as e:
                logger.debug("Failed to serialize tools: %s", e)

        if messages:
            # Only capture first 200 chars to avoid sensitive data and span size issues
            first_message = str(messages[0])[:200]
            attrs["gen_ai.request.first_message"] = first_message

        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from OpenRouter response.

        OpenRouter returns OpenAI-compatible responses with usage information.

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

            # OpenRouter may include additional usage details
            if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                details = usage.completion_tokens_details
                usage_dict["completion_tokens_details"] = {
                    "reasoning_tokens": getattr(details, "reasoning_tokens", 0)
                }

            return usage_dict
        return None

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        """Extract response attributes from OpenRouter response.

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

            # Tool calls extraction
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
        """Extract finish reason from OpenRouter response.

        Args:
            result: The OpenRouter API response object.

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
