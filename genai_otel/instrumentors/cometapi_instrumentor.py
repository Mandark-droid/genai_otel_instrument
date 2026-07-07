"""OpenTelemetry instrumentor for CometAPI.

CometAPI (https://www.cometapi.com) is an all-in-one aggregator that exposes
500+ models behind a single API key. It is reachable through BOTH the OpenAI
SDK (OpenAI-compatible ``/v1/chat/completions``) and the Anthropic SDK
(Anthropic-compatible ``/v1/messages``) by pointing the client's ``base_url``
at ``https://api.cometapi.com``.

This instrumentor detects CometAPI usage by inspecting the ``base_url`` of
newly-created OpenAI and Anthropic clients and instruments only those clients,
capturing relevant attributes such as the model name, message count, and token
usage from either response shape.
"""

import json
import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class CometAPIInstrumentor(BaseInstrumentor):
    """Instrumentor for CometAPI (OpenAI- and Anthropic-compatible interfaces)"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._openai_available = False
        self._anthropic_available = False
        self._cometapi_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if OpenAI or Anthropic libraries are available (CometAPI uses either)."""
        try:
            import openai

            self._openai_available = True
        except ImportError:
            self._openai_available = False

        try:
            import anthropic

            self._anthropic_available = True
        except ImportError:
            self._anthropic_available = False

        self._cometapi_available = self._openai_available or self._anthropic_available
        if self._cometapi_available:
            logger.debug("OpenAI/Anthropic library detected, CometAPI instrumentation available")
        else:
            logger.debug(
                "Neither OpenAI nor Anthropic library installed, "
                "CometAPI instrumentation will be skipped"
            )

    def instrument(self, config: OTelConfig):
        """Instrument CometAPI calls if available.

        CometAPI is used through the OpenAI or Anthropic client libraries with a
        custom base_url. We detect CometAPI usage by checking the base_url
        attribute of newly-created clients.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._cometapi_available:
            logger.debug("Skipping CometAPI instrumentation - library not available")
            return

        self.config = config

        try:
            import wrapt

            if self._openai_available:
                import openai

                # Instrument OpenAI client initialization to detect CometAPI usage
                if hasattr(openai, "OpenAI"):
                    original_openai_init = openai.OpenAI.__init__

                    def wrapped_openai_init(wrapped, instance, args, kwargs):
                        result = wrapped(*args, **kwargs)
                        # Only instrument if this is a CometAPI client
                        if self._is_cometapi_client(instance):
                            self._instrument_openai_client(instance)
                            logger.debug("CometAPI client (OpenAI SDK) detected and instrumented")
                        return result

                    openai.OpenAI.__init__ = wrapt.FunctionWrapper(
                        original_openai_init, wrapped_openai_init
                    )
                    self._instrumented = True

            if self._anthropic_available:
                import anthropic

                # Instrument Anthropic client initialization to detect CometAPI usage
                if hasattr(anthropic, "Anthropic"):
                    original_anthropic_init = anthropic.Anthropic.__init__

                    def wrapped_anthropic_init(wrapped, instance, args, kwargs):
                        result = wrapped(*args, **kwargs)
                        # Only instrument if this is a CometAPI client
                        if self._is_cometapi_client(instance):
                            self._instrument_anthropic_client(instance)
                            logger.debug(
                                "CometAPI client (Anthropic SDK) detected and instrumented"
                            )
                        return result

                    anthropic.Anthropic.__init__ = wrapt.FunctionWrapper(
                        original_anthropic_init, wrapped_anthropic_init
                    )
                    self._instrumented = True

            if self._instrumented:
                logger.info("CometAPI instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument CometAPI: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _is_cometapi_client(self, client) -> bool:
        """Check if the client is configured for CometAPI.

        Args:
            client: The OpenAI or Anthropic client instance.

        Returns:
            bool: True if this is a CometAPI client, False otherwise.
        """
        if hasattr(client, "base_url") and client.base_url:
            base_url = str(client.base_url).lower()
            return "cometapi.com" in base_url
        return False

    def _instrument_openai_client(self, client):
        """Instrument a CometAPI client created via the OpenAI SDK.

        Args:
            client: The CometAPI (OpenAI) client instance to instrument.
        """
        if (
            hasattr(client, "chat")
            and hasattr(client.chat, "completions")
            and hasattr(client.chat.completions, "create")
        ):
            original_create = client.chat.completions.create
            instrumented_create_method = self.create_span_wrapper(
                span_name="cometapi.chat.completion",
                extract_attributes=self._extract_cometapi_attributes,
            )(original_create)
            client.chat.completions.create = instrumented_create_method

    def _instrument_anthropic_client(self, client):
        """Instrument a CometAPI client created via the Anthropic SDK.

        Args:
            client: The CometAPI (Anthropic) client instance to instrument.
        """
        if hasattr(client, "messages") and hasattr(client.messages, "create"):
            original_create = client.messages.create
            instrumented_create_method = self.create_span_wrapper(
                span_name="cometapi.messages.create",
                extract_attributes=self._extract_cometapi_attributes,
            )(original_create)
            client.messages.create = instrumented_create_method

    def _extract_cometapi_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Extract attributes from CometAPI API call.

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
        attrs["gen_ai.system"] = "cometapi"
        attrs["gen_ai.request.model"] = model
        attrs["gen_ai.operation.name"] = "chat"
        attrs["gen_ai.request.message_count"] = len(messages)

        # Request parameters (shared by OpenAI- and Anthropic-style calls)
        if "temperature" in kwargs:
            attrs["gen_ai.request.temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            attrs["gen_ai.request.top_p"] = kwargs["top_p"]
        if "max_tokens" in kwargs:
            attrs["gen_ai.request.max_tokens"] = kwargs["max_tokens"]

        # Tool/function definitions
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
        """Extract token usage from a CometAPI response.

        Handles both response shapes:
        - OpenAI-compatible: ``usage.prompt_tokens`` / ``usage.completion_tokens``
        - Anthropic-compatible: ``usage.input_tokens`` / ``usage.output_tokens``

        Args:
            result: The API response object.

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        if hasattr(result, "usage") and result.usage:
            usage = result.usage

            # OpenAI-compatible usage shape
            if hasattr(usage, "prompt_tokens") or hasattr(usage, "completion_tokens"):
                usage_dict = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                }
                if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                    details = usage.completion_tokens_details
                    usage_dict["completion_tokens_details"] = {
                        "reasoning_tokens": getattr(details, "reasoning_tokens", 0)
                    }
                return usage_dict

            # Anthropic-compatible usage shape
            if hasattr(usage, "input_tokens") or hasattr(usage, "output_tokens"):
                input_tokens = getattr(usage, "input_tokens", 0) or 0
                output_tokens = getattr(usage, "output_tokens", 0) or 0
                usage_dict = {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }
                if hasattr(usage, "cache_read_input_tokens"):
                    usage_dict["cache_read_input_tokens"] = (
                        getattr(usage, "cache_read_input_tokens", 0) or 0
                    )
                if hasattr(usage, "cache_creation_input_tokens"):
                    usage_dict["cache_creation_input_tokens"] = (
                        getattr(usage, "cache_creation_input_tokens", 0) or 0
                    )
                return usage_dict

        return None

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        """Extract response attributes from a CometAPI response.

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

        # OpenAI-compatible responses: finish reasons from choices
        if hasattr(result, "choices") and result.choices:
            finish_reasons = [
                choice.finish_reason
                for choice in result.choices
                if hasattr(choice, "finish_reason")
            ]
            if finish_reasons:
                attrs["gen_ai.response.finish_reasons"] = finish_reasons
        # Anthropic-compatible responses: stop_reason
        elif hasattr(result, "stop_reason") and result.stop_reason:
            attrs["gen_ai.response.finish_reasons"] = [result.stop_reason]

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

        response_text = None

        # OpenAI-compatible responses: completion content from choices
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
                    if idx == 0:
                        response_text = str(content)
        # Anthropic-compatible responses: completion content from content blocks
        elif hasattr(result, "content") and result.content:
            for idx, content_block in enumerate(result.content):
                if hasattr(content_block, "text"):
                    span.add_event(
                        f"gen_ai.completion.{idx}",
                        attributes={
                            "gen_ai.completion.role": "assistant",
                            "gen_ai.completion.content": content_block.text,
                        },
                    )
                    if idx == 0:
                        response_text = content_block.text

        # Set as attribute for evaluation processor
        if response_text:
            span.set_attribute("gen_ai.response", response_text)

    def _extract_finish_reason(self, result) -> Optional[str]:
        """Extract finish reason from a CometAPI response.

        Args:
            result: The CometAPI response object.

        Returns:
            Optional[str]: The finish reason string or None if not available.
        """
        try:
            # OpenAI-compatible responses
            if hasattr(result, "choices") and result.choices:
                first_choice = result.choices[0]
                if hasattr(first_choice, "finish_reason"):
                    return first_choice.finish_reason
            # Anthropic-compatible responses
            if hasattr(result, "stop_reason") and result.stop_reason:
                return result.stop_reason
        except Exception as e:
            logger.debug("Failed to extract finish_reason: %s", e)
        return None
