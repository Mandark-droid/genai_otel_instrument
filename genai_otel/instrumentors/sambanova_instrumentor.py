"""OpenTelemetry instrumentor for the SambaNova SDK.

This instrumentor automatically traces chat completion calls to SambaNova models,
capturing relevant attributes such as the model name and token usage.
"""

import json
import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class SambaNovaInstrumentor(BaseInstrumentor):
    """Instrumentor for SambaNova"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._sambanova_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if SambaNova library is available."""
        try:
            import sambanova

            self._sambanova_available = True
            logger.debug("SambaNova library detected and available for instrumentation")
        except ImportError:
            logger.debug("SambaNova library not installed, instrumentation will be skipped")
            self._sambanova_available = False

    def instrument(self, config: OTelConfig):
        """Instrument SambaNova SDK if available.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._sambanova_available:
            logger.debug("Skipping SambaNova instrumentation - library not available")
            return

        self.config = config

        try:
            import sambanova

            original_init = sambanova.SambaNova.__init__

            def wrapped_init(instance, *args, **kwargs):
                original_init(instance, *args, **kwargs)
                self._instrument_client(instance)
                return instance

            sambanova.SambaNova.__init__ = wrapped_init
            self._instrumented = True
            logger.info("SambaNova instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument SambaNova: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _instrument_client(self, client):
        """Instrument SambaNova client methods.

        Args:
            client: The SambaNova client instance to instrument.
        """
        if (
            hasattr(client, "chat")
            and hasattr(client.chat, "completions")
            and hasattr(client.chat.completions, "create")
        ):
            original_create = client.chat.completions.create
            instrumented_create_method = self.create_span_wrapper(
                span_name="sambanova.chat.completion",
                extract_attributes=self._extract_sambanova_attributes,
            )(original_create)
            client.chat.completions.create = instrumented_create_method

    def _extract_sambanova_attributes(
        self, instance: Any, args: Any, kwargs: Any
    ) -> Dict[str, Any]:
        """Extract attributes from SambaNova API call.

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
        attrs["gen_ai.system"] = "sambanova"
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
        """Extract token usage from SambaNova response.

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
        """Extract response attributes from SambaNova response.

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

        return attrs

    def _extract_finish_reason(self, result) -> Optional[str]:
        """Extract finish reason from SambaNova response.

        Args:
            result: The SambaNova API response object.

        Returns:
            Optional[str]: The finish reason string or None if not available.
        """
        try:
            if hasattr(result, "choices") and result.choices:
                first_choice = result.choices[0]
                if hasattr(first_choice, "finish_reason"):
                    return first_choice.finish_reason
        except Exception as e:
            logger.debug("Failed to extract finish_reason: %s", e)
        return None
