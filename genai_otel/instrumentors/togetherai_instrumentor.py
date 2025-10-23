"""OpenTelemetry instrumentor for the Together AI SDK.

This instrumentor automatically traces completion calls to Together AI models,
capturing relevant attributes such as the model name and token usage.
"""

import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class TogetherAIInstrumentor(BaseInstrumentor):
    """Instrumentor for Together AI"""

    def instrument(self, config: OTelConfig):
        """Instrument Together AI SDK if available."""
        self.config = config
        try:
            import together

            # Instrument chat completions (newer API)
            if hasattr(together, "Together"):
                # This is the newer Together SDK with client-based API
                original_init = together.Together.__init__

                def wrapped_init(instance, *args, **kwargs):
                    original_init(instance, *args, **kwargs)
                    self._instrument_client(instance)

                together.Together.__init__ = wrapped_init
                self._instrumented = True
                logger.info("Together AI instrumentation enabled (client-based API)")
            # Fallback to older Complete API if available
            elif hasattr(together, "Complete"):
                original_complete = together.Complete.create

                wrapped_complete = self.create_span_wrapper(
                    span_name="together.complete",
                    extract_attributes=self._extract_complete_attributes,
                )(original_complete)

                together.Complete.create = wrapped_complete
                self._instrumented = True
                logger.info("Together AI instrumentation enabled (Complete API)")

        except ImportError:
            logger.debug("Together AI library not installed, instrumentation will be skipped")
        except Exception as e:
            logger.error("Failed to instrument Together AI: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _instrument_client(self, client):
        """Instrument Together AI client methods."""
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            original_create = client.chat.completions.create

            wrapped_create = self.create_span_wrapper(
                span_name="together.chat.completion",
                extract_attributes=self._extract_chat_attributes,
            )(original_create)

            client.chat.completions.create = wrapped_create

    def _extract_chat_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Extract attributes from Together AI chat completion call.

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

        attrs["gen_ai.system"] = "together"
        attrs["gen_ai.request.model"] = model
        attrs["gen_ai.operation.name"] = "chat"
        attrs["gen_ai.request.message_count"] = len(messages)

        # Optional parameters
        if "temperature" in kwargs:
            attrs["gen_ai.request.temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            attrs["gen_ai.request.top_p"] = kwargs["top_p"]
        if "max_tokens" in kwargs:
            attrs["gen_ai.request.max_tokens"] = kwargs["max_tokens"]

        return attrs

    def _extract_complete_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Extract attributes from Together AI complete call.

        Args:
            instance: The instance (None for class methods).
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}
        model = kwargs.get("model", "unknown")

        attrs["gen_ai.system"] = "together"
        attrs["gen_ai.request.model"] = model
        attrs["gen_ai.operation.name"] = "complete"

        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from Together AI response.

        Together AI uses OpenAI-compatible format with usage field containing:
        - prompt_tokens: Input tokens
        - completion_tokens: Output tokens
        - total_tokens: Total tokens

        Args:
            result: The API response object.

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        try:
            # Handle OpenAI-compatible response format
            if hasattr(result, "usage") and result.usage:
                usage = result.usage
                return {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                }

            return None
        except Exception as e:
            logger.debug("Failed to extract usage from Together AI response: %s", e)
            return None
