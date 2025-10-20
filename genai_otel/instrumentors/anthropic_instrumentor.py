"""OpenTelemetry instrumentor for the Anthropic Claude SDK.

This instrumentor automatically traces calls to the Anthropic API, capturing
relevant attributes such as model name, message count, and token usage.
"""

import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class AnthropicInstrumentor(BaseInstrumentor):
    """Instrumentor for Anthropic Claude SDK"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._anthropic_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if Anthropic library is available."""
        try:
            import anthropic

            self._anthropic_available = True
            logger.debug("Anthropic library detected and available for instrumentation")
        except ImportError:
            logger.debug("Anthropic library not installed, instrumentation will be skipped")
            self._anthropic_available = False

    def instrument(self, config: OTelConfig):
        """Instrument Anthropic SDK if available.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._anthropic_available:
            logger.debug("Skipping Anthropic instrumentation - library not available")
            return

        self.config = config

        try:
            import anthropic
            import wrapt

            if hasattr(anthropic, "Anthropic"):
                original_init = anthropic.Anthropic.__init__

                def wrapped_init(wrapped, instance, args, kwargs):
                    result = wrapped(*args, **kwargs)
                    self._instrument_client(instance)
                    return result

                anthropic.Anthropic.__init__ = wrapt.FunctionWrapper(original_init, wrapped_init)
                self._instrumented = True
                logger.info("Anthropic instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument Anthropic: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _instrument_client(self, client):
        """Instrument Anthropic client methods.

        Args:
            client: The Anthropic client instance to instrument.
        """
        if hasattr(client, "messages") and hasattr(client.messages, "create"):
            original_create = client.messages.create
            instrumented_create_method = self.create_span_wrapper(
                span_name="anthropic.messages.create",
                extract_attributes=self._extract_anthropic_attributes,
            )(original_create)
            client.messages.create = instrumented_create_method

    def _extract_anthropic_attributes(
        self, instance: Any, args: Any, kwargs: Any
    ) -> Dict[str, Any]:
        """Extract attributes from Anthropic API call.

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

        attrs["gen_ai.system"] = "anthropic"
        attrs["gen_ai.request.model"] = model
        attrs["gen_ai.request.message_count"] = len(messages)
        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from Anthropic response.

        Args:
            result: The API response object.

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        if hasattr(result, "usage") and result.usage:
            usage = result.usage
            usage_dict = {
                "prompt_tokens": getattr(usage, "input_tokens", 0),
                "completion_tokens": getattr(usage, "output_tokens", 0),
                "total_tokens": getattr(usage, "input_tokens", 0)
                + getattr(usage, "output_tokens", 0),
            }

            # Extract cache tokens for Anthropic models (Phase 3.2)
            # cache_read_input_tokens: Tokens that were read from cache
            # cache_creation_input_tokens: Tokens that were written to cache
            if hasattr(usage, "cache_read_input_tokens"):
                usage_dict["cache_read_input_tokens"] = getattr(usage, "cache_read_input_tokens", 0)
            if hasattr(usage, "cache_creation_input_tokens"):
                usage_dict["cache_creation_input_tokens"] = getattr(
                    usage, "cache_creation_input_tokens", 0
                )

            return usage_dict
        return None
