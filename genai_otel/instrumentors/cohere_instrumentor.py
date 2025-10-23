"""OpenTelemetry instrumentor for the Cohere SDK.

This instrumentor automatically traces calls to Cohere models, capturing
relevant attributes such as the model name and token usage.
"""

import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class CohereInstrumentor(BaseInstrumentor):
    """Instrumentor for Cohere"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._cohere_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if cohere library is available."""
        try:
            import cohere

            self._cohere_available = True
            logger.debug("cohere library detected and available for instrumentation")
        except ImportError:
            logger.debug("cohere library not installed, instrumentation will be skipped")
            self._cohere_available = False

    def instrument(self, config: OTelConfig):
        """Instrument cohere if available."""
        if not self._cohere_available:
            logger.debug("Skipping instrumentation - library not available")
            return

        self.config = config
        try:
            import cohere

            original_init = cohere.Client.__init__

            def wrapped_init(instance, *args, **kwargs):
                original_init(instance, *args, **kwargs)
                self._instrument_client(instance)

            cohere.Client.__init__ = wrapped_init
            self._instrumented = True
            logger.info("Cohere instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument Cohere: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _instrument_client(self, client):
        """Instrument Cohere client methods."""
        original_generate = client.generate

        # Wrap using create_span_wrapper
        wrapped_generate = self.create_span_wrapper(
            span_name="cohere.generate",
            extract_attributes=self._extract_generate_attributes,
        )(original_generate)

        client.generate = wrapped_generate

    def _extract_generate_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Extract attributes from Cohere generate call.

        Args:
            instance: The client instance.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}
        model = kwargs.get("model", "command")
        prompt = kwargs.get("prompt", "")

        attrs["gen_ai.system"] = "cohere"
        attrs["gen_ai.request.model"] = model
        attrs["gen_ai.operation.name"] = "generate"
        attrs["gen_ai.request.message_count"] = 1 if prompt else 0

        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from Cohere response.

        Cohere responses include meta.tokens with:
        - input_tokens: Input tokens
        - output_tokens: Output tokens

        Args:
            result: The API response object.

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        try:
            # Handle object response
            if hasattr(result, "meta") and result.meta:
                meta = result.meta
                # Check for tokens object
                if hasattr(meta, "tokens") and meta.tokens:
                    tokens = meta.tokens
                    input_tokens = getattr(tokens, "input_tokens", 0)
                    output_tokens = getattr(tokens, "output_tokens", 0)

                    if input_tokens or output_tokens:
                        return {
                            "prompt_tokens": int(input_tokens) if input_tokens else 0,
                            "completion_tokens": int(output_tokens) if output_tokens else 0,
                            "total_tokens": int(input_tokens or 0) + int(output_tokens or 0),
                        }
                # Fallback to billed_units
                elif hasattr(meta, "billed_units") and meta.billed_units:
                    billed = meta.billed_units
                    input_tokens = getattr(billed, "input_tokens", 0)
                    output_tokens = getattr(billed, "output_tokens", 0)

                    if input_tokens or output_tokens:
                        return {
                            "prompt_tokens": int(input_tokens) if input_tokens else 0,
                            "completion_tokens": int(output_tokens) if output_tokens else 0,
                            "total_tokens": int(input_tokens or 0) + int(output_tokens or 0),
                        }

            return None
        except Exception as e:
            logger.debug("Failed to extract usage from Cohere response: %s", e)
            return None
