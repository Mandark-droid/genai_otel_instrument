"""OpenTelemetry instrumentor for the Replicate API client.

This instrumentor automatically traces calls to Replicate models, capturing
relevant attributes such as the model name.

Note: Replicate uses hardware-based pricing (per second of GPU/CPU time),
not token-based pricing. Cost tracking is not applicable as the pricing model
is fundamentally different from token-based LLM APIs.
"""

import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class ReplicateInstrumentor(BaseInstrumentor):
    """Instrumentor for Replicate.

    Note: Replicate uses hardware-based pricing ($/second), not token-based.
    Cost tracking returns None as pricing is based on execution time and hardware type.
    """

    def instrument(self, config: OTelConfig):
        """Instrument Replicate SDK if available."""
        self.config = config
        try:
            import replicate

            original_run = replicate.run

            # Wrap using create_span_wrapper
            wrapped_run = self.create_span_wrapper(
                span_name="replicate.run",
                extract_attributes=self._extract_run_attributes,
            )(original_run)

            replicate.run = wrapped_run
            self._instrumented = True
            logger.info("Replicate instrumentation enabled")

        except ImportError:
            logger.debug("Replicate library not installed, instrumentation will be skipped")
        except Exception as e:
            logger.error("Failed to instrument Replicate: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _extract_run_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Extract attributes from Replicate run call.

        Args:
            instance: The instance (None for module-level functions).
            args: Positional arguments (first arg is typically the model).
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}
        model = args[0] if args else kwargs.get("model", "unknown")

        attrs["gen_ai.system"] = "replicate"
        attrs["gen_ai.request.model"] = model
        attrs["gen_ai.operation.name"] = "run"

        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from Replicate response.

        Note: Replicate uses hardware-based pricing ($/second of GPU/CPU time),
        not token-based pricing. Returns None as the pricing model is incompatible
        with token-based cost calculation.

        Args:
            result: The API response.

        Returns:
            None: Replicate uses hardware-based pricing, not token-based.
        """
        # Replicate uses hardware-based pricing ($/second), not tokens
        # Cannot track costs with token-based calculator
        return None
