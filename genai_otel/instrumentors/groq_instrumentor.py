"""OpenTelemetry instrumentor for the Groq SDK.

This instrumentor automatically traces chat completion calls to Groq models,
capturing relevant attributes such as the model name and token usage.
"""

import logging
from typing import Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class GroqInstrumentor(BaseInstrumentor):
    """Instrumentor for Groq"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._groq_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if Groq library is available."""
        try:
            import groq

            self._groq_available = True
            logger.debug("Groq library detected and available for instrumentation")
        except ImportError:
            logger.debug("Groq library not installed, instrumentation will be skipped")
            self._groq_available = False

    def instrument(self, config: OTelConfig):
        """Instrument Groq SDK if available.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._groq_available:
            logger.debug("Skipping Groq instrumentation - library not available")
            return

        self.config = config

        try:
            import groq

            original_init = groq.Groq.__init__

            def wrapped_init(instance, *args, **kwargs):
                original_init(instance, *args, **kwargs)
                self._instrument_client(instance)
                return instance

            groq.Groq.__init__ = wrapped_init
            self._instrumented = True
            logger.info("Groq instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument Groq: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _instrument_client(self, client):
        """Instrument Groq client methods.

        Args:
            client: The Groq client instance to instrument.
        """
        original_create = client.chat.completions.create

        def wrapped_create(*args, **kwargs):
            with self.tracer.start_as_current_span("groq.chat.completions") as span:
                model = kwargs.get("model", "unknown")

                span.set_attribute("gen_ai.system", "groq")
                span.set_attribute("gen_ai.request.model", model)

                if self.request_counter:
                    self.request_counter.add(1, {"model": model, "provider": "groq"})

                result = original_create(*args, **kwargs)
                self._record_result_metrics(span, result, 0)
                return result

        client.chat.completions.create = wrapped_create

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from Groq response.

        Args:
            result: The API response object.

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        if hasattr(result, "usage"):
            return {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
            }
        return None
