"""OpenTelemetry instrumentor for the Cohere SDK.

This instrumentor automatically traces calls to Cohere models, capturing
relevant attributes such as the model name.
"""

from typing import Dict, Optional
import logging

from .base import BaseInstrumentor
from ..config import OTelConfig

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
        """Instrument  cohere available if available."""
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

        except ImportError:
            pass

    def _instrument_client(self, client):
        original_generate = client.generate

        def wrapped_generate(*args, **kwargs):
            with self.tracer.start_as_current_span("cohere.generate") as span:
                model = kwargs.get("model", "command")

                span.set_attribute("gen_ai.system", "cohere")
                span.set_attribute("gen_ai.request.model", model)

                self.request_counter.add(1, {"model": model, "provider": "cohere"})

                result = original_generate(*args, **kwargs)
                return result

        client.generate = wrapped_generate

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        return None
