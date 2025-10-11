"""OpenTelemetry instrumentor for the Ollama library.

This instrumentor automatically traces calls to Ollama models for both
generation and chat functionalities, capturing relevant attributes such as
the model name.
"""

from typing import Dict, Optional
import logging


from .base import BaseInstrumentor
from ..config import OTelConfig

logger = logging.getLogger(__name__)


class OllamaInstrumentor(BaseInstrumentor):
    """Instrumentor for Ollama"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._ollama_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if Ollama library is available."""
        try:
            import ollama

            self._ollama_available = True
            logger.debug("Ollama library detected and available for instrumentation")
        except ImportError:
            logger.debug("Ollama library not installed, instrumentation will be skipped")
            self._ollama_available = False

    def instrument(self, config: OTelConfig):
        self.config = config
        try:
            import ollama

            original_generate = ollama.generate
            original_chat = ollama.chat

            def wrapped_generate(*args, **kwargs):
                with self.tracer.start_as_current_span("ollama.generate") as span:
                    model = kwargs.get("model", "unknown")

                    span.set_attribute("gen_ai.system", "ollama")
                    span.set_attribute("gen_ai.request.model", model)

                    self.request_counter.add(1, {"model": model, "provider": "ollama"})

                    result = original_generate(*args, **kwargs)
                    return result

            def wrapped_chat(*args, **kwargs):
                with self.tracer.start_as_current_span("ollama.chat") as span:
                    model = kwargs.get("model", "unknown")

                    span.set_attribute("gen_ai.system", "ollama")
                    span.set_attribute("gen_ai.request.model", model)

                    self.request_counter.add(1, {"model": model, "provider": "ollama"})

                    result = original_chat(*args, **kwargs)
                    return result

            ollama.generate = wrapped_generate
            ollama.chat = wrapped_chat

        except ImportError:
            pass

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        return None
