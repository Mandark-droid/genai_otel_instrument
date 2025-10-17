"""OpenTelemetry instrumentor for the Ollama library.

This instrumentor automatically traces calls to Ollama models for both
generation and chat functionalities, capturing relevant attributes such as
the model name.
"""

import logging
from typing import Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class OllamaInstrumentor(BaseInstrumentor):
    """Instrumentor for Ollama"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._ollama_available = False
        self._ollama_module = None
        self._original_generate = None  # Add this
        self._original_chat = None  # Add this
        self._check_availability()

    def _check_availability(self):
        """Check if Ollama library is available."""
        try:
            import ollama

            self._ollama_available = True
            self._ollama_module = ollama
            logger.debug("Ollama library detected and available for instrumentation")
        except ImportError:
            logger.debug("Ollama library not installed, instrumentation will be skipped")
            self._ollama_available = False
            self._ollama_module = None

    def instrument(self, config: OTelConfig):
        """Instrument the Ollama library."""
        self.config = config

        if not self._ollama_available or self._ollama_module is None:
            return

        # Store original methods
        self._original_generate = self._ollama_module.generate
        self._original_chat = self._ollama_module.chat

        def wrapped_generate(*args, **kwargs):
            with self.tracer.start_as_current_span("ollama.generate") as span:
                model = kwargs.get("model", "unknown")

                span.set_attribute("gen_ai.system", "ollama")
                span.set_attribute("gen_ai.request.model", model)

                if self.request_counter:
                    self.request_counter.add(1, {"model": model, "provider": "ollama"})

                result = self._original_generate(*args, **kwargs)
                return result

        def wrapped_chat(*args, **kwargs):
            with self.tracer.start_as_current_span("ollama.chat") as span:
                model = kwargs.get("model", "unknown")

                span.set_attribute("gen_ai.system", "ollama")
                span.set_attribute("gen_ai.request.model", model)

                if self.request_counter:
                    self.request_counter.add(1, {"model": model, "provider": "ollama"})

                result = self._original_chat(*args, **kwargs)
                return result

        self._ollama_module.generate = wrapped_generate
        self._ollama_module.chat = wrapped_chat

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        return None
