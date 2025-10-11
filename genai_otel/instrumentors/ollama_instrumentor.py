from .base import BaseInstrumentor
from ..config import OTelConfig
from typing import Dict, Optional


class OllamaInstrumentor(BaseInstrumentor):
    """Instrumentor for Ollama"""

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