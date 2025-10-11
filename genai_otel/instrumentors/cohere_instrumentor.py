from .base import BaseInstrumentor
from ..config import OTelConfig
from typing import Dict, Optional


class CohereInstrumentor(BaseInstrumentor):
    """Instrumentor for Cohere"""

    def instrument(self, config: OTelConfig):
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
