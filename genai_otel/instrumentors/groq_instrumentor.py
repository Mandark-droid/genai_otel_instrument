from .base import BaseInstrumentor
from ..config import OTelConfig
from typing import Dict, Optional


class GroqInstrumentor(BaseInstrumentor):
    """Instrumentor for Groq"""

    def instrument(self, config: OTelConfig):
        self.config = config
        try:
            from groq import Groq

            original_init = Groq.__init__

            def wrapped_init(instance, *args, **kwargs):
                original_init(instance, *args, **kwargs)
                self._instrument_client(instance)

            Groq.__init__ = wrapped_init

        except ImportError:
            pass

    def _instrument_client(self, client):
        original_create = client.chat.completions.create

        def wrapped_create(*args, **kwargs):
            with self.tracer.start_as_current_span("groq.chat.completions") as span:
                model = kwargs.get("model", "unknown")

                span.set_attribute("gen_ai.system", "groq")
                span.set_attribute("gen_ai.request.model", model)

                self.request_counter.add(1, {"model": model, "provider": "groq"})

                result = original_create(*args, **kwargs)
                self._record_result_metrics(span, result, 0)
                return result

        client.chat.completions.create = wrapped_create

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        if hasattr(result, "usage"):
            return {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens
            }
        return None