from .base import BaseInstrumentor
from ..config import OTelConfig
from typing import Dict, Optional


class MistralAIInstrumentor(BaseInstrumentor):
    """Instrumentor for Mistral AI"""

    def instrument(self, config: OTelConfig):
        self.config = config
        try:
            from mistralai.client import MistralClient

            original_chat = MistralClient.chat

            def wrapped_chat(instance, *args, **kwargs):
                with self.tracer.start_as_current_span("mistralai.chat") as span:
                    model = kwargs.get("model", "unknown")

                    span.set_attribute("gen_ai.system", "mistralai")
                    span.set_attribute("gen_ai.request.model", model)

                    self.request_counter.add(1, {"model": model, "provider": "mistralai"})

                    result = original_chat(instance, *args, **kwargs)
                    self._record_result_metrics(span, result, 0)
                    return result

            MistralClient.chat = wrapped_chat

        except ImportError:
            pass

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        if hasattr(result, "usage"):
            return {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens
            }
        return None
