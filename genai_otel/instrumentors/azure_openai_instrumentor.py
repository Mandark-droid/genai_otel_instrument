from .base import BaseInstrumentor
from ..config import OTelConfig
import wrapt
from typing import Dict, Optional


class AzureOpenAIInstrumentor(BaseInstrumentor):
    """Instrumentor for Azure OpenAI"""

    def instrument(self, config: OTelConfig):
        self.config = config
        try:
            from azure.ai.openai import OpenAIClient

            original_complete = OpenAIClient.complete

            def wrapped_complete(instance, *args, **kwargs):
                with self.tracer.start_as_current_span("azure.openai.complete") as span:
                    model = kwargs.get("model", "unknown")

                    span.set_attribute("gen_ai.system", "azure_openai")
                    span.set_attribute("gen_ai.request.model", model)

                    self.request_counter.add(1, {"model": model, "provider": "azure_openai"})

                    result = original_complete(instance, *args, **kwargs)
                    self._record_result_metrics(span, result, 0)
                    return result

            OpenAIClient.complete = wrapped_complete

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
