"""OpenTelemetry instrumentor for Azure OpenAI SDK.

This instrumentor automatically traces calls to Azure OpenAI models, capturing
relevant attributes such as model name and token usage.
"""

import logging
from typing import Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class AzureOpenAIInstrumentor(BaseInstrumentor):
    """Instrumentor for Azure OpenAI"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._azure_openai_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if Azure AI OpenAI library is available."""
        try:
            import azure.ai.openai  # Moved to top

            self._azure_openai_available = True
            logger.debug("Azure AI OpenAI library detected and available for instrumentation")
        except ImportError:
            logger.debug("Azure AI OpenAI library not installed, instrumentation will be skipped")
            self._azure_openai_available = False

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

                    if self.request_counter:
                        self.request_counter.add(1, {"model": model, "provider": "azure_openai"})

                    result = original_complete(instance, *args, **kwargs)
                    self._record_result_metrics(span, result, 0)
                    return result

            OpenAIClient.complete = wrapped_complete

        except ImportError:
            pass

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        if hasattr(result, "usage") and result.usage:
            return {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
            }
        return None
