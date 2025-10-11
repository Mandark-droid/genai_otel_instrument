"""OpenTelemetry instrumentor for Google Vertex AI SDK.

This instrumentor automatically traces content generation calls to Vertex AI models,
capturing relevant attributes such as the model name.
"""

from typing import Dict, Optional

from .base import BaseInstrumentor
from ..config import OTelConfig


class VertexAIInstrumentor(BaseInstrumentor):
    """Instrumentor for Google Vertex AI"""

    def instrument(self, config: OTelConfig):
        self.config = config
        try:
            from vertexai.preview.generative_models import GenerativeModel

            original_generate = GenerativeModel.generate_content

            def wrapped_generate(instance, *args, **kwargs):
                with self.tracer.start_as_current_span("vertexai.generate_content") as span:
                    model_name = getattr(instance, "_model_name", "unknown")

                    span.set_attribute("gen_ai.system", "vertexai")
                    span.set_attribute("gen_ai.request.model", model_name)

                    self.request_counter.add(1, {"model": model_name, "provider": "vertexai"})

                    result = original_generate(instance, *args, **kwargs)
                    return result

            GenerativeModel.generate_content = wrapped_generate

        except ImportError:
            pass

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        return None
