"""OpenTelemetry instrumentor for the Together AI SDK.

This instrumentor automatically traces completion calls to Together AI models,
capturing relevant attributes such as the model name.
"""

from typing import Dict, Optional

from .base import BaseInstrumentor
from ..config import OTelConfig


class TogetherAIInstrumentor(BaseInstrumentor):
    """Instrumentor for Together AI"""

    def instrument(self, config: OTelConfig):
        self.config = config
        try:
            import together

            original_complete = together.Complete.create

            def wrapped_complete(*args, **kwargs):
                with self.tracer.start_as_current_span("together.complete") as span:
                    model = kwargs.get("model", "unknown")

                    span.set_attribute("gen_ai.system", "together")
                    span.set_attribute("gen_ai.request.model", model)

                    self.request_counter.add(1, {"model": model, "provider": "together"})

                    result = original_complete(*args, **kwargs)
                    return result

            together.Complete.create = wrapped_complete

        except ImportError:
            pass

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        return None
