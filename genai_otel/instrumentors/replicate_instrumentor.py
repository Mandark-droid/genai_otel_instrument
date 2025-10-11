from .base import BaseInstrumentor
from ..config import OTelConfig
from typing import Dict, Optional


class ReplicateInstrumentor(BaseInstrumentor):
    """Instrumentor for Replicate"""

    def instrument(self, config: OTelConfig):
        self.config = config
        try:
            import replicate

            original_run = replicate.run

            def wrapped_run(*args, **kwargs):
                with self.tracer.start_as_current_span("replicate.run") as span:
                    model = args[0] if args else "unknown"

                    span.set_attribute("gen_ai.system", "replicate")
                    span.set_attribute("gen_ai.request.model", model)

                    self.request_counter.add(1, {"model": model, "provider": "replicate"})

                    result = original_run(*args, **kwargs)
                    return result

            replicate.run = wrapped_run

        except ImportError:
            pass

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        return None
