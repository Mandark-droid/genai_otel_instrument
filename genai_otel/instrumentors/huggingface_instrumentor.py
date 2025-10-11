from .base import BaseInstrumentor
from ..config import OTelConfig
from typing import Dict, Optional


class HuggingFaceInstrumentor(BaseInstrumentor):
    """Instrumentor for HuggingFace Transformers"""

    def instrument(self, config: OTelConfig):
        self.config = config
        try:
            from transformers import pipeline, PreTrainedModel

            original_pipeline = pipeline

            def wrapped_pipeline(*args, **kwargs):
                pipe = original_pipeline(*args, **kwargs)
                original_call = pipe.__call__

                def wrapped_call(*call_args, **call_kwargs):
                    with self.tracer.start_as_current_span("huggingface.pipeline") as span:
                        task = getattr(pipe, "task", "unknown")
                        model = getattr(pipe.model, "name_or_path", "unknown")

                        span.set_attribute("gen_ai.system", "huggingface")
                        span.set_attribute("gen_ai.request.model", model)
                        span.set_attribute("huggingface.task", task)

                        self.request_counter.add(1, {"model": model, "provider": "huggingface"})

                        result = original_call(*call_args, **call_kwargs)
                        return result

                pipe.__call__ = wrapped_call
                return pipe

            import transformers
            transformers.pipeline = wrapped_pipeline

        except ImportError:
            pass

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        return None
