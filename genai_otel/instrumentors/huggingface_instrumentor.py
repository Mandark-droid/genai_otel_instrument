"""OpenTelemetry instrumentor for HuggingFace Transformers library.

This instrumentor automatically traces calls made through HuggingFace pipelines,
capturing relevant attributes such as the model name and task type.
"""

from typing import Dict, Optional
import logging
from .base import BaseInstrumentor
from ..config import OTelConfig

logger = logging.getLogger(__name__)


class HuggingFaceInstrumentor(BaseInstrumentor):
    """Instrumentor for HuggingFace Transformers"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._transformers_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if Transformers library is available."""
        try:
            import transformers

            self._transformers_available = True
            logger.debug("Transformers library detected and available for instrumentation")
        except ImportError:
            logger.debug("Transformers library not installed, instrumentation will be skipped")
            self._transformers_available = False

    def instrument(self, config: OTelConfig):
        self.config = config
        try:
            from transformers import pipeline

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

            transformers.pipeline = wrapped_pipeline

        except ImportError:
            pass

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        return None
