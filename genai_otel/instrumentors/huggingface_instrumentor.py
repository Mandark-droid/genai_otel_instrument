"""OpenTelemetry instrumentor for HuggingFace Transformers library.

This instrumentor automatically traces calls made through HuggingFace pipelines,
capturing relevant attributes such as the model name and task type.
"""

import logging
import types
from typing import Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

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
        super().instrument(config)

        if not self._transformers_available:
            return

        try:
            import importlib

            transformers_module = importlib.import_module("transformers")
            original_pipeline = transformers_module.pipeline

            def wrapped_pipeline(*args, **kwargs):
                pipe = original_pipeline(*args, **kwargs)

                class WrappedPipeline:
                    def __init__(self, original_pipe):
                        self._original_pipe = original_pipe

                    def __call__(self, *call_args, **call_kwargs):
                        with config.tracer.start_as_current_span("huggingface.pipeline") as span:
                            task = getattr(self._original_pipe, "task", "unknown")
                            model = getattr(
                                getattr(self._original_pipe, "model", None),
                                "name_or_path",
                                "unknown",
                            )

                            span.set_attribute("gen_ai.system", "huggingface")
                            span.set_attribute("gen_ai.request.model", model)
                            span.set_attribute("huggingface.task", task)

                            config.request_counter.add(
                                1, {"model": model, "provider": "huggingface"}
                            )

                            result = self._original_pipe(*call_args, **call_kwargs)
                            return result

                    def __getattr__(self, name):
                        # Delegate all other attribute access to the original pipe
                        return getattr(self._original_pipe, name)

                return WrappedPipeline(pipe)

            transformers_module.pipeline = wrapped_pipeline

        except ImportError:
            pass

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        return None
