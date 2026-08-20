"""OpenTelemetry instrumentation for SentenceTransformers embeddings."""

import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class SentenceTransformersInstrumentor(BaseInstrumentor):
    """Trace ``SentenceTransformer.encode`` calls as embedding spans."""

    def __init__(self):
        super().__init__()
        self._available = False
        try:
            import sentence_transformers  # noqa: F401

            self._available = True
        except ImportError:
            logger.debug("sentence-transformers is not installed, skipping instrumentation")

    def instrument(self, config: OTelConfig):
        self.config = config
        if not self._available:
            return
        try:
            from sentence_transformers import SentenceTransformer

            if getattr(SentenceTransformer, "_genai_otel_st_instrumented", False):
                self._instrumented = True
                return

            original_encode = SentenceTransformer.encode
            SentenceTransformer.encode = self.create_span_wrapper(
                span_name="sentence_transformers.embeddings",
                extract_attributes=self._extract_embedding_attributes,
            )(original_encode)
            SentenceTransformer._genai_otel_st_instrumented = True
            self._instrumented = True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to instrument SentenceTransformers: %s", exc, exc_info=True)
            if config.fail_on_error:
                raise

    @staticmethod
    def _count_inputs(value: Any) -> int:
        if isinstance(value, str):
            return 1
        if isinstance(value, (list, tuple)):
            return len(value)
        return 1 if value is not None else 0

    def _extract_embedding_attributes(
        self, instance: Any, args: Any, kwargs: Any
    ) -> Dict[str, Any]:
        model = getattr(instance, "model_name_or_path", None) or getattr(
            instance, "_model_name", "unknown"
        )
        sentences = kwargs.get("sentences")
        if sentences is None and args:
            sentences = args[0]
        return {
            "gen_ai.system": "sentence-transformers",
            "gen_ai.request.model": str(model),
            "gen_ai.operation.name": "embeddings",
            "gen_ai.request.type": "embedding",
            "gen_ai.request.input_count": self._count_inputs(sentences),
        }

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {}
        shape = getattr(result, "shape", None)
        if shape is not None:
            try:
                dims = tuple(int(value) for value in shape)
                if dims:
                    if len(dims) == 1:
                        # A 1D array is a single embedding vector (e.g.
                        # encode("one sentence")), not `dims[0]` embeddings.
                        attrs["gen_ai.response.embedding_count"] = 1
                        attrs["gen_ai.response.vector_size"] = dims[0]
                    else:
                        attrs["gen_ai.response.embedding_count"] = dims[0]
                        attrs["gen_ai.response.vector_size"] = dims[-1]
                    return attrs
            except (TypeError, ValueError):
                pass
        if isinstance(result, (list, tuple)):
            attrs["gen_ai.response.embedding_count"] = len(result)
            if result and isinstance(result[0], (list, tuple)):
                attrs["gen_ai.response.vector_size"] = len(result[0])
        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        return None
