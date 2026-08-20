"""OpenTelemetry instrumentation for the Azure AI Inference SDK."""

import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class AzureAIInferenceInstrumentor(BaseInstrumentor):
    """Trace ``azure.ai.inference.EmbeddingsClient.embed`` calls."""

    def __init__(self):
        super().__init__()
        try:
            from azure.ai.inference import EmbeddingsClient  # noqa: F401

            self._available = True
        except ImportError:
            self._available = False

    def instrument(self, config: OTelConfig):
        self.config = config
        if not self._available:
            return
        try:
            from azure.ai.inference import EmbeddingsClient

            if getattr(EmbeddingsClient, "_genai_otel_azure_inference_instrumented", False):
                self._instrumented = True
                return
            original_embed = EmbeddingsClient.embed
            EmbeddingsClient.embed = self.create_span_wrapper(
                span_name="azure.ai.inference.embeddings",
                extract_attributes=self._extract_embedding_attributes,
            )(original_embed)
            EmbeddingsClient._genai_otel_azure_inference_instrumented = True
            self._instrumented = True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to instrument Azure AI Inference: %s", exc, exc_info=True)
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
        model = kwargs.get("model") or getattr(instance, "model", "unknown")
        value = kwargs.get("input")
        if value is None and args:
            value = args[0]
        return {
            "gen_ai.system": "azure_ai_inference",
            "gen_ai.request.model": str(model),
            "gen_ai.operation.name": "embeddings",
            "gen_ai.request.type": "embedding",
            "gen_ai.request.input_count": self._count_inputs(value),
        }

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        data = result.get("data") if isinstance(result, dict) else getattr(result, "data", None)
        if data is None:
            return {}
        try:
            items = list(data)
            attrs: Dict[str, Any] = {"gen_ai.response.embedding_count": len(items)}
            if items:
                first = items[0]
                vector = (
                    first.get("embedding")
                    if isinstance(first, dict)
                    else getattr(first, "embedding", None)
                )
                if isinstance(vector, (list, tuple)):
                    attrs["gen_ai.response.vector_size"] = len(vector)
            return attrs
        except (TypeError, AttributeError, ValueError):
            return {}

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        usage = result.get("usage") if isinstance(result, dict) else getattr(result, "usage", None)
        if usage is None:
            return None
        get = usage.get if isinstance(usage, dict) else lambda key: getattr(usage, key, None)
        prompt = get("prompt_tokens") or get("input_tokens") or 0
        total = get("total_tokens") or prompt
        return {"prompt_tokens": int(prompt), "completion_tokens": 0, "total_tokens": int(total)}
