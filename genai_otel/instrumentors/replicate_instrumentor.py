"""OpenTelemetry instrumentor for the Replicate API client.

This instrumentor automatically traces calls to Replicate models, capturing
relevant attributes such as the model name.

Note: Replicate uses hardware-based pricing (per second of GPU/CPU time),
not token-based pricing. Cost tracking is not applicable as the pricing model
is fundamentally different from token-based LLM APIs.
"""

import contextvars
import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)

# Replicate model references are freeform "owner/repo" slugs with no naming
# convention Replicate itself enforces, so "embed" alone misses real,
# commonly-deployed embedding models (e.g. "nateraw/bge-large-en-v1.5" has no
# "embed" substring at all). These additional markers are established
# embedding-only model architecture families (BAAI BGE, intfloat E5, Alibaba
# GTE, sentence-transformers MPNet/MiniLM) that aren't used for chat/generation,
# so matching on them doesn't risk misclassifying a generation model the way a
# broader guess (e.g. "text") would.
_EMBEDDING_MODEL_MARKERS = (
    "embed",
    "bge-",
    "e5-",
    "gte-",
    "mpnet-base",
    "minilm",
    "sentence-transformers",
)

# Replicate's `run()` is a single generic entry point for any model, so a
# response shaped like a bare list of numbers is genuinely ambiguous - it
# could be an embedding vector, audio samples, bounding boxes, or numeric
# predictions. _extract_response_attributes() has no access to the request,
# so this carries the classification decided in _extract_run_attributes()
# across to it, scoped per call (thread/task) like base.py's own
# _INNER_MEASUREMENT contextvar.
_LAST_REQUEST_IS_EMBEDDING: contextvars.ContextVar = contextvars.ContextVar(
    "genai_otel_replicate_is_embedding", default=False
)


class ReplicateInstrumentor(BaseInstrumentor):
    """Instrumentor for Replicate.

    Note: Replicate uses hardware-based pricing ($/second), not token-based.
    Cost tracking returns None as pricing is based on execution time and hardware type.
    """

    def instrument(self, config: OTelConfig):
        """Instrument Replicate SDK if available."""
        self.config = config
        try:
            import replicate

            # Idempotency guard: never stack wrappers if instrument() runs twice.
            if getattr(replicate, "_genai_otel_replicate_instrumented", False) is True:
                logger.debug("Replicate already instrumented, skipping")
                self._instrumented = True
                return

            original_run = replicate.run

            # Wrap using create_span_wrapper
            wrapped_run = self.create_span_wrapper(
                span_name="replicate.run",
                extract_attributes=self._extract_run_attributes,
            )(original_run)

            replicate.run = wrapped_run
            try:
                replicate._genai_otel_replicate_instrumented = True
            except Exception:  # noqa: BLE001
                pass
            self._instrumented = True
            logger.info("Replicate instrumentation enabled")

        except ImportError:
            logger.debug("Replicate library not installed, instrumentation will be skipped")
        except Exception as e:
            logger.error("Failed to instrument Replicate: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _extract_run_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Extract attributes from Replicate run call.

        Args:
            instance: The instance (None for module-level functions).
            args: Positional arguments (first arg is typically the model).
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}
        model = args[0] if args else kwargs.get("model", "unknown")

        attrs["gen_ai.system"] = "replicate"
        attrs["gen_ai.request.model"] = model

        # Replicate's `run()` is a single generic entry point for any hosted
        # model, so there's no dedicated embeddings method to hook the way
        # other providers' SDKs offer. The model reference is the only
        # available classification signal.
        model_input = kwargs.get("input")
        if model_input is None and len(args) > 1:
            model_input = args[1]
        model_lower = str(model).lower()
        is_embedding = any(marker in model_lower for marker in _EMBEDDING_MODEL_MARKERS)
        _LAST_REQUEST_IS_EMBEDDING.set(is_embedding)
        if is_embedding:
            attrs["gen_ai.operation.name"] = "embeddings"
            attrs["gen_ai.request.type"] = "embedding"
            attrs["gen_ai.request.input_count"] = self._count_inputs(model_input)
        else:
            attrs["gen_ai.operation.name"] = "run"

        return attrs

    @staticmethod
    def _count_inputs(model_input: Any) -> int:
        """Count embedding inputs from a Replicate `input={...}` payload.

        Replicate has no fixed input schema (it's per-model), so this only
        recognizes the field names embedding models conventionally use.
        """
        if not isinstance(model_input, dict):
            return 1 if model_input is not None else 0
        for key in ("texts", "inputs"):
            value = model_input.get(key)
            if isinstance(value, (list, tuple)):
                return len(value)
        for key in ("text", "input", "prompt"):
            if model_input.get(key) is not None:
                return 1
        return 0

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        """Extract embedding response shape for embedding model runs.

        A Replicate embedding model's `run()` result is typically the raw
        vector itself (a flat list of numbers) for a single input, or a list
        of vectors for a batch - there's no wrapper object to introspect. Only
        applies this interpretation when the request was itself classified as
        an embedding call (see `_LAST_REQUEST_IS_EMBEDDING`); otherwise a
        non-embedding model's list-shaped output (audio samples, bounding
        boxes, numeric predictions, ...) would be mislabeled the same way.
        """
        if not _LAST_REQUEST_IS_EMBEDDING.get():
            return {}
        if not isinstance(result, (list, tuple)) or not result:
            return {}
        first = result[0]
        if isinstance(first, (list, tuple)):
            attrs: Dict[str, Any] = {"gen_ai.response.embedding_count": len(result)}
            if first:
                attrs["gen_ai.response.vector_size"] = len(first)
            return attrs
        if isinstance(first, (int, float)):
            # A single flat vector, not a batch of them.
            return {
                "gen_ai.response.embedding_count": 1,
                "gen_ai.response.vector_size": len(result),
            }
        return {}

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from Replicate response.

        Note: Replicate uses hardware-based pricing ($/second of GPU/CPU time),
        not token-based pricing. Returns None as the pricing model is incompatible
        with token-based cost calculation.

        Args:
            result: The API response.

        Returns:
            None: Replicate uses hardware-based pricing, not token-based.
        """
        # Replicate uses hardware-based pricing ($/second), not tokens
        # Cannot track costs with token-based calculator
        return None
