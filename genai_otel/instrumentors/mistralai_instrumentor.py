"""OpenTelemetry instrumentor for the Mistral AI SDK (v1.0+).

This instrumentor automatically traces chat calls to Mistral AI models,
capturing relevant attributes such as the model name and token usage.

Supports Mistral SDK v1.0+ with the new API structure:
- Mistral.chat.complete()
- Mistral.chat.stream()
- Mistral.embeddings.create()
"""

import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class MistralAIInstrumentor(BaseInstrumentor):
    """Instrumentor for Mistral AI SDK v1.0+"""

    def instrument(self, config: OTelConfig):
        self.config = config
        try:
            import wrapt
            from mistralai import Mistral

            # Wrap the Mistral client __init__ to instrument each instance
            original_init = Mistral.__init__

            def wrapped_init(wrapped, instance, args, kwargs):
                result = wrapped(*args, **kwargs)
                self._instrument_client(instance)
                return result

            Mistral.__init__ = wrapt.FunctionWrapper(original_init, wrapped_init)
            logger.info("MistralAI instrumentation enabled (v1.0+ SDK)")

        except ImportError:
            logger.warning("mistralai package not available, skipping instrumentation")
        except Exception as e:
            logger.error(f"Failed to instrument mistralai: {e}", exc_info=True)

    def _instrument_client(self, client):
        """Instrument Mistral client instance methods."""
        # Instrument chat.complete()
        if hasattr(client, "chat") and hasattr(client.chat, "complete"):
            original_complete = client.chat.complete
            instrumented_complete = self.create_span_wrapper(
                span_name="mistralai.chat.complete",
                extract_attributes=self._extract_chat_attributes,
            )(original_complete)
            client.chat.complete = instrumented_complete

        # Instrument chat.stream()
        if hasattr(client, "chat") and hasattr(client.chat, "stream"):
            original_stream = client.chat.stream
            instrumented_stream = self.create_span_wrapper(
                span_name="mistralai.chat.stream",
                extract_attributes=self._extract_chat_attributes,
            )(original_stream)
            client.chat.stream = instrumented_stream

        # Instrument embeddings.create()
        if hasattr(client, "embeddings") and hasattr(client.embeddings, "create"):
            original_embeddings = client.embeddings.create
            instrumented_embeddings = self.create_span_wrapper(
                span_name="mistralai.embeddings.create",
                extract_attributes=self._extract_embeddings_attributes,
            )(original_embeddings)
            client.embeddings.create = instrumented_embeddings

    def _extract_chat_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Extract attributes from chat.complete() or chat.stream() call."""
        model = kwargs.get("model", "unknown")
        attributes = {
            "gen_ai.system": "mistralai",
            "gen_ai.request.model": model,
            "gen_ai.request.type": "chat",
        }

        # Add optional parameters
        if "temperature" in kwargs and kwargs["temperature"] is not None:
            attributes["gen_ai.request.temperature"] = kwargs["temperature"]
        if "top_p" in kwargs and kwargs["top_p"] is not None:
            attributes["gen_ai.request.top_p"] = kwargs["top_p"]
        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
            attributes["gen_ai.request.max_tokens"] = kwargs["max_tokens"]

        return attributes

    def _extract_embeddings_attributes(
        self, instance: Any, args: Any, kwargs: Any
    ) -> Dict[str, Any]:
        """Extract attributes from embeddings.create() call."""
        model = kwargs.get("model", "mistral-embed")
        attributes = {
            "gen_ai.system": "mistralai",
            "gen_ai.request.model": model,
            "gen_ai.request.type": "embedding",
        }
        return attributes

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract usage information from Mistral AI response"""
        try:
            if hasattr(result, "usage"):
                usage = result.usage
                return {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                }
        except Exception as e:
            logger.debug(f"Could not extract usage from MistralAI response: {e}")

        return None
