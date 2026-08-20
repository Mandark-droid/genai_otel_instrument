"""OpenTelemetry instrumentor for the Together AI SDK.

This instrumentor automatically traces completion calls to Together AI models,
capturing relevant attributes such as the model name and token usage.
"""

import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class TogetherAIInstrumentor(BaseInstrumentor):
    """Instrumentor for Together AI"""

    def instrument(self, config: OTelConfig):
        """Instrument Together AI SDK if available."""
        self.config = config
        try:
            import together

            # Idempotency guard: never stack wrappers if instrument() runs twice.
            if getattr(together, "_genai_otel_together_instrumented", False) is True:
                logger.debug("Together AI already instrumented, skipping")
                self._instrumented = True
                return

            # Instrument chat completions (newer API)
            if hasattr(together, "Together"):
                # This is the newer Together SDK with client-based API
                original_init = together.Together.__init__

                def wrapped_init(instance, *args, **kwargs):
                    original_init(instance, *args, **kwargs)
                    self._instrument_client(instance)

                together.Together.__init__ = wrapped_init
                async_client = getattr(together, "AsyncTogether", None)
                if isinstance(async_client, type) and not getattr(
                    async_client, "_genai_otel_together_instrumented", False
                ):
                    original_async_init = async_client.__init__

                    def wrapped_async_init(instance, *args, **kwargs):
                        original_async_init(instance, *args, **kwargs)
                        self._instrument_client(instance)

                    async_client.__init__ = wrapped_async_init
                    async_client._genai_otel_together_instrumented = True
                self._instrumented = True
                logger.info("Together AI instrumentation enabled (client-based API)")
            # Fallback to older Complete API if available
            elif hasattr(together, "Complete"):
                original_complete = together.Complete.create

                wrapped_complete = self.create_span_wrapper(
                    span_name="together.complete",
                    extract_attributes=self._extract_complete_attributes,
                )(original_complete)

                together.Complete.create = wrapped_complete
                self._instrumented = True
                logger.info("Together AI instrumentation enabled (Complete API)")

            if self._instrumented:
                try:
                    together._genai_otel_together_instrumented = True
                except Exception:  # noqa: BLE001
                    pass

        except ImportError:
            logger.debug("Together AI library not installed, instrumentation will be skipped")
        except Exception as e:
            logger.error("Failed to instrument Together AI: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _instrument_client(self, client):
        """Instrument Together AI client methods."""
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            original_create = client.chat.completions.create

            wrapped_create = self.create_span_wrapper(
                span_name="together.chat.completion",
                extract_attributes=self._extract_chat_attributes,
            )(original_create)

            client.chat.completions.create = wrapped_create

        embeddings = getattr(client, "embeddings", None)
        if embeddings is not None and hasattr(embeddings, "create"):
            original_create = embeddings.create
            embeddings.create = self.create_span_wrapper(
                span_name="together.embeddings",
                extract_attributes=self._extract_embedding_attributes,
            )(original_create)

    @staticmethod
    def _count_embedding_inputs(value: Any) -> int:
        if isinstance(value, str):
            return 1
        if isinstance(value, (list, tuple)):
            if not value:
                return 0
            if all(isinstance(item, int) for item in value):
                return 1
            return len(value)
        return 1 if value is not None else 0

    def _extract_embedding_attributes(
        self, instance: Any, args: Any, kwargs: Any
    ) -> Dict[str, Any]:
        """Extract canonical attributes from Together's embeddings endpoint."""
        model = kwargs.get("model", "unknown")
        value = kwargs.get("input")
        if value is None and args:
            value = args[0]
        return {
            "gen_ai.system": "together",
            "gen_ai.request.model": str(model),
            "gen_ai.operation.name": "embeddings",
            "gen_ai.request.type": "embedding",
            "gen_ai.request.input_count": self._count_embedding_inputs(value),
        }

    def _extract_chat_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Extract attributes from Together AI chat completion call.

        Args:
            instance: The client instance.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])

        attrs["gen_ai.system"] = "together"
        attrs["gen_ai.request.model"] = model
        attrs["gen_ai.operation.name"] = "chat"
        attrs["gen_ai.request.message_count"] = len(messages)

        # Optional parameters
        if "temperature" in kwargs:
            attrs["gen_ai.request.temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            attrs["gen_ai.request.top_p"] = kwargs["top_p"]
        if "max_tokens" in kwargs:
            attrs["gen_ai.request.max_tokens"] = kwargs["max_tokens"]

        return attrs

    def _extract_complete_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Extract attributes from Together AI complete call.

        Args:
            instance: The instance (None for class methods).
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}
        model = kwargs.get("model", "unknown")

        attrs["gen_ai.system"] = "together"
        attrs["gen_ai.request.model"] = model
        attrs["gen_ai.operation.name"] = "complete"

        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from Together AI response.

        Together AI uses OpenAI-compatible format with usage field containing:
        - prompt_tokens: Input tokens
        - completion_tokens: Output tokens
        - total_tokens: Total tokens

        Args:
            result: The API response object.

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        try:
            # Handle OpenAI-compatible response format
            if hasattr(result, "usage") and result.usage:
                usage = result.usage
                return {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                }

            return None
        except Exception as e:
            logger.debug("Failed to extract usage from Together AI response: %s", e)
            return None

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        """Extract response model and embedding dimensions when available."""
        attrs: Dict[str, Any] = {}
        try:
            if isinstance(result, dict):
                data = result.get("data", result.get("embeddings"))
            else:
                data = getattr(result, "data", getattr(result, "embeddings", None))
            if data is not None:
                items = list(data)
                attrs["gen_ai.response.embedding_count"] = len(items)
                if items:
                    first = items[0]
                    vector = (
                        first.get("embedding")
                        if isinstance(first, dict)
                        else getattr(first, "embedding", first)
                    )
                    if isinstance(vector, (list, tuple)):
                        attrs["gen_ai.response.vector_size"] = len(vector)
            model = (
                result.get("model") if isinstance(result, dict) else getattr(result, "model", None)
            )
            if model:
                attrs["gen_ai.response.model"] = str(model)
        except (TypeError, AttributeError, ValueError) as e:
            logger.debug("Failed to extract Together embedding response: %s", e)
        return attrs
