"""OpenTelemetry instrumentor for Google Vertex AI SDK.

This instrumentor automatically traces content generation calls to Vertex AI models,
capturing relevant attributes such as the model name and token usage.
"""

import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from ._gemini_usage import gemini_usage
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class VertexAIInstrumentor(BaseInstrumentor):
    """Instrumentor for Google Vertex AI"""

    def instrument(self, config: OTelConfig):
        """Instrument Vertex AI SDK if available."""
        self.config = config
        try:
            from vertexai.preview.generative_models import GenerativeModel

            # Idempotency guard: never stack wrappers if instrument() runs twice.
            if getattr(GenerativeModel, "_genai_otel_vertexai_instrumented", False) is True:
                logger.debug("Vertex AI already instrumented, skipping")
                self._instrumented = True
                return

            original_generate = GenerativeModel.generate_content

            # Wrap using create_span_wrapper
            wrapped_generate = self.create_span_wrapper(
                span_name="vertexai.generate_content",
                extract_attributes=self._extract_generate_attributes,
            )(original_generate)

            GenerativeModel.generate_content = wrapped_generate
            try:
                GenerativeModel._genai_otel_vertexai_instrumented = True
            except Exception:  # noqa: BLE001
                pass
            self._instrumented = True
            logger.info("Vertex AI instrumentation enabled")

        except ImportError:
            logger.debug("Vertex AI library not installed, instrumentation will be skipped")
        except Exception as e:
            logger.error("Failed to instrument Vertex AI: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _extract_generate_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Extract attributes from Vertex AI generate_content call.

        Args:
            instance: The GenerativeModel instance.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}
        model_name = getattr(instance, "_model_name", "unknown")

        attrs["gen_ai.system"] = "vertexai"
        attrs["gen_ai.request.model"] = model_name
        attrs["gen_ai.operation.name"] = "generate_content"

        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from Vertex AI response.

        Vertex AI responses include usage_metadata with:
        - prompt_token_count: Input tokens
        - candidates_token_count: Output tokens
        - total_token_count: Total tokens

        Args:
            result: The API response object.

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        try:
            # Handle response with usage_metadata
            if hasattr(result, "usage_metadata") and result.usage_metadata:
                # Thinking and cached tokens included; snake and camel case; see _gemini_usage.
                return gemini_usage(result.usage_metadata)

            return None
        except Exception as e:
            logger.debug("Failed to extract usage from Vertex AI response: %s", e)
            return None
