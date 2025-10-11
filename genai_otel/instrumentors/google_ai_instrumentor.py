"""OpenTelemetry instrumentor for Google Generative AI (Gemini) SDK.

This instrumentor automatically traces calls to Google Gemini models, capturing
relevant attributes such as the model name and token usage.
"""

import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor


class GoogleAIInstrumentor(BaseInstrumentor):
    """Instrumentor for Google Generative AI (Gemini)"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._google_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if Google Generative AI library is available."""
        try:
            import google.generativeai as genai

            self._google_available = True
            logging.debug("Google Generative AI library detected and available for instrumentation")
        except ImportError:
            logging.debug(
                "Google Generative AI library not installed, instrumentation will be skipped"
            )
            self._google_available = False

    def instrument(self, config: OTelConfig):
        if not self._google_available:
            logging.debug("Skipping Google Generative AI instrumentation - library not available")
            return

        self.config = config
        try:
            import google.generativeai as genai

            if hasattr(genai, "GenerativeModel") and hasattr(
                genai.GenerativeModel, "generate_content"
            ):
                instrumented_generate_method = self.create_span_wrapper(
                    span_name="google.generativeai.generate_content",
                    extract_attributes=self._extract_google_ai_attributes,
                )
                genai.GenerativeModel.generate_content = instrumented_generate_method
            self._instrumented = True
            logging.info("Google Generative AI instrumentation enabled")

        except Exception as e:
            logging.error("Failed to instrument Google Generative AI: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _extract_google_ai_attributes(
        self, instance: Any, args: Any, kwargs: Any
    ) -> Dict[str, Any]:  # pylint: disable=W0613

        attrs = {}

        model_name = getattr(instance, "model_name", "unknown")

        attrs["gen_ai.system"] = "google"

        attrs["gen_ai.request.model"] = model_name

        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:

        if hasattr(result, "usage_metadata") and result.usage_metadata:

            usage = result.usage_metadata

            return {
                "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                "completion_tokens": getattr(usage, "candidates_token_count", 0),
                "total_tokens": getattr(usage, "total_token_count", 0),
            }

        return None
