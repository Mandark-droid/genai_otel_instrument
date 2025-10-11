import wrapt
from typing import Dict, Optional, Any, Callable
from .base import BaseInstrumentor
from ..config import OTelConfig


class GoogleAIInstrumentor(BaseInstrumentor):
    """Instrumentor for Google Generative AI (Gemini)"""

    def instrument(self, config: OTelConfig):
        self.config = config
        try:
            import google.generativeai as genai

            # Instrument GenerativeModel.generate_content using create_span_wrapper
            if hasattr(genai, "GenerativeModel") and hasattr(genai.GenerativeModel, "generate_content"):
                original_generate_method = genai.GenerativeModel.generate_content
                instrumented_generate_method = self.create_span_wrapper(
                    span_name="google.generativeai.generate_content",
                    extract_attributes=self._extract_google_ai_attributes
                )
                genai.GenerativeModel.generate_content = instrumented_generate_method

        except ImportError:
            pass

    # New method to extract attributes, now accepting instance
    def _extract_google_ai_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        attrs = {}
        # Access instance directly to get model_name
        model_name = getattr(instance, "model_name", "unknown")

        attrs["gen_ai.system"] = "google"
        attrs["gen_ai.request.model"] = model_name
        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        # Add safety checks for result and result.usage_metadata
        if hasattr(result, "usage_metadata") and result.usage_metadata:
            usage = result.usage_metadata
            return {
                "prompt_tokens": getattr(usage, 'prompt_token_count', 0),
                "completion_tokens": getattr(usage, 'candidates_token_count', 0),
                "total_tokens": getattr(usage, 'total_token_count', 0)
            }
        return None