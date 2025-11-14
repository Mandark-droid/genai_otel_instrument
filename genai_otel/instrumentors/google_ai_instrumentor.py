"""OpenTelemetry instrumentor for Google Generative AI (Gemini) SDK.

This instrumentor supports both the legacy google-generativeai SDK and the new
google-genai unified SDK. It automatically detects which SDK is installed and
instruments accordingly.

Legacy SDK (deprecated Nov 30, 2025): import google.generativeai as genai
New SDK (GA May 2025): from google import genai
"""

import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class GoogleAIInstrumentor(BaseInstrumentor):
    """Instrumentor for Google Generative AI (Gemini)

    Supports both:
    - Legacy SDK: google-generativeai (pip install google-generativeai)
    - New SDK: google-genai (pip install google-genai)
    """

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._google_available = False
        self._using_new_sdk = False
        self._check_availability()

    def _check_availability(self):
        """Check if Google Generative AI library is available.

        Checks for new SDK first, falls back to legacy SDK.
        """
        # Try new SDK first (google-genai)
        try:
            from google import genai

            self._google_available = True
            self._using_new_sdk = True
            logger.debug(
                "Google GenAI (new unified SDK) detected and available for instrumentation"
            )
            return
        except ImportError:
            pass

        # Fall back to legacy SDK (google-generativeai)
        try:
            import google.generativeai as genai

            self._google_available = True
            self._using_new_sdk = False
            logger.debug(
                "Google Generative AI (legacy SDK) detected and available for instrumentation. "
                "Consider migrating to google-genai (support for legacy SDK ends Nov 30, 2025)"
            )
            return
        except ImportError:
            logger.debug(
                "Google Generative AI library not installed, instrumentation will be skipped"
            )
            self._google_available = False

    def instrument(self, config: OTelConfig):
        """Instrument Google Generative AI SDK if available.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._google_available:
            logger.debug("Skipping Google Generative AI instrumentation - library not available")
            return

        self.config = config

        try:
            if self._using_new_sdk:
                self._instrument_new_sdk()
            else:
                self._instrument_legacy_sdk()

            self._instrumented = True
            sdk_type = "new unified SDK" if self._using_new_sdk else "legacy SDK"
            logger.info(f"Google Generative AI instrumentation enabled ({sdk_type})")

        except Exception as e:
            logger.error("Failed to instrument Google Generative AI: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _instrument_new_sdk(self):
        """Instrument the new google-genai unified SDK."""
        import wrapt
        from google import genai

        # The new SDK uses a Client-based approach
        # Instrument the Client class initialization to wrap generate_content methods
        if hasattr(genai, "Client"):
            original_init = genai.Client.__init__

            def wrapped_init(wrapped, instance, args, kwargs):
                result = wrapped(*args, **kwargs)
                self._instrument_client(instance)
                return result

            genai.Client.__init__ = wrapt.FunctionWrapper(original_init, wrapped_init)

        # Also instrument GenerativeModel if it exists (for backward compatibility)
        if hasattr(genai, "GenerativeModel"):
            if hasattr(genai.GenerativeModel, "generate_content"):
                original_generate = genai.GenerativeModel.generate_content
                genai.GenerativeModel.generate_content = self.create_span_wrapper(
                    span_name="google.genai.generate_content",
                    extract_attributes=self._extract_google_ai_attributes,
                )(original_generate)

    def _instrument_client(self, client):
        """Instrument a google-genai Client instance.

        Args:
            client: The genai.Client instance to instrument.
        """
        # Instrument models.generate_content if available
        if hasattr(client, "models"):
            if hasattr(client.models, "generate_content"):
                original_generate = client.models.generate_content
                client.models.generate_content = self.create_span_wrapper(
                    span_name="google.genai.models.generate_content",
                    extract_attributes=self._extract_google_ai_attributes_new_sdk,
                )(original_generate)

    def _instrument_legacy_sdk(self):
        """Instrument the legacy google-generativeai SDK."""
        import google.generativeai as genai

        # Legacy SDK: Instrument GenerativeModel.generate_content
        if hasattr(genai, "GenerativeModel"):
            if hasattr(genai.GenerativeModel, "generate_content"):
                original_generate = genai.GenerativeModel.generate_content
                genai.GenerativeModel.generate_content = self.create_span_wrapper(
                    span_name="google.generativeai.generate_content",
                    extract_attributes=self._extract_google_ai_attributes,
                )(original_generate)

    def _extract_google_ai_attributes(
        self, instance: Any, args: Any, kwargs: Any
    ) -> Dict[str, Any]:
        """Extract attributes from Google AI API call (legacy SDK).

        Args:
            instance: The GenerativeModel instance.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Extract model name from instance
        model_name = getattr(instance, "model_name", "unknown")
        attrs["gen_ai.system"] = "google"
        attrs["gen_ai.request.model"] = model_name
        attrs["gen_ai.operation.name"] = "chat"

        # Extract generation config if available
        if "generation_config" in kwargs:
            config = kwargs["generation_config"]
            if hasattr(config, "temperature"):
                attrs["gen_ai.request.temperature"] = config.temperature
            if hasattr(config, "top_p"):
                attrs["gen_ai.request.top_p"] = config.top_p
            if hasattr(config, "max_output_tokens"):
                attrs["gen_ai.request.max_tokens"] = config.max_output_tokens

        # Extract safety settings count if available
        if "safety_settings" in kwargs:
            attrs["gen_ai.request.safety_settings_count"] = len(kwargs["safety_settings"])

        return attrs

    def _extract_google_ai_attributes_new_sdk(
        self, instance: Any, args: Any, kwargs: Any
    ) -> Dict[str, Any]:
        """Extract attributes from Google AI API call (new SDK).

        Args:
            instance: The client instance.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        attrs["gen_ai.system"] = "google"
        attrs["gen_ai.operation.name"] = "chat"

        # Extract model from kwargs (new SDK uses model parameter)
        if "model" in kwargs:
            attrs["gen_ai.request.model"] = kwargs["model"]

        # Extract config parameters if available
        if "config" in kwargs:
            config = kwargs["config"]
            if isinstance(config, dict):
                if "temperature" in config:
                    attrs["gen_ai.request.temperature"] = config["temperature"]
                if "top_p" in config:
                    attrs["gen_ai.request.top_p"] = config["top_p"]
                if "max_output_tokens" in config:
                    attrs["gen_ai.request.max_tokens"] = config["max_output_tokens"]

        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from Google AI response.

        Works with both legacy and new SDK response formats.

        Args:
            result: The API response object.

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        # Try new SDK format first (usage_metadata)
        if hasattr(result, "usage_metadata") and result.usage_metadata:
            usage = result.usage_metadata
            return {
                "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                "completion_tokens": getattr(usage, "candidates_token_count", 0),
                "total_tokens": getattr(usage, "total_token_count", 0),
            }

        # Try alternative attribute names (in case SDK changes)
        if hasattr(result, "usage") and result.usage:
            usage = result.usage
            return {
                "prompt_tokens": getattr(
                    usage, "prompt_tokens", getattr(usage, "prompt_token_count", 0)
                ),
                "completion_tokens": getattr(
                    usage, "completion_tokens", getattr(usage, "candidates_token_count", 0)
                ),
                "total_tokens": getattr(
                    usage, "total_tokens", getattr(usage, "total_token_count", 0)
                ),
            }

        return None

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        """Extract response attributes from Google AI response.

        Args:
            result: The API response object.

        Returns:
            Dict[str, Any]: Dictionary of response attributes.
        """
        attrs = {}

        # Extract model name from response if available
        if hasattr(result, "model"):
            attrs["gen_ai.response.model"] = result.model

        # Extract finish reasons from candidates
        if hasattr(result, "candidates") and result.candidates:
            finish_reasons = []
            for candidate in result.candidates:
                if hasattr(candidate, "finish_reason"):
                    finish_reasons.append(str(candidate.finish_reason))

            if finish_reasons:
                attrs["gen_ai.response.finish_reasons"] = finish_reasons

        # Extract safety ratings if available
        if hasattr(result, "candidates") and result.candidates:
            for idx, candidate in enumerate(result.candidates[:1]):  # Limit to first candidate
                if hasattr(candidate, "safety_ratings") and candidate.safety_ratings:
                    for rating in candidate.safety_ratings:
                        category = getattr(rating, "category", "unknown")
                        probability = getattr(rating, "probability", "unknown")
                        attrs[f"gen_ai.safety.{category}"] = str(probability)

        return attrs

    def _extract_finish_reason(self, result) -> Optional[str]:
        """Extract finish reason from Google AI response.

        Args:
            result: The Google AI API response object.

        Returns:
            Optional[str]: The finish reason string or None if not available.
        """
        if hasattr(result, "candidates") and result.candidates:
            first_candidate = result.candidates[0]
            if hasattr(first_candidate, "finish_reason"):
                return str(first_candidate.finish_reason)

        return None
