"""OpenTelemetry instrumentor for the Ollama library.

This instrumentor automatically traces calls to Ollama models for both
generation and chat functionalities, capturing relevant attributes such as
the model name and token usage.

Optionally enables server metrics polling via /api/ps endpoint to track
VRAM usage and running models.
"""

import logging
import os
import sys
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor
from .ollama_server_metrics_poller import start_ollama_metrics_poller

logger = logging.getLogger(__name__)


class OllamaInstrumentor(BaseInstrumentor):
    """Instrumentor for Ollama"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._ollama_available = False
        self._ollama_module = None
        self._original_generate = None
        self._original_chat = None
        self._check_availability()

    def _check_availability(self):
        """Check if Ollama library is available."""
        try:
            import ollama

            self._ollama_available = True
            self._ollama_module = ollama
            logger.debug("Ollama library detected and available for instrumentation")
        except ImportError:
            logger.debug("Ollama library not installed, instrumentation will be skipped")
            self._ollama_available = False
            self._ollama_module = None

    def instrument(self, config: OTelConfig):
        """Instrument the Ollama library."""
        self.config = config

        if not self._ollama_available or self._ollama_module is None:
            return

        try:
            # Store original methods and wrap them
            self._original_generate = self._ollama_module.generate
            self._original_chat = self._ollama_module.chat

            # Wrap generate method
            wrapped_generate = self.create_span_wrapper(
                span_name="ollama.generate",
                extract_attributes=self._extract_generate_attributes,
            )(self._original_generate)
            self._ollama_module.generate = wrapped_generate

            # Wrap chat method
            wrapped_chat = self.create_span_wrapper(
                span_name="ollama.chat",
                extract_attributes=self._extract_chat_attributes,
            )(self._original_chat)
            self._ollama_module.chat = wrapped_chat

            self._instrumented = True
            logger.info("Ollama instrumentation enabled")

            # Start server metrics poller if enabled
            # Note: Server metrics poller requires Python 3.11+ due to implementation dependencies
            python_version = sys.version_info
            if python_version < (3, 11):
                logger.debug(
                    "Ollama server metrics poller requires Python 3.11+, skipping "
                    f"(current: {python_version.major}.{python_version.minor})"
                )
                return

            enable_server_metrics = (
                os.getenv("GENAI_ENABLE_OLLAMA_SERVER_METRICS", "true").lower() == "true"
            )

            if enable_server_metrics:
                try:
                    # Get configuration from environment variables
                    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                    ollama_metrics_interval = float(
                        os.getenv("GENAI_OLLAMA_METRICS_INTERVAL", "5.0")
                    )
                    ollama_max_vram_gb = os.getenv("GENAI_OLLAMA_MAX_VRAM_GB")
                    max_vram = float(ollama_max_vram_gb) if ollama_max_vram_gb else None

                    # Start the poller
                    start_ollama_metrics_poller(
                        base_url=ollama_base_url,
                        interval=ollama_metrics_interval,
                        max_vram_gb=max_vram,
                    )
                    logger.info(
                        f"Ollama server metrics poller started (url={ollama_base_url}, "
                        f"interval={ollama_metrics_interval}s)"
                    )
                except Exception as e:
                    logger.warning(f"Failed to start Ollama server metrics poller: {e}")
                    if config.fail_on_error:
                        raise

        except Exception as e:
            logger.error("Failed to instrument Ollama: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _extract_generate_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Extract attributes from Ollama generate call.

        Args:
            instance: The client instance (None for module-level functions).
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}
        model = kwargs.get("model", "unknown")

        attrs["gen_ai.system"] = "ollama"
        attrs["gen_ai.request.model"] = model
        attrs["gen_ai.operation.name"] = "generate"

        return attrs

    def _extract_chat_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Extract attributes from Ollama chat call.

        Args:
            instance: The client instance (None for module-level functions).
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])

        attrs["gen_ai.system"] = "ollama"
        attrs["gen_ai.request.model"] = model
        attrs["gen_ai.operation.name"] = "chat"
        attrs["gen_ai.request.message_count"] = len(messages)

        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from Ollama response.

        Ollama responses include:
        - prompt_eval_count: Input tokens
        - eval_count: Output tokens

        Args:
            result: The API response object or dictionary.

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        try:
            # Handle both dict and object responses
            if isinstance(result, dict):
                prompt_tokens = result.get("prompt_eval_count", 0)
                completion_tokens = result.get("eval_count", 0)
            elif hasattr(result, "prompt_eval_count") and hasattr(result, "eval_count"):
                prompt_tokens = getattr(result, "prompt_eval_count", 0)
                completion_tokens = getattr(result, "eval_count", 0)
            else:
                return None

            if prompt_tokens == 0 and completion_tokens == 0:
                return None

            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        except Exception as e:
            logger.debug("Failed to extract usage from Ollama response: %s", e)
            return None

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        """Extract response attributes from Ollama response.

        Args:
            result: The API response object or dictionary.

        Returns:
            Dict[str, Any]: Dictionary of response attributes.
        """
        attrs = {}

        try:
            # Handle both dict and object responses
            if isinstance(result, dict):
                # Response model (actual model used)
                if "model" in result:
                    attrs["gen_ai.response.model"] = result["model"]

                # Done reason (equivalent to finish_reason)
                if "done_reason" in result:
                    attrs["gen_ai.response.finish_reason"] = result["done_reason"]

                # Response content length (for observability)
                if "response" in result:
                    response_text = result["response"]
                    if isinstance(response_text, str):
                        attrs["gen_ai.response.length"] = len(response_text)
                elif "message" in result:
                    # For chat responses
                    message = result["message"]
                    if isinstance(message, dict) and "content" in message:
                        content = message["content"]
                        if isinstance(content, str):
                            attrs["gen_ai.response.length"] = len(content)

            elif hasattr(result, "model"):
                # Object-like response
                if hasattr(result, "model"):
                    attrs["gen_ai.response.model"] = result.model

                if hasattr(result, "done_reason"):
                    attrs["gen_ai.response.finish_reason"] = result.done_reason

                # Response content length
                if hasattr(result, "response"):
                    response_text = result.response
                    if isinstance(response_text, str):
                        attrs["gen_ai.response.length"] = len(response_text)
                elif hasattr(result, "message"):
                    message = result.message
                    if hasattr(message, "content"):
                        content = message.content
                        if isinstance(content, str):
                            attrs["gen_ai.response.length"] = len(content)

        except Exception as e:
            logger.debug("Failed to extract response attributes from Ollama response: %s", e)

        return attrs

    def _extract_finish_reason(self, result) -> Optional[str]:
        """Extract finish reason from Ollama response.

        Args:
            result: The Ollama API response object or dictionary.

        Returns:
            Optional[str]: The finish reason string or None if not available.
        """
        try:
            # Handle both dict and object responses
            if isinstance(result, dict):
                return result.get("done_reason")
            elif hasattr(result, "done_reason"):
                return result.done_reason
        except Exception as e:
            logger.debug("Failed to extract finish_reason from Ollama response: %s", e)
        return None
