"""OpenTelemetry instrumentor for the Mistral AI SDK.

This instrumentor automatically traces chat calls to Mistral AI models,
capturing relevant attributes such as the model name and token usage.
"""

import logging
from typing import Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class MistralAIInstrumentor(BaseInstrumentor):
    """Instrumentor for Mistral AI"""

    def instrument(self, config: OTelConfig):
        self.config = config
        try:
            import mistralai
            from mistralai.client import MistralClient

            # Instrument the chat method using proper patching
            self._instrument_chat()
            self._instrument_embeddings()

            logger.info("MistralAI instrumentation enabled")

        except ImportError:
            logger.warning("mistralai package not available, skipping instrumentation")
        except Exception as e:
            logger.error(f"Failed to instrument mistralai: {e}")

    def _instrument_chat(self):
        """Instrument the chat completion method"""
        try:
            import wrapt
            from mistralai.client import MistralClient

            @wrapt.patch_function_wrapper("mistralai.client", "MistralClient.chat")
            def wrapped_chat(wrapped, instance, args, kwargs):
                with self.tracer.start_as_current_span("mistralai.chat") as span:
                    model = kwargs.get("model", "unknown")

                    span.set_attribute("gen_ai.system", "mistralai")
                    span.set_attribute("gen_ai.request.model", model)
                    span.set_attribute("llm.request.type", "chat")

                    self.request_counter.add(
                        1, {"model": model, "provider": "mistralai", "operation": "chat"}
                    )

                    try:
                        result = wrapped(*args, **kwargs)
                        self._record_result_metrics(span, result, 0)
                        return result
                    except Exception as e:
                        span.set_attribute("error", True)
                        span.set_attribute("error.message", str(e))
                        self.request_counter.add(
                            1,
                            {
                                "model": model,
                                "provider": "mistralai",
                                "operation": "chat",
                                "error": "true",
                            },
                        )
                        raise

            return wrapped_chat

        except Exception as e:
            logger.debug(f"Could not instrument MistralAI chat: {e}")

    def _instrument_embeddings(self):
        """Instrument the embeddings method"""
        try:
            import wrapt
            from mistralai.client import MistralClient

            @wrapt.patch_function_wrapper("mistralai.client", "MistralClient.embeddings")
            def wrapped_embeddings(wrapped, instance, args, kwargs):
                with self.tracer.start_as_current_span("mistralai.embeddings") as span:
                    model = kwargs.get("model", "mistral-embed")

                    span.set_attribute("gen_ai.system", "mistralai")
                    span.set_attribute("gen_ai.request.model", model)
                    span.set_attribute("llm.request.type", "embeddings")

                    self.request_counter.add(
                        1, {"model": model, "provider": "mistralai", "operation": "embeddings"}
                    )

                    try:
                        result = wrapped(*args, **kwargs)
                        self._record_embedding_metrics(span, result)
                        return result
                    except Exception as e:
                        span.set_attribute("error", True)
                        span.set_attribute("error.message", str(e))
                        self.request_counter.add(
                            1,
                            {
                                "model": model,
                                "provider": "mistralai",
                                "operation": "embeddings",
                                "error": "true",
                            },
                        )
                        raise

            return wrapped_embeddings

        except Exception as e:
            logger.debug(f"Could not instrument MistralAI embeddings: {e}")

    def _record_result_metrics(self, span, result, cost: float):
        """Record metrics from chat completion result"""
        if hasattr(result, "usage"):
            usage = self._extract_usage(result)
            if usage:
                span.set_attribute("gen_ai.response.finish_reasons", ["stop"])
                span.set_attribute("gen_ai.usage.prompt_tokens", usage.get("prompt_tokens", 0))
                span.set_attribute(
                    "gen_ai.usage.completion_tokens", usage.get("completion_tokens", 0)
                )
                span.set_attribute("gen_ai.usage.total_tokens", usage.get("total_tokens", 0))

                # Record token metrics
                self.token_counter.add(
                    usage.get("prompt_tokens", 0), {"type": "input", "provider": "mistralai"}
                )
                self.token_counter.add(
                    usage.get("completion_tokens", 0), {"type": "output", "provider": "mistralai"}
                )

                # Calculate and record cost
                if cost > 0:
                    span.set_attribute("gen_ai.usage.cost", cost)
                    self.cost_counter.add(cost, {"provider": "mistralai"})

    def _record_embedding_metrics(self, span, result):
        """Record metrics from embeddings result"""
        if hasattr(result, "usage"):
            usage = self._extract_usage(result)
            if usage:
                span.set_attribute("gen_ai.usage.prompt_tokens", usage.get("prompt_tokens", 0))
                span.set_attribute("gen_ai.usage.total_tokens", usage.get("total_tokens", 0))

                self.token_counter.add(
                    usage.get("prompt_tokens", 0),
                    {"type": "input", "provider": "mistralai", "operation": "embeddings"},
                )

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
