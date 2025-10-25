"""OpenTelemetry instrumentor for the Mistral AI SDK (v1.0+).

This instrumentor automatically traces chat calls to Mistral AI models,
capturing relevant attributes such as the model name and token usage.

Supports Mistral SDK v1.0+ with the new API structure:
- Mistral.chat.complete()
- Mistral.chat.stream()
- Mistral.embeddings.create()
"""

import logging
import time
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

            # Get access to the chat and embeddings modules
            # In Mistral SDK v1.0+, structure is:
            # - Mistral client has .chat and .embeddings properties
            # - These are bound methods that call internal APIs
            # Store original methods at module level before any instances are created
            if not hasattr(Mistral, "_genai_otel_instrumented"):
                self._wrap_mistral_methods(Mistral, wrapt)
                Mistral._genai_otel_instrumented = True
                logger.info("MistralAI instrumentation enabled (v1.0+ SDK)")

        except ImportError:
            logger.warning("mistralai package not available, skipping instrumentation")
        except Exception as e:
            logger.error(f"Failed to instrument mistralai: {e}", exc_info=True)
            if config.fail_on_error:
                raise

    def _wrap_mistral_methods(self, Mistral, wrapt):
        """Wrap Mistral client methods at the class level."""
        # Import the internal classes that handle chat and embeddings
        try:
            from mistralai.chat import Chat
            from mistralai.embeddings import Embeddings

            # Wrap Chat.complete method
            if hasattr(Chat, "complete"):
                wrapt.wrap_function_wrapper(
                    "mistralai.chat", "Chat.complete", self._wrap_chat_complete
                )
                logger.debug("Wrapped Mistral Chat.complete")

            # Wrap Chat.stream method
            if hasattr(Chat, "stream"):
                wrapt.wrap_function_wrapper("mistralai.chat", "Chat.stream", self._wrap_chat_stream)
                logger.debug("Wrapped Mistral Chat.stream")

            # Wrap Embeddings.create method
            if hasattr(Embeddings, "create"):
                wrapt.wrap_function_wrapper(
                    "mistralai.embeddings", "Embeddings.create", self._wrap_embeddings_create
                )
                logger.debug("Wrapped Mistral Embeddings.create")

        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not access Mistral internal classes: {e}")

    def _wrap_chat_complete(self, wrapped, instance, args, kwargs):
        """Wrapper for chat.complete() method."""
        model = kwargs.get("model", "mistral-small-latest")
        span_name = f"mistralai.chat.complete {model}"

        with self.tracer.start_span(span_name) as span:
            # Set attributes
            attributes = self._extract_chat_attributes(instance, args, kwargs)
            for key, value in attributes.items():
                span.set_attribute(key, value)

            # Record request metric
            if self.request_counter:
                self.request_counter.add(1, {"model": model, "provider": "mistralai"})

            # Execute the call
            start_time = time.time()
            try:
                response = wrapped(*args, **kwargs)

                # Record metrics from response
                self._record_result_metrics(span, response, start_time, kwargs)

                return response

            except Exception as e:
                if self.error_counter:
                    self.error_counter.add(
                        1, {"operation": span_name, "error.type": type(e).__name__}
                    )
                span.record_exception(e)
                raise

    def _wrap_chat_stream(self, wrapped, instance, args, kwargs):
        """Wrapper for chat.stream() method - handles streaming responses."""
        model = kwargs.get("model", "mistral-small-latest")
        span_name = f"mistralai.chat.stream {model}"

        # Start the span
        span = self.tracer.start_span(span_name)

        # Set attributes
        attributes = self._extract_chat_attributes(instance, args, kwargs)
        for key, value in attributes.items():
            span.set_attribute(key, value)

        # Record request metric
        if self.request_counter:
            self.request_counter.add(1, {"model": model, "provider": "mistralai"})

        start_time = time.time()

        # Execute and get the stream
        try:
            stream = wrapped(*args, **kwargs)

            # Wrap the stream with our tracking wrapper
            return self._StreamWrapper(stream, span, self, model, start_time, span_name)

        except Exception as e:
            if self.error_counter:
                self.error_counter.add(1, {"operation": span_name, "error.type": type(e).__name__})
            span.record_exception(e)
            span.end()
            raise

    def _wrap_embeddings_create(self, wrapped, instance, args, kwargs):
        """Wrapper for embeddings.create() method."""
        model = kwargs.get("model", "mistral-embed")
        span_name = f"mistralai.embeddings.create {model}"

        with self.tracer.start_span(span_name) as span:
            # Set attributes
            attributes = self._extract_embeddings_attributes(instance, args, kwargs)
            for key, value in attributes.items():
                span.set_attribute(key, value)

            # Record request metric
            if self.request_counter:
                self.request_counter.add(1, {"model": model, "provider": "mistralai"})

            # Execute the call
            start_time = time.time()
            try:
                response = wrapped(*args, **kwargs)

                # Record metrics from response
                self._record_result_metrics(span, response, start_time, kwargs)

                return response

            except Exception as e:
                if self.error_counter:
                    self.error_counter.add(
                        1, {"operation": span_name, "error.type": type(e).__name__}
                    )
                span.record_exception(e)
                raise

    class _StreamWrapper:
        """Wrapper for streaming responses that collects metrics."""

        def __init__(self, stream, span, instrumentor, model, start_time, span_name):
            self._stream = stream
            self._span = span
            self._instrumentor = instrumentor
            self._model = model
            self._start_time = start_time
            self._span_name = span_name
            self._usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            self._response_text = ""
            self._first_chunk = True
            self._ttft = None

        def __iter__(self):
            return self

        def __next__(self):
            try:
                chunk = next(self._stream)

                # Record time to first token
                if self._first_chunk:
                    self._ttft = time.time() - self._start_time
                    self._first_chunk = False

                # Process chunk to extract usage and content
                self._process_chunk(chunk)

                return chunk

            except StopIteration:
                # Stream completed - record final metrics
                try:
                    # Set TTFT if we got any chunks
                    if self._ttft is not None:
                        self._span.set_attribute("gen_ai.server.ttft", self._ttft)

                    # Record usage metrics if available
                    if self._usage["total_tokens"] > 0:
                        # Create a mock response object with usage for _record_result_metrics
                        class MockUsage:
                            def __init__(self, usage_dict):
                                self.prompt_tokens = usage_dict["prompt_tokens"]
                                self.completion_tokens = usage_dict["completion_tokens"]
                                self.total_tokens = usage_dict["total_tokens"]

                        class MockResponse:
                            def __init__(self, usage_dict):
                                self.usage = MockUsage(usage_dict)

                        mock_response = MockResponse(self._usage)
                        self._instrumentor._record_result_metrics(
                            self._span, mock_response, self._start_time, {"model": self._model}
                        )

                finally:
                    self._span.end()

                raise

        def _process_chunk(self, chunk):
            """Process a streaming chunk to extract usage."""
            try:
                # Mistral streaming chunks have: data.choices[0].delta.content
                if hasattr(chunk, "data"):
                    data = chunk.data
                    if hasattr(data, "choices") and len(data.choices) > 0:
                        delta = data.choices[0].delta
                        if hasattr(delta, "content") and delta.content:
                            self._response_text += delta.content

                    # Extract usage if available on final chunk
                    if hasattr(data, "usage") and data.usage:
                        usage = data.usage
                        if hasattr(usage, "prompt_tokens"):
                            self._usage["prompt_tokens"] = usage.prompt_tokens
                        if hasattr(usage, "completion_tokens"):
                            self._usage["completion_tokens"] = usage.completion_tokens
                        if hasattr(usage, "total_tokens"):
                            self._usage["total_tokens"] = usage.total_tokens

            except Exception as e:
                logger.debug(f"Error processing Mistral stream chunk: {e}")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                self._span.record_exception(exc_val)
            self._span.end()
            return False

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
