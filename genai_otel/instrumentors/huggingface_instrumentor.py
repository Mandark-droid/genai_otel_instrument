"""OpenTelemetry instrumentor for HuggingFace Transformers and Inference API.

This instrumentor automatically traces:
1. HuggingFace Transformers pipelines (local model execution)
2. HuggingFace Inference API calls via InferenceClient (used by smolagents)
3. Direct model usage via AutoModelForCausalLM.generate() and forward()

Note: Transformers runs models locally (no API costs), but InferenceClient makes
API calls to HuggingFace endpoints which may have costs based on usage.
Local model costs are estimated based on parameter count and token usage.
"""

import logging
import threading
import weakref
from typing import Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)

# Global storage for last used tokenizer per thread (for content capture)
_thread_local = threading.local()


class HuggingFaceInstrumentor(BaseInstrumentor):
    """Instrumentor for HuggingFace Transformers and Inference API.

    Instruments:
    - transformers.pipeline (local execution, estimated costs)
    - transformers.AutoModelForCausalLM.generate() (local execution, estimated costs)
    - transformers.AutoModelForCausalLM.forward() (local execution, estimated costs)
    - huggingface_hub.InferenceClient (API calls, may have costs)
    """

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._transformers_available = False
        self._inference_client_available = False
        self._model_classes_instrumented = False
        self._check_availability()

    def _check_availability(self):
        """Check if Transformers and InferenceClient libraries are available."""
        try:
            import transformers

            self._transformers_available = True
            logger.debug("Transformers library detected and available for instrumentation")
        except ImportError:
            logger.debug("Transformers library not installed, instrumentation will be skipped")
            self._transformers_available = False

        try:
            from huggingface_hub import InferenceClient

            self._inference_client_available = True
            logger.debug("HuggingFace InferenceClient detected and available for instrumentation")
        except ImportError:
            logger.debug(
                "huggingface_hub not installed, InferenceClient instrumentation will be skipped"
            )
            self._inference_client_available = False

    def instrument(self, config: OTelConfig):
        """Instrument HuggingFace Transformers pipelines, model classes, and InferenceClient."""
        self._setup_config(config)

        instrumented_count = 0

        # Instrument transformers components if available
        if self._transformers_available:
            # Instrument pipeline
            try:
                self._instrument_transformers()
                instrumented_count += 1
            except Exception as e:
                logger.error("Failed to instrument HuggingFace Transformers: %s", e, exc_info=True)
                if config.fail_on_error:
                    raise

            # Instrument model classes (AutoModelForCausalLM, etc.)
            try:
                self._instrument_model_classes()
                instrumented_count += 1
            except Exception as e:
                logger.error("Failed to instrument HuggingFace model classes: %s", e, exc_info=True)
                if config.fail_on_error:
                    raise

            # Instrument tokenizers for content capture
            if config.enable_content_capture:
                try:
                    self._instrument_tokenizers()
                except Exception as e:
                    logger.debug(f"Could not instrument tokenizers for content capture: {e}")

        # Instrument InferenceClient if available
        if self._inference_client_available:
            try:
                self._instrument_inference_client()
                instrumented_count += 1
            except Exception as e:
                logger.error(
                    "Failed to instrument HuggingFace InferenceClient: %s", e, exc_info=True
                )
                if config.fail_on_error:
                    raise

        if instrumented_count > 0:
            self._instrumented = True
            logger.info(f"HuggingFace instrumentation enabled ({instrumented_count} components)")

    def _instrument_transformers(self):
        """Instrument transformers.pipeline for local model execution."""
        try:
            import importlib

            transformers_module = importlib.import_module("transformers")
            original_pipeline = transformers_module.pipeline

            # Capture self reference for use in nested classes
            instrumentor = self

            def wrapped_pipeline(*args, **kwargs):
                pipe = original_pipeline(*args, **kwargs)

                class WrappedPipeline:
                    def __init__(self, original_pipe):
                        self._original_pipe = original_pipe

                    def __call__(self, *call_args, **call_kwargs):
                        # Use instrumentor.tracer instead of config.tracer
                        with instrumentor.tracer.start_span("huggingface.pipeline") as span:
                            task = getattr(self._original_pipe, "task", "unknown")
                            model = getattr(
                                getattr(self._original_pipe, "model", None),
                                "name_or_path",
                                "unknown",
                            )

                            span.set_attribute("gen_ai.system", "huggingface")
                            span.set_attribute("gen_ai.request.model", model)
                            span.set_attribute("gen_ai.operation.name", task)
                            span.set_attribute("huggingface.task", task)

                            if instrumentor.request_counter:
                                instrumentor.request_counter.add(
                                    1, {"model": model, "provider": "huggingface"}
                                )

                            result = self._original_pipe(*call_args, **call_kwargs)

                            # End span manually
                            span.end()
                            return result

                    def __getattr__(self, name):
                        # Delegate all other attribute access to the original pipe
                        return getattr(self._original_pipe, name)

                return WrappedPipeline(pipe)

            transformers_module.pipeline = wrapped_pipeline
            logger.debug("HuggingFace Transformers pipeline instrumented")

        except Exception as e:
            raise  # Re-raise to be caught by instrument() method

    def _instrument_inference_client(self):
        """Instrument HuggingFace InferenceClient for API calls."""
        from huggingface_hub import InferenceClient

        # Store original methods
        original_chat_completion = InferenceClient.chat_completion
        original_text_generation = InferenceClient.text_generation

        # Wrap chat_completion method
        wrapped_chat_completion = self.create_span_wrapper(
            span_name="huggingface.inference.chat_completion",
            extract_attributes=self._extract_inference_client_attributes,
        )(original_chat_completion)

        # Wrap text_generation method
        wrapped_text_generation = self.create_span_wrapper(
            span_name="huggingface.inference.text_generation",
            extract_attributes=self._extract_inference_client_attributes,
        )(original_text_generation)

        InferenceClient.chat_completion = wrapped_chat_completion
        InferenceClient.text_generation = wrapped_text_generation
        logger.debug("HuggingFace InferenceClient instrumented")

    def _instrument_model_classes(self):
        """Instrument HuggingFace model classes for direct model usage."""
        try:
            import wrapt

            # Import GenerationMixin - the base class that provides generate() method
            # All generative models (AutoModelForCausalLM, AutoModelForSeq2SeqLM, etc.) inherit from it
            try:
                from transformers.generation.utils import GenerationMixin
            except ImportError:
                # Fallback for older transformers versions
                from transformers.generation import GenerationMixin

            # Store reference to instrumentor for use in wrapper
            instrumentor = self

            # Wrap the generate() method at GenerationMixin level (all models inherit from this)
            original_generate = GenerationMixin.generate

            @wrapt.decorator
            def generate_wrapper(wrapped, instance, args, kwargs):
                """Wrapper for model.generate() method."""
                # Extract model info
                model_name = getattr(instance, "name_or_path", "unknown")
                if hasattr(instance.config, "_name_or_path"):
                    model_name = instance.config._name_or_path

                # Get input token count and retrieve original text from thread-local
                input_ids = kwargs.get("input_ids") or (args[0] if args else None)
                prompt_tokens = 0
                input_text = None

                if input_ids is not None:
                    if hasattr(input_ids, "shape"):
                        prompt_tokens = int(input_ids.shape[-1])
                    elif isinstance(input_ids, (list, tuple)):
                        prompt_tokens = len(input_ids[0]) if input_ids else 0

                    # Try to get original text from thread-local storage
                    if hasattr(_thread_local, "last_tokenized_text"):
                        input_text = _thread_local.last_tokenized_text.get("current")

                # Create span
                with instrumentor.tracer.start_as_current_span(
                    "huggingface.model.generate"
                ) as span:
                    # Set attributes
                    span.set_attribute("gen_ai.system", "huggingface")
                    span.set_attribute("gen_ai.request.model", model_name)
                    span.set_attribute("gen_ai.operation.name", "text_generation")
                    span.set_attribute("gen_ai.request.type", "chat")

                    # Set first message for evaluation if we decoded the input
                    if input_text:
                        # Format as dict-string for consistency with other instrumentors
                        first_message = str({"role": "user", "content": input_text[:200]})
                        span.set_attribute("gen_ai.request.first_message", first_message)

                    # Extract generation parameters
                    if "max_length" in kwargs:
                        span.set_attribute("gen_ai.request.max_tokens", kwargs["max_length"])
                    if "max_new_tokens" in kwargs:
                        span.set_attribute("gen_ai.request.max_tokens", kwargs["max_new_tokens"])
                    if "temperature" in kwargs:
                        span.set_attribute("gen_ai.request.temperature", kwargs["temperature"])
                    if "top_p" in kwargs:
                        span.set_attribute("gen_ai.request.top_p", kwargs["top_p"])

                    # Call original generate
                    import time

                    start_time = time.time()
                    result = wrapped(*args, **kwargs)
                    duration = time.time() - start_time

                    # Extract output token count
                    completion_tokens = 0
                    if hasattr(result, "shape"):
                        # result is a tensor
                        total_length = int(result.shape[-1])
                        completion_tokens = max(0, total_length - prompt_tokens)
                    elif isinstance(result, (list, tuple)):
                        # result is a list of sequences
                        if result and hasattr(result[0], "shape"):
                            total_length = int(result[0].shape[-1])
                            completion_tokens = max(0, total_length - prompt_tokens)

                    total_tokens = prompt_tokens + completion_tokens

                    # Set token usage attributes
                    if prompt_tokens > 0:
                        span.set_attribute("gen_ai.usage.prompt_tokens", prompt_tokens)
                    if completion_tokens > 0:
                        span.set_attribute("gen_ai.usage.completion_tokens", completion_tokens)
                    if total_tokens > 0:
                        span.set_attribute("gen_ai.usage.total_tokens", total_tokens)

                    # Record metrics
                    if instrumentor.request_counter:
                        instrumentor.request_counter.add(
                            1, {"model": model_name, "provider": "huggingface"}
                        )

                    if instrumentor.token_counter and total_tokens > 0:
                        if prompt_tokens > 0:
                            instrumentor.token_counter.add(
                                prompt_tokens, {"token_type": "prompt", "operation": span.name}
                            )
                        if completion_tokens > 0:
                            instrumentor.token_counter.add(
                                completion_tokens,
                                {"token_type": "completion", "operation": span.name},
                            )

                    if instrumentor.latency_histogram:
                        instrumentor.latency_histogram.record(duration, {"operation": span.name})

                    # Calculate and record cost if enabled
                    if (
                        instrumentor.config
                        and instrumentor.config.enable_cost_tracking
                        and total_tokens > 0
                    ):
                        try:
                            usage = {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": total_tokens,
                            }

                            costs = instrumentor.cost_calculator.calculate_granular_cost(
                                model=model_name, usage=usage, call_type="chat"
                            )

                            if costs["total"] > 0:
                                if instrumentor.cost_counter:
                                    instrumentor.cost_counter.add(
                                        costs["total"], {"model": model_name}
                                    )
                                span.set_attribute("gen_ai.usage.cost.total", costs["total"])
                                if costs["prompt"] > 0:
                                    span.set_attribute("gen_ai.usage.cost.prompt", costs["prompt"])
                                if costs["completion"] > 0:
                                    span.set_attribute(
                                        "gen_ai.usage.cost.completion", costs["completion"]
                                    )

                                logger.debug(
                                    f"HuggingFace model {model_name}: {total_tokens} tokens, "
                                    f"cost: ${costs['total']:.6f}"
                                )
                        except Exception as e:
                            logger.warning(f"Failed to calculate cost: {e}")

                    # Add content events and attributes if content capture is enabled
                    if instrumentor.config and instrumentor.config.enable_content_capture:
                        try:
                            # Add prompt event if we have the original text
                            if input_text:
                                span.add_event(
                                    "gen_ai.prompt.0",
                                    attributes={
                                        "gen_ai.prompt.role": "user",
                                        "gen_ai.prompt.content": str(input_text),
                                    },
                                )

                            # Try to get tokenizer from thread-local and decode output
                            # Check if we stored a tokenizer reference
                            if hasattr(_thread_local, "last_tokenizer"):
                                tokenizer = _thread_local.last_tokenizer
                                try:
                                    output_text = None
                                    if hasattr(result, "tolist"):
                                        # Convert tensor to list
                                        output_list = result.tolist()
                                        if isinstance(output_list[0], list):
                                            # Batch output, take first item
                                            output_text = tokenizer.decode(
                                                output_list[0], skip_special_tokens=True
                                            )
                                        else:
                                            output_text = tokenizer.decode(
                                                output_list, skip_special_tokens=True
                                            )

                                    if output_text:
                                        # Set as attribute for evaluation processor
                                        span.set_attribute("gen_ai.response", output_text)
                                        # Also add as event for observability
                                        span.add_event(
                                            "gen_ai.completion.0",
                                            attributes={
                                                "gen_ai.completion.role": "assistant",
                                                "gen_ai.completion.content": output_text,
                                            },
                                        )
                                except Exception as e:
                                    logger.debug(f"Could not decode output for content: {e}")
                        except Exception as e:
                            logger.debug(f"Failed to add content events: {e}")

                    # Run evaluation checks BEFORE ending the span
                    try:
                        instrumentor._run_evaluation_checks(span, args, kwargs, result)
                    except Exception as e:
                        logger.warning(
                            f"Failed to run evaluation checks for HuggingFace span: {e}",
                            exc_info=True,
                        )

                    return result

            # Apply wrapper to GenerationMixin.generate (all models inherit this)
            GenerationMixin.generate = generate_wrapper(original_generate)

            self._model_classes_instrumented = True
            logger.debug(
                "HuggingFace GenerationMixin.generate() instrumented "
                "(covers all models: AutoModelForCausalLM, AutoModelForSeq2SeqLM, etc.)"
            )

        except ImportError as e:
            logger.debug(f"Could not import model classes for instrumentation: {e}")
        except Exception as e:
            raise  # Re-raise to be caught by instrument() method

    def _extract_inference_client_attributes(self, instance, args, kwargs) -> Dict[str, str]:
        """Extract attributes from Inference API call."""
        attrs = {}
        model = kwargs.get("model") or (args[0] if args else "unknown")

        attrs["gen_ai.system"] = "huggingface"
        attrs["gen_ai.request.model"] = str(model)
        attrs["gen_ai.operation.name"] = "chat"  # Default to chat

        # Extract parameters if available
        if "max_tokens" in kwargs:
            attrs["gen_ai.request.max_tokens"] = kwargs["max_tokens"]
        if "temperature" in kwargs:
            attrs["gen_ai.request.temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            attrs["gen_ai.request.top_p"] = kwargs["top_p"]

        # Capture first message content for evaluation features
        messages = kwargs.get("messages", [])
        if messages:
            # Only capture first 200 chars to avoid sensitive data and span size issues
            # Messages should already be in dict format, str() preserves the dict-string format
            first_message = str(messages[0])[:200]
            attrs["gen_ai.request.first_message"] = first_message
        elif "prompt" in kwargs:
            # For text_generation calls - wrap plain text in dict format
            prompt_text = str(kwargs["prompt"])[:200]
            first_message = str({"role": "user", "content": prompt_text})
            attrs["gen_ai.request.first_message"] = first_message

        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from HuggingFace response.

        Handles both:
        1. Transformers pipeline (local execution) - returns None
        2. InferenceClient API calls - extracts token usage from response

        Args:
            result: The pipeline output or InferenceClient response.

        Returns:
            Dict with token counts for InferenceClient calls, None for local execution.
        """
        # Check if this is an InferenceClient API response
        if result is not None and hasattr(result, "usage"):
            usage = result.usage

            # Extract token counts from usage object
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)

            # If usage is a dict instead of object
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                total_tokens = usage.get("total_tokens")

            # Return token counts if available
            if prompt_tokens is not None or completion_tokens is not None:
                return {
                    "prompt_tokens": prompt_tokens or 0,
                    "completion_tokens": completion_tokens or 0,
                    "total_tokens": total_tokens or (prompt_tokens or 0) + (completion_tokens or 0),
                }

        # HuggingFace Transformers is free (local execution)
        # No token-based costs to track
        return None

    def _add_content_events(self, span, result, request_kwargs: dict):
        """Add prompt and completion content as span events and attributes.

        Args:
            span: The OpenTelemetry span.
            result: The API response object.
            request_kwargs: The original request kwargs.
        """
        # Add prompt content events for InferenceClient
        messages = request_kwargs.get("messages", [])
        if messages:
            for idx, message in enumerate(messages):
                if isinstance(message, dict):
                    role = message.get("role", "unknown")
                    content = message.get("content", "")
                    span.add_event(
                        f"gen_ai.prompt.{idx}",
                        attributes={
                            "gen_ai.prompt.role": role,
                            "gen_ai.prompt.content": str(content),
                        },
                    )
        elif "prompt" in request_kwargs:
            # For text_generation calls
            prompt = request_kwargs["prompt"]
            span.add_event(
                "gen_ai.prompt.0",
                attributes={
                    "gen_ai.prompt.role": "user",
                    "gen_ai.prompt.content": str(prompt),
                },
            )

        # Add completion content events AND attributes (for evaluation processor)
        try:
            response_text = None

            # For InferenceClient responses
            if hasattr(result, "choices") and result.choices:
                for idx, choice in enumerate(result.choices):
                    if hasattr(choice, "message") and hasattr(choice.message, "content"):
                        content = choice.message.content
                        if idx == 0:
                            response_text = str(content)
                        span.add_event(
                            f"gen_ai.completion.{idx}",
                            attributes={
                                "gen_ai.completion.role": "assistant",
                                "gen_ai.completion.content": str(content),
                            },
                        )
            # For text_generation responses
            elif hasattr(result, "generated_text"):
                response_text = str(result.generated_text)
                span.add_event(
                    "gen_ai.completion.0",
                    attributes={
                        "gen_ai.completion.role": "assistant",
                        "gen_ai.completion.content": response_text,
                    },
                )
            # For pipeline responses (list of dicts)
            elif isinstance(result, list) and result:
                for idx, item in enumerate(result):
                    if isinstance(item, dict) and "generated_text" in item:
                        response_text = str(item["generated_text"])
                        span.add_event(
                            f"gen_ai.completion.{idx}",
                            attributes={
                                "gen_ai.completion.role": "assistant",
                                "gen_ai.completion.content": response_text,
                            },
                        )
                        break  # Only capture first completion

            # Set response as attribute for evaluation processor
            if response_text:
                span.set_attribute("gen_ai.response", response_text)

        except Exception as e:
            logger.debug("Failed to add content events for HuggingFace response: %s", e)

    def _instrument_tokenizers(self):
        """Instrument tokenizers to capture original text for content events."""
        try:
            import wrapt
            from transformers import PreTrainedTokenizerBase

            # Store original __call__ method
            original_call = PreTrainedTokenizerBase.__call__

            @wrapt.decorator
            def tokenizer_call_wrapper(wrapped, instance, args, kwargs):
                """Wrap tokenizer __call__ to store original text and tokenizer reference."""
                # Get the text input (first positional arg or 'text' kwarg)
                text_input = None
                if args:
                    text_input = args[0]
                elif "text" in kwargs:
                    text_input = kwargs["text"]

                # Call original tokenizer
                result = wrapped(*args, **kwargs)

                # Store text and tokenizer reference in thread-local if we have input_ids
                if text_input and hasattr(result, "get") and "input_ids" in result:
                    if not hasattr(_thread_local, "last_tokenized_text"):
                        _thread_local.last_tokenized_text = {}
                    # Store the text mapped to a simple key we can retrieve
                    _thread_local.last_tokenized_text["current"] = text_input
                    # Store tokenizer reference for decoding output
                    _thread_local.last_tokenizer = instance

                return result

            # Apply wrapper
            PreTrainedTokenizerBase.__call__ = tokenizer_call_wrapper(original_call)
            logger.debug("HuggingFace tokenizers instrumented for content capture")

        except Exception as e:
            logger.debug(f"Could not instrument tokenizers: {e}")
