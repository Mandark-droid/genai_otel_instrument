"""OpenTelemetry instrumentor for the OpenAI Python SDK.

This instrumentor automatically traces chat completion calls made using the
OpenAI SDK, capturing relevant attributes such as the model name, message count,
and token usage.
"""

import json
import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor, find_base_url_claim

logger = logging.getLogger(__name__)


def _cap_content(config, text):
    """Bound captured content to config.content_max_length (0/None/unset = unlimited)."""
    if text is None:
        return text
    text = str(text)
    max_len = getattr(config, "content_max_length", 0) if config else 0
    if isinstance(max_len, int) and max_len > 0:
        return text[:max_len]
    return text


def _responses_content_to_text(content: Any) -> str:
    """Flatten a Responses content value to plain text.

    A content field is either a bare string or a list of typed parts
    (``input_text``, ``output_text``, ...). Only the textual parts carry
    anything worth putting on a span.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for part in content:
            text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return ""


class OpenAIInstrumentor(BaseInstrumentor):
    """Instrumentor for OpenAI SDK"""

    MEDIA_PROVIDER = "openai"

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._openai_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if OpenAI library is available."""
        try:
            import openai

            self._openai_available = True
            logger.debug("OpenAI library detected and available for instrumentation")
        except ImportError:
            logger.debug("OpenAI library not installed, instrumentation will be skipped")
            self._openai_available = False

    def instrument(self, config: OTelConfig):
        """Instrument OpenAI SDK if available.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._openai_available:
            logger.debug("Skipping OpenAI instrumentation - library not available")
            return

        self.config = config

        try:
            import openai
            import wrapt

            # Idempotency guard: never stack wrappers if instrument() runs twice.
            if getattr(openai, "_genai_otel_openai_instrumented", False) is True:
                logger.debug("OpenAI already instrumented, skipping")
                self._instrumented = True
                return

            # Instrument sync OpenAI client initialization
            if hasattr(openai, "OpenAI"):
                original_init = openai.OpenAI.__init__

                def wrapped_init(wrapped, instance, args, kwargs):
                    result = wrapped(*args, **kwargs)
                    self._instrument_client(instance)
                    return result

                openai.OpenAI.__init__ = wrapt.FunctionWrapper(original_init, wrapped_init)
                self._instrumented = True
                logger.info("OpenAI instrumentation enabled")

            # Instrument async OpenAI client initialization
            if hasattr(openai, "AsyncOpenAI"):
                original_async_init = openai.AsyncOpenAI.__init__

                def wrapped_async_init(wrapped, instance, args, kwargs):
                    result = wrapped(*args, **kwargs)
                    self._instrument_async_client(instance)
                    return result

                openai.AsyncOpenAI.__init__ = wrapt.FunctionWrapper(
                    original_async_init, wrapped_async_init
                )
                self._instrumented = True
                logger.info("AsyncOpenAI instrumentation enabled")

            if self._instrumented:
                try:
                    openai._genai_otel_openai_instrumented = True
                except Exception:  # noqa: BLE001
                    pass

        except Exception as e:
            logger.error("Failed to instrument OpenAI: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _instrument_client(self, client):
        """Instrument OpenAI client methods.

        Args:
            client: The OpenAI client instance to instrument.
        """
        # A client pointed at an aggregator (OpenRouter, CometAPI) is traced by
        # its dedicated instrumentor; wrapping it here too would emit a
        # duplicate span and double-count token/cost metrics.
        claimed = find_base_url_claim(getattr(client, "base_url", None))
        if claimed:
            logger.debug(
                "Skipping generic OpenAI instrumentation for client handled by "
                "the '%s' instrumentor",
                claimed,
            )
            return
        if (
            hasattr(client, "chat")
            and hasattr(client.chat, "completions")
            and hasattr(client.chat.completions, "create")
        ):
            original_create = client.chat.completions.create
            instrumented_create_method = self.create_span_wrapper(
                span_name="openai.chat.completion",
                extract_attributes=self._extract_openai_attributes,
            )(original_create)
            client.chat.completions.create = instrumented_create_method

        # Embeddings are the retrieval half of a RAG call. Without a span here
        # the trace shows only the generation leg: the lookup that selected the
        # context is invisible, and its tokens and cost go unrecorded.
        if hasattr(client, "embeddings") and hasattr(client.embeddings, "create"):
            original_embeddings_create = client.embeddings.create
            client.embeddings.create = self.create_span_wrapper(
                span_name="openai.embeddings",
                extract_attributes=self._extract_embedding_attributes,
            )(original_embeddings_create)

        # The Responses API is the default path for native GPT-5.6+ models --
        # Chat Completions rejects function tools combined with reasoning, so
        # agent runtimes route there. Unwrapped it produced no span at all: the
        # instrumentor loaded, reported success, and captured nothing (#26).
        if hasattr(client, "responses") and hasattr(client.responses, "create"):
            original_responses_create = client.responses.create
            client.responses.create = self.create_span_wrapper(
                span_name="openai.responses",
                extract_attributes=self._extract_responses_attributes,
            )(original_responses_create)

    def _instrument_async_client(self, client):
        """Instrument AsyncOpenAI client methods.

        Args:
            client: The AsyncOpenAI client instance to instrument.
        """
        if (
            hasattr(client, "chat")
            and hasattr(client.chat, "completions")
            and hasattr(client.chat.completions, "create")
        ):
            original_create = client.chat.completions.create
            instrumented_create_method = self._create_async_span_wrapper(
                span_name="openai.chat.completion",
                extract_attributes=self._extract_openai_attributes,
            )(original_create)
            client.chat.completions.create = instrumented_create_method

        if hasattr(client, "embeddings") and hasattr(client.embeddings, "create"):
            original_embeddings_create = client.embeddings.create
            client.embeddings.create = self._create_async_span_wrapper(
                span_name="openai.embeddings",
                extract_attributes=self._extract_embedding_attributes,
            )(original_embeddings_create)

        if hasattr(client, "responses") and hasattr(client.responses, "create"):
            original_responses_create = client.responses.create
            client.responses.create = self._create_async_span_wrapper(
                span_name="openai.responses",
                extract_attributes=self._extract_responses_attributes,
            )(original_responses_create)

    def _create_async_span_wrapper(self, span_name, extract_attributes=None):
        """Create an async wrapper that adds OpenTelemetry spans around async calls.

        Args:
            span_name: Name for the span.
            extract_attributes: Optional callable to extract span attributes.

        Returns:
            A decorator function for async methods.
        """
        import asyncio
        import time

        from opentelemetry import context as otel_context
        from opentelemetry import trace
        from opentelemetry.trace import SpanKind, Status, StatusCode

        instrumentor = self

        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                tracer = trace.get_tracer(__name__)
                is_streaming = bool(kwargs.get("stream", False))
                # Started explicitly rather than via start_as_current_span: a
                # streamed call outlives this coroutine, and the context
                # manager would close the span at the `await` below.
                span = tracer.start_span(span_name, kind=SpanKind.CLIENT)
                token = otel_context.attach(trace.set_span_in_context(span))
                handed_to_stream = False
                try:
                    if extract_attributes:
                        attrs = extract_attributes(None, args, kwargs)
                        for key, value in attrs.items():
                            span.set_attribute(key, value)

                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)

                        # Awaiting a streamed call yields the iterator, not the
                        # answer: the generation - and with it TTFT, the final
                        # usage chunk and the true duration - happens while the
                        # caller iterates. Closing the span here timed the
                        # handshake instead, which is why async clients
                        # reported no streaming latency and no token usage.
                        #
                        # This also covers callers that ask for response headers
                        # via with_raw_response (litellm does), where the stream
                        # only appears once .parse() is called.
                        handled, value = instrumentor._install_stream_measurement(
                            span, result, start_time, kwargs.get("model", "unknown"), kwargs
                        )
                        if handled:
                            handed_to_stream = True
                            return value

                        duration = time.time() - start_time
                        span.set_attribute("gen_ai.latency", duration)

                        # Extract response attributes
                        response_attrs = instrumentor._extract_response_attributes(result)
                        for key, value in response_attrs.items():
                            span.set_attribute(key, value)

                        # Record metrics
                        instrumentor._record_result_metrics(span, result, start_time, kwargs)

                        # Add content events
                        instrumentor._add_content_events(span, result, kwargs)

                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise
                finally:
                    otel_context.detach(token)
                    # The stream wrapper owns the span's end once it takes over.
                    if not handed_to_stream:
                        span.end()

            return async_wrapper

        return decorator

    def _extract_openai_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Extract attributes from OpenAI API call.

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

        # Core attributes
        attrs["gen_ai.system"] = "openai"
        attrs["gen_ai.request.model"] = model
        attrs["gen_ai.operation.name"] = "chat"  # NEW: operation name
        attrs["gen_ai.request.message_count"] = len(messages)

        # Request parameters (NEW)
        if "temperature" in kwargs:
            attrs["gen_ai.request.temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            attrs["gen_ai.request.top_p"] = kwargs["top_p"]
        if "max_tokens" in kwargs:
            attrs["gen_ai.request.max_tokens"] = kwargs["max_tokens"]
        if "frequency_penalty" in kwargs:
            attrs["gen_ai.request.frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            attrs["gen_ai.request.presence_penalty"] = kwargs["presence_penalty"]

        # Tool/function definitions (Phase 3.1)
        if "tools" in kwargs:
            try:
                attrs["llm.tools"] = json.dumps(kwargs["tools"])
            except (TypeError, ValueError) as e:
                logger.debug("Failed to serialize tools: %s", e)

        first_message = self._build_first_message(messages)
        if first_message:
            attrs["gen_ai.request.first_message"] = first_message

        return attrs

    @staticmethod
    def _count_embedding_inputs(value: Any) -> int:
        """Count how many texts an embeddings request covers.

        The API accepts a string, a list of strings, a list of token ids, or a
        list of token-id lists. A list of ints is one pre-tokenised input, not
        N inputs, so counting it by length would overstate a batch of one.
        """
        if value is None:
            return 0
        if isinstance(value, str):
            return 1
        if isinstance(value, (list, tuple)):
            if not value:
                return 0
            if all(isinstance(item, int) for item in value):
                return 1
            return len(value)
        return 1

    def _extract_embedding_attributes(
        self, instance: Any, args: Any, kwargs: Any
    ) -> Dict[str, Any]:
        """Extract attributes from an OpenAI embeddings call.

        Args:
            instance: The client instance.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        model = kwargs.get("model", "unknown")

        attrs: Dict[str, Any] = {
            "gen_ai.system": "openai",
            "gen_ai.request.model": model,
            "gen_ai.operation.name": "embeddings",
            # "embedding", singular, is the value CostCalculator.calculate_cost
            # dispatches on. The pricing table's category is "embeddings", so
            # the two deliberately differ; using the plural here would fall
            # through to chat pricing and bill the call at the wrong rate.
            "gen_ai.request.type": "embedding",
            "gen_ai.request.input_count": self._count_embedding_inputs(kwargs.get("input")),
        }

        self.add_embedding_request_attributes(
            attrs,
            dimensions=kwargs.get("dimensions"),
            encoding_format=kwargs.get("encoding_format"),
        )

        return attrs

    @staticmethod
    def _embedding_items(result) -> Optional[list]:
        """Return the embedding items of a response, or None if not embeddings.

        Deliberately strict about the list type: a mock or a chat response
        would otherwise satisfy a bare ``hasattr(result, "data")`` and get
        embedding attributes attached to it.
        """
        data = getattr(result, "data", None)
        if isinstance(data, (list, tuple)) and data and hasattr(data[0], "embedding"):
            return list(data)
        return None

    @staticmethod
    def _responses_output_items(result) -> Optional[list]:
        """Return the Responses ``output`` items, or None if not that shape.

        Checked against ``choices`` first so a Chat Completions response can
        never be read as a Responses one.
        """
        if getattr(result, "choices", None):
            return None
        output = getattr(result, "output", None)
        if isinstance(output, (list, tuple)):
            return list(output)
        return None

    @staticmethod
    def _responses_input_as_messages(raw_input: Any) -> list:
        """Normalise a Responses ``input`` onto the chat ``messages`` shape.

        ``input`` is either a bare string or a list of typed items. Mapping it
        onto the message shape lets the existing content-capture and
        first-message machinery apply unchanged.
        """
        if raw_input is None:
            return []
        if isinstance(raw_input, str):
            return [{"role": "user", "content": raw_input}]
        if isinstance(raw_input, (list, tuple)):
            messages = []
            for item in raw_input:
                if isinstance(item, dict):
                    messages.append(
                        {
                            "role": item.get("role", "user"),
                            "content": _responses_content_to_text(item.get("content", "")),
                        }
                    )
                else:
                    role = getattr(item, "role", None)
                    if role is not None:
                        messages.append(
                            {
                                "role": role,
                                "content": _responses_content_to_text(getattr(item, "content", "")),
                            }
                        )
            return messages
        return []

    @staticmethod
    def _count_responses_input(raw_input: Any) -> int:
        """Count the messages a Responses request carries."""
        if raw_input is None:
            return 0
        if isinstance(raw_input, str):
            return 1
        if isinstance(raw_input, (list, tuple)):
            return len(raw_input)
        return 1

    def _extract_responses_attributes(
        self, instance: Any, args: Any, kwargs: Any
    ) -> Dict[str, Any]:
        """Extract span attributes from a Responses API call.

        Args:
            instance: The client instance.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs: Dict[str, Any] = {}
        raw_input = kwargs.get("input")

        attrs["gen_ai.system"] = "openai"
        attrs["gen_ai.request.model"] = kwargs.get("model", "unknown")
        attrs["gen_ai.operation.name"] = "chat"
        attrs["gen_ai.request.message_count"] = self._count_responses_input(raw_input)

        config = getattr(self, "config", None)
        instructions = kwargs.get("instructions")
        if instructions:
            attrs["gen_ai.request.instructions"] = _cap_content(config, instructions)

        if "temperature" in kwargs:
            attrs["gen_ai.request.temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            attrs["gen_ai.request.top_p"] = kwargs["top_p"]
        # Responses spells the output cap `max_output_tokens`; the semantic
        # convention attribute stays `gen_ai.request.max_tokens`.
        if "max_output_tokens" in kwargs:
            attrs["gen_ai.request.max_tokens"] = kwargs["max_output_tokens"]

        if "tools" in kwargs:
            try:
                attrs["llm.tools"] = json.dumps(kwargs["tools"])
            except (TypeError, ValueError) as e:
                logger.debug("Failed to serialize tools: %s", e)

        first_message = self._build_first_message(self._responses_input_as_messages(raw_input))
        if first_message:
            attrs["gen_ai.request.first_message"] = first_message

        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from OpenAI response.

        Args:
            result: The API response object.

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        if hasattr(result, "usage") and result.usage:
            usage = result.usage

            # The Responses API reports input_tokens/output_tokens where Chat
            # Completions reports prompt_tokens/completion_tokens. Map onto the
            # canonical keys base.py consumes -- it emits both spellings. If
            # this returned None instead, base.py would treat the span as
            # having no usage and price it at zero rather than flag it.
            if getattr(usage, "prompt_tokens", None) is None and hasattr(usage, "input_tokens"):
                responses_usage: Dict[str, Any] = {
                    "prompt_tokens": getattr(usage, "input_tokens", 0) or 0,
                    "completion_tokens": getattr(usage, "output_tokens", 0) or 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                }
                # Reasoning tokens are billed as output, so they are attributed
                # there rather than tracked as a separate bucket.
                out_details = getattr(usage, "output_tokens_details", None)
                reasoning = getattr(out_details, "reasoning_tokens", 0) if out_details else 0
                if reasoning:
                    responses_usage["completion_tokens_details"] = {"reasoning_tokens": reasoning}
                in_details = getattr(usage, "input_tokens_details", None)
                cached = getattr(in_details, "cached_tokens", None) if in_details else None
                if cached:
                    responses_usage["cache_read_input_tokens"] = cached
                return responses_usage

            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }

            # Extract reasoning tokens for o1/o3 models (Phase 3.2). Surfaced
            # to base.py as `gen_ai.usage.reasoning_tokens` (upstream
            # semantic-conventions-genai#76).
            if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                details = usage.completion_tokens_details
                usage_dict["completion_tokens_details"] = {
                    "reasoning_tokens": getattr(details, "reasoning_tokens", 0)
                }

            # Extract OpenAI prompt-cache reads. The Chat Completions API
            # reports cached prompt tokens under prompt_tokens_details.
            # cached_tokens; conceptually identical to Anthropic's
            # cache_read_input_tokens, so surface it under the same canonical
            # key for base.py to emit as
            # `gen_ai.usage.cache_read.input_tokens`.
            if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
                cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", None)
                if cached_tokens:
                    usage_dict["cache_read_input_tokens"] = cached_tokens

            # Per-modality breakdown (semantic-conventions-genai#440). OpenAI
            # reports audio tokens on both detail objects for the realtime and
            # audio-preview models, and text/image tokens on the Responses API.
            # Normalised into the flat keys base.py emits from.
            self._collect_modality_tokens(usage, usage_dict)

            return usage_dict
        return None

    @staticmethod
    def _collect_modality_tokens(usage, usage_dict: Dict[str, Any]) -> None:
        """Copy per-modality token counts off OpenAI's usage detail objects.

        Only counts the provider actually reported are copied; a missing detail
        is left absent rather than defaulted to zero, so a consumer can tell
        "no audio in this request" from "this model does not report modality".
        """
        for details_attr, direction in (
            ("prompt_tokens_details", "input"),
            ("completion_tokens_details", "output"),
        ):
            details = getattr(usage, details_attr, None)
            if not details:
                continue
            for modality in ("text", "image", "audio"):
                value = getattr(details, f"{modality}_tokens", None)
                if isinstance(value, (int, float)) and value > 0:
                    usage_dict[f"{modality}_{direction}_tokens"] = int(value)

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        """Extract response attributes from OpenAI response.

        Args:
            result: The API response object.

        Returns:
            Dict[str, Any]: Dictionary of response attributes.
        """
        attrs = {}

        # Response ID
        if hasattr(result, "id"):
            attrs["gen_ai.response.id"] = result.id

        # Response model (actual model used, may differ from request)
        if hasattr(result, "model"):
            attrs["gen_ai.response.model"] = result.model

        # Embeddings responses carry vectors instead of choices. Recording the
        # count and the dimension is what makes a retrieval span diagnosable:
        # a silently truncated `dimensions` request or a short batch shows up
        # here and nowhere else.
        items = self._embedding_items(result)
        if items is not None:
            attrs["gen_ai.response.embedding_count"] = len(items)
            try:
                attrs["gen_ai.response.vector_size"] = len(items[0].embedding)
            except (TypeError, AttributeError) as e:
                logger.debug("Failed to measure embedding vector: %s", e)
            return attrs

        # Responses carries output[] rather than choices[]: the completion,
        # the tool calls and the reasoning items all live there.
        output_items = self._responses_output_items(result)
        if output_items is not None:
            finish_reason = self._extract_finish_reason(result)
            if finish_reason:
                attrs["gen_ai.response.finish_reasons"] = [finish_reason]

            tc_idx = 0
            for item in output_items:
                if getattr(item, "type", None) != "function_call":
                    continue
                prefix = f"llm.output_messages.0.message.tool_calls.{tc_idx}"
                call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
                if call_id:
                    attrs[f"{prefix}.tool_call.id"] = call_id
                name = getattr(item, "name", None)
                if name:
                    attrs[f"{prefix}.tool_call.function.name"] = name
                arguments = getattr(item, "arguments", None)
                if arguments is not None:
                    attrs[f"{prefix}.tool_call.function.arguments"] = arguments
                tc_idx += 1
            return attrs

        # Finish reasons
        if hasattr(result, "choices") and result.choices:
            finish_reasons = [
                choice.finish_reason
                for choice in result.choices
                if hasattr(choice, "finish_reason")
            ]
            if finish_reasons:
                attrs["gen_ai.response.finish_reasons"] = finish_reasons

            # Tool calls extraction (Phase 3.1)
            for choice_idx, choice in enumerate(result.choices):
                message = getattr(choice, "message", None)
                if message and hasattr(message, "tool_calls") and message.tool_calls:
                    for tc_idx, tool_call in enumerate(message.tool_calls):
                        prefix = f"llm.output_messages.{choice_idx}.message.tool_calls.{tc_idx}"
                        if hasattr(tool_call, "id"):
                            attrs[f"{prefix}.tool_call.id"] = tool_call.id
                        if hasattr(tool_call, "function"):
                            if hasattr(tool_call.function, "name"):
                                attrs[f"{prefix}.tool_call.function.name"] = tool_call.function.name
                            if hasattr(tool_call.function, "arguments"):
                                attrs[f"{prefix}.tool_call.function.arguments"] = (
                                    tool_call.function.arguments
                                )

        return attrs

    def _add_content_events(self, span, result, request_kwargs: dict):
        """Add prompt and completion content as span events and attributes.

        Args:
            span: The OpenTelemetry span.
            result: The API response object.
            request_kwargs: The original request kwargs.
        """
        config = getattr(self, "config", None)

        # Embeddings requests carry `input` rather than `messages`. The text
        # that was embedded is the whole point of a retrieval span - without it
        # you can see that a lookup happened but not what it looked for.
        # Reached only when enable_content_capture is on; base.py gates the
        # call to this method on it.
        #
        # The Responses API also keys on `input`, so the request alone cannot
        # tell the two apart -- routing on it sent every Responses call down
        # the retrieval path, dropping the completion and labelling the span
        # `embedding.model_name`. Decide on the response shape, which is
        # unambiguous.
        if (
            "input" in request_kwargs
            and not request_kwargs.get("messages")
            and self._embedding_items(result) is not None
        ):
            self._add_embedding_content(span, result, request_kwargs, config)
            return

        output_items = self._responses_output_items(result)
        if output_items is not None:
            self._add_responses_content(span, result, request_kwargs, config, output_items)
            return

        # Add prompt content events
        messages = request_kwargs.get("messages", [])
        for idx, message in enumerate(messages):
            if isinstance(message, dict):
                role = message.get("role", "unknown")
                content = message.get("content", "")
                span.add_event(
                    f"gen_ai.prompt.{idx}",
                    attributes={
                        "gen_ai.prompt.role": role,
                        "gen_ai.prompt.content": _cap_content(config, content),
                    },
                )

        # Add completion content events AND attributes (for evaluation processor)
        if hasattr(result, "choices") and result.choices:
            response_text = None
            for idx, choice in enumerate(result.choices):
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    content = _cap_content(config, choice.message.content)
                    # Add as event for observability
                    span.add_event(
                        f"gen_ai.completion.{idx}",
                        attributes={
                            "gen_ai.completion.role": "assistant",
                            "gen_ai.completion.content": content,
                        },
                    )
                    # Capture first completion for evaluation
                    if idx == 0:
                        response_text = content

            # Set as attribute for evaluation processor
            if response_text:
                span.set_attribute("gen_ai.response", response_text)

    def _add_responses_content(self, span, result, request_kwargs: dict, config, output_items):
        """Attach Responses prompts and completions to a span.

        Prompts come from `input` (normalised onto the message shape) and the
        completion from the `message` items of `output[]`; `reasoning` and
        `function_call` items carry no user-visible text and are recorded as
        attributes elsewhere.
        """
        instructions = request_kwargs.get("instructions")
        if instructions:
            span.add_event(
                "gen_ai.prompt.system",
                attributes={
                    "gen_ai.prompt.role": "system",
                    "gen_ai.prompt.content": _cap_content(config, instructions),
                },
            )

        for idx, message in enumerate(
            self._responses_input_as_messages(request_kwargs.get("input"))
        ):
            span.add_event(
                f"gen_ai.prompt.{idx}",
                attributes={
                    "gen_ai.prompt.role": message["role"],
                    "gen_ai.prompt.content": _cap_content(config, message["content"]),
                },
            )

        response_text = None
        completion_idx = 0
        for item in output_items:
            if getattr(item, "type", None) != "message":
                continue
            text = _responses_content_to_text(getattr(item, "content", ""))
            if not text:
                continue
            content = _cap_content(config, text)
            span.add_event(
                f"gen_ai.completion.{completion_idx}",
                attributes={
                    "gen_ai.completion.role": getattr(item, "role", "assistant"),
                    "gen_ai.completion.content": content,
                },
            )
            if completion_idx == 0:
                response_text = content
            completion_idx += 1

        # Set as attribute for the evaluation processor.
        if response_text:
            span.set_attribute("gen_ai.response", response_text)

    def _add_embedding_content(self, span, result, request_kwargs: dict, config):
        """Attach the embedded text (and optionally vectors) to a span.

        Args:
            span: The OpenTelemetry span.
            result: The embeddings response object.
            request_kwargs: The original request kwargs.
            config: The active OTelConfig, or None.
        """
        model = request_kwargs.get("model", "unknown")
        span.set_attribute("embedding.model_name", model)

        raw_input = request_kwargs.get("input")
        if isinstance(raw_input, str):
            texts = [raw_input]
        elif isinstance(raw_input, (list, tuple)) and not all(
            isinstance(item, int) for item in raw_input
        ):
            texts = [item for item in raw_input if isinstance(item, str)]
        else:
            # Pre-tokenised input: there is no text to record.
            texts = []

        if texts:
            span.set_attribute("embedding.text", _cap_content(config, texts[0]))
            for idx, text in enumerate(texts):
                span.add_event(
                    f"gen_ai.embedding.{idx}",
                    attributes={"gen_ai.embedding.content": _cap_content(config, text)},
                )

        # Vectors are large enough to dominate a span and are rarely wanted, so
        # they stay off unless explicitly requested.
        if config is not None and getattr(config, "capture_embedding_vectors", False):
            items = self._embedding_items(result)
            if items:
                try:
                    span.set_attribute("embedding.vector", json.dumps(list(items[0].embedding)))
                    span.set_attribute("embedding.vector.dimension", len(items[0].embedding))
                except (TypeError, ValueError, AttributeError) as e:
                    logger.debug("Failed to serialize embedding vector: %s", e)

    def _extract_finish_reason(self, result) -> Optional[str]:
        """Extract finish reason from OpenAI response.

        Args:
            result: The OpenAI API response object.

        Returns:
            Optional[str]: The finish reason string or None if not available.
        """
        try:
            if hasattr(result, "choices") and result.choices:
                # Get the first finish_reason from the first choice
                first_choice = result.choices[0]
                if hasattr(first_choice, "finish_reason"):
                    return first_choice.finish_reason

            # Responses reports a terminal `status`. When it stopped short,
            # incomplete_details.reason says why, which is the answer an
            # operator actually needs -- prefer it over the bare "incomplete".
            details = getattr(result, "incomplete_details", None)
            reason = getattr(details, "reason", None) if details is not None else None
            if isinstance(reason, str) and reason:
                return reason
            status = getattr(result, "status", None)
            if isinstance(status, str) and status:
                return status
        except Exception as e:
            logger.debug("Failed to extract finish_reason: %s", e)
        return None
