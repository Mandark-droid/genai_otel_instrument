"""OpenTelemetry instrumentor for the Anthropic Claude SDK.

This instrumentor automatically traces calls to the Anthropic API, capturing
relevant attributes such as model name, message count, and token usage.
"""

import logging
import time
from typing import Any, Dict, Optional

import wrapt
from opentelemetry import context as otel_context
from opentelemetry import trace

from ..config import OTelConfig
from ..server_metrics import get_server_metrics
from .base import BaseInstrumentor, _StreamTiming, find_base_url_claim

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


def _measured_iter(wrapped, instrumentor, span, timing, model):
    """Yield a sync MessageStream's events, observing each one."""
    for event in wrapped:
        instrumentor._observe_stream_chunk(span, timing, event, model)
        yield event


async def _measured_aiter(wrapped, instrumentor, span, timing, model):
    """Yield an async MessageStream's events, observing each one."""
    async for event in wrapped:
        instrumentor._observe_stream_chunk(span, timing, event, model)
        yield event


class _MeasuredMessageStream(wrapt.ObjectProxy):
    """A MessageStream whose iteration is measured, and nothing else changed.

    A transparent proxy rather than a replacement iterator: callers reach past
    iteration for ``get_final_message()``, ``get_final_text()``,
    ``until_done()``, ``text_stream`` and ``response``, and handing back a bare
    generator would break every one of them.
    """

    def __init__(self, wrapped, instrumentor, span, timing, model):
        super().__init__(wrapped)
        # ObjectProxy reserves the `_self_` prefix for attributes belonging to
        # the proxy rather than to the wrapped object.
        self._self_instrumentor = instrumentor
        self._self_span = span
        self._self_timing = timing
        self._self_model = model

    def __iter__(self):
        return _measured_iter(
            self.__wrapped__,
            self._self_instrumentor,
            self._self_span,
            self._self_timing,
            self._self_model,
        )


class _MeasuredAsyncMessageStream(_MeasuredMessageStream):
    """Async counterpart of :class:`_MeasuredMessageStream`."""

    def __aiter__(self):
        return _measured_aiter(
            self.__wrapped__,
            self._self_instrumentor,
            self._self_span,
            self._self_timing,
            self._self_model,
        )


class _MeasuredStreamManagerBase:
    """Owns the span for one ``messages.stream()`` call.

    The context manager owns the span's lifetime, not the iterator. A caller
    may exhaust the stream, abandon it half-way, or never iterate at all and
    ask only for ``get_final_message()`` -- but it must always leave the
    ``with`` block, so ``__exit__`` is the one place that reliably runs. Giving
    ownership to the iterator instead would leak a span for every caller who
    took the final message without iterating.
    """

    def __init__(self, manager, instrumentor, span, model):
        self._manager = manager
        self._instrumentor = instrumentor
        self._span = span
        self._model = model
        self._timing = None
        self._stream = None
        self._finished = False
        self._token = None

    def _begin(self):
        # Timed from __enter__, not from the stream() call: stream() only
        # builds the manager, the request is issued on entry.
        self._timing = _StreamTiming(time.time())
        try:
            self._token = otel_context.attach(trace.set_span_in_context(self._span))
        except Exception as e:  # noqa: BLE001 - context must not break the call
            logger.debug("Could not attach context for anthropic.messages.stream: %s", e)
        server_metrics = get_server_metrics()
        if server_metrics:
            server_metrics.increment_requests_running()

    def _finish(self, exc):
        if self._finished:
            return
        self._finished = True
        if self._token is not None:
            try:
                otel_context.detach(self._token)
            except Exception as e:  # noqa: BLE001
                logger.debug("Could not detach context for anthropic.messages.stream: %s", e)
        if exc is not None:
            self._instrumentor._fail_stream(self._span, exc)
            return
        timing = self._timing if self._timing is not None else _StreamTiming(time.time())
        if timing.usage is None:
            # Nothing was observed during iteration, so the caller took the
            # answer some other way. The accumulated message still carries the
            # usage; without this the span would report no cost purely because
            # of how the caller chose to read the stream.
            self._instrumentor._adopt_final_usage(timing, self._stream)
        self._instrumentor._finalize_stream(self._span, timing, self._model)


class _MeasuredStreamManager(_MeasuredStreamManagerBase):
    """Sync ``with client.messages.stream(...) as stream:``."""

    def __enter__(self):
        self._begin()
        try:
            stream = self._manager.__enter__()
        except BaseException as e:
            self._finish(e)
            raise
        self._stream = stream
        return _MeasuredMessageStream(
            stream, self._instrumentor, self._span, self._timing, self._model
        )

    def __exit__(self, exc_type, exc, tb):
        try:
            return self._manager.__exit__(exc_type, exc, tb)
        finally:
            self._finish(exc)


class _MeasuredAsyncStreamManager(_MeasuredStreamManagerBase):
    """Async ``async with client.messages.stream(...) as stream:``."""

    async def __aenter__(self):
        self._begin()
        try:
            stream = await self._manager.__aenter__()
        except BaseException as e:
            self._finish(e)
            raise
        self._stream = stream
        return _MeasuredAsyncMessageStream(
            stream, self._instrumentor, self._span, self._timing, self._model
        )

    async def __aexit__(self, exc_type, exc, tb):
        try:
            return await self._manager.__aexit__(exc_type, exc, tb)
        finally:
            self._finish(exc)


class AnthropicInstrumentor(BaseInstrumentor):
    """Instrumentor for Anthropic Claude SDK"""

    MEDIA_PROVIDER = "anthropic"

    # Every client class the SDK exposes that carries a `messages` resource.
    # They are independent classes, so wrapping `Anthropic` alone catches none
    # of the others: an application built on `AsyncAnthropic` instrumented
    # cleanly and emitted no spans whatsoever.
    #
    # The Bedrock and Vertex entries are the *Anthropic* SDK pointed at those
    # clouds. They do not overlap with `aws_bedrock_instrumentor` (which wraps
    # boto3) or `vertexai_instrumentor` (which wraps Google's SDK), so there is
    # no double-counting. The cloud-hosted variants ship behind optional
    # extras, so each name is probed rather than imported.
    CLIENT_CLASSES = (
        "Anthropic",
        "AsyncAnthropic",
        "AnthropicBedrock",
        "AsyncAnthropicBedrock",
        "AnthropicVertex",
        "AsyncAnthropicVertex",
        "AnthropicAWS",
        "AsyncAnthropicAWS",
        "AnthropicFoundry",
        "AsyncAnthropicFoundry",
        "AnthropicGoogleCloud",
        "AsyncAnthropicGoogleCloud",
    )

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._anthropic_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if Anthropic library is available."""
        try:
            import anthropic

            self._anthropic_available = True
            logger.debug("Anthropic library detected and available for instrumentation")
        except ImportError:
            logger.debug("Anthropic library not installed, instrumentation will be skipped")
            self._anthropic_available = False

    def instrument(self, config: OTelConfig):
        """Instrument Anthropic SDK if available.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._anthropic_available:
            logger.debug("Skipping Anthropic instrumentation - library not available")
            return

        self.config = config

        try:
            import anthropic
            import wrapt

            # Idempotency guard: never stack wrappers if instrument() runs twice.
            if getattr(anthropic, "_genai_otel_anthropic_instrumented", False) is True:
                logger.debug("Anthropic already instrumented, skipping")
                self._instrumented = True
                return

            def wrapped_init(wrapped, instance, args, kwargs):
                result = wrapped(*args, **kwargs)
                self._instrument_client(instance)
                return result

            wrapped_classes = []
            for class_name in self.CLIENT_CLASSES:
                client_class = getattr(anthropic, class_name, None)
                if client_class is None:
                    continue
                try:
                    client_class.__init__ = wrapt.FunctionWrapper(
                        client_class.__init__, wrapped_init
                    )
                except (AttributeError, TypeError) as e:
                    logger.debug("Could not wrap anthropic.%s.__init__: %s", class_name, e)
                    continue
                wrapped_classes.append(class_name)

            if wrapped_classes:
                # Marked done only once every client class is wrapped. Setting
                # this inside the `Anthropic` branch meant a run that wrapped
                # the sync client alone flagged the module as instrumented,
                # and every later run returned early -- so the async and
                # cloud clients could never be picked up afterwards.
                try:
                    anthropic._genai_otel_anthropic_instrumented = True
                except Exception:  # noqa: BLE001
                    pass
                self._instrumented = True
                logger.debug("Instrumented Anthropic clients: %s", ", ".join(wrapped_classes))
                logger.info("Anthropic instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument Anthropic: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _instrument_client(self, client):
        """Instrument Anthropic client methods.

        Args:
            client: The Anthropic client instance to instrument.
        """
        # A client pointed at an aggregator (e.g. CometAPI) is traced by its
        # dedicated instrumentor; wrapping it here too would emit a duplicate
        # span and double-count token/cost metrics.
        claimed = find_base_url_claim(getattr(client, "base_url", None))
        if claimed:
            logger.debug(
                "Skipping generic Anthropic instrumentation for client handled "
                "by the '%s' instrumentor",
                claimed,
            )
            return
        if hasattr(client, "messages") and hasattr(client.messages, "create"):
            original_create = client.messages.create
            instrumented_create_method = self.create_span_wrapper(
                span_name="anthropic.messages.create",
                extract_attributes=self._extract_anthropic_attributes,
            )(original_create)
            client.messages.create = instrumented_create_method

        if hasattr(client, "messages") and hasattr(client.messages, "stream"):
            self._instrument_stream_method(client)

    def _instrument_stream_method(self, client):
        """Wrap ``messages.stream()``, whose span has to outlive the call.

        ``stream()`` hands back a context manager rather than an iterator, and
        never sets ``stream=True``. So ``create_span_wrapper`` sees an ordinary
        buffered call and closes the span on the handshake -- while the
        generation, the tokens and the cost all happen later, inside the
        caller's ``with`` block. Measuring it needs the context-manager
        protocol, which is why this path is wrapped by hand.
        """
        original_stream = client.messages.stream
        instrumentor = self

        def wrapped_stream(*args, **kwargs):
            if not instrumentor._instrumented:
                return original_stream(*args, **kwargs)

            manager = original_stream(*args, **kwargs)
            try:
                span = instrumentor._start_stream_span(kwargs)
            except Exception as e:  # noqa: BLE001 - never break the caller's call
                logger.debug("Could not start span for anthropic.messages.stream: %s", e)
                return manager

            model = kwargs.get("model", "unknown")
            if hasattr(manager, "__aenter__"):
                return _MeasuredAsyncStreamManager(manager, instrumentor, span, model)
            return _MeasuredStreamManager(manager, instrumentor, span, model)

        client.messages.stream = wrapped_stream

    def _start_stream_span(self, kwargs):
        """Open the span for a ``messages.stream()`` call."""
        attributes = {}
        try:
            attributes = self._with_provider_aliases(
                self._extract_anthropic_attributes(None, (), kwargs)
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to extract attributes for anthropic.messages.stream: %s", e)
        return self.tracer.start_span("anthropic.messages.stream", attributes=attributes)

    def _adopt_final_usage(self, timing, stream) -> None:
        """Take usage off the accumulated message when nothing was iterated.

        ``until_done()`` and ``get_final_message()`` consume the stream
        internally, so our measured ``__iter__`` never runs and no event is
        ever observed. The snapshot holds the same usage the events carried.
        """
        if stream is None:
            return
        try:
            snapshot = getattr(stream, "current_message_snapshot", None)
            usage = self._extract_usage(snapshot) if snapshot is not None else None
        except Exception as e:  # noqa: BLE001 - a missing count is not an outage
            logger.debug("Could not read the final Anthropic message snapshot: %s", e)
            return
        if usage:
            timing.usage = usage

    def _extract_anthropic_attributes(
        self, instance: Any, args: Any, kwargs: Any
    ) -> Dict[str, Any]:
        """Extract attributes from Anthropic API call.

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

        attrs["gen_ai.system"] = "anthropic"
        attrs["gen_ai.request.model"] = model
        attrs["gen_ai.request.message_count"] = len(messages)

        first_message = self._build_first_message(messages)
        if first_message:
            attrs["gen_ai.request.first_message"] = first_message

        return attrs

    @staticmethod
    def _usage_object(result):
        """Return the usage payload of a response or stream event, or None.

        Anthropic reports it in two shapes. A buffered ``Message`` and a
        ``message_delta`` event carry ``.usage`` directly; ``message_start``
        and the high-level ``message_stop`` carry it one level down, on the
        partial or accumulated ``.message``.

        A ``.usage`` that exists but is None means the provider reported no
        usage for this event -- that is an answer, not an invitation to go
        looking on ``.message``.
        """
        if hasattr(result, "usage"):
            return result.usage
        message = getattr(result, "message", None)
        if message is not None:
            return getattr(message, "usage", None)
        return None

    @staticmethod
    def _token_count(usage, name: str) -> int:
        """Read one token count, treating anything non-numeric as absent.

        The streaming usage models declare several of these Optional, so a
        real response can hand back None where the buffered one gives an int.
        Adding None to an int would raise inside the finalizer and lose the
        whole usage dict, cost included.
        """
        value = getattr(usage, name, 0)
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from an Anthropic response or stream event.

        Args:
            result: The API response object, or one streamed event.

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        usage = self._usage_object(result)
        if not usage:
            return None

        input_tokens = self._token_count(usage, "input_tokens")
        output_tokens = self._token_count(usage, "output_tokens")
        usage_dict = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

        # Extract cache tokens for Anthropic models (Phase 3.2)
        # cache_read_input_tokens: Tokens that were read from cache
        # cache_creation_input_tokens: Tokens that were written to cache
        if hasattr(usage, "cache_read_input_tokens"):
            usage_dict["cache_read_input_tokens"] = self._token_count(
                usage, "cache_read_input_tokens"
            )
        if hasattr(usage, "cache_creation_input_tokens"):
            usage_dict["cache_creation_input_tokens"] = self._token_count(
                usage, "cache_creation_input_tokens"
            )

        return usage_dict

    def _accumulate_stream_usage(self, timing, chunk) -> None:
        """Fold usage across a streamed Anthropic message.

        Anthropic spreads usage over the stream: ``message_start`` carries the
        input tokens, ``message_delta`` the output tokens, and the final
        ``message_stop`` carries none at all. The base finalizer reads only the
        last chunk, so before this existed a streamed call produced a span with
        latency and no economics -- no token counts, and cost never calculated
        rather than calculated as zero.

        Counts are merged with ``max`` rather than summed. This is not a
        defensive guess -- it is what the provider documents. Anthropic's
        streaming reference says, in a warning block:

            "The token counts shown in the ``usage`` field of the
            ``message_delta`` event are *cumulative*."

            https://platform.claude.com/docs/en/build-with-claude/streaming

        A tool-use stream sends several ``message_delta`` events, each
        restating the running total rather than adding to it, so summing them
        would report several times the tokens actually used and bill the
        customer for them. **Do not "fix" this into a sum.**
        ``test_cumulative_message_deltas_are_not_summed`` pins it: deltas of
        5 then 9 must yield 9, and a sum would yield 14.
        """
        usage = self._extract_usage(chunk)
        if not usage:
            return

        merged = dict(timing.usage) if timing.usage else {}
        for key, value in usage.items():
            if not isinstance(value, (int, float)) or value <= 0:
                continue
            previous = merged.get(key, 0)
            if not isinstance(previous, (int, float)):
                previous = 0
            merged[key] = max(previous, value)

        if "prompt_tokens" in merged or "completion_tokens" in merged:
            merged["total_tokens"] = merged.get("prompt_tokens", 0) + merged.get(
                "completion_tokens", 0
            )

        if merged:
            timing.usage = merged

    def _add_content_events(self, span, result, request_kwargs: dict):
        """Add prompt and completion content as span events and attributes.

        Args:
            span: The OpenTelemetry span.
            result: The API response object.
            request_kwargs: The original request kwargs.
        """
        config = getattr(self, "config", None)

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
        if hasattr(result, "content") and result.content:
            response_text = None
            for idx, content_block in enumerate(result.content):
                if hasattr(content_block, "text"):
                    content = _cap_content(config, content_block.text)
                    # Add as event for observability
                    span.add_event(
                        f"gen_ai.completion.{idx}",
                        attributes={
                            "gen_ai.completion.role": "assistant",
                            "gen_ai.completion.content": content,
                        },
                    )
                    # Capture first text block for evaluation
                    if idx == 0:
                        response_text = content

            # Set as attribute for evaluation processor
            if response_text:
                span.set_attribute("gen_ai.response", response_text)
