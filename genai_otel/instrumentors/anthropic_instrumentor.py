"""OpenTelemetry instrumentor for the Anthropic Claude SDK.

This instrumentor automatically traces calls to the Anthropic API, capturing
relevant attributes such as model name, message count, and token usage.
"""

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

        Counts are merged with ``max`` rather than summed because Anthropic
        reports them cumulatively: a tool-use stream sends several
        ``message_delta`` events, each restating the running output total, and
        adding those would multiply the bill.
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
