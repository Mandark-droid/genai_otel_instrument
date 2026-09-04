"""OpenTelemetry instrumentor for llama.cpp (``llama-cpp-python``).

Instruments the in-process ``Llama`` class -- ``create_chat_completion``,
``create_completion`` and ``__call__`` -- which is how llama.cpp is used when
embedded in an application rather than run as a server. As with vLLM and
SGLang, none of this traffic crosses HTTP, so no OpenAI-SDK-level
instrumentation can observe it.

llama.cpp returns OpenAI-shaped responses, so token usage comes off the usual
``usage`` block. Where the build reports them, its millisecond ``timings``
(``prompt_ms`` / ``predicted_ms``) map onto the prefill and decode phases of the
latency vocabulary in :mod:`genai_otel.engine_latency`.
"""

import contextvars
import inspect
import logging
from typing import Any, Dict, Optional

import wrapt
from opentelemetry.trace import Status, StatusCode

from ..config import OTelConfig
from ..engine_latency import apply_latency_attributes, llamacpp_latency_attributes
from ..semconv import SemanticConvention as SC
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)

PROVIDER = "llamacpp"

# llama.cpp's entry points delegate to each other: create_chat_completion calls
# create_completion, and __call__ calls it too. Wrapping each one independently
# therefore produced TWO spans for every chat call, both carrying the same token
# counts -- so tokens and cost were counted twice in the metrics. This flag marks
# that a llama.cpp span is already open on this call stack, and the inner call
# runs untraced.
#
# A contextvar rather than an instance attribute, so concurrent calls on
# different threads or tasks do not suppress each other's spans.
_LLAMACPP_SPAN_ACTIVE: contextvars.ContextVar = contextvars.ContextVar(
    "genai_otel_llamacpp_span_active", default=False
)


def _reset_after(generator, token):
    """Re-yield a stream, releasing the dedup guard once it is exhausted."""
    try:
        for item in generator:
            yield item
    finally:
        try:
            _LLAMACPP_SPAN_ACTIVE.reset(token)
        except Exception:  # pragma: no cover - reset in a different context
            _LLAMACPP_SPAN_ACTIVE.set(False)


class LlamaCppInstrumentor(BaseInstrumentor):
    """Instrumentor for the in-process llama.cpp Python bindings."""

    def __init__(self):
        super().__init__()
        self._llamacpp_available = False
        self._llamacpp_module = None
        self._originals: Dict[str, Any] = {}
        self._check_availability()

    def _check_availability(self):
        try:
            import llama_cpp

            self._llamacpp_module = llama_cpp
            self._llamacpp_available = hasattr(llama_cpp, "Llama")
            if not self._llamacpp_available:
                logger.debug("llama_cpp imported but exposes no Llama class; skipping")
        except ImportError:
            logger.debug("llama-cpp-python not installed, instrumentation will be skipped")
            self._llamacpp_available = False
            self._llamacpp_module = None
        except Exception as e:  # pragma: no cover - native library load can fail
            # The bindings load a compiled shared library on import, which fails
            # on a machine without a matching build. Not a reason to fail setup.
            logger.debug("llama_cpp import failed (%s); instrumentation will be skipped", e)
            self._llamacpp_available = False
            self._llamacpp_module = None

    def instrument(self, config: OTelConfig):
        """Wrap the ``Llama`` completion entry points."""
        self.config = config

        if not self._llamacpp_available or self._llamacpp_module is None:
            return

        try:
            llama_cls = self._llamacpp_module.Llama

            if getattr(llama_cls, "_genai_otel_llamacpp_instrumented", False) is True:
                logger.debug("llama.cpp already instrumented, skipping")
                self._instrumented = True
                return

            for method_name, span_name, extractor in (
                (
                    "create_chat_completion",
                    "llamacpp.chat",
                    self._extract_chat_attributes,
                ),
                (
                    "create_completion",
                    "llamacpp.completion",
                    self._extract_completion_attributes,
                ),
                (
                    "create_embedding",
                    "llamacpp.embeddings",
                    self._extract_embedding_attributes,
                ),
            ):
                original = getattr(llama_cls, method_name, None)
                if not callable(original):
                    continue
                self._originals[method_name] = original
                setattr(
                    llama_cls,
                    method_name,
                    self._dedup(
                        self._stream_aware(original, span_name, extractor),
                        original,
                    ),
                )

            llama_cls._genai_otel_llamacpp_instrumented = True
            self._instrumented = True
            logger.info("llama.cpp instrumentation enabled")
        except Exception as e:
            logger.error("Failed to instrument llama.cpp: %s", e)
            if config and config.fail_on_error:
                raise

    def _stream_aware(self, original, span_name, extractor):
        """Trace a call, handling streamed and non-streamed results alike.

        The generic generator path in :mod:`~genai_otel.instrumentors.base`
        ends the span but never inspects the chunks, so a streamed llama.cpp
        call produced a span with no outcome at all. Here the stream is
        re-yielded chunk by chunk and the final chunk's finish reason is
        recorded when it is exhausted.

        Token counts stay absent for streamed calls: llama.cpp emits no
        ``usage`` block in stream mode, and deriving a count from the number of
        chunks would be a guess presented as a measurement.
        """
        traced = self.create_span_wrapper(span_name=span_name, extract_attributes=extractor)(
            original
        )
        instrumentor = self

        @wrapt.decorator
        def dispatch(wrapped, instance, args, kwargs):
            if not kwargs.get("stream"):
                return wrapped(*args, **kwargs)

            try:
                attrs = extractor(instance, args, kwargs)
                attrs[SC.GEN_AI_REQUEST_STREAM] = True
            except Exception:  # pragma: no cover - defensive
                attrs = {"gen_ai.system": PROVIDER, SC.GEN_AI_REQUEST_STREAM: True}
            span = instrumentor.tracer.start_span(span_name, attributes=attrs)

            def traced_stream():
                chunks = []
                try:
                    for chunk in original(instance, *args, **kwargs):
                        chunks.append(chunk)
                        yield chunk
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    span.end()
                    raise
                try:
                    reason = instrumentor._extract_finish_reason(chunks)
                    if reason:
                        span.set_attribute(SC.GEN_AI_RESPONSE_FINISH_REASONS, [reason])
                    span.set_status(Status(StatusCode.OK))
                except Exception as e:  # pragma: no cover - defensive
                    logger.debug("Could not record streamed llama.cpp outcome: %s", e)
                finally:
                    span.end()

            return traced_stream()

        return dispatch(traced)

    @staticmethod
    def _dedup(traced, original):
        """Emit one span per user call, not one per internal delegation.

        ``create_chat_completion`` calls ``create_completion`` internally, so
        without this a single chat call produced a ``llamacpp.chat`` span and a
        nested ``llamacpp.completion`` span carrying the same usage -- double
        counting tokens and cost. The outermost call wins, because that is the
        operation the application actually made.
        """

        # wrapt, not a plain function: these are set as class attributes, so a
        # plain wrapper is never bound and the instrumentor receives
        # instance=None -- which silently lost gen_ai.request.model, since the
        # gguf path is read off the Llama instance.
        @wrapt.decorator
        def guarded(wrapped, instance, args, kwargs):
            if _LLAMACPP_SPAN_ACTIVE.get():
                # Already inside a llama.cpp span: run the untraced original so
                # the delegation does not produce a second span.
                if instance is not None:
                    return original(instance, *args, **kwargs)
                return original(*args, **kwargs)
            token = _LLAMACPP_SPAN_ACTIVE.set(True)
            try:
                result = wrapped(*args, **kwargs)
            except BaseException:
                _LLAMACPP_SPAN_ACTIVE.reset(token)
                raise
            if inspect.isgenerator(result):
                # A streamed call: the guard must stay set for as long as the
                # caller is consuming chunks, not just until the generator
                # object is handed back, or a delegated inner call mid-stream
                # would open a second span.
                return _reset_after(result, token)
            _LLAMACPP_SPAN_ACTIVE.reset(token)
            return result

        return guarded(traced)

    # ------------------------------------------------------------------
    # Attribute extraction
    # ------------------------------------------------------------------

    def _base_attributes(self, instance: Any) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {"gen_ai.system": PROVIDER}
        model = self._model_name(instance)
        if model:
            attrs["gen_ai.request.model"] = model
        return attrs

    @staticmethod
    def _model_name(instance: Any) -> Optional[str]:
        """The loaded GGUF path, which is the only model identity llama.cpp has."""
        for attr in ("model_path", "_model_path"):
            value = getattr(instance, attr, None)
            if isinstance(value, str) and value:
                return value
        return None

    def _extract_chat_attributes(self, instance, args, kwargs) -> Dict[str, Any]:
        attrs = self._base_attributes(instance)
        attrs["gen_ai.operation.name"] = "chat"
        attrs["gen_ai.request.type"] = "chat"
        messages = kwargs.get("messages", args[0] if args else None)
        if isinstance(messages, (list, tuple)):
            attrs["gen_ai.request.input_count"] = len(messages)
        return attrs

    def _extract_completion_attributes(self, instance, args, kwargs) -> Dict[str, Any]:
        attrs = self._base_attributes(instance)
        attrs["gen_ai.operation.name"] = "text_completion"
        attrs["gen_ai.request.type"] = "chat"
        return attrs

    def _extract_embedding_attributes(self, instance, args, kwargs) -> Dict[str, Any]:
        attrs = self._base_attributes(instance)
        attrs["gen_ai.operation.name"] = "embeddings"
        # Singular: this is what CostCalculator dispatches on. The pricing table
        # category is the plural "embeddings" and the two deliberately differ.
        attrs["gen_ai.request.type"] = "embedding"
        return attrs

    # ------------------------------------------------------------------
    # Result handling
    # ------------------------------------------------------------------

    def _extract_usage(self, result) -> Optional[Dict[str, Any]]:
        """Token usage from llama.cpp's OpenAI-shaped ``usage`` block."""
        if not isinstance(result, dict):
            return None
        usage = result.get("usage")
        if not isinstance(usage, dict):
            return None

        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        if not any(isinstance(v, (int, float)) for v in (prompt_tokens, completion_tokens)):
            return None

        extracted: Dict[str, Any] = {
            "prompt_tokens": prompt_tokens or 0,
            "completion_tokens": completion_tokens or 0,
        }
        extracted["total_tokens"] = (
            total_tokens
            if isinstance(total_tokens, (int, float))
            else extracted["prompt_tokens"] + extracted["completion_tokens"]
        )
        return extracted

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return {}

        attrs: Dict[str, Any] = {}
        response_id = result.get("id")
        if response_id:
            attrs["gen_ai.response.id"] = str(response_id)
        model = result.get("model")
        if model:
            attrs["gen_ai.response.model"] = str(model)

        # Present on server-backed responses and on builds compiled with
        # timing support; absent otherwise, in which case no phase attributes
        # are emitted rather than zeros.
        apply_latency_attributes(attrs, llamacpp_latency_attributes(result.get("timings")))
        return attrs

    def _extract_finish_reason(self, result) -> Optional[str]:
        """Finish reason from a completed response or a final streamed chunk.

        Streamed chunks carry no ``usage`` block -- llama.cpp does not report
        token counts in stream mode -- but the last chunk does carry the finish
        reason, so that much is recoverable. Tokens are left absent rather than
        derived from a chunk count, which would be a guess presented as a
        measurement.
        """
        if isinstance(result, (list, tuple)):
            # A drained stream: the reason lives on the final chunk.
            for chunk in reversed(result):
                reason = self._extract_finish_reason(chunk)
                if reason:
                    return reason
            return None
        if not isinstance(result, dict):
            return None
        choices = result.get("choices")
        if not isinstance(choices, (list, tuple)) or not choices:
            return None
        first = choices[0]
        if not isinstance(first, dict):
            return None
        reason = first.get("finish_reason")
        return str(reason) if reason else None
