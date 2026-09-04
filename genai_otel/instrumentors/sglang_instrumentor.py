"""OpenTelemetry instrumentor for SGLang.

Instruments the in-process SGLang engine (``sglang.Engine.generate`` and its
async counterpart), for the same reason as the vLLM instrumentor: offline
generation through the Python API never makes an HTTP request, so no
OpenAI-SDK-level instrumentation can see it.

SGLang returns per-request metadata in a ``meta_info`` mapping alongside the
generated text -- token counts, finish reason, and timing fields whose coverage
varies by release. Each is read independently so a version that omits one still
produces a complete span for everything else.
"""

import contextvars
import inspect
import logging
from typing import Any, Dict, List, Optional

import wrapt

from ..config import OTelConfig
from ..engine_latency import apply_latency_attributes, sglang_latency_attributes
from ..semconv import SemanticConvention as SC
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)

PROVIDER = "sglang"

# SGLang's synchronous Engine.generate drives async_generate through an event
# loop on several releases, so wrapping both independently would emit two spans
# for one call and count its tokens twice. This marks that an SGLang span is
# already open on this call stack; the inner call then runs untraced.
#
# A contextvar rather than an instance attribute, so concurrent calls on
# different threads or tasks cannot suppress each other's spans.
_SGLANG_SPAN_ACTIVE: contextvars.ContextVar = contextvars.ContextVar(
    "genai_otel_sglang_span_active", default=False
)


def _reset_after(generator, token):
    """Re-yield a stream, releasing the dedup guard once it is exhausted."""
    try:
        for item in generator:
            yield item
    finally:
        try:
            _SGLANG_SPAN_ACTIVE.reset(token)
        except Exception:  # pragma: no cover - reset in a different context
            _SGLANG_SPAN_ACTIVE.set(False)


def _dedup(traced, original):
    """Emit one span per user call, not one per internal delegation."""

    @wrapt.decorator
    def guarded(wrapped, instance, args, kwargs):
        if _SGLANG_SPAN_ACTIVE.get():
            if instance is not None:
                return original(instance, *args, **kwargs)
            return original(*args, **kwargs)
        token = _SGLANG_SPAN_ACTIVE.set(True)
        try:
            result = wrapped(*args, **kwargs)
        except BaseException:
            _SGLANG_SPAN_ACTIVE.reset(token)
            raise
        if inspect.isgenerator(result):
            return _reset_after(result, token)
        if inspect.isasyncgen(result) or inspect.iscoroutine(result):
            # Async results are consumed after this returns; releasing the guard
            # now is correct because the consumer runs in its own context.
            _SGLANG_SPAN_ACTIVE.reset(token)
            return result
        _SGLANG_SPAN_ACTIVE.reset(token)
        return result

    return guarded(traced)


class SGLangInstrumentor(BaseInstrumentor):
    """Instrumentor for the in-process SGLang engine."""

    def __init__(self):
        super().__init__()
        self._sglang_available = False
        self._sglang_module = None
        self._original_generate = None
        self._original_async_generate = None
        self._check_availability()

    def _check_availability(self):
        try:
            import sglang

            self._sglang_module = sglang
            self._sglang_available = hasattr(sglang, "Engine")
            if not self._sglang_available:
                logger.debug("sglang imported but exposes no Engine class; skipping")
        except ImportError:
            logger.debug("SGLang not installed, instrumentation will be skipped")
            self._sglang_available = False
            self._sglang_module = None
        except Exception as e:  # pragma: no cover - import needs a GPU runtime
            logger.debug("SGLang import failed (%s); instrumentation will be skipped", e)
            self._sglang_available = False
            self._sglang_module = None

    def instrument(self, config: OTelConfig):
        """Wrap ``sglang.Engine.generate`` and ``async_generate``."""
        self.config = config

        if not self._sglang_available or self._sglang_module is None:
            return

        try:
            engine_cls = self._sglang_module.Engine

            if getattr(engine_cls, "_genai_otel_sglang_instrumented", False) is True:
                logger.debug("SGLang already instrumented, skipping")
                self._instrumented = True
                return

            original = getattr(engine_cls, "generate", None)
            if callable(original):
                self._original_generate = original
                engine_cls.generate = _dedup(
                    self.create_span_wrapper(
                        span_name="sglang.generate",
                        extract_attributes=self._extract_generate_attributes,
                    )(original),
                    original,
                )

            # Async entry point, present on current releases. Wrapped through
            # the same span wrapper, which handles awaitable results.
            original_async = getattr(engine_cls, "async_generate", None)
            if callable(original_async):
                self._original_async_generate = original_async
                engine_cls.async_generate = _dedup(
                    self.create_span_wrapper(
                        span_name="sglang.generate",
                        extract_attributes=self._extract_generate_attributes,
                    )(original_async),
                    original_async,
                )

            engine_cls._genai_otel_sglang_instrumented = True
            self._instrumented = True
            logger.info("SGLang instrumentation enabled")
        except Exception as e:
            logger.error("Failed to instrument SGLang: %s", e)
            if config and config.fail_on_error:
                raise

    # ------------------------------------------------------------------
    # Attribute extraction
    # ------------------------------------------------------------------

    def _extract_generate_attributes(self, instance, args, kwargs) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {
            "gen_ai.system": PROVIDER,
            "gen_ai.operation.name": "text_completion",
            "gen_ai.request.type": "chat",
        }

        model = self._model_name(instance)
        if model:
            attrs["gen_ai.request.model"] = model

        prompt = kwargs.get("prompt", args[0] if args else None)
        count = self._prompt_count(prompt)
        if count is not None:
            attrs["gen_ai.request.input_count"] = count

        # SGLang takes sampling settings as a plain dict, so the kwargs-based
        # derivation in base.py does not reach them.
        sampling = kwargs.get("sampling_params")
        if isinstance(sampling, dict):
            for source, target in (
                ("max_new_tokens", "gen_ai.request.max_tokens"),
                ("temperature", "gen_ai.request.temperature"),
                ("top_p", "gen_ai.request.top_p"),
                ("top_k", SC.GEN_AI_REQUEST_TOP_K),
                ("n", SC.GEN_AI_REQUEST_CHOICE_COUNT),
            ):
                value = sampling.get(source)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    attrs[target] = value

        return attrs

    @staticmethod
    def _model_name(instance: Any) -> Optional[str]:
        """Served model path from an ``Engine`` instance, across layouts."""
        for path in (
            ("server_args", "model_path"),
            ("tokenizer_manager", "server_args", "model_path"),
        ):
            target = instance
            for step in path:
                target = getattr(target, step, None)
                if target is None:
                    break
            if isinstance(target, str) and target:
                return target
        return None

    @staticmethod
    def _prompt_count(prompt: Any) -> Optional[int]:
        if prompt is None or isinstance(prompt, (str, bytes, dict)):
            return 1 if prompt is not None else None
        try:
            return len(prompt)
        except TypeError:
            return None

    # ------------------------------------------------------------------
    # Result handling
    # ------------------------------------------------------------------

    @staticmethod
    def _as_result_list(result: Any) -> List[Dict[str, Any]]:
        """Normalise an SGLang result to a list of result dicts."""
        if result is None:
            return []
        if isinstance(result, dict):
            return [result]
        if isinstance(result, (list, tuple)):
            return [r for r in result if isinstance(r, dict)]
        return []

    def _extract_usage(self, result) -> Optional[Dict[str, Any]]:
        """Sum token usage across an SGLang batch result."""
        results = self._as_result_list(result)
        if not results:
            return None

        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0
        for item in results:
            meta = item.get("meta_info")
            if not isinstance(meta, dict):
                continue
            for key, target in (
                ("prompt_tokens", "prompt"),
                ("completion_tokens", "completion"),
                ("cached_tokens", "cached"),
            ):
                value = meta.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if target == "prompt":
                        prompt_tokens += int(value)
                    elif target == "completion":
                        completion_tokens += int(value)
                    else:
                        cached_tokens += int(value)

        if not prompt_tokens and not completion_tokens:
            return None

        usage: Dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if cached_tokens:
            # SGLang's cached_tokens is a prefix-cache hit count, the same
            # concept the conventions call cache_read.
            usage["cache_read_input_tokens"] = cached_tokens
        return usage

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        results = self._as_result_list(result)
        if not results:
            return {}

        attrs: Dict[str, Any] = {}
        first_meta = results[0].get("meta_info")
        if isinstance(first_meta, dict):
            request_id = first_meta.get("id")
            if request_id:
                attrs[SC.GEN_AI_REQUEST_ID] = str(request_id)
            apply_latency_attributes(attrs, sglang_latency_attributes(first_meta))

        # The conventions define finish_reasons as an array, and an SGLang
        # batch genuinely ends different ways -- one request hits EOS while
        # another hits the token cap. Reporting only the first would hide the
        # truncated requests, which are the ones worth finding.
        reasons = self._extract_finish_reasons(results)
        if reasons:
            attrs[SC.GEN_AI_RESPONSE_FINISH_REASONS] = reasons

        if len(results) > 1:
            attrs["gen_ai.response.output_count"] = len(results)
        return attrs

    def _extract_finish_reasons(self, results: List[Dict[str, Any]]) -> List[str]:
        """Every distinct finish reason in a batch, in first-seen order.

        De-duplicated so a large batch that all hit the cap reports
        ``["length"]`` rather than the same string once per request.
        """
        reasons: List[str] = []
        for item in results:
            meta = item.get("meta_info")
            if not isinstance(meta, dict):
                continue
            reason = meta.get("finish_reason")
            if isinstance(reason, dict):
                reason = reason.get("type")
            if reason and str(reason) not in reasons:
                reasons.append(str(reason))
        return reasons

    def _extract_finish_reason(self, result) -> Optional[str]:
        for item in self._as_result_list(result):
            meta = item.get("meta_info")
            if not isinstance(meta, dict):
                continue
            reason = meta.get("finish_reason")
            # SGLang reports this as either a bare string or {"type": ...}.
            if isinstance(reason, dict):
                reason = reason.get("type")
            if reason:
                return str(reason)
        return None
