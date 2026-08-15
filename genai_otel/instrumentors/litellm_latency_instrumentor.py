"""Streaming-latency instrumentation for litellm's own entry points.

litellm reaches OpenAI, Azure and OpenAI-compatible endpoints through the OpenAI
SDK, which this library already instruments -- those calls are measured on the
inner provider span. Every other provider (Anthropic, Bedrock, Gemini, Cohere,
HuggingFace, ...) is implemented with litellm's own httpx handlers, which no
provider instrumentor ever sees. Without this, streaming latency is simply
absent for that whole set.

Wrapping ``litellm.completion`` / ``litellm.acompletion`` catches every route,
because litellm hands back a ``CustomStreamWrapper`` regardless of the transport
underneath.

The span created here is the parent of any inner provider span. When an inner
span measured the request, this one deliberately publishes no TTFT/TPOT/usage of
its own -- a single request must not be counted twice. See issue #22.
"""

import importlib.util
import logging
import sys
import time
from typing import Any, Dict, Optional

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from ..config import OTelConfig
from .base import BaseInstrumentor, close_inner_measurement_scope, inner_measurement_scope

logger = logging.getLogger(__name__)


class LiteLLMLatencyInstrumentor(BaseInstrumentor):
    """Measure litellm calls, including providers it reaches over its own HTTP client."""

    def __init__(self):
        super().__init__()
        # find_spec, not import: importing litellm costs seconds, and this class
        # is constructed during setup. The real import is deferred to
        # instrument(), which only runs when the operator has opted in.
        try:
            self._litellm_available = importlib.util.find_spec("litellm") is not None
        except (ImportError, ValueError):
            # ValueError: something is already in sys.modules under this name
            # but carries no __spec__. Fall back to the plain presence check
            # rather than letting a probe break instrumentation setup.
            self._litellm_available = "litellm" in sys.modules
        logger.debug(
            "litellm %s", "detected" if self._litellm_available else "not installed, skipping"
        )

    def instrument(self, config: OTelConfig):
        self.config = config
        if not self._litellm_available:
            logger.debug("Skipping litellm latency instrumentation - library not available")
            return

        try:
            import litellm

            # Idempotency guard: never stack wrappers if instrument() runs twice.
            if getattr(litellm, "_genai_otel_latency_instrumented", False) is True:
                logger.debug("litellm latency already instrumented, skipping")
                return

            if hasattr(litellm, "acompletion"):
                litellm.acompletion = self._wrap_async(litellm.acompletion, "litellm.acompletion")
            if hasattr(litellm, "completion"):
                litellm.completion = self._wrap_sync(litellm.completion, "litellm.completion")

            try:
                litellm._genai_otel_latency_instrumented = True
            except Exception:  # noqa: BLE001
                pass

            self._instrumented = True
            logger.info("litellm streaming latency instrumentation enabled")
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to instrument litellm: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _start_span(self, span_name: str, kwargs):
        span = self.tracer.start_span(span_name, kind=SpanKind.CLIENT)
        model = kwargs.get("model", "unknown")
        span.set_attribute("gen_ai.system", "litellm")
        span.set_attribute("gen_ai.request.model", str(model))
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.request.type", "chat")
        return span, model

    def _wrap_sync(self, original, span_name: str):
        instrumentor = self

        def wrapper(*args, **kwargs):
            span, model = instrumentor._start_span(span_name, kwargs)
            token = otel_context.attach(trace.set_span_in_context(span))
            # Watch whether an inner provider SDK we instrument takes over.
            holder, scope_token = inner_measurement_scope()
            start_time = time.time()
            handed_to_stream = False
            try:
                result = original(*args, **kwargs)
                handled, value = instrumentor._install_stream_measurement(
                    span,
                    result,
                    start_time,
                    model,
                    kwargs,
                    emit_measurements=not holder,
                )
                if handled:
                    handed_to_stream = True
                    return value

                if not holder:
                    instrumentor._record_result_metrics(span, result, start_time, kwargs)
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            finally:
                close_inner_measurement_scope(scope_token)
                otel_context.detach(token)
                if not handed_to_stream:
                    span.end()

        return wrapper

    def _wrap_async(self, original, span_name: str):
        instrumentor = self

        async def wrapper(*args, **kwargs):
            span, model = instrumentor._start_span(span_name, kwargs)
            token = otel_context.attach(trace.set_span_in_context(span))
            holder, scope_token = inner_measurement_scope()
            start_time = time.time()
            handed_to_stream = False
            try:
                result = await original(*args, **kwargs)
                handled, value = instrumentor._install_stream_measurement(
                    span,
                    result,
                    start_time,
                    model,
                    kwargs,
                    emit_measurements=not holder,
                )
                if handled:
                    handed_to_stream = True
                    return value

                if not holder:
                    instrumentor._record_result_metrics(span, result, start_time, kwargs)
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            finally:
                close_inner_measurement_scope(scope_token)
                otel_context.detach(token)
                if not handed_to_stream:
                    span.end()

        return wrapper

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from a litellm response or final stream chunk."""
        usage: Any = getattr(result, "usage", None)
        if usage is None and isinstance(result, dict):
            usage = result.get("usage")
        if usage is None:
            return None

        def _get(name):
            if isinstance(usage, dict):
                return usage.get(name)
            return getattr(usage, name, None)

        prompt = _get("prompt_tokens")
        completion = _get("completion_tokens")
        total = _get("total_tokens")
        if prompt is None and completion is None and total is None:
            return None
        return {
            "prompt_tokens": int(prompt or 0),
            "completion_tokens": int(completion or 0),
            "total_tokens": int(total or ((prompt or 0) + (completion or 0))),
        }
