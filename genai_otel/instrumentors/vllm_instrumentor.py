"""OpenTelemetry instrumentor for vLLM.

Instruments the **in-process** vLLM Python API (``vllm.LLM.generate`` and
``vllm.LLM.chat``), not the OpenAI-compatible HTTP server. That distinction is
the point: a vLLM server can be traced by pointing any OpenAI-SDK
instrumentation at it, but offline batch inference through ``LLM.generate()``
is invisible to every HTTP-based approach, because no HTTP request is ever
made.

**Engine latency availability.** The queue / prefill / decode breakdown comes
from the ``RequestMetrics`` object vLLM attaches to each ``RequestOutput``. The
**V1 engine does not populate it** -- ``RequestOutput.metrics`` is ``None``, and
V1 exposes no per-request timing on the Python API at all (verified against
vLLM 0.24 and 0.27). So on any current vLLM the ``gen_ai.latency.*`` attributes
are simply absent, and everything else on the span is unaffected.

They are still emitted wherever ``metrics`` is populated (V0-era engines and
builds that fill it in), because reporting a real breakdown when one exists is
worth more than dropping the capability, and inventing timings from wall-clock
guesses would be worse than emitting nothing: the span duration already records
end-to-end time honestly, and a fabricated "prefill" number is not recoverable
by a consumer that trusts it.

``num_cached_tokens`` *is* available on V1 and carries the prefix-cache hit
count, so that is read separately. See :mod:`genai_otel.engine_latency`.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import wrapt
from opentelemetry.trace import Status, StatusCode

from ..config import OTelConfig
from ..engine_latency import apply_latency_attributes, vllm_latency_attributes
from ..semconv import SemanticConvention as SC
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)

PROVIDER = "vllm"


class VLLMInstrumentor(BaseInstrumentor):
    """Instrumentor for the in-process vLLM engine."""

    def __init__(self):
        super().__init__()
        self._vllm_available = False
        self._vllm_module = None
        self._original_generate = None
        self._original_chat = None
        self._check_availability()

    def _check_availability(self):
        """Check whether vLLM is importable.

        Importing vLLM is expensive and can initialise CUDA, so this only
        touches the top-level module, which is enough to know whether the
        ``LLM`` class is there to wrap.
        """
        try:
            import vllm

            self._vllm_module = vllm
            self._vllm_available = hasattr(vllm, "LLM")
            if not self._vllm_available:
                logger.debug("vllm imported but exposes no LLM class; skipping instrumentation")
        except ImportError:
            logger.debug("vLLM not installed, instrumentation will be skipped")
            self._vllm_available = False
            self._vllm_module = None
        except Exception as e:  # pragma: no cover - import side effects vary by build
            # vLLM can raise on import when no compatible GPU/driver is present.
            # That is a perfectly normal machine for this library to run on, so
            # it must not take instrumentation setup down with it.
            logger.debug("vLLM import failed (%s); instrumentation will be skipped", e)
            self._vllm_available = False
            self._vllm_module = None

    def instrument(self, config: OTelConfig):
        """Wrap ``vllm.LLM.generate`` and ``vllm.LLM.chat``."""
        self.config = config

        if not self._vllm_available or self._vllm_module is None:
            return

        try:
            llm_cls = self._vllm_module.LLM

            if getattr(llm_cls, "_genai_otel_vllm_instrumented", False) is True:
                logger.debug("vLLM already instrumented, skipping")
                self._instrumented = True
                return

            for method_name, span_name, extractor in (
                ("generate", "vllm.generate", self._extract_generate_attributes),
                ("chat", "vllm.chat", self._extract_chat_attributes),
                # Offline embeddings. Without this an embeddings workload run
                # through vLLM produces no retrieval leg at all.
                ("encode", "vllm.embeddings", self._extract_encode_attributes),
            ):
                original = getattr(llm_cls, method_name, None)
                if not callable(original):
                    # `chat` only exists on newer vLLM releases.
                    continue
                setattr(self, f"_original_{method_name}", original)
                setattr(
                    llm_cls,
                    method_name,
                    self.create_span_wrapper(span_name=span_name, extract_attributes=extractor)(
                        original
                    ),
                )

            self._instrument_async_engines()

            llm_cls._genai_otel_vllm_instrumented = True
            self._instrumented = True
            logger.info("vLLM instrumentation enabled")
        except Exception as e:
            logger.error("Failed to instrument vLLM: %s", e)
            if config and config.fail_on_error:
                raise

    def _instrument_async_engines(self) -> None:
        """Wrap the streaming entry points.

        ``LLM.generate`` returns completed outputs, so a streaming or
        server-side deployment never goes through it -- it uses ``AsyncLLM``
        (V1) or ``AsyncLLMEngine`` (V0 shim), whose ``generate`` is an async
        generator yielding partial ``RequestOutput``s. Without these, streaming
        traffic produces no spans at all.

        Both classes are tried because which one exists depends on the vLLM
        version, and a release that exposes neither simply gets no async
        instrumentation rather than an error.
        """
        targets = []
        try:
            from vllm.v1.engine.async_llm import AsyncLLM  # type: ignore

            targets.append(AsyncLLM)
        except Exception:  # pragma: no cover - depends on vLLM version
            pass
        async_engine = getattr(self._vllm_module, "AsyncLLMEngine", None)
        if isinstance(async_engine, type):
            targets.append(async_engine)

        for cls in targets:
            if getattr(cls, "_genai_otel_vllm_instrumented", False) is True:
                continue
            original = getattr(cls, "generate", None)
            if not callable(original):
                continue
            try:
                cls.generate = self._async_generate_wrapper(original)
                cls._genai_otel_vllm_instrumented = True
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("Could not instrument %s.generate: %s", cls.__name__, e)

    def _async_generate_wrapper(self, original):
        """Trace a streamed ``AsyncLLM.generate`` including its token usage.

        The generic async-generator path in :mod:`~genai_otel.instrumentors.base`
        ends the span correctly but never records result metrics, so a streamed
        call produced a span with no tokens, no cost and no finish reason --
        verified live before this wrapper existed. vLLM makes the fix easy: each
        yielded ``RequestOutput`` is cumulative, so the final one carries the
        whole completion and can be handed to the normal metric path.

        Failures in the telemetry are swallowed and the item is always
        re-yielded. Breaking a caller's token stream to record a span would be a
        far worse outcome than losing the span.
        """
        instrumentor = self

        @wrapt.decorator
        def wrapper(wrapped, instance, args, kwargs):
            if not instrumentor._instrumented:
                return wrapped(*args, **kwargs)

            agen = wrapped(*args, **kwargs)
            try:
                attrs = instrumentor._extract_async_generate_attributes(instance, args, kwargs)
            except Exception:  # pragma: no cover - defensive
                attrs = {"gen_ai.system": PROVIDER}
            span = instrumentor.tracer.start_span("vllm.generate", attributes=attrs)
            start_time = time.time()

            async def traced():
                final = None
                try:
                    async for item in agen:
                        final = item
                        yield item
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    span.end()
                    raise
                try:
                    if final is not None:
                        instrumentor._record_result_metrics(span, [final], start_time, kwargs)
                    span.set_status(Status(StatusCode.OK))
                except Exception as e:  # pragma: no cover - defensive
                    logger.debug("Could not record streamed vLLM metrics: %s", e)
                finally:
                    span.end()

            return traced()

        return wrapper(original)

    # ------------------------------------------------------------------
    # Attribute extraction
    # ------------------------------------------------------------------

    def _base_attributes(self, instance: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Attributes common to every vLLM entry point."""
        attrs: Dict[str, Any] = {
            "gen_ai.system": PROVIDER,
            "gen_ai.request.type": "chat",
        }

        model = self._model_name(instance)
        if model:
            attrs["gen_ai.request.model"] = model

        # Sampling parameters live on a SamplingParams object rather than in
        # kwargs, so the central kwargs-based derivation in base.py cannot see
        # them.
        sampling = kwargs.get("sampling_params")
        if sampling is not None and not isinstance(sampling, (list, tuple)):
            for source, target in (
                ("max_tokens", "gen_ai.request.max_tokens"),
                ("temperature", "gen_ai.request.temperature"),
                ("top_p", "gen_ai.request.top_p"),
                ("top_k", SC.GEN_AI_REQUEST_TOP_K),
                ("seed", SC.GEN_AI_REQUEST_SEED),
                ("n", SC.GEN_AI_REQUEST_CHOICE_COUNT),
            ):
                value = getattr(sampling, source, None)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    attrs[target] = value

        return attrs

    @staticmethod
    def _model_name(instance: Any) -> Optional[str]:
        """Best-effort served-model name from an ``LLM`` instance.

        vLLM has moved this between releases (``llm_engine.model_config`` in
        older versions, an ``llm_engine.vllm_config`` wrapper in newer ones), so
        both shapes are tried before giving up.
        """
        engine = getattr(instance, "llm_engine", None)
        if engine is None:
            return None
        for path in (("model_config",), ("vllm_config", "model_config")):
            target = engine
            for step in path:
                target = getattr(target, step, None)
                if target is None:
                    break
            model = getattr(target, "model", None) if target is not None else None
            if isinstance(model, str) and model:
                return model
        return None

    def _extract_generate_attributes(self, instance, args, kwargs) -> Dict[str, Any]:
        attrs = self._base_attributes(instance, kwargs)
        attrs["gen_ai.operation.name"] = "text_completion"
        prompts = kwargs.get("prompts", args[0] if args else None)
        count = self._prompt_count(prompts)
        if count is not None:
            attrs["gen_ai.request.input_count"] = count
        return attrs

    def _extract_chat_attributes(self, instance, args, kwargs) -> Dict[str, Any]:
        attrs = self._base_attributes(instance, kwargs)
        attrs["gen_ai.operation.name"] = "chat"
        messages = kwargs.get("messages", args[0] if args else None)
        count = self._prompt_count(messages)
        if count is not None:
            attrs["gen_ai.request.input_count"] = count
        return attrs

    def _extract_encode_attributes(self, instance, args, kwargs) -> Dict[str, Any]:
        attrs = self._base_attributes(instance, kwargs)
        attrs["gen_ai.operation.name"] = "embeddings"
        # Singular: CostCalculator dispatches on this. The pricing table
        # category is the plural "embeddings" and the two deliberately differ.
        attrs["gen_ai.request.type"] = "embedding"
        prompts = kwargs.get("prompts", args[0] if args else None)
        count = self._prompt_count(prompts)
        if count is not None:
            attrs["gen_ai.request.input_count"] = count
        return attrs

    def _extract_async_generate_attributes(self, instance, args, kwargs) -> Dict[str, Any]:
        """Attributes for a streamed AsyncLLM/AsyncLLMEngine generate.

        The async engines take one prompt per call rather than a batch, and
        carry the caller's request id, which is the handle an operator uses to
        line a span up against the engine's own logs.
        """
        attrs: Dict[str, Any] = {
            "gen_ai.system": PROVIDER,
            "gen_ai.operation.name": "text_completion",
            "gen_ai.request.type": "chat",
            SC.GEN_AI_REQUEST_STREAM: True,
            "gen_ai.request.input_count": 1,
        }
        request_id = kwargs.get("request_id")
        if request_id is None and len(args) >= 3:
            request_id = args[2]
        if request_id:
            attrs[SC.GEN_AI_REQUEST_ID] = str(request_id)

        sampling = kwargs.get("sampling_params")
        if sampling is not None and not isinstance(sampling, (list, tuple)):
            for source, target in (
                ("max_tokens", "gen_ai.request.max_tokens"),
                ("temperature", "gen_ai.request.temperature"),
                ("top_p", "gen_ai.request.top_p"),
            ):
                value = getattr(sampling, source, None)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    attrs[target] = value
        return attrs

    @staticmethod
    def _prompt_count(prompts: Any) -> Optional[int]:
        """How many prompts a batch call covers.

        vLLM's defining feature is batching, so this is the difference between
        a span that says "one call" and one that says "one call, 512 prompts".
        A single string is one prompt, not its character count.
        """
        if prompts is None or isinstance(prompts, (str, bytes, dict)):
            return 1 if prompts is not None else None
        try:
            return len(prompts)
        except TypeError:
            return None

    # ------------------------------------------------------------------
    # Result handling
    # ------------------------------------------------------------------

    def _extract_usage(self, result) -> Optional[Dict[str, Any]]:
        """Sum token usage across a vLLM batch result.

        ``generate`` returns a *list* of ``RequestOutput``, one per prompt, and
        each carries its own prompt and generated token ids. Totals are summed
        across the batch so the span's token count matches what the engine
        actually processed for the call.
        """
        outputs = self._as_output_list(result)
        if not outputs:
            return None

        prompt_tokens = 0
        completion_tokens = 0
        for output in outputs:
            prompt_ids = getattr(output, "prompt_token_ids", None)
            if prompt_ids is not None:
                try:
                    prompt_tokens += len(prompt_ids)
                except TypeError:
                    pass
            for completion in getattr(output, "outputs", None) or []:
                token_ids = getattr(completion, "token_ids", None)
                if token_ids is not None:
                    try:
                        completion_tokens += len(token_ids)
                    except TypeError:
                        pass

        if not prompt_tokens and not completion_tokens:
            return None

        usage: Dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        # Prefix-cache hits. Unlike RequestMetrics this IS populated by the V1
        # engine, and it is the same concept the conventions call cache_read.
        cached = 0
        for output in outputs:
            value = getattr(output, "num_cached_tokens", None)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                cached += int(value)
        if cached:
            usage["cache_read_input_tokens"] = cached

        return usage

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        """Response attributes, including the engine latency breakdown."""
        outputs = self._as_output_list(result)
        if not outputs:
            return {}

        attrs: Dict[str, Any] = {}
        first = outputs[0]

        request_id = getattr(first, "request_id", None)
        if request_id:
            attrs[SC.GEN_AI_REQUEST_ID] = str(request_id)

        # For a batch, the first output's metrics describe one request rather
        # than the batch. Report the slowest request instead: a batch is only
        # as fast as its tail, and averaging would hide exactly the stragglers
        # this breakdown exists to find.
        #
        # `metrics` is None on the V1 engine, which is every current vLLM, so
        # this branch is skipped and no latency attributes are emitted. That is
        # deliberate: absent means "the engine did not report it", which a
        # consumer can act on, whereas a wall-clock substitute would look like
        # an engine-internal measurement and quietly mislead.
        metrics = self._slowest_metrics(outputs)
        if metrics is not None:
            apply_latency_attributes(attrs, vllm_latency_attributes(metrics))

        # The conventions define finish_reasons as an array, and a vLLM batch
        # genuinely ends different ways: one request hits EOS while another
        # hits the token cap. Reporting only the first would hide exactly the
        # requests an operator is looking for -- the truncated ones.
        reasons = self._extract_finish_reasons(outputs)
        if reasons:
            attrs[SC.GEN_AI_RESPONSE_FINISH_REASONS] = reasons

        if len(outputs) > 1:
            attrs["gen_ai.response.output_count"] = len(outputs)

        return attrs

    @staticmethod
    def _extract_finish_reasons(outputs: List[Any]) -> List[str]:
        """Every distinct finish reason in a batch, in first-seen order.

        De-duplicated because a 512-prompt batch that all hit the cap should
        report ``["length"]``, not the same string 512 times.
        """
        reasons: List[str] = []
        for output in outputs:
            for completion in getattr(output, "outputs", None) or []:
                reason = getattr(completion, "finish_reason", None)
                if reason and str(reason) not in reasons:
                    reasons.append(str(reason))
        return reasons

    @staticmethod
    def _slowest_metrics(outputs: List[Any]) -> Any:
        """The RequestMetrics of the longest-running output in a batch."""
        best = None
        best_duration = None
        for output in outputs:
            metrics = getattr(output, "metrics", None)
            if metrics is None:
                continue
            arrival = getattr(metrics, "arrival_time", None)
            finished = getattr(metrics, "finished_time", None)
            duration = None
            if isinstance(arrival, (int, float)) and isinstance(finished, (int, float)):
                duration = finished - arrival
            if best is None or (
                duration is not None and (best_duration is None or duration > best_duration)
            ):
                best, best_duration = metrics, duration
        return best

    def _extract_finish_reason(self, result) -> Optional[str]:
        outputs = self._as_output_list(result)
        for output in outputs:
            for completion in getattr(output, "outputs", None) or []:
                reason = getattr(completion, "finish_reason", None)
                if reason:
                    return str(reason)
        return None

    @staticmethod
    def _as_output_list(result: Any) -> List[Any]:
        """Normalise a vLLM result to a list of RequestOutput-like objects."""
        if result is None:
            return []
        if isinstance(result, (list, tuple)):
            return [r for r in result if r is not None]
        return [result]
