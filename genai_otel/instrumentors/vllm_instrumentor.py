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
from typing import Any, Dict, List, Optional

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

            llm_cls._genai_otel_vllm_instrumented = True
            self._instrumented = True
            logger.info("vLLM instrumentation enabled")
        except Exception as e:
            logger.error("Failed to instrument vLLM: %s", e)
            if config and config.fail_on_error:
                raise

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

        if len(outputs) > 1:
            attrs["gen_ai.response.output_count"] = len(outputs)

        return attrs

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
