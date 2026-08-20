"""OpenTelemetry instrumentation for Liquid AI's ``liquid-audio`` runtime.

The runtime exposes ``LFM2AudioModel.generate_sequential`` and
``generate_interleaved`` as generators. This instrumentor keeps the span open
while those generators are consumed, making the first yielded token observable
as TTFT instead of ending the span at method return.
"""

import logging
import time
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class LiquidAudioInstrumentor(BaseInstrumentor):
    """Instrument Liquid AI LFM2 audio generation methods."""

    PROVIDER = "liquid_audio"

    def __init__(self):
        super().__init__()
        self._available = False
        try:
            import liquid_audio  # noqa: F401

            self._available = True
        except ImportError:
            logger.debug("Liquid Audio library not installed, skipping instrumentation")

    def instrument(self, config: OTelConfig):
        if not self._available:
            return
        self.config = config
        try:
            import wrapt
            from liquid_audio import LFM2AudioModel

            if getattr(LFM2AudioModel, "_genai_otel_liquid_audio_instrumented", False):
                return
            for method_name in ("generate_sequential", "generate_interleaved"):
                if hasattr(LFM2AudioModel, method_name):
                    wrapt.wrap_function_wrapper(
                        "liquid_audio",
                        f"LFM2AudioModel.{method_name}",
                        self._wrap_generation,
                    )
            LFM2AudioModel._genai_otel_liquid_audio_instrumented = True
            self._instrumented = True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to instrument Liquid Audio: %s", exc, exc_info=True)
            if config.fail_on_error:
                raise

    def _wrap_generation(self, wrapped, instance, args, kwargs):
        model = self._model_name(instance)
        is_asr = self._has_audio_input(kwargs)
        operation = "speech_to_text" if is_asr else "text_to_speech"
        span = self.tracer.start_span(f"liquid_audio.{operation}")
        start_time = time.time()
        try:
            span.set_attribute("gen_ai.system", self.PROVIDER)
            span.set_attribute("gen_ai.request.model", model)
            span.set_attribute("gen_ai.operation.name", operation)
            span.set_attribute("gen_ai.request.type", operation)
            span.set_attribute("gen_ai.request.streamed", True)
            if is_asr:
                audio_seconds = self._audio_seconds(kwargs)
                if audio_seconds > 0:
                    span.set_attribute("gen_ai.usage.audio_duration_seconds", audio_seconds)
            result = wrapped(*args, **kwargs)
            if hasattr(result, "__aiter__"):
                return self._wrap_async_streaming_response(result, span, start_time, model)
            if hasattr(result, "__iter__") and not isinstance(
                result, (str, bytes, dict, list, tuple)
            ):
                return self._wrap_streaming_response(result, span, start_time, model)
            span.end()
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.end()
            raise

    @staticmethod
    def _model_name(instance: Any) -> str:
        config = getattr(instance, "config", None)
        return str(
            getattr(instance, "name_or_path", None)
            or getattr(config, "_name_or_path", None)
            or "unknown"
        )

    @staticmethod
    def _has_audio_input(kwargs: Dict[str, Any]) -> bool:
        audio = kwargs.get("audio_in")
        if audio is None:
            return False
        shape = getattr(audio, "shape", None)
        return bool(shape is None or (len(shape) > 1 and int(shape[-1]) > 0))

    @staticmethod
    def _audio_seconds(kwargs: Dict[str, Any]) -> float:
        audio = kwargs.get("audio_in")
        shape = getattr(audio, "shape", None)
        if shape is None or len(shape) < 2:
            return 0.0
        return float(shape[-1]) / 16000.0

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        return None
