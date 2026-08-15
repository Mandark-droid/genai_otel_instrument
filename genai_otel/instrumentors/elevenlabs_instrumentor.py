"""ElevenLabs instrumentor for text-to-speech and Scribe speech-to-text.

ElevenLabs is billed by unit of media rather than by token: text-to-speech per
character of input text, Scribe per second of audio. Both resolve through the
``audio`` pricing category, which stores TTS rates per 1000 characters and
transcription rates per second.

Audio payloads are never captured. TTS returns an iterator of audio bytes and
Scribe consumes a caller-supplied file; the instrumentation records sizes and
durations only. For voice workloads that content is frequently personal data, so
reference-only is the sane default rather than an opt-out.
"""

import logging
import time

from ..config import OTelConfig
from ..semconv import genai_semconv_modes
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)

# SDK defaults, used when the caller does not pass model_id explicitly.
DEFAULT_TTS_MODEL = "eleven_multilingual_v2"
DEFAULT_STT_MODEL = "scribe_v1"
PROVIDER = "elevenlabs"


def _arg(kwargs, args, name, index, default=None):
    """Read a parameter that may have been passed positionally or by keyword.

    The wrapped callables are bound methods, so ``args`` excludes ``self`` and the
    positional indices line up with the SDK signature.
    """
    if name in kwargs:
        return kwargs[name]
    if len(args) > index:
        return args[index]
    return default


class ElevenLabsInstrumentor(BaseInstrumentor):
    """Instrumentor for the ElevenLabs SDK (``elevenlabs`` package)."""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._elevenlabs_available = False
        self._check_availability()

    def _check_availability(self):
        """Check whether the ElevenLabs library is importable."""
        try:
            import elevenlabs  # noqa: F401

            self._elevenlabs_available = True
            logger.debug("ElevenLabs library detected and available for instrumentation")
        except ImportError:
            logger.debug("ElevenLabs library not installed, instrumentation will be skipped")
            self._elevenlabs_available = False

    def _extract_usage(self, result):
        """ElevenLabs responses carry no token usage.

        Returning ``None`` keeps :meth:`BaseInstrumentor._wrap_streaming_response`
        from trying to read token counts off an audio chunk.
        """
        return None

    def _set_provider(self, span):
        """Set the provider attribute under the spellings the config asks for."""
        span.set_attribute("gen_ai.provider.name", PROVIDER)
        _, emit_superseded = genai_semconv_modes(
            self.config.semconv_stability_opt_in if self.config else None
        )
        if emit_superseded:
            span.set_attribute("gen_ai.system", PROVIDER)

    def _record_media_cost(self, span, model: str, usage: dict):
        """Price a call from its media usage and record the cost.

        ``usage`` is either ``{"characters": n}`` for text-to-speech or
        ``{"seconds": n}`` for transcription; ``_calculate_audio_cost`` owns both
        units. Cost is only recorded when a rate is actually found - a zero
        written for an unpriced model would read as "this was free".

        The pricing table is inconsistent about qualification: TTS models are
        keyed bare (``eleven_multilingual_v2``) while Scribe is keyed by provider
        (``elevenlabs/scribe_v1``). The SDK passes the bare ``model_id`` in both
        cases, so the provider-qualified key is tried first and the bare name
        second. Without this, ``scribe_v1`` matches nothing and every
        transcription would be priced at zero.
        """
        if not (self.config and self.config.enable_cost_tracking):
            return
        try:
            cost = self.cost_calculator.calculate_cost(f"{PROVIDER}/{model}", usage, "audio")
            if not cost:
                cost = self.cost_calculator.calculate_cost(model, usage, "audio")
            if cost and cost > 0:
                span.set_attribute("gen_ai.usage.cost.total", cost)
                if self.cost_counter:
                    self.cost_counter.add(cost, {"model": model, "provider": PROVIDER})
            else:
                logger.debug("No audio pricing found for ElevenLabs model '%s'", model)
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to calculate ElevenLabs cost for '%s': %s", model, e)

    def instrument(self, config: OTelConfig):
        """Instrument the ElevenLabs SDK if available.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._elevenlabs_available:
            logger.debug("Skipping ElevenLabs instrumentation - library not available")
            return

        self.config = config

        try:
            import elevenlabs

            # Idempotency guard: never stack wrappers if instrument() runs twice.
            if getattr(elevenlabs.ElevenLabs, "_genai_otel_elevenlabs_instrumented", False) is True:
                logger.debug("ElevenLabs already instrumented, skipping")
                self._instrumented = True
                return

            original_init = elevenlabs.ElevenLabs.__init__

            def wrapped_init(instance, *args, **kwargs):
                original_init(instance, *args, **kwargs)
                self._instrument_client(instance)
                # __init__ must return None, not instance

            elevenlabs.ElevenLabs.__init__ = wrapped_init

            try:
                if hasattr(elevenlabs, "AsyncElevenLabs") and isinstance(
                    elevenlabs.AsyncElevenLabs, type
                ):
                    original_async_init = elevenlabs.AsyncElevenLabs.__init__

                    def wrapped_async_init(instance, *args, **kwargs):
                        original_async_init(instance, *args, **kwargs)
                        self._instrument_client(instance, is_async=True)
                        # __init__ must return None, not instance

                    elevenlabs.AsyncElevenLabs.__init__ = wrapped_async_init
                    logger.debug("ElevenLabs async client instrumentation enabled")
            except Exception as e:  # noqa: BLE001
                logger.debug("ElevenLabs async client instrumentation skipped: %s", e)

            try:
                elevenlabs.ElevenLabs._genai_otel_elevenlabs_instrumented = True
            except Exception:  # noqa: BLE001
                pass
            self._instrumented = True
            logger.info("ElevenLabs instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument ElevenLabs: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _instrument_client(self, client, is_async: bool = False):
        """Wrap the text-to-speech and speech-to-text methods on a client instance."""
        tts = getattr(client, "text_to_speech", None)
        if tts is not None:
            for method_name in ("convert", "stream"):
                if hasattr(tts, method_name):
                    self._wrap_tts(tts, method_name, is_async)

        stt = getattr(client, "speech_to_text", None)
        if stt is not None and hasattr(stt, "convert"):
            self._wrap_stt(stt, is_async)

    def _tts_span_setup(self, span, args, kwargs):
        """Populate a text-to-speech span and return ``(model, character count)``."""
        model = str(_arg(kwargs, args, "model_id", 6, DEFAULT_TTS_MODEL))
        voice_id = _arg(kwargs, args, "voice_id", 0)
        text = _arg(kwargs, args, "text", 1, "") or ""
        char_count = len(text)

        self._set_provider(span)
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.operation.name", "text_to_speech")
        span.set_attribute("gen_ai.request.type", "text_to_speech")
        span.set_attribute("gen_ai.usage.characters", char_count)
        if voice_id:
            span.set_attribute("gen_ai.request.voice_id", str(voice_id))
        output_format = _arg(kwargs, args, "output_format", 4)
        if output_format:
            span.set_attribute("elevenlabs.output_format", str(output_format))

        if self.request_counter:
            self.request_counter.add(1, {"model": model, "provider": PROVIDER})

        # Characters are known before the audio arrives, so cost does not depend
        # on the caller draining the iterator.
        self._record_media_cost(span, model, {"characters": char_count})
        return model

    def _wrap_tts(self, tts, method_name: str, is_async: bool):
        """Wrap a TTS method, preserving its streaming return value.

        ``convert`` and ``stream`` both return an iterator of audio bytes. The
        iterator is wrapped rather than consumed, so time-to-first-byte reflects
        what the caller actually waits for on a voice turn.
        """
        original = getattr(tts, method_name)
        instrumentor = self
        span_name = f"elevenlabs.text_to_speech.{method_name}"

        if is_async:

            async def wrapped_async_tts(*args, **kwargs):
                span = instrumentor.tracer.start_span(span_name)
                start_time = time.time()
                try:
                    model = instrumentor._tts_span_setup(span, args, kwargs)
                    stream = original(*args, **kwargs)
                    if hasattr(stream, "__aiter__"):
                        return instrumentor._wrap_async_audio_stream(
                            stream, span, start_time, model
                        )
                    result = await stream if hasattr(stream, "__await__") else stream
                    span.end()
                    return result
                except Exception:
                    span.end()
                    raise

            setattr(tts, method_name, wrapped_async_tts)
            return

        def wrapped_tts(*args, **kwargs):
            span = instrumentor.tracer.start_span(span_name)
            start_time = time.time()
            try:
                model = instrumentor._tts_span_setup(span, args, kwargs)
                stream = original(*args, **kwargs)
            except Exception:
                span.end()
                raise
            # _wrap_streaming_response records TTFT/TBT and ends the span when the
            # iterator is drained or raises.
            return instrumentor._wrap_streaming_response(stream, span, start_time, model)

        setattr(tts, method_name, wrapped_tts)

    async def _wrap_async_audio_stream(self, stream, span, start_time: float, model: str):
        """Async counterpart of ``_wrap_streaming_response`` for audio byte streams."""
        from opentelemetry.trace import Status, StatusCode

        first = True
        chunks = 0
        try:
            async for chunk in stream:
                if first:
                    # Time to the first audio byte. TPOT has no meaning here --
                    # a TTS stream has no output tokens to divide by - so it is
                    # left off rather than approximated from chunk counts.
                    self._record_time_to_first_token(span, time.time() - start_time, model)
                    first = False
                chunks += 1
                yield chunk
            span.set_attribute("gen_ai.streaming.chunk_count", chunks)
            if self.latency_histogram:
                self.latency_histogram.record(time.time() - start_time, {"operation": span.name})
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            span.end()

    def _stt_span_setup(self, span, args, kwargs):
        """Populate a speech-to-text span and return the model name."""
        model = str(_arg(kwargs, args, "model_id", 0, DEFAULT_STT_MODEL))
        self._set_provider(span)
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.operation.name", "speech_to_text")
        span.set_attribute("gen_ai.request.type", "speech_to_text")
        language_code = _arg(kwargs, args, "language_code", 5)
        if language_code:
            span.set_attribute("gen_ai.request.language_code", str(language_code))
        if self.request_counter:
            self.request_counter.add(1, {"model": model, "provider": PROVIDER})
        return model

    def _stt_record_result(self, span, model: str, result, start_time: float):
        """Record transcription duration, transcript size and cost."""
        duration = time.time() - start_time
        if self.latency_histogram:
            self.latency_histogram.record(duration, {"operation": span.name})

        audio_seconds = getattr(result, "audio_duration_secs", None)
        if isinstance(audio_seconds, (int, float)) and audio_seconds > 0:
            span.set_attribute("gen_ai.usage.audio_duration_seconds", float(audio_seconds))
            # Scribe is billed per second of audio, not per character of transcript.
            self._record_media_cost(span, model, {"seconds": float(audio_seconds)})
        else:
            logger.debug(
                "ElevenLabs transcription returned no audio_duration_secs; cost not recorded"
            )

        text = getattr(result, "text", None)
        if isinstance(text, str):
            span.set_attribute("gen_ai.response.transcript_length", len(text))
        language = getattr(result, "language_code", None)
        if language:
            span.set_attribute("gen_ai.response.language_code", str(language))

        # start_as_current_span records a raised exception as ERROR but never sets
        # OK, so without this the span exports UNSET and reads as "never finished".
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.OK))

    def _wrap_stt(self, stt, is_async: bool):
        """Wrap ``speech_to_text.convert`` on a client instance."""
        original = stt.convert
        instrumentor = self

        if is_async:

            async def wrapped_async_stt(*args, **kwargs):
                with instrumentor.tracer.start_as_current_span(
                    "elevenlabs.speech_to_text.convert"
                ) as span:
                    start_time = time.time()
                    model = instrumentor._stt_span_setup(span, args, kwargs)
                    result = await original(*args, **kwargs)
                    instrumentor._stt_record_result(span, model, result, start_time)
                    return result

            stt.convert = wrapped_async_stt
            return

        def wrapped_stt(*args, **kwargs):
            with instrumentor.tracer.start_as_current_span(
                "elevenlabs.speech_to_text.convert"
            ) as span:
                start_time = time.time()
                model = instrumentor._stt_span_setup(span, args, kwargs)
                result = original(*args, **kwargs)
                instrumentor._stt_record_result(span, model, result, start_time)
                return result

        stt.convert = wrapped_stt
