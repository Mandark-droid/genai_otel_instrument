"""OpenTelemetry instrumentor for the Sarvam AI SDK.

This instrumentor automatically traces calls to Sarvam AI's APIs including
chat completions, translation, transliteration, language detection,
speech-to-text, and text-to-speech. It captures relevant attributes such as
the model name, languages, token usage, and all Sarvam-specific parameters.

Sarvam AI is India's sovereign AI platform supporting 22+ Indian languages.
"""

import logging
import time
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


def _safe_kwarg(kwargs, key, default=None):
    """Safely extract a kwarg value, handling SDK OMIT/NotGiven sentinels.

    The Sarvam SDK uses OMIT sentinel objects for optional params that weren't
    passed by the user. This helper returns the default for such values.
    """
    value = kwargs.get(key, default)
    if value is None:
        return default
    class_name = getattr(type(value), "__name__", "")
    if class_name in ("NotGiven", "OMIT", "Omit"):
        return default
    return value


class SarvamAIInstrumentor(BaseInstrumentor):
    """Instrumentor for Sarvam AI SDK (sarvamai package)."""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._sarvam_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if Sarvam AI library is available."""
        try:
            import sarvamai

            self._sarvam_available = True
            logger.debug("Sarvam AI library detected and available for instrumentation")
        except ImportError:
            logger.debug("Sarvam AI library not installed, instrumentation will be skipped")
            self._sarvam_available = False

    @staticmethod
    def _normalize_sarvam_tts_model(model_name: str) -> str:
        """Normalize Sarvam TTS model names from SDK format to pricing key format.

        The SDK uses colon format (bulbul:v2) while pricing keys use hyphen (bulbul-v2).
        Only applies to TTS model names where the convention differs.
        """
        if model_name and model_name.startswith("bulbul"):
            return model_name.replace(":", "-")
        return model_name

    def _record_sarvam_cost(self, span, model: str, char_count: int, start_time: float):
        """Record latency and character-based cost for Sarvam text/audio operations.

        Sarvam's non-chat APIs (translate, TTS, STT, etc.) are priced per 1K characters.
        The pricing data is stored in the 'speech_to_text' category with promptPrice per 1K.

        Args:
            span: The OpenTelemetry span.
            model: The pricing key name (e.g., 'mayura:v1', 'bulbul-v2').
            char_count: Number of input characters.
            start_time: The time.time() when the operation started.
        """
        # Record latency
        try:
            duration = time.time() - start_time
            if self.latency_histogram:
                self.latency_histogram.record(duration, {"operation": span.name})
        except Exception as e:
            logger.debug("Failed to record latency for span '%s': %s", span.name, e)

        # Record character count as a span attribute
        span.set_attribute("gen_ai.usage.characters", char_count)

        # Calculate and record cost
        if self.config and self.config.enable_cost_tracking and char_count > 0:
            try:
                pricing_data = self.cost_calculator.pricing_data.get("speech_to_text", {})
                pricing = pricing_data.get(model)

                if pricing is None:
                    # Try case-insensitive and substring match
                    model_lower = model.lower()
                    for key in pricing_data:
                        if key.lower() == model_lower or key.lower() in model_lower:
                            pricing = pricing_data[key]
                            break

                if pricing and isinstance(pricing, dict):
                    prompt_price = pricing.get("promptPrice", 0.0)
                    total_cost = (char_count / 1000) * prompt_price

                    span.set_attribute("gen_ai.usage.cost.total", total_cost)
                    if total_cost > 0 and self.cost_counter:
                        self.cost_counter.add(total_cost, {"model": model})

                    logger.debug(
                        "Sarvam cost for %s: model=%s, chars=%d, cost=$%.6f",
                        span.name,
                        model,
                        char_count,
                        total_cost,
                    )
                else:
                    logger.debug(
                        "No pricing found for Sarvam model '%s' in speech_to_text category", model
                    )
            except Exception as e:
                logger.debug("Failed to calculate cost for span '%s': %s", span.name, e)

    def instrument(self, config: OTelConfig):
        """Instrument Sarvam AI SDK if available.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._sarvam_available:
            logger.debug("Skipping Sarvam AI instrumentation - library not available")
            return

        self.config = config

        try:
            import sarvamai

            original_init = sarvamai.SarvamAI.__init__

            def wrapped_init(instance, *args, **kwargs):
                original_init(instance, *args, **kwargs)
                self._instrument_client(instance)
                # __init__ must return None, not instance

            sarvamai.SarvamAI.__init__ = wrapped_init

            # Also instrument async client if available
            try:
                if hasattr(sarvamai, "AsyncSarvamAI") and isinstance(sarvamai.AsyncSarvamAI, type):
                    original_async_init = sarvamai.AsyncSarvamAI.__init__

                    def wrapped_async_init(instance, *args, **kwargs):
                        original_async_init(instance, *args, **kwargs)
                        self._instrument_client(instance)
                        # __init__ must return None, not instance

                    sarvamai.AsyncSarvamAI.__init__ = wrapped_async_init
                    logger.debug("Sarvam AI async client instrumentation enabled")
            except Exception as e:
                logger.debug("Sarvam AI async client instrumentation skipped: %s", e)

            self._instrumented = True
            logger.info("Sarvam AI instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument Sarvam AI: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _instrument_client(self, client):
        """Instrument Sarvam AI client methods.

        Args:
            client: The SarvamAI client instance to instrument.
        """
        # Instrument chat completions
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            original_completions = client.chat.completions
            instrumentor = self

            def wrapped_completions(*args, **kwargs):
                with instrumentor.tracer.start_as_current_span("sarvam.chat.completions") as span:
                    start_time = time.time()
                    model = kwargs.get("model", "sarvam-m")
                    span.set_attribute("gen_ai.system", "sarvam")
                    span.set_attribute("gen_ai.request.model", model)
                    span.set_attribute("gen_ai.operation.name", "chat")
                    span.set_attribute("gen_ai.request.type", "chat")

                    messages = kwargs.get("messages", [])
                    if messages:
                        try:
                            first_message = messages[0]
                            if isinstance(first_message, dict):
                                content = first_message.get("content", "")
                            else:
                                content = getattr(first_message, "content", "")
                            truncated_content = str(content)[:150]
                            request_str = str({"role": "user", "content": truncated_content})
                            span.set_attribute("gen_ai.request.first_message", request_str[:200])
                        except (IndexError, AttributeError) as e:
                            logger.debug("Failed to extract request content: %s", e)

                    if instrumentor.request_counter:
                        instrumentor.request_counter.add(1, {"model": model, "provider": "sarvam"})

                    result = original_completions(*args, **kwargs)
                    instrumentor._record_result_metrics(span, result, start_time)

                    response_attrs = instrumentor._extract_response_attributes(result)
                    for key, value in response_attrs.items():
                        span.set_attribute(key, value)

                    return result

            client.chat.completions = wrapped_completions

        # Instrument text translation
        if hasattr(client, "text") and hasattr(client.text, "translate"):
            original_translate = client.text.translate
            instrumentor = self

            def wrapped_translate(*args, **kwargs):
                with instrumentor.tracer.start_as_current_span("sarvam.text.translate") as span:
                    start_time = time.time()
                    # Translate supports mayura:v1 and sarvam-translate:v1
                    model = _safe_kwarg(kwargs, "model", "mayura:v1")
                    model = str(model)
                    span.set_attribute("gen_ai.system", "sarvam")
                    span.set_attribute("gen_ai.request.model", model)
                    span.set_attribute("gen_ai.operation.name", "translate")
                    span.set_attribute("gen_ai.request.type", "translate")

                    source_lang = _safe_kwarg(kwargs, "source_language_code", "auto")
                    target_lang = _safe_kwarg(kwargs, "target_language_code", "unknown")
                    span.set_attribute("sarvam.source_language", str(source_lang))
                    span.set_attribute("sarvam.target_language", str(target_lang))

                    input_text = kwargs.get("input", "")
                    if input_text:
                        span.set_attribute(
                            "gen_ai.request.first_message",
                            str({"role": "user", "content": str(input_text)[:150]})[:200],
                        )

                    # Capture Sarvam-specific translate params
                    mode = _safe_kwarg(kwargs, "mode")
                    if mode:
                        span.set_attribute("sarvam.translate.mode", str(mode))
                    speaker_gender = _safe_kwarg(kwargs, "speaker_gender")
                    if speaker_gender:
                        span.set_attribute("sarvam.translate.speaker_gender", str(speaker_gender))
                    numerals_format = _safe_kwarg(kwargs, "numerals_format")
                    if numerals_format:
                        span.set_attribute("sarvam.translate.numerals_format", str(numerals_format))
                    output_script = _safe_kwarg(kwargs, "output_script")
                    if output_script:
                        span.set_attribute("sarvam.translate.output_script", str(output_script))
                    enable_preprocessing = _safe_kwarg(kwargs, "enable_preprocessing")
                    if enable_preprocessing is not None:
                        span.set_attribute(
                            "sarvam.translate.enable_preprocessing", bool(enable_preprocessing)
                        )

                    if instrumentor.request_counter:
                        instrumentor.request_counter.add(
                            1, {"model": model, "operation": "translate", "provider": "sarvam"}
                        )

                    result = original_translate(*args, **kwargs)

                    if hasattr(result, "translated_text"):
                        span.set_attribute(
                            "sarvam.translated_text", str(result.translated_text)[:500]
                        )

                    # Record latency and character-based cost
                    char_count = len(str(input_text)) if input_text else 0
                    instrumentor._record_sarvam_cost(span, model, char_count, start_time)

                    return result

            client.text.translate = wrapped_translate

        # Instrument text transliteration
        if hasattr(client, "text") and hasattr(client.text, "transliterate"):
            original_transliterate = client.text.transliterate
            instrumentor = self

            def wrapped_transliterate(*args, **kwargs):
                with instrumentor.tracer.start_as_current_span("sarvam.text.transliterate") as span:
                    start_time = time.time()
                    model = "sarvam-transliterate"
                    span.set_attribute("gen_ai.system", "sarvam")
                    span.set_attribute("gen_ai.request.model", model)
                    span.set_attribute("gen_ai.operation.name", "transliterate")
                    span.set_attribute("gen_ai.request.type", "transliterate")

                    source_lang = _safe_kwarg(kwargs, "source_language_code", "auto")
                    target_lang = _safe_kwarg(kwargs, "target_language_code", "unknown")
                    span.set_attribute("sarvam.source_language", str(source_lang))
                    span.set_attribute("sarvam.target_language", str(target_lang))

                    input_text = kwargs.get("input", "")

                    # Capture Sarvam-specific transliterate params
                    numerals_format = _safe_kwarg(kwargs, "numerals_format")
                    if numerals_format:
                        span.set_attribute(
                            "sarvam.transliterate.numerals_format", str(numerals_format)
                        )
                    spoken_form = _safe_kwarg(kwargs, "spoken_form")
                    if spoken_form is not None:
                        span.set_attribute("sarvam.transliterate.spoken_form", bool(spoken_form))
                    spoken_form_numerals_language = _safe_kwarg(
                        kwargs, "spoken_form_numerals_language"
                    )
                    if spoken_form_numerals_language:
                        span.set_attribute(
                            "sarvam.transliterate.spoken_form_numerals_language",
                            str(spoken_form_numerals_language),
                        )

                    if instrumentor.request_counter:
                        instrumentor.request_counter.add(
                            1, {"model": model, "operation": "transliterate", "provider": "sarvam"}
                        )

                    result = original_transliterate(*args, **kwargs)

                    # Record latency and character-based cost
                    char_count = len(str(input_text)) if input_text else 0
                    instrumentor._record_sarvam_cost(span, model, char_count, start_time)

                    return result

            client.text.transliterate = wrapped_transliterate

        # Instrument language identification
        if hasattr(client, "text") and hasattr(client.text, "identify_language"):
            original_identify = client.text.identify_language
            instrumentor = self

            def wrapped_identify_language(*args, **kwargs):
                with instrumentor.tracer.start_as_current_span(
                    "sarvam.text.identify_language"
                ) as span:
                    start_time = time.time()
                    model = "sarvam-detect-language"
                    span.set_attribute("gen_ai.system", "sarvam")
                    span.set_attribute("gen_ai.request.model", model)
                    span.set_attribute("gen_ai.operation.name", "identify_language")
                    span.set_attribute("gen_ai.request.type", "identify_language")

                    input_text = kwargs.get("input", "")

                    if instrumentor.request_counter:
                        instrumentor.request_counter.add(
                            1,
                            {
                                "model": model,
                                "operation": "identify_language",
                                "provider": "sarvam",
                            },
                        )

                    result = original_identify(*args, **kwargs)

                    if hasattr(result, "language_code"):
                        span.set_attribute("sarvam.detected_language", result.language_code)

                    # Record latency and character-based cost
                    char_count = len(str(input_text)) if input_text else 0
                    instrumentor._record_sarvam_cost(span, model, char_count, start_time)

                    return result

            client.text.identify_language = wrapped_identify_language

        # Instrument speech-to-text transcription
        if hasattr(client, "speech_to_text") and hasattr(client.speech_to_text, "transcribe"):
            original_transcribe = client.speech_to_text.transcribe
            instrumentor = self

            def wrapped_transcribe(*args, **kwargs):
                with instrumentor.tracer.start_as_current_span(
                    "sarvam.speech_to_text.transcribe"
                ) as span:
                    start_time = time.time()
                    model = _safe_kwarg(kwargs, "model", "saarika:v2.5")
                    model = str(model)
                    span.set_attribute("gen_ai.system", "sarvam")
                    span.set_attribute("gen_ai.request.model", model)
                    span.set_attribute("gen_ai.operation.name", "speech_to_text")
                    span.set_attribute("gen_ai.request.type", "speech_to_text")

                    language_code = _safe_kwarg(kwargs, "language_code", "unknown")
                    span.set_attribute("sarvam.language_code", str(language_code))

                    # Capture STT-specific params
                    mode = _safe_kwarg(kwargs, "mode")
                    if mode:
                        span.set_attribute("sarvam.stt.mode", str(mode))
                    input_audio_codec = _safe_kwarg(kwargs, "input_audio_codec")
                    if input_audio_codec:
                        span.set_attribute("sarvam.stt.input_audio_codec", str(input_audio_codec))

                    if instrumentor.request_counter:
                        instrumentor.request_counter.add(1, {"model": model, "provider": "sarvam"})

                    result = original_transcribe(*args, **kwargs)

                    # Record transcript length and calculate cost based on transcript chars
                    transcript_chars = 0
                    if hasattr(result, "transcript"):
                        transcript_chars = len(result.transcript)
                        span.set_attribute("sarvam.transcript_length", transcript_chars)

                    instrumentor._record_sarvam_cost(span, model, transcript_chars, start_time)

                    return result

            client.speech_to_text.transcribe = wrapped_transcribe

        # Instrument speech-to-text translation
        if hasattr(client, "speech_to_text") and hasattr(client.speech_to_text, "translate"):
            original_stt_translate = client.speech_to_text.translate
            instrumentor = self

            def wrapped_stt_translate(*args, **kwargs):
                with instrumentor.tracer.start_as_current_span(
                    "sarvam.speech_to_text.translate"
                ) as span:
                    start_time = time.time()
                    model = _safe_kwarg(kwargs, "model", "saaras:v2.5")
                    model = str(model)
                    span.set_attribute("gen_ai.system", "sarvam")
                    span.set_attribute("gen_ai.request.model", model)
                    span.set_attribute("gen_ai.operation.name", "speech_to_text_translate")
                    span.set_attribute("gen_ai.request.type", "speech_to_text_translate")

                    # Capture STT translate-specific params
                    prompt = _safe_kwarg(kwargs, "prompt")
                    if prompt:
                        span.set_attribute("sarvam.stt.prompt", str(prompt)[:200])
                    input_audio_codec = _safe_kwarg(kwargs, "input_audio_codec")
                    if input_audio_codec:
                        span.set_attribute("sarvam.stt.input_audio_codec", str(input_audio_codec))

                    if instrumentor.request_counter:
                        instrumentor.request_counter.add(1, {"model": model, "provider": "sarvam"})

                    result = original_stt_translate(*args, **kwargs)

                    # Record latency (use transcript chars as proxy for cost)
                    transcript_chars = 0
                    if hasattr(result, "transcript"):
                        transcript_chars = len(result.transcript)
                        span.set_attribute("sarvam.transcript_length", transcript_chars)

                    instrumentor._record_sarvam_cost(span, model, transcript_chars, start_time)

                    return result

            client.speech_to_text.translate = wrapped_stt_translate

        # Instrument text-to-speech
        if hasattr(client, "text_to_speech") and hasattr(client.text_to_speech, "convert"):
            original_convert = client.text_to_speech.convert
            instrumentor = self

            def wrapped_convert(*args, **kwargs):
                with instrumentor.tracer.start_as_current_span(
                    "sarvam.text_to_speech.convert"
                ) as span:
                    start_time = time.time()
                    # Extract model from kwargs; SDK default is bulbul:v2 when not specified
                    raw_model = _safe_kwarg(kwargs, "model", "bulbul-v2")
                    model = instrumentor._normalize_sarvam_tts_model(str(raw_model))
                    span.set_attribute("gen_ai.system", "sarvam")
                    span.set_attribute("gen_ai.operation.name", "text_to_speech")
                    span.set_attribute("gen_ai.request.type", "text_to_speech")
                    span.set_attribute("gen_ai.request.model", model)

                    target_lang = _safe_kwarg(kwargs, "target_language_code", "unknown")
                    span.set_attribute("sarvam.target_language", str(target_lang))
                    speaker = _safe_kwarg(kwargs, "speaker", "unknown")
                    span.set_attribute("sarvam.speaker", str(speaker))

                    text = kwargs.get("text", "")
                    char_count = len(text) if text else 0
                    if text:
                        span.set_attribute("sarvam.input_text_length", char_count)

                    # Capture TTS-specific params
                    pace = _safe_kwarg(kwargs, "pace")
                    if pace is not None:
                        span.set_attribute("sarvam.tts.pace", float(pace))
                    temperature = _safe_kwarg(kwargs, "temperature")
                    if temperature is not None:
                        span.set_attribute("sarvam.tts.temperature", float(temperature))
                    pitch = _safe_kwarg(kwargs, "pitch")
                    if pitch is not None:
                        span.set_attribute("sarvam.tts.pitch", float(pitch))
                    loudness = _safe_kwarg(kwargs, "loudness")
                    if loudness is not None:
                        span.set_attribute("sarvam.tts.loudness", float(loudness))
                    speech_sample_rate = _safe_kwarg(kwargs, "speech_sample_rate")
                    if speech_sample_rate is not None:
                        span.set_attribute("sarvam.tts.speech_sample_rate", int(speech_sample_rate))
                    enable_preprocessing = _safe_kwarg(kwargs, "enable_preprocessing")
                    if enable_preprocessing is not None:
                        span.set_attribute(
                            "sarvam.tts.enable_preprocessing", bool(enable_preprocessing)
                        )
                    output_audio_codec = _safe_kwarg(kwargs, "output_audio_codec")
                    if output_audio_codec:
                        span.set_attribute("sarvam.tts.output_audio_codec", str(output_audio_codec))

                    if instrumentor.request_counter:
                        instrumentor.request_counter.add(
                            1, {"model": model, "operation": "text_to_speech", "provider": "sarvam"}
                        )

                    result = original_convert(*args, **kwargs)

                    # Record latency and character-based cost
                    instrumentor._record_sarvam_cost(span, model, char_count, start_time)

                    return result

            client.text_to_speech.convert = wrapped_convert

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from Sarvam AI response.

        Args:
            result: The API response object.

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        if hasattr(result, "usage") and result.usage:
            usage_dict = {}
            if hasattr(result.usage, "prompt_tokens"):
                usage_dict["prompt_tokens"] = result.usage.prompt_tokens
            if hasattr(result.usage, "completion_tokens"):
                usage_dict["completion_tokens"] = result.usage.completion_tokens
            if hasattr(result.usage, "total_tokens"):
                usage_dict["total_tokens"] = result.usage.total_tokens
            return usage_dict if usage_dict else None
        return None

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        """Extract response attributes from Sarvam AI response for evaluation support.

        Args:
            result: The API response object.

        Returns:
            Dict[str, Any]: Dictionary of response attributes.
        """
        attrs = {}

        try:
            # Chat completions use choices[].message.content format
            if hasattr(result, "choices") and result.choices:
                first_choice = result.choices[0]
                if hasattr(first_choice, "message") and hasattr(first_choice.message, "content"):
                    response_content = first_choice.message.content
                    if response_content:
                        attrs["gen_ai.response"] = response_content
        except (IndexError, AttributeError) as e:
            logger.debug("Failed to extract response content: %s", e)

        return attrs
