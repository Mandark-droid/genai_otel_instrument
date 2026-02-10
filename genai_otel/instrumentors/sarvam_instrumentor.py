"""OpenTelemetry instrumentor for the Sarvam AI SDK.

This instrumentor automatically traces calls to Sarvam AI's APIs including
chat completions, translation, transliteration, language detection,
speech-to-text, and text-to-speech. It captures relevant attributes such as
the model name, languages, and token usage.

Sarvam AI is India's sovereign AI platform supporting 22+ Indian languages.
"""

import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


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
                return instance

            sarvamai.SarvamAI.__init__ = wrapped_init

            # Also instrument async client if available
            try:
                if hasattr(sarvamai, "AsyncSarvamAI") and isinstance(sarvamai.AsyncSarvamAI, type):
                    original_async_init = sarvamai.AsyncSarvamAI.__init__

                    def wrapped_async_init(instance, *args, **kwargs):
                        original_async_init(instance, *args, **kwargs)
                        self._instrument_client(instance)
                        return instance

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
                    instrumentor._record_result_metrics(span, result, 0)

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
                    span.set_attribute("gen_ai.system", "sarvam")
                    span.set_attribute("gen_ai.operation.name", "translate")
                    span.set_attribute("gen_ai.request.type", "translate")

                    source_lang = kwargs.get("source_language_code", "auto")
                    target_lang = kwargs.get("target_language_code", "unknown")
                    span.set_attribute("sarvam.source_language", source_lang)
                    span.set_attribute("sarvam.target_language", target_lang)

                    input_text = kwargs.get("input", "")
                    if input_text:
                        span.set_attribute(
                            "gen_ai.request.first_message",
                            str({"role": "user", "content": str(input_text)[:150]})[:200],
                        )

                    if instrumentor.request_counter:
                        instrumentor.request_counter.add(
                            1, {"operation": "translate", "provider": "sarvam"}
                        )

                    result = original_translate(*args, **kwargs)

                    if hasattr(result, "translated_text"):
                        span.set_attribute(
                            "sarvam.translated_text", str(result.translated_text)[:500]
                        )

                    return result

            client.text.translate = wrapped_translate

        # Instrument text transliteration
        if hasattr(client, "text") and hasattr(client.text, "transliterate"):
            original_transliterate = client.text.transliterate
            instrumentor = self

            def wrapped_transliterate(*args, **kwargs):
                with instrumentor.tracer.start_as_current_span("sarvam.text.transliterate") as span:
                    span.set_attribute("gen_ai.system", "sarvam")
                    span.set_attribute("gen_ai.operation.name", "transliterate")

                    source_lang = kwargs.get("source_language_code", "auto")
                    target_lang = kwargs.get("target_language_code", "unknown")
                    span.set_attribute("sarvam.source_language", source_lang)
                    span.set_attribute("sarvam.target_language", target_lang)

                    if instrumentor.request_counter:
                        instrumentor.request_counter.add(
                            1, {"operation": "transliterate", "provider": "sarvam"}
                        )

                    result = original_transliterate(*args, **kwargs)
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
                    span.set_attribute("gen_ai.system", "sarvam")
                    span.set_attribute("gen_ai.operation.name", "identify_language")

                    if instrumentor.request_counter:
                        instrumentor.request_counter.add(
                            1, {"operation": "identify_language", "provider": "sarvam"}
                        )

                    result = original_identify(*args, **kwargs)

                    if hasattr(result, "language_code"):
                        span.set_attribute("sarvam.detected_language", result.language_code)

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
                    model = kwargs.get("model", "saarika:v2.5")
                    span.set_attribute("gen_ai.system", "sarvam")
                    span.set_attribute("gen_ai.request.model", model)
                    span.set_attribute("gen_ai.operation.name", "speech_to_text")
                    span.set_attribute("gen_ai.request.type", "speech_to_text")

                    language_code = kwargs.get("language_code", "unknown")
                    span.set_attribute("sarvam.language_code", language_code)

                    if instrumentor.request_counter:
                        instrumentor.request_counter.add(1, {"model": model, "provider": "sarvam"})

                    result = original_transcribe(*args, **kwargs)

                    if hasattr(result, "transcript"):
                        span.set_attribute("sarvam.transcript_length", len(result.transcript))

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
                    model = kwargs.get("model", "saaras:v2.5")
                    span.set_attribute("gen_ai.system", "sarvam")
                    span.set_attribute("gen_ai.request.model", model)
                    span.set_attribute("gen_ai.operation.name", "speech_to_text_translate")
                    span.set_attribute("gen_ai.request.type", "speech_to_text_translate")

                    if instrumentor.request_counter:
                        instrumentor.request_counter.add(1, {"model": model, "provider": "sarvam"})

                    result = original_stt_translate(*args, **kwargs)
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
                    span.set_attribute("gen_ai.system", "sarvam")
                    span.set_attribute("gen_ai.operation.name", "text_to_speech")
                    span.set_attribute("gen_ai.request.type", "text_to_speech")
                    span.set_attribute("gen_ai.request.model", "bulbul")

                    target_lang = kwargs.get("target_language_code", "unknown")
                    speaker = kwargs.get("speaker", "unknown")
                    span.set_attribute("sarvam.target_language", target_lang)
                    span.set_attribute("sarvam.speaker", speaker)

                    text = kwargs.get("text", "")
                    if text:
                        span.set_attribute("sarvam.input_text_length", len(text))

                    if instrumentor.request_counter:
                        instrumentor.request_counter.add(
                            1, {"operation": "text_to_speech", "provider": "sarvam"}
                        )

                    result = original_convert(*args, **kwargs)
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
