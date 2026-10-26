"""OpenTelemetry support for the official xAI Python SDK (``xai-sdk``)."""

import inspect
import logging
import time
from urllib.parse import urlparse

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from ..config import OTelConfig
from .base import BaseInstrumentor, content_capture_on

logger = logging.getLogger(__name__)


class XAIInstrumentor(BaseInstrumentor):
    """Trace chat sampling and streaming through xAI's sync and async clients."""

    def __init__(self):
        super().__init__()
        try:
            import xai_sdk  # noqa: F401

            self._available = True
        except ImportError:
            self._available = False

    def instrument(self, config: OTelConfig):
        if not self._available:
            logger.debug("Skipping xAI instrumentation: xai-sdk is not installed")
            return
        self._setup_config(config)
        try:
            import xai_sdk

            for client_name in ("Client", "AsyncClient"):
                client_type = getattr(xai_sdk, client_name, None)
                if client_type is None or getattr(client_type, "_genai_otel_xai_wrapped", False):
                    continue
                original_init = client_type.__init__
                instrumentor = self

                def wrapped_init(instance, *args, _init=original_init, **kwargs):
                    _init(instance, *args, **kwargs)
                    instrumentor._wrap_chat_create(instance)
                    instrumentor._wrap_media_apis(instance)

                client_type.__init__ = wrapped_init
                client_type._genai_otel_xai_wrapped = True
            self._instrument_voice_http()
            self._instrumented = True
        except Exception as exc:  # instrumentation must not break SDK setup
            logger.error("Failed to instrument xai-sdk: %s", exc, exc_info=True)
            if config.fail_on_error:
                raise

    def _wrap_chat_create(self, client):
        chat_api = getattr(client, "chat", None)
        create = getattr(chat_api, "create", None)
        if not callable(create) or getattr(create, "_genai_otel_xai_wrapped", False):
            return

        instrumentor = self

        def wrapped_create(*args, **kwargs):
            chat = create(*args, **kwargs)
            model = kwargs.get("model", "unknown")
            instrumentor._wrap_chat(chat, model)
            return chat

        wrapped_create._genai_otel_xai_wrapped = True
        chat_api.create = wrapped_create

    def _wrap_chat(self, chat, model):
        for method_name in ("sample", "stream"):
            original = getattr(chat, method_name, None)
            if not callable(original) or getattr(original, "_genai_otel_xai_wrapped", False):
                continue

            instrumentor = self
            if method_name == "sample":
                if inspect.iscoroutinefunction(original):

                    async def wrapped(*args, _original=original, **kwargs):
                        start = time.time()
                        span = instrumentor._start_span("xai.chat", model, "chat")
                        token = otel_context.attach(trace.set_span_in_context(span))
                        try:
                            result = await _original(*args, **kwargs)
                            instrumentor._finish_span(span, result, start)
                            return result
                        except Exception as exc:
                            instrumentor._fail_span(span, exc)
                            raise
                        finally:
                            otel_context.detach(token)

                else:

                    def wrapped(*args, _original=original, **kwargs):
                        start = time.time()
                        span = instrumentor._start_span("xai.chat", model, "chat")
                        token = otel_context.attach(trace.set_span_in_context(span))
                        try:
                            result = _original(*args, **kwargs)
                            instrumentor._finish_span(span, result, start)
                            return result
                        except Exception as exc:
                            instrumentor._fail_span(span, exc)
                            raise
                        finally:
                            otel_context.detach(token)

            elif inspect.isasyncgenfunction(original):

                async def wrapped(*args, _original=original, **kwargs):
                    start = time.time()
                    span = instrumentor._start_span("xai.chat", model, "chat")
                    token = otel_context.attach(trace.set_span_in_context(span))
                    result = None
                    try:
                        async for item in _original(*args, **kwargs):
                            if isinstance(item, tuple) and item:
                                result = item[0]
                            yield item
                        instrumentor._finish_span(span, result, start)
                    except Exception as exc:
                        instrumentor._fail_span(span, exc)
                        raise
                    finally:
                        if span.is_recording():
                            span.end()
                        otel_context.detach(token)

            else:

                def wrapped(*args, _original=original, **kwargs):
                    start = time.time()
                    span = instrumentor._start_span("xai.chat", model, "chat")
                    token = otel_context.attach(trace.set_span_in_context(span))
                    result = None
                    try:
                        for item in _original(*args, **kwargs):
                            if isinstance(item, tuple) and item:
                                result = item[0]
                            yield item
                        instrumentor._finish_span(span, result, start)
                    except Exception as exc:
                        instrumentor._fail_span(span, exc)
                        raise
                    finally:
                        if span.is_recording():
                            span.end()
                        otel_context.detach(token)

            wrapped._genai_otel_xai_wrapped = True
            try:
                setattr(chat, method_name, wrapped)
            except (AttributeError, TypeError):
                logger.debug("xAI Chat object does not allow wrapping %s", method_name)

    def _wrap_media_apis(self, client):
        """Wrap the SDK's image and video generation methods when present."""
        for namespace_name, methods in (
            ("image", ("sample", "sample_batch")),
            ("video", ("generate", "extend")),
            ("voice", ("speak", "transcribe")),
        ):
            namespace = getattr(client, namespace_name, None)
            if namespace is None:
                continue
            for method_name in methods:
                original = getattr(namespace, method_name, None)
                if not callable(original) or getattr(original, "_genai_otel_xai_wrapped", False):
                    continue
                instrumentor = self
                operation = {
                    "image": "image",
                    "video": "video",
                    "voice": "audio" if method_name == "speak" else "speech_to_text",
                }[namespace_name]
                default_model = {
                    "audio": "xai-tts",
                    "speech_to_text": "xai-stt",
                }.get(operation, "unknown")
                span_name = "xai.%s.%s" % (namespace_name, method_name)
                if inspect.iscoroutinefunction(original):

                    async def wrapped(
                        *args,
                        _original=original,
                        _operation=operation,
                        _span_name=span_name,
                        _default_model=default_model,
                        **kwargs,
                    ):
                        start = time.time()
                        span = instrumentor._start_span(
                            _span_name, kwargs.get("model", _default_model), _operation
                        )
                        token = otel_context.attach(trace.set_span_in_context(span))
                        try:
                            response = await _original(*args, **kwargs)
                            instrumentor._finish_span(
                                span,
                                response,
                                start,
                                _operation,
                                instrumentor._voice_usage(_operation, kwargs, response),
                            )
                            return response
                        except Exception as exc:
                            instrumentor._fail_span(span, exc)
                            raise
                        finally:
                            otel_context.detach(token)

                else:

                    def wrapped(
                        *args,
                        _original=original,
                        _operation=operation,
                        _span_name=span_name,
                        _default_model=default_model,
                        **kwargs,
                    ):
                        start = time.time()
                        span = instrumentor._start_span(
                            _span_name, kwargs.get("model", _default_model), _operation
                        )
                        token = otel_context.attach(trace.set_span_in_context(span))
                        try:
                            response = _original(*args, **kwargs)
                            instrumentor._finish_span(
                                span,
                                response,
                                start,
                                _operation,
                                instrumentor._voice_usage(_operation, kwargs, response),
                            )
                            return response
                        except Exception as exc:
                            instrumentor._fail_span(span, exc)
                            raise
                        finally:
                            otel_context.detach(token)

                wrapped._genai_otel_xai_wrapped = True
                try:
                    setattr(namespace, method_name, wrapped)
                except (AttributeError, TypeError):
                    logger.debug(
                        "xAI %s API does not allow wrapping %s", namespace_name, method_name
                    )

    def _instrument_voice_http(self):
        """Instrument xAI's voice REST endpoints for Python SDK versions without voice helpers."""
        try:
            import requests
        except ImportError:
            return
        session_type = requests.sessions.Session
        if getattr(session_type, "_genai_otel_xai_voice_wrapped", False):
            return
        original = session_type.request
        instrumentor = self

        def wrapped_request(session, method, url, *args, **kwargs):
            parsed = urlparse(str(url))
            path = parsed.path.rstrip("/")
            is_xai_api = parsed.hostname == "api.x.ai" or bool(
                parsed.hostname and parsed.hostname.endswith(".api.x.ai")
            )
            if not is_xai_api or path not in {"/v1/tts", "/v1/stt"}:
                return original(session, method, url, *args, **kwargs)

            is_tts = path.endswith("/tts")
            operation = "audio" if is_tts else "speech_to_text"
            model = "xai-tts" if is_tts else "xai-stt"
            span = instrumentor._start_span("xai.voice.%s" % operation, model, operation)
            token = otel_context.attach(trace.set_span_in_context(span))
            started = time.time()
            payload = kwargs.get("json") or {}
            usage = None
            if is_tts and isinstance(payload, dict) and isinstance(payload.get("text"), str):
                usage = {"characters": len(payload["text"])}
            try:
                response = original(session, method, url, *args, **kwargs)
                if not is_tts:
                    try:
                        metadata = response.json()
                    except (ValueError, AttributeError):
                        metadata = {}
                    if isinstance(metadata, dict):
                        seconds = next(
                            (
                                metadata.get(key)
                                for key in ("audio_duration", "duration_seconds", "duration")
                                if isinstance(metadata.get(key), (int, float))
                            ),
                            None,
                        )
                        if seconds is not None and seconds >= 0:
                            usage = {"seconds": seconds}
                if getattr(response, "status_code", 200) >= 400:
                    span.set_status(Status(StatusCode.ERROR, "HTTP %s" % response.status_code))
                    span.end()
                else:
                    instrumentor._finish_span(
                        span, response, started, operation, usage_override=usage
                    )
                return response
            except Exception as exc:
                instrumentor._fail_span(span, exc)
                raise
            finally:
                otel_context.detach(token)

        session_type.request = wrapped_request
        session_type._genai_otel_xai_voice_wrapped = True

    @staticmethod
    def _voice_usage(operation, request, response):
        """Keep billable voice quantities without recording user audio or text."""
        if operation == "audio":
            text = request.get("text")
            return {"characters": len(text)} if isinstance(text, str) else None
        if operation == "speech_to_text":
            for candidate in (response, getattr(response, "usage", None)):
                if candidate is None:
                    continue
                for key in ("seconds", "duration_seconds", "audio_duration"):
                    seconds = getattr(candidate, key, None)
                    if isinstance(seconds, (int, float)) and seconds >= 0:
                        return {"seconds": seconds}
        return None

    def _start_span(self, name, model, operation):
        span = self.tracer.start_span(name)
        span.set_attribute("gen_ai.system", "xai")
        span.set_attribute("gen_ai.operation.name", operation)
        span.set_attribute("gen_ai.request.type", operation)
        span.set_attribute("gen_ai.request.model", str(model))
        return span

    def _extract_usage(self, response):
        """Normalize xAI protobuf usage fields for the shared token/cost pipeline."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        fields = {
            "prompt_tokens": ("prompt_tokens", "input_tokens"),
            "completion_tokens": ("completion_tokens", "output_tokens"),
            "total_tokens": ("total_tokens",),
            "reasoning_tokens": ("reasoning_tokens",),
            "cost_in_usd_ticks": ("cost_in_usd_ticks",),
        }
        result = {}
        for target, candidates in fields.items():
            for name in candidates:
                value = getattr(usage, name, None)
                if isinstance(value, (int, float)) and value >= 0:
                    result[target] = value
                    break
        return result or None

    @staticmethod
    def _reported_cost(response):
        """Read xAI's convenience USD cost or its integer tick representation."""
        cost = getattr(response, "cost_usd", None)
        if isinstance(cost, (int, float)) and cost >= 0:
            return float(cost)
        usage = getattr(response, "usage", None)
        ticks = getattr(usage, "cost_in_usd_ticks", None) if usage is not None else None
        if isinstance(ticks, (int, float)) and ticks >= 0:
            return float(ticks) / 10_000_000_000
        return None

    def _finish_span(self, span, response, start, operation="chat", usage_override=None):
        usage = self._extract_usage(response) or {}
        for source, target in (
            ("prompt_tokens", "gen_ai.usage.input_tokens"),
            ("completion_tokens", "gen_ai.usage.output_tokens"),
            ("total_tokens", "gen_ai.usage.total_tokens"),
            ("reasoning_tokens", "gen_ai.usage.reasoning_tokens"),
        ):
            value = usage.get(source)
            if isinstance(value, (int, float)):
                span.set_attribute(target, value)
        cost = self._reported_cost(response)
        content = getattr(response, "content", None) if operation == "chat" else None
        if content and content_capture_on(self.config):
            limit = getattr(self.config, "content_max_length", 0) or 0
            content = str(content)
            span.set_attribute("gen_ai.response", content[:limit] if limit > 0 else content)
        self._record_result_metrics(
            span,
            response,
            start,
            reported_cost=cost,
            usage_override=usage_override,
        )
        span.end()

    @staticmethod
    def _fail_span(span, exc):
        span.set_status(Status(StatusCode.ERROR, str(exc)))
        span.record_exception(exc)
        span.end()
