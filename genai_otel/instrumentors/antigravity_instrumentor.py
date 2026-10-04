"""Native OpenTelemetry instrumentation for the Google Antigravity SDK."""

from __future__ import annotations

import importlib
import json
import logging
import time
from typing import Any, Dict, Optional

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class AntigravityInstrumentor(BaseInstrumentor):
    """Trace Antigravity ``Agent.chat`` turns, including lazy stream consumption."""

    def __init__(self):
        super().__init__()
        self._antigravity_available = False
        self._check_availability()

    def _check_availability(self) -> None:
        try:
            module = importlib.import_module("google.antigravity")
            self._antigravity_available = callable(getattr(module, "Agent", None))
        except ImportError:
            self._antigravity_available = False
            logger.debug("google-antigravity is not installed; skipping instrumentation")

    def instrument(self, config: OTelConfig) -> None:
        if not self._antigravity_available:
            logger.debug("Skipping Antigravity instrumentation - optional SDK missing")
            return
        if self._instrumented:
            return
        self._setup_config(config)
        try:
            import wrapt

            wrapt.wrap_function_wrapper("google.antigravity.agent", "Agent.chat", self._wrap_chat)
            self._instrumented = True
            logger.info("Google Antigravity instrumentation enabled")
        except Exception as exc:  # pragma: no cover - SDK-version defensive path
            logger.error("Failed to instrument Google Antigravity: %s", exc, exc_info=True)
            if config.fail_on_error:
                raise

    @staticmethod
    def _get(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    @classmethod
    def _first(cls, value: Any, *names: str) -> Any:
        for name in names:
            result = cls._get(value, name)
            if result is not None:
                return result
        return None

    @staticmethod
    def _json(value: Any) -> str:
        try:
            return json.dumps(value, default=str, sort_keys=True)
        except Exception:  # pragma: no cover - defensive telemetry path
            return str(value)

    def _request_attributes(self, instance: Any, prompt: Any) -> Dict[str, Any]:
        cfg = self._first(instance, "_config", "config")
        model = self._first(cfg, "model", "model_name", "default_model")
        if model is None:
            model = self._first(self._first(cfg, "model_target", "endpoint"), "model", "model_name")
        attrs: Dict[str, Any] = {
            "gen_ai.system": "antigravity",
            "gen_ai.operation.name": "agent.chat",
            "gen_ai.request.type": "chat",
            "openinference.span.kind": "CHAIN",
        }
        if model is not None:
            attrs["gen_ai.request.model"] = str(model)
            attrs["llm.request.model_name"] = str(model)
        if self.config and self.config.enable_content_capture:
            maximum = self.config.content_max_length
            encoded = self._json(prompt)
            attrs["input.value"] = encoded[:maximum] if maximum > 0 else encoded
        return attrs

    @classmethod
    def _usage_attributes(cls, usage: Any) -> Dict[str, int]:
        aliases = {
            "gen_ai.usage.input_tokens": ("prompt_token_count", "input_tokens", "prompt_tokens"),
            "gen_ai.usage.output_tokens": (
                "candidates_token_count",
                "output_tokens",
                "completion_tokens",
            ),
            "gen_ai.usage.cache_read_input_tokens": (
                "cached_content_token_count",
                "cache_read_input_tokens",
            ),
            "gen_ai.usage.reasoning_tokens": ("thoughts_token_count", "reasoning_tokens"),
            "gen_ai.usage.total_tokens": ("total_token_count", "total_tokens"),
        }
        attrs: Dict[str, int] = {}
        for attribute, names in aliases.items():
            number = AntigravityInstrumentor._first(usage, *names)
            if isinstance(number, (int, float)) and not isinstance(number, bool):
                attrs[attribute] = int(number)
        output = attrs.get("gen_ai.usage.output_tokens", 0) + attrs.get(
            "gen_ai.usage.reasoning_tokens", 0
        )
        if output and "gen_ai.usage.output_tokens" in attrs:
            attrs["gen_ai.usage.output_tokens"] = output
        if "gen_ai.usage.total_tokens" not in attrs:
            total = attrs.get("gen_ai.usage.input_tokens", 0) + output
            if total:
                attrs["gen_ai.usage.total_tokens"] = total
        return attrs

    def _wrap_chat(self, wrapped, instance, args, kwargs):
        """Wrap the SDK coroutine and hold its span open through lazy iteration."""
        if not self._instrumented:
            return wrapped(*args, **kwargs)

        prompt = kwargs.get("prompt", args[0] if args else None)
        attributes = self._request_attributes(instance, prompt)
        span = self.tracer.start_span("antigravity.agent.chat", attributes=attributes)
        started = time.time()

        async def _run():
            token = otel_context.attach(trace.set_span_in_context(span))
            try:
                response = await wrapped(*args, **kwargs)
            except BaseException as exc:
                if self.error_counter:
                    self.error_counter.add(
                        1,
                        {
                            "operation": "antigravity.agent.chat",
                            "error_type": type(exc).__name__,
                        },
                    )
                span.set_attribute("error.type", type(exc).__name__)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                if isinstance(exc, Exception):
                    span.record_exception(exc)
                span.end()
                raise
            finally:
                otel_context.detach(token)

            source = getattr(response, "_chunk_stream", None)
            if source is None or not hasattr(source, "__aiter__"):
                if self.request_counter:
                    self.request_counter.add(1, {"operation": "antigravity.agent.chat"})
                self._record_result_metrics(span, response, started, kwargs)
                span.set_status(Status(StatusCode.OK))
                span.end()
                return response

            async def _traced_chunks():
                text_parts = []
                captured_length = 0
                capture = bool(self.config and self.config.enable_content_capture)
                content_limit = self.config.content_max_length if self.config else 0
                try:
                    iterator = source.__aiter__()
                    while True:
                        chunk_token = otel_context.attach(trace.set_span_in_context(span))
                        try:
                            chunk = await iterator.__anext__()
                        except StopAsyncIteration:
                            break
                        finally:
                            otel_context.detach(chunk_token)
                        if capture and chunk.__class__.__name__ == "Text":
                            chunk_text = getattr(chunk, "text", None)
                            if isinstance(chunk_text, str):
                                if content_limit > 0:
                                    remaining = content_limit - captured_length
                                    if remaining > 0:
                                        text_parts.append(chunk_text[:remaining])
                                        captured_length += min(len(chunk_text), remaining)
                                else:
                                    text_parts.append(chunk_text)
                        yield chunk

                    usage = getattr(response, "usage_metadata", None)
                    if self.request_counter:
                        self.request_counter.add(1, {"operation": "antigravity.agent.chat"})
                    for key, value in self._usage_attributes(usage).items():
                        span.set_attribute(key, value)
                    if capture and text_parts:
                        span.set_attribute("output.value", "".join(text_parts))
                    self._record_result_metrics(span, usage, started, kwargs)
                    span.set_status(Status(StatusCode.OK))
                except BaseException as exc:
                    if self.error_counter:
                        self.error_counter.add(
                            1,
                            {
                                "operation": "antigravity.agent.chat",
                                "error_type": type(exc).__name__,
                            },
                        )
                    span.set_attribute("error.type", type(exc).__name__)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    if isinstance(exc, Exception):
                        span.record_exception(exc)
                    raise
                finally:
                    span.end()

            response._chunk_stream = _traced_chunks()
            return response

        return _run()

    def _extract_usage(self, result: Any) -> Optional[Dict[str, int]]:
        attrs = self._usage_attributes(result)
        if not attrs:
            return None
        return {
            "prompt_tokens": attrs.get("gen_ai.usage.input_tokens", 0),
            "completion_tokens": attrs.get("gen_ai.usage.output_tokens", 0),
            "total_tokens": attrs.get("gen_ai.usage.total_tokens", 0),
        }


__all__ = ["AntigravityInstrumentor"]
