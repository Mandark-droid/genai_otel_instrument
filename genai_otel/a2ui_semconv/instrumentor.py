"""A2UI (Agent-to-UI) instrumentation for the official agent SDK, ``a2ui-agent-sdk`` 0.11+.

This works whatever carries the UI: A2A 0.3 (through ``a2ui-a2a``), A2A 1.x, Google ADK,
MCP or something of the application's own. The A2A 1.x enrichment of an A2A span lives
with the A2A instrumentor and shares :mod:`genai_otel.a2ui_semconv.extract`.

What it records
---------------
* ``A2uiRequestProcessor.parse_response`` opens an INTERNAL span ``a2ui.parse_response``.
  When parsing or validation raises, the span is ERROR with ``error.type`` set to the
  exception class and ``a2ui.validation.outcome=invalid``: the model wrote UI the agent
  SDK rejected, and the user sees nothing. Under an LLM span it is that span's child.
* ``A2uiRequestProcessor.open_stream`` opens ``a2ui.parse_stream``. It ends when the
  stream does (``close()`` returns, or ``feed``/``close`` raises) and carries the totals.
  A stream opened with ``after_rejected_block="continue"`` reads past a rejected block;
  the span is ERROR then too, because a block failed validation.
* ``RendererToAgentMessageProcessor.process_message`` opens ``a2ui.process {type}``
  (``a2ui.process action``) and records the renderer message, the number of replies and
  the codes of renderer errors and function errors.
* ``a2ui.a2a.parts.parse_content_to_parts`` catches every exception and logs only its
  class. Our wrapper notices that log line and records ``a2ui.validation.outcome=invalid``
  on the current span. Its return value and its logging are untouched. That module needs
  ``a2ui-a2a`` (and so ``a2a-sdk`` 0.3), and is wrapped only when it can be imported.

It does nothing when ``a2ui-agent-sdk`` is missing or older than 0.11.
"""

import contextlib
import contextvars
import importlib
import importlib.util
import inspect
import logging
import weakref
from typing import Any, Callable, Dict, List, Optional, Tuple

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from ..config import OTelConfig
from ..instrumentors.base import BaseInstrumentor
from .attributes import (
    CONTENT_KEYS,
    SIDE_AGENT,
    SIDE_RENDERER,
    VALIDATION_INVALID,
    VALIDATION_VALID,
    A2UIAttributes,
)
from .extract import RENDERER_KINDS, A2uiObservation, apply_to_span, message_to_dict, snake

logger = logging.getLogger(__name__)

SDK_DISTRIBUTION = "a2ui-agent-sdk"
MIN_SDK_VERSION = (0, 11)
_INSTRUMENTATION_SCOPE = "genai_otel.a2ui"
_MARKER = "_genai_otel_a2ui_instrumented"
_STREAM_MARKER = "_genai_otel_a2ui_stream"

PARTS_MODULE = "a2ui.a2a.parts"
PARTS_LOGGER = "a2ui.a2a.parts"
SWALLOWED_PREFIX = "Failed to parse A2UI"
SWALLOWED_EVENT = "a2ui.validation_failed"

# While ``parse_content_to_parts`` runs under our wrapper, the class names it logged.
_swallowed: "contextvars.ContextVar[Optional[List[str]]]" = contextvars.ContextVar(
    "genai_otel_a2ui_swallowed", default=None
)


def _sdk_version() -> Optional[Tuple[int, ...]]:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            text = version(SDK_DISTRIBUTION)
        except PackageNotFoundError:
            return None
    except Exception:  # noqa: BLE001  # pragma: no cover - importlib.metadata always exists
        return None
    pieces: List[int] = []
    for piece in text.split(".")[:2]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            return None
        pieces.append(int(digits))
    return tuple(pieces)


def sdk_supported() -> bool:
    """True when ``a2ui-agent-sdk`` 0.11 or later is installed. Does not import it."""
    try:
        if importlib.util.find_spec("a2ui") is None:
            return False
        if importlib.util.find_spec("a2ui.processor") is None:
            return False
    except (ImportError, ValueError):
        return False
    installed = _sdk_version()
    if installed is None:
        # Installed without metadata (vendored, a zipapp): go by what it offers.
        return importlib.util.find_spec("a2ui.rpc") is not None
    return installed >= MIN_SDK_VERSION


def _enabled(config: Optional[OTelConfig]) -> bool:
    return bool(getattr(config, "a2ui_enabled", True)) if config is not None else True


class _SwallowedFilter(logging.Filter):
    """Notes the warning ``parse_content_to_parts`` logs for a failure it swallowed."""

    def filter(self, record: logging.LogRecord) -> bool:
        holder = _swallowed.get()
        if holder is not None and record.levelno >= logging.WARNING:
            try:
                if str(record.msg).startswith(SWALLOWED_PREFIX):
                    args = record.args
                    holder.append(str(args[0]) if isinstance(args, tuple) and args else "Error")
            except Exception:  # noqa: BLE001
                pass
        return True  # never suppresses: the logging is not ours to change


class _Scope:
    """One A2UI span, what it saw, and how it ends."""

    def __init__(self, instrumentor: "A2UIInstrumentor", span: Any, owned_end: bool = True) -> None:
        self.instrumentor = instrumentor
        self.span = span
        self.obs = A2uiObservation(instrumentor._capture())
        self.error: Optional[BaseException] = None
        self._owned_end = owned_end
        self._finished = False

    def fail(self, exc: BaseException) -> None:
        # A cancelled task or an abandoned stream is not a failed parse.
        if isinstance(exc, Exception) and self.error is None:
            self.error = exc

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            span = self.span
            apply_to_span(span, self.obs, self.instrumentor._put_content)
            if self.error is not None:
                etype = type(self.error).__name__
                span.set_attribute(A2UIAttributes.ERROR_TYPE, etype)
                if self.instrumentor._capture():
                    # A validation error can quote the model's output or user data, so the
                    # text is kept only with content capture on.
                    span.set_status(Status(StatusCode.ERROR, str(self.error)[:200]))
                    span.record_exception(self.error)
                else:
                    span.set_status(Status(StatusCode.ERROR, etype))
                    span.add_event("exception", {"exception.type": etype})
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - never break the call
            logger.debug("Could not finish an A2UI span: %s", exc)
        finally:
            if self._owned_end:
                with contextlib.suppress(Exception):
                    self.span.end()


class A2UIInstrumentor(BaseInstrumentor):
    """Instrumentor for the A2UI agent SDK (``a2ui-agent-sdk`` 0.11+)."""

    CONTENT_ATTRIBUTES = CONTENT_KEYS

    def __init__(self) -> None:
        super().__init__()
        self.tracer = trace.get_tracer(_INSTRUMENTATION_SCOPE)
        self._patched: List[Tuple[Any, str, Any]] = []
        self._filter: Optional[_SwallowedFilter] = None

    @staticmethod
    def available() -> bool:
        """True when a supported ``a2ui-agent-sdk`` is installed."""
        return sdk_supported()

    # -- instrument / uninstrument -------------------------------------------------

    def instrument(self, config: OTelConfig) -> None:
        """Wrap the agent SDK's processors. Does nothing without ``a2ui-agent-sdk`` 0.11+."""
        self.config = config
        if not _enabled(config):
            logger.debug("GENAI_A2UI_ENABLED is false; A2UI instrumentation skipped")
            return
        if not sdk_supported():
            logger.debug("a2ui-agent-sdk 0.11+ is not installed; A2UI instrumentation skipped")
            return
        try:
            wrapped = self._wrap_request_processor() + self._wrap_rpc_processor()
            wrapped += self._wrap_parts_module()
            self._instrumented = wrapped > 0
            if wrapped:
                logger.info("A2UI instrumentation enabled")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to instrument a2ui-agent-sdk: %s", exc, exc_info=True)
            if getattr(config, "fail_on_error", False):
                raise

    def uninstrument(self) -> None:
        """Restore every method we wrapped."""
        for owner, name, original in reversed(self._patched):
            try:
                setattr(owner, name, original)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not restore %s.%s: %s", owner, name, exc)
        for owner in {id(o): o for o, _, _ in self._patched}.values():
            with contextlib.suppress(Exception):
                if _MARKER in vars(owner):
                    delattr(owner, _MARKER)
        self._patched = []
        if self._filter is not None:
            with contextlib.suppress(Exception):
                logging.getLogger(PARTS_LOGGER).removeFilter(self._filter)
            self._filter = None
        self._instrumented = False

    def _patch(self, owner: Any, name: str, replacement: Any) -> None:
        original = vars(owner)[name] if not inspect.ismodule(owner) else getattr(owner, name)
        self._patched.append((owner, name, original))
        setattr(owner, name, replacement)

    def _capture(self) -> bool:
        cfg = self.config
        return cfg is None or bool(getattr(cfg, "enable_content_capture", False))

    def _put_content(self, span: Any, key: str, value: Any) -> None:
        """Write a content attribute, only when capture is on, capped to the length limit."""
        if not self._capture():
            return
        limit = getattr(self.config, "content_max_length", 0) if self.config else 0
        limit = limit if isinstance(limit, int) and limit > 0 else 0
        try:
            if isinstance(value, (list, tuple)):
                span.set_attribute(key, [str(v)[:limit] if limit else str(v) for v in value])
            else:
                text = str(value)
                span.set_attribute(key, text[:limit] if limit else text)
        except Exception as exc:  # noqa: BLE001 - capture must never fail the call
            logger.debug("Could not set content attribute %s: %s", key, exc)

    # -- A2uiRequestProcessor ------------------------------------------------------

    def _wrap_request_processor(self) -> int:
        try:
            processor_class = importlib.import_module("a2ui.processor").A2uiRequestProcessor
        except Exception as exc:  # noqa: BLE001
            logger.debug("a2ui request processor not instrumented: %s", exc)
            return 0
        if vars(processor_class).get(_MARKER):
            return 0
        count = 0
        original = vars(processor_class).get("parse_response")
        if original is not None:
            self._patch(processor_class, "parse_response", self._parse_response_wrapper(original))
            count += 1
        original = vars(processor_class).get("open_stream")
        if original is not None:
            self._patch(processor_class, "open_stream", self._open_stream_wrapper(original))
            count += 1
        if count:
            setattr(processor_class, _MARKER, True)
        return count

    def _parse_response_wrapper(self, original: Callable[..., Any]) -> Any:
        instrumentor = self

        def wrapper(processor: Any, *args: Any, **kwargs: Any) -> Any:
            if not instrumentor._instrumented:
                return original(processor, *args, **kwargs)
            with instrumentor.tracer.start_as_current_span(
                "a2ui.parse_response",
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                scope = _Scope(instrumentor, span, owned_end=False)
                try:
                    result = original(processor, *args, **kwargs)
                except BaseException as exc:
                    scope.obs.set_outcome(VALIDATION_INVALID)
                    scope.fail(exc)
                    scope.finish()
                    raise
                scope.obs.sides_seen.add(SIDE_AGENT)
                _add_parts(scope.obs, result)
                scope.obs.set_outcome(VALIDATION_VALID)
                scope.finish()
                return result

        _copy_metadata(wrapper, original)
        return wrapper

    def _open_stream_wrapper(self, original: Callable[..., Any]) -> Any:
        instrumentor = self

        def wrapper(processor: Any, *args: Any, **kwargs: Any) -> Any:
            if not instrumentor._instrumented:
                return original(processor, *args, **kwargs)
            # Not made current: the stream outlives this call, and the caller's code
            # between chunks must not run with our span as its parent.
            span = instrumentor.tracer.start_span("a2ui.parse_stream")
            scope = _Scope(instrumentor, span)
            scope.obs.sides_seen.add(SIDE_AGENT)
            try:
                stream = original(processor, *args, **kwargs)
            except BaseException as exc:
                scope.obs.set_outcome(VALIDATION_INVALID)
                scope.fail(exc)
                scope.finish()
                raise
            if not instrumentor._wrap_stream(stream, scope):
                scope.finish()
            return stream

        _copy_metadata(wrapper, original)
        return wrapper

    def _wrap_stream(self, stream: Any, scope: _Scope) -> bool:
        """Make ``feed`` and ``close`` of one stream feed ``scope``. False when they cannot."""
        try:
            if getattr(stream, _STREAM_MARKER, False):
                return False
            feed = stream.feed
            close = stream.close
        except Exception:  # noqa: BLE001
            return False

        def read(parts: Any) -> None:
            _add_parts(scope.obs, parts)
            for part in parts or ():
                if type(part).__name__ == "ErrorPart":
                    scope.obs.set_outcome(VALIDATION_INVALID)
                    scope.fail(getattr(part, "error", None) or RuntimeError("rejected block"))

        def feed_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                parts = feed(*args, **kwargs)
            except BaseException as exc:
                scope.obs.set_outcome(VALIDATION_INVALID)
                scope.fail(exc)
                scope.finish()
                raise
            read(parts)
            return parts

        def close_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                parts = close(*args, **kwargs)
            except BaseException as exc:
                scope.obs.set_outcome(VALIDATION_INVALID)
                scope.fail(exc)
                scope.finish()
                raise
            read(parts)
            if scope.obs.outcome is None:
                scope.obs.set_outcome(VALIDATION_VALID)
            scope.finish()
            return parts

        try:
            stream.feed = feed_wrapper
            stream.close = close_wrapper
            setattr(stream, _STREAM_MARKER, True)
        except Exception:  # noqa: BLE001 - a stream with __slots__ cannot be patched
            return False
        # A stream that is never closed still ends its span, when it is collected.
        with contextlib.suppress(Exception):
            weakref.finalize(stream, scope.finish).atexit = False
        return True

    # -- RendererToAgentMessageProcessor ------------------------------------------

    def _wrap_rpc_processor(self) -> int:
        try:
            processor_class = importlib.import_module(
                "a2ui.rpc.processor"
            ).RendererToAgentMessageProcessor
        except Exception as exc:  # noqa: BLE001
            logger.debug("a2ui renderer message processor not instrumented: %s", exc)
            return 0
        if vars(processor_class).get(_MARKER):
            return 0
        original = vars(processor_class).get("process_message")
        if original is None or not inspect.iscoroutinefunction(original):
            return 0
        self._patch(processor_class, "process_message", self._process_message_wrapper(original))
        setattr(processor_class, _MARKER, True)
        return 1

    def _process_message_wrapper(self, original: Callable[..., Any]) -> Any:
        instrumentor = self

        async def wrapper(processor: Any, message: Any, *args: Any, **kwargs: Any) -> Any:
            if not instrumentor._instrumented:
                return await original(processor, message, *args, **kwargs)
            data = message_to_dict(message)
            kind = next((k for k in (data or {}) if k in RENDERER_KINDS), None)
            name = f"a2ui.process {snake(kind)}" if kind else "a2ui.process unknown"
            with instrumentor.tracer.start_as_current_span(
                name, record_exception=False, set_status_on_exception=False
            ) as span:
                scope = _Scope(instrumentor, span, owned_end=False)
                if data is not None:
                    scope.obs.add_payload(data, SIDE_RENDERER)
                try:
                    result = await original(processor, message, *args, **kwargs)
                except BaseException as exc:
                    if "Validation" in type(exc).__name__:
                        scope.obs.set_outcome(VALIDATION_INVALID)
                    scope.fail(exc)
                    scope.finish()
                    raise
                # The replies go back to the renderer, so they are the agent's messages
                # (an agentFunctionResponse, which may carry a function error).
                scope.obs.add_payload(list(getattr(result, "replies", None) or []), SIDE_AGENT)
                scope.finish()
                return result

        _copy_metadata(wrapper, original)
        return wrapper

    # -- parse_content_to_parts ----------------------------------------------------

    def _wrap_parts_module(self) -> int:
        try:
            parts_module = importlib.import_module(PARTS_MODULE)
        except Exception as exc:  # noqa: BLE001 - needs a2ui-a2a, and so a2a-sdk 0.3
            logger.debug("%s not instrumented: %s", PARTS_MODULE, exc)
            return 0
        original = getattr(parts_module, "parse_content_to_parts", None)
        if original is None or getattr(original, _MARKER, False):
            return 0
        if self._filter is None:
            self._filter = _SwallowedFilter()
            logging.getLogger(PARTS_LOGGER).addFilter(self._filter)
        instrumentor = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not instrumentor._instrumented:
                return original(*args, **kwargs)
            holder: List[str] = []
            token = _swallowed.set(holder)
            try:
                return original(*args, **kwargs)
            finally:
                _swallowed.reset(token)
                if holder:
                    instrumentor._record_swallowed(holder[0])

        _copy_metadata(wrapper, original)
        setattr(wrapper, _MARKER, True)
        self._patch(parts_module, "parse_content_to_parts", wrapper)
        return 1

    def _record_swallowed(self, error_class: str) -> None:
        """Mark the current span: the agent SDK dropped a response it could not parse."""
        try:
            span = trace.get_current_span()
            if not span.is_recording():
                return
            span.set_attribute(A2UIAttributes.VALIDATION_OUTCOME, VALIDATION_INVALID)
            span.add_event(SWALLOWED_EVENT, {A2UIAttributes.ERROR_TYPE: error_class})
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not record a swallowed A2UI failure: %s", exc)

    # -- BaseInstrumentor ----------------------------------------------------------

    def _extract_usage(self, result: Any) -> Optional[Dict[str, int]]:
        return None


def _add_parts(obs: A2uiObservation, parts: Any) -> None:
    """Feed the A2UI messages of parsed response parts to ``obs`` as the agent's."""
    for part in parts or ():
        messages = getattr(part, "a2ui", None)
        if isinstance(messages, (list, tuple)):
            obs.add_payload(list(messages), SIDE_AGENT)


def _copy_metadata(wrapper: Any, original: Any) -> None:
    for attribute in ("__name__", "__qualname__", "__doc__", "__module__"):
        with contextlib.suppress(Exception):
            setattr(wrapper, attribute, getattr(original, attribute))
    with contextlib.suppress(Exception):
        wrapper.__wrapped__ = original
