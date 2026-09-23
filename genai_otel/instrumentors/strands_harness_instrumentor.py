"""Native OpenTelemetry instrumentation for Strands Harness.

The Strands SDK already emits provider/model/tool spans.  This integration adds
the harness execution structure around those spans through the public hook API,
and enriches an existing SDK span when one is active instead of creating a
duplicate model or tool span.

``strands-harness`` is intentionally imported only after the optional
instrumentor is enabled.  The core package therefore remains usable without
Strands installed.
"""

from __future__ import annotations

import contextvars
import importlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


_ACTIVE_RUNS: contextvars.ContextVar[Tuple["_Run", ...]] = contextvars.ContextVar(
    "genai_otel_strands_active_runs", default=()
)
_ACTIVE_OPERATIONS: contextvars.ContextVar[Tuple["_Operation", ...]] = contextvars.ContextVar(
    "genai_otel_strands_active_operations", default=()
)


def _value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _first(value: Any, *names: str) -> Any:
    for name in names:
        found = _value(value, name)
        if found is not None:
            return found
    return None


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    for name in ("model_id", "id", "name", "value"):
        candidate = _value(value, name)
        if candidate is not None and candidate is not value:
            return _text(candidate)
    return str(value)


def _agent_name(agent: Any) -> str:
    return _text(_first(agent, "name", "agent_id", "id")) or agent.__class__.__name__


def _agent_id(agent: Any) -> Optional[str]:
    return _text(_first(agent, "agent_id", "id"))


def _model_name(agent: Any) -> Optional[str]:
    model = _first(agent, "model", "model_id", "model_name")
    return _text(model)


def _json(value: Any, maximum: int = 200) -> str:
    try:
        encoded = json.dumps(value, default=str, sort_keys=True)
    except Exception:  # pragma: no cover - defensive telemetry path
        encoded = str(value)
    return encoded[:maximum] if maximum > 0 else encoded


def _exception_type(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, BaseException):
        return value.__class__.__name__
    return _text(_first(value, "type", "error_type", "name")) or value.__class__.__name__


def _usage_attributes(value: Any) -> Dict[str, int]:
    """Extract common usage spellings without importing a provider SDK."""
    candidates = [value]
    for name in ("usage", "metrics", "token_usage", "usage_metadata"):
        nested = _value(value, name)
        if nested is not None:
            candidates.insert(0, nested)

    attrs: Dict[str, int] = {}
    aliases = {
        "input_tokens": ("input_tokens", "prompt_tokens", "input"),
        "output_tokens": ("output_tokens", "completion_tokens", "output"),
        "cache_read_input_tokens": (
            "cache_read_input_tokens",
            "cache_read_tokens",
            "prompt_cache_hit_tokens",
        ),
        "cache_creation_input_tokens": (
            "cache_creation_input_tokens",
            "cache_write_tokens",
            "prompt_cache_miss_tokens",
        ),
        "total_tokens": ("total_tokens", "total"),
    }
    for target, names in aliases.items():
        for candidate in candidates:
            number = _first(candidate, *names)
            if isinstance(number, (int, float)) and not isinstance(number, bool):
                attrs[target] = int(number)
                break
    if "total_tokens" not in attrs:
        total = attrs.get("input_tokens", 0) + attrs.get("output_tokens", 0)
        if total:
            attrs["total_tokens"] = total
    return attrs


@dataclass
class _Run:
    agent: Any
    span: Span
    token: Any
    root_span: Optional[Span] = None
    root_token: Any = None
    state_token: Any = None


@dataclass
class _Operation:
    key: str
    span: Span
    owned: bool
    token: Any = None


class StrandsHarnessHook:
    """HookProvider-compatible callbacks used by ``create_harness``.

    The class deliberately follows the small public HookProvider protocol
    (``register_hooks``) without importing Strands at module import time.
    """

    _OUR_SPANS = frozenset(
        {"harness.run", "agent.run", "subagent.run", "llm.request", "tool.call", "mcp.call"}
    )

    def __init__(self, instrumentor: "StrandsHarnessInstrumentor") -> None:
        self.instrumentor = instrumentor
        self._registered = set()

    @property
    def tracer(self):
        return self.instrumentor.tracer

    @property
    def config(self):
        return self.instrumentor.config

    def register_hooks(self, registry: Any, **kwargs: Any) -> None:
        """Register only events present in the installed Strands version."""
        modules = []
        for module_name in ("strands.hooks.events", "strands.hooks"):
            try:
                modules.append(importlib.import_module(module_name))
            except ImportError:
                continue

        event_methods = {
            "AgentInitializedEvent": self.on_agent_initialized,
            "BeforeInvocationEvent": self.on_before_invocation,
            "AfterInvocationEvent": self.on_after_invocation,
            "BeforeModelCallEvent": self.on_before_model_call,
            "AfterModelCallEvent": self.on_after_model_call,
            "BeforeToolCallEvent": self.on_before_tool_call,
            "AfterToolCallEvent": self.on_after_tool_call,
            "ModelStreamChunkEvent": self.on_model_stream_chunk,
            "BeforeContextCompactionEvent": self.on_before_context_compaction,
            "AfterContextCompactionEvent": self.on_after_context_compaction,
            "BeforeContextReductionEvent": self.on_before_context_compaction,
            "AfterContextReductionEvent": self.on_after_context_compaction,
            "BeforeContextOverflowEvent": self.on_before_context_compaction,
            "AfterContextOverflowEvent": self.on_after_context_compaction,
            "BeforeContextRecoveryEvent": self.on_before_context_compaction,
            "AfterContextRecoveryEvent": self.on_after_context_compaction,
            "BeforeMemoryReadEvent": self.on_before_memory_read,
            "AfterMemoryReadEvent": self.on_after_memory_read,
            "BeforeMemoryWriteEvent": self.on_before_memory_write,
            "AfterMemoryWriteEvent": self.on_after_memory_write,
        }
        add_callback = getattr(registry, "add_callback", None)
        if not callable(add_callback):
            logger.debug("Strands hook registry has no add_callback method")
            return
        for event_name, callback in event_methods.items():
            event_type = next((getattr(module, event_name, None) for module in modules), None)
            if event_type is None or event_type in self._registered:
                continue
            try:
                add_callback(event_type, callback)
                self._registered.add(event_type)
            except Exception:  # pragma: no cover - SDK-version defensive path
                logger.debug("Could not register Strands event %s", event_name, exc_info=True)

    def _capture(self, value: Any) -> Optional[str]:
        if not self.config or not self.config.enable_content_capture:
            return None
        if (
            getattr(self.config, "enable_pii_detection", False)
            and str(getattr(self.config, "pii_mode", "detect")).lower() == "redact"
        ):
            return "[REDACTED]"
        maximum = getattr(self.config, "content_max_length", 200) or 0
        return _json(value, maximum)

    def _agent_attributes(self, agent: Any, event: Any = None) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {
            "gen_ai.system": "strands",
            "gen_ai.operation.name": "invoke_agent",
            "strands.harness.name": "strands_harness",
            "strands.harness.version": self.instrumentor.harness_version,
            "gen_ai.agent.name": _agent_name(agent),
        }
        agent_id = _agent_id(agent)
        if agent_id:
            attrs["gen_ai.agent.id"] = agent_id
        model = _model_name(agent)
        if model:
            attrs["gen_ai.request.model"] = model
        state = _value(event, "invocation_state") if event else None
        for attr, names in {
            "strands.session.id": ("session_id", "sessionId"),
            "strands.run.id": ("run_id", "runId", "task_id", "taskId"),
            "strands.parent_run.id": ("parent_run_id", "parentRunId", "parent_task_id"),
        }.items():
            found = _first(state, *names)
            if found is None:
                found = _first(agent, *names)
            if found is not None:
                attrs[attr] = _text(found)
        session_manager = _first(
            agent, "session_manager", "_session_manager", "sessionManager", "_sessionManager"
        )
        if session_manager is not None:
            attrs["strands.session.enabled"] = True
        session_id = _first(agent, "session_id", "_session_id", "sessionId", "_sessionId")
        if session_id is not None:
            attrs["strands.session.id"] = _text(session_id)
        resumed = _first(state, "resume", "resumed", "session_resumed")
        if resumed is not None:
            attrs["strands.session.resumed"] = bool(resumed)
        if session_id is not None and resumed is None:
            attrs["strands.session.resumed"] = False
        memory_manager = _first(agent, "memory_manager", "_memory_manager", "memoryManager")
        if memory_manager is not None:
            attrs["strands.memory.enabled"] = True
        return attrs

    def _event_span(self, name: str, attrs: Dict[str, Any], key: str) -> _Operation:
        current = trace.get_current_span()
        current_name = getattr(current, "name", None)
        if current is not None and current.is_recording() and current_name not in self._OUR_SPANS:
            current.set_attributes(attrs)
            return _Operation(key, current, False)
        span = self.tracer.start_span(name, attributes=attrs)
        token = otel_context.attach(trace.set_span_in_context(span))
        return _Operation(key, span, True, token)

    @staticmethod
    def _end_operation(operation: _Operation) -> None:
        if not operation.owned:
            return
        token = operation.token
        if token is not None:
            try:
                otel_context.detach(token)
            except (ValueError, LookupError):
                pass
        operation.span.end()

    def _finish_run(self, event: Any) -> None:
        runs = list(_ACTIVE_RUNS.get())
        agent = _value(event, "agent")
        index = next((i for i in range(len(runs) - 1, -1, -1) if runs[i].agent is agent), None)
        if index is None:
            return
        run = runs.pop(index)
        _ACTIVE_RUNS.set(tuple(runs))
        exception = _first(event, "exception", "error")
        result = _value(event, "result")
        stop_reason = _text(_first(result, "stop_reason", "stopReason", "status"))
        error_type = _exception_type(exception)
        if error_type:
            run.span.set_attribute("error.type", error_type)
            run.span.set_status(Status(StatusCode.ERROR, error_type))
            if run.root_span:
                run.root_span.set_attribute("error.type", error_type)
                run.root_span.set_status(Status(StatusCode.ERROR, error_type))
        elif stop_reason:
            run.span.set_attribute("strands.stop_reason", stop_reason)
            run.span.set_attribute("strands.run.status", stop_reason)
            if stop_reason.lower() in {
                "error",
                "failed",
                "failure",
                "timeout",
                "cancelled",
                "canceled",
            }:
                run.span.set_status(Status(StatusCode.ERROR, stop_reason))
        captured = self._capture(result)
        if captured is not None:
            run.span.set_attribute("output.value", captured)
        self._detach(run.token)
        run.span.end()
        if run.root_span is not None:
            self._detach(run.root_token)
            run.root_span.end()

    @staticmethod
    def _detach(token: Any) -> None:
        try:
            otel_context.detach(token)
        except (ValueError, LookupError):
            pass

    def on_agent_initialized(self, event: Any) -> None:
        # Initialization is represented on the next run span. Keep this callback
        # intentionally side-effect free so constructing a harness emits no fake
        # work span.
        return None

    def on_before_invocation(self, event: Any) -> None:
        agent = _value(event, "agent")
        attrs = self._agent_attributes(agent, event)
        active = _ACTIVE_RUNS.get()
        if active:
            span = self.tracer.start_span("subagent.run", attributes=attrs)
            token = otel_context.attach(trace.set_span_in_context(span))
            run = _Run(agent=agent, span=span, token=token, state_token=None)
        else:
            root = self.tracer.start_span("harness.run", attributes=attrs)
            root_token = otel_context.attach(trace.set_span_in_context(root))
            span = self.tracer.start_span("agent.run", attributes=attrs)
            token = otel_context.attach(trace.set_span_in_context(span))
            run = _Run(
                agent=agent,
                span=span,
                token=token,
                root_span=root,
                root_token=root_token,
                state_token=None,
            )
        run.state_token = _ACTIVE_RUNS.set((*active, run))
        messages = _value(event, "messages")
        captured = self._capture(messages)
        if captured is not None:
            run.span.set_attribute("input.value", captured)

    def on_after_invocation(self, event: Any) -> None:
        self._finish_run(event)

    def on_before_model_call(self, event: Any) -> None:
        agent = _value(event, "agent")
        attrs: Dict[str, Any] = {
            "gen_ai.operation.name": "chat",
            "strands.operation": "llm.request",
            "gen_ai.system": "strands",
        }
        model = _model_name(agent)
        if model:
            attrs["gen_ai.request.model"] = model
        projected = _first(event, "projected_input_tokens", "input_tokens")
        if isinstance(projected, (int, float)):
            attrs["gen_ai.usage.input_tokens"] = int(projected)
        operation = self._event_span("llm.request", attrs, "model")
        _ACTIVE_OPERATIONS.set((*_ACTIVE_OPERATIONS.get(), operation))

    def on_after_model_call(self, event: Any) -> None:
        operations = list(_ACTIVE_OPERATIONS.get())
        index = next(
            (i for i in range(len(operations) - 1, -1, -1) if operations[i].key == "model"), None
        )
        if index is None:
            return
        operation = operations.pop(index)
        _ACTIVE_OPERATIONS.set(tuple(operations))
        usage = _usage_attributes(_first(event, "stop_response", "response", "result", "metrics"))
        for key, number in usage.items():
            operation.span.set_attribute(f"gen_ai.usage.{key}", number)
            operation.span.set_attribute(f"llm.token_count.{key.replace('_tokens', '')}", number)
        exception = _first(event, "exception", "error")
        if exception is not None:
            error_type = _exception_type(exception)
            operation.span.set_attribute("error.type", error_type)
            operation.span.set_status(Status(StatusCode.ERROR, error_type))
        streaming = _first(event, "streaming", "is_streaming")
        if streaming is not None:
            operation.span.set_attribute("gen_ai.request.stream", bool(streaming))
        self._end_operation(operation)

    def on_model_stream_chunk(self, event: Any) -> None:
        operations = _ACTIVE_OPERATIONS.get()
        for operation in reversed(operations):
            if operation.key == "model":
                operation.span.set_attribute("gen_ai.request.stream", True)
                operation.span.set_attribute("strands.response.streaming", True)
                return

    def _tool_attributes(self, event: Any) -> Tuple[str, Dict[str, Any]]:
        selected = _first(event, "selected_tool", "tool", "tool_name")
        tool_use = _value(event, "tool_use")
        name = (
            _text(_first(selected, "name", "tool_name"))
            or _text(_first(tool_use, "name", "tool_name"))
            or "unknown"
        )
        module = _text(_first(selected, "__module__")) or ""
        category = "mcp" if "mcp" in name.lower() or "mcp" in module.lower() else "tool"
        span_name = "mcp.call" if category == "mcp" else "tool.call"
        attrs: Dict[str, Any] = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": name,
            "strands.tool.category": category,
            "gen_ai.system": "strands",
        }
        server = _first(selected, "server_name", "server", "mcp_server")
        if server is None:
            server = _first(tool_use, "server_name", "server", "mcp_server")
        if server is not None:
            attrs["strands.mcp.server"] = _text(server)
        call_id = _first(tool_use, "toolUseId", "tool_use_id", "call_id", "id")
        if call_id is not None:
            attrs["gen_ai.tool.call.id"] = _text(call_id)
        captured = self._capture(_first(tool_use, "input", "arguments", "args"))
        if captured is not None:
            attrs["gen_ai.tool.call.arguments"] = captured
        return span_name, attrs

    def on_before_tool_call(self, event: Any) -> None:
        name, attrs = self._tool_attributes(event)
        operation = self._event_span(name, attrs, "tool")
        _ACTIVE_OPERATIONS.set((*_ACTIVE_OPERATIONS.get(), operation))

    def on_after_tool_call(self, event: Any) -> None:
        operations = list(_ACTIVE_OPERATIONS.get())
        index = next(
            (i for i in range(len(operations) - 1, -1, -1) if operations[i].key == "tool"), None
        )
        if index is None:
            return
        operation = operations.pop(index)
        _ACTIVE_OPERATIONS.set(tuple(operations))
        exception = _first(event, "exception", "error")
        if exception is not None:
            error_type = _exception_type(exception)
            operation.span.set_attribute("error.type", error_type)
            operation.span.set_status(Status(StatusCode.ERROR, error_type))
        captured = self._capture(_first(event, "result", "output"))
        if captured is not None:
            operation.span.set_attribute("output.value", captured)
        self._end_operation(operation)

    def on_before_context_compaction(self, event: Any) -> None:
        attrs = {
            "gen_ai.operation.name": "context.compaction",
            "strands.context.operation": "compaction",
            "gen_ai.system": "strands",
        }
        operation = self._event_span("context.compaction", attrs, "context")
        _ACTIVE_OPERATIONS.set((*_ACTIVE_OPERATIONS.get(), operation))

    def on_after_context_compaction(self, event: Any) -> None:
        operations = list(_ACTIVE_OPERATIONS.get())
        index = next(
            (i for i in range(len(operations) - 1, -1, -1) if operations[i].key == "context"), None
        )
        if index is None:
            return
        operation = operations.pop(index)
        _ACTIVE_OPERATIONS.set(tuple(operations))
        exception = _first(event, "exception", "error")
        if exception is not None:
            error_type = _exception_type(exception)
            operation.span.set_attribute("error.type", error_type)
            operation.span.set_status(Status(StatusCode.ERROR, error_type))
        self._end_operation(operation)

    def _before_memory(self, event: Any, operation_name: str) -> None:
        attrs = {
            "gen_ai.operation.name": operation_name,
            "strands.memory.operation": operation_name.rsplit(".", 1)[-1],
            "gen_ai.system": "strands",
        }
        operation = self._event_span(operation_name, attrs, operation_name)
        _ACTIVE_OPERATIONS.set((*_ACTIVE_OPERATIONS.get(), operation))

    def _after_memory(self, event: Any, operation_name: str) -> None:
        operations = list(_ACTIVE_OPERATIONS.get())
        index = next(
            (i for i in range(len(operations) - 1, -1, -1) if operations[i].key == operation_name),
            None,
        )
        if index is None:
            return
        operation = operations.pop(index)
        _ACTIVE_OPERATIONS.set(tuple(operations))
        exception = _first(event, "exception", "error")
        if exception is not None:
            error_type = _exception_type(exception)
            operation.span.set_attribute("error.type", error_type)
            operation.span.set_status(Status(StatusCode.ERROR, error_type))
        self._end_operation(operation)

    def on_before_memory_read(self, event: Any) -> None:
        self._before_memory(event, "memory.read")

    def on_after_memory_read(self, event: Any) -> None:
        self._after_memory(event, "memory.read")

    def on_before_memory_write(self, event: Any) -> None:
        self._before_memory(event, "memory.write")

    def on_after_memory_write(self, event: Any) -> None:
        self._after_memory(event, "memory.write")


class StrandsHarnessInstrumentor(BaseInstrumentor):
    """Instrument the public ``strands_harness.create_harness`` factory."""

    def __init__(self):
        super().__init__()
        self._strands_available = False
        self._hook: Optional[StrandsHarnessHook] = None
        self.harness_version = "unknown"
        self._check_availability()

    def _check_availability(self) -> None:
        try:
            module = importlib.import_module("strands_harness")
            self._strands_available = callable(getattr(module, "create_harness", None))
            self.harness_version = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            self._strands_available = False
            logger.debug("strands-harness is not installed; skipping Strands instrumentation")

    def instrument(self, config: OTelConfig) -> None:
        if not self._strands_available:
            logger.debug("Skipping Strands Harness instrumentation - optional dependency missing")
            return
        if self._instrumented:
            return
        self._setup_config(config)
        self._hook = StrandsHarnessHook(self)
        try:
            import wrapt

            wrapt.wrap_function_wrapper(
                "strands_harness", "create_harness", self._wrap_create_harness
            )
            self._instrumented = True
            logger.info("Strands Harness instrumentation enabled")
        except Exception as exc:  # pragma: no cover - dependency-version defensive path
            logger.error("Failed to instrument Strands Harness: %s", exc, exc_info=True)
            if config.fail_on_error:
                raise

    def _wrap_create_harness(self, wrapped, instance, args, kwargs):
        hook = self._hook
        if hook is not None:
            updated = dict(kwargs)
            hooks = list(updated.get("hooks") or [])
            if not any(item is hook for item in hooks):
                hooks.append(hook)
            updated["hooks"] = hooks
            kwargs = updated
        return wrapped(*args, **kwargs)

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        usage = _usage_attributes(result)
        if not usage:
            return None
        return {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }


__all__ = ["StrandsHarnessHook", "StrandsHarnessInstrumentor"]
