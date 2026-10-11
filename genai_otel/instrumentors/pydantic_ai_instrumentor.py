"""OpenTelemetry instrumentor for Pydantic AI framework.

This instrumentor automatically traces agent runs made with the Pydantic AI
type-safe agent framework: ``Agent.run()``, ``run_sync()``, ``run_stream()`` and
``iter()``, plus ``run_stream_sync()`` and ``run_stream_events()``, which are built
on those. The model requests an agent makes are traced by the provider
instrumentor (OpenAI, Anthropic, ...) and nest under the agent span. Tools are
recorded by name on the agent span; their executions get no span of their own.

Pydantic AI is a new framework (Dec 2024) by the Pydantic team that provides
type-safe agent development with full Pydantic validation and multi-provider support.

Requirements:
    pip install pydantic-ai
"""

import contextvars
import importlib.util
import inspect
import json
import logging
import sys
import time
from contextlib import AbstractAsyncContextManager
from typing import Any, Dict, Optional

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from ..config import OTelConfig
from ..server_metrics import get_server_metrics
from .base import BaseInstrumentor, _context_with_usage_holder, strip_content_attributes

logger = logging.getLogger(__name__)

# The agents with a traced run open in the current context. Agent.run() and
# Agent.run_stream() are implemented on top of Agent.iter(), and run_sync() on top
# of run(). Without this every run() carried a second, identical agent span from
# iter(), and each layer opened an agent budget frame of its own.
_RUNNING_AGENTS: contextvars.ContextVar = contextvars.ContextVar(
    "genai_otel_pydantic_ai_running_agents", default=()
)

# Set while a run_stream_sync() that never leaves its run_stream() block is under
# way (Pydantic AI before 2.0). A span opened for that block could never end.
_UNTRACED_STREAM: contextvars.ContextVar = contextvars.ContextVar(
    "genai_otel_pydantic_ai_untraced_stream", default=False
)

_MISSING = object()


def _stream_sync_leaves_its_block() -> bool:
    """Whether run_stream_sync() exits the run_stream() block it enters.

    From 2.0 it runs the stream through a bridge (``pydantic_ai._sync_stream``) and
    does. Before that it entered the block inside a helper generator and never left.
    """
    if "pydantic_ai._sync_stream" in sys.modules:
        return True
    try:
        return importlib.util.find_spec("pydantic_ai._sync_stream") is not None
    except Exception:  # noqa: BLE001 - a stub or partial install has no spec to search
        return False


def _read(obj: Any, name: str) -> Any:
    """``getattr`` that treats a property that raises as a missing attribute.

    Several result properties raise until the run has produced a response.
    """
    try:
        return getattr(obj, name, _MISSING)
    except Exception:  # noqa: BLE001
        return _MISSING


def _frame(depth: int):
    """The frame ``depth`` levels above the caller, or None."""
    try:
        return sys._getframe(depth + 1)
    except ValueError:
        return None


def _is_wrapt_frame(frame: Any) -> bool:
    module = str(frame.f_globals.get("__name__", ""))
    return module == "wrapt" or module.startswith("wrapt.")


def _is_running(instance: Any) -> bool:
    return any(agent is instance for agent in _RUNNING_AGENTS.get())


def _finished_result(value: Any) -> Any:
    """The settled result behind an ``iter()`` run or a consumed stream, else ``value``."""
    try:
        settled = value.result
    except Exception:  # noqa: BLE001 - a stream has no result until it has been consumed
        return value
    return value if settled is None else settled


class _AgentRunSpan:
    """The span for one agent run, opened and closed in the context that runs it.

    ``Agent.run()`` is a coroutine function and ``run_stream()`` / ``iter()`` return
    async context managers: calling them does no work. The base wrapper opens its
    span, and makes it the current span, when the wrapped function is called, which
    for these is the wrong moment and often the wrong context:

    * ``run_sync()`` runs ``run()`` in a new task, so the span attached at the call
      was detached in another Context and OpenTelemetry logged "Failed to detach
      context" on every run;
    * ``asyncio.gather(a.run(...), b.run(...))`` attached both spans to the caller
      before either run started: the second run became a child of the first, both
      runs' model requests were parented to the second, and the caller was left
      with a finished span as its current span;
    * a ``run_stream()`` span ended before the context manager was entered, so it
      lasted microseconds, parented nothing and was OK whatever the stream did.

    This does the base wrapper's bookkeeping around the part that actually runs.
    """

    def __init__(self, instrumentor, span_name: str, instance: Any, args: Any, kwargs: Any):
        self._instrumentor = instrumentor
        self._span_name = span_name
        self._instance = instance
        self._args = args
        self._kwargs = kwargs
        self._span: Any = None
        self._context_token: Any = None
        self._running_token: Any = None
        self._counted_running = False
        self._start_time = 0.0
        self._finished = False

    def infer_agent_name(self, frame) -> None:
        self._instrumentor._infer_agent_name(self._instance, frame)

    def begin(self) -> None:
        instrumentor = self._instrumentor
        self._start_time = time.time()
        try:
            # run_sync() -> run() is one invocation with two spans; the outer one
            # holds its budget frame.
            nested = _is_running(self._instance)
            self._running_token = _RUNNING_AGENTS.set(_RUNNING_AGENTS.get() + (self._instance,))
            attrs = instrumentor._initial_attributes(self._instance, self._args, self._kwargs)
            span = instrumentor.tracer.start_span(self._span_name, attributes=attrs)
            self._span = span
            if not span.is_recording():
                # Sampled out: keep it current so the children inherit the decision.
                self._context_token = otel_context.attach(trace.set_span_in_context(span))
                return
            self._context_token = otel_context.attach(
                _context_with_usage_holder(span, trace.set_span_in_context(span))
            )
            if not nested:
                instrumentor._begin_agent_budget(span, attrs)
            server_metrics = get_server_metrics()
            if server_metrics:
                server_metrics.increment_requests_running()
                self._counted_running = True
            instrumentor._set_session_attributes(span, self._instance, self._args, self._kwargs)
        except Exception as e:  # noqa: BLE001 - tracing must never break the run
            logger.debug("Could not open span '%s': %s", self._span_name, e)

    def finish(self, result: Any = None, error: Optional[BaseException] = None) -> None:
        if self._finished:
            return
        self._finished = True
        instrumentor = self._instrumentor
        span = self._span
        try:
            if span is None:
                pass
            elif not span.is_recording():
                # Client metrics stay independent of trace sampling, as in the base wrapper.
                if error is None:
                    if instrumentor.request_counter:
                        instrumentor.request_counter.add(1, {"operation": self._span_name})
                    if instrumentor.latency_histogram:
                        instrumentor.latency_histogram.record(
                            time.time() - self._start_time, {"operation": self._span_name}
                        )
            elif error is None:
                if instrumentor.request_counter:
                    instrumentor.request_counter.add(1, {"operation": self._span_name})
                try:
                    instrumentor._record_result_metrics(
                        span, result, self._start_time, self._kwargs
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to record metrics for span '%s': %s", self._span_name, e)
                try:
                    instrumentor._run_evaluation_checks(span, self._args, self._kwargs, result)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Failed to run evaluation checks for span '%s': %s", self._span_name, e
                    )
                instrumentor._finish_agent_budget(span)
                span.set_status(Status(StatusCode.OK))
            else:
                instrumentor._finish_agent_budget(span)
                # A cancelled run ends its span without a status, as the base wrapper does.
                if isinstance(error, Exception):
                    if instrumentor.error_counter:
                        instrumentor.error_counter.add(
                            1,
                            {"operation": self._span_name, "error_type": type(error).__name__},
                        )
                    span.set_status(Status(StatusCode.ERROR, str(error)))
                    span.record_exception(error)
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not finish span '%s': %s", self._span_name, e)
        finally:
            self._close()

    def _close(self) -> None:
        try:
            if self._span is not None:
                self._span.end()
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not end span '%s': %s", self._span_name, e)
        # Whatever happened above, the caller's context must be handed back.
        try:
            if self._context_token is not None:
                otel_context.detach(self._context_token)
            if self._running_token is not None:
                try:
                    _RUNNING_AGENTS.reset(self._running_token)
                except ValueError:  # closed from another Context; nothing to restore here
                    pass
            if self._counted_running:
                server_metrics = get_server_metrics()
                if server_metrics:
                    server_metrics.decrement_requests_running()
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not close span '%s': %s", self._span_name, e)


class _TracedAgentContext(AbstractAsyncContextManager):
    """``async with agent.run_stream(...)`` / ``agent.iter(...)`` with the span held open.

    The context manager owns the span's lifetime: a caller may consume the stream,
    drive the graph node by node or leave early, but it always leaves the block.
    """

    def __init__(self, manager, call: _AgentRunSpan, infer_name: bool):
        self._manager = manager
        self._call = call
        self._infer_name = infer_name
        self._value = None

    async def __aenter__(self):
        if self._infer_name:
            self._call.infer_agent_name(_frame(1))
        self._call.begin()
        try:
            self._value = await self._manager.__aenter__()
        except BaseException as exc:
            self._call.finish(error=exc)
            raise
        return self._value

    async def __aexit__(self, exc_type, exc, tb):
        try:
            suppressed = await self._manager.__aexit__(exc_type, exc, tb)
        except BaseException as raised:
            self._call.finish(error=raised)
            raise
        if exc is not None and not suppressed:
            self._call.finish(error=exc)
        else:
            self._call.finish(result=_finished_result(self._value))
        return suppressed


class PydanticAIInstrumentor(BaseInstrumentor):
    """Instrumentor for Pydantic AI agent framework"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._pydantic_ai_available = False
        # Whether each wrapped entry point takes ``infer_name``, by function.
        self._infer_name_support: Dict[Any, bool] = {}
        self._check_availability()

    def _check_availability(self):
        """Check if Pydantic AI library is available."""
        try:
            import pydantic_ai

            self._pydantic_ai_available = True
            logger.debug("Pydantic AI library detected and available for instrumentation")
        except ImportError:
            logger.debug("Pydantic AI library not installed, instrumentation will be skipped")
            self._pydantic_ai_available = False

    def instrument(self, config: OTelConfig):
        """Instrument Pydantic AI framework if available.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._pydantic_ai_available:
            logger.debug("Skipping Pydantic AI instrumentation - library not available")
            return

        # Idempotency guard: repeated instrument() calls must not stack wrappers
        # on the shared Agent class.
        if self._instrumented:
            logger.debug("Pydantic AI already instrumented, skipping repeat instrument()")
            return

        self.config = config

        try:
            import wrapt
            from pydantic_ai import Agent

            # Instrument Agent.run (async execution). run_stream_events() runs this
            # in a background task, so it is traced through it.
            if hasattr(Agent, "run"):
                original_run = Agent.run
                Agent.run = wrapt.FunctionWrapper(original_run, self._wrap_agent_run)

            # Instrument Agent.run_sync (explicit synchronous execution)
            if hasattr(Agent, "run_sync"):
                original_run_sync = Agent.run_sync
                Agent.run_sync = wrapt.FunctionWrapper(original_run_sync, self._wrap_agent_run_sync)

            # Instrument Agent.run_stream (streaming execution). run_stream_sync()
            # enters this on its own event loop, so it is traced through it.
            if hasattr(Agent, "run_stream"):
                original_run_stream = Agent.run_stream
                Agent.run_stream = wrapt.FunctionWrapper(
                    original_run_stream, self._wrap_agent_run_stream
                )

            if hasattr(Agent, "run_stream_sync") and not _stream_sync_leaves_its_block():
                Agent.run_stream_sync = wrapt.FunctionWrapper(
                    Agent.run_stream_sync, self._wrap_agent_run_stream_sync_untraced
                )

            # Instrument Agent.iter (node-by-node execution). Called directly it had
            # no span, and each of its model requests started a trace of its own.
            if hasattr(Agent, "iter"):
                original_iter = Agent.iter
                Agent.iter = wrapt.FunctionWrapper(original_iter, self._wrap_agent_iter)

            self._instrumented = True
            logger.info("Pydantic AI instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument Pydantic AI: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _wrap_agent_run(self, wrapped, instance, args, kwargs):
        """Wrap Agent.run method with span.

        Args:
            wrapped: The original method.
            instance: The Agent instance.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        if not self._instrumented or not inspect.iscoroutinefunction(wrapped):
            return self.create_span_wrapper(
                span_name="pydantic_ai.agent.run",
                extract_attributes=self._extract_agent_attributes,
            )(wrapped)(*args, **kwargs)

        infer_name = self._claims_name_inference(instance, wrapped, kwargs)
        call_kwargs = {**kwargs, "infer_name": False} if infer_name else kwargs

        async def _run():
            if infer_name:
                # Pydantic AI looks at whoever is awaiting run(); so does this.
                self._infer_agent_name(instance, _frame(1))
            call = _AgentRunSpan(self, "pydantic_ai.agent.run", instance, args, kwargs)
            call.begin()
            try:
                result = await wrapped(*args, **call_kwargs)
            except BaseException as exc:
                call.finish(error=exc)
                raise
            call.finish(result=result)
            return result

        return _run()

    def _wrap_agent_run_sync(self, wrapped, instance, args, kwargs):
        """Wrap Agent.run_sync method with span.

        Args:
            wrapped: The original method.
            instance: The Agent instance.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        if self._claims_name_inference(instance, wrapped, kwargs):
            caller = _frame(1)
            # wrapt's pure-Python fallback puts frames of its own in between.
            while caller is not None and _is_wrapt_frame(caller):
                caller = caller.f_back
            self._infer_agent_name(instance, caller)
            kwargs = {**kwargs, "infer_name": False}

        token = _RUNNING_AGENTS.set(_RUNNING_AGENTS.get() + (instance,))
        try:
            return self.create_span_wrapper(
                span_name="pydantic_ai.agent.run_sync",
                extract_attributes=self._extract_agent_attributes,
            )(wrapped)(*args, **kwargs)
        finally:
            _RUNNING_AGENTS.reset(token)

    def _wrap_agent_run_stream(self, wrapped, instance, args, kwargs):
        """Wrap Agent.run_stream method with span.

        Args:
            wrapped: The original method.
            instance: The Agent instance.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        if not self._instrumented or _UNTRACED_STREAM.get():
            return wrapped(*args, **kwargs)

        infer_name = self._claims_name_inference(instance, wrapped, kwargs)
        manager = wrapped(*args, **({**kwargs, "infer_name": False} if infer_name else kwargs))
        if not hasattr(manager, "__aenter__"):
            return manager
        call = _AgentRunSpan(self, "pydantic_ai.agent.run_stream", instance, args, kwargs)
        return _TracedAgentContext(manager, call, infer_name)

    def _wrap_agent_run_stream_sync_untraced(self, wrapped, instance, args, kwargs):
        """Run a pre-2.0 run_stream_sync() with no agent span of its own.

        It enters run_stream() and never leaves the block, so the span would stay
        open, and count as a running request, for good. Its model requests are still
        traced by the provider instrumentor.
        """
        token = _UNTRACED_STREAM.set(True)
        try:
            return wrapped(*args, **kwargs)
        finally:
            try:
                _UNTRACED_STREAM.reset(token)
            except (ValueError, LookupError):
                pass

    def _wrap_agent_iter(self, wrapped, instance, args, kwargs):
        """Wrap Agent.iter method with span.

        Args:
            wrapped: The original method.
            instance: The Agent instance.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        manager = wrapped(*args, **kwargs)
        # Inside this agent's own traced run() / run_stream() the span already exists.
        if (
            not self._instrumented
            or _UNTRACED_STREAM.get()
            or _is_running(instance)
            or not hasattr(manager, "__aenter__")
        ):
            return manager
        call = _AgentRunSpan(self, "pydantic_ai.agent.iter", instance, args, kwargs)
        return _TracedAgentContext(manager, call, False)

    def _claims_name_inference(self, instance: Any, wrapped: Any, kwargs: Any) -> bool:
        """Whether this call would have Pydantic AI infer the agent's name.

        Pydantic AI names an unnamed agent after the variable that holds it, which it
        finds by scanning the locals of the frame that called the entry point. Wrapped,
        that frame belongs to this library. The base wrapper keeps the agent in a local
        called ``instance``, so every unnamed agent run through ``run_sync()`` was renamed
        "instance", and one awaited through ``run()`` was left unnamed. When this returns
        True the wrapper does the same scan against the application's frame and passes
        ``infer_name=False`` on.
        """
        try:
            if instance is None or not kwargs.get("infer_name", True):
                return False
            if getattr(instance, "name", None) is not None:
                return False
            func = getattr(wrapped, "__func__", wrapped)
            supported = self._infer_name_support.get(func)
            if supported is None:
                supported = "infer_name" in inspect.signature(func).parameters
                self._infer_name_support[func] = supported
            return supported
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not tell whether the agent name is inferred: %s", e)
            return False

    @staticmethod
    def _infer_agent_name(instance: Any, frame: Any) -> None:
        """Name ``instance`` after the variable holding it in ``frame``, as Pydantic AI does."""
        try:
            if frame is None or instance.name is not None:
                return
            for scope in (frame.f_locals, frame.f_globals):
                for name, item in list(scope.items()):
                    if item is instance:
                        instance.name = name
                        return
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to infer agent name: %s", e)

    def _initial_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Span attributes for a run, prepared the way the base wrapper prepares them."""
        attrs: Dict[str, Any] = {}
        try:
            extracted = self._extract_agent_attributes(instance, args, kwargs)
            extracted = self._with_provider_aliases(extracted)
            extracted = strip_content_attributes(extracted, self.config)
            for key, value in extracted.items():
                attrs[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to extract attributes for a Pydantic AI span: %s", e)
        return attrs

    def _set_session_attributes(self, span, instance: Any, args: Any, kwargs: Any) -> None:
        """Apply the configured session / user extractors, as the base wrapper does."""
        config = self.config
        if not config:
            return
        session_id_extractor = getattr(config, "session_id_extractor", None)
        if session_id_extractor:
            try:
                session_id = session_id_extractor(instance, args, kwargs)
                if session_id:
                    span.set_attribute("session.id", session_id)
                    span.set_attribute("gen_ai.conversation.id", session_id)
            except Exception as e:  # noqa: BLE001
                logger.debug("Failed to extract session ID: %s", e)
        user_id_extractor = getattr(config, "user_id_extractor", None)
        if user_id_extractor:
            try:
                user_id = user_id_extractor(instance, args, kwargs)
                if user_id:
                    span.set_attribute("user.id", user_id)
            except Exception as e:  # noqa: BLE001
                logger.debug("Failed to extract user ID: %s", e)

    def _cap_content(self, text: Any, default: int = 200) -> str:
        """Cap captured content per ``config.content_max_length``.

        Content capture is a required audit feature, so content keeps flowing;
        this only bounds its size. ``content_max_length == 0`` (or missing) means
        unlimited. When no usable config is present (unit tests) the historical
        default cap is used.
        """
        s = text if isinstance(text, str) else str(text)
        cfg = getattr(self, "config", None)
        max_len = default
        cfg_len = getattr(cfg, "content_max_length", None) if cfg is not None else None
        if isinstance(cfg_len, int):
            max_len = cfg_len
        if max_len and max_len > 0:
            return s[:max_len]
        return s

    def _extract_agent_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Extract attributes from Agent.run call.

        Args:
            instance: The Agent instance.
            args: Positional arguments (user prompt, etc.).
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "pydantic_ai"
        attrs["gen_ai.operation.name"] = "agent.run"

        # Extract agent name if available
        if hasattr(instance, "name") and instance.name:
            attrs["pydantic_ai.agent.name"] = instance.name
            # Cross-framework agent attribution per upstream proposal
            # semantic-conventions-genai#91.
            attrs["gen_ai.agent.name"] = instance.name

        # Extract model information
        if hasattr(instance, "model") and instance.model:
            model = instance.model
            # Get model name/provider
            if hasattr(model, "name"):
                attrs["gen_ai.request.model"] = model.name
                attrs["pydantic_ai.model.name"] = model.name
            elif hasattr(model, "model_name"):
                attrs["gen_ai.request.model"] = model.model_name
                attrs["pydantic_ai.model.name"] = model.model_name

            # Extract provider if available
            if hasattr(model, "__class__"):
                provider = model.__class__.__name__
                attrs["pydantic_ai.model.provider"] = provider

        # Extract system prompts
        if hasattr(instance, "_system_prompts") and instance._system_prompts:
            try:
                # System prompts can be list of strings or functions
                prompts = []
                for prompt in instance._system_prompts:
                    if callable(prompt):
                        prompts.append("<function>")
                    else:
                        prompts.append(self._cap_content(prompt))
                attrs["pydantic_ai.system_prompts"] = prompts[:5]  # Limit to 5
            except Exception as e:
                logger.debug("Failed to extract system prompts: %s", e)

        # Extract tools/functions if available. Current releases keep the agent's
        # own tools in a toolset; older ones had them in `_function_tools`.
        function_tools = getattr(instance, "_function_tools", None)
        if not function_tools:
            function_tools = getattr(getattr(instance, "_function_toolset", None), "tools", None)
        if function_tools:
            try:
                tool_names = []
                for tool_name in function_tools.keys():
                    tool_names.append(tool_name)
                attrs["pydantic_ai.tools"] = tool_names[:10]  # Limit to 10
                attrs["pydantic_ai.tools.count"] = len(tool_names)
            except Exception as e:
                logger.debug("Failed to extract tools: %s", e)

        # Extract result type if specified. `_result_type` in older releases,
        # `output_type` in current ones.
        try:
            result_type = getattr(instance, "_result_type", None)
            if not result_type:
                result_type = getattr(instance, "output_type", None)
            if result_type:
                attrs["pydantic_ai.result_type"] = str(result_type)
        except Exception as e:
            logger.debug("Failed to extract result type: %s", e)

        # Extract user prompt (first positional argument)
        user_prompt = None
        if len(args) > 0:
            user_prompt = args[0]
        elif "user_prompt" in kwargs:
            user_prompt = kwargs["user_prompt"]
        elif "prompt" in kwargs:
            user_prompt = kwargs["prompt"]

        if user_prompt:
            attrs["pydantic_ai.user_prompt"] = self._cap_content(user_prompt)

        # Extract message history if provided
        if "message_history" in kwargs:
            try:
                history = kwargs["message_history"]
                if history and hasattr(history, "__len__"):
                    attrs["pydantic_ai.message_history.count"] = len(history)
            except Exception as e:
                logger.debug("Failed to extract message history: %s", e)

        # Extract model settings if provided
        if "model_settings" in kwargs:
            try:
                settings = kwargs["model_settings"]
                if isinstance(settings, dict):
                    if "temperature" in settings:
                        attrs["gen_ai.request.temperature"] = settings["temperature"]
                    if "max_tokens" in settings:
                        attrs["gen_ai.request.max_tokens"] = settings["max_tokens"]
                    if "top_p" in settings:
                        attrs["gen_ai.request.top_p"] = settings["top_p"]
            except Exception as e:
                logger.debug("Failed to extract model settings: %s", e)

        return attrs

    @staticmethod
    def _usage_counts(usage: Any) -> Optional[Dict[str, int]]:
        """Token counts from a Pydantic AI usage object, or None if it carries none.

        ``input_tokens`` / ``output_tokens`` in current releases; ``request_tokens`` /
        ``response_tokens`` in older ones.
        """
        if usage is None:
            return None

        def number(*names):
            for name in names:
                value = _read(usage, name)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return value
            return None

        prompt = number("input_tokens", "request_tokens")
        completion = number("output_tokens", "response_tokens")
        total = number("total_tokens")
        if not prompt and not completion and not total:
            # Absent, or a usage object nothing has been counted into yet. Reported
            # as usage, zeros told every enclosing span its tokens were recorded.
            return None
        prompt = prompt or 0
        completion = completion or 0
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total if total is not None else prompt + completion,
        }

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from agent result.

        Args:
            result: The agent execution result.

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        # Try to extract usage from result
        try:
            # Pydantic AI results typically have usage information
            usage = _read(result, "usage")
            if usage is not _MISSING and usage:
                counts = self._usage_counts(usage)
                if counts is None and callable(usage):
                    # usage() was a method in older releases; it is a property now.
                    counts = self._usage_counts(usage())
                if counts is not None:
                    return counts

            # Try alternative attribute names
            usage = _read(result, "_usage")
            if usage is not _MISSING:
                return self._usage_counts(usage)

        except Exception as e:
            logger.debug("Failed to extract usage: %s", e)

        # Token usage is also captured by underlying provider instrumentors
        return None

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        """Extract response attributes from agent result.

        Args:
            result: The agent execution result.

        Returns:
            Dict[str, Any]: Dictionary of response attributes.
        """
        attrs = {}

        try:
            # Extract result data. `output` in current releases, `data` in older ones.
            data = _read(result, "output")
            if data is _MISSING:
                data = _read(result, "data")
            if data is not _MISSING:
                try:
                    # If data is a Pydantic model, convert to dict
                    if hasattr(data, "model_dump"):
                        data_dict = data.model_dump()
                        attrs["pydantic_ai.result.data"] = self._cap_content(data_dict)
                    else:
                        attrs["pydantic_ai.result.data"] = self._cap_content(data)
                except Exception as e:
                    logger.debug("Failed to extract result data: %s", e)

            # Extract messages from result
            messages = _read(result, "messages")
            if messages is not _MISSING and messages:
                try:
                    attrs["pydantic_ai.result.messages.count"] = len(messages)

                    # Extract last message content
                    if messages:
                        last_msg = messages[-1]
                        if hasattr(last_msg, "content"):
                            attrs["pydantic_ai.result.last_message"] = self._cap_content(
                                last_msg.content
                            )
                        if hasattr(last_msg, "role"):
                            attrs["pydantic_ai.result.last_role"] = last_msg.role
                except Exception as e:
                    logger.debug("Failed to extract messages: %s", e)
            elif messages is _MISSING and callable(_read(result, "all_messages")):
                # The message list is behind all_messages() in current releases.
                try:
                    attrs["pydantic_ai.result.messages.count"] = len(result.all_messages())
                except Exception as e:
                    logger.debug("Failed to count messages: %s", e)

            # Extract timestamp if available
            timestamp = _read(result, "timestamp")
            if timestamp is not _MISSING:
                try:
                    # A method before Pydantic AI 2.0, a property since. str() of the
                    # bound method embeds the repr of the result it belongs to, which
                    # for a stream is every message of the conversation.
                    if callable(timestamp):
                        timestamp = timestamp()
                    attrs["pydantic_ai.result.timestamp"] = str(timestamp)
                except Exception as e:
                    logger.debug("Failed to extract timestamp: %s", e)

            # Extract cost if available
            cost = _read(result, "cost")
            if cost is not _MISSING and cost:
                try:
                    attrs["pydantic_ai.result.cost"] = float(cost)
                except Exception as e:
                    logger.debug("Failed to extract cost: %s", e)

            # Extract model name from result if available. Current releases report
            # it on the final model response.
            model = _read(result, "model")
            if model is _MISSING:
                model = _read(_read(result, "response"), "model_name")
            if model is not _MISSING and model is not None:
                attrs["gen_ai.response.model"] = str(model)

        except Exception as e:
            logger.debug("Failed to extract response attributes: %s", e)

        return attrs

    def _extract_finish_reason(self, result) -> Optional[str]:
        """Extract finish reason from agent result.

        Args:
            result: The agent execution result.

        Returns:
            Optional[str]: The finish reason string or None if not available.
        """
        try:
            # Check if result has finish_reason
            if hasattr(result, "finish_reason"):
                return str(result.finish_reason)

            # Check messages for finish reason
            if hasattr(result, "messages") and result.messages:
                last_msg = result.messages[-1]
                if hasattr(last_msg, "finish_reason"):
                    return str(last_msg.finish_reason)

            # Current releases report it on the final model response
            finish_reason = _read(_read(result, "response"), "finish_reason")
            if finish_reason is not _MISSING and finish_reason:
                return str(finish_reason)

            # If we have data, assume completion
            if _read(result, "output") is not _MISSING or hasattr(result, "data"):
                return "completed"

        except Exception as e:
            logger.debug("Failed to extract finish reason: %s", e)

        return None
