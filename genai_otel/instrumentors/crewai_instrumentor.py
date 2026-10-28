"""OpenTelemetry instrumentor for the CrewAI framework.

This instrumentor automatically traces crew execution, agents, tasks, and
collaborative workflows using the CrewAI multi-agent framework.

Context propagation is handled by the base class create_span_wrapper() which
sets each span as the current span in context, enabling proper parent-child
trace hierarchy. ThreadPoolExecutor is patched to propagate context across
threads used internally by CrewAI.

IMPORTANT: CrewAI has built-in telemetry that conflicts with OpenTelemetry.
Up to CrewAI 1.15.14 it installs its own TracerProvider as the global one while
CrewAI is being imported and creates its spans through the global tracer, so either
its spans land in the application's traces or the application's spans go to CrewAI's
collector, depending on who set the provider first. Later releases keep that
provider private and only export their own spans.

This instrumentor switches that telemetry off by setting CREWAI_DISABLE_TELEMETRY=true
before it imports CrewAI. That is the variable CrewAI reads: 0.130 and later have it,
0.86 does not, and for releases without it there is nothing to set. A value the
application has set itself is left alone.
"""

import contextvars
import functools
import inspect
import logging
import os
import time
import uuid
from typing import Any, Callable, Dict, Optional

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from ..agent_budget import extract_budget_attributes
from ..config import OTelConfig
from ..sessions import current_session_id, inbound_session_id
from .base import BaseInstrumentor, _context_with_usage_holder, content_capture_on

logger = logging.getLogger(__name__)

# The variable CrewAI's own telemetry reads.
_TELEMETRY_SWITCH = "CREWAI_DISABLE_TELEMETRY"

# Crew instances whose run is already being traced in the current context.
# Crew.kickoff_async() runs Crew.kickoff() on a worker thread and both are wrapped,
# so without this one crew run is reported as two nested crew spans.
_ACTIVE_CREWS: contextvars.ContextVar = contextvars.ContextVar(
    "genai_otel_crewai_active_crews", default=frozenset()
)

# Where a kickoff's own token usage is left on its result for _extract_usage().
_KICKOFF_USAGE_ATTR = "_genai_otel_kickoff_usage"
_USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens", "cached_prompt_tokens")


def _disable_crewai_telemetry() -> bool:
    """Set CrewAI's telemetry switch unless the application already chose a value.

    Returns:
        bool: True when this call set the variable.
    """
    if _TELEMETRY_SWITCH in os.environ:
        return False
    os.environ[_TELEMETRY_SWITCH] = "true"
    return True


def _count(value: Any) -> int:
    """A token counter as an int; anything else (a mock, None) counts as zero."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _forget_active_crew(token) -> None:
    """Stop treating a crew as running. Never raises into the caller.

    A kickoff coroutine that is closed from another context (a cancelled task
    finalised at loop shutdown) cannot reset a token made in its own.
    """
    try:
        _ACTIVE_CREWS.reset(token)
    except (ValueError, LookupError):
        pass


class CrewAIInstrumentor(BaseInstrumentor):
    """Instrumentor for CrewAI multi-agent framework"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._crewai_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if CrewAI library is available."""
        # CrewAI builds its telemetry exporter while the package is being imported,
        # so the switch has to be in the environment before the import below.
        switched = _disable_crewai_telemetry()
        try:
            import crewai

            self._crewai_available = True
            logger.debug("CrewAI library detected and available for instrumentation")
        except ImportError:
            if switched:
                os.environ.pop(_TELEMETRY_SWITCH, None)
            logger.debug("CrewAI library not installed, instrumentation will be skipped")
            self._crewai_available = False

    def instrument(self, config: OTelConfig):
        """Instrument CrewAI framework if available.

        Instruments all crew execution methods (sync and async):
        - Crew.kickoff() / kickoff_async() / akickoff() - Main execution
        - Crew.kickoff_for_each() / kickoff_for_each_async() / akickoff_for_each() - Batch
        - Task.execute_sync() / execute_async() / aexecute_sync() - Task execution
        - Agent.execute_task() / aexecute_task() - Agent task execution
        - concurrent.futures.ThreadPoolExecutor - Automatic context propagation

        The a-prefixed Task and Agent methods are the ones Crew.akickoff() runs.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._crewai_available:
            logger.debug("Skipping CrewAI instrumentation - library not available")
            return

        # Idempotency guard: repeated instrument() calls must not stack wrappers
        # on the shared Crew / Task / Agent classes or re-patch ThreadPoolExecutor.
        if self._instrumented:
            logger.debug("CrewAI already instrumented, skipping repeat instrument()")
            return

        self.config = config

        # Disable CrewAI's built-in telemetry to prevent conflicts with OpenTelemetry
        # CrewAI has its own telemetry system that interferes with OTel tracing.
        # CREWAI_TELEMETRY_OPT_OUT is what this library has always set, but CrewAI does
        # not read it (checked from 0.28 to 1.15.27); it stays so that nothing looking
        # for it sees a change.
        os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
        _disable_crewai_telemetry()
        switch = os.environ.get(_TELEMETRY_SWITCH, "")
        if switch.strip().lower() == "true":
            logger.info("Disabled CrewAI built-in telemetry (%s=true)", _TELEMETRY_SWITCH)
        else:
            logger.info(
                "CrewAI built-in telemetry left as the application set it (%s=%s)",
                _TELEMETRY_SWITCH,
                switch,
            )

        try:
            import crewai

            self._warn_if_crewai_telemetry_is_global(crewai)

            # Create span wrappers for each operation type
            crew_wrapper = self.create_span_wrapper(
                span_name="crewai.crew.execution",
                extract_attributes=self._extract_crew_attributes,
            )
            task_wrapper = self.create_span_wrapper(
                span_name="crewai.task.execution",
                extract_attributes=self._extract_task_attributes,
            )
            agent_wrapper = self.create_span_wrapper(
                span_name="crewai.agent.execution",
                extract_attributes=self._extract_agent_attributes,
            )

            # Instrument all Crew kickoff methods (sync, async, and batch variants)
            if hasattr(crewai, "Crew"):
                for method_name in [
                    "kickoff",
                    "kickoff_async",
                    "akickoff",
                    "kickoff_for_each",
                    "kickoff_for_each_async",
                    "akickoff_for_each",
                ]:
                    if hasattr(crewai.Crew, method_name):
                        original = getattr(crewai.Crew, method_name)
                        setattr(
                            crewai.Crew,
                            method_name,
                            self._traced_crew_method(original, crew_wrapper),
                        )
                        logger.debug("Instrumented Crew.%s()", method_name)

            # Instrument Task execution methods
            if hasattr(crewai, "Task"):
                for method_name in ["execute_sync", "aexecute_sync"]:
                    if hasattr(crewai.Task, method_name):
                        original = getattr(crewai.Task, method_name)
                        setattr(
                            crewai.Task, method_name, self._traced_method(original, task_wrapper)
                        )
                        logger.debug("Instrumented Task.%s()", method_name)
                # execute_async hands back a Future, so its span is ended by the Future
                if hasattr(crewai.Task, "execute_async"):
                    crewai.Task.execute_async = self._traced_future_method(
                        crewai.Task.execute_async,
                        "crewai.task.execution",
                        self._extract_task_attributes,
                    )
                    logger.debug("Instrumented Task.execute_async()")

            # Instrument Agent execution
            if hasattr(crewai, "Agent"):
                for method_name in ["execute_task", "aexecute_task"]:
                    if hasattr(crewai.Agent, method_name):
                        original = getattr(crewai.Agent, method_name)
                        setattr(
                            crewai.Agent, method_name, self._traced_method(original, agent_wrapper)
                        )
                        logger.debug("Instrumented Agent.%s()", method_name)

            # Patch ThreadPoolExecutor.submit to propagate context automatically
            # This ensures any threaded execution by CrewAI maintains trace context
            self._patch_thread_pool_executor()

            self._instrumented = True
            logger.info("CrewAI instrumentation enabled with automatic context propagation")

        except Exception as e:
            logger.error("Failed to instrument CrewAI: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _warn_if_crewai_telemetry_is_global(self, crewai_module: Any) -> None:
        """Say so when CrewAI's telemetry provider is the process-wide TracerProvider.

        CrewAI up to 1.15.14 installs its telemetry provider globally while the package
        is imported. An application that imports CrewAI before calling instrument()
        has therefore already lost the set-once race: every span this library creates
        is exported to CrewAI's collector and none reaches the application's exporter.
        Switching CrewAI's telemetry off afterwards does not undo that.
        """
        try:
            telemetry = getattr(getattr(crewai_module, "telemetry", None), "Telemetry", None)
            provider = getattr(getattr(telemetry, "_instance", None), "provider", None)
            if provider is not None and provider is trace.get_tracer_provider():
                logger.warning(
                    "CrewAI's telemetry TracerProvider is the global OpenTelemetry provider: "
                    "spans from this library are being exported to CrewAI's telemetry "
                    "collector, not to your exporter. This CrewAI release installs its "
                    "provider when it is imported, and it was imported before "
                    "genai_otel.instrument(). Call genai_otel.instrument() before importing "
                    "crewai, set %s=true in the environment before the process starts, or "
                    "upgrade CrewAI to 1.15.15 or later.",
                    _TELEMETRY_SWITCH,
                )
        except Exception as e:
            logger.debug("Could not inspect CrewAI's telemetry provider: %s", e)

    def _traced_method(self, original: Callable, span_wrapper: Callable) -> Callable:
        """Wrap one CrewAI method in a span.

        For an ``async def`` the span is opened when the coroutine runs, not when it is
        created. The base wrapper attaches the span's context as soon as it is called
        and detaches it once the awaited call has finished; applied directly to a
        coroutine function those two happen in different contexts whenever the
        coroutine is handed to ``asyncio.run`` or ``create_task``. The detach then
        fails ("Failed to detach context") and the span stays current in the caller,
        so whatever the caller does next is parented to a span that has ended.
        """
        traced = span_wrapper(original)
        if not inspect.iscoroutinefunction(original):
            return traced

        @functools.wraps(original)
        async def method(instance, *args, **kwargs):
            return await traced.__get__(instance, type(instance))(*args, **kwargs)

        return method

    def _traced_crew_method(self, original: Callable, span_wrapper: Callable) -> Callable:
        """Wrap one Crew entry point: one span per crew run, carrying that run's usage.

        A crew whose run is already being traced in this context is called straight
        through, which is what keeps kickoff_async() -> kickoff() from producing a
        second crew span. kickoff_for_each() runs copies of the crew, so each copy
        still gets its own span under the batch span.
        """
        is_async = inspect.iscoroutinefunction(original)
        traced = span_wrapper(self._measuring_usage(original, is_async))

        if is_async:

            @functools.wraps(original)
            async def method(crew, *args, **kwargs):
                active = _ACTIVE_CREWS.get()
                if id(crew) in active:
                    return await original(crew, *args, **kwargs)
                token = _ACTIVE_CREWS.set(active | {id(crew)})
                try:
                    return await traced.__get__(crew, type(crew))(*args, **kwargs)
                finally:
                    _forget_active_crew(token)

        else:

            @functools.wraps(original)
            def method(crew, *args, **kwargs):
                active = _ACTIVE_CREWS.get()
                if id(crew) in active:
                    return original(crew, *args, **kwargs)
                token = _ACTIVE_CREWS.set(active | {id(crew)})
                try:
                    return traced.__get__(crew, type(crew))(*args, **kwargs)
                finally:
                    _forget_active_crew(token)

        return method

    def _measuring_usage(self, original: Callable, is_async: bool) -> Callable:
        """Return ``original`` with each run's own token usage attached to its result."""
        if is_async:

            @functools.wraps(original)
            async def method(crew, *args, **kwargs):
                before = self._llm_counters(crew)
                result = await original(crew, *args, **kwargs)
                self._attach_kickoff_usage(result, crew, before)
                return result

        else:

            @functools.wraps(original)
            def method(crew, *args, **kwargs):
                before = self._llm_counters(crew)
                result = original(crew, *args, **kwargs)
                self._attach_kickoff_usage(result, crew, before)
                return result

        return method

    def _llm_counters(self, crew: Any) -> Optional[Dict[int, Dict[str, int]]]:
        """Read the lifetime token counters of each distinct LLM object the crew uses.

        Returns:
            Counters keyed by LLM object identity, or None when an agent's LLM keeps
            none (CrewAI before 1.0), in which case CrewOutput.token_usage is the only
            figure there is.
        """
        try:
            agents = list(getattr(crew, "agents", None) or [])
            manager = getattr(crew, "manager_agent", None)
            if manager is not None:
                agents.append(manager)
            llms = [getattr(agent, "llm", None) for agent in agents]
            if not llms or not all(
                callable(getattr(llm, "get_token_usage_summary", None)) for llm in llms
            ):
                return None
            # The manager's LLM exists before the manager agent does (hierarchical process)
            manager_llm = getattr(crew, "manager_llm", None)
            if callable(getattr(manager_llm, "get_token_usage_summary", None)):
                llms.append(manager_llm)

            counters: Dict[int, Dict[str, int]] = {}
            for llm in llms:
                # Keyed by the counter, not the LLM object: Crew.copy() gives every
                # copied agent a shallow copy of its LLM, and the copies share one
                # counter. Counting per object reported it once for each copy.
                key = id(getattr(llm, "_token_usage", None) or llm)
                if key not in counters:
                    summary = llm.get_token_usage_summary()
                    counters[key] = {
                        name: _count(getattr(summary, name, 0)) for name in _USAGE_FIELDS
                    }
            return counters
        except Exception as e:
            logger.debug("Failed to read CrewAI LLM token counters: %s", e)
            return None

    def _attach_kickoff_usage(
        self, result: Any, crew: Any, before: Optional[Dict[int, Dict[str, int]]]
    ) -> None:
        """Leave what this run used on its result, where _extract_usage() looks first.

        CrewOutput.token_usage adds up, over the crew's agents, each agent's LLM
        lifetime counters. It therefore grows with every kickoff of the same crew, and
        an LLM object shared by several agents is added once per agent. The change in
        each distinct LLM's counters across the call is what this run used. Runs that
        share an LLM object and overlap in time are counted into each other; no
        snapshot can tell them apart.
        """
        if before is None:
            return
        try:
            if not hasattr(result, "token_usage"):
                return
            after = self._llm_counters(crew)
            if after is None:
                return
            used = dict.fromkeys(_USAGE_FIELDS, 0)
            for key, counters in after.items():
                start = before.get(key, {})
                for name in _USAGE_FIELDS:
                    used[name] += max(counters[name] - start.get(name, 0), 0)
            setattr(result, _KICKOFF_USAGE_ATTR, used)
        except Exception as e:
            logger.debug("Failed to measure CrewAI kickoff usage: %s", e)

    def _traced_future_method(
        self,
        original: Callable,
        span_name: str,
        extract_attributes: Callable[[Any, Any, Any], Dict[str, Any]],
    ) -> Callable:
        """Wrap a method that returns a ``concurrent.futures.Future`` straight away.

        Task.execute_async() starts the work on a thread and returns its Future. The
        base wrapper ends the span when the method returns, which is before the work
        has started: the task span lasted a few milliseconds, had no output, and was OK
        whatever happened on the thread. Here the Future ends the span instead.
        """

        @functools.wraps(original)
        def method(instance, *args, **kwargs):
            if not self._instrumented:
                return original(instance, *args, **kwargs)
            try:
                span = self.tracer.start_span(
                    span_name,
                    attributes=self._initial_attributes(
                        span_name, extract_attributes, instance, args, kwargs
                    ),
                )
                # Current while the thread is started: CrewAI copies the context into
                # it, which is what parents the agent and LLM spans to this one.
                token = otel_context.attach(
                    _context_with_usage_holder(span, trace.set_span_in_context(span))
                )
            except Exception as e:
                logger.debug("Could not start span '%s': %s", span_name, e)
                return original(instance, *args, **kwargs)

            start_time = time.time()
            try:
                result = original(instance, *args, **kwargs)
            except Exception as e:
                self._end_span(span, start_time, kwargs, error=e)
                raise
            finally:
                otel_context.detach(token)

            add_done_callback = getattr(result, "add_done_callback", None)
            if callable(add_done_callback):
                add_done_callback(
                    lambda future: self._end_span(span, start_time, kwargs, future=future)
                )
            else:
                self._end_span(span, start_time, kwargs, result=result)
            return result

        return method

    def _initial_attributes(
        self,
        span_name: str,
        extract_attributes: Callable[[Any, Any, Any], Dict[str, Any]],
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> Dict[str, Any]:
        """Build span-start attributes the way the base wrapper does."""
        attrs: Dict[str, Any] = {}
        try:
            extracted = self._with_provider_aliases(extract_attributes(instance, args, kwargs))
            for key, value in extracted.items():
                attrs[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
        except Exception as e:
            logger.warning("Failed to extract attributes for span '%s': %s", span_name, e)

        for key, extractor in (
            ("session.id", getattr(self.config, "session_id_extractor", None)),
            ("user.id", getattr(self.config, "user_id_extractor", None)),
        ):
            if not extractor:
                continue
            try:
                value = extractor(instance, args, kwargs)
                if value:
                    attrs[key] = value
                    if key == "session.id":
                        attrs["gen_ai.conversation.id"] = value
            except Exception as e:
                logger.debug("Failed to extract %s: %s", key, e)
        return attrs

    def _end_span(
        self,
        span: Any,
        start_time: float,
        kwargs: Any,
        result: Any = None,
        future: Any = None,
        error: Optional[BaseException] = None,
    ) -> None:
        """Record the outcome on a span opened by _traced_future_method() and end it."""
        try:
            if future is not None:
                try:
                    error = future.exception()
                except BaseException as e:  # noqa: BLE001 - a cancelled Future raises
                    error = e
                if error is None:
                    result = future.result()
            if not span.is_recording():
                return
            if error is None:
                if self.request_counter:
                    self.request_counter.add(1, {"operation": span.name})
                self._record_result_metrics(span, result, start_time, kwargs)
                span.set_status(Status(StatusCode.OK))
            else:
                if self.error_counter:
                    self.error_counter.add(
                        1, {"operation": span.name, "error_type": type(error).__name__}
                    )
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.record_exception(error)
        except Exception as e:
            logger.debug("Failed to record the outcome of a CrewAI span: %s", e)
        finally:
            span.end()

    def _patch_thread_pool_executor(self):
        """Patch ThreadPoolExecutor to automatically propagate OpenTelemetry context.

        This ensures that any code using ThreadPoolExecutor (including CrewAI internally)
        will automatically propagate trace context to worker threads.

        NOTE: this patches the stdlib ``concurrent.futures.ThreadPoolExecutor.submit``
        process-wide, so it affects every thread pool in the host application, not
        only CrewAI's. To keep that blast radius safe:

        - a double-wrap guard prevents stacking the wrapper on repeat/parallel
          instrument() calls; and
        - the wrapper is fully defensive: if OTel context capture/attach fails for
          any reason, the original callable is submitted and executed unchanged, so
          an unrelated submit is never broken and never double-submitted.
        """
        try:
            from concurrent.futures import ThreadPoolExecutor

            original_submit = ThreadPoolExecutor.submit

            # Double-wrap guard: if submit was already patched by this library
            # (this instance, or another CrewAI instrumentor in the same process),
            # do not wrap it again.
            if getattr(original_submit, "_genai_otel_wrapped", False):
                logger.debug("ThreadPoolExecutor.submit already patched, skipping")
                return

            def context_propagating_submit(self, fn, /, *args, **kwargs):
                """Submit with automatic context propagation.

                Never breaks an unrelated submit: on any failure capturing or
                attaching context, the original callable is submitted unchanged.
                ``original_submit`` is called exactly once on every path.
                """
                try:
                    ctx = otel_context.get_current()
                except Exception:
                    # Could not capture context; submit the callable unchanged.
                    return original_submit(self, fn, *args, **kwargs)

                def wrapper():
                    # The business callable must always run, even if context
                    # attach fails in the worker thread.
                    try:
                        token = otel_context.attach(ctx)
                    except Exception:
                        return fn(*args, **kwargs)
                    try:
                        return fn(*args, **kwargs)
                    finally:
                        otel_context.detach(token)

                return original_submit(self, wrapper)

            context_propagating_submit._genai_otel_wrapped = True
            ThreadPoolExecutor.submit = context_propagating_submit
            logger.debug("Patched ThreadPoolExecutor.submit() for automatic context propagation")

        except Exception as e:
            logger.debug(f"Could not patch ThreadPoolExecutor: {e}")
            # Not critical - continue with other instrumentation

    def _cap_content(self, text: Any, default: int = 200) -> str:
        """Cap captured content per ``config.content_max_length``.

        Content capture is a required audit feature, so content keeps flowing;
        this only bounds its size. ``content_max_length == 0`` (or missing) means
        unlimited. When no config is present (unit tests) the historical default
        cap is used.
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

    def _extract_task_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Extract attributes from Task execution.

        Args:
            instance: The Task instance.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "crewai"
        attrs["gen_ai.operation.name"] = "task.execution"

        try:
            # Extract task description
            if hasattr(instance, "description"):
                attrs["crewai.task.description"] = str(instance.description)[:200]

            # Extract task expected_output
            if hasattr(instance, "expected_output"):
                attrs["crewai.task.expected_output"] = str(instance.expected_output)[:200]

            # Extract assigned agent role if available
            if hasattr(instance, "agent") and hasattr(instance.agent, "role"):
                attrs["crewai.task.agent_role"] = str(instance.agent.role)
                # CrewAI agents are identified by `role` (no separate `name`
                # attribute in v0.x). Cross-framework attribution per
                # semantic-conventions-genai#91 uses role as the canonical name.
                attrs["gen_ai.agent.name"] = str(instance.agent.role)

            # Extract task ID if available
            if hasattr(instance, "id"):
                attrs["crewai.task.id"] = str(instance.id)

            # Propagate session.id from parent crew instance
            # Task doesn't have a direct 'crew' attr, but task.agent.crew does
            crew = getattr(instance, "crew", None)
            if not crew:
                agent = getattr(instance, "agent", None)
                if agent:
                    crew = getattr(agent, "crew", None)
            if crew:
                session_id = getattr(crew, "_genai_otel_session_id", None)
                if session_id:
                    attrs["session.id"] = session_id
                    # Cross-framework conversation correlation per upstream
                    # proposal semantic-conventions-genai#145.
                    attrs["gen_ai.conversation.id"] = session_id

        except Exception as e:
            logger.debug("Failed to extract task attributes: %s", e)

        return attrs

    def _extract_agent_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Extract attributes from Agent execution.

        Args:
            instance: The Agent instance.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "crewai"
        attrs["gen_ai.operation.name"] = "agent.execution"
        # CrewAI puts max_iter / max_tokens on the Agent itself.
        attrs.update(extract_budget_attributes(kwargs, instance))

        try:
            # Extract agent role
            if hasattr(instance, "role"):
                attrs["crewai.agent.role"] = str(instance.role)
                # Same canonical name as the task span (role), so the agent
                # and the task it runs form one node in the agent graph.
                attrs["gen_ai.agent.name"] = str(instance.role)

            # Extract agent goal
            if hasattr(instance, "goal"):
                attrs["crewai.agent.goal"] = str(instance.goal)[:200]

            # Extract agent backstory
            if hasattr(instance, "backstory"):
                attrs["crewai.agent.backstory"] = str(instance.backstory)[:200]

            # Extract LLM model if available
            if hasattr(instance, "llm") and hasattr(instance.llm, "model_name"):
                attrs["crewai.agent.llm_model"] = instance.llm.model_name
            elif hasattr(instance, "llm") and hasattr(instance.llm, "model"):
                attrs["crewai.agent.llm_model"] = instance.llm.model

            # Propagate session.id from parent crew instance
            crew = getattr(instance, "crew", None)
            if crew:
                session_id = getattr(crew, "_genai_otel_session_id", None)
                if session_id:
                    attrs["session.id"] = session_id
                    # Cross-framework conversation correlation per upstream
                    # proposal semantic-conventions-genai#145.
                    attrs["gen_ai.conversation.id"] = session_id

        except Exception as e:
            logger.debug("Failed to extract agent attributes: %s", e)

        return attrs

    def _extract_crew_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Extract attributes from Crew.kickoff() call.

        Args:
            instance: The Crew instance.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "crewai"
        attrs["gen_ai.operation.name"] = "crew.execution"

        # Extract LLM model from agents for cost enrichment.
        # agent.llm can be a string (model name) or a BaseLLM/LLM object with .model attr.
        try:
            model_name = self._extract_model_from_crew(instance)
            if model_name:
                attrs["gen_ai.request.model"] = model_name
        except Exception as e:
            logger.debug("Failed to extract model from crew: %s", e)

        # Extract crew ID if available
        if hasattr(instance, "id"):
            attrs["crewai.crew.id"] = str(instance.id)

        # Extract crew name if available
        if hasattr(instance, "name"):
            attrs["crewai.crew.name"] = instance.name

        # Extract process type (sequential, hierarchical, etc.)
        if hasattr(instance, "process"):
            process = instance.process
            # Process might be an enum or string
            process_type = str(process).split(".")[-1] if hasattr(process, "name") else str(process)
            attrs["crewai.process.type"] = process_type

        # Extract agents
        if hasattr(instance, "agents") and instance.agents:
            try:
                agent_count = len(instance.agents)
                attrs["crewai.agent_count"] = agent_count

                # Extract agent roles
                agent_roles = []
                agent_goals = []
                for agent in instance.agents:
                    if hasattr(agent, "role"):
                        agent_roles.append(str(agent.role)[:100])  # Truncate long roles
                    if hasattr(agent, "goal"):
                        agent_goals.append(str(agent.goal)[:100])  # Truncate long goals

                if agent_roles:
                    attrs["crewai.agent.roles"] = agent_roles
                if agent_goals:
                    attrs["crewai.agent.goals"] = agent_goals

                # Extract tools from agents
                all_tools = []
                for agent in instance.agents:
                    if hasattr(agent, "tools") and agent.tools:
                        for tool in agent.tools:
                            tool_name = str(getattr(tool, "name", type(tool).__name__))
                            if tool_name and tool_name not in all_tools:
                                all_tools.append(tool_name)

                if all_tools:
                    attrs["crewai.tools"] = all_tools[:10]  # Limit to 10 tools
                    attrs["crewai.tool_count"] = len(all_tools)

            except Exception as e:
                logger.debug("Failed to extract agent information: %s", e)

        # Extract tasks
        if hasattr(instance, "tasks") and instance.tasks:
            try:
                task_count = len(instance.tasks)
                attrs["crewai.task_count"] = task_count

                # Extract task descriptions (truncated)
                task_descriptions = []
                for task in instance.tasks:
                    if hasattr(task, "description"):
                        desc = str(task.description)[:100]  # Truncate
                        task_descriptions.append(desc)

                if task_descriptions:
                    attrs["crewai.task.descriptions"] = task_descriptions

            except Exception as e:
                logger.debug("Failed to extract task information: %s", e)

        # Extract verbose setting
        if hasattr(instance, "verbose"):
            attrs["crewai.verbose"] = instance.verbose

        # Extract inputs passed to kickoff (first positional arg or 'inputs' kwarg)
        inputs = None
        if len(args) > 0:
            inputs = args[0]
        elif "inputs" in kwargs:
            inputs = kwargs["inputs"]

        if inputs:
            try:
                if isinstance(inputs, dict):
                    # Store input keys and truncated values
                    input_keys = list(inputs.keys())
                    attrs["crewai.inputs.keys"] = input_keys

                    # Store input values (bounded via content_max_length)
                    for key, value in list(inputs.items())[:5]:  # Limit to 5 inputs
                        attrs[f"crewai.inputs.{key}"] = self._cap_content(value)
                else:
                    # Non-dict inputs
                    attrs["crewai.inputs"] = self._cap_content(inputs)
            except Exception as e:
                logger.debug("Failed to extract inputs: %s", e)

        # Extract manager agent for hierarchical process
        if hasattr(instance, "manager_agent") and instance.manager_agent:
            try:
                if hasattr(instance.manager_agent, "role"):
                    attrs["crewai.manager.role"] = str(instance.manager_agent.role)[:100]
            except Exception as e:
                logger.debug("Failed to extract manager agent: %s", e)

        # --- Session ID extraction (backends group traces into sessions by it) ---
        session_id = None

        # Priority 1: App-provided session_id in kickoff inputs
        if isinstance(inputs, dict):
            session_id = inputs.get("session_id") or inputs.get("session.id")

        # Priority 2: an enclosing genai_otel.session() block
        if not session_id:
            session_id = current_session_id()

        # Priority 3: OTelConfig.session_id_extractor callable
        if not session_id and self.config and self.config.session_id_extractor:
            try:
                session_id = self.config.session_id_extractor(instance, args, kwargs)
            except Exception as e:
                logger.debug("Failed to extract session ID via extractor: %s", e)

        # Priority 4: a session an upstream service sent, when accepted. It ranks
        # above the crew id, which CrewAI generates rather than the application.
        if not session_id:
            session_id = inbound_session_id()

        # Priority 5: Crew instance ID (CrewAI assigns a UUID to each Crew)
        if not session_id and hasattr(instance, "id"):
            session_id = str(instance.id)

        # Priority 6: Auto-generate a new UUID for this kickoff
        if not session_id:
            session_id = str(uuid.uuid4())

        attrs["session.id"] = session_id
        attrs["crewai.session.id"] = session_id
        # Cross-framework conversation correlation per upstream proposal
        # semantic-conventions-genai#145.
        attrs["gen_ai.conversation.id"] = session_id

        # Store on instance for child span access (cross-thread safe)
        instance._genai_otel_session_id = session_id

        return attrs

    def _extract_model_from_crew(self, instance: Any) -> Optional[str]:
        """Extract the LLM model name from a Crew instance.

        Iterates over agents to find the first available model name.
        agent.llm can be a string (model name) or a BaseLLM/LLM object with .model attr.

        Args:
            instance: The Crew instance.

        Returns:
            The model name string, or None if not found.
        """
        agents = getattr(instance, "agents", None) or []
        for agent in agents:
            llm = getattr(agent, "llm", None)
            if llm is None:
                continue
            # agent.llm can be a plain string (e.g. "openai/gpt-4o")
            if isinstance(llm, str):
                return llm
            # Or a BaseLLM / LLM object with a .model attribute
            model = getattr(llm, "model", None)
            if model:
                return str(model)
        # Fallback: check manager_agent (hierarchical process)
        manager = getattr(instance, "manager_agent", None)
        if manager:
            llm = getattr(manager, "llm", None)
            if isinstance(llm, str):
                return llm
            model = getattr(llm, "model", None) if llm else None
            if model:
                return str(model)
        return None

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from crew execution result.

        CrewAI's CrewOutput.token_usage is a UsageMetrics pydantic model with:
        total_tokens, prompt_tokens, cached_prompt_tokens, completion_tokens,
        successful_requests. CrewAI aggregates these internally from each agent's
        token_process, so usage is available even when individual LLM provider
        responses don't include it.

        From CrewAI 1.0 that aggregate is built from counters that live on the LLM
        objects for their whole lifetime, so it is not the usage of one kickoff. When
        the run was measured (see _attach_kickoff_usage) that measurement is used
        instead.

        Args:
            result: The crew execution result (CrewOutput).

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        if hasattr(result, "token_usage"):
            try:
                usage = result.token_usage
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                total_tokens = getattr(usage, "total_tokens", 0) or 0
                cached_prompt_tokens = getattr(usage, "cached_prompt_tokens", 0) or 0

                measured = getattr(result, _KICKOFF_USAGE_ATTR, None)
                if isinstance(measured, dict):
                    prompt_tokens = measured["prompt_tokens"]
                    completion_tokens = measured["completion_tokens"]
                    total_tokens = measured["total_tokens"]
                    cached_prompt_tokens = measured["cached_prompt_tokens"]

                # Only return if there are actual tokens recorded
                if total_tokens > 0 or prompt_tokens > 0 or completion_tokens > 0:
                    usage_dict = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    }
                    # Include cached tokens for Anthropic cache cost calculation
                    if cached_prompt_tokens > 0:
                        usage_dict["cache_read_input_tokens"] = cached_prompt_tokens
                    return usage_dict
            except Exception as e:
                logger.debug("Failed to extract token usage: %s", e)
        return None

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        """Extract response attributes from crew execution result.

        Args:
            result: The crew execution result.

        Returns:
            Dict[str, Any]: Dictionary of response attributes.
        """
        attrs = {}

        # CrewAI result can be a string or a CrewOutput object
        try:
            # If result is a string (common case)
            if isinstance(result, str):
                attrs["crewai.output"] = self._cap_content(result)
                attrs["crewai.output_length"] = len(result)

            # If result is a CrewOutput object
            elif hasattr(result, "raw"):
                attrs["crewai.output"] = self._cap_content(result.raw)
                attrs["crewai.output_length"] = len(str(result.raw))

            # An agent given a response model returns the model itself (CrewAI 1.x)
            elif callable(getattr(result, "model_dump_json", None)):
                dumped = result.model_dump_json()
                attrs["crewai.output"] = self._cap_content(dumped)
                attrs["crewai.output_length"] = len(dumped)

            # If result has tasks_output attribute (list of task results)
            if hasattr(result, "tasks_output"):
                try:
                    tasks_output = result.tasks_output
                    if tasks_output:
                        attrs["crewai.tasks_completed"] = len(tasks_output)

                        # Extract output from each task
                        task_outputs = []
                        for idx, task_output in enumerate(tasks_output[:5]):  # Limit to 5 tasks
                            if hasattr(task_output, "raw"):
                                task_outputs.append(self._cap_content(task_output.raw))

                        if task_outputs:
                            attrs["crewai.task_outputs"] = task_outputs
                except Exception as e:
                    logger.debug("Failed to extract tasks_output: %s", e)

            # If result has json attribute. CrewOutput.json and TaskOutput.json raise
            # unless the task asked for JSON, and on a bare pydantic model `json` is
            # a method rather than the output.
            json_output = self._read_output_field(result, "json")
            if json_output is not None and not callable(json_output):
                attrs["crewai.output.json"] = str(json_output)[:500]

            # If result has pydantic attribute
            # The structured output is the model's answer: content, recorded only
            # on request.
            pydantic_output = self._read_output_field(result, "pydantic")
            if pydantic_output is not None and content_capture_on(getattr(self, "config", None)):
                attrs["crewai.output.pydantic"] = str(pydantic_output)[:500]

        except Exception as e:
            logger.debug("Failed to extract response attributes: %s", e)

        return attrs

    @staticmethod
    def _read_output_field(result: Any, name: str) -> Any:
        """Read an optional field of a CrewAI result; a property that raises is None."""
        try:
            return getattr(result, name, None)
        except Exception:
            # Expected for every plain-text result, so not worth a log line
            return None

    def _extract_finish_reason(self, result) -> Optional[str]:
        """Extract finish reason from crew execution result.

        Args:
            result: The crew execution result.

        Returns:
            Optional[str]: The finish reason string or None if not available.
        """
        # CrewAI doesn't typically provide a finish_reason
        # We could infer completion status
        if result:
            return "completed"
        return None
