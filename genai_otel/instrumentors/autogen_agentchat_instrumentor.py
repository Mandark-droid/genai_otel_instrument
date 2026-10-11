"""OpenTelemetry instrumentor for AutoGen AgentChat (v0.4+).

This instrumentor traces the newer AutoGen AgentChat framework which uses
a different API from the legacy pyautogen package. It instruments:
- BaseChatAgent.run() / run_stream() - Agent execution, for every agent class
- Team classes (RoundRobinGroupChat, SelectorGroupChat, Swarm) - Team execution
- on_messages() of each agent class (AssistantAgent, ...) - Message handling

Inside a team the participants are driven through on_messages_stream(), which is not
wrapped: their turns appear as the LLM spans beneath the team span, and as AutoGen's
own invoke_agent spans where autogen-core emits them (0.7 does, 0.4 does not).

Requirements:
    pip install autogen-agentchat>=0.4.0
"""

import contextvars
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional

import wrapt

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)

# The team whose traced run() is executing in the current context. BaseGroupChat.run()
# is implemented on top of its own run_stream(), so without this one team.run() call
# produced two identical nested spans.
_RUNNING_TEAM: contextvars.ContextVar = contextvars.ContextVar(
    "genai_otel_autogen_agentchat_running_team", default=None
)

# Collects the generator a traced *_stream() call created, so the wrapper the caller
# iterates can close it. Set only for the duration of that (synchronous) call.
_OPENED_STREAMS: contextvars.ContextVar = contextvars.ContextVar(
    "genai_otel_autogen_agentchat_opened_streams", default=None
)


def _traced_when_started(
    span_wrapper: Callable[[Callable], Callable], original: Callable
) -> Callable:
    """Trace ``original`` with ``span_wrapper``, opening the span when the call runs.

    The base wrapper starts its span, and makes it the current span, as soon as the
    wrapped function is called. An ``async def`` does nothing until it is awaited, so
    ``asyncio.gather(a.run(...), b.run(...))`` or ``asyncio.create_task(a.run(...))``
    attached both spans to the caller's context before either run started: the second
    run became a child of the first, each run's LLM calls were parented to the wrong
    span, and the caller was left with a finished span as its current span.

    Synchronous callables are traced as they are.
    """
    if inspect.iscoroutinefunction(original):

        @wrapt.decorator
        def when_awaited(wrapped, instance, args, kwargs):
            async def _call():
                return await wrapped(*args, **kwargs)

            return _call()

        return when_awaited(span_wrapper(original))

    if inspect.isasyncgenfunction(original):

        @wrapt.decorator
        def remembered(wrapped, instance, args, kwargs):
            stream = wrapped(*args, **kwargs)
            opened = _OPENED_STREAMS.get()
            if opened is not None:
                opened.append(stream)
            return stream

        @wrapt.decorator
        def when_iterated(wrapped, instance, args, kwargs):
            async def _iterate():
                opened: List[Any] = []
                token = _OPENED_STREAMS.set(opened)
                try:
                    stream = wrapped(*args, **kwargs)
                finally:
                    _OPENED_STREAMS.reset(token)
                try:
                    async for item in stream:
                        yield item
                finally:
                    # A caller that stops early closes this generator only. The base
                    # wrapper does not close the generator it iterates, so AutoGen's
                    # own one was left to the garbage collector: its cleanup (ending
                    # AutoGen's span, marking a team as no longer running) ran later
                    # in another task. Close innermost first, in this call, so each
                    # context is detached where it was attached.
                    try:
                        for inner in opened:
                            await inner.aclose()
                    finally:
                        await stream.aclose()

            return _iterate()

        return when_iterated(span_wrapper(remembered(original)))

    return span_wrapper(original)


def _marking_running_team(original: Callable) -> Callable:
    """Record the team in ``_RUNNING_TEAM`` while its ``run()`` executes."""
    if not inspect.iscoroutinefunction(original):
        return original

    @wrapt.decorator
    def run(wrapped, instance, args, kwargs):
        async def _call():
            token = _RUNNING_TEAM.set(instance)
            try:
                return await wrapped(*args, **kwargs)
            finally:
                _RUNNING_TEAM.reset(token)

        return _call()

    return run(original)


def _untraced_inside_own_run(traced: Callable, original: Callable) -> Callable:
    """Use ``original`` when the team's own traced ``run()`` is the caller."""

    @wrapt.decorator
    def run_stream(wrapped, instance, args, kwargs):
        if instance is None:
            return traced(*args, **kwargs)
        if _RUNNING_TEAM.get() is instance:
            return wrapped(*args, **kwargs)
        return traced.__get__(instance, type(instance))(*args, **kwargs)

    return run_stream(original)


def _classes_implementing(base: Any, method: str) -> List[Any]:
    """``base`` and its subclasses that define a concrete ``method`` of their own."""
    found: List[Any] = []
    seen = set()
    pending = [base]
    while pending:
        cls = pending.pop()
        if id(cls) in seen:
            continue
        seen.add(id(cls))
        impl = getattr(cls, "__dict__", {}).get(method)
        if callable(impl) and not getattr(impl, "__isabstractmethod__", False):
            found.append(cls)
        try:
            pending.extend(cls.__subclasses__())
        except (AttributeError, TypeError):
            pass
    return found


class AutoGenAgentChatInstrumentor(BaseInstrumentor):
    """Instrumentor for AutoGen AgentChat framework (v0.4+)"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._agentchat_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if autogen-agentchat library is available."""
        try:
            import autogen_agentchat  # noqa: F401

            self._agentchat_available = True
            logger.debug("autogen-agentchat library detected and available for instrumentation")
        except ImportError:
            logger.debug("autogen-agentchat library not installed, instrumentation will be skipped")
            self._agentchat_available = False

    def instrument(self, config: OTelConfig):
        """Instrument AutoGen AgentChat framework if available.

        Instruments:
        - BaseChatAgent.run() / run_stream() - Agent execution
        - BaseGroupChat.run() / run_stream() - Team execution
        - on_messages() of each agent class - Message handling

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._agentchat_available:
            logger.debug("Skipping AutoGen AgentChat instrumentation - library not available")
            return

        # Idempotency guard: repeated instrument() calls must not stack wrappers
        # on the shared ChatAgent / BaseGroupChat classes.
        if self._instrumented:
            logger.debug("AutoGen AgentChat already instrumented, skipping repeat instrument()")
            return

        self.config = config

        try:
            # Instrument agent run() / run_stream() / on_messages().
            # run() and run_stream() are implemented by agents.BaseChatAgent, which every
            # agent subclasses. base.ChatAgent only inherits the TaskRunner protocol stubs
            # and declares on_messages() abstract, so wrappers placed on it are shadowed
            # and never run. It stays as the fallback for a layout without BaseChatAgent.
            agent_cls: Any = None
            try:
                from autogen_agentchat.agents import BaseChatAgent

                agent_cls = BaseChatAgent
            except ImportError:
                try:
                    from autogen_agentchat.base import ChatAgent

                    agent_cls = ChatAgent
                except ImportError:
                    logger.debug("Could not import BaseChatAgent or ChatAgent")

            if agent_cls is not None:
                agent_run_wrapper = self.create_span_wrapper(
                    span_name="autogen_agentchat.agent.run",
                    extract_attributes=self._extract_agent_run_attributes,
                )
                agent_run_stream_wrapper = self.create_span_wrapper(
                    span_name="autogen_agentchat.agent.run_stream",
                    extract_attributes=self._extract_agent_run_attributes,
                )
                on_messages_wrapper = self.create_span_wrapper(
                    span_name="autogen_agentchat.agent.on_messages",
                    extract_attributes=self._extract_on_messages_attributes,
                )

                if hasattr(agent_cls, "run"):
                    agent_cls.run = _traced_when_started(agent_run_wrapper, agent_cls.run)
                    logger.debug("Instrumented %s.run()", agent_cls.__name__)

                if hasattr(agent_cls, "run_stream"):
                    agent_cls.run_stream = _traced_when_started(
                        agent_run_stream_wrapper, agent_cls.run_stream
                    )
                    logger.debug("Instrumented %s.run_stream()", agent_cls.__name__)

                # on_messages() is abstract on both base classes; each agent class
                # (AssistantAgent, UserProxyAgent, ...) brings its own. Only classes
                # that exist now can be wrapped: an agent class defined later is still
                # traced through run() / run_stream().
                for cls in _classes_implementing(agent_cls, "on_messages"):
                    cls.on_messages = _traced_when_started(on_messages_wrapper, cls.on_messages)
                    logger.debug("Instrumented %s.on_messages()", cls.__name__)

            # Instrument team classes via BaseGroupChat
            try:
                from autogen_agentchat.teams import BaseGroupChat

                team_run_wrapper = self.create_span_wrapper(
                    span_name="autogen_agentchat.team.run",
                    extract_attributes=self._extract_team_run_attributes,
                )
                team_run_stream_wrapper = self.create_span_wrapper(
                    span_name="autogen_agentchat.team.run_stream",
                    extract_attributes=self._extract_team_run_attributes,
                )

                if hasattr(BaseGroupChat, "run"):
                    BaseGroupChat.run = _traced_when_started(
                        team_run_wrapper, _marking_running_team(BaseGroupChat.run)
                    )
                    logger.debug("Instrumented BaseGroupChat.run()")

                if hasattr(BaseGroupChat, "run_stream"):
                    original = BaseGroupChat.run_stream
                    BaseGroupChat.run_stream = _untraced_inside_own_run(
                        _traced_when_started(team_run_stream_wrapper, original), original
                    )
                    logger.debug("Instrumented BaseGroupChat.run_stream()")
            except ImportError:
                logger.debug("Could not import autogen_agentchat.teams.BaseGroupChat")

            self._instrumented = True
            logger.info("AutoGen AgentChat instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument AutoGen AgentChat: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _begin_agent_budget(self, span, attrs: Dict[str, Any]) -> None:
        """Open no agent budget frame for AgentChat spans.

        The base wrapper closes a frame only on its synchronous path, and every
        AgentChat entry point is a coroutine or an async generator. A frame opened
        here was never closed: each ``agent.run()`` left two behind in the caller's
        context, and no consumption attribute was ever written from them. Drop this
        override once the base wrapper closes frames on its async paths.
        """

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

    @staticmethod
    def _model_name_of(agent: Any) -> Optional[str]:
        """Return the model an agent is configured with, or None.

        AssistantAgent keeps its client in the private ``_model_client``, and the
        autogen-ext clients (OpenAI, Azure OpenAI, Anthropic, Ollama) expose no
        ``model`` attribute: the name is in ``_create_args``. The public spellings
        are tried first.
        """
        model_client = getattr(agent, "model_client", None) or getattr(agent, "_model_client", None)
        if not model_client:
            return None
        if hasattr(model_client, "model"):
            return str(model_client.model)
        for holder in ("_create_args", "_raw_config"):
            settings = getattr(model_client, holder, None)
            if isinstance(settings, dict) and settings.get("model"):
                return str(settings["model"])
        return None

    def _extract_agent_run_attributes(
        self, instance: Any, args: Any, kwargs: Any
    ) -> Dict[str, Any]:
        """Extract attributes from ChatAgent.run() / run_stream() call.

        Args:
            instance: The agent instance.
            args: Positional arguments (task).
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "autogen_agentchat"
        attrs["gen_ai.operation.name"] = "agent.run"

        # Extract agent name and type
        if hasattr(instance, "name"):
            attrs["autogen_agentchat.agent.name"] = str(instance.name)
            # Cross-framework agent attribution per upstream proposal
            # semantic-conventions-genai#91.
            attrs["gen_ai.agent.name"] = str(instance.name)
        attrs["autogen_agentchat.agent.type"] = type(instance).__name__

        # Extract model info if available
        model_name = self._model_name_of(instance)
        if model_name:
            attrs["gen_ai.request.model"] = model_name

        # Extract task from first arg or kwarg
        task = None
        if args and len(args) > 0:
            task = args[0]
        else:
            task = kwargs.get("task")

        if task:
            if isinstance(task, str):
                attrs["autogen_agentchat.task"] = self._cap_content(task)
            elif isinstance(task, list):
                # List of BaseMessage objects
                attrs["autogen_agentchat.task.message_count"] = len(task)
                if task and hasattr(task[0], "content"):
                    attrs["autogen_agentchat.task"] = self._cap_content(task[0].content)

        # Extract produced_message_types if available
        if hasattr(instance, "produced_message_types"):
            try:
                msg_types = [t.__name__ for t in instance.produced_message_types]
                attrs["autogen_agentchat.produced_message_types"] = msg_types
            except Exception:
                pass

        return attrs

    def _extract_on_messages_attributes(
        self, instance: Any, args: Any, kwargs: Any
    ) -> Dict[str, Any]:
        """Extract attributes from ChatAgent.on_messages() call.

        Args:
            instance: The agent instance.
            args: Positional arguments (messages).
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "autogen_agentchat"
        attrs["gen_ai.operation.name"] = "agent.on_messages"

        # Extract agent info
        if hasattr(instance, "name"):
            attrs["autogen_agentchat.agent.name"] = str(instance.name)
            # Cross-framework agent attribution per upstream proposal
            # semantic-conventions-genai#91.
            attrs["gen_ai.agent.name"] = str(instance.name)
        attrs["autogen_agentchat.agent.type"] = type(instance).__name__

        # Extract message count
        messages = None
        if args and len(args) > 0:
            messages = args[0]
        else:
            messages = kwargs.get("messages")

        if messages:
            attrs["autogen_agentchat.message_count"] = len(messages)

        return attrs

    def _extract_team_run_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Extract attributes from BaseGroupChat.run() / run_stream() call.

        Args:
            instance: The team instance.
            args: Positional arguments (task).
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "autogen_agentchat"
        attrs["gen_ai.operation.name"] = "team.run"

        # Extract team type
        team_type = type(instance).__name__
        attrs["autogen_agentchat.team.type"] = team_type

        # Extract team name if available
        if hasattr(instance, "name"):
            attrs["autogen_agentchat.team.name"] = str(instance.name)

        # Extract participants
        if hasattr(instance, "_participants") and instance._participants:
            attrs["autogen_agentchat.team.participant_count"] = len(instance._participants)
            participant_names = []
            for p in instance._participants:
                if hasattr(p, "name"):
                    participant_names.append(str(p.name))
            if participant_names:
                attrs["autogen_agentchat.team.participants"] = participant_names[:10]

        # Extract task
        task = None
        if args and len(args) > 0:
            task = args[0]
        else:
            task = kwargs.get("task")

        if task and isinstance(task, str):
            attrs["autogen_agentchat.task"] = self._cap_content(task)

        # Extract termination condition
        if hasattr(instance, "_termination_condition") and instance._termination_condition:
            attrs["autogen_agentchat.team.termination_condition"] = type(
                instance._termination_condition
            ).__name__

        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from AgentChat execution result.

        Note: Token usage is captured by underlying LLM provider instrumentors.

        Args:
            result: The execution result (TaskResult).

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        return None

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        """Extract response attributes from execution result.

        Args:
            result: The execution result.

        Returns:
            Dict[str, Any]: Dictionary of response attributes.
        """
        attrs = {}

        try:
            # TaskResult has messages and stop_reason
            if hasattr(result, "messages"):
                attrs["autogen_agentchat.result.message_count"] = len(result.messages)

            if hasattr(result, "stop_reason") and result.stop_reason:
                attrs["autogen_agentchat.result.stop_reason"] = str(result.stop_reason)[:200]
        except Exception as e:
            logger.debug("Failed to extract response attributes: %s", e)

        return attrs

    def _extract_finish_reason(self, result) -> Optional[str]:
        """Extract finish reason from execution result.

        Args:
            result: The execution result.

        Returns:
            Optional[str]: The finish reason string or None.
        """
        if hasattr(result, "stop_reason") and result.stop_reason:
            return str(result.stop_reason)[:100]
        if result:
            return "completed"
        return None
