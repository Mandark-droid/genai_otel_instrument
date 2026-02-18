"""OpenTelemetry instrumentor for AutoGen AgentChat (v0.4+).

This instrumentor traces the newer AutoGen AgentChat framework which uses
a different API from the legacy pyautogen package. It instruments:
- AssistantAgent.run() / run_stream() - Agent execution
- Team classes (RoundRobinGroupChat, SelectorGroupChat, Swarm) - Team execution
- BaseChatAgent.on_messages() - Message handling

Requirements:
    pip install autogen-agentchat>=0.4.0
"""

import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


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
        - BaseChatAgent.on_messages() / on_messages_stream() - Message handling

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._agentchat_available:
            logger.debug("Skipping AutoGen AgentChat instrumentation - library not available")
            return

        self.config = config

        try:
            # Instrument BaseChatAgent.run() and run_stream()
            try:
                from autogen_agentchat.base import ChatAgent

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

                if hasattr(ChatAgent, "run"):
                    ChatAgent.run = agent_run_wrapper(ChatAgent.run)
                    logger.debug("Instrumented ChatAgent.run()")

                if hasattr(ChatAgent, "run_stream"):
                    ChatAgent.run_stream = agent_run_stream_wrapper(ChatAgent.run_stream)
                    logger.debug("Instrumented ChatAgent.run_stream()")

                if hasattr(ChatAgent, "on_messages"):
                    ChatAgent.on_messages = on_messages_wrapper(ChatAgent.on_messages)
                    logger.debug("Instrumented ChatAgent.on_messages()")
            except ImportError:
                logger.debug("Could not import autogen_agentchat.base.ChatAgent")

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
                    BaseGroupChat.run = team_run_wrapper(BaseGroupChat.run)
                    logger.debug("Instrumented BaseGroupChat.run()")

                if hasattr(BaseGroupChat, "run_stream"):
                    BaseGroupChat.run_stream = team_run_stream_wrapper(BaseGroupChat.run_stream)
                    logger.debug("Instrumented BaseGroupChat.run_stream()")
            except ImportError:
                logger.debug("Could not import autogen_agentchat.teams.BaseGroupChat")

            self._instrumented = True
            logger.info("AutoGen AgentChat instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument AutoGen AgentChat: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

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
        attrs["autogen_agentchat.agent.type"] = type(instance).__name__

        # Extract model info if available
        if hasattr(instance, "model_client") and instance.model_client:
            model_client = instance.model_client
            if hasattr(model_client, "model"):
                attrs["gen_ai.request.model"] = str(model_client.model)

        # Extract task from first arg or kwarg
        task = None
        if args and len(args) > 0:
            task = args[0]
        else:
            task = kwargs.get("task")

        if task:
            if isinstance(task, str):
                attrs["autogen_agentchat.task"] = task[:200]
            elif isinstance(task, list):
                # List of BaseMessage objects
                attrs["autogen_agentchat.task.message_count"] = len(task)
                if task and hasattr(task[0], "content"):
                    attrs["autogen_agentchat.task"] = str(task[0].content)[:200]

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
            attrs["autogen_agentchat.task"] = task[:200]

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
