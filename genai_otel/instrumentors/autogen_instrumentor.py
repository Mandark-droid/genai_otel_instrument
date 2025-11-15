"""OpenTelemetry instrumentor for Microsoft AutoGen framework.

This instrumentor automatically traces multi-agent conversations, group chats,
and agent interactions using the Microsoft AutoGen framework.

Note: AutoGen is entering maintenance mode and merging with Semantic Kernel
into the Microsoft Agent Framework (public preview Oct 2025). This instrumentor
supports the current AutoGen release.

Requirements:
    pip install pyautogen  # or autogen (legacy package name)
"""

import json
import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class AutoGenInstrumentor(BaseInstrumentor):
    """Instrumentor for Microsoft AutoGen framework"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._autogen_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if AutoGen library is available."""
        # Try pyautogen first (newer package name)
        try:
            import autogen

            self._autogen_available = True
            logger.debug("AutoGen library detected and available for instrumentation")
            return
        except ImportError:
            pass

        # Fall back to older package name
        try:
            import pyautogen

            self._autogen_available = True
            logger.debug("AutoGen library (pyautogen) detected and available for instrumentation")
            return
        except ImportError:
            logger.debug("AutoGen library not installed, instrumentation will be skipped")
            self._autogen_available = False

    def instrument(self, config: OTelConfig):
        """Instrument AutoGen framework if available.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._autogen_available:
            logger.debug("Skipping AutoGen instrumentation - library not available")
            return

        self.config = config

        try:
            import wrapt

            # Try both package names
            try:
                import autogen
            except ImportError:
                import pyautogen as autogen

            # Instrument ConversableAgent.initiate_chat (main conversation method)
            if hasattr(autogen, "ConversableAgent"):
                if hasattr(autogen.ConversableAgent, "initiate_chat"):
                    original_initiate = autogen.ConversableAgent.initiate_chat
                    autogen.ConversableAgent.initiate_chat = wrapt.FunctionWrapper(
                        original_initiate, self._wrap_initiate_chat
                    )

            # Instrument GroupChat.run (group chat orchestration)
            if hasattr(autogen, "GroupChat"):
                # GroupChat typically has select_speaker method for agent selection
                if hasattr(autogen.GroupChat, "select_speaker"):
                    original_select = autogen.GroupChat.select_speaker
                    autogen.GroupChat.select_speaker = wrapt.FunctionWrapper(
                        original_select, self._wrap_select_speaker
                    )

            # Instrument GroupChatManager if available
            if hasattr(autogen, "GroupChatManager"):
                if hasattr(autogen.GroupChatManager, "run"):
                    original_run = autogen.GroupChatManager.run
                    autogen.GroupChatManager.run = wrapt.FunctionWrapper(
                        original_run, self._wrap_group_chat_run
                    )

            self._instrumented = True
            logger.info("AutoGen instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument AutoGen: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _wrap_initiate_chat(self, wrapped, instance, args, kwargs):
        """Wrap ConversableAgent.initiate_chat method with span.

        Args:
            wrapped: The original method.
            instance: The ConversableAgent instance.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        return self.create_span_wrapper(
            span_name="autogen.initiate_chat",
            extract_attributes=self._extract_chat_attributes,
        )(wrapped)(instance, *args, **kwargs)

    def _wrap_select_speaker(self, wrapped, instance, args, kwargs):
        """Wrap GroupChat.select_speaker method with span.

        Args:
            wrapped: The original method.
            instance: The GroupChat instance.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        return self.create_span_wrapper(
            span_name="autogen.group_chat.select_speaker",
            extract_attributes=self._extract_group_chat_attributes,
        )(wrapped)(instance, *args, **kwargs)

    def _wrap_group_chat_run(self, wrapped, instance, args, kwargs):
        """Wrap GroupChatManager.run method with span.

        Args:
            wrapped: The original method.
            instance: The GroupChatManager instance.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        return self.create_span_wrapper(
            span_name="autogen.group_chat.run",
            extract_attributes=self._extract_group_chat_manager_attributes,
        )(wrapped)(instance, *args, **kwargs)

    def _extract_chat_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Extract attributes from ConversableAgent.initiate_chat call.

        Args:
            instance: The ConversableAgent instance (sender).
            args: Positional arguments (recipient, message, etc.).
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "autogen"
        attrs["gen_ai.operation.name"] = "conversation.initiate"

        # Extract sender agent name
        if hasattr(instance, "name"):
            attrs["autogen.agent.name"] = instance.name
            attrs["autogen.conversation.sender"] = instance.name

        # Extract sender agent type
        agent_type = type(instance).__name__
        attrs["autogen.agent.type"] = agent_type

        # Extract recipient agent (first positional argument)
        recipient = None
        if len(args) > 0:
            recipient = args[0]
        else:
            recipient = kwargs.get("recipient")

        if recipient:
            if hasattr(recipient, "name"):
                attrs["autogen.conversation.recipient"] = recipient.name

            recipient_type = type(recipient).__name__
            attrs["autogen.recipient.type"] = recipient_type

        # Extract message (second positional argument or kwarg)
        message = None
        if len(args) > 1:
            message = args[1]
        else:
            message = kwargs.get("message")

        if message:
            # Message can be string or dict
            if isinstance(message, str):
                attrs["autogen.message"] = message[:500]  # Truncate
                attrs["autogen.message.type"] = "string"
            elif isinstance(message, dict):
                if "content" in message:
                    attrs["autogen.message"] = str(message["content"])[:500]
                attrs["autogen.message.type"] = "dict"
            else:
                attrs["autogen.message.type"] = str(type(message).__name__)

        # Extract max_turns if specified
        max_turns = kwargs.get("max_turns")
        if max_turns is not None:
            attrs["autogen.conversation.max_turns"] = max_turns

        # Extract silent mode
        if "silent" in kwargs:
            attrs["autogen.conversation.silent"] = kwargs["silent"]

        # Extract clear_history flag
        if "clear_history" in kwargs:
            attrs["autogen.conversation.clear_history"] = kwargs["clear_history"]

        return attrs

    def _extract_group_chat_attributes(
        self, instance: Any, args: Any, kwargs: Any
    ) -> Dict[str, Any]:
        """Extract attributes from GroupChat.select_speaker call.

        Args:
            instance: The GroupChat instance.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "autogen"
        attrs["gen_ai.operation.name"] = "group_chat.select_speaker"

        # Extract agent count
        if hasattr(instance, "agents") and instance.agents:
            attrs["autogen.group_chat.agent_count"] = len(instance.agents)

            # Extract agent names
            agent_names = [getattr(agent, "name", "unknown") for agent in instance.agents]
            attrs["autogen.group_chat.agents"] = agent_names[:10]  # Limit to 10

        # Extract selection mode/method
        if hasattr(instance, "speaker_selection_method"):
            attrs["autogen.group_chat.selection_mode"] = instance.speaker_selection_method

        # Extract max_round
        if hasattr(instance, "max_round"):
            attrs["autogen.group_chat.max_round"] = instance.max_round

        return attrs

    def _extract_group_chat_manager_attributes(
        self, instance: Any, args: Any, kwargs: Any
    ) -> Dict[str, Any]:
        """Extract attributes from GroupChatManager.run call.

        Args:
            instance: The GroupChatManager instance.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "autogen"
        attrs["gen_ai.operation.name"] = "group_chat.run"

        # Extract manager name
        if hasattr(instance, "name"):
            attrs["autogen.manager.name"] = instance.name

        # Extract groupchat info
        if hasattr(instance, "groupchat"):
            groupchat = instance.groupchat

            if hasattr(groupchat, "agents") and groupchat.agents:
                attrs["autogen.group_chat.agent_count"] = len(groupchat.agents)

                agent_names = [getattr(agent, "name", "unknown") for agent in groupchat.agents]
                attrs["autogen.group_chat.agents"] = agent_names[:10]

            if hasattr(groupchat, "speaker_selection_method"):
                attrs["autogen.group_chat.selection_mode"] = groupchat.speaker_selection_method

        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from conversation result.

        Note: AutoGen doesn't directly expose token usage in results.
        Token usage is captured by underlying LLM provider instrumentors.

        Args:
            result: The conversation result.

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        # AutoGen doesn't expose usage directly
        # Token usage is captured by LLM provider instrumentors (OpenAI, etc.)
        # We could try to aggregate if AutoGen provides usage info in the future
        if hasattr(result, "usage"):
            try:
                usage = result.usage
                return {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                }
            except Exception as e:
                logger.debug("Failed to extract token usage: %s", e)

        return None

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        """Extract response attributes from conversation result.

        Args:
            result: The conversation result.

        Returns:
            Dict[str, Any]: Dictionary of response attributes.
        """
        attrs = {}

        try:
            # AutoGen results are typically ChatResult objects
            if hasattr(result, "chat_history"):
                # Extract chat history length
                attrs["autogen.conversation.messages"] = len(result.chat_history)

                # Extract last message content (truncated)
                if result.chat_history:
                    last_message = result.chat_history[-1]
                    if isinstance(last_message, dict):
                        if "content" in last_message:
                            attrs["autogen.conversation.last_message"] = str(
                                last_message["content"]
                            )[:500]
                        if "role" in last_message:
                            attrs["autogen.conversation.last_role"] = last_message["role"]
                        if "name" in last_message:
                            attrs["autogen.conversation.last_speaker"] = last_message["name"]

            # Extract cost if available
            if hasattr(result, "cost"):
                try:
                    cost = result.cost
                    if isinstance(cost, dict):
                        for key, value in cost.items():
                            attrs[f"autogen.cost.{key}"] = value
                    else:
                        attrs["autogen.cost"] = cost
                except Exception as e:
                    logger.debug("Failed to extract cost: %s", e)

            # Extract summary if available
            if hasattr(result, "summary"):
                attrs["autogen.conversation.summary"] = str(result.summary)[:500]

        except Exception as e:
            logger.debug("Failed to extract response attributes: %s", e)

        return attrs

    def _extract_finish_reason(self, result) -> Optional[str]:
        """Extract finish reason from conversation result.

        Args:
            result: The conversation result.

        Returns:
            Optional[str]: The finish reason string or None if not available.
        """
        # Try to determine if conversation completed successfully
        if hasattr(result, "chat_history"):
            # If we have chat history, conversation completed
            return "completed"

        return None
