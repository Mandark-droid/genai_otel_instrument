"""OpenTelemetry instrumentor for Microsoft AutoGen framework.

This instrumentor automatically traces multi-agent conversations, group chats,
and agent interactions using the Microsoft AutoGen framework.

Note: AutoGen is entering maintenance mode and merging with Semantic Kernel
into the Microsoft Agent Framework (public preview Oct 2025). This instrumentor
supports the current AutoGen release.

Requirements:
    A package that provides ``import autogen`` with ConversableAgent:
    pip install "pyautogen~=0.2.0"  # Microsoft's 0.2 line (0.2.35 verified)
    pip install autogen             # AG2 classic (0.14.1 verified)
    ``pyautogen`` 0.10 is an empty proxy for autogen-agentchat and ``ag2`` 1.x has
    no ``autogen`` module; neither has anything to wrap here.
"""

import logging
from typing import Any, Dict, Optional

from opentelemetry import trace

from ..agent_budget import extract_budget_attributes
from ..config import OTelConfig
from .base import BaseInstrumentor, _descendant_recorded_usage

logger = logging.getLogger(__name__)


class AutoGenInstrumentor(BaseInstrumentor):
    """Instrumentor for Microsoft AutoGen framework"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._autogen_available = False
        self._check_availability()

    @staticmethod
    def _has_legacy_api(module: Any) -> bool:
        """Whether ``module`` carries the classic ConversableAgent API.

        ``pyautogen`` 0.10 and later is an empty proxy package for
        ``autogen-agentchat``: it imports, but there is nothing in it to wrap.
        """
        try:
            return hasattr(module, "ConversableAgent")
        except Exception:  # a lazy attribute that fails; instrument() reports it
            return True

    def _check_availability(self):
        """Check if AutoGen library is available."""
        found_without_api = None

        # Try pyautogen first (newer package name)
        try:
            import autogen

            if self._has_legacy_api(autogen):
                self._autogen_available = True
                logger.debug("AutoGen library detected and available for instrumentation")
                return
            found_without_api = "autogen"
        except ImportError:
            pass

        # Fall back to older package name
        try:
            import pyautogen

            if self._has_legacy_api(pyautogen):
                self._autogen_available = True
                logger.debug(
                    "AutoGen library (pyautogen) detected and available for instrumentation"
                )
                return
            found_without_api = "pyautogen"
        except ImportError:
            pass

        self._autogen_available = False
        if found_without_api:
            logger.debug(
                "'%s' is installed but has no ConversableAgent API (pyautogen 0.10+ is a "
                "proxy for autogen-agentchat); legacy AutoGen instrumentation will be skipped",
                found_without_api,
            )
        else:
            logger.debug("AutoGen library not installed, instrumentation will be skipped")

    def instrument(self, config: OTelConfig):
        """Instrument AutoGen framework if available.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._autogen_available:
            logger.debug("Skipping AutoGen instrumentation - library not available")
            return

        # Idempotency guard: repeated instrument() calls must not stack wrappers
        # on the shared ConversableAgent / GroupChat classes.
        if self._instrumented:
            logger.debug("AutoGen already instrumented, skipping repeat instrument()")
            return

        self.config = config

        try:
            # Try both package names
            try:
                import autogen
            except ImportError:
                import pyautogen as autogen

            # Create span wrappers (same pattern as CrewAI - direct application)
            chat_wrapper = self.create_span_wrapper(
                span_name="autogen.initiate_chat",
                extract_attributes=self._extract_chat_attributes,
            )
            select_speaker_wrapper = self.create_span_wrapper(
                span_name="autogen.group_chat.select_speaker",
                extract_attributes=self._extract_group_chat_attributes,
            )
            group_run_wrapper = self.create_span_wrapper(
                span_name="autogen.group_chat.run",
                extract_attributes=self._extract_group_chat_manager_attributes,
            )

            # Instrument ConversableAgent.initiate_chat (main conversation method)
            if hasattr(autogen, "ConversableAgent"):
                if hasattr(autogen.ConversableAgent, "initiate_chat"):
                    original = autogen.ConversableAgent.initiate_chat
                    autogen.ConversableAgent.initiate_chat = chat_wrapper(original)
                    logger.debug("Instrumented ConversableAgent.initiate_chat()")

            # Instrument GroupChat.select_speaker (agent selection)
            if hasattr(autogen, "GroupChat"):
                if hasattr(autogen.GroupChat, "select_speaker"):
                    original = autogen.GroupChat.select_speaker
                    autogen.GroupChat.select_speaker = select_speaker_wrapper(original)
                    logger.debug("Instrumented GroupChat.select_speaker()")

            # Instrument the group chat run. The group chat is driven by
            # GroupChatManager.run_chat(), which the manager registers as a reply
            # function when it is built. pyautogen 0.2 has no GroupChatManager.run()
            # at all, and the one AG2 inherits from ConversableAgent only starts a
            # background thread and returns, so a span around it ends before the
            # chat has run. run() stays as the fallback where run_chat() is missing.
            if hasattr(autogen, "GroupChatManager"):
                if hasattr(autogen.GroupChatManager, "run_chat"):
                    original = autogen.GroupChatManager.run_chat
                    autogen.GroupChatManager.run_chat = group_run_wrapper(original)
                    logger.debug("Instrumented GroupChatManager.run_chat()")
                elif hasattr(autogen.GroupChatManager, "run"):
                    original = autogen.GroupChatManager.run
                    autogen.GroupChatManager.run = group_run_wrapper(original)
                    logger.debug("Instrumented GroupChatManager.run()")

            self._instrumented = True
            logger.info("AutoGen instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument AutoGen: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

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
        attrs.update(extract_budget_attributes(kwargs, instance))

        # Extract sender agent name
        if hasattr(instance, "name"):
            attrs["autogen.agent.name"] = instance.name
            # Cross-framework agent attribution per upstream proposal
            # semantic-conventions-genai#91.
            attrs["gen_ai.agent.name"] = instance.name
            attrs["autogen.conversation.sender"] = instance.name

        # Extract sender agent type
        agent_type = type(instance).__name__
        attrs["autogen.agent.type"] = agent_type

        # Extract model from recipient's llm_config for cost enrichment
        recipient = None
        if len(args) > 0:
            recipient = args[0]
        else:
            recipient = kwargs.get("recipient")

        # Try to extract model name from recipient (usually the LLM-backed agent)
        model_name = self._extract_model_from_agent(recipient)
        if not model_name:
            # Fallback: try sender
            model_name = self._extract_model_from_agent(instance)
        if model_name:
            attrs["gen_ai.request.model"] = model_name

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
                attrs["autogen.message"] = self._cap_content(message)
                attrs["autogen.message.type"] = "string"
            elif isinstance(message, dict):
                if "content" in message:
                    attrs["autogen.message"] = self._cap_content(message["content"])
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

    def _extract_model_from_agent(self, agent) -> Optional[str]:
        """Extract model name from an AutoGen agent's llm_config.

        Args:
            agent: An AutoGen agent instance.

        Returns:
            Optional[str]: The model name or None.
        """
        if agent is None:
            return None
        try:
            llm_config = getattr(agent, "llm_config", None)
            if not llm_config:
                return None

            if not isinstance(llm_config, dict):
                # AG2 replaces the dict with an LLMConfig object whose config_list
                # holds one entry object per model.
                model = self._config_string(llm_config, "model")
                if model:
                    return model
                config_list = getattr(llm_config, "config_list", None)
                if isinstance(config_list, (list, tuple)) and config_list:
                    return self._config_string(config_list[0], "model")
                return None

            # Direct model field
            if "model" in llm_config:
                return str(llm_config["model"])

            # config_list[0]["model"]
            config_list = llm_config.get("config_list")
            if config_list and isinstance(config_list, list) and len(config_list) > 0:
                first = config_list[0]
                if isinstance(first, dict) and "model" in first:
                    return str(first["model"])
        except Exception:
            pass
        return None

    @staticmethod
    def _config_string(source: Any, key: str) -> Optional[str]:
        """Read a non-empty string setting from a dict or a config object."""
        try:
            value = source.get(key) if isinstance(source, dict) else getattr(source, key, None)
        except Exception:
            return None
        return value if isinstance(value, str) and value else None

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from AutoGen ChatResult.cost.

        AutoGen's ChatResult has a `cost` field (CostDict) with:
        - usage_including_cached_inference: {model: {prompt_tokens, completion_tokens, cost, ...}}
        - usage_excluding_cached_inference: {model: {prompt_tokens, completion_tokens, cost, ...}}

        Args:
            result: The conversation result (ChatResult).

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        try:
            cost_dict = getattr(result, "cost", None)
            if not cost_dict or not isinstance(cost_dict, dict):
                return None

            # Use usage_including_cached_inference for complete picture
            usage_data = cost_dict.get("usage_including_cached_inference", {})
            if not usage_data:
                usage_data = cost_dict.get("usage_excluding_cached_inference", {})
            if not usage_data:
                return None

            # Aggregate tokens across all models used in this conversation
            total_prompt = 0
            total_completion = 0
            total_cost = 0.0

            for model_name, model_usage in usage_data.items():
                if not isinstance(model_usage, dict):
                    continue
                total_prompt += model_usage.get("prompt_tokens", 0)
                total_completion += model_usage.get("completion_tokens", 0)
                total_cost += model_usage.get("cost", 0.0)

            if total_prompt == 0 and total_completion == 0:
                return None

            return {
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
            }

        except Exception as e:
            logger.debug("Failed to extract token usage from ChatResult: %s", e)

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
                            attrs["autogen.conversation.last_message"] = self._cap_content(
                                last_message["content"]
                            )
                        if "role" in last_message:
                            attrs["autogen.conversation.last_role"] = last_message["role"]
                        if "name" in last_message:
                            attrs["autogen.conversation.last_speaker"] = last_message["name"]

            # Extract cost from ChatResult.cost (CostDict). It is the sum of the LLM
            # calls made inside the conversation: when an instrumented provider span
            # beneath this one already recorded them, the base wrapper leaves tokens
            # and cost to that span, and repeating the cost here would double it.
            if hasattr(result, "cost") and not _descendant_recorded_usage(trace.get_current_span()):
                try:
                    cost_dict = result.cost
                    if isinstance(cost_dict, dict):
                        # Calculate total cost across all models
                        usage_data = cost_dict.get("usage_including_cached_inference", {})
                        total_cost = 0.0
                        for model_name, model_usage in usage_data.items():
                            if isinstance(model_usage, dict):
                                total_cost += model_usage.get("cost", 0.0)
                        if total_cost > 0:
                            attrs["gen_ai.usage.cost.total"] = total_cost
                except Exception as e:
                    logger.debug("Failed to extract cost: %s", e)

            # Extract summary if available
            if hasattr(result, "summary"):
                attrs["autogen.conversation.summary"] = self._cap_content(result.summary)

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
