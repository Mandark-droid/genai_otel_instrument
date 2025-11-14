"""OpenTelemetry instrumentor for the OpenAI Agents SDK.

This instrumentor automatically traces agent execution, handoffs, sessions,
and guardrails using the OpenAI Agents SDK (openai-agents package).
"""

import json
import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class OpenAIAgentsInstrumentor(BaseInstrumentor):
    """Instrumentor for OpenAI Agents SDK"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._agents_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if OpenAI Agents SDK is available."""
        try:
            import agents

            self._agents_available = True
            logger.debug("OpenAI Agents SDK detected and available for instrumentation")
        except ImportError:
            logger.debug("OpenAI Agents SDK not installed, instrumentation will be skipped")
            self._agents_available = False

    def instrument(self, config: OTelConfig):
        """Instrument OpenAI Agents SDK if available.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._agents_available:
            logger.debug("Skipping OpenAI Agents instrumentation - library not available")
            return

        self.config = config

        try:
            import agents
            import wrapt

            # Instrument Runner.run() (async) and Runner.run_sync() (sync)
            if hasattr(agents, "Runner"):
                # Instrument async run method
                if hasattr(agents.Runner, "run"):
                    original_run = agents.Runner.run
                    agents.Runner.run = wrapt.FunctionWrapper(original_run, self._wrap_runner_run)

                # Instrument sync run method
                if hasattr(agents.Runner, "run_sync"):
                    original_run_sync = agents.Runner.run_sync
                    agents.Runner.run_sync = wrapt.FunctionWrapper(
                        original_run_sync, self._wrap_runner_run_sync
                    )

                self._instrumented = True
                logger.info("OpenAI Agents SDK instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument OpenAI Agents SDK: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _wrap_runner_run(self, wrapped, instance, args, kwargs):
        """Wrap Runner.run() async method with span.

        Args:
            wrapped: The original method.
            instance: The Runner instance.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        return self.create_span_wrapper(
            span_name="openai_agents.runner.run",
            extract_attributes=self._extract_runner_attributes,
        )(wrapped)(instance, *args, **kwargs)

    def _wrap_runner_run_sync(self, wrapped, instance, args, kwargs):
        """Wrap Runner.run_sync() sync method with span.

        Args:
            wrapped: The original method.
            instance: The Runner instance.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        return self.create_span_wrapper(
            span_name="openai_agents.runner.run_sync",
            extract_attributes=self._extract_runner_attributes,
        )(wrapped)(instance, *args, **kwargs)

    def _extract_runner_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Extract attributes from Runner.run() or Runner.run_sync() call.

        Args:
            instance: The Runner instance.
            args: Positional arguments (agent, input_data, session, etc.).
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "openai_agents"
        attrs["gen_ai.operation.name"] = "agent.run"

        # Extract agent from args (first positional argument)
        agent = None
        if len(args) > 0:
            agent = args[0]
        else:
            agent = kwargs.get("agent")

        if agent:
            # Agent attributes
            if hasattr(agent, "name"):
                attrs["openai.agent.name"] = agent.name
                attrs["gen_ai.request.model"] = agent.name  # Use agent name as "model"

            if hasattr(agent, "model"):
                attrs["openai.agent.model"] = agent.model

            if hasattr(agent, "instructions"):
                # Truncate instructions to avoid span size issues
                instructions = str(agent.instructions)[:500]
                attrs["openai.agent.instructions"] = instructions

            # Extract tools
            if hasattr(agent, "tools") and agent.tools:
                try:
                    tool_names = [getattr(tool, "name", str(tool)[:50]) for tool in agent.tools]
                    attrs["openai.agent.tools"] = tool_names
                    attrs["openai.agent.tool_count"] = len(agent.tools)
                except Exception as e:
                    logger.debug("Failed to extract agent tools: %s", e)

            # Extract handoffs
            if hasattr(agent, "handoffs") and agent.handoffs:
                try:
                    handoff_names = [getattr(h, "name", str(h)[:50]) for h in agent.handoffs]
                    attrs["openai.agent.handoffs"] = handoff_names
                    attrs["openai.agent.handoff_count"] = len(agent.handoffs)
                except Exception as e:
                    logger.debug("Failed to extract agent handoffs: %s", e)

            # Extract guardrails
            if hasattr(agent, "guardrails") and agent.guardrails:
                try:
                    attrs["openai.agent.guardrails_enabled"] = True
                    attrs["openai.agent.guardrail_count"] = len(agent.guardrails)
                except Exception as e:
                    logger.debug("Failed to extract agent guardrails: %s", e)

        # Extract input data (second positional argument)
        input_data = None
        if len(args) > 1:
            input_data = args[1]
        else:
            input_data = kwargs.get("input_data")

        if input_data:
            # Truncate input to avoid span size issues
            input_str = str(input_data)[:500]
            attrs["openai.agent.input"] = input_str
            attrs["openai.agent.input_length"] = len(str(input_data))

        # Extract session (third positional argument or kwarg)
        session = None
        if len(args) > 2:
            session = args[2]
        else:
            session = kwargs.get("session")

        if session:
            # Session attributes
            if hasattr(session, "session_id"):
                attrs["openai.session.id"] = session.session_id
                attrs["session.id"] = session.session_id  # Generic session ID

            # Detect session type
            session_type = type(session).__name__
            attrs["openai.session.type"] = session_type

        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from agent run result.

        Note: The OpenAI Agents SDK may not directly expose token usage in the result.
        Token usage is captured by the underlying OpenAI SDK instrumentor.

        Args:
            result: The agent run result object.

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        # OpenAI Agents SDK doesn't directly expose usage in the result
        # Token usage is captured by the OpenAI SDK instrumentor for underlying LLM calls
        # We can try to extract if the result has usage information
        if hasattr(result, "usage") and result.usage:
            usage = result.usage
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        return None

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        """Extract response attributes from agent run result.

        Args:
            result: The agent run result object.

        Returns:
            Dict[str, Any]: Dictionary of response attributes.
        """
        attrs = {}

        # Extract final output
        if hasattr(result, "final_output"):
            output = str(result.final_output)[:500]  # Truncate to avoid span size issues
            attrs["openai.agent.output"] = output
            attrs["openai.agent.output_length"] = len(str(result.final_output))

        # Extract handoff information if the agent handed off to another agent
        if hasattr(result, "handoff") and result.handoff:
            attrs["openai.handoff.occurred"] = True
            if hasattr(result.handoff, "target_agent"):
                attrs["openai.handoff.to_agent"] = result.handoff.target_agent

        # Extract guardrail validation results
        if hasattr(result, "guardrail_results"):
            try:
                guardrail_results = result.guardrail_results
                if guardrail_results:
                    # Count violations
                    violation_count = sum(
                        1 for r in guardrail_results if not getattr(r, "passed", True)
                    )
                    attrs["openai.guardrail.violations"] = violation_count
                    attrs["openai.guardrail.validated"] = True
            except Exception as e:
                logger.debug("Failed to extract guardrail results: %s", e)

        # Extract metadata if available
        if hasattr(result, "metadata"):
            try:
                metadata = result.metadata
                if isinstance(metadata, dict):
                    # Add selected metadata fields
                    for key in ["run_id", "agent_id", "session_id"]:
                        if key in metadata:
                            attrs[f"openai.agent.metadata.{key}"] = metadata[key]
            except Exception as e:
                logger.debug("Failed to extract result metadata: %s", e)

        return attrs

    def _extract_finish_reason(self, result) -> Optional[str]:
        """Extract finish reason from agent run result.

        Args:
            result: The agent run result object.

        Returns:
            Optional[str]: The finish reason string or None if not available.
        """
        # Try to extract finish reason if available
        if hasattr(result, "finish_reason"):
            return result.finish_reason

        # Check if result has a status
        if hasattr(result, "status"):
            return result.status

        return None
