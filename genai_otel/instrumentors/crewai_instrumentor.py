"""OpenTelemetry instrumentor for the CrewAI framework.

This instrumentor automatically traces crew execution, agents, tasks, and
collaborative workflows using the CrewAI multi-agent framework.
"""

import json
import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class CrewAIInstrumentor(BaseInstrumentor):
    """Instrumentor for CrewAI multi-agent framework"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._crewai_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if CrewAI library is available."""
        try:
            import crewai

            self._crewai_available = True
            logger.debug("CrewAI library detected and available for instrumentation")
        except ImportError:
            logger.debug("CrewAI library not installed, instrumentation will be skipped")
            self._crewai_available = False

    def instrument(self, config: OTelConfig):
        """Instrument CrewAI framework if available.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._crewai_available:
            logger.debug("Skipping CrewAI instrumentation - library not available")
            return

        self.config = config

        try:
            import crewai
            import wrapt

            # Instrument Crew.kickoff() method (main execution entry point)
            if hasattr(crewai, "Crew"):
                if hasattr(crewai.Crew, "kickoff"):
                    original_kickoff = crewai.Crew.kickoff
                    crewai.Crew.kickoff = wrapt.FunctionWrapper(
                        original_kickoff, self._wrap_crew_kickoff
                    )

                self._instrumented = True
                logger.info("CrewAI instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument CrewAI: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _wrap_crew_kickoff(self, wrapped, instance, args, kwargs):
        """Wrap Crew.kickoff() method with span.

        Args:
            wrapped: The original method.
            instance: The Crew instance.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        return self.create_span_wrapper(
            span_name="crewai.crew.execution",
            extract_attributes=self._extract_crew_attributes,
        )(wrapped)(instance, *args, **kwargs)

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

                    # Store input values (truncated)
                    for key, value in list(inputs.items())[:5]:  # Limit to 5 inputs
                        value_str = str(value)[:200]  # Truncate long values
                        attrs[f"crewai.inputs.{key}"] = value_str
                else:
                    # Non-dict inputs
                    attrs["crewai.inputs"] = str(inputs)[:200]
            except Exception as e:
                logger.debug("Failed to extract inputs: %s", e)

        # Extract manager agent for hierarchical process
        if hasattr(instance, "manager_agent") and instance.manager_agent:
            try:
                if hasattr(instance.manager_agent, "role"):
                    attrs["crewai.manager.role"] = str(instance.manager_agent.role)[:100]
            except Exception as e:
                logger.debug("Failed to extract manager agent: %s", e)

        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from crew execution result.

        Note: CrewAI doesn't directly expose token usage in the result.
        Token usage is captured by underlying LLM provider instrumentors.

        Args:
            result: The crew execution result.

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        # CrewAI doesn't directly expose usage in the result
        # Token usage is captured by LLM provider instrumentors (OpenAI, Anthropic, etc.)
        # We could try to aggregate if CrewAI provides usage info in the future
        if hasattr(result, "token_usage"):
            try:
                usage = result.token_usage
                return {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                }
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
                output = result[:500]  # Truncate to avoid span size issues
                attrs["crewai.output"] = output
                attrs["crewai.output_length"] = len(result)

            # If result is a CrewOutput object
            elif hasattr(result, "raw"):
                output = str(result.raw)[:500]
                attrs["crewai.output"] = output
                attrs["crewai.output_length"] = len(str(result.raw))

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
                                task_result = str(task_output.raw)[:200]  # Truncate
                                task_outputs.append(task_result)

                        if task_outputs:
                            attrs["crewai.task_outputs"] = task_outputs
                except Exception as e:
                    logger.debug("Failed to extract tasks_output: %s", e)

            # If result has json attribute
            if hasattr(result, "json"):
                try:
                    attrs["crewai.output.json"] = str(result.json)[:500]
                except Exception as e:
                    logger.debug("Failed to extract JSON output: %s", e)

            # If result has pydantic attribute
            if hasattr(result, "pydantic"):
                try:
                    attrs["crewai.output.pydantic"] = str(result.pydantic)[:500]
                except Exception as e:
                    logger.debug("Failed to extract Pydantic output: %s", e)

        except Exception as e:
            logger.debug("Failed to extract response attributes: %s", e)

        return attrs

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
