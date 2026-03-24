"""OpenTelemetry instrumentor for the CrewAI framework.

This instrumentor automatically traces crew execution, agents, tasks, and
collaborative workflows using the CrewAI multi-agent framework.

Context propagation is handled by the base class create_span_wrapper() which
sets each span as the current span in context, enabling proper parent-child
trace hierarchy. ThreadPoolExecutor is patched to propagate context across
threads used internally by CrewAI.

IMPORTANT: CrewAI has built-in telemetry that conflicts with OpenTelemetry.
This instrumentor automatically disables CrewAI's built-in telemetry by setting
the CREWAI_TELEMETRY_OPT_OUT environment variable to prevent conflicts.
"""

import logging
import os
import uuid
from typing import Any, Dict, Optional

from opentelemetry import context as otel_context

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

        Instruments all crew execution methods (sync and async):
        - Crew.kickoff() / kickoff_async() / akickoff() - Main execution
        - Crew.kickoff_for_each() / kickoff_for_each_async() / akickoff_for_each() - Batch
        - Task.execute_sync() / execute_async() - Task execution
        - Agent.execute_task() - Agent task execution
        - concurrent.futures.ThreadPoolExecutor - Automatic context propagation

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._crewai_available:
            logger.debug("Skipping CrewAI instrumentation - library not available")
            return

        self.config = config

        # Disable CrewAI's built-in telemetry to prevent conflicts with OpenTelemetry
        # CrewAI has its own telemetry system that interferes with OTel tracing
        os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
        logger.info("Disabled CrewAI built-in telemetry (CREWAI_TELEMETRY_OPT_OUT=true)")

        try:
            import crewai

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
                        setattr(crewai.Crew, method_name, crew_wrapper(original))
                        logger.debug("Instrumented Crew.%s()", method_name)

            # Instrument Task execution methods
            if hasattr(crewai, "Task"):
                for method_name in ["execute_sync", "execute_async"]:
                    if hasattr(crewai.Task, method_name):
                        original = getattr(crewai.Task, method_name)
                        setattr(crewai.Task, method_name, task_wrapper(original))
                        logger.debug("Instrumented Task.%s()", method_name)

            # Instrument Agent execution
            if hasattr(crewai, "Agent"):
                if hasattr(crewai.Agent, "execute_task"):
                    crewai.Agent.execute_task = agent_wrapper(crewai.Agent.execute_task)
                    logger.debug("Instrumented Agent.execute_task()")

            # Patch ThreadPoolExecutor.submit to propagate context automatically
            # This ensures any threaded execution by CrewAI maintains trace context
            self._patch_thread_pool_executor()

            self._instrumented = True
            logger.info("CrewAI instrumentation enabled with automatic context propagation")

        except Exception as e:
            logger.error("Failed to instrument CrewAI: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _patch_thread_pool_executor(self):
        """Patch ThreadPoolExecutor to automatically propagate OpenTelemetry context.

        This ensures that any code using ThreadPoolExecutor (including CrewAI internally)
        will automatically propagate trace context to worker threads.
        """
        try:
            from concurrent.futures import ThreadPoolExecutor

            original_submit = ThreadPoolExecutor.submit

            def context_propagating_submit(self, fn, /, *args, **kwargs):
                """Submit with automatic context propagation."""
                # Capture current context
                ctx = otel_context.get_current()

                # Wrap the function to propagate context
                def wrapper():
                    token = otel_context.attach(ctx)
                    try:
                        return fn(*args, **kwargs)
                    finally:
                        otel_context.detach(token)

                # Submit the wrapped function
                return original_submit(self, wrapper)

            ThreadPoolExecutor.submit = context_propagating_submit
            logger.debug("Patched ThreadPoolExecutor.submit() for automatic context propagation")

        except Exception as e:
            logger.debug(f"Could not patch ThreadPoolExecutor: {e}")
            # Not critical - continue with other instrumentation

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

        try:
            # Extract agent role
            if hasattr(instance, "role"):
                attrs["crewai.agent.role"] = str(instance.role)

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

        # --- Session ID extraction (required for TraceVerse session aggregation) ---
        session_id = None

        # Priority 1: App-provided session_id in kickoff inputs
        if isinstance(inputs, dict):
            session_id = inputs.get("session_id") or inputs.get("session.id")

        # Priority 2: OTelConfig.session_id_extractor callable
        if not session_id and self.config and self.config.session_id_extractor:
            try:
                session_id = self.config.session_id_extractor(instance, args, kwargs)
            except Exception as e:
                logger.debug("Failed to extract session ID via extractor: %s", e)

        # Priority 3: Crew instance ID (CrewAI assigns a UUID to each Crew)
        if not session_id and hasattr(instance, "id"):
            session_id = str(instance.id)

        # Priority 4: Auto-generate a new UUID for this kickoff
        if not session_id:
            session_id = str(uuid.uuid4())

        attrs["session.id"] = session_id
        attrs["crewai.session.id"] = session_id

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
