"""OpenTelemetry instrumentor for Google Agent Development Kit (ADK).

This instrumentor automatically traces agent execution via Google ADK's
Runner and InMemoryRunner classes.

Google ADK has built-in OpenTelemetry tracing oriented toward Google Cloud.
This instrumentor provides provider-agnostic instrumentation that works
with any OTLP-compatible backend.

Requirements:
    pip install google-adk>=1.17.0
"""

import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class GoogleADKInstrumentor(BaseInstrumentor):
    """Instrumentor for Google Agent Development Kit (ADK)"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._adk_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if Google ADK library is available."""
        try:
            import google.adk  # noqa: F401

            self._adk_available = True
            logger.debug("Google ADK library detected and available for instrumentation")
        except ImportError:
            logger.debug("Google ADK library not installed, instrumentation will be skipped")
            self._adk_available = False

    def instrument(self, config: OTelConfig):
        """Instrument Google ADK framework if available.

        Instruments:
        - Runner.run_async() - Main async execution method
        - InMemoryRunner.run_debug() - Simplified debug execution

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._adk_available:
            logger.debug("Skipping Google ADK instrumentation - library not available")
            return

        self.config = config

        try:
            # Instrument Runner.run_async()
            try:
                from google.adk.runners import Runner

                run_wrapper = self.create_span_wrapper(
                    span_name="google_adk.runner.run",
                    extract_attributes=self._extract_runner_attributes,
                )
                if hasattr(Runner, "run_async"):
                    Runner.run_async = run_wrapper(Runner.run_async)
                    logger.debug("Instrumented Runner.run_async()")
            except ImportError:
                logger.debug("Could not import google.adk.runners.Runner")

            # Instrument InMemoryRunner.run_debug()
            try:
                from google.adk.runners import InMemoryRunner

                debug_wrapper = self.create_span_wrapper(
                    span_name="google_adk.runner.run_debug",
                    extract_attributes=self._extract_runner_debug_attributes,
                )
                if hasattr(InMemoryRunner, "run_debug"):
                    InMemoryRunner.run_debug = debug_wrapper(InMemoryRunner.run_debug)
                    logger.debug("Instrumented InMemoryRunner.run_debug()")
            except ImportError:
                logger.debug("Could not import google.adk.runners.InMemoryRunner")

            self._instrumented = True
            logger.info("Google ADK instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument Google ADK: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _extract_runner_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Extract attributes from Runner.run_async() call.

        Args:
            instance: The Runner instance.
            args: Positional arguments (user_id, session_id, new_message).
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "google_adk"
        attrs["gen_ai.operation.name"] = "runner.run"

        # Extract app_name from runner
        if hasattr(instance, "app_name"):
            attrs["google_adk.app_name"] = str(instance.app_name)

        # Extract agent info
        if hasattr(instance, "agent"):
            agent = instance.agent
            if hasattr(agent, "name"):
                attrs["google_adk.agent.name"] = str(agent.name)
            if hasattr(agent, "model"):
                attrs["gen_ai.request.model"] = str(agent.model)
            if hasattr(agent, "description") and agent.description:
                attrs["google_adk.agent.description"] = str(agent.description)[:200]

            # Extract sub_agents count if available
            if hasattr(agent, "sub_agents") and agent.sub_agents:
                attrs["google_adk.sub_agent_count"] = len(agent.sub_agents)
                sub_agent_names = []
                for sub_agent in agent.sub_agents:
                    if hasattr(sub_agent, "name"):
                        sub_agent_names.append(str(sub_agent.name))
                if sub_agent_names:
                    attrs["google_adk.sub_agent_names"] = sub_agent_names[:10]

            # Extract tools
            if hasattr(agent, "tools") and agent.tools:
                tool_names = []
                for tool in agent.tools:
                    name = getattr(tool, "name", None) or type(tool).__name__
                    tool_names.append(str(name))
                if tool_names:
                    attrs["google_adk.tools"] = tool_names[:10]
                    attrs["google_adk.tool_count"] = len(agent.tools)

        # Extract user_id and session_id from kwargs
        user_id = kwargs.get("user_id")
        if user_id:
            attrs["google_adk.user_id"] = str(user_id)

        session_id = kwargs.get("session_id")
        if session_id:
            attrs["google_adk.session_id"] = str(session_id)

        # Extract message content (from new_message kwarg)
        new_message = kwargs.get("new_message")
        if new_message:
            # new_message is a Content object with parts
            if hasattr(new_message, "parts"):
                for part in new_message.parts:
                    if hasattr(part, "text") and part.text:
                        attrs["google_adk.input_message"] = str(part.text)[:200]
                        break

        return attrs

    def _extract_runner_debug_attributes(
        self, instance: Any, args: Any, kwargs: Any
    ) -> Dict[str, Any]:
        """Extract attributes from InMemoryRunner.run_debug() call.

        Args:
            instance: The InMemoryRunner instance.
            args: Positional arguments (message string).
            kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "google_adk"
        attrs["gen_ai.operation.name"] = "runner.run_debug"

        # Extract app_name from runner
        if hasattr(instance, "app_name"):
            attrs["google_adk.app_name"] = str(instance.app_name)

        # Extract agent info
        if hasattr(instance, "agent"):
            agent = instance.agent
            if hasattr(agent, "name"):
                attrs["google_adk.agent.name"] = str(agent.name)
            if hasattr(agent, "model"):
                attrs["gen_ai.request.model"] = str(agent.model)

        # Extract the input message (first positional arg)
        if args and len(args) > 0:
            attrs["google_adk.input_message"] = str(args[0])[:200]

        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from ADK execution result.

        Note: Google ADK doesn't directly expose token usage in results.
        Token usage is captured by underlying LLM provider instrumentors.

        Args:
            result: The execution result.

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        return None

    def _extract_response_attributes(self, result) -> Dict[str, Any]:
        """Extract response attributes from ADK execution result.

        Args:
            result: The execution result.

        Returns:
            Dict[str, Any]: Dictionary of response attributes.
        """
        attrs = {}

        try:
            if isinstance(result, str):
                attrs["google_adk.output"] = result[:500]
            elif hasattr(result, "text"):
                attrs["google_adk.output"] = str(result.text)[:500]
        except Exception as e:
            logger.debug("Failed to extract response attributes: %s", e)

        return attrs

    def _extract_finish_reason(self, result) -> Optional[str]:
        """Extract finish reason from ADK execution result.

        Args:
            result: The execution result.

        Returns:
            Optional[str]: The finish reason string or None.
        """
        if result:
            return "completed"
        return None
