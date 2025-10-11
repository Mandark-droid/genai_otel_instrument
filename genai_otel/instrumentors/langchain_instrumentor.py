"""OpenTelemetry instrumentor for the LangChain framework.

This instrumentor automatically traces various components within LangChain,
including chains and agents, capturing relevant attributes for observability.
"""

from typing import Dict, Optional

import logging

from .base import BaseInstrumentor
from ..config import OTelConfig

logger = logging.getLogger(__name__)


class LangChainInstrumentor(BaseInstrumentor):
    """Instrumentor for LangChain"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._langchain_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if langchain library is available."""
        try:
            import langchain

            self._langchain_available = True
            logger.debug("langchain library detected and available for instrumentation")
        except ImportError:
            logger.debug("langchain library not installed, instrumentation will be skipped")
            self._langchain_available = False

    def instrument(self, config: OTelConfig):
        """Instrument  langchain available if available."""
        if not self._langchain_available:
            logger.debug("Skipping instrumentation - library not available")
            return

        self.config = config
        try:
            from langchain.agents.agent import AgentExecutor
            from langchain.chains.base import Chain

            # Instrument Chains
            original_call = Chain.__call__

            def wrapped_call(instance, *args, **kwargs):
                chain_type = instance.__class__.__name__
                with self.tracer.start_as_current_span(f"langchain.chain.{chain_type}") as span:
                    span.set_attribute("langchain.chain.type", chain_type)
                    result = original_call(instance, *args, **kwargs)
                    return result

            Chain.__call__ = wrapped_call

            # Instrument Agents
            original_agent_call = AgentExecutor.__call__

            def wrapped_agent_call(instance, *args, **kwargs):
                with self.tracer.start_as_current_span("langchain.agent.execute") as span:
                    agent_name = getattr(instance, "agent", {}).get("name", "unknown")
                    span.set_attribute("langchain.agent.name", agent_name)
                    result = original_agent_call(instance, *args, **kwargs)
                    return result

            AgentExecutor.__call__ = wrapped_agent_call

        except ImportError:
            pass

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        return None
