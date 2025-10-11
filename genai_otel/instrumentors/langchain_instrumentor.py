from .base import BaseInstrumentor
from ..config import OTelConfig
import wrapt
from typing import Dict, Optional


class LangChainInstrumentor(BaseInstrumentor):
    """Instrumentor for LangChain"""

    def instrument(self, config: OTelConfig):
        self.config = config
        try:
            from langchain.chains.base import Chain
            from langchain.agents.agent import AgentExecutor

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
                    span.set_attribute("langchain.agent.name", getattr(instance, "agent", {}).get("name", "unknown"))
                    result = original_agent_call(instance, *args, **kwargs)
                    return result

            AgentExecutor.__call__ = wrapped_agent_call

        except ImportError:
            pass

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        return None