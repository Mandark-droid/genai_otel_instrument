from .base import BaseInstrumentor
from ..config import OTelConfig
from typing import Dict, Optional


class LlamaIndexInstrumentor(BaseInstrumentor):
    """Instrumentor for LlamaIndex"""

    def instrument(self, config: OTelConfig):
        self.config = config
        try:
            from llama_index.core.query_engine import BaseQueryEngine

            original_query = BaseQueryEngine.query

            def wrapped_query(instance, *args, **kwargs):
                with self.tracer.start_as_current_span("llamaindex.query_engine") as span:
                    query_text = args[0] if args else kwargs.get("query_str", "")
                    span.set_attribute("llamaindex.query", str(query_text)[:200])
                    result = original_query(instance, *args, **kwargs)
                    return result

            BaseQueryEngine.query = wrapped_query

        except ImportError:
            pass

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        return None
