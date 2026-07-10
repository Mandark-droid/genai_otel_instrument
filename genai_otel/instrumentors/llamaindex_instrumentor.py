"""OpenTelemetry instrumentor for the LlamaIndex framework.

This instrumentor automatically traces query engine operations within LlamaIndex,
capturing relevant attributes such as the query text.
"""

import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class LlamaIndexInstrumentor(BaseInstrumentor):
    """Instrumentor for LlamaIndex"""

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

    def instrument(self, config: OTelConfig):
        self.config = config

        # Idempotency guard: repeated instrument() calls must not stack wrappers
        # on the shared BaseQueryEngine.query method.
        if self._instrumented:
            logger.debug("LlamaIndex already instrumented, skipping repeat instrument()")
            return

        try:
            from llama_index.core.query_engine import BaseQueryEngine

            original_query = BaseQueryEngine.query

            def wrapped_query(instance, *args, **kwargs):
                with self.tracer.start_as_current_span("llamaindex.query_engine") as span:
                    # Pre-call attribute extraction must never break the business
                    # query: contain any failure here and still run the query.
                    try:
                        query_text = args[0] if args else kwargs.get("query_str", "")
                        span.set_attribute("llamaindex.query", self._cap_content(query_text))
                    except Exception as e:
                        logger.debug("Failed to set llamaindex query attribute: %s", e)
                    result = original_query(instance, *args, **kwargs)
                    return result

            BaseQueryEngine.query = wrapped_query
            self._instrumented = True

        except ImportError:
            pass

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        return None
