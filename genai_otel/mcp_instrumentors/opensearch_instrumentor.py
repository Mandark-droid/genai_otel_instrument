"""OpenTelemetry instrumentor for OpenSearch clients.

This module provides the `OpenSearchInstrumentor` class, which automatically
instruments OpenSearch operations, enabling tracing of search and indexing
interactions within GenAI applications.
"""

import logging

from ..config import OTelConfig

logger = logging.getLogger(__name__)


class OpenSearchInstrumentor:  # pylint: disable=R0903
    """Instrument OpenSearch clients"""

    def __init__(self, config: OTelConfig):
        self.config = config

    def instrument(self):
        """Instrument OpenSearch"""
        try:
            from opentelemetry.instrumentation.opensearch import (
                OpenSearchInstrumentor as OTelOpenSearchInstrumentor,
            )

            OTelOpenSearchInstrumentor().instrument()
            logger.info("OpenSearch instrumentation enabled")
        except ImportError:
            logger.debug("opensearch-py not installed, skipping instrumentation.")
        except Exception as e:
            logger.warning("OpenSearch instrumentation failed: %s", e)
