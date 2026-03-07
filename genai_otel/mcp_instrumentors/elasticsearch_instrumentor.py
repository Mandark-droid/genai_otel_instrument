"""OpenTelemetry instrumentor for Elasticsearch clients.

This module provides the `ElasticsearchInstrumentor` class, which automatically
instruments Elasticsearch operations, enabling tracing of search and indexing
interactions within GenAI applications.
"""

import logging

from ..config import OTelConfig

logger = logging.getLogger(__name__)


class ElasticsearchInstrumentor:  # pylint: disable=R0903
    """Instrument Elasticsearch clients"""

    def __init__(self, config: OTelConfig):
        self.config = config

    def instrument(self):
        """Instrument Elasticsearch"""
        try:
            from opentelemetry.instrumentation.elasticsearch import (
                ElasticsearchInstrumentor as OTelElasticsearchInstrumentor,
            )

            OTelElasticsearchInstrumentor().instrument()
            logger.info("Elasticsearch instrumentation enabled")
        except ImportError:
            logger.debug("elasticsearch-py not installed, skipping instrumentation.")
        except Exception as e:
            logger.warning("Elasticsearch instrumentation failed: %s", e)
