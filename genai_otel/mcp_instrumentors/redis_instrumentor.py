"""OpenTelemetry instrumentor for Redis clients.

This module provides the `RedisInstrumentor` class, which automatically
instruments Redis operations, enabling tracing of caching interactions
within GenAI applications.
"""

import logging

from opentelemetry.instrumentation.redis import RedisInstrumentor as OTelRedisInstrumentor

from ..config import OTelConfig

logger = logging.getLogger(__name__)


class RedisInstrumentor:  # pylint: disable=R0903
    """Instrument Redis clients"""

    def __init__(self, config: OTelConfig):
        self.config = config

    def instrument(self):
        """Instrument Redis"""
        try:
            OTelRedisInstrumentor().instrument()
            logger.info("Redis instrumentation enabled")
        except ImportError:
            logger.debug("Redis-py not installed, skipping instrumentation.")
        except Exception as e:
            logger.warning(f"Redis instrumentation failed: {e}")
