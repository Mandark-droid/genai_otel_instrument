"""OpenTelemetry instrumentor for Redis clients.

This module provides the `RedisInstrumentor` class, which automatically
instruments Redis operations, enabling tracing of caching interactions
within GenAI applications.
"""

import logging

try:
    from opentelemetry.instrumentation.redis import RedisInstrumentor as OTelRedisInstrumentor
except ImportError:
    OTelRedisInstrumentor = None

from ..config import OTelConfig

logger = logging.getLogger(__name__)


class RedisInstrumentor:  # pylint: disable=R0903
    """Instrument Redis clients"""

    def __init__(self, config: OTelConfig):
        self.config = config

    def instrument(self):
        """Instrument Redis"""
        # Optional dependency absent: the name is None rather than missing, so
        # calling it raises TypeError, not ImportError. Without this guard that
        # fell through to the handler below and logged a warning that reads like
        # a failure on a fresh install, where none of these are expected.
        if OTelRedisInstrumentor is None:
            logger.debug("Redis-py not installed, skipping instrumentation.")
        else:
            try:
                OTelRedisInstrumentor().instrument()
                logger.info("Redis instrumentation enabled")
            except ImportError:
                logger.debug("Redis-py not installed, skipping instrumentation.")
            except Exception as e:
                logger.warning(f"Redis instrumentation failed: {e}")
