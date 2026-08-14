"""OpenTelemetry instrumentor for Apache Kafka clients.

This module provides the `KafkaInstrumentor` class, which automatically
instruments Kafka producers and consumers, enabling tracing of message
queue operations within GenAI applications.
"""

import logging

try:
    from opentelemetry.instrumentation.kafka import KafkaInstrumentor as OTelKafkaInstrumentor
except ImportError:
    OTelKafkaInstrumentor = None

from ..config import OTelConfig

logger = logging.getLogger(__name__)


class KafkaInstrumentor:  # pylint: disable=R0903
    """Instrument Kafka producers and consumers"""

    def __init__(self, config: OTelConfig):
        self.config = config

    def instrument(self):
        """Instrument Kafka"""
        # Optional dependency absent: the name is None rather than missing, so
        # calling it raises TypeError, not ImportError. Without this guard that
        # fell through to the handler below and logged a warning that reads like
        # a failure on a fresh install, where none of these are expected.
        if OTelKafkaInstrumentor is None:
            logger.debug("Kafka-python not installed, skipping instrumentation.")
        else:
            try:
                OTelKafkaInstrumentor().instrument()
                logger.info("Kafka instrumentation enabled")
            except ImportError:
                logger.debug("Kafka-python not installed, skipping instrumentation.")
            except Exception as e:
                logger.warning(f"Kafka instrumentation failed: {e}")
