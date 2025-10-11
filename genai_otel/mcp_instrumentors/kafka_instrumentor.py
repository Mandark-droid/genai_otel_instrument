"""OpenTelemetry instrumentor for Apache Kafka clients.

This module provides the `KafkaInstrumentor` class, which automatically
instruments Kafka producers and consumers, enabling tracing of message
queue operations within GenAI applications.
"""

import logging

from opentelemetry.instrumentation.kafka import KafkaInstrumentor as OTelKafkaInstrumentor

from ..config import OTelConfig

logger = logging.getLogger(__name__)


class KafkaInstrumentor:  # pylint: disable=R0903
    """Instrument Kafka producers and consumers"""

    def __init__(self, config: OTelConfig):
        self.config = config

    def instrument(self):
        """Instrument Kafka"""
        try:
            OTelKafkaInstrumentor().instrument()
            logger.info("Kafka instrumentation enabled")
        except ImportError:
            logger.debug("Kafka-python not installed, skipping instrumentation.")
        except Exception as e:
            logger.warning(f"Kafka instrumentation failed: {e}")
