import logging
from opentelemetry.instrumentation.kafka import KafkaInstrumentor as OTelKafkaInstrumentor
from ..config import OTelConfig

logger = logging.getLogger(__name__)


class KafkaInstrumentor:
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

