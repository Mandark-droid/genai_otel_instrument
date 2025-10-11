import logging
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from ..config import OTelConfig
from .database_instrumentor import DatabaseInstrumentor
from .redis_instrumentor import RedisInstrumentor
from .kafka_instrumentor import KafkaInstrumentor
from .vector_db_instrumentor import VectorDBInstrumentor
from .api_instrumentor import APIInstrumentor

logger = logging.getLogger(__name__)


class MCPInstrumentorManager:
    """Manager for MCP (Model Context Protocol) tool instrumentation"""

    def __init__(self, config: OTelConfig):
        self.config = config
        self.instrumentors = []

    def instrument_all(self, fail_on_error: bool = False):
        """Instrument all detected MCP tools"""

        success_count = 0
        failure_count = 0

        # HTTP/API instrumentation
        try:
            logger.info("Instrumenting HTTP/API calls")
            RequestsInstrumentor().instrument()
            HTTPXClientInstrumentor().instrument()
            api_instrumentor = APIInstrumentor(self.config)
            api_instrumentor.instrument()
            logger.info("✓ HTTP/API instrumentation enabled")
            success_count += 1
        except ImportError as e:
            failure_count += 1
            logger.debug(f"✗ HTTP/API instrumentation skipped due to missing dependency: {e}")
        except Exception as e:
            failure_count += 1
            logger.error(f"✗ Failed to instrument HTTP/API: {e}", exc_info=True)
            if fail_on_error:
                raise

        # Database instrumentation
        try:
            logger.info("Instrumenting databases")
            db_instrumentor = DatabaseInstrumentor(self.config)
            result = db_instrumentor.instrument()
            if result > 0:
                success_count += 1
                logger.info(f"✓ Database instrumentation enabled ({result} databases)")
        except ImportError as e:
            failure_count += 1
            logger.debug(f"✗ Database instrumentation skipped due to missing dependency: {e}")
        except Exception as e:
            failure_count += 1
            logger.error(f"✗ Failed to instrument databases: {e}", exc_info=True)
            if fail_on_error:
                raise

        # Redis instrumentation
        try:
            logger.info("Instrumenting Redis")
            redis_instrumentor = RedisInstrumentor(self.config)
            redis_instrumentor.instrument()
            success_count += 1
        except ImportError as e:
            failure_count += 1
            logger.debug(f"✗ Redis instrumentation skipped due to missing dependency: {e}")
        except Exception as e:
            failure_count += 1
            logger.error(f"✗ Failed to instrument Redis: {e}", exc_info=True)
            if fail_on_error:
                raise

        # Kafka instrumentation
        try:
            logger.info("Instrumenting Kafka")
            kafka_instrumentor = KafkaInstrumentor(self.config)
            kafka_instrumentor.instrument()
            success_count += 1
        except ImportError as e:
            failure_count += 1
            logger.debug(f"✗ Kafka instrumentation skipped due to missing dependency: {e}")
        except Exception as e:
            failure_count += 1
            logger.error(f"✗ Failed to instrument Kafka: {e}", exc_info=True)
            if fail_on_error:
                raise

        # Vector DB instrumentation
        try:
            logger.info("Instrumenting Vector DBs")
            vector_db_instrumentor = VectorDBInstrumentor(self.config)
            result = vector_db_instrumentor.instrument()
            if result > 0:
                success_count += 1
                logger.info(f"✓ Vector DB instrumentation enabled ({result} databases)")
        except ImportError as e:
            failure_count += 1
            logger.debug(f"✗ Vector DB instrumentation skipped due to missing dependency: {e}")
        except Exception as e:
            failure_count += 1
            logger.error(f"✗ Failed to instrument Vector DBs: {e}", exc_info=True)
            if fail_on_error:
                raise

        logger.info(
            f"MCP instrumentation summary: {success_count} succeeded, "
            f"{failure_count} failed"
        )