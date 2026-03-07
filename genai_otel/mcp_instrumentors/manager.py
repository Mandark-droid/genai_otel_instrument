"""Manager for OpenTelemetry instrumentation of Model Context Protocol (MCP) tools.

This module provides the `MCPInstrumentorManager` class, which orchestrates
the automatic instrumentation of various MCP tools, including databases, caching
layers, message queues, vector databases, and generic API calls. It ensures
that these components are integrated into the OpenTelemetry tracing and metrics
system.
"""

import asyncio
import logging

try:
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
except ImportError:
    HTTPXClientInstrumentor = None

try:
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
except ImportError:
    RequestsInstrumentor = None

from ..config import OTelConfig
from .api_instrumentor import APIInstrumentor
from .database_instrumentor import DatabaseInstrumentor
from .elasticsearch_instrumentor import ElasticsearchInstrumentor
from .kafka_instrumentor import KafkaInstrumentor
from .minio_instrumentor import MinIOInstrumentor
from .opensearch_instrumentor import OpenSearchInstrumentor
from .rabbitmq_instrumentor import RabbitMQInstrumentor
from .redis_instrumentor import RedisInstrumentor
from .timescaledb_instrumentor import TimescaleDBInstrumentor
from .vector_db_instrumentor import VectorDBInstrumentor

logger = logging.getLogger(__name__)


class MCPInstrumentorManager:  # pylint: disable=R0903
    """Manager for MCP (Model Context Protocol) tool instrumentation"""

    def __init__(self, config: OTelConfig):
        self.config = config
        self.instrumentors = []

    def instrument_all(self, fail_on_error: bool = False):  # pylint: disable=R0912, R0915
        """Instrument all detected MCP tools"""

        success_count = 0
        failure_count = 0

        # HTTP/API instrumentation (disabled by default to avoid conflicts)
        if self.config.enable_http_instrumentation:
            try:
                logger.info("Instrumenting HTTP/API calls")
                # CRITICAL: Do NOT instrument requests library when using OTLP HTTP exporters
                # RequestsInstrumentor patches requests.Session at class level, breaking OTLP exporters
                # that use requests internally. The OTEL_PYTHON_REQUESTS_EXCLUDED_URLS doesn't help
                # because it only works at request-time, not at instrumentation-time.
                #
                # TODO: Find a way to instrument user requests without breaking OTLP exporters
                # RequestsInstrumentor().instrument()

                logger.warning(
                    "Requests library instrumentation is disabled to prevent conflicts with OTLP exporters"
                )

                # HTTPx is safe to instrument
                HTTPXClientInstrumentor().instrument()
                api_instrumentor = APIInstrumentor(self.config)
                api_instrumentor.instrument(self.config)
                logger.info("[OK] HTTP/API instrumentation enabled (requests library excluded)")
                success_count += 1
            except ImportError as e:
                failure_count += 1
                logger.debug(
                    "[SKIP] HTTP/API instrumentation skipped due to missing dependency: %s", e
                )
            except Exception as e:
                failure_count += 1
                logger.error("[ERROR] Failed to instrument HTTP/API: %s", e, exc_info=True)
                if fail_on_error:
                    raise
        else:
            logger.info("HTTP/API instrumentation disabled (enable_http_instrumentation=False)")

        # Database instrumentation
        try:
            logger.info("Instrumenting databases")
            db_instrumentor = DatabaseInstrumentor(self.config)
            result = db_instrumentor.instrument()
            if result > 0:
                success_count += 1
                logger.info("[OK] Database instrumentation enabled (%s databases)", result)
        except ImportError as e:
            failure_count += 1
            logger.debug("[SKIP] Database instrumentation skipped due to missing dependency: %s", e)
        except Exception as e:
            failure_count += 1
            logger.error("[ERROR] Failed to instrument databases: %s", e, exc_info=True)
            if fail_on_error:
                raise

        # Elasticsearch instrumentation
        try:
            logger.info("Instrumenting Elasticsearch")
            es_instrumentor = ElasticsearchInstrumentor(self.config)
            es_instrumentor.instrument()
            success_count += 1
        except ImportError as e:
            failure_count += 1
            logger.debug(
                "[SKIP] Elasticsearch instrumentation skipped due to missing dependency: %s", e
            )
        except Exception as e:
            failure_count += 1
            logger.error("[ERROR] Failed to instrument Elasticsearch: %s", e, exc_info=True)
            if fail_on_error:
                raise

        # TimescaleDB instrumentation
        try:
            logger.info("Instrumenting TimescaleDB")
            timescaledb_instrumentor = TimescaleDBInstrumentor(self.config)
            if timescaledb_instrumentor.instrument():
                success_count += 1
                logger.info("[OK] TimescaleDB instrumentation enabled")
        except ImportError as e:
            failure_count += 1
            logger.debug(
                "[SKIP] TimescaleDB instrumentation skipped due to missing dependency: %s", e
            )
        except Exception as e:
            failure_count += 1
            logger.error("[ERROR] Failed to instrument TimescaleDB: %s", e, exc_info=True)
            if fail_on_error:
                raise

        # OpenSearch instrumentation
        try:
            logger.info("Instrumenting OpenSearch")
            opensearch_instrumentor = OpenSearchInstrumentor(self.config)
            opensearch_instrumentor.instrument()
            success_count += 1
        except ImportError as e:
            failure_count += 1
            logger.debug(
                "[SKIP] OpenSearch instrumentation skipped due to missing dependency: %s", e
            )
        except Exception as e:
            failure_count += 1
            logger.error("[ERROR] Failed to instrument OpenSearch: %s", e, exc_info=True)
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
            logger.debug("[SKIP] Redis instrumentation skipped due to missing dependency: %s", e)
        except Exception as e:
            failure_count += 1
            logger.error("[ERROR] Failed to instrument Redis: %s", e, exc_info=True)
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
            logger.debug("[SKIP] Kafka instrumentation skipped due to missing dependency: %s", e)
        except Exception as e:
            failure_count += 1
            logger.error("[ERROR] Failed to instrument Kafka: %s", e, exc_info=True)
            if fail_on_error:
                raise

        # MinIO instrumentation
        try:
            logger.info("Instrumenting MinIO")
            minio_instrumentor = MinIOInstrumentor(self.config)
            if minio_instrumentor.instrument():
                success_count += 1
                logger.info("[OK] MinIO instrumentation enabled")
        except ImportError as e:
            failure_count += 1
            logger.debug("[SKIP] MinIO instrumentation skipped due to missing dependency: %s", e)
        except Exception as e:
            failure_count += 1
            logger.error("[ERROR] Failed to instrument MinIO: %s", e, exc_info=True)
            if fail_on_error:
                raise

        # RabbitMQ instrumentation
        try:
            logger.info("Instrumenting RabbitMQ")
            rabbitmq_instrumentor = RabbitMQInstrumentor(self.config)
            if rabbitmq_instrumentor.instrument():
                success_count += 1
                logger.info("[OK] RabbitMQ instrumentation enabled")
        except ImportError as e:
            failure_count += 1
            logger.debug("[SKIP] RabbitMQ instrumentation skipped due to missing dependency: %s", e)
        except Exception as e:
            failure_count += 1
            logger.error("[ERROR] Failed to instrument RabbitMQ: %s", e, exc_info=True)
            if fail_on_error:
                raise

        # Vector DB instrumentation
        try:
            logger.info("Instrumenting Vector DBs")
            vector_db_instrumentor = VectorDBInstrumentor(self.config)
            result = vector_db_instrumentor.instrument()
            if result > 0:
                success_count += 1
                logger.info("[OK] Vector DB instrumentation enabled (%s databases)", result)
        except ImportError as e:
            failure_count += 1
            logger.debug(
                "[SKIP] Vector DB instrumentation skipped due to missing dependency: %s", e
            )
        except Exception as e:
            failure_count += 1
            logger.error("[ERROR] Failed to instrument Vector DBs: %s", e, exc_info=True)
            if fail_on_error:
                raise

        logger.info(
            "MCP instrumentation summary: %s succeeded, %s failed",
            success_count,
            failure_count,
        )
