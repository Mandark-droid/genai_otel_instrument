"""Manager for OpenTelemetry instrumentation of Model Context Protocol (MCP) tools.

This module provides the `MCPInstrumentorManager` class, which orchestrates
the automatic instrumentation of various MCP tools, including databases, caching
layers, message queues, vector databases, and generic API calls. It ensures
that these components are integrated into the OpenTelemetry tracing and metrics
system.
"""

import asyncio
import logging

import httpx
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

from ..config import OTelConfig
from .api_instrumentor import APIInstrumentor
from .database_instrumentor import DatabaseInstrumentor
from .kafka_instrumentor import KafkaInstrumentor
from .redis_instrumentor import RedisInstrumentor
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
                logger.info("✓ HTTP/API instrumentation enabled (requests library excluded)")
                success_count += 1
            except ImportError as e:
                failure_count += 1
                logger.debug(f"✗ HTTP/API instrumentation skipped due to missing dependency: {e}")
            except Exception as e:
                failure_count += 1
                logger.error(f"✗ Failed to instrument HTTP/API: {e}", exc_info=True)
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
            f"MCP instrumentation summary: {success_count} succeeded, " f"{failure_count} failed"
        )
