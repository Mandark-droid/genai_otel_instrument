"""OpenTelemetry instrumentor for TimescaleDB operations.

This module provides the ``TimescaleDBInstrumentor`` class, which automatically
instruments TimescaleDB-specific operations such as hypertable creation,
continuous aggregates, and compression, enabling tracing of time-series
database operations within GenAI applications.
"""

import logging

import wrapt
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from ..config import OTelConfig

logger = logging.getLogger(__name__)


class TimescaleDBInstrumentor:  # pylint: disable=R0903
    """Instrument TimescaleDB-specific operations"""

    def __init__(self, config: OTelConfig):
        self.config = config
        self.tracer = trace.get_tracer(__name__)

    def instrument(self):
        """Instrument TimescaleDB operations via psycopg2 cursor wrapping"""
        try:
            import psycopg2  # noqa: F401
            from psycopg2.extensions import cursor as Psycopg2Cursor  # noqa: F401

            tracer = self.tracer

            def wrapped_execute(wrapped, instance, args, kwargs):
                query = args[0] if args else kwargs.get("query", "")
                query_upper = str(query).upper().strip()

                # Detect TimescaleDB-specific operations
                timescale_op = _detect_timescale_operation(query_upper)

                if timescale_op:
                    span_name = f"timescaledb.{timescale_op}"
                    with tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
                        span.set_attribute("db.system", "timescaledb")
                        span.set_attribute("db.operation", timescale_op)
                        try:
                            result = wrapped(*args, **kwargs)
                            span.set_status(Status(StatusCode.OK))
                            return result
                        except Exception as e:
                            span.set_status(Status(StatusCode.ERROR, str(e)))
                            span.record_exception(e)
                            raise
                else:
                    return wrapped(*args, **kwargs)

            wrapt.wrap_function_wrapper("psycopg2.extensions", "cursor.execute", wrapped_execute)
            logger.info("TimescaleDB instrumentation enabled")
            return True

        except ImportError:
            logger.debug("psycopg2 not installed, skipping TimescaleDB instrumentation")
            return False
        except Exception as e:
            logger.error("Failed to instrument TimescaleDB: %s", e, exc_info=True)
            return False


def _detect_timescale_operation(query_upper: str) -> str:
    """Detect TimescaleDB-specific operations from SQL query text.

    Returns the operation name if detected, empty string otherwise.
    """
    # Order matters: longer/more specific patterns must come before shorter ones
    # e.g. DECOMPRESS_CHUNK before COMPRESS_CHUNK, CREATE_DISTRIBUTED_HYPERTABLE before CREATE_HYPERTABLE
    timescale_patterns = [
        ("CREATE_DISTRIBUTED_HYPERTABLE", "create_distributed_hypertable"),
        ("CREATE_HYPERTABLE", "create_hypertable"),
        ("REMOVE_COMPRESSION_POLICY", "remove_compression_policy"),
        ("ADD_COMPRESSION_POLICY", "add_compression_policy"),
        ("REMOVE_RETENTION_POLICY", "remove_retention_policy"),
        ("ADD_RETENTION_POLICY", "add_retention_policy"),
        ("REMOVE_CONTINUOUS_AGGREGATE_POLICY", "remove_continuous_aggregate_policy"),
        ("ADD_CONTINUOUS_AGGREGATE_POLICY", "add_continuous_aggregate_policy"),
        ("CREATE MATERIALIZED VIEW", "create_continuous_aggregate"),
        ("TIME_BUCKET", "time_bucket_query"),
        ("DECOMPRESS_CHUNK", "decompress_chunk"),
        ("COMPRESS_CHUNK", "compress_chunk"),
        ("CHUNKS_DETAILED_SIZE", "chunks_detailed_size"),
        ("HYPERTABLE_SIZE", "hypertable_size"),
        ("DROP_CHUNKS", "drop_chunks"),
        ("SHOW_CHUNKS", "show_chunks"),
    ]

    for pattern, operation in timescale_patterns:
        if pattern in query_upper:
            return operation
    return ""
