"""OpenTelemetry instrumentor for TimescaleDB operations.

This module provides the ``TimescaleDBInstrumentor`` class, which automatically
instruments TimescaleDB-specific operations such as hypertable creation,
continuous aggregates, and compression, enabling tracing of time-series
database operations within GenAI applications.

Performance note
----------------
Operation detection runs on **every** psycopg2 query process-wide (the
instrumentor hijacks ``psycopg2.connect`` to inject a tracing cursor). To avoid
uppercasing/copying multi-MB statements on the hot path, only a bounded prefix
of the query (``_OPERATION_SCAN_LIMIT`` characters) is inspected to detect the
TimescaleDB operation.

Content bounding
----------------
Error-status descriptions are bounded to ``config.content_max_length`` (0 =
unlimited) so a failing statement's text is not echoed unbounded onto the span.

Double-wrap safety
------------------
``instrument()`` is idempotent: repeated calls do not re-hijack
``psycopg2.connect``. A best-effort ``uninstrument()`` restores the original.
"""

import logging

import wrapt
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from ..config import OTelConfig

logger = logging.getLogger(__name__)

# Marker appended when a captured error message is truncated.
_TRUNCATION_SUFFIX = "...[truncated]"

# Maximum number of leading characters of a query to inspect when detecting a
# TimescaleDB operation. Detection only needs the operation token, which for
# these statements appears near the start; scanning a bounded prefix keeps the
# per-query cost constant regardless of statement size.
_OPERATION_SCAN_LIMIT = 1024

# Identity-tracked registry of already-instrumented ``psycopg2`` module objects.
_INSTRUMENTED_MODULES = []


def _bound_text(value, config) -> str:
    """Bound a captured string to ``config.content_max_length`` when > 0.

    A cap of 0 (or an unset/unknown config) means unlimited. ``getattr`` with a
    default keeps this safe against ``MagicMock(spec=OTelConfig)`` in tests.
    """
    text = value if isinstance(value, str) else str(value)
    max_len = getattr(config, "content_max_length", 0) or 0
    if max_len > 0 and len(text) > max_len:
        return text[:max_len] + _TRUNCATION_SUFFIX
    return text


def _is_module_instrumented(module) -> bool:
    """Return True if ``module`` (by identity) was already instrumented."""
    return any(m is module for m in _INSTRUMENTED_MODULES)


class TimescaleDBInstrumentor:  # pylint: disable=R0903
    """Instrument TimescaleDB-specific operations"""

    def __init__(self, config: OTelConfig):
        self.config = config
        self.tracer = trace.get_tracer(__name__)

    def instrument(self):
        """Instrument TimescaleDB operations via psycopg2 connection wrapping.

        Wraps ``psycopg2.connect`` so that every connection returned uses a
        custom cursor subclass whose ``execute`` method creates TimescaleDB-
        specific spans for recognised operations.
        """
        try:
            import psycopg2  # noqa: F401

            # Double-wrap guard: do not re-hijack psycopg2.connect on repeat calls.
            if _is_module_instrumented(psycopg2):
                logger.debug("TimescaleDB already instrumented; skipping to prevent double-wrap")
                return True

            tracer = self.tracer
            config = self.config

            class _TimescaleCursor(psycopg2.extensions.cursor):
                """Cursor subclass that traces TimescaleDB-specific operations."""

                def execute(self, query, vars=None):  # pylint: disable=W0622
                    # Inspect only a bounded prefix to detect the operation; this
                    # avoids uppercasing/copying very large statements on every query.
                    timescale_op = _operation_from_query(query)

                    if timescale_op:
                        span_name = f"timescaledb.{timescale_op}"
                        with tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
                            span.set_attribute("db.system", "timescaledb")
                            span.set_attribute("db.operation", timescale_op)
                            try:
                                result = super().execute(query, vars)
                                span.set_status(Status(StatusCode.OK))
                                return result
                            except Exception as e:
                                span.set_status(Status(StatusCode.ERROR, _bound_text(e, config)))
                                span.record_exception(e)
                                raise
                    else:
                        return super().execute(query, vars)

            def wrapped_connect(wrapped, instance, args, kwargs):
                # Inject our cursor factory unless the caller specified one
                if "cursor_factory" not in kwargs:
                    kwargs["cursor_factory"] = _TimescaleCursor
                return wrapped(*args, **kwargs)

            wrapt.wrap_function_wrapper("psycopg2", "connect", wrapped_connect)
            _INSTRUMENTED_MODULES.append(psycopg2)
            logger.info("TimescaleDB instrumentation enabled")
            return True

        except ImportError:
            logger.debug("psycopg2 not installed, skipping TimescaleDB instrumentation")
            return False
        except Exception as e:
            logger.error("Failed to instrument TimescaleDB: %s", e, exc_info=True)
            return False

    def uninstrument(self):
        """Best-effort removal of TimescaleDB instrumentation.

        Restores the original ``psycopg2.connect`` (via wrapt's ``__wrapped__``)
        and clears the double-wrap registry. Returns True if it was instrumented.
        """
        try:
            import psycopg2  # noqa: F401
        except ImportError:
            return False

        was_instrumented = _is_module_instrumented(psycopg2)
        try:
            connect = getattr(psycopg2, "connect", None)
            original = getattr(connect, "__wrapped__", None)
            if original is not None:
                psycopg2.connect = original
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("Failed to unwrap psycopg2.connect: %s", e)
        _INSTRUMENTED_MODULES[:] = [m for m in _INSTRUMENTED_MODULES if m is not psycopg2]
        if was_instrumented:
            logger.info("TimescaleDB instrumentation removed")
        return was_instrumented


def _operation_from_query(query) -> str:
    """Detect a TimescaleDB operation from a query, inspecting only a bounded prefix.

    Casting/uppercasing is applied to at most ``_OPERATION_SCAN_LIMIT`` leading
    characters so the per-query cost stays constant for very large statements.
    """
    prefix = str(query)[:_OPERATION_SCAN_LIMIT].upper().strip()
    return _detect_timescale_operation(prefix)


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
