"""OpenTelemetry instrumentor for various database clients.

This module provides the `DatabaseInstrumentor` class, which automatically
instruments popular Python database libraries such as SQLAlchemy, psycopg2,
pymongo, and mysql, enabling tracing of database operations within GenAI applications.

This instrumentor uses a hybrid approach:
1. Built-in OTel instrumentors create spans with full trace context
2. Custom wrapt wrappers add MCP-specific metrics (duration, payload sizes)
"""

import logging
import time

import wrapt

try:
    from opentelemetry.instrumentation.mysql import MySQLInstrumentor
except ImportError:
    MySQLInstrumentor = None

try:
    from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
except ImportError:
    Psycopg2Instrumentor = None

try:
    from opentelemetry.instrumentation.pymongo import PymongoInstrumentor
except ImportError:
    PymongoInstrumentor = None

try:
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
except ImportError:
    SQLAlchemyInstrumentor = None

from ..config import OTelConfig
from .base import BaseMCPInstrumentor

logger = logging.getLogger(__name__)

# Conditional imports for database libraries
try:
    import psycopg2
    from psycopg2.extensions import cursor as Psycopg2Cursor

    PSYCOPG2_AVAILABLE = True
except ImportError:
    psycopg2 = None
    Psycopg2Cursor = None
    PSYCOPG2_AVAILABLE = False

try:
    import pymongo
    from pymongo.collection import Collection as PymongoCollection

    PYMONGO_AVAILABLE = True
except ImportError:
    pymongo = None
    PymongoCollection = None
    PYMONGO_AVAILABLE = False

try:
    import mysql.connector
    from mysql.connector.cursor import MySQLCursor

    MYSQL_AVAILABLE = True
except ImportError:
    mysql = None
    MySQLCursor = None
    MYSQL_AVAILABLE = False


# Sampling cap for the cheap payload-size estimator. Large containers are sized
# by sampling this many elements and extrapolating, instead of serializing all.
_SIZE_SAMPLE_LIMIT = 10
_SIZE_MAX_DEPTH = 2

# Identity-tracked registry of cursor/collection classes already wrapped with the
# custom MCP-metrics wrappers, to prevent stacking on repeated instrument() calls.
# Keyed on object identity so mocked classes in unit tests never collide.
_METRICS_WRAPPED = []


def _estimate_payload_size(obj, depth=0):
    """Cheap, bounded estimate of a payload's serialized size in bytes.

    This runs on the database hot path purely to feed request/response-size
    histograms, so it must stay roughly constant-time regardless of payload
    size. Instead of fully serializing (e.g. ``json.dumps`` on a multi-MB
    ``executemany`` batch), it samples a bounded number of elements from
    containers and extrapolates by the container length.
    """
    if obj is None:
        return 0
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return len(obj)
    if isinstance(obj, str):
        return len(obj)
    if isinstance(obj, bool):
        return 1
    if isinstance(obj, int):
        return 8
    if isinstance(obj, float):
        return 8
    if depth >= _SIZE_MAX_DEPTH:
        # Stop recursing; approximate deeper structures by their length.
        try:
            return len(obj)
        except TypeError:
            return 16
    if isinstance(obj, dict):
        n = len(obj)
        if n == 0:
            return 2
        total = 0
        sampled = 0
        for key, value in obj.items():
            total += _estimate_payload_size(key, depth + 1)
            total += _estimate_payload_size(value, depth + 1)
            sampled += 1
            if sampled >= _SIZE_SAMPLE_LIMIT:
                break
        return int(total / sampled * n) + 2
    if isinstance(obj, (list, tuple, set, frozenset)):
        n = len(obj)
        if n == 0:
            return 2
        total = 0
        sampled = 0
        for item in obj:
            total += _estimate_payload_size(item, depth + 1)
            sampled += 1
            if sampled >= _SIZE_SAMPLE_LIMIT:
                break
        return int(total / sampled * n) + 2
    # Opaque object: cheap constant; do NOT str()/serialize it (could be huge).
    return 16


def _metrics_already_wrapped(cls) -> bool:
    """Return True if ``cls`` (by identity) already has the custom metric wrappers."""
    return any(c is cls for c in _METRICS_WRAPPED)


class DatabaseInstrumentor(BaseMCPInstrumentor):  # pylint: disable=R0903
    """Instrument various database clients with traces and MCP metrics.

    Uses a hybrid approach:
    - Built-in OTel instrumentors for spans/traces
    - Custom wrappers for MCP-specific metrics
    """

    def __init__(self, config: OTelConfig):
        super().__init__()
        self.config = config

    def instrument(self):
        """Instrument all detected database libraries with traces and MCP metrics.

        Uses hybrid approach:
        1. Built-in OTel instrumentors for spans/traces
        2. Custom wrappers for MCP metrics (duration, payload sizes)
        """
        instrumented_count = 0

        # Step 1: Use built-in instrumentors for traces/spans
        # SQLAlchemy
        try:
            SQLAlchemyInstrumentor().instrument()
            logger.info("SQLAlchemy instrumentation enabled")
            instrumented_count += 1
        except ImportError:
            logger.debug("SQLAlchemy not installed, skipping instrumentation.")
        except Exception as e:
            logger.warning(f"SQLAlchemy instrumentation failed: {e}")

        # PostgreSQL (psycopg2)
        try:
            Psycopg2Instrumentor().instrument()
            logger.info("PostgreSQL (psycopg2) instrumentation enabled")
            instrumented_count += 1
        except ImportError:
            logger.debug("Psycopg2 not installed, skipping instrumentation.")
        except Exception as e:
            logger.warning(f"PostgreSQL instrumentation failed: {e}")

        # MongoDB
        try:
            PymongoInstrumentor().instrument()
            logger.info("MongoDB instrumentation enabled")
            instrumented_count += 1
        except ImportError:
            logger.debug("Pymongo not installed, skipping instrumentation.")
        except Exception as e:
            logger.warning(f"MongoDB instrumentation failed: {e}")

        # MySQL
        try:
            MySQLInstrumentor().instrument()
            logger.info("MySQL instrumentation enabled")
            instrumented_count += 1
        except ImportError:
            logger.debug("MySQL-python not installed, skipping instrumentation.")
        except Exception as e:
            logger.warning(f"MySQL instrumentation failed: {e}")

        # Step 2: Add custom MCP metrics wrappers
        if self.mcp_request_counter is not None:
            # Add metrics collection for databases that are available
            if PSYCOPG2_AVAILABLE:
                self._add_psycopg2_metrics()
            if PYMONGO_AVAILABLE:
                self._add_pymongo_metrics()
            if MYSQL_AVAILABLE:
                self._add_mysql_metrics()

        return instrumented_count

    def _add_psycopg2_metrics(self):
        """Add MCP metrics collection to psycopg2 cursor execute methods."""
        try:
            # Double-wrap guard: skip if this cursor class was already wrapped.
            if Psycopg2Cursor is not None and _metrics_already_wrapped(Psycopg2Cursor):
                logger.debug("PostgreSQL MCP metrics already enabled; skipping double-wrap")
                return
            # Wrap psycopg2 cursor execute methods
            if hasattr(Psycopg2Cursor, "execute"):
                wrapt.wrap_function_wrapper(
                    "psycopg2.extensions", "cursor.execute", self._db_execute_wrapper("psycopg2")
                )
            if hasattr(Psycopg2Cursor, "executemany"):
                wrapt.wrap_function_wrapper(
                    "psycopg2.extensions",
                    "cursor.executemany",
                    self._db_execute_wrapper("psycopg2"),
                )
            if Psycopg2Cursor is not None:
                _METRICS_WRAPPED.append(Psycopg2Cursor)
            logger.debug("PostgreSQL MCP metrics enabled")
        except Exception as e:
            logger.debug(f"Failed to add PostgreSQL MCP metrics: {e}")

    def _add_pymongo_metrics(self):
        """Add MCP metrics collection to pymongo collection methods."""
        try:
            # Double-wrap guard: skip if this collection class was already wrapped.
            if PymongoCollection is not None and _metrics_already_wrapped(PymongoCollection):
                logger.debug("MongoDB MCP metrics already enabled; skipping double-wrap")
                return
            # Wrap common pymongo collection methods
            methods_to_wrap = [
                "find",
                "find_one",
                "insert_one",
                "insert_many",
                "update_one",
                "update_many",
                "delete_one",
                "delete_many",
                "count_documents",
                "aggregate",
            ]
            for method_name in methods_to_wrap:
                if hasattr(PymongoCollection, method_name):
                    wrapt.wrap_function_wrapper(
                        "pymongo.collection",
                        f"Collection.{method_name}",
                        self._db_operation_wrapper("pymongo", method_name),
                    )
            if PymongoCollection is not None:
                _METRICS_WRAPPED.append(PymongoCollection)
            logger.debug("MongoDB MCP metrics enabled")
        except Exception as e:
            logger.debug(f"Failed to add MongoDB MCP metrics: {e}")

    def _add_mysql_metrics(self):
        """Add MCP metrics collection to MySQL cursor execute methods."""
        try:
            # Double-wrap guard: skip if this cursor class was already wrapped.
            if MySQLCursor is not None and _metrics_already_wrapped(MySQLCursor):
                logger.debug("MySQL MCP metrics already enabled; skipping double-wrap")
                return
            # Wrap MySQL cursor execute methods
            if hasattr(MySQLCursor, "execute"):
                wrapt.wrap_function_wrapper(
                    "mysql.connector.cursor",
                    "MySQLCursor.execute",
                    self._db_execute_wrapper("mysql"),
                )
            if hasattr(MySQLCursor, "executemany"):
                wrapt.wrap_function_wrapper(
                    "mysql.connector.cursor",
                    "MySQLCursor.executemany",
                    self._db_execute_wrapper("mysql"),
                )
            if MySQLCursor is not None:
                _METRICS_WRAPPED.append(MySQLCursor)
            logger.debug("MySQL MCP metrics enabled")
        except Exception as e:
            logger.debug(f"Failed to add MySQL MCP metrics: {e}")

    def _db_execute_wrapper(self, db_system: str):
        """Create a wrapper for database execute methods that records MCP metrics.

        Args:
            db_system: Name of the database system (e.g., "psycopg2", "mysql")

        Returns:
            Wrapper function compatible with wrapt
        """

        def wrapper(wrapped, instance, args, kwargs):
            start_time = time.time()
            try:
                result = wrapped(*args, **kwargs)
                return result
            finally:
                # Record duration
                duration = time.time() - start_time
                if self.mcp_duration_histogram:
                    self.mcp_duration_histogram.record(
                        duration, {"db.system": db_system, "mcp.operation": "execute"}
                    )

                # Record request count
                if self.mcp_request_counter:
                    self.mcp_request_counter.add(
                        1, {"db.system": db_system, "mcp.operation": "execute"}
                    )

                # Estimate request size (query + params).
                # Use a cheap, bounded estimate rather than fully serializing the
                # payload: a large executemany batch must not be re-serialized on
                # the hot path just to measure its size.
                try:
                    query = args[0] if args else ""
                    params = (
                        args[1] if len(args) > 1 else kwargs.get("vars") or kwargs.get("params")
                    )
                    request_size = _estimate_payload_size(query)
                    if params is not None:
                        request_size += _estimate_payload_size(params)

                    if self.mcp_request_size_histogram:
                        self.mcp_request_size_histogram.record(
                            request_size, {"db.system": db_system}
                        )

                    # Estimate response size from rowcount
                    if hasattr(instance, "rowcount") and instance.rowcount > 0:
                        # Rough estimate: 100 bytes per row
                        response_size = instance.rowcount * 100
                        if self.mcp_response_size_histogram:
                            self.mcp_response_size_histogram.record(
                                response_size, {"db.system": db_system}
                            )
                except Exception as e:
                    logger.debug(f"Failed to record payload size for {db_system}: {e}")

        return wrapper

    def _db_operation_wrapper(self, db_system: str, operation: str):
        """Create a wrapper for database operations that records MCP metrics.

        Args:
            db_system: Name of the database system (e.g., "pymongo")
            operation: Name of the operation (e.g., "find", "insert_one")

        Returns:
            Wrapper function compatible with wrapt
        """

        def wrapper(wrapped, instance, args, kwargs):
            start_time = time.time()
            # Initialize before the call so the finally block can reference it
            # even when the wrapped operation raises (avoids a NameError that
            # would otherwise mask the real exception).
            result = None
            try:
                result = wrapped(*args, **kwargs)
                return result
            finally:
                # Record duration
                duration = time.time() - start_time
                if self.mcp_duration_histogram:
                    self.mcp_duration_histogram.record(
                        duration, {"db.system": db_system, "mcp.operation": operation}
                    )

                # Record request count
                if self.mcp_request_counter:
                    self.mcp_request_counter.add(
                        1, {"db.system": db_system, "mcp.operation": operation}
                    )

                # Estimate payload sizes (cheap, bounded; see _estimate_payload_size)
                try:
                    # Request size: sample args and kwargs rather than serializing
                    request_size = 0
                    if args:
                        for arg in args:
                            if arg is not None:
                                request_size += _estimate_payload_size(arg)
                    if kwargs:
                        for val in kwargs.values():
                            if val is not None:
                                request_size += _estimate_payload_size(val)

                    if self.mcp_request_size_histogram and request_size > 0:
                        self.mcp_request_size_histogram.record(
                            request_size, {"db.system": db_system, "mcp.operation": operation}
                        )

                    # Response size: estimate based on result type
                    response_size = 0
                    if result is not None:
                        if isinstance(result, dict):
                            response_size = _estimate_payload_size(result)
                        elif isinstance(result, (list, tuple)):
                            response_size = len(result) * 100  # Estimate 100 bytes per item
                        elif isinstance(result, int):
                            response_size = 8  # Integer size
                        elif hasattr(result, "inserted_ids"):
                            response_size = len(str(result.inserted_ids))
                        elif hasattr(result, "matched_count"):
                            response_size = 8

                    if self.mcp_response_size_histogram and response_size > 0:
                        self.mcp_response_size_histogram.record(
                            response_size, {"db.system": db_system, "mcp.operation": operation}
                        )
                except Exception as e:
                    logger.debug(f"Failed to record payload size for {db_system}.{operation}: {e}")

        return wrapper
