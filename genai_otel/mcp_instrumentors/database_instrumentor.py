"""OpenTelemetry instrumentor for various database clients.

This module provides the `DatabaseInstrumentor` class, which automatically
instruments popular Python database libraries such as SQLAlchemy, psycopg2,
pymongo, and mysql, enabling tracing of database operations within GenAI applications.

This instrumentor uses a hybrid approach:
1. Built-in OTel instrumentors create spans with full trace context
2. Custom wrapt wrappers add MCP-specific metrics (duration, payload sizes)
"""

import json
import logging
import time

import wrapt
from opentelemetry.instrumentation.mysql import MySQLInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.instrumentation.pymongo import PymongoInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

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
            logger.debug("PostgreSQL MCP metrics enabled")
        except Exception as e:
            logger.debug(f"Failed to add PostgreSQL MCP metrics: {e}")

    def _add_pymongo_metrics(self):
        """Add MCP metrics collection to pymongo collection methods."""
        try:
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
            logger.debug("MongoDB MCP metrics enabled")
        except Exception as e:
            logger.debug(f"Failed to add MongoDB MCP metrics: {e}")

    def _add_mysql_metrics(self):
        """Add MCP metrics collection to MySQL cursor execute methods."""
        try:
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

                # Estimate request size (query + params)
                try:
                    query = args[0] if args else ""
                    params = (
                        args[1] if len(args) > 1 else kwargs.get("vars") or kwargs.get("params")
                    )
                    request_size = len(str(query))
                    if params:
                        try:
                            request_size += len(json.dumps(params, default=str))
                        except (TypeError, ValueError):
                            request_size += len(str(params))

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

                # Estimate payload sizes
                try:
                    # Request size: serialize args and kwargs
                    request_size = 0
                    if args:
                        for arg in args:
                            if arg is not None:
                                try:
                                    request_size += len(json.dumps(arg, default=str))
                                except (TypeError, ValueError):
                                    request_size += len(str(arg))
                    if kwargs:
                        for val in kwargs.values():
                            if val is not None:
                                try:
                                    request_size += len(json.dumps(val, default=str))
                                except (TypeError, ValueError):
                                    request_size += len(str(val))

                    if self.mcp_request_size_histogram and request_size > 0:
                        self.mcp_request_size_histogram.record(
                            request_size, {"db.system": db_system, "mcp.operation": operation}
                        )

                    # Response size: estimate based on result type
                    response_size = 0
                    if result is not None:
                        if isinstance(result, dict):
                            try:
                                response_size = len(json.dumps(result, default=str))
                            except (TypeError, ValueError):
                                response_size = len(str(result))
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
