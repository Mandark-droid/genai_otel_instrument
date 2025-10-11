"""OpenTelemetry instrumentor for various database clients.

This module provides the `DatabaseInstrumentor` class, which automatically
instruments popular Python database libraries such as SQLAlchemy, psycopg2,
pymongo, and mysql, enabling tracing of database operations within GenAI applications.
"""

import logging
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.instrumentation.pymongo import PymongoInstrumentor
from opentelemetry.instrumentation.mysql import MySQLInstrumentor
from ..config import OTelConfig

logger = logging.getLogger(__name__)


class DatabaseInstrumentor:  # pylint: disable=R0903
    """Instrument various database clients"""

    def __init__(self, config: OTelConfig):
        self.config = config

    def instrument(self):
        """Instrument all detected database libraries"""
        instrumented_count = 0

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

        return instrumented_count
