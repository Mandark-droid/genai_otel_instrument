import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.mcp_instrumentors.database_instrumentor import DatabaseInstrumentor


class TestDatabaseInstrumentor(unittest.TestCase):
    """Tests for DatabaseInstrumentor"""

    def test_init(self):
        """Test that __init__ sets config correctly."""
        config = OTelConfig()
        instrumentor = DatabaseInstrumentor(config)
        self.assertEqual(instrumentor.config, config)

    @patch("genai_otel.mcp_instrumentors.database_instrumentor.logger")
    @patch("genai_otel.mcp_instrumentors.database_instrumentor.SQLAlchemyInstrumentor")
    def test_instrument_sqlalchemy_success(self, mock_sqlalchemy, mock_logger):
        """Test successful SQLAlchemy instrumentation."""
        config = OTelConfig()
        instrumentor = DatabaseInstrumentor(config)

        # Mock the instrumentor instance
        mock_instance = MagicMock()
        mock_sqlalchemy.return_value = mock_instance

        count = instrumentor.instrument()

        # Verify SQLAlchemyInstrumentor was called
        mock_sqlalchemy.assert_called_once()
        mock_instance.instrument.assert_called_once()
        mock_logger.info.assert_any_call("SQLAlchemy instrumentation enabled")
        # Count should be at least 1 (SQLAlchemy was instrumented successfully)
        self.assertGreaterEqual(count, 1)

    @patch("genai_otel.mcp_instrumentors.database_instrumentor.logger")
    @patch("genai_otel.mcp_instrumentors.database_instrumentor.SQLAlchemyInstrumentor")
    def test_instrument_sqlalchemy_import_error(self, mock_sqlalchemy, mock_logger):
        """Test SQLAlchemy instrumentation with ImportError."""
        config = OTelConfig()
        instrumentor = DatabaseInstrumentor(config)

        # Make SQLAlchemyInstrumentor().instrument() raise ImportError
        mock_instance = MagicMock()
        mock_instance.instrument.side_effect = ImportError("SQLAlchemy not found")
        mock_sqlalchemy.return_value = mock_instance

        count = instrumentor.instrument()

        mock_logger.debug.assert_any_call("SQLAlchemy not installed, skipping instrumentation.")
        # Other databases may still be instrumented, so count could be > 0
        self.assertGreaterEqual(count, 0)

    @patch("genai_otel.mcp_instrumentors.database_instrumentor.logger")
    @patch("genai_otel.mcp_instrumentors.database_instrumentor.SQLAlchemyInstrumentor")
    def test_instrument_sqlalchemy_general_exception(self, mock_sqlalchemy, mock_logger):
        """Test SQLAlchemy instrumentation with general exception."""
        config = OTelConfig()
        instrumentor = DatabaseInstrumentor(config)

        # Make SQLAlchemyInstrumentor().instrument() raise RuntimeError
        mock_instance = MagicMock()
        mock_instance.instrument.side_effect = RuntimeError("Unexpected error")
        mock_sqlalchemy.return_value = mock_instance

        count = instrumentor.instrument()

        mock_logger.warning.assert_any_call("SQLAlchemy instrumentation failed: Unexpected error")
        # Other databases may still be instrumented, so count could be > 0
        self.assertGreaterEqual(count, 0)

    @patch("genai_otel.mcp_instrumentors.database_instrumentor.logger")
    @patch("genai_otel.mcp_instrumentors.database_instrumentor.Psycopg2Instrumentor")
    def test_instrument_psycopg2_success(self, mock_psycopg2, mock_logger):
        """Test successful PostgreSQL (psycopg2) instrumentation."""
        config = OTelConfig()
        instrumentor = DatabaseInstrumentor(config)

        # Mock the instrumentor instance
        mock_instance = MagicMock()
        mock_psycopg2.return_value = mock_instance

        count = instrumentor.instrument()

        # Verify Psycopg2Instrumentor was called
        mock_psycopg2.assert_called_once()
        mock_instance.instrument.assert_called_once()
        mock_logger.info.assert_any_call("PostgreSQL (psycopg2) instrumentation enabled")
        self.assertGreaterEqual(count, 1)

    @patch("genai_otel.mcp_instrumentors.database_instrumentor.logger")
    @patch("genai_otel.mcp_instrumentors.database_instrumentor.Psycopg2Instrumentor")
    def test_instrument_psycopg2_import_error(self, mock_psycopg2, mock_logger):
        """Test PostgreSQL instrumentation with ImportError."""
        config = OTelConfig()
        instrumentor = DatabaseInstrumentor(config)

        # Make Psycopg2Instrumentor().instrument() raise ImportError
        mock_instance = MagicMock()
        mock_instance.instrument.side_effect = ImportError("psycopg2 not found")
        mock_psycopg2.return_value = mock_instance

        count = instrumentor.instrument()

        mock_logger.debug.assert_any_call("Psycopg2 not installed, skipping instrumentation.")
        self.assertGreaterEqual(count, 0)

    @patch("genai_otel.mcp_instrumentors.database_instrumentor.logger")
    @patch("genai_otel.mcp_instrumentors.database_instrumentor.Psycopg2Instrumentor")
    def test_instrument_psycopg2_general_exception(self, mock_psycopg2, mock_logger):
        """Test PostgreSQL instrumentation with general exception."""
        config = OTelConfig()
        instrumentor = DatabaseInstrumentor(config)

        # Make Psycopg2Instrumentor().instrument() raise RuntimeError
        mock_instance = MagicMock()
        mock_instance.instrument.side_effect = RuntimeError("Unexpected error")
        mock_psycopg2.return_value = mock_instance

        count = instrumentor.instrument()

        mock_logger.warning.assert_any_call("PostgreSQL instrumentation failed: Unexpected error")
        self.assertGreaterEqual(count, 0)

    @patch("genai_otel.mcp_instrumentors.database_instrumentor.logger")
    @patch("genai_otel.mcp_instrumentors.database_instrumentor.PymongoInstrumentor")
    def test_instrument_pymongo_success(self, mock_pymongo, mock_logger):
        """Test successful MongoDB (pymongo) instrumentation."""
        config = OTelConfig()
        instrumentor = DatabaseInstrumentor(config)

        # Mock the instrumentor instance
        mock_instance = MagicMock()
        mock_pymongo.return_value = mock_instance

        count = instrumentor.instrument()

        # Verify PymongoInstrumentor was called
        mock_pymongo.assert_called_once()
        mock_instance.instrument.assert_called_once()
        mock_logger.info.assert_any_call("MongoDB instrumentation enabled")
        self.assertGreaterEqual(count, 1)

    @patch("genai_otel.mcp_instrumentors.database_instrumentor.logger")
    @patch("genai_otel.mcp_instrumentors.database_instrumentor.PymongoInstrumentor")
    def test_instrument_pymongo_import_error(self, mock_pymongo, mock_logger):
        """Test MongoDB instrumentation with ImportError."""
        config = OTelConfig()
        instrumentor = DatabaseInstrumentor(config)

        # Make PymongoInstrumentor().instrument() raise ImportError
        mock_instance = MagicMock()
        mock_instance.instrument.side_effect = ImportError("pymongo not found")
        mock_pymongo.return_value = mock_instance

        count = instrumentor.instrument()

        mock_logger.debug.assert_any_call("Pymongo not installed, skipping instrumentation.")
        self.assertGreaterEqual(count, 0)

    @patch("genai_otel.mcp_instrumentors.database_instrumentor.logger")
    @patch("genai_otel.mcp_instrumentors.database_instrumentor.PymongoInstrumentor")
    def test_instrument_pymongo_general_exception(self, mock_pymongo, mock_logger):
        """Test MongoDB instrumentation with general exception."""
        config = OTelConfig()
        instrumentor = DatabaseInstrumentor(config)

        # Make PymongoInstrumentor().instrument() raise RuntimeError
        mock_instance = MagicMock()
        mock_instance.instrument.side_effect = RuntimeError("Unexpected error")
        mock_pymongo.return_value = mock_instance

        count = instrumentor.instrument()

        mock_logger.warning.assert_any_call("MongoDB instrumentation failed: Unexpected error")
        self.assertGreaterEqual(count, 0)

    @patch("genai_otel.mcp_instrumentors.database_instrumentor.logger")
    @patch("genai_otel.mcp_instrumentors.database_instrumentor.MySQLInstrumentor")
    def test_instrument_mysql_success(self, mock_mysql, mock_logger):
        """Test successful MySQL instrumentation."""
        config = OTelConfig()
        instrumentor = DatabaseInstrumentor(config)

        # Mock the instrumentor instance
        mock_instance = MagicMock()
        mock_mysql.return_value = mock_instance

        count = instrumentor.instrument()

        # Verify MySQLInstrumentor was called
        mock_mysql.assert_called_once()
        mock_instance.instrument.assert_called_once()
        mock_logger.info.assert_any_call("MySQL instrumentation enabled")
        self.assertGreaterEqual(count, 1)

    @patch("genai_otel.mcp_instrumentors.database_instrumentor.logger")
    @patch("genai_otel.mcp_instrumentors.database_instrumentor.MySQLInstrumentor")
    def test_instrument_mysql_import_error(self, mock_mysql, mock_logger):
        """Test MySQL instrumentation with ImportError."""
        config = OTelConfig()
        instrumentor = DatabaseInstrumentor(config)

        # Make MySQLInstrumentor().instrument() raise ImportError
        mock_instance = MagicMock()
        mock_instance.instrument.side_effect = ImportError("mysql not found")
        mock_mysql.return_value = mock_instance

        count = instrumentor.instrument()

        mock_logger.debug.assert_any_call("MySQL-python not installed, skipping instrumentation.")
        self.assertGreaterEqual(count, 0)

    @patch("genai_otel.mcp_instrumentors.database_instrumentor.logger")
    @patch("genai_otel.mcp_instrumentors.database_instrumentor.MySQLInstrumentor")
    def test_instrument_mysql_general_exception(self, mock_mysql, mock_logger):
        """Test MySQL instrumentation with general exception."""
        config = OTelConfig()
        instrumentor = DatabaseInstrumentor(config)

        # Make MySQLInstrumentor().instrument() raise RuntimeError
        mock_instance = MagicMock()
        mock_instance.instrument.side_effect = RuntimeError("Unexpected error")
        mock_mysql.return_value = mock_instance

        count = instrumentor.instrument()

        mock_logger.warning.assert_any_call("MySQL instrumentation failed: Unexpected error")
        self.assertGreaterEqual(count, 0)

    @patch("genai_otel.mcp_instrumentors.database_instrumentor.logger")
    @patch("genai_otel.mcp_instrumentors.database_instrumentor.MySQLInstrumentor")
    @patch("genai_otel.mcp_instrumentors.database_instrumentor.PymongoInstrumentor")
    @patch("genai_otel.mcp_instrumentors.database_instrumentor.Psycopg2Instrumentor")
    @patch("genai_otel.mcp_instrumentors.database_instrumentor.SQLAlchemyInstrumentor")
    def test_instrument_all_databases_success(
        self, mock_sqlalchemy, mock_psycopg2, mock_pymongo, mock_mysql, mock_logger
    ):
        """Test instrumentation of all databases successfully."""
        config = OTelConfig()
        instrumentor = DatabaseInstrumentor(config)

        # Mock all instrumentor instances
        for mock_db in [mock_sqlalchemy, mock_psycopg2, mock_pymongo, mock_mysql]:
            mock_instance = MagicMock()
            mock_db.return_value = mock_instance

        count = instrumentor.instrument()

        # All 4 databases should be instrumented
        self.assertEqual(count, 4)
        mock_logger.info.assert_any_call("SQLAlchemy instrumentation enabled")
        mock_logger.info.assert_any_call("PostgreSQL (psycopg2) instrumentation enabled")
        mock_logger.info.assert_any_call("MongoDB instrumentation enabled")
        mock_logger.info.assert_any_call("MySQL instrumentation enabled")


if __name__ == "__main__":
    unittest.main(verbosity=2)
