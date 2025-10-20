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


class TestDatabaseInstrumentorMCPMetrics(unittest.TestCase):
    """Tests for DatabaseInstrumentor MCP metrics wrappers"""

    def setUp(self):
        """Set up test fixtures."""
        self.config = OTelConfig()
        self.instrumentor = DatabaseInstrumentor(self.config)

    def test_db_execute_wrapper_psycopg2(self):
        """Test that _db_execute_wrapper creates a valid wrapper."""
        wrapper = self.instrumentor._db_execute_wrapper("psycopg2")

        # Test that wrapper is callable
        self.assertTrue(callable(wrapper))

        # Mock a database execute function
        mock_execute = MagicMock(return_value=None)
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 5

        # Call the wrapper
        result = wrapper(mock_execute, mock_cursor, ("SELECT * FROM users",), {})

        # Verify the original function was called
        mock_execute.assert_called_once_with("SELECT * FROM users")
        self.assertIsNone(result)

    def test_db_execute_wrapper_with_params(self):
        """Test wrapper with query parameters."""
        wrapper = self.instrumentor._db_execute_wrapper("mysql")

        mock_execute = MagicMock(return_value=None)
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 3

        # Call with parameters
        result = wrapper(mock_execute, mock_cursor, ("SELECT * FROM users WHERE id = ?", (1,)), {})

        mock_execute.assert_called_once_with("SELECT * FROM users WHERE id = ?", (1,))

    def test_db_operation_wrapper_pymongo(self):
        """Test that _db_operation_wrapper creates a valid wrapper."""
        wrapper = self.instrumentor._db_operation_wrapper("pymongo", "find_one")

        # Test that wrapper is callable
        self.assertTrue(callable(wrapper))

        # Mock a pymongo operation
        mock_find_one = MagicMock(return_value={"_id": "123", "name": "test"})
        mock_collection = MagicMock()

        # Call the wrapper
        result = wrapper(mock_find_one, mock_collection, ({"name": "test"},), {})

        # Verify the original function was called
        mock_find_one.assert_called_once_with({"name": "test"})
        self.assertEqual(result, {"_id": "123", "name": "test"})

    def test_db_operation_wrapper_with_list_result(self):
        """Test wrapper with list result."""
        wrapper = self.instrumentor._db_operation_wrapper("pymongo", "find")

        mock_find = MagicMock(return_value=[{"_id": "1"}, {"_id": "2"}])
        mock_collection = MagicMock()

        result = wrapper(mock_find, mock_collection, ({},), {})

        self.assertEqual(len(result), 2)

    @patch("genai_otel.mcp_instrumentors.database_instrumentor.PSYCOPG2_AVAILABLE", True)
    @patch("genai_otel.mcp_instrumentors.database_instrumentor.Psycopg2Cursor")
    @patch("genai_otel.mcp_instrumentors.database_instrumentor.wrapt")
    def test_add_psycopg2_metrics(self, mock_wrapt, mock_cursor_class, *args):
        """Test _add_psycopg2_metrics wraps cursor methods."""
        mock_cursor_class.execute = MagicMock()
        mock_cursor_class.executemany = MagicMock()

        # Set hasattr to return True
        with patch("builtins.hasattr", return_value=True):
            self.instrumentor._add_psycopg2_metrics()

        # Verify wrapt.wrap_function_wrapper was called
        self.assertGreaterEqual(mock_wrapt.wrap_function_wrapper.call_count, 1)

    @patch("genai_otel.mcp_instrumentors.database_instrumentor.PYMONGO_AVAILABLE", True)
    @patch("genai_otel.mcp_instrumentors.database_instrumentor.PymongoCollection")
    @patch("genai_otel.mcp_instrumentors.database_instrumentor.wrapt")
    def test_add_pymongo_metrics(self, mock_wrapt, mock_collection_class, *args):
        """Test _add_pymongo_metrics wraps collection methods."""
        # Mock hasattr to return True for methods
        with patch("builtins.hasattr", return_value=True):
            self.instrumentor._add_pymongo_metrics()

        # Verify wrapt.wrap_function_wrapper was called multiple times (for each method)
        self.assertGreaterEqual(mock_wrapt.wrap_function_wrapper.call_count, 1)

    @patch("genai_otel.mcp_instrumentors.database_instrumentor.MYSQL_AVAILABLE", True)
    @patch("genai_otel.mcp_instrumentors.database_instrumentor.MySQLCursor")
    @patch("genai_otel.mcp_instrumentors.database_instrumentor.wrapt")
    def test_add_mysql_metrics(self, mock_wrapt, mock_cursor_class, *args):
        """Test _add_mysql_metrics wraps cursor methods."""
        mock_cursor_class.execute = MagicMock()
        mock_cursor_class.executemany = MagicMock()

        # Set hasattr to return True
        with patch("builtins.hasattr", return_value=True):
            self.instrumentor._add_mysql_metrics()

        # Verify wrapt.wrap_function_wrapper was called
        self.assertGreaterEqual(mock_wrapt.wrap_function_wrapper.call_count, 1)

    @patch("genai_otel.mcp_instrumentors.database_instrumentor.PSYCOPG2_AVAILABLE", False)
    @patch("genai_otel.mcp_instrumentors.database_instrumentor.logger")
    def test_add_psycopg2_metrics_not_available(self, mock_logger, *args):
        """Test that _add_psycopg2_metrics handles unavailable psycopg2."""
        self.instrumentor._add_psycopg2_metrics()

        # Should log debug message
        mock_logger.debug.assert_called()

    @patch("genai_otel.mcp_instrumentors.database_instrumentor.PYMONGO_AVAILABLE", False)
    @patch("genai_otel.mcp_instrumentors.database_instrumentor.logger")
    def test_add_pymongo_metrics_not_available(self, mock_logger, *args):
        """Test that _add_pymongo_metrics handles unavailable pymongo."""
        self.instrumentor._add_pymongo_metrics()

        # Should log debug message
        mock_logger.debug.assert_called()

    @patch("genai_otel.mcp_instrumentors.database_instrumentor.MYSQL_AVAILABLE", False)
    @patch("genai_otel.mcp_instrumentors.database_instrumentor.logger")
    def test_add_mysql_metrics_not_available(self, mock_logger, *args):
        """Test that _add_mysql_metrics handles unavailable MySQL."""
        self.instrumentor._add_mysql_metrics()

        # Should log debug message
        mock_logger.debug.assert_called()

    def test_db_execute_wrapper_exception_handling(self):
        """Test that wrapper handles exceptions in payload size calculation."""
        wrapper = self.instrumentor._db_execute_wrapper("psycopg2")

        # Mock function that raises exception
        mock_execute = MagicMock(return_value=None)
        mock_cursor = MagicMock()
        # Make rowcount raise exception when accessed
        type(mock_cursor).rowcount = property(lambda self: (_ for _ in ()).throw(Exception("test")))

        # This should not raise, just log debug
        result = wrapper(mock_execute, mock_cursor, ("SELECT * FROM users",), {})

        # Should still return result
        self.assertIsNone(result)

    def test_db_operation_wrapper_exception_handling(self):
        """Test that operation wrapper handles exceptions gracefully."""
        wrapper = self.instrumentor._db_operation_wrapper("pymongo", "find_one")

        mock_find_one = MagicMock(return_value={"_id": "123"})
        mock_collection = MagicMock()

        # Call with non-serializable argument
        class NonSerializable:
            pass

        result = wrapper(mock_find_one, mock_collection, (NonSerializable(),), {})

        # Should still work
        self.assertEqual(result, {"_id": "123"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
