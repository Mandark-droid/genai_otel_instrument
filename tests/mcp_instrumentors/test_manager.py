import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.mcp_instrumentors.manager import MCPInstrumentorManager


class TestMCPInstrumentorManager(unittest.TestCase):
    """Tests for MCPInstrumentorManager"""

    def setUp(self):
        # Enable HTTP instrumentation for tests that expect it
        self.config = OTelConfig(enable_http_instrumentation=True)

    def test_init(self):
        """Test that manager initializes correctly"""
        manager = MCPInstrumentorManager(self.config)
        self.assertEqual(manager.config, self.config)
        self.assertEqual(manager.instrumentors, [])

    @patch("genai_otel.mcp_instrumentors.manager.logger")
    @patch("genai_otel.mcp_instrumentors.manager.HTTPXClientInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.APIInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.DatabaseInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.RedisInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.KafkaInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.VectorDBInstrumentor")
    def test_instrument_all_success(
        self,
        mock_vector_db,
        mock_kafka,
        mock_redis,
        mock_database,
        mock_api,
        mock_httpx,
        mock_logger,
    ):
        """Test successful instrumentation of all components"""
        # Setup mocks
        mock_httpx_instance = MagicMock()
        mock_httpx.return_value = mock_httpx_instance

        mock_api_instance = MagicMock()
        mock_api.return_value = mock_api_instance

        mock_db_instance = MagicMock()
        mock_db_instance.instrument.return_value = 3  # 3 databases instrumented
        mock_database.return_value = mock_db_instance

        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance

        mock_kafka_instance = MagicMock()
        mock_kafka.return_value = mock_kafka_instance

        mock_vector_instance = MagicMock()
        mock_vector_instance.instrument.return_value = 2  # 2 vector DBs instrumented
        mock_vector_db.return_value = mock_vector_instance

        # Execute
        manager = MCPInstrumentorManager(self.config)
        manager.instrument_all()

        # Verify HTTPx instrumentation
        mock_httpx_instance.instrument.assert_called_once()

        # Verify API instrumentation
        mock_api_instance.instrument.assert_called_once_with(self.config)

        # Verify Database instrumentation
        mock_db_instance.instrument.assert_called_once()

        # Verify Redis instrumentation
        mock_redis_instance.instrument.assert_called_once()

        # Verify Kafka instrumentation
        mock_kafka_instance.instrument.assert_called_once()

        # Verify Vector DB instrumentation
        mock_vector_instance.instrument.assert_called_once()

        # Verify success messages logged
        self.assertTrue(any("5 succeeded" in str(call) for call in mock_logger.info.call_args_list))

    @patch("genai_otel.mcp_instrumentors.manager.logger")
    @patch("genai_otel.mcp_instrumentors.manager.HTTPXClientInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.APIInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.DatabaseInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.RedisInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.KafkaInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.VectorDBInstrumentor")
    def test_instrument_all_with_import_errors(
        self,
        mock_vector_db,
        mock_kafka,
        mock_redis,
        mock_database,
        mock_api,
        mock_httpx,
        mock_logger,
    ):
        """Test instrumentation when some components have import errors"""
        # Setup mocks - HTTPx succeeds
        mock_httpx_instance = MagicMock()
        mock_httpx.return_value = mock_httpx_instance
        mock_api_instance = MagicMock()
        mock_api.return_value = mock_api_instance

        # Database raises ImportError
        mock_database.side_effect = ImportError("No database module")

        # Redis succeeds
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance

        # Kafka raises ImportError
        mock_kafka.side_effect = ImportError("No kafka module")

        # Vector DB succeeds
        mock_vector_instance = MagicMock()
        mock_vector_instance.instrument.return_value = 1
        mock_vector_db.return_value = mock_vector_instance

        # Execute
        manager = MCPInstrumentorManager(self.config)
        manager.instrument_all()

        # Verify debug logs for import errors
        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
        self.assertTrue(any("Database instrumentation skipped" in call for call in debug_calls))
        self.assertTrue(any("Kafka instrumentation skipped" in call for call in debug_calls))

        # Verify summary includes failures
        self.assertTrue(any("2 failed" in str(call) for call in mock_logger.info.call_args_list))

    @patch("genai_otel.mcp_instrumentors.manager.logger")
    @patch("genai_otel.mcp_instrumentors.manager.HTTPXClientInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.APIInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.DatabaseInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.RedisInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.KafkaInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.VectorDBInstrumentor")
    def test_instrument_all_with_exceptions(
        self,
        mock_vector_db,
        mock_kafka,
        mock_redis,
        mock_database,
        mock_api,
        mock_httpx,
        mock_logger,
    ):
        """Test instrumentation when components raise exceptions"""
        # HTTPx raises RuntimeError
        mock_httpx_instance = MagicMock()
        mock_httpx_instance.instrument.side_effect = RuntimeError("HTTPx error")
        mock_httpx.return_value = mock_httpx_instance

        # API succeeds
        mock_api_instance = MagicMock()
        mock_api.return_value = mock_api_instance

        # Database succeeds
        mock_db_instance = MagicMock()
        mock_db_instance.instrument.return_value = 1
        mock_database.return_value = mock_db_instance

        # Redis raises RuntimeError
        mock_redis_instance = MagicMock()
        mock_redis_instance.instrument.side_effect = RuntimeError("Redis error")
        mock_redis.return_value = mock_redis_instance

        # Kafka succeeds
        mock_kafka_instance = MagicMock()
        mock_kafka.return_value = mock_kafka_instance

        # Vector DB raises RuntimeError
        mock_vector_instance = MagicMock()
        mock_vector_instance.instrument.side_effect = RuntimeError("Vector DB error")
        mock_vector_db.return_value = mock_vector_instance

        # Execute
        manager = MCPInstrumentorManager(self.config)
        manager.instrument_all(fail_on_error=False)

        # Verify error logs
        error_calls = [str(call) for call in mock_logger.error.call_args_list]
        self.assertTrue(any("Failed to instrument HTTP/API" in call for call in error_calls))
        self.assertTrue(any("Failed to instrument Redis" in call for call in error_calls))
        self.assertTrue(any("Failed to instrument Vector DBs" in call for call in error_calls))

        # Verify summary includes failures
        self.assertTrue(any("3 failed" in str(call) for call in mock_logger.info.call_args_list))

    @patch("genai_otel.mcp_instrumentors.manager.logger")
    @patch("genai_otel.mcp_instrumentors.manager.HTTPXClientInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.APIInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.DatabaseInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.RedisInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.KafkaInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.VectorDBInstrumentor")
    def test_instrument_all_fail_on_error_true(
        self,
        mock_vector_db,
        mock_kafka,
        mock_redis,
        mock_database,
        mock_api,
        mock_httpx,
        mock_logger,
    ):
        """Test that fail_on_error=True raises exceptions"""
        # HTTPx raises RuntimeError
        mock_httpx_instance = MagicMock()
        mock_httpx_instance.instrument.side_effect = RuntimeError("HTTPx error")
        mock_httpx.return_value = mock_httpx_instance

        # Execute and expect exception
        manager = MCPInstrumentorManager(self.config)
        with self.assertRaises(RuntimeError) as context:
            manager.instrument_all(fail_on_error=True)

        self.assertEqual(str(context.exception), "HTTPx error")

    @patch("genai_otel.mcp_instrumentors.manager.logger")
    @patch("genai_otel.mcp_instrumentors.manager.HTTPXClientInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.APIInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.DatabaseInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.RedisInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.KafkaInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.VectorDBInstrumentor")
    def test_instrument_all_database_zero_count(
        self,
        mock_vector_db,
        mock_kafka,
        mock_redis,
        mock_database,
        mock_api,
        mock_httpx,
        mock_logger,
    ):
        """Test when database instrumentor returns 0 (no databases found)"""
        # Setup mocks
        mock_httpx_instance = MagicMock()
        mock_httpx.return_value = mock_httpx_instance
        mock_api_instance = MagicMock()
        mock_api.return_value = mock_api_instance

        # Database returns 0
        mock_db_instance = MagicMock()
        mock_db_instance.instrument.return_value = 0
        mock_database.return_value = mock_db_instance

        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance

        mock_kafka_instance = MagicMock()
        mock_kafka.return_value = mock_kafka_instance

        # Vector DB returns 0
        mock_vector_instance = MagicMock()
        mock_vector_instance.instrument.return_value = 0
        mock_vector_db.return_value = mock_vector_instance

        # Execute
        manager = MCPInstrumentorManager(self.config)
        manager.instrument_all()

        # Verify database was instrumented but didn't increment success count
        mock_db_instance.instrument.assert_called_once()

        # Success count should be 3 (HTTPx/API, Redis, Kafka) not 5
        self.assertTrue(any("3 succeeded" in str(call) for call in mock_logger.info.call_args_list))

    @patch("genai_otel.mcp_instrumentors.manager.logger")
    @patch("genai_otel.mcp_instrumentors.manager.HTTPXClientInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.APIInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.DatabaseInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.RedisInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.KafkaInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.VectorDBInstrumentor")
    def test_requests_library_warning(
        self,
        mock_vector_db,
        mock_kafka,
        mock_redis,
        mock_database,
        mock_api,
        mock_httpx,
        mock_logger,
    ):
        """Test that requests library instrumentation warning is logged"""
        # Setup mocks to succeed
        mock_httpx.return_value = MagicMock()
        mock_api.return_value = MagicMock()
        mock_database.return_value = MagicMock()
        mock_database.return_value.instrument.return_value = 0
        mock_redis.return_value = MagicMock()
        mock_kafka.return_value = MagicMock()
        mock_vector_db.return_value = MagicMock()
        mock_vector_db.return_value.instrument.return_value = 0

        # Execute
        manager = MCPInstrumentorManager(self.config)
        manager.instrument_all()

        # Verify warning about requests library
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        self.assertTrue(
            any("Requests library instrumentation is disabled" in call for call in warning_calls)
        )

    @patch("genai_otel.mcp_instrumentors.manager.logger")
    @patch("genai_otel.mcp_instrumentors.manager.HTTPXClientInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.APIInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.DatabaseInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.RedisInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.KafkaInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.VectorDBInstrumentor")
    def test_http_import_error(
        self,
        mock_vector_db,
        mock_kafka,
        mock_redis,
        mock_database,
        mock_api,
        mock_httpx,
        mock_logger,
    ):
        """Test HTTP/API instrumentation with ImportError"""
        # HTTPx raises ImportError
        mock_httpx.side_effect = ImportError("No httpx module")

        # Other components succeed
        mock_api.return_value = MagicMock()
        mock_database.return_value = MagicMock()
        mock_database.return_value.instrument.return_value = 1
        mock_redis.return_value = MagicMock()
        mock_kafka.return_value = MagicMock()
        mock_vector_db.return_value = MagicMock()
        mock_vector_db.return_value.instrument.return_value = 1

        # Execute
        manager = MCPInstrumentorManager(self.config)
        manager.instrument_all()

        # Verify debug log for import error
        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
        self.assertTrue(any("HTTP/API instrumentation skipped" in call for call in debug_calls))

    @patch("genai_otel.mcp_instrumentors.manager.logger")
    @patch("genai_otel.mcp_instrumentors.manager.HTTPXClientInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.APIInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.DatabaseInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.RedisInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.KafkaInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.VectorDBInstrumentor")
    def test_database_fail_on_error(
        self,
        mock_vector_db,
        mock_kafka,
        mock_redis,
        mock_database,
        mock_api,
        mock_httpx,
        mock_logger,
    ):
        """Test database instrumentation failure with fail_on_error=True"""
        # HTTP succeeds
        mock_httpx.return_value = MagicMock()
        mock_api.return_value = MagicMock()

        # Database raises exception
        mock_db_instance = MagicMock()
        mock_db_instance.instrument.side_effect = RuntimeError("Database error")
        mock_database.return_value = mock_db_instance

        # Execute and expect exception
        manager = MCPInstrumentorManager(self.config)
        with self.assertRaises(RuntimeError) as context:
            manager.instrument_all(fail_on_error=True)

        self.assertEqual(str(context.exception), "Database error")

    @patch("genai_otel.mcp_instrumentors.manager.logger")
    @patch("genai_otel.mcp_instrumentors.manager.HTTPXClientInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.APIInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.DatabaseInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.RedisInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.KafkaInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.VectorDBInstrumentor")
    def test_redis_fail_on_error(
        self,
        mock_vector_db,
        mock_kafka,
        mock_redis,
        mock_database,
        mock_api,
        mock_httpx,
        mock_logger,
    ):
        """Test Redis instrumentation failure with fail_on_error=True"""
        # HTTP and database succeed
        mock_httpx.return_value = MagicMock()
        mock_api.return_value = MagicMock()
        mock_db_instance = MagicMock()
        mock_db_instance.instrument.return_value = 1
        mock_database.return_value = mock_db_instance

        # Redis raises exception
        mock_redis_instance = MagicMock()
        mock_redis_instance.instrument.side_effect = RuntimeError("Redis error")
        mock_redis.return_value = mock_redis_instance

        # Execute and expect exception
        manager = MCPInstrumentorManager(self.config)
        with self.assertRaises(RuntimeError) as context:
            manager.instrument_all(fail_on_error=True)

        self.assertEqual(str(context.exception), "Redis error")

    @patch("genai_otel.mcp_instrumentors.manager.logger")
    @patch("genai_otel.mcp_instrumentors.manager.HTTPXClientInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.APIInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.DatabaseInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.RedisInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.KafkaInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.VectorDBInstrumentor")
    def test_kafka_fail_on_error(
        self,
        mock_vector_db,
        mock_kafka,
        mock_redis,
        mock_database,
        mock_api,
        mock_httpx,
        mock_logger,
    ):
        """Test Kafka instrumentation failure with fail_on_error=True"""
        # HTTP, database, and redis succeed
        mock_httpx.return_value = MagicMock()
        mock_api.return_value = MagicMock()
        mock_db_instance = MagicMock()
        mock_db_instance.instrument.return_value = 1
        mock_database.return_value = mock_db_instance
        mock_redis.return_value = MagicMock()

        # Kafka raises exception
        mock_kafka_instance = MagicMock()
        mock_kafka_instance.instrument.side_effect = RuntimeError("Kafka error")
        mock_kafka.return_value = mock_kafka_instance

        # Execute and expect exception
        manager = MCPInstrumentorManager(self.config)
        with self.assertRaises(RuntimeError) as context:
            manager.instrument_all(fail_on_error=True)

        self.assertEqual(str(context.exception), "Kafka error")

    @patch("genai_otel.mcp_instrumentors.manager.logger")
    @patch("genai_otel.mcp_instrumentors.manager.HTTPXClientInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.APIInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.DatabaseInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.RedisInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.KafkaInstrumentor")
    @patch("genai_otel.mcp_instrumentors.manager.VectorDBInstrumentor")
    def test_vector_db_fail_on_error(
        self,
        mock_vector_db,
        mock_kafka,
        mock_redis,
        mock_database,
        mock_api,
        mock_httpx,
        mock_logger,
    ):
        """Test Vector DB instrumentation failure with fail_on_error=True"""
        # All previous components succeed
        mock_httpx.return_value = MagicMock()
        mock_api.return_value = MagicMock()
        mock_db_instance = MagicMock()
        mock_db_instance.instrument.return_value = 1
        mock_database.return_value = mock_db_instance
        mock_redis.return_value = MagicMock()
        mock_kafka.return_value = MagicMock()

        # Vector DB raises exception
        mock_vector_instance = MagicMock()
        mock_vector_instance.instrument.side_effect = RuntimeError("Vector DB error")
        mock_vector_db.return_value = mock_vector_instance

        # Execute and expect exception
        manager = MCPInstrumentorManager(self.config)
        with self.assertRaises(RuntimeError) as context:
            manager.instrument_all(fail_on_error=True)

        self.assertEqual(str(context.exception), "Vector DB error")


if __name__ == "__main__":
    unittest.main(verbosity=2)
