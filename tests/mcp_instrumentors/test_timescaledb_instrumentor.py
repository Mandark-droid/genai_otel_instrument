"""Tests for TimescaleDBInstrumentor."""

import logging
import sys
import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.mcp_instrumentors.timescaledb_instrumentor import (
    TimescaleDBInstrumentor,
    _detect_timescale_operation,
)


class TestTimescaleDBInstrumentor(unittest.TestCase):
    """Tests for TimescaleDBInstrumentor"""

    def test_init(self):
        """Test that __init__ stores config correctly."""
        config = MagicMock(spec=OTelConfig)
        instrumentor = TimescaleDBInstrumentor(config)

        self.assertEqual(instrumentor.config, config)
        self.assertIsNotNone(instrumentor.tracer)

    @patch("genai_otel.mcp_instrumentors.timescaledb_instrumentor.wrapt")
    @patch("genai_otel.mcp_instrumentors.timescaledb_instrumentor.logger")
    def test_instrument_success(self, mock_logger, mock_wrapt):
        """Test successful TimescaleDB instrumentation when psycopg2 is available."""
        # Create a mock psycopg2 module with extensions.cursor
        mock_psycopg2 = MagicMock()
        mock_cursor_class = MagicMock()
        mock_cursor_class.execute = MagicMock()
        mock_psycopg2.extensions.cursor = mock_cursor_class

        with patch.dict(
            "sys.modules",
            {"psycopg2": mock_psycopg2, "psycopg2.extensions": mock_psycopg2.extensions},
        ):
            config = MagicMock(spec=OTelConfig)
            instrumentor = TimescaleDBInstrumentor(config)

            result = instrumentor.instrument()

            self.assertTrue(result)
            mock_wrapt.wrap_function_wrapper.assert_called_once_with(
                "psycopg2.extensions", "cursor.execute", unittest.mock.ANY
            )
            mock_logger.info.assert_called_with("TimescaleDB instrumentation enabled")

    @patch("genai_otel.mcp_instrumentors.timescaledb_instrumentor.wrapt")
    @patch("genai_otel.mcp_instrumentors.timescaledb_instrumentor.logger")
    def test_instrument_import_error(self, mock_logger, mock_wrapt):
        """Test graceful handling when psycopg2 is not installed."""
        config = MagicMock(spec=OTelConfig)
        instrumentor = TimescaleDBInstrumentor(config)

        # Force ImportError by patching the import inside instrument()
        with patch.dict("sys.modules", {"psycopg2": None}):
            result = instrumentor.instrument()

        self.assertFalse(result)
        mock_logger.debug.assert_called_with(
            "psycopg2 not installed, skipping TimescaleDB instrumentation"
        )
        mock_wrapt.wrap_function_wrapper.assert_not_called()

    @patch("genai_otel.mcp_instrumentors.timescaledb_instrumentor.wrapt")
    @patch("genai_otel.mcp_instrumentors.timescaledb_instrumentor.logger")
    def test_instrument_general_exception(self, mock_logger, mock_wrapt):
        """Test error logging on general exception."""
        mock_psycopg2 = MagicMock()
        mock_psycopg2.extensions.cursor = MagicMock()

        mock_wrapt.wrap_function_wrapper.side_effect = RuntimeError("Wrapping failed")

        with patch.dict(
            "sys.modules",
            {"psycopg2": mock_psycopg2, "psycopg2.extensions": mock_psycopg2.extensions},
        ):
            config = MagicMock(spec=OTelConfig)
            instrumentor = TimescaleDBInstrumentor(config)

            result = instrumentor.instrument()

        self.assertFalse(result)
        mock_logger.error.assert_called_once()
        error_args = mock_logger.error.call_args[0]
        self.assertIn("Failed to instrument TimescaleDB", error_args[0])

    def test_detect_timescale_operation(self):
        """Test the pattern detection function with various queries."""
        # TimescaleDB-specific queries
        self.assertEqual(
            _detect_timescale_operation("SELECT CREATE_HYPERTABLE('conditions', 'time')"),
            "create_hypertable",
        )
        self.assertEqual(
            _detect_timescale_operation(
                "SELECT ADD_COMPRESSION_POLICY('conditions', INTERVAL '7 days')"
            ),
            "add_compression_policy",
        )
        self.assertEqual(
            _detect_timescale_operation(
                "SELECT TIME_BUCKET('5 minutes', time) AS BUCKET FROM conditions"
            ),
            "time_bucket_query",
        )
        self.assertEqual(
            _detect_timescale_operation("CREATE MATERIALIZED VIEW daily_avg AS SELECT ..."),
            "create_continuous_aggregate",
        )
        self.assertEqual(
            _detect_timescale_operation(
                "SELECT DROP_CHUNKS('conditions', OLDER_THAN => INTERVAL '3 months')"
            ),
            "drop_chunks",
        )
        self.assertEqual(
            _detect_timescale_operation("SELECT SHOW_CHUNKS('conditions')"),
            "show_chunks",
        )
        self.assertEqual(
            _detect_timescale_operation(
                "SELECT COMPRESS_CHUNK('_timescaledb_internal._hyper_1_1_chunk')"
            ),
            "compress_chunk",
        )
        self.assertEqual(
            _detect_timescale_operation(
                "SELECT DECOMPRESS_CHUNK('_timescaledb_internal._hyper_1_1_chunk')"
            ),
            "decompress_chunk",
        )
        self.assertEqual(
            _detect_timescale_operation("SELECT HYPERTABLE_SIZE('conditions')"),
            "hypertable_size",
        )
        self.assertEqual(
            _detect_timescale_operation("SELECT CHUNKS_DETAILED_SIZE('conditions')"),
            "chunks_detailed_size",
        )
        self.assertEqual(
            _detect_timescale_operation(
                "SELECT ADD_RETENTION_POLICY('conditions', INTERVAL '6 months')"
            ),
            "add_retention_policy",
        )
        self.assertEqual(
            _detect_timescale_operation("SELECT REMOVE_RETENTION_POLICY('conditions')"),
            "remove_retention_policy",
        )
        self.assertEqual(
            _detect_timescale_operation("SELECT REMOVE_COMPRESSION_POLICY('conditions')"),
            "remove_compression_policy",
        )
        self.assertEqual(
            _detect_timescale_operation("SELECT ADD_CONTINUOUS_AGGREGATE_POLICY('daily_avg', ...)"),
            "add_continuous_aggregate_policy",
        )
        self.assertEqual(
            _detect_timescale_operation("SELECT REMOVE_CONTINUOUS_AGGREGATE_POLICY('daily_avg')"),
            "remove_continuous_aggregate_policy",
        )
        self.assertEqual(
            _detect_timescale_operation(
                "SELECT CREATE_DISTRIBUTED_HYPERTABLE('conditions', 'time')"
            ),
            "create_distributed_hypertable",
        )

        # Non-TimescaleDB queries should return empty string
        self.assertEqual(
            _detect_timescale_operation("SELECT * FROM users WHERE id = 1"),
            "",
        )
        self.assertEqual(
            _detect_timescale_operation("INSERT INTO logs (message) VALUES ('hello')"),
            "",
        )
        self.assertEqual(
            _detect_timescale_operation("UPDATE settings SET value = 'new' WHERE key = 'test'"),
            "",
        )

    @patch("genai_otel.mcp_instrumentors.timescaledb_instrumentor.wrapt")
    @patch("genai_otel.mcp_instrumentors.timescaledb_instrumentor.trace")
    def test_wrapped_execute_timescale_query(self, mock_trace, mock_wrapt):
        """Test that TimescaleDB queries create spans."""
        mock_psycopg2 = MagicMock()
        mock_psycopg2.extensions.cursor = MagicMock()

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
        mock_trace.get_tracer.return_value = mock_tracer

        # Capture the wrapped_execute function passed to wrapt
        captured_wrapper = None

        def capture_wrapper(module, name, wrapper):
            nonlocal captured_wrapper
            captured_wrapper = wrapper

        mock_wrapt.wrap_function_wrapper.side_effect = capture_wrapper

        with patch.dict(
            "sys.modules",
            {"psycopg2": mock_psycopg2, "psycopg2.extensions": mock_psycopg2.extensions},
        ):
            config = MagicMock(spec=OTelConfig)
            instrumentor = TimescaleDBInstrumentor(config)
            instrumentor.instrument()

        self.assertIsNotNone(captured_wrapper)

        # Simulate calling the wrapped execute with a TimescaleDB query
        mock_original = MagicMock(return_value=None)
        mock_instance = MagicMock()
        query = "SELECT create_hypertable('conditions', 'time')"

        captured_wrapper(mock_original, mock_instance, (query,), {})

        # Verify span was created
        mock_tracer.start_as_current_span.assert_called_once()
        call_args = mock_tracer.start_as_current_span.call_args
        self.assertEqual(call_args[0][0], "timescaledb.create_hypertable")

        # Verify the original function was called
        mock_original.assert_called_once_with(query)

    @patch("genai_otel.mcp_instrumentors.timescaledb_instrumentor.wrapt")
    @patch("genai_otel.mcp_instrumentors.timescaledb_instrumentor.trace")
    def test_wrapped_execute_non_timescale_query(self, mock_trace, mock_wrapt):
        """Test that normal SQL queries pass through without extra spans."""
        mock_psycopg2 = MagicMock()
        mock_psycopg2.extensions.cursor = MagicMock()

        mock_tracer = MagicMock()
        mock_trace.get_tracer.return_value = mock_tracer

        # Capture the wrapped_execute function passed to wrapt
        captured_wrapper = None

        def capture_wrapper(module, name, wrapper):
            nonlocal captured_wrapper
            captured_wrapper = wrapper

        mock_wrapt.wrap_function_wrapper.side_effect = capture_wrapper

        with patch.dict(
            "sys.modules",
            {"psycopg2": mock_psycopg2, "psycopg2.extensions": mock_psycopg2.extensions},
        ):
            config = MagicMock(spec=OTelConfig)
            instrumentor = TimescaleDBInstrumentor(config)
            instrumentor.instrument()

        self.assertIsNotNone(captured_wrapper)

        # Simulate calling the wrapped execute with a normal SQL query
        mock_original = MagicMock(return_value="result")
        mock_instance = MagicMock()
        query = "SELECT * FROM users WHERE id = 1"

        result = captured_wrapper(mock_original, mock_instance, (query,), {})

        # Verify NO span was created
        mock_tracer.start_as_current_span.assert_not_called()

        # Verify the original function was called and result returned
        mock_original.assert_called_once_with(query)
        self.assertEqual(result, "result")


if __name__ == "__main__":
    unittest.main(verbosity=2)
