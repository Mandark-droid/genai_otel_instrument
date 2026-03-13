"""Tests for FalkorDB MCP instrumentor."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from genai_otel.config import OTelConfig


@pytest.fixture
def config():
    return OTelConfig()


class TestFalkorDBInstrumentor:
    """Tests for FalkorDBInstrumentor."""

    def test_init(self, config):
        """Test that config is stored on the instrumentor."""
        from genai_otel.mcp_instrumentors.falkordb_instrumentor import FalkorDBInstrumentor

        instrumentor = FalkorDBInstrumentor(config)
        assert instrumentor.config is config

    @patch("wrapt.wrap_function_wrapper")
    def test_instrument_success(self, mock_wrap, config, caplog):
        """Test successful instrumentation wraps all expected methods."""
        mock_falkordb = MagicMock()
        with patch.dict("sys.modules", {"falkordb": mock_falkordb}):
            from genai_otel.mcp_instrumentors.falkordb_instrumentor import FalkorDBInstrumentor

            instrumentor = FalkorDBInstrumentor(config)
            caplog.set_level(logging.INFO)
            result = instrumentor.instrument()

            assert result is True
            assert "FalkorDB instrumentation enabled" in caplog.text

            expected_methods = [
                "Graph.query",
                "Graph.ro_query",
                "Graph.delete",
                "Graph.copy",
                "FalkorDB.select_graph",
            ]
            assert mock_wrap.call_count == len(expected_methods)
            wrapped_methods = [call_args[0][1] for call_args in mock_wrap.call_args_list]
            for method_name in expected_methods:
                assert method_name in wrapped_methods
            # Verify all calls use "falkordb" as the module
            for call_args in mock_wrap.call_args_list:
                assert call_args[0][0] == "falkordb"

    def test_instrument_missing(self, config, caplog):
        """Test graceful handling when falkordb is not installed."""
        with patch.dict("sys.modules", {"falkordb": None}):
            from genai_otel.mcp_instrumentors.falkordb_instrumentor import FalkorDBInstrumentor

            instrumentor = FalkorDBInstrumentor(config)
            caplog.set_level(logging.DEBUG)
            result = instrumentor.instrument()

            assert result is False
            assert any(
                "falkordb not installed, skipping FalkorDB instrumentation" in r.message
                and r.levelno == logging.DEBUG
                for r in caplog.records
            )

    @patch("wrapt.wrap_function_wrapper", side_effect=RuntimeError("Mock error"))
    def test_instrument_error(self, mock_wrap, config, caplog):
        """Test error logging on general exception."""
        mock_falkordb = MagicMock()
        with patch.dict("sys.modules", {"falkordb": mock_falkordb}):
            from genai_otel.mcp_instrumentors.falkordb_instrumentor import FalkorDBInstrumentor

            instrumentor = FalkorDBInstrumentor(config)
            caplog.set_level(logging.ERROR)
            result = instrumentor.instrument()

            assert result is False
            assert any(
                "Failed to instrument FalkorDB: Mock error" in r.message
                and r.levelno == logging.ERROR
                for r in caplog.records
            )

    def test_wrapper_query_creates_span_with_attributes(self, config):
        """Test that wrapped query method creates spans with correct attributes."""
        mock_falkordb = MagicMock()
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        wrappers = {}

        def capture_wrapper(module, name, wrapper_func):
            wrappers[name] = wrapper_func

        with patch.dict("sys.modules", {"falkordb": mock_falkordb}):
            with patch("wrapt.wrap_function_wrapper", side_effect=capture_wrapper):
                with patch(
                    "genai_otel.mcp_instrumentors.falkordb_instrumentor.trace"
                ) as mock_trace:
                    mock_trace.get_tracer.return_value = mock_tracer

                    from genai_otel.mcp_instrumentors.falkordb_instrumentor import (
                        FalkorDBInstrumentor,
                    )

                    instrumentor = FalkorDBInstrumentor(config)
                    instrumentor.instrument()

        assert "Graph.query" in wrappers

        # Call the wrapper with a Cypher query
        mock_wrapped = MagicMock(return_value="result")
        mock_instance = MagicMock()
        mock_instance.name = "social"
        wrappers["Graph.query"](mock_wrapped, mock_instance, ("MATCH (n) RETURN n",), {})

        mock_span.set_attribute.assert_any_call("db.system", "falkordb")
        mock_span.set_attribute.assert_any_call("db.operation", "query")
        mock_span.set_attribute.assert_any_call("db.name", "social")
        mock_span.set_attribute.assert_any_call("db.statement", "MATCH (n) RETURN n")

    def test_wrapper_ro_query_creates_span(self, config):
        """Test that wrapped ro_query method creates spans with correct attributes."""
        mock_falkordb = MagicMock()
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        wrappers = {}

        def capture_wrapper(module, name, wrapper_func):
            wrappers[name] = wrapper_func

        with patch.dict("sys.modules", {"falkordb": mock_falkordb}):
            with patch("wrapt.wrap_function_wrapper", side_effect=capture_wrapper):
                with patch(
                    "genai_otel.mcp_instrumentors.falkordb_instrumentor.trace"
                ) as mock_trace:
                    mock_trace.get_tracer.return_value = mock_tracer

                    from genai_otel.mcp_instrumentors.falkordb_instrumentor import (
                        FalkorDBInstrumentor,
                    )

                    instrumentor = FalkorDBInstrumentor(config)
                    instrumentor.instrument()

        assert "Graph.ro_query" in wrappers

        mock_wrapped = MagicMock(return_value="result")
        mock_instance = MagicMock()
        mock_instance.name = "social"
        wrappers["Graph.ro_query"](
            mock_wrapped, mock_instance, ("MATCH (n) RETURN n LIMIT 10",), {}
        )

        mock_span.set_attribute.assert_any_call("db.system", "falkordb")
        mock_span.set_attribute.assert_any_call("db.operation", "ro_query")
        mock_span.set_attribute.assert_any_call("db.name", "social")

    def test_wrapper_select_graph_creates_span(self, config):
        """Test that wrapped select_graph method creates spans with correct attributes."""
        mock_falkordb = MagicMock()
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        wrappers = {}

        def capture_wrapper(module, name, wrapper_func):
            wrappers[name] = wrapper_func

        with patch.dict("sys.modules", {"falkordb": mock_falkordb}):
            with patch("wrapt.wrap_function_wrapper", side_effect=capture_wrapper):
                with patch(
                    "genai_otel.mcp_instrumentors.falkordb_instrumentor.trace"
                ) as mock_trace:
                    mock_trace.get_tracer.return_value = mock_tracer

                    from genai_otel.mcp_instrumentors.falkordb_instrumentor import (
                        FalkorDBInstrumentor,
                    )

                    instrumentor = FalkorDBInstrumentor(config)
                    instrumentor.instrument()

        assert "FalkorDB.select_graph" in wrappers

        mock_wrapped = MagicMock(return_value="graph_obj")
        wrappers["FalkorDB.select_graph"](mock_wrapped, None, ("social",), {})

        mock_span.set_attribute.assert_any_call("db.system", "falkordb")
        mock_span.set_attribute.assert_any_call("db.operation", "select_graph")
        mock_span.set_attribute.assert_any_call("db.name", "social")

    def test_wrapper_handles_exception(self, config):
        """Test that exceptions are recorded on spans."""
        mock_falkordb = MagicMock()
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        wrappers = {}

        def capture_wrapper(module, name, wrapper_func):
            wrappers[name] = wrapper_func

        with patch.dict("sys.modules", {"falkordb": mock_falkordb}):
            with patch("wrapt.wrap_function_wrapper", side_effect=capture_wrapper):
                with patch(
                    "genai_otel.mcp_instrumentors.falkordb_instrumentor.trace"
                ) as mock_trace:
                    mock_trace.get_tracer.return_value = mock_tracer

                    from genai_otel.mcp_instrumentors.falkordb_instrumentor import (
                        FalkorDBInstrumentor,
                    )

                    instrumentor = FalkorDBInstrumentor(config)
                    instrumentor.instrument()

        assert "Graph.query" in wrappers

        # Make the wrapped function raise an exception
        test_error = ConnectionError("Connection refused")
        mock_wrapped = MagicMock(side_effect=test_error)

        with pytest.raises(ConnectionError, match="Connection refused"):
            wrappers["Graph.query"](mock_wrapped, None, ("MATCH (n) RETURN n",), {})

        mock_span.record_exception.assert_called_once_with(test_error)

    def test_wrapper_copy_creates_span(self, config):
        """Test that wrapped copy method creates spans with destination graph."""
        mock_falkordb = MagicMock()
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        wrappers = {}

        def capture_wrapper(module, name, wrapper_func):
            wrappers[name] = wrapper_func

        with patch.dict("sys.modules", {"falkordb": mock_falkordb}):
            with patch("wrapt.wrap_function_wrapper", side_effect=capture_wrapper):
                with patch(
                    "genai_otel.mcp_instrumentors.falkordb_instrumentor.trace"
                ) as mock_trace:
                    mock_trace.get_tracer.return_value = mock_tracer

                    from genai_otel.mcp_instrumentors.falkordb_instrumentor import (
                        FalkorDBInstrumentor,
                    )

                    instrumentor = FalkorDBInstrumentor(config)
                    instrumentor.instrument()

        assert "Graph.copy" in wrappers

        mock_wrapped = MagicMock(return_value="copied_graph")
        mock_instance = MagicMock()
        mock_instance.name = "social"
        wrappers["Graph.copy"](mock_wrapped, mock_instance, ("social_backup",), {})

        mock_span.set_attribute.assert_any_call("db.system", "falkordb")
        mock_span.set_attribute.assert_any_call("db.operation", "copy")
        mock_span.set_attribute.assert_any_call("db.name", "social")
        mock_span.set_attribute.assert_any_call("falkordb.destination_graph", "social_backup")
