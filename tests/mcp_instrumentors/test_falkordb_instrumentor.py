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


def _capture_query_wrapper(config, mock_falkordb):
    """Instrument with a captured tracer/span and return (wrappers, mock_span)."""
    mock_tracer = MagicMock()
    mock_span = MagicMock()
    mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
    mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

    wrappers = {}

    def capture_wrapper(module, name, wrapper_func):
        wrappers[name] = wrapper_func

    with patch.dict("sys.modules", {"falkordb": mock_falkordb}):
        with patch("wrapt.wrap_function_wrapper", side_effect=capture_wrapper):
            with patch("genai_otel.mcp_instrumentors.falkordb_instrumentor.trace") as mock_trace:
                mock_trace.get_tracer.return_value = mock_tracer
                from genai_otel.mcp_instrumentors.falkordb_instrumentor import FalkorDBInstrumentor

                instrumentor = FalkorDBInstrumentor(config)
                instrumentor.instrument()
    return wrappers, mock_span


class TestFalkorDBStatementBounding:
    """Tests for the content_max_length bounding of db.statement and errors."""

    def test_bound_text_truncates_when_cap_positive(self):
        from genai_otel.mcp_instrumentors.falkordb_instrumentor import (
            _TRUNCATION_SUFFIX,
            _bound_text,
        )

        cfg = OTelConfig(content_max_length=10)
        result = _bound_text("MATCH (n) WHERE n.name = 1 RETURN n", cfg)
        assert result == "MATCH (n) " + _TRUNCATION_SUFFIX
        assert result.startswith("MATCH (n) ")

    def test_bound_text_keeps_full_when_cap_zero(self):
        from genai_otel.mcp_instrumentors.falkordb_instrumentor import _bound_text

        long_statement = "MATCH (n) WHERE n.name = 1 RETURN n" * 50
        assert _bound_text(long_statement, OTelConfig(content_max_length=0)) == long_statement

    def test_bound_text_safe_without_field(self):
        """A config missing content_max_length (e.g. spec mock) means unlimited."""
        from unittest.mock import MagicMock as MM

        from genai_otel.mcp_instrumentors.falkordb_instrumentor import _bound_text

        cfg = MM(spec=OTelConfig)  # spec has no content_max_length attribute
        assert _bound_text("some cypher", cfg) == "some cypher"

    def test_db_statement_is_bounded_on_span(self):
        cfg = OTelConfig(content_max_length=10)
        wrappers, mock_span = _capture_query_wrapper(cfg, MagicMock())

        cypher = "MATCH (n) WHERE n.name = 'x' RETURN n"
        wrappers["Graph.query"](MagicMock(return_value="ok"), None, (cypher,), {})

        stmt_calls = [
            c.args[1] for c in mock_span.set_attribute.call_args_list if c.args[0] == "db.statement"
        ]
        assert stmt_calls, "db.statement was not captured"
        assert stmt_calls[0].startswith(cypher[:10])
        assert len(stmt_calls[0]) < len(cypher) + 20  # bounded, not the full statement

    def test_db_statement_full_when_cap_zero(self):
        cfg = OTelConfig(content_max_length=0)
        wrappers, mock_span = _capture_query_wrapper(cfg, MagicMock())

        cypher = "MATCH (n) WHERE n.name = 'longvalue-for-audit' RETURN n"
        wrappers["Graph.query"](MagicMock(return_value="ok"), None, (cypher,), {})

        mock_span.set_attribute.assert_any_call("db.statement", cypher)

    def test_error_status_is_bounded(self):
        cfg = OTelConfig(content_max_length=8)
        wrappers, mock_span = _capture_query_wrapper(cfg, MagicMock())

        long_msg = "syntax error near VERY_LONG_SQL_ECHOED_BACK_IN_ERROR" * 3
        with pytest.raises(RuntimeError):
            wrappers["Graph.query"](
                MagicMock(side_effect=RuntimeError(long_msg)), None, ("MATCH (n) RETURN n",), {}
            )

        assert mock_span.set_status.called
        status = mock_span.set_status.call_args.args[0]
        assert len(status.description) <= 8 + len("...[truncated]")


class TestFalkorDBDoubleWrapGuard:
    """Tests for the idempotency / double-wrap guard."""

    def test_repeated_instrument_does_not_stack(self, config):
        from genai_otel.mcp_instrumentors import falkordb_instrumentor as fmod

        fmod._INSTRUMENTED_MODULES.clear()
        fmod._WRAPPED_TARGETS.clear()

        mock_falkordb = MagicMock()
        with patch.dict("sys.modules", {"falkordb": mock_falkordb}):
            with patch("wrapt.wrap_function_wrapper") as mock_wrap:
                instrumentor = fmod.FalkorDBInstrumentor(config)
                assert instrumentor.instrument() is True
                first = mock_wrap.call_count
                assert first == 5
                # Second call is guarded: no additional wrapping.
                assert instrumentor.instrument() is True
                assert mock_wrap.call_count == first

    def test_uninstrument_resets_guard(self, config):
        from genai_otel.mcp_instrumentors import falkordb_instrumentor as fmod

        fmod._INSTRUMENTED_MODULES.clear()
        fmod._WRAPPED_TARGETS.clear()

        mock_falkordb = MagicMock()
        with patch.dict("sys.modules", {"falkordb": mock_falkordb}):
            with patch("wrapt.wrap_function_wrapper"):
                instrumentor = fmod.FalkorDBInstrumentor(config)
                instrumentor.instrument()
                assert fmod._is_module_instrumented(mock_falkordb) is True
                assert instrumentor.uninstrument() is True
                assert fmod._is_module_instrumented(mock_falkordb) is False
