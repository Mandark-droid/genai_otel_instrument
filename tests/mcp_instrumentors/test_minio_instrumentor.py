"""Tests for MinIO MCP instrumentor."""

import logging
import sys
from unittest.mock import MagicMock, call, patch

import pytest

from genai_otel.config import OTelConfig


@pytest.fixture
def config():
    return OTelConfig()


class TestMinIOInstrumentor:
    """Tests for MinIOInstrumentor."""

    def test_init(self, config):
        """Test that config is stored on the instrumentor."""
        from genai_otel.mcp_instrumentors.minio_instrumentor import MinIOInstrumentor

        instrumentor = MinIOInstrumentor(config)
        assert instrumentor.config is config

    @patch("wrapt.wrap_function_wrapper")
    def test_instrument_success(self, mock_wrap, config, caplog):
        """Test successful instrumentation wraps all expected methods."""
        # Create a mock minio module
        mock_minio = MagicMock()
        with patch.dict("sys.modules", {"minio": mock_minio}):
            from genai_otel.mcp_instrumentors.minio_instrumentor import MinIOInstrumentor

            instrumentor = MinIOInstrumentor(config)
            caplog.set_level(logging.INFO)
            result = instrumentor.instrument()

            assert result is True
            assert "MinIO instrumentation enabled" in caplog.text

            expected_methods = [
                "put_object",
                "get_object",
                "remove_object",
                "list_objects",
                "make_bucket",
                "remove_bucket",
                "list_buckets",
                "stat_object",
                "fput_object",
                "fget_object",
            ]
            assert mock_wrap.call_count == len(expected_methods)
            wrapped_methods = [call_args[0][1] for call_args in mock_wrap.call_args_list]
            for method_name in expected_methods:
                assert f"Minio.{method_name}" in wrapped_methods
                # Verify all calls use "minio" as the module
            for call_args in mock_wrap.call_args_list:
                assert call_args[0][0] == "minio"

    def test_instrument_missing(self, config, caplog):
        """Test graceful handling when minio is not installed."""
        with patch.dict("sys.modules", {"minio": None}):
            from genai_otel.mcp_instrumentors.minio_instrumentor import MinIOInstrumentor

            instrumentor = MinIOInstrumentor(config)
            caplog.set_level(logging.DEBUG)
            result = instrumentor.instrument()

            assert result is False
            assert any(
                "minio not installed, skipping MinIO instrumentation" in r.message
                and r.levelno == logging.DEBUG
                for r in caplog.records
            )

    @patch("wrapt.wrap_function_wrapper", side_effect=RuntimeError("Mock error"))
    def test_instrument_error(self, mock_wrap, config, caplog):
        """Test error logging on general exception."""
        mock_minio = MagicMock()
        with patch.dict("sys.modules", {"minio": mock_minio}):
            from genai_otel.mcp_instrumentors.minio_instrumentor import MinIOInstrumentor

            instrumentor = MinIOInstrumentor(config)
            caplog.set_level(logging.ERROR)
            result = instrumentor.instrument()

            assert result is False
            assert any(
                "Failed to instrument MinIO: Mock error" in r.message and r.levelno == logging.ERROR
                for r in caplog.records
            )

    def test_wrapper_creates_span_with_attributes(self, config):
        """Test that wrapped methods create spans with correct attributes."""
        mock_minio = MagicMock()
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        # Capture the wrapper functions passed to wrapt
        wrappers = {}

        def capture_wrapper(module, name, wrapper_func):
            method_name = name.split(".")[-1]
            wrappers[method_name] = wrapper_func

        with patch.dict("sys.modules", {"minio": mock_minio}):
            with patch("wrapt.wrap_function_wrapper", side_effect=capture_wrapper):
                with patch("genai_otel.mcp_instrumentors.minio_instrumentor.trace") as mock_trace:
                    mock_trace.get_tracer.return_value = mock_tracer

                    from genai_otel.mcp_instrumentors.minio_instrumentor import MinIOInstrumentor

                    instrumentor = MinIOInstrumentor(config)
                    instrumentor.instrument()

        assert "put_object" in wrappers

        # Call the wrapper with positional args (bucket, object_name)
        mock_wrapped = MagicMock(return_value="result")
        wrappers["put_object"](mock_wrapped, None, ("my-bucket", "my-object.txt"), {})

        mock_tracer.start_as_current_span.assert_called_with(
            "minio.put_object",
            kind=pytest.approx(mock_tracer.start_as_current_span.call_args[1]["kind"], abs=1),
        )
        mock_span.set_attribute.assert_any_call("db.system", "minio")
        mock_span.set_attribute.assert_any_call("db.operation", "put_object")
        mock_span.set_attribute.assert_any_call("minio.bucket", "my-bucket")
        mock_span.set_attribute.assert_any_call("minio.object", "my-object.txt")

    def test_wrapper_handles_exception(self, config):
        """Test that exceptions are recorded on spans."""
        mock_minio = MagicMock()
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        wrappers = {}

        def capture_wrapper(module, name, wrapper_func):
            method_name = name.split(".")[-1]
            wrappers[method_name] = wrapper_func

        with patch.dict("sys.modules", {"minio": mock_minio}):
            with patch("wrapt.wrap_function_wrapper", side_effect=capture_wrapper):
                with patch("genai_otel.mcp_instrumentors.minio_instrumentor.trace") as mock_trace:
                    mock_trace.get_tracer.return_value = mock_tracer

                    from genai_otel.mcp_instrumentors.minio_instrumentor import MinIOInstrumentor

                    instrumentor = MinIOInstrumentor(config)
                    instrumentor.instrument()

        assert "get_object" in wrappers

        # Make the wrapped function raise an exception
        test_error = ConnectionError("Connection refused")
        mock_wrapped = MagicMock(side_effect=test_error)

        with pytest.raises(ConnectionError, match="Connection refused"):
            wrappers["get_object"](mock_wrapped, None, ("my-bucket", "my-key"), {})

        mock_span.record_exception.assert_called_once_with(test_error)
