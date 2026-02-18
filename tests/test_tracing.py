"""Tests for genai_otel.tracing module."""

import unittest
from unittest.mock import MagicMock, call, patch

from opentelemetry import trace

from genai_otel.tracing import trace_operation


class TestTraceOperation(unittest.TestCase):
    """Tests for trace_operation context manager."""

    def test_creates_span_with_name(self):
        """Test that trace_operation creates a span with the given name."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("genai_otel.tracing.trace.get_tracer", return_value=mock_tracer):
            with trace_operation("test_op"):
                pass

        mock_tracer.start_as_current_span.assert_called_once_with(
            "test_op", kind=trace.SpanKind.INTERNAL, attributes={}
        )

    def test_span_with_attributes(self):
        """Test that trace_operation passes attributes to the span."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("genai_otel.tracing.trace.get_tracer", return_value=mock_tracer):
            with trace_operation("test_op", {"key1": "value1", "key2": "value2"}):
                pass

        mock_tracer.start_as_current_span.assert_called_once_with(
            "test_op",
            kind=trace.SpanKind.INTERNAL,
            attributes={"key1": "value1", "key2": "value2"},
        )

    def test_yields_span_object(self):
        """Test that trace_operation yields the span for further customization."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("genai_otel.tracing.trace.get_tracer", return_value=mock_tracer):
            with trace_operation("test_op") as span:
                span.set_attribute("custom_key", "custom_value")

        mock_span.set_attribute.assert_called_once_with("custom_key", "custom_value")

    def test_custom_span_kind(self):
        """Test that span kind can be configured."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("genai_otel.tracing.trace.get_tracer", return_value=mock_tracer):
            with trace_operation("server_op", kind=trace.SpanKind.SERVER):
                pass

        mock_tracer.start_as_current_span.assert_called_once_with(
            "server_op", kind=trace.SpanKind.SERVER, attributes={}
        )

    def test_default_span_kind_is_internal(self):
        """Test that default span kind is INTERNAL."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("genai_otel.tracing.trace.get_tracer", return_value=mock_tracer):
            with trace_operation("test_op"):
                pass

        call_args = mock_tracer.start_as_current_span.call_args
        self.assertEqual(call_args[1]["kind"], trace.SpanKind.INTERNAL)

    def test_none_attributes_uses_empty_dict(self):
        """Test trace_operation with None attributes uses empty dict."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("genai_otel.tracing.trace.get_tracer", return_value=mock_tracer):
            with trace_operation("test_op", None):
                pass

        call_args = mock_tracer.start_as_current_span.call_args
        self.assertEqual(call_args[1]["attributes"], {})

    def test_exception_propagates(self):
        """Test that exceptions within the context manager propagate."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("genai_otel.tracing.trace.get_tracer", return_value=mock_tracer):
            with self.assertRaises(ValueError):
                with trace_operation("failing_op"):
                    raise ValueError("test error")

    def test_uses_correct_tracer_name(self):
        """Test that trace_operation uses the correct tracer name."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("genai_otel.tracing.trace.get_tracer", return_value=mock_tracer) as mock_get:
            with trace_operation("test_op"):
                pass

        mock_get.assert_called_once_with("genai_otel")

    def test_import_from_package(self):
        """Test that trace_operation can be imported from the package."""
        from genai_otel import trace_operation as imported_func

        self.assertIsNotNone(imported_func)
        self.assertTrue(callable(imported_func))


if __name__ == "__main__":
    unittest.main(verbosity=2)
