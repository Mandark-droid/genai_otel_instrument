import unittest
from unittest.mock import MagicMock, patch

from genai_otel.instrumentors.llamaindex_instrumentor import LlamaIndexInstrumentor


class TestLlamaIndexInstrumentor(unittest.TestCase):
    """Tests for LlamaIndexInstrumentor"""

    def test_instrument_when_llamaindex_not_installed(self):
        """Test that instrument handles missing llama_index gracefully."""
        instrumentor = LlamaIndexInstrumentor()
        config = MagicMock()

        # Mock the import to fail
        with patch.dict("sys.modules", {"llama_index.core.query_engine": None}):
            # Should not raise any exception
            instrumentor.instrument(config)

            # Config should be stored
            self.assertEqual(instrumentor.config, config)

    def test_instrument_with_llamaindex_installed(self):
        """Test that instrument wraps LlamaIndex query engine when installed."""
        instrumentor = LlamaIndexInstrumentor()
        config = MagicMock()

        # Create a mock tracer and set it on the instrumentor
        mock_tracer = MagicMock()
        instrumentor.tracer = mock_tracer

        # Create a mock span context manager
        mock_span = MagicMock()
        mock_span_context = MagicMock()
        mock_span_context.__enter__ = MagicMock(return_value=mock_span)
        mock_span_context.__exit__ = MagicMock(return_value=None)
        mock_tracer.start_as_current_span.return_value = mock_span_context

        # Create mock BaseQueryEngine class
        mock_base_query_engine_class = MagicMock()
        original_query = MagicMock(return_value="query result")
        mock_base_query_engine_class.query = original_query

        # Mock the llama_index.core.query_engine module
        mock_query_engine_module = MagicMock()
        mock_query_engine_module.BaseQueryEngine = mock_base_query_engine_class

        with patch.dict("sys.modules", {"llama_index.core.query_engine": mock_query_engine_module}):
            # Act
            instrumentor.instrument(config)

            # The query method should now be wrapped
            self.assertNotEqual(mock_base_query_engine_class.query, original_query)

            # Create a mock instance and call the wrapped query
            mock_instance = MagicMock()
            result = mock_base_query_engine_class.query(mock_instance, "test query")

            # Assertions
            self.assertEqual(result, "query result")

            # Verify tracing was called
            mock_tracer.start_as_current_span.assert_called_once_with("llamaindex.query_engine")

            # Verify span attributes were set
            mock_span.set_attribute.assert_called_once_with("llamaindex.query", "test query")

    def test_wrapped_query_with_kwargs(self):
        """Test that wrapped query handles kwargs properly."""
        instrumentor = LlamaIndexInstrumentor()
        config = MagicMock()

        # Create a mock tracer and set it on the instrumentor
        mock_tracer = MagicMock()
        instrumentor.tracer = mock_tracer

        # Create a mock span context manager
        mock_span = MagicMock()
        mock_span_context = MagicMock()
        mock_span_context.__enter__ = MagicMock(return_value=mock_span)
        mock_span_context.__exit__ = MagicMock(return_value=None)
        mock_tracer.start_as_current_span.return_value = mock_span_context

        # Create mock BaseQueryEngine class
        mock_base_query_engine_class = MagicMock()
        original_query = MagicMock(return_value="query result")
        mock_base_query_engine_class.query = original_query

        # Mock the llama_index.core.query_engine module
        mock_query_engine_module = MagicMock()
        mock_query_engine_module.BaseQueryEngine = mock_base_query_engine_class

        with patch.dict("sys.modules", {"llama_index.core.query_engine": mock_query_engine_module}):
            # Act
            instrumentor.instrument(config)

            # Call with kwargs
            mock_instance = MagicMock()
            result = mock_base_query_engine_class.query(
                mock_instance, query_str="test query via kwargs"
            )

            # Assertions
            self.assertEqual(result, "query result")

            # Verify span attribute was set with the query from kwargs
            mock_span.set_attribute.assert_called_once_with(
                "llamaindex.query", "test query via kwargs"
            )

    def test_wrapped_query_truncates_long_queries(self):
        """Test that wrapped query truncates long query strings to 200 chars."""
        instrumentor = LlamaIndexInstrumentor()
        config = MagicMock()

        # Create a mock tracer and set it on the instrumentor
        mock_tracer = MagicMock()
        instrumentor.tracer = mock_tracer

        # Create a mock span context manager
        mock_span = MagicMock()
        mock_span_context = MagicMock()
        mock_span_context.__enter__ = MagicMock(return_value=mock_span)
        mock_span_context.__exit__ = MagicMock(return_value=None)
        mock_tracer.start_as_current_span.return_value = mock_span_context

        # Create mock BaseQueryEngine class
        mock_base_query_engine_class = MagicMock()
        original_query = MagicMock(return_value="query result")
        mock_base_query_engine_class.query = original_query

        # Mock the llama_index.core.query_engine module
        mock_query_engine_module = MagicMock()
        mock_query_engine_module.BaseQueryEngine = mock_base_query_engine_class

        with patch.dict("sys.modules", {"llama_index.core.query_engine": mock_query_engine_module}):
            # Act
            instrumentor.instrument(config)

            # Call with a very long query
            long_query = "x" * 300
            mock_instance = MagicMock()
            result = mock_base_query_engine_class.query(mock_instance, long_query)

            # Assertions
            self.assertEqual(result, "query result")

            # Verify span attribute was set with truncated query (200 chars)
            mock_span.set_attribute.assert_called_once_with("llamaindex.query", "x" * 200)

    def test_extract_usage(self):
        """Test that _extract_usage returns None."""
        instrumentor = LlamaIndexInstrumentor()
        result = instrumentor._extract_usage("any_result")

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
