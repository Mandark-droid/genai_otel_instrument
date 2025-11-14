import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.langgraph_instrumentor import LangGraphInstrumentor


class TestLangGraphInstrumentor(unittest.TestCase):
    """Tests for LangGraphInstrumentor"""

    @patch("genai_otel.instrumentors.langgraph_instrumentor.logger")
    def test_init_with_langgraph_available(self, mock_logger):
        """Test that __init__ detects LangGraph availability."""
        with patch.dict("sys.modules", {"langgraph": MagicMock()}):
            instrumentor = LangGraphInstrumentor()

            self.assertTrue(instrumentor._langgraph_available)
            mock_logger.debug.assert_called_with(
                "LangGraph library detected and available for instrumentation"
            )

    @patch("genai_otel.instrumentors.langgraph_instrumentor.logger")
    def test_init_with_langgraph_not_available(self, mock_logger):
        """Test that __init__ handles missing LangGraph gracefully."""
        with patch.dict("sys.modules", {"langgraph": None}):
            instrumentor = LangGraphInstrumentor()

            self.assertFalse(instrumentor._langgraph_available)
            mock_logger.debug.assert_called_with(
                "LangGraph library not installed, instrumentation will be skipped"
            )

    @patch("genai_otel.instrumentors.langgraph_instrumentor.logger")
    def test_instrument_when_langgraph_not_available(self, mock_logger):
        """Test that instrument skips when LangGraph is not available."""
        with patch.dict("sys.modules", {"langgraph": None}):
            instrumentor = LangGraphInstrumentor()
            config = MagicMock()

            instrumentor.instrument(config)

            mock_logger.debug.assert_any_call(
                "Skipping LangGraph instrumentation - library not available"
            )

    @patch("genai_otel.instrumentors.langgraph_instrumentor.logger")
    def test_instrument_with_langgraph_available(self, mock_logger):
        """Test that instrument wraps StateGraph.compile when available."""

        # Create a real StateGraph class
        class MockStateGraph:
            def compile(self):
                # Return a mock compiled graph
                compiled = MagicMock()
                compiled.invoke = MagicMock(return_value={"result": "test"})
                compiled.stream = MagicMock()
                return compiled

        # Create mock langgraph module
        mock_langgraph = MagicMock()
        mock_langgraph.graph.StateGraph = MockStateGraph

        # Create a mock wrapt module
        mock_wrapt = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "langgraph": mock_langgraph,
                "langgraph.graph": mock_langgraph.graph,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = LangGraphInstrumentor()
            config = MagicMock()

            # Act
            instrumentor.instrument(config)

            # Assert
            self.assertEqual(instrumentor.config, config)
            self.assertTrue(instrumentor._instrumented)
            mock_logger.info.assert_called_with("LangGraph instrumentation enabled")
            # Verify FunctionWrapper was called to wrap compile
            mock_wrapt.FunctionWrapper.assert_called()

    @patch("genai_otel.instrumentors.langgraph_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_false(self, mock_logger):
        """Test that instrument handles exceptions gracefully when fail_on_error is False."""
        # Create mock langgraph module
        mock_langgraph = MagicMock()

        # Make hasattr fail to trigger exception
        def mock_hasattr_side_effect(obj, name):
            if name == "StateGraph":
                raise RuntimeError("Test error")
            return True

        with patch.dict("sys.modules", {"langgraph": mock_langgraph, "wrapt": MagicMock()}):
            with patch("builtins.hasattr", side_effect=mock_hasattr_side_effect):
                instrumentor = LangGraphInstrumentor()
                config = MagicMock()
                config.fail_on_error = False

                # Should not raise exception
                instrumentor.instrument(config)

                mock_logger.error.assert_called_once()

    @patch("genai_otel.instrumentors.langgraph_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_true(self, mock_logger):
        """Test that instrument raises exceptions when fail_on_error is True."""
        # Create mock langgraph modules
        mock_langgraph = MagicMock()

        # Make the StateGraph access raise an exception
        mock_langgraph_graph = MagicMock()
        type(mock_langgraph_graph).StateGraph = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Test error"))
        )

        with patch.dict(
            "sys.modules",
            {
                "langgraph": mock_langgraph,
                "langgraph.graph": mock_langgraph_graph,
                "wrapt": MagicMock(),
            },
        ):
            instrumentor = LangGraphInstrumentor()
            config = MagicMock()
            config.fail_on_error = True

            # Should raise exception
            with self.assertRaises(RuntimeError):
                instrumentor.instrument(config)

    def test_extract_graph_attributes_basic(self):
        """Test extraction of basic graph attributes."""
        with patch.dict("sys.modules", {"langgraph": MagicMock()}):
            instrumentor = LangGraphInstrumentor()

            # Create mock state graph
            mock_state_graph = MagicMock()
            mock_state_graph.nodes = {"node1": MagicMock(), "node2": MagicMock()}
            mock_state_graph.edges = [("node1", "node2")]
            mock_state_graph.channels = {"messages": MagicMock(), "state": MagicMock()}

            # Input state
            input_state = {"messages": ["Hello"], "query": "test"}
            args = (input_state,)
            kwargs = {}

            attrs = instrumentor._extract_graph_attributes(None, args, kwargs, mock_state_graph)

            # Assert
            self.assertEqual(attrs["gen_ai.system"], "langgraph")
            self.assertEqual(attrs["gen_ai.operation.name"], "graph.execution")
            self.assertEqual(attrs["langgraph.node_count"], 2)
            self.assertIn("node1", attrs["langgraph.nodes"])
            self.assertIn("node2", attrs["langgraph.nodes"])
            self.assertEqual(attrs["langgraph.edge_count"], 1)
            self.assertEqual(attrs["langgraph.channel_count"], 2)
            self.assertIn("messages", attrs["langgraph.channels"])

    def test_extract_graph_attributes_with_input_state(self):
        """Test extraction of graph attributes with input state."""
        with patch.dict("sys.modules", {"langgraph": MagicMock()}):
            instrumentor = LangGraphInstrumentor()

            # Create mock state graph
            mock_state_graph = MagicMock()

            # Input state with various keys
            input_state = {
                "messages": ["message1", "message2"],
                "query": "What is LangGraph?",
                "custom_key": "custom_value",
            }
            args = (input_state,)
            kwargs = {}

            attrs = instrumentor._extract_graph_attributes(None, args, kwargs, mock_state_graph)

            # Assert
            self.assertIn("messages", attrs["langgraph.input.keys"])
            self.assertIn("query", attrs["langgraph.input.keys"])
            self.assertIn("custom_key", attrs["langgraph.input.keys"])
            self.assertIn("langgraph.input.messages", attrs)
            self.assertIn("langgraph.input.query", attrs)

    def test_extract_graph_attributes_with_config(self):
        """Test extraction of graph attributes with config."""
        with patch.dict("sys.modules", {"langgraph": MagicMock()}):
            instrumentor = LangGraphInstrumentor()

            # Create mock state graph
            mock_state_graph = MagicMock()

            # Input with config
            input_state = {"messages": []}
            config = {
                "configurable": {"thread_id": "thread_123", "checkpoint_id": "checkpoint_456"},
                "recursion_limit": 25,
            }
            args = (input_state,)
            kwargs = {"config": config}

            attrs = instrumentor._extract_graph_attributes(None, args, kwargs, mock_state_graph)

            # Assert
            self.assertEqual(attrs["langgraph.thread_id"], "thread_123")
            self.assertEqual(attrs["langgraph.checkpoint_id"], "checkpoint_456")
            self.assertEqual(attrs["langgraph.recursion_limit"], 25)

    def test_extract_response_attributes_dict_result(self):
        """Test extraction of response attributes from dict result."""
        with patch.dict("sys.modules", {"langgraph": MagicMock()}):
            instrumentor = LangGraphInstrumentor()

            # Test with dict result
            result = {
                "messages": ["msg1", "msg2", "msg3"],
                "answer": "This is the answer",
                "output": "Final output",
            }

            attrs = instrumentor._extract_response_attributes(result)

            # Assert
            self.assertIn("messages", attrs["langgraph.output.keys"])
            self.assertIn("answer", attrs["langgraph.output.keys"])
            self.assertIn("output", attrs["langgraph.output.keys"])
            self.assertEqual(attrs["langgraph.message_count"], 3)
            self.assertIn("langgraph.output.messages", attrs)
            self.assertIn("langgraph.output.answer", attrs)

    def test_extract_response_attributes_with_metadata(self):
        """Test extraction of response attributes with metadata."""
        with patch.dict("sys.modules", {"langgraph": MagicMock()}):
            instrumentor = LangGraphInstrumentor()

            # Create mock result with metadata using a class that acts like a dict
            class ResultWithMetadata(dict):
                pass

            result = ResultWithMetadata({"result": "test"})
            result.__metadata__ = {"step": 5}

            attrs = instrumentor._extract_response_attributes(result)

            # Assert
            self.assertEqual(attrs["langgraph.steps"], 5)

    def test_extract_usage_returns_none(self):
        """Test that extract_usage returns None (LangGraph doesn't provide usage)."""
        with patch.dict("sys.modules", {"langgraph": MagicMock()}):
            instrumentor = LangGraphInstrumentor()

            # Create mock result
            mock_result = MagicMock()

            usage = instrumentor._extract_usage(mock_result)

            # Assert
            self.assertIsNone(usage)

    def test_extract_finish_reason(self):
        """Test extraction of finish reason from result."""
        with patch.dict("sys.modules", {"langgraph": MagicMock()}):
            instrumentor = LangGraphInstrumentor()

            # Create mock result
            mock_result = {"output": "test"}

            finish_reason = instrumentor._extract_finish_reason(mock_result)

            # Assert
            self.assertEqual(finish_reason, "completed")

    def test_extract_finish_reason_none_result(self):
        """Test extraction of finish reason with None result."""
        with patch.dict("sys.modules", {"langgraph": MagicMock()}):
            instrumentor = LangGraphInstrumentor()

            # None result
            finish_reason = instrumentor._extract_finish_reason(None)

            # Assert
            self.assertIsNone(finish_reason)


if __name__ == "__main__":
    unittest.main()
