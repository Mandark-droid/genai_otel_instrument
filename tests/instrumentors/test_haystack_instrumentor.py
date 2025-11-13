import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.haystack_instrumentor import HaystackInstrumentor


class TestHaystackInstrumentor(unittest.TestCase):
    """Tests for HaystackInstrumentor"""

    @patch("genai_otel.instrumentors.haystack_instrumentor.logger")
    def test_init_with_haystack_available(self, mock_logger):
        """Test that __init__ detects Haystack availability."""
        with patch.dict("sys.modules", {"haystack": MagicMock()}):
            instrumentor = HaystackInstrumentor()

            self.assertTrue(instrumentor._haystack_available)
            mock_logger.debug.assert_called_with(
                "Haystack library detected and available for instrumentation"
            )

    @patch("genai_otel.instrumentors.haystack_instrumentor.logger")
    def test_init_with_haystack_not_available(self, mock_logger):
        """Test that __init__ handles missing Haystack gracefully."""
        with patch.dict("sys.modules", {"haystack": None}):
            instrumentor = HaystackInstrumentor()

            self.assertFalse(instrumentor._haystack_available)
            mock_logger.debug.assert_called_with(
                "Haystack library not installed, instrumentation will be skipped"
            )

    @patch("genai_otel.instrumentors.haystack_instrumentor.logger")
    def test_instrument_when_haystack_not_available(self, mock_logger):
        """Test that instrument skips when Haystack is not available."""
        with patch.dict("sys.modules", {"haystack": None}):
            instrumentor = HaystackInstrumentor()
            config = MagicMock()

            instrumentor.instrument(config)

            mock_logger.debug.assert_any_call(
                "Skipping Haystack instrumentation - library not available"
            )

    @patch("genai_otel.instrumentors.haystack_instrumentor.logger")
    def test_instrument_with_pipeline(self, mock_logger):
        """Test that instrument wraps Pipeline.run method."""

        # Create mock Pipeline class
        class MockPipeline:
            def run(self, data=None):
                return {"output": "result"}

        # Create mock haystack module
        mock_haystack = MagicMock()
        mock_haystack.Pipeline = MockPipeline

        mock_wrapt = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "haystack": mock_haystack,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = HaystackInstrumentor()
            config = MagicMock()

            # Call instrument
            instrumentor.instrument(config)

            # Assert
            self.assertEqual(instrumentor.config, config)
            self.assertTrue(instrumentor._instrumented)
            mock_logger.info.assert_called_with("Haystack instrumentation enabled")
            # Verify FunctionWrapper was called
            self.assertTrue(mock_wrapt.FunctionWrapper.called)

    @patch("genai_otel.instrumentors.haystack_instrumentor.logger")
    def test_instrument_with_pipeline_run_async(self, mock_logger):
        """Test that instrument wraps Pipeline.run_async method."""

        # Create mock Pipeline class with async method
        class MockPipeline:
            async def run_async(self, data=None):
                return {"output": "async result"}

        # Create mock haystack module
        mock_haystack = MagicMock()
        mock_haystack.Pipeline = MockPipeline

        mock_wrapt = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "haystack": mock_haystack,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = HaystackInstrumentor()
            config = MagicMock()

            # Call instrument
            instrumentor.instrument(config)

            # Assert
            self.assertTrue(instrumentor._instrumented)
            mock_logger.info.assert_called_with("Haystack instrumentation enabled")

    @patch("genai_otel.instrumentors.haystack_instrumentor.logger")
    def test_instrument_with_generators(self, mock_logger):
        """Test that instrument wraps Generator components."""

        # Create mock Generator classes
        class MockOpenAIGenerator:
            def run(self, prompt=None):
                return {"replies": ["Generated text"]}

        class MockOpenAIChatGenerator:
            def run(self, messages=None):
                return {"replies": ["Chat response"]}

        # Create mock haystack module
        mock_haystack = MagicMock()
        mock_generators = MagicMock()
        mock_generators.OpenAIGenerator = MockOpenAIGenerator
        mock_generators.OpenAIChatGenerator = MockOpenAIChatGenerator

        # Mock the import path
        mock_components = MagicMock()
        mock_components.generators = mock_generators

        mock_wrapt = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "haystack": mock_haystack,
                "haystack.components": mock_components,
                "haystack.components.generators": mock_generators,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = HaystackInstrumentor()
            config = MagicMock()

            # Call instrument
            instrumentor.instrument(config)

            # Assert
            self.assertTrue(instrumentor._instrumented)

    @patch("genai_otel.instrumentors.haystack_instrumentor.logger")
    def test_instrument_with_retriever(self, mock_logger):
        """Test that instrument wraps Retriever components."""

        # Create mock Retriever class
        class MockInMemoryBM25Retriever:
            def run(self, query=None, top_k=10):
                return {"documents": []}

        # Create mock haystack module
        mock_haystack = MagicMock()
        mock_retrievers = MagicMock()
        mock_retrievers.InMemoryBM25Retriever = MockInMemoryBM25Retriever

        # Mock the import path
        mock_components = MagicMock()
        mock_components.retrievers = mock_retrievers

        mock_wrapt = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "haystack": mock_haystack,
                "haystack.components": mock_components,
                "haystack.components.retrievers": mock_retrievers,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = HaystackInstrumentor()
            config = MagicMock()

            # Call instrument
            instrumentor.instrument(config)

            # Assert
            self.assertTrue(instrumentor._instrumented)

    @patch("genai_otel.instrumentors.haystack_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_false(self, mock_logger):
        """Test that exceptions are logged when fail_on_error is False."""
        # Create mock that raises
        mock_haystack = MagicMock()
        type(mock_haystack).Pipeline = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Access failed"))
        )

        with patch.dict("sys.modules", {"haystack": mock_haystack, "wrapt": MagicMock()}):
            instrumentor = HaystackInstrumentor()
            config = MagicMock()
            config.fail_on_error = False

            # Should not raise
            instrumentor.instrument(config)

            mock_logger.error.assert_called_once()

    @patch("genai_otel.instrumentors.haystack_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_true(self, mock_logger):
        """Test that exceptions are raised when fail_on_error is True."""
        # Create mock that raises
        mock_haystack = MagicMock()
        type(mock_haystack).Pipeline = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Access failed"))
        )

        with patch.dict("sys.modules", {"haystack": mock_haystack, "wrapt": MagicMock()}):
            instrumentor = HaystackInstrumentor()
            config = MagicMock()
            config.fail_on_error = True

            # Should raise
            with self.assertRaises(RuntimeError) as context:
                instrumentor.instrument(config)

            self.assertEqual(str(context.exception), "Access failed")

    def test_extract_pipeline_attributes_basic(self):
        """Test extraction of basic pipeline attributes."""
        with patch.dict("sys.modules", {"haystack": MagicMock()}):
            instrumentor = HaystackInstrumentor()

            # Create mock pipeline instance
            mock_pipeline = MagicMock()
            mock_pipeline.metadata = {"name": "test_pipeline", "version": "1.0"}

            # Create mock graph
            mock_graph = MagicMock()
            mock_graph.nodes.return_value = ["node1", "node2", "node3"]
            mock_graph.edges.return_value = [("node1", "node2"), ("node2", "node3")]
            mock_pipeline.graph = mock_graph

            args = ()
            kwargs = {"data": {"query": "What is Haystack?"}}

            attrs = instrumentor._extract_pipeline_attributes(mock_pipeline, args, kwargs)

            # Assert
            self.assertEqual(attrs["gen_ai.system"], "haystack")
            self.assertEqual(attrs["gen_ai.operation.name"], "pipeline.run")
            self.assertEqual(attrs["haystack.pipeline.components.count"], 3)
            self.assertEqual(attrs["haystack.pipeline.connections.count"], 2)
            self.assertEqual(attrs["haystack.pipeline.input.query"], "What is Haystack?")

    def test_extract_pipeline_attributes_with_metadata(self):
        """Test extraction of pipeline attributes with metadata."""
        with patch.dict("sys.modules", {"haystack": MagicMock()}):
            instrumentor = HaystackInstrumentor()

            # Create mock pipeline instance
            mock_pipeline = MagicMock()
            mock_pipeline.metadata = {
                "pipeline_name": "qa_pipeline",
                "author": "test_user",
                "description": "Question answering pipeline",
            }

            args = ()
            kwargs = {}

            attrs = instrumentor._extract_pipeline_attributes(mock_pipeline, args, kwargs)

            # Assert
            self.assertIn("haystack.pipeline.metadata.pipeline_name", attrs)
            self.assertEqual(attrs["haystack.pipeline.metadata.pipeline_name"], "qa_pipeline")

    def test_extract_generator_attributes(self):
        """Test extraction of generator attributes."""
        with patch.dict("sys.modules", {"haystack": MagicMock()}):
            instrumentor = HaystackInstrumentor()

            # Create mock generator instance
            mock_generator = MagicMock()
            mock_generator.model = "gpt-4"
            mock_generator.generation_kwargs = {
                "max_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.9,
            }

            args = ()
            kwargs = {"prompt": "Generate a summary"}

            attrs = instrumentor._extract_generator_attributes(mock_generator, args, kwargs)

            # Assert
            self.assertEqual(attrs["gen_ai.system"], "haystack")
            self.assertEqual(attrs["gen_ai.operation.name"], "generator.run")
            self.assertEqual(attrs["haystack.component.type"], "generator")
            self.assertEqual(attrs["gen_ai.request.model"], "gpt-4")
            self.assertEqual(attrs["gen_ai.request.max_tokens"], 500)
            self.assertEqual(attrs["gen_ai.request.temperature"], 0.7)
            self.assertEqual(attrs["gen_ai.request.top_p"], 0.9)
            self.assertEqual(attrs["haystack.generator.prompt"], "Generate a summary")

    def test_extract_chat_generator_attributes(self):
        """Test extraction of chat generator attributes."""
        with patch.dict("sys.modules", {"haystack": MagicMock()}):
            instrumentor = HaystackInstrumentor()

            # Create mock chat generator instance
            mock_chat_generator = MagicMock()
            mock_chat_generator.model = "gpt-3.5-turbo"
            mock_chat_generator.generation_kwargs = {"max_tokens": 1000, "temperature": 0.5}

            # Create mock messages
            mock_msg1 = MagicMock()
            mock_msg1.content = "What is AI?"
            mock_msg1.role = "user"

            mock_msg2 = MagicMock()
            mock_msg2.content = "AI stands for Artificial Intelligence"
            mock_msg2.role = "assistant"

            args = ()
            kwargs = {"messages": [mock_msg1, mock_msg2]}

            attrs = instrumentor._extract_chat_generator_attributes(
                mock_chat_generator, args, kwargs
            )

            # Assert
            self.assertEqual(attrs["gen_ai.system"], "haystack")
            self.assertEqual(attrs["gen_ai.operation.name"], "chat_generator.run")
            self.assertEqual(attrs["haystack.component.type"], "chat_generator")
            self.assertEqual(attrs["gen_ai.request.model"], "gpt-3.5-turbo")
            self.assertEqual(attrs["haystack.chat_generator.messages.count"], 2)
            self.assertEqual(
                attrs["haystack.chat_generator.last_message"],
                "AI stands for Artificial Intelligence",
            )
            self.assertEqual(attrs["haystack.chat_generator.last_role"], "assistant")

    def test_extract_retriever_attributes(self):
        """Test extraction of retriever attributes."""
        with patch.dict("sys.modules", {"haystack": MagicMock()}):
            instrumentor = HaystackInstrumentor()

            # Create mock retriever instance
            mock_retriever = MagicMock()

            args = ()
            kwargs = {
                "query": "search query",
                "top_k": 5,
                "filters": {"year": 2024},
            }

            attrs = instrumentor._extract_retriever_attributes(mock_retriever, args, kwargs)

            # Assert
            self.assertEqual(attrs["gen_ai.system"], "haystack")
            self.assertEqual(attrs["gen_ai.operation.name"], "retriever.run")
            self.assertEqual(attrs["haystack.component.type"], "retriever")
            self.assertEqual(attrs["haystack.retriever.query"], "search query")
            self.assertEqual(attrs["haystack.retriever.top_k"], 5)
            self.assertIn("2024", attrs["haystack.retriever.filters"])

    def test_extract_usage_from_result(self):
        """Test that _extract_usage extracts from generator output."""
        with patch.dict("sys.modules", {"haystack": MagicMock()}):
            instrumentor = HaystackInstrumentor()

            # Create mock result with usage info
            result = {
                "generator": {
                    "replies": ["Generated text"],
                    "meta": [
                        {
                            "usage": {
                                "prompt_tokens": 50,
                                "completion_tokens": 100,
                                "total_tokens": 150,
                            }
                        }
                    ],
                }
            }

            usage = instrumentor._extract_usage(result)

            self.assertIsNotNone(usage)
            self.assertEqual(usage["prompt_tokens"], 50)
            self.assertEqual(usage["completion_tokens"], 100)
            self.assertEqual(usage["total_tokens"], 150)

    def test_extract_usage_without_usage_info(self):
        """Test that _extract_usage returns None when no usage."""
        with patch.dict("sys.modules", {"haystack": MagicMock()}):
            instrumentor = HaystackInstrumentor()

            # Create mock result without usage
            result = {"output": "some result"}

            usage = instrumentor._extract_usage(result)

            self.assertIsNone(usage)

    def test_extract_response_attributes_with_generator_output(self):
        """Test extraction of response attributes with generator output."""
        with patch.dict("sys.modules", {"haystack": MagicMock()}):
            instrumentor = HaystackInstrumentor()

            # Create mock result
            result = {
                "generator": {
                    "replies": ["First reply", "Second reply"],
                    "meta": [{"model": "gpt-4"}],
                },
                "retriever": {"documents": [{"content": "doc1"}, {"content": "doc2"}]},
            }

            attrs = instrumentor._extract_response_attributes(result)

            # Assert
            self.assertIn("generator", attrs["haystack.pipeline.output.keys"])
            self.assertIn("retriever", attrs["haystack.pipeline.output.keys"])
            self.assertEqual(attrs["haystack.output.generator.replies.count"], 2)
            self.assertEqual(attrs["haystack.output.generator.first_reply"], "First reply")
            self.assertEqual(attrs["haystack.output.retriever.documents.count"], 2)

    def test_extract_response_attributes_empty_result(self):
        """Test extraction of response attributes with empty result."""
        with patch.dict("sys.modules", {"haystack": MagicMock()}):
            instrumentor = HaystackInstrumentor()

            # Create empty result
            result = {}

            attrs = instrumentor._extract_response_attributes(result)

            # Assert
            self.assertEqual(attrs["haystack.pipeline.output.keys"], [])

    def test_extract_finish_reason_with_finish_reason(self):
        """Test extraction of finish reason from generator output."""
        with patch.dict("sys.modules", {"haystack": MagicMock()}):
            instrumentor = HaystackInstrumentor()

            # Create mock result with finish_reason
            result = {
                "generator": {
                    "replies": ["Generated text"],
                    "meta": [{"finish_reason": "stop"}],
                }
            }

            finish_reason = instrumentor._extract_finish_reason(result)

            # Assert
            self.assertEqual(finish_reason, "stop")

    def test_extract_finish_reason_completed(self):
        """Test extraction of finish reason when result exists."""
        with patch.dict("sys.modules", {"haystack": MagicMock()}):
            instrumentor = HaystackInstrumentor()

            # Create mock result without finish_reason
            result = {"output": "some result"}

            finish_reason = instrumentor._extract_finish_reason(result)

            # Assert
            self.assertEqual(finish_reason, "completed")

    def test_extract_finish_reason_none(self):
        """Test extraction of finish reason returns None for empty result."""
        with patch.dict("sys.modules", {"haystack": MagicMock()}):
            instrumentor = HaystackInstrumentor()

            # Empty result
            result = None

            finish_reason = instrumentor._extract_finish_reason(result)

            # Assert
            self.assertIsNone(finish_reason)


if __name__ == "__main__":
    unittest.main()
