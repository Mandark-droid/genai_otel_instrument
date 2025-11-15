import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.dspy_instrumentor import DSPyInstrumentor


class TestDSPyInstrumentor(unittest.TestCase):
    """Tests for DSPyInstrumentor"""

    @patch("genai_otel.instrumentors.dspy_instrumentor.logger")
    def test_init_with_dspy_available(self, mock_logger):
        """Test that __init__ detects DSPy availability."""
        # Create mock dspy module
        mock_dspy = MagicMock()

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            instrumentor = DSPyInstrumentor()

            self.assertTrue(instrumentor._dspy_available)
            mock_logger.debug.assert_called_with(
                "DSPy framework detected and available for instrumentation"
            )

    @patch("genai_otel.instrumentors.dspy_instrumentor.logger")
    def test_init_with_dspy_not_available(self, mock_logger):
        """Test that __init__ handles missing DSPy."""
        with patch.dict("sys.modules", {"dspy": None}):
            instrumentor = DSPyInstrumentor()

            self.assertFalse(instrumentor._dspy_available)
            mock_logger.debug.assert_called_with(
                "DSPy not installed, instrumentation will be skipped"
            )

    @patch("genai_otel.instrumentors.dspy_instrumentor.logger")
    def test_instrument_when_dspy_not_available(self, mock_logger):
        """Test that instrument skips when DSPy is not available."""
        with patch.dict("sys.modules", {"dspy": None}):
            instrumentor = DSPyInstrumentor()
            config = MagicMock()

            instrumentor.instrument(config)

            mock_logger.debug.assert_any_call(
                "Skipping DSPy instrumentation - library not available"
            )

    @patch("genai_otel.instrumentors.dspy_instrumentor.logger")
    def test_instrument_wraps_module_call(self, mock_logger):
        """Test that instrument wraps Module.__call__ method."""
        # Create mock modules
        mock_dspy = MagicMock()
        mock_module = MagicMock()
        mock_dspy.primitives.module.BaseModule = mock_module

        mock_wrapt = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "dspy": mock_dspy,
                "dspy.primitives": mock_dspy.primitives,
                "dspy.primitives.module": mock_dspy.primitives.module,
                "dspy.predict": mock_dspy.predict,
                "dspy.predict.predict": mock_dspy.predict.predict,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = DSPyInstrumentor()
            config = MagicMock()

            # Call instrument
            instrumentor.instrument(config)

            # Assert wrapt.wrap_function_wrapper was called
            self.assertTrue(mock_wrapt.wrap_function_wrapper.called)
            self.assertEqual(instrumentor.config, config)
            self.assertTrue(instrumentor._instrumented)
            mock_logger.info.assert_called_with("DSPy instrumentation enabled")

    @patch("genai_otel.instrumentors.dspy_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_false(self, mock_logger):
        """Test that exceptions are logged when fail_on_error is False."""
        mock_dspy = MagicMock()
        mock_wrapt = MagicMock()
        mock_wrapt.wrap_function_wrapper.side_effect = RuntimeError("Wrap failed")

        with patch.dict(
            "sys.modules",
            {
                "dspy": mock_dspy,
                "dspy.primitives": mock_dspy.primitives,
                "dspy.primitives.module": mock_dspy.primitives.module,
                "dspy.predict": mock_dspy.predict,
                "dspy.predict.predict": mock_dspy.predict.predict,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = DSPyInstrumentor()
            config = MagicMock()
            config.fail_on_error = False

            # Should not raise
            instrumentor.instrument(config)

            mock_logger.error.assert_called_once()

    @patch("genai_otel.instrumentors.dspy_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_true(self, mock_logger):
        """Test that exceptions are raised when fail_on_error is True."""
        mock_dspy = MagicMock()
        mock_wrapt = MagicMock()
        mock_wrapt.wrap_function_wrapper.side_effect = RuntimeError("Wrap failed")

        with patch.dict(
            "sys.modules",
            {
                "dspy": mock_dspy,
                "dspy.primitives": mock_dspy.primitives,
                "dspy.primitives.module": mock_dspy.primitives.module,
                "dspy.predict": mock_dspy.predict,
                "dspy.predict.predict": mock_dspy.predict.predict,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = DSPyInstrumentor()
            config = MagicMock()
            config.fail_on_error = True

            # Should raise
            with self.assertRaises(RuntimeError) as context:
                instrumentor.instrument(config)

            self.assertEqual(str(context.exception), "Wrap failed")

    def test_extract_module_attributes(self):
        """Test extraction of module attributes."""
        mock_dspy = MagicMock()

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            instrumentor = DSPyInstrumentor()

            # Create mock module instance
            mock_module = MagicMock()
            mock_module.__class__.__name__ = "CustomModule"
            mock_module.name = "my_module"

            # Create mock signature
            mock_signature = MagicMock()
            mock_signature.__name__ = "QuestionAnswer"
            mock_module.signature = mock_signature

            kwargs = {"question": "What is DSPy?", "context": "DSPy is a framework"}

            attrs = instrumentor._extract_module_attributes(mock_module, (), kwargs)

            # Assert
            self.assertEqual(attrs["gen_ai.system"], "dspy")
            self.assertEqual(attrs["gen_ai.operation.name"], "module.call")
            self.assertEqual(attrs["dspy.module.name"], "CustomModule")
            self.assertEqual(attrs["dspy.module.instance_name"], "my_module")
            self.assertEqual(attrs["dspy.module.signature"], "QuestionAnswer")
            self.assertIn("question", attrs["dspy.module.input_keys"])
            self.assertEqual(attrs["dspy.module.input_count"], 2)

    def test_extract_predict_attributes(self):
        """Test extraction of predict attributes."""
        mock_dspy = MagicMock()

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            instrumentor = DSPyInstrumentor()

            # Create mock Predict instance
            mock_predict = MagicMock()

            # Create mock signature
            mock_signature = MagicMock()
            mock_signature.__name__ = "QuestionAnswer"
            mock_signature.instructions = "Answer the question based on context"

            # Create mock fields
            input_field = MagicMock()
            input_field.input_variable = "question"
            output_field = MagicMock()
            output_field.output_variable = "answer"

            mock_signature.input_fields = [input_field]
            mock_signature.output_fields = [output_field]

            mock_predict.signature = mock_signature

            kwargs = {"question": "What is DSPy?"}

            attrs = instrumentor._extract_predict_attributes(mock_predict, kwargs)

            # Assert
            self.assertEqual(attrs["gen_ai.system"], "dspy")
            self.assertEqual(attrs["gen_ai.operation.name"], "predict")
            self.assertEqual(attrs["dspy.predict.signature"], "QuestionAnswer")
            self.assertIn("Answer the question", attrs["dspy.predict.instructions"])
            self.assertEqual(attrs["dspy.predict.input_fields"], ["question"])
            self.assertEqual(attrs["dspy.predict.output_fields"], ["answer"])
            self.assertEqual(attrs["dspy.predict.input.question"], "What is DSPy?")

    def test_extract_cot_attributes(self):
        """Test extraction of chain-of-thought attributes."""
        mock_dspy = MagicMock()

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            instrumentor = DSPyInstrumentor()

            # Create mock ChainOfThought instance
            mock_cot = MagicMock()

            # Create mock signature
            mock_signature = MagicMock()
            mock_signature.__name__ = "ReasoningSignature"
            mock_cot.signature = mock_signature

            # Create mock extended signature
            mock_ext_sig = MagicMock()
            output_field1 = MagicMock()
            output_field1.output_variable = "rationale"
            output_field2 = MagicMock()
            output_field2.output_variable = "answer"

            mock_ext_sig.output_fields = [output_field1, output_field2]
            mock_cot.extended_signature = mock_ext_sig

            kwargs = {"question": "Explain DSPy chain-of-thought"}

            attrs = instrumentor._extract_cot_attributes(mock_cot, kwargs)

            # Assert
            self.assertEqual(attrs["gen_ai.system"], "dspy")
            self.assertEqual(attrs["gen_ai.operation.name"], "chain_of_thought")
            self.assertEqual(attrs["dspy.cot.signature"], "ReasoningSignature")
            self.assertEqual(attrs["dspy.cot.output_fields"], ["rationale", "answer"])
            self.assertIn("Explain DSPy", attrs["dspy.cot.input.question"])

    def test_extract_react_attributes(self):
        """Test extraction of ReAct attributes."""
        mock_dspy = MagicMock()

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            instrumentor = DSPyInstrumentor()

            # Create mock ReAct instance
            mock_react = MagicMock()

            # Create mock signature
            mock_signature = MagicMock()
            mock_signature.__name__ = "AgentSignature"
            mock_react.signature = mock_signature

            # Create mock tools
            tool1 = MagicMock()
            tool1.__name__ = "search"
            tool2 = MagicMock()
            tool2.__name__ = "calculate"

            mock_react.tools = [tool1, tool2]

            kwargs = {"task": "Find and calculate average temperature"}

            attrs = instrumentor._extract_react_attributes(mock_react, kwargs)

            # Assert
            self.assertEqual(attrs["gen_ai.system"], "dspy")
            self.assertEqual(attrs["gen_ai.operation.name"], "react")
            self.assertEqual(attrs["dspy.react.signature"], "AgentSignature")
            self.assertEqual(attrs["dspy.react.tools"], ["search", "calculate"])
            self.assertEqual(attrs["dspy.react.tools_count"], 2)
            self.assertIn("Find and calculate", attrs["dspy.react.input.task"])

    def test_extract_optimizer_attributes(self):
        """Test extraction of optimizer attributes."""
        mock_dspy = MagicMock()

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            instrumentor = DSPyInstrumentor()

            # Create mock COPRO optimizer
            mock_optimizer = MagicMock()
            mock_optimizer.__class__.__name__ = "COPRO"
            mock_optimizer.metric = MagicMock()
            mock_optimizer.breadth = 10
            mock_optimizer.depth = 3

            # Create mock datasets
            trainset = [MagicMock() for _ in range(100)]
            valset = [MagicMock() for _ in range(50)]

            kwargs = {"trainset": trainset, "valset": valset}

            attrs = instrumentor._extract_optimizer_attributes(mock_optimizer, (), kwargs)

            # Assert
            self.assertEqual(attrs["gen_ai.system"], "dspy")
            self.assertEqual(attrs["gen_ai.operation.name"], "optimizer.compile")
            self.assertEqual(attrs["dspy.optimizer.name"], "COPRO")
            self.assertTrue(attrs["dspy.optimizer.has_metric"])
            self.assertEqual(attrs["dspy.optimizer.trainset_size"], 100)
            self.assertEqual(attrs["dspy.optimizer.valset_size"], 50)
            self.assertEqual(attrs["dspy.optimizer.copro.breadth"], 10)
            self.assertEqual(attrs["dspy.optimizer.copro.depth"], 3)

    def test_extract_module_response_attributes(self):
        """Test extraction of module response attributes."""
        mock_dspy = MagicMock()

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            instrumentor = DSPyInstrumentor()

            # Create mock Prediction result
            mock_result = MagicMock()
            mock_result.__class__.__name__ = "Prediction"
            mock_result._store = {"answer": "DSPy is a framework", "confidence": 0.95}

            attrs = instrumentor._extract_module_response_attributes(mock_result)

            # Assert
            self.assertIn("answer", attrs["dspy.module.output_keys"])
            self.assertIn("confidence", attrs["dspy.module.output_keys"])
            self.assertEqual(attrs["dspy.module.output_count"], 2)

    def test_extract_predict_response_attributes(self):
        """Test extraction of predict response attributes."""
        mock_dspy = MagicMock()

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            instrumentor = DSPyInstrumentor()

            # Create mock Prediction result
            mock_result = MagicMock()
            mock_result._store = {
                "answer": "DSPy is a Stanford NLP framework for programming language models",
                "confidence": 0.95,
            }

            attrs = instrumentor._extract_predict_response_attributes(mock_result)

            # Assert
            self.assertIn("Stanford NLP", attrs["dspy.predict.output.answer"])
            self.assertEqual(attrs["dspy.predict.output.confidence"], "0.95")

    def test_extract_cot_response_attributes(self):
        """Test extraction of chain-of-thought response attributes."""
        mock_dspy = MagicMock()

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            instrumentor = DSPyInstrumentor()

            # Create mock result with reasoning
            mock_result = MagicMock()
            mock_result._store = {
                "rationale": "First, I'll break down the question. DSPy is designed for declarative programming...",
                "answer": "DSPy is a framework for programming language models declaratively",
            }

            attrs = instrumentor._extract_cot_response_attributes(mock_result)

            # Assert
            self.assertIn("break down", attrs["dspy.cot.reasoning"])
            self.assertIn("declaratively", attrs["dspy.cot.output.answer"])

    def test_extract_react_response_attributes(self):
        """Test extraction of ReAct response attributes."""
        mock_dspy = MagicMock()

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            instrumentor = DSPyInstrumentor()

            # Create mock result with trajectory
            mock_result = MagicMock()
            mock_result._store = {
                "trajectory": [
                    {"action": "search", "observation": "Found 10 results"},
                    {"action": "calculate", "observation": "Average is 72.5"},
                ],
                "answer": "The average temperature is 72.5 degrees",
            }

            attrs = instrumentor._extract_react_response_attributes(mock_result)

            # Assert
            self.assertTrue(attrs["dspy.react.has_trajectory"])
            self.assertIn("average temperature", attrs["dspy.react.output.answer"])

    def test_extract_optimizer_response_attributes(self):
        """Test extraction of optimizer response attributes."""
        mock_dspy = MagicMock()

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            instrumentor = DSPyInstrumentor()

            # Create mock compiled program
            mock_program = MagicMock()
            mock_program.__class__.__name__ = "CompiledQA"
            mock_program.demos = [MagicMock() for _ in range(5)]

            attrs = instrumentor._extract_optimizer_response_attributes(mock_program)

            # Assert
            self.assertEqual(attrs["dspy.optimizer.result_type"], "CompiledQA")
            self.assertEqual(attrs["dspy.optimizer.demos_count"], 5)

    def test_extract_usage_returns_none(self):
        """Test that _extract_usage returns None."""
        mock_dspy = MagicMock()

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            instrumentor = DSPyInstrumentor()

            # DSPy usage tracked by underlying LM providers
            result = MagicMock()

            usage = instrumentor._extract_usage(result)

            self.assertIsNone(usage)

    def test_extract_finish_reason_with_finish_reason_in_store(self):
        """Test extraction of finish reason from result store."""
        mock_dspy = MagicMock()

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            instrumentor = DSPyInstrumentor()

            # Create mock result with finish reason
            mock_result = MagicMock()
            mock_result._store = {"finish_reason": "stop", "answer": "Response"}

            finish_reason = instrumentor._extract_finish_reason(mock_result)

            # Assert
            self.assertEqual(finish_reason, "stop")

    def test_extract_finish_reason_completed(self):
        """Test extraction of finish reason defaults to completed."""
        mock_dspy = MagicMock()

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            instrumentor = DSPyInstrumentor()

            # Create mock result with store but no finish_reason
            mock_result = MagicMock()
            mock_result._store = {"answer": "Response"}

            finish_reason = instrumentor._extract_finish_reason(mock_result)

            # Assert
            self.assertEqual(finish_reason, "completed")

    def test_extract_finish_reason_none(self):
        """Test extraction of finish reason returns None when not available."""
        mock_dspy = MagicMock()

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            instrumentor = DSPyInstrumentor()

            # Create mock result without _store
            mock_result = MagicMock(spec=[])

            finish_reason = instrumentor._extract_finish_reason(mock_result)

            # Assert
            self.assertIsNone(finish_reason)

    def test_wrap_module_call_creates_span(self):
        """Test that _wrap_module_call creates appropriate span."""
        mock_dspy = MagicMock()

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            instrumentor = DSPyInstrumentor()
            instrumentor.config = MagicMock()

            # Create mock module
            mock_module = MagicMock()
            mock_module.__class__.__name__ = "QAModule"

            # Mock the create_span_wrapper
            mock_wrapper = MagicMock(return_value=lambda f: f)
            instrumentor.create_span_wrapper = mock_wrapper

            # Mock wrapped function
            mock_wrapped = MagicMock(return_value="result")

            # Call wrap
            result = instrumentor._wrap_module_call(mock_wrapped, mock_module, (), {})

            # Assert span wrapper was called
            mock_wrapper.assert_called_once()
            call_kwargs = mock_wrapper.call_args[1]
            self.assertEqual(call_kwargs["span_name"], "dspy.module.qamodule")

    def test_wrap_predict_forward_creates_span(self):
        """Test that _wrap_predict_forward creates appropriate span."""
        mock_dspy = MagicMock()

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            instrumentor = DSPyInstrumentor()
            instrumentor.config = MagicMock()

            # Create mock Predict instance
            mock_predict = MagicMock()

            # Mock the create_span_wrapper
            mock_wrapper = MagicMock(return_value=lambda f: f)
            instrumentor.create_span_wrapper = mock_wrapper

            # Mock wrapped function
            mock_wrapped = MagicMock(return_value="result")

            # Call wrap
            result = instrumentor._wrap_predict_forward(mock_wrapped, mock_predict, (), {})

            # Assert span wrapper was called
            mock_wrapper.assert_called_once()
            call_kwargs = mock_wrapper.call_args[1]
            self.assertEqual(call_kwargs["span_name"], "dspy.predict")

    def test_wrap_chain_of_thought_forward_creates_span(self):
        """Test that _wrap_chain_of_thought_forward creates appropriate span."""
        mock_dspy = MagicMock()

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            instrumentor = DSPyInstrumentor()
            instrumentor.config = MagicMock()

            # Create mock ChainOfThought instance
            mock_cot = MagicMock()

            # Mock the create_span_wrapper
            mock_wrapper = MagicMock(return_value=lambda f: f)
            instrumentor.create_span_wrapper = mock_wrapper

            # Mock wrapped function
            mock_wrapped = MagicMock(return_value="result")

            # Call wrap
            result = instrumentor._wrap_chain_of_thought_forward(mock_wrapped, mock_cot, (), {})

            # Assert span wrapper was called
            mock_wrapper.assert_called_once()
            call_kwargs = mock_wrapper.call_args[1]
            self.assertEqual(call_kwargs["span_name"], "dspy.chain_of_thought")

    def test_wrap_react_forward_creates_span(self):
        """Test that _wrap_react_forward creates appropriate span."""
        mock_dspy = MagicMock()

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            instrumentor = DSPyInstrumentor()
            instrumentor.config = MagicMock()

            # Create mock ReAct instance
            mock_react = MagicMock()

            # Mock the create_span_wrapper
            mock_wrapper = MagicMock(return_value=lambda f: f)
            instrumentor.create_span_wrapper = mock_wrapper

            # Mock wrapped function
            mock_wrapped = MagicMock(return_value="result")

            # Call wrap
            result = instrumentor._wrap_react_forward(mock_wrapped, mock_react, (), {})

            # Assert span wrapper was called
            mock_wrapper.assert_called_once()
            call_kwargs = mock_wrapper.call_args[1]
            self.assertEqual(call_kwargs["span_name"], "dspy.react")

    def test_wrap_optimizer_compile_creates_span(self):
        """Test that _wrap_optimizer_compile creates appropriate span."""
        mock_dspy = MagicMock()

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            instrumentor = DSPyInstrumentor()
            instrumentor.config = MagicMock()

            # Create mock optimizer
            mock_optimizer = MagicMock()
            mock_optimizer.__class__.__name__ = "COPRO"

            # Mock the create_span_wrapper
            mock_wrapper = MagicMock(return_value=lambda f: f)
            instrumentor.create_span_wrapper = mock_wrapper

            # Mock wrapped function
            mock_wrapped = MagicMock(return_value="result")

            # Call wrap
            result = instrumentor._wrap_optimizer_compile(mock_wrapped, mock_optimizer, (), {})

            # Assert span wrapper was called
            mock_wrapper.assert_called_once()
            call_kwargs = mock_wrapper.call_args[1]
            self.assertEqual(call_kwargs["span_name"], "dspy.optimizer.copro")


if __name__ == "__main__":
    unittest.main()
