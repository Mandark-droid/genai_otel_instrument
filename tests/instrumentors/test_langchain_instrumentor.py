import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.langchain_instrumentor import LangChainInstrumentor


class TestLangChainInstrumentor(unittest.TestCase):
    """Tests for LangChainInstrumentor"""

    @patch("genai_otel.instrumentors.langchain_instrumentor.logger")
    def test_init_with_langchain_available(self, mock_logger):
        """Test that __init__ detects langchain availability."""
        with patch.dict("sys.modules", {"langchain": MagicMock()}):
            instrumentor = LangChainInstrumentor()

            self.assertTrue(instrumentor._langchain_available)
            mock_logger.debug.assert_any_call(
                "langchain library detected and available for instrumentation"
            )

    @patch("genai_otel.instrumentors.langchain_instrumentor.logger")
    def test_init_with_langchain_not_available(self, mock_logger):
        """Test that __init__ handles missing langchain gracefully."""
        with patch.dict("sys.modules", {"langchain": None}):
            instrumentor = LangChainInstrumentor()

            self.assertFalse(instrumentor._langchain_available)
            mock_logger.debug.assert_any_call(
                "langchain library not installed, instrumentation will be skipped"
            )

    @patch("genai_otel.instrumentors.langchain_instrumentor.logger")
    def test_instrument_with_langchain_not_available(self, mock_logger):
        """Test that instrument skips when langchain is not available."""
        with patch.dict("sys.modules", {"langchain": None}):
            instrumentor = LangChainInstrumentor()
            config = OTelConfig()

            instrumentor.instrument(config)

            mock_logger.debug.assert_any_call("Skipping instrumentation - library not available")

    def test_instrument_with_langchain_available(self):
        """Test that instrument wraps langchain components when available."""

        # Create mock Chain and AgentExecutor classes
        class MockChain:
            def __call__(self, *args, **kwargs):
                return "chain_result"

        class MockAgentExecutor:
            def __call__(self, *args, **kwargs):
                return "agent_result"

        # Create mock langchain modules
        mock_langchain = MagicMock()
        mock_chains_module = MagicMock()
        mock_agents_module = MagicMock()

        mock_chains_module.base.Chain = MockChain
        mock_agents_module.agent.AgentExecutor = MockAgentExecutor

        with patch.dict(
            "sys.modules",
            {
                "langchain": mock_langchain,
                "langchain.chains": mock_chains_module,
                "langchain.chains.base": mock_chains_module.base,
                "langchain.agents": mock_agents_module,
                "langchain.agents.agent": mock_agents_module.agent,
            },
        ):
            instrumentor = LangChainInstrumentor()
            config = OTelConfig()

            # Store original methods
            original_chain_call = MockChain.__call__
            original_agent_call = MockAgentExecutor.__call__

            # Call instrument
            instrumentor.instrument(config)

            # Verify that methods were replaced
            self.assertNotEqual(MockChain.__call__, original_chain_call)
            self.assertNotEqual(MockAgentExecutor.__call__, original_agent_call)
            self.assertEqual(instrumentor.config, config)

    def test_instrument_with_import_error(self):
        """Test that instrument handles ImportError gracefully."""
        # Create mock langchain that raises ImportError on submodule import
        mock_langchain = MagicMock()

        def raise_import_error(name, *args, **kwargs):
            if "langchain.chains" in name or "langchain.agents" in name:
                raise ImportError(f"No module named '{name}'")
            return MagicMock()

        with patch.dict("sys.modules", {"langchain": mock_langchain}):
            with patch("builtins.__import__", side_effect=raise_import_error):
                instrumentor = LangChainInstrumentor()
                config = OTelConfig()

                # Should not raise
                instrumentor.instrument(config)

    def test_wrapped_chain_call(self):
        """Test that wrapped Chain.__call__ creates spans correctly."""

        # Create mock Chain class
        class MockChain:
            def __call__(self, *args, **kwargs):
                return {"result": "chain_output"}

        # Create mock langchain modules
        mock_langchain = MagicMock()
        mock_chains_module = MagicMock()
        mock_agents_module = MagicMock()

        mock_chains_module.base.Chain = MockChain
        mock_agents_module.agent.AgentExecutor = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "langchain": mock_langchain,
                "langchain.chains": mock_chains_module,
                "langchain.chains.base": mock_chains_module.base,
                "langchain.agents": mock_agents_module,
                "langchain.agents.agent": mock_agents_module.agent,
            },
        ):
            instrumentor = LangChainInstrumentor()
            instrumentor.tracer = MagicMock()
            config = OTelConfig()

            # Mock span
            mock_span = MagicMock()
            instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = (
                mock_span
            )

            # Instrument
            instrumentor.instrument(config)

            # Create chain instance and call it
            chain = MockChain()
            result = chain("input")

            # Verify span was created
            instrumentor.tracer.start_as_current_span.assert_called_once_with(
                "langchain.chain.MockChain"
            )
            mock_span.set_attribute.assert_called_once_with("langchain.chain.type", "MockChain")
            self.assertEqual(result, {"result": "chain_output"})

    def test_wrapped_agent_call(self):
        """Test that wrapped AgentExecutor.__call__ creates spans correctly."""

        # Create mock AgentExecutor class
        class MockAgentExecutor:
            def __init__(self):
                self.agent = {"name": "test_agent"}

            def __call__(self, *args, **kwargs):
                return {"result": "agent_output"}

        # Create mock langchain modules
        mock_langchain = MagicMock()
        mock_chains_module = MagicMock()
        mock_agents_module = MagicMock()

        mock_chains_module.base.Chain = MagicMock()
        mock_agents_module.agent.AgentExecutor = MockAgentExecutor

        with patch.dict(
            "sys.modules",
            {
                "langchain": mock_langchain,
                "langchain.chains": mock_chains_module,
                "langchain.chains.base": mock_chains_module.base,
                "langchain.agents": mock_agents_module,
                "langchain.agents.agent": mock_agents_module.agent,
            },
        ):
            instrumentor = LangChainInstrumentor()
            instrumentor.tracer = MagicMock()
            config = OTelConfig()

            # Mock span
            mock_span = MagicMock()
            instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = (
                mock_span
            )

            # Instrument
            instrumentor.instrument(config)

            # Create agent instance and call it
            agent = MockAgentExecutor()
            result = agent("input")

            # Verify span was created
            instrumentor.tracer.start_as_current_span.assert_called_once_with(
                "langchain.agent.execute"
            )
            mock_span.set_attribute.assert_called_once_with("langchain.agent.name", "test_agent")
            self.assertEqual(result, {"result": "agent_output"})

    def test_wrapped_agent_call_with_unknown_agent(self):
        """Test that wrapped AgentExecutor.__call__ handles missing agent name."""

        # Create mock AgentExecutor class without agent attribute
        class MockAgentExecutor:
            def __call__(self, *args, **kwargs):
                return {"result": "agent_output"}

        # Create mock langchain modules
        mock_langchain = MagicMock()
        mock_chains_module = MagicMock()
        mock_agents_module = MagicMock()

        mock_chains_module.base.Chain = MagicMock()
        mock_agents_module.agent.AgentExecutor = MockAgentExecutor

        with patch.dict(
            "sys.modules",
            {
                "langchain": mock_langchain,
                "langchain.chains": mock_chains_module,
                "langchain.chains.base": mock_chains_module.base,
                "langchain.agents": mock_agents_module,
                "langchain.agents.agent": mock_agents_module.agent,
            },
        ):
            instrumentor = LangChainInstrumentor()
            instrumentor.tracer = MagicMock()
            config = OTelConfig()

            # Mock span
            mock_span = MagicMock()
            instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = (
                mock_span
            )

            # Instrument
            instrumentor.instrument(config)

            # Create agent instance and call it
            agent = MockAgentExecutor()
            result = agent("input")

            # Verify span was created with "unknown" agent name
            mock_span.set_attribute.assert_called_once_with("langchain.agent.name", "unknown")

    def test_extract_usage(self):
        """Test that _extract_usage returns None for objects without usage."""
        instrumentor = LangChainInstrumentor()

        # Create a mock without usage metadata attributes
        mock_result = MagicMock(spec=[])

        result = instrumentor._extract_usage(mock_result)

        self.assertIsNone(result)

    def test_extract_usage_with_usage_metadata(self):
        """Test that _extract_usage extracts from usage_metadata."""
        instrumentor = LangChainInstrumentor()

        # Mock result with usage_metadata
        mock_result = MagicMock()
        mock_result.usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 20,
        }

        result = instrumentor._extract_usage(mock_result)

        self.assertIsNotNone(result)
        self.assertEqual(result["prompt_tokens"], 10)
        self.assertEqual(result["completion_tokens"], 20)
        self.assertEqual(result["total_tokens"], 30)

    def test_extract_usage_with_response_metadata(self):
        """Test that _extract_usage extracts from response_metadata."""
        instrumentor = LangChainInstrumentor()

        # Mock result with response_metadata
        mock_result = MagicMock()
        mock_result.usage_metadata = None
        mock_result.response_metadata = {
            "token_usage": {
                "prompt_tokens": 15,
                "completion_tokens": 25,
            }
        }

        result = instrumentor._extract_usage(mock_result)

        self.assertIsNotNone(result)
        self.assertEqual(result["prompt_tokens"], 15)
        self.assertEqual(result["completion_tokens"], 25)
        self.assertEqual(result["total_tokens"], 40)

    @patch("genai_otel.instrumentors.langchain_instrumentor.logger")
    def test_init_with_langchain_core_available(self, mock_logger):
        """Test that __init__ detects langchain_core availability."""
        with patch.dict("sys.modules", {"langchain": MagicMock(), "langchain_core": MagicMock()}):
            instrumentor = LangChainInstrumentor()

            self.assertTrue(instrumentor._langchain_available)
            self.assertTrue(instrumentor._langchain_core_available)

    def test_instrument_chat_models_invoke(self):
        """Test that chat model invoke() method is instrumented."""

        # Create mock BaseChatModel class
        class MockBaseChatModel:
            model_name = "gpt-4"

            def invoke(self, *args, **kwargs):
                # Mock response with usage_metadata
                response = MagicMock()
                response.usage_metadata = {
                    "input_tokens": 10,
                    "output_tokens": 20,
                }
                return response

        # Create mock langchain modules
        mock_langchain = MagicMock()
        mock_langchain_core = MagicMock()
        mock_chat_models = MagicMock()
        mock_chains_module = MagicMock()
        mock_agents_module = MagicMock()

        mock_chat_models.BaseChatModel = MockBaseChatModel
        mock_chains_module.base.Chain = MagicMock()
        mock_agents_module.agent.AgentExecutor = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "langchain": mock_langchain,
                "langchain_core": mock_langchain_core,
                "langchain_core.language_models": mock_chat_models,
                "langchain_core.language_models.chat_models": mock_chat_models,
                "langchain.chains": mock_chains_module,
                "langchain.chains.base": mock_chains_module.base,
                "langchain.agents": mock_agents_module,
                "langchain.agents.agent": mock_agents_module.agent,
            },
        ):
            instrumentor = LangChainInstrumentor()
            instrumentor.tracer = MagicMock()
            config = OTelConfig()

            # Mock span
            mock_span = MagicMock()
            instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = (
                mock_span
            )

            # Instrument
            instrumentor.instrument(config)

            # Create chat model instance and call invoke
            chat_model = MockBaseChatModel()
            result = chat_model.invoke("test message")

            # Verify span was created
            instrumentor.tracer.start_as_current_span.assert_called_with(
                "langchain.chat_model.invoke"
            )

            # Verify attributes were set
            calls = mock_span.set_attribute.call_args_list
            attribute_dict = {call[0][0]: call[0][1] for call in calls}

            self.assertEqual(attribute_dict.get("langchain.chat_model.name"), "gpt-4")
            self.assertEqual(attribute_dict.get("langchain.chat_model.operation"), "invoke")
            self.assertEqual(attribute_dict.get("gen_ai.usage.prompt_tokens"), 10)
            self.assertEqual(attribute_dict.get("gen_ai.usage.completion_tokens"), 20)
            self.assertEqual(attribute_dict.get("gen_ai.usage.total_tokens"), 30)

    def test_instrument_chat_models_batch(self):
        """Test that chat model batch() method is instrumented."""

        # Create mock BaseChatModel class
        class MockBaseChatModel:
            model_name = "gpt-4"

            def invoke(self, *args, **kwargs):
                return MagicMock()

            async def ainvoke(self, *args, **kwargs):
                return MagicMock()

            def batch(self, *args, **kwargs):
                return ["response1", "response2", "response3"]

            async def abatch(self, *args, **kwargs):
                return ["response1", "response2", "response3"]

        # Create mock langchain modules
        mock_langchain = MagicMock()
        mock_langchain_core = MagicMock()
        mock_chat_models = MagicMock()
        mock_chains_module = MagicMock()
        mock_agents_module = MagicMock()

        mock_chat_models.BaseChatModel = MockBaseChatModel
        mock_chains_module.base.Chain = MagicMock()
        mock_agents_module.agent.AgentExecutor = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "langchain": mock_langchain,
                "langchain_core": mock_langchain_core,
                "langchain_core.language_models": mock_chat_models,
                "langchain_core.language_models.chat_models": mock_chat_models,
                "langchain.chains": mock_chains_module,
                "langchain.chains.base": mock_chains_module.base,
                "langchain.agents": mock_agents_module,
                "langchain.agents.agent": mock_agents_module.agent,
            },
        ):
            instrumentor = LangChainInstrumentor()
            instrumentor.tracer = MagicMock()
            config = OTelConfig()

            # Mock span
            mock_span = MagicMock()
            instrumentor.tracer.start_as_current_span.return_value.__enter__.return_value = (
                mock_span
            )

            # Instrument
            instrumentor.instrument(config)

            # Create chat model instance and call batch
            chat_model = MockBaseChatModel()
            messages = ["msg1", "msg2", "msg3"]
            result = chat_model.batch(messages)

            # Verify span was created
            instrumentor.tracer.start_as_current_span.assert_called_with(
                "langchain.chat_model.batch"
            )

            # Verify attributes were set
            calls = mock_span.set_attribute.call_args_list
            attribute_dict = {call[0][0]: call[0][1] for call in calls}

            self.assertEqual(attribute_dict.get("langchain.chat_model.name"), "gpt-4")
            self.assertEqual(attribute_dict.get("langchain.chat_model.operation"), "batch")
            self.assertEqual(attribute_dict.get("langchain.chat_model.batch_size"), 3)

    def test_get_model_name(self):
        """Test model name extraction from various attributes."""
        instrumentor = LangChainInstrumentor()

        # Test with model_name attribute
        mock_instance = MagicMock()
        mock_instance.model_name = "gpt-4"
        mock_instance.model = None
        mock_instance.model_id = None
        self.assertEqual(instrumentor._get_model_name(mock_instance), "gpt-4")

        # Test with model attribute
        mock_instance = MagicMock()
        mock_instance.model_name = None
        mock_instance.model = "claude-3"
        self.assertEqual(instrumentor._get_model_name(mock_instance), "claude-3")

        # Test with model_id attribute
        mock_instance = MagicMock()
        mock_instance.model_name = None
        mock_instance.model = None
        mock_instance.model_id = "gemini-pro"
        self.assertEqual(instrumentor._get_model_name(mock_instance), "gemini-pro")

        # Test fallback to class name
        class TestModel:
            pass

        instance = TestModel()
        self.assertEqual(instrumentor._get_model_name(instance), "TestModel")

    def test_extract_provider(self):
        """Test provider extraction from class name and module."""
        instrumentor = LangChainInstrumentor()

        # Test OpenAI provider
        class ChatOpenAI:
            pass

        ChatOpenAI.__module__ = "langchain_openai.chat_models"
        instance = ChatOpenAI()
        self.assertEqual(instrumentor._extract_provider(instance), "openai")

        # Test Anthropic provider
        class ChatAnthropic:
            pass

        ChatAnthropic.__module__ = "langchain_anthropic.chat_models"
        instance = ChatAnthropic()
        self.assertEqual(instrumentor._extract_provider(instance), "anthropic")

        # Test unknown provider
        class CustomModel:
            pass

        CustomModel.__module__ = "custom.module"
        instance = CustomModel()
        self.assertIsNone(instrumentor._extract_provider(instance))


if __name__ == "__main__":
    unittest.main(verbosity=2)
