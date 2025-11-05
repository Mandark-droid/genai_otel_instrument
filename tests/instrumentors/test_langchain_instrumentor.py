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
            mock_logger.debug.assert_called_with(
                "langchain library detected and available for instrumentation"
            )

    @patch("genai_otel.instrumentors.langchain_instrumentor.logger")
    def test_init_with_langchain_not_available(self, mock_logger):
        """Test that __init__ handles missing langchain gracefully."""
        with patch.dict("sys.modules", {"langchain": None}):
            instrumentor = LangChainInstrumentor()

            self.assertFalse(instrumentor._langchain_available)
            mock_logger.debug.assert_called_with(
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
        """Test that _extract_usage returns None."""
        instrumentor = LangChainInstrumentor()

        result = instrumentor._extract_usage(MagicMock())

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
