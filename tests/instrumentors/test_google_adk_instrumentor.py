import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.google_adk_instrumentor import GoogleADKInstrumentor


class TestGoogleADKInstrumentor(unittest.TestCase):
    """Tests for GoogleADKInstrumentor"""

    @patch("genai_otel.instrumentors.google_adk_instrumentor.logger")
    def test_init_with_adk_available(self, mock_logger):
        """Test that __init__ detects Google ADK availability."""
        mock_google_adk = MagicMock()
        with patch.dict("sys.modules", {"google": MagicMock(), "google.adk": mock_google_adk}):
            instrumentor = GoogleADKInstrumentor()

            self.assertTrue(instrumentor._adk_available)
            mock_logger.debug.assert_called_with(
                "Google ADK library detected and available for instrumentation"
            )

    @patch("genai_otel.instrumentors.google_adk_instrumentor.logger")
    def test_init_with_adk_not_available(self, mock_logger):
        """Test that __init__ handles missing Google ADK gracefully."""
        with patch.dict("sys.modules", {"google.adk": None}):
            instrumentor = GoogleADKInstrumentor()

            self.assertFalse(instrumentor._adk_available)
            mock_logger.debug.assert_called_with(
                "Google ADK library not installed, instrumentation will be skipped"
            )

    @patch("genai_otel.instrumentors.google_adk_instrumentor.logger")
    def test_instrument_when_adk_not_available(self, mock_logger):
        """Test that instrument skips when Google ADK is not available."""
        with patch.dict("sys.modules", {"google.adk": None}):
            instrumentor = GoogleADKInstrumentor()
            config = MagicMock()

            instrumentor.instrument(config)

            mock_logger.debug.assert_any_call(
                "Skipping Google ADK instrumentation - library not available"
            )

    @patch("genai_otel.instrumentors.google_adk_instrumentor.logger")
    def test_instrument_with_adk_available(self, mock_logger):
        """Test that instrument wraps Runner and InMemoryRunner methods."""

        class MockRunner:
            def run_async(self, **kwargs):
                pass

        class MockInMemoryRunner:
            def run_debug(self, message):
                pass

        mock_runners = MagicMock()
        mock_runners.Runner = MockRunner
        mock_runners.InMemoryRunner = MockInMemoryRunner

        mock_wrapt = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.adk": MagicMock(),
                "google.adk.runners": mock_runners,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = GoogleADKInstrumentor()
            config = MagicMock()

            instrumentor.instrument(config)

            self.assertTrue(instrumentor._instrumented)
            mock_logger.info.assert_called_with("Google ADK instrumentation enabled")
            # Runner.run_async + InMemoryRunner.run_debug = 2
            self.assertEqual(mock_wrapt.FunctionWrapper.call_count, 2)

    def test_extract_runner_attributes_basic(self):
        """Test extraction of basic runner attributes."""
        mock_google_adk = MagicMock()
        with patch.dict("sys.modules", {"google": MagicMock(), "google.adk": mock_google_adk}):
            instrumentor = GoogleADKInstrumentor()

            # Create mock runner
            mock_runner = MagicMock()
            mock_runner.app_name = "my_app"
            mock_runner.agent.name = "my_agent"
            mock_runner.agent.model = "gemini-2.5-flash"
            mock_runner.agent.description = "A test agent"
            mock_runner.agent.sub_agents = []
            mock_runner.agent.tools = []

            kwargs = {"user_id": "user_123", "session_id": "session_456"}

            attrs = instrumentor._extract_runner_attributes(mock_runner, (), kwargs)

            self.assertEqual(attrs["gen_ai.system"], "google_adk")
            self.assertEqual(attrs["gen_ai.operation.name"], "runner.run")
            self.assertEqual(attrs["google_adk.app_name"], "my_app")
            self.assertEqual(attrs["google_adk.agent.name"], "my_agent")
            self.assertEqual(attrs["gen_ai.request.model"], "gemini-2.5-flash")
            self.assertEqual(attrs["google_adk.user_id"], "user_123")
            self.assertEqual(attrs["google_adk.session_id"], "session_456")

    def test_extract_runner_attributes_with_tools_and_sub_agents(self):
        """Test extraction of runner attributes with tools and sub-agents."""
        mock_google_adk = MagicMock()
        with patch.dict("sys.modules", {"google": MagicMock(), "google.adk": mock_google_adk}):
            instrumentor = GoogleADKInstrumentor()

            # Create mock tools
            mock_tool1 = MagicMock()
            mock_tool1.name = "search_tool"
            mock_tool2 = MagicMock()
            mock_tool2.name = "calc_tool"

            # Create mock sub-agents
            mock_sub1 = MagicMock()
            mock_sub1.name = "helper_agent"

            # Create mock runner
            mock_runner = MagicMock()
            mock_runner.app_name = "my_app"
            mock_runner.agent.name = "coordinator"
            mock_runner.agent.model = "gemini-2.5-pro"
            mock_runner.agent.sub_agents = [mock_sub1]
            mock_runner.agent.tools = [mock_tool1, mock_tool2]

            attrs = instrumentor._extract_runner_attributes(mock_runner, (), {})

            self.assertEqual(attrs["google_adk.sub_agent_count"], 1)
            self.assertIn("helper_agent", attrs["google_adk.sub_agent_names"])
            self.assertEqual(attrs["google_adk.tool_count"], 2)
            self.assertIn("search_tool", attrs["google_adk.tools"])
            self.assertIn("calc_tool", attrs["google_adk.tools"])

    def test_extract_runner_debug_attributes(self):
        """Test extraction of debug runner attributes."""
        mock_google_adk = MagicMock()
        with patch.dict("sys.modules", {"google": MagicMock(), "google.adk": mock_google_adk}):
            instrumentor = GoogleADKInstrumentor()

            mock_runner = MagicMock()
            mock_runner.app_name = "test_app"
            mock_runner.agent.name = "test_agent"
            mock_runner.agent.model = "gemini-2.5-flash"

            attrs = instrumentor._extract_runner_debug_attributes(
                mock_runner, ("Hello there!",), {}
            )

            self.assertEqual(attrs["gen_ai.system"], "google_adk")
            self.assertEqual(attrs["gen_ai.operation.name"], "runner.run_debug")
            self.assertEqual(attrs["google_adk.app_name"], "test_app")
            self.assertEqual(attrs["google_adk.input_message"], "Hello there!")

    def test_extract_usage_returns_none(self):
        """Test that _extract_usage returns None (ADK relies on provider instrumentors)."""
        mock_google_adk = MagicMock()
        with patch.dict("sys.modules", {"google": MagicMock(), "google.adk": mock_google_adk}):
            instrumentor = GoogleADKInstrumentor()
            self.assertIsNone(instrumentor._extract_usage(MagicMock()))

    def test_extract_finish_reason(self):
        """Test extraction of finish reason."""
        mock_google_adk = MagicMock()
        with patch.dict("sys.modules", {"google": MagicMock(), "google.adk": mock_google_adk}):
            instrumentor = GoogleADKInstrumentor()
            self.assertEqual(instrumentor._extract_finish_reason("result"), "completed")
            self.assertIsNone(instrumentor._extract_finish_reason(None))

    @patch("genai_otel.instrumentors.google_adk_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_false(self, mock_logger):
        """Test that instrument handles exceptions gracefully when fail_on_error is False."""
        mock_google_adk = MagicMock()
        mock_runners = MagicMock()

        # Make wrapt.FunctionWrapper raise to trigger the except block
        mock_wrapt = MagicMock()
        mock_wrapt.FunctionWrapper.side_effect = RuntimeError("Test error")

        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.adk": mock_google_adk,
                "google.adk.runners": mock_runners,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = GoogleADKInstrumentor()
            config = MagicMock()
            config.fail_on_error = False

            instrumentor.instrument(config)

            mock_logger.error.assert_called_once()

    @patch("genai_otel.instrumentors.google_adk_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_true(self, mock_logger):
        """Test that instrument raises exceptions when fail_on_error is True."""
        mock_google_adk = MagicMock()
        mock_runners = MagicMock()

        mock_wrapt = MagicMock()
        mock_wrapt.FunctionWrapper.side_effect = RuntimeError("Test error")

        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.adk": mock_google_adk,
                "google.adk.runners": mock_runners,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = GoogleADKInstrumentor()
            config = MagicMock()
            config.fail_on_error = True

            with self.assertRaises(RuntimeError):
                instrumentor.instrument(config)


if __name__ == "__main__":
    unittest.main()
