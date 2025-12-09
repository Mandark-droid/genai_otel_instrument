import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.crewai_instrumentor import CrewAIInstrumentor


class TestCrewAIInstrumentor(unittest.TestCase):
    """Tests for CrewAIInstrumentor"""

    @patch("genai_otel.instrumentors.crewai_instrumentor.logger")
    def test_init_with_crewai_available(self, mock_logger):
        """Test that __init__ detects CrewAI availability."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            self.assertTrue(instrumentor._crewai_available)
            mock_logger.debug.assert_called_with(
                "CrewAI library detected and available for instrumentation"
            )

    @patch("genai_otel.instrumentors.crewai_instrumentor.logger")
    def test_init_with_crewai_not_available(self, mock_logger):
        """Test that __init__ handles missing CrewAI gracefully."""
        with patch.dict("sys.modules", {"crewai": None}):
            instrumentor = CrewAIInstrumentor()

            self.assertFalse(instrumentor._crewai_available)
            mock_logger.debug.assert_called_with(
                "CrewAI library not installed, instrumentation will be skipped"
            )

    @patch("genai_otel.instrumentors.crewai_instrumentor.logger")
    def test_instrument_when_crewai_not_available(self, mock_logger):
        """Test that instrument skips when CrewAI is not available."""
        with patch.dict("sys.modules", {"crewai": None}):
            instrumentor = CrewAIInstrumentor()
            config = MagicMock()

            instrumentor.instrument(config)

            mock_logger.debug.assert_any_call(
                "Skipping CrewAI instrumentation - library not available"
            )

    @patch("genai_otel.instrumentors.crewai_instrumentor.logger")
    def test_instrument_with_crewai_available(self, mock_logger):
        """Test that instrument wraps Crew.kickoff, Task, and Agent methods when available."""

        # Create a real Crew class
        class MockCrew:
            def kickoff(self, inputs=None):
                return "crew_result"

        # Create mock Task and Agent classes
        class MockTask:
            def execute_sync(self):
                return "task_result"

            def execute_async(self):
                return "task_result"

        class MockAgent:
            def execute_task(self):
                return "agent_result"

        # Create mock crewai module
        mock_crewai = MagicMock()
        mock_crewai.Crew = MockCrew
        mock_crewai.Task = MockTask
        mock_crewai.Agent = MockAgent

        # Create a mock wrapt module
        mock_wrapt = MagicMock()

        with patch.dict("sys.modules", {"crewai": mock_crewai, "wrapt": mock_wrapt}):
            instrumentor = CrewAIInstrumentor()
            config = MagicMock()

            # Act
            instrumentor.instrument(config)

            # Assert
            self.assertEqual(instrumentor.config, config)
            self.assertTrue(instrumentor._instrumented)
            mock_logger.info.assert_called_with(
                "CrewAI instrumentation enabled with automatic context propagation"
            )
            # Verify FunctionWrapper was called to wrap all methods (kickoff, execute_sync, execute_async, execute_task)
            self.assertEqual(mock_wrapt.FunctionWrapper.call_count, 4)

    @patch("genai_otel.instrumentors.crewai_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_false(self, mock_logger):
        """Test that instrument handles exceptions gracefully when fail_on_error is False."""
        # Create mock crewai module
        mock_crewai = MagicMock()

        # Make hasattr fail to trigger exception
        def mock_hasattr_side_effect(obj, name):
            if name == "Crew":
                raise RuntimeError("Test error")
            return True

        with patch.dict("sys.modules", {"crewai": mock_crewai, "wrapt": MagicMock()}):
            with patch("builtins.hasattr", side_effect=mock_hasattr_side_effect):
                instrumentor = CrewAIInstrumentor()
                config = MagicMock()
                config.fail_on_error = False

                # Should not raise exception
                instrumentor.instrument(config)

                mock_logger.error.assert_called_once()

    @patch("genai_otel.instrumentors.crewai_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_true(self, mock_logger):
        """Test that instrument raises exceptions when fail_on_error is True."""
        # Create mock crewai module
        mock_crewai = MagicMock()

        # Make hasattr fail to trigger exception
        def mock_hasattr_side_effect(obj, name):
            if name == "Crew":
                raise RuntimeError("Test error")
            return True

        with patch.dict("sys.modules", {"crewai": mock_crewai, "wrapt": MagicMock()}):
            with patch("builtins.hasattr", side_effect=mock_hasattr_side_effect):
                instrumentor = CrewAIInstrumentor()
                config = MagicMock()
                config.fail_on_error = True

                # Should raise exception
                with self.assertRaises(RuntimeError):
                    instrumentor.instrument(config)

    def test_extract_crew_attributes_basic(self):
        """Test extraction of basic crew attributes."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            # Create mock crew
            mock_crew = MagicMock()
            mock_crew.id = "crew_123"
            mock_crew.name = "Research Crew"
            mock_crew.process = "sequential"
            mock_crew.verbose = True

            args = ()
            kwargs = {}

            attrs = instrumentor._extract_crew_attributes(mock_crew, args, kwargs)

            # Assert
            self.assertEqual(attrs["gen_ai.system"], "crewai")
            self.assertEqual(attrs["gen_ai.operation.name"], "crew.execution")
            self.assertEqual(attrs["crewai.crew.id"], "crew_123")
            self.assertEqual(attrs["crewai.crew.name"], "Research Crew")
            self.assertEqual(attrs["crewai.process.type"], "sequential")
            self.assertTrue(attrs["crewai.verbose"])

    def test_extract_crew_attributes_with_agents(self):
        """Test extraction of crew attributes with agents."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            # Create mock agents
            mock_agent1 = MagicMock()
            mock_agent1.role = "Researcher"
            mock_agent1.goal = "Research AI trends"

            # Create mock tools with proper name attributes
            search_tool = MagicMock()
            search_tool.name = "search_tool"
            scrape_tool = MagicMock()
            scrape_tool.name = "scrape_tool"
            mock_agent1.tools = [search_tool, scrape_tool]

            mock_agent2 = MagicMock()
            mock_agent2.role = "Writer"
            mock_agent2.goal = "Write articles"
            mock_agent2.tools = []

            # Create mock crew
            mock_crew = MagicMock()
            mock_crew.agents = [mock_agent1, mock_agent2]

            args = ()
            kwargs = {}

            attrs = instrumentor._extract_crew_attributes(mock_crew, args, kwargs)

            # Assert
            self.assertEqual(attrs["crewai.agent_count"], 2)
            self.assertIn("Researcher", attrs["crewai.agent.roles"])
            self.assertIn("Writer", attrs["crewai.agent.roles"])
            self.assertIn("Research AI trends", attrs["crewai.agent.goals"])
            self.assertEqual(attrs["crewai.tool_count"], 2)
            self.assertIn("search_tool", attrs["crewai.tools"])
            self.assertIn("scrape_tool", attrs["crewai.tools"])

    def test_extract_crew_attributes_with_tasks(self):
        """Test extraction of crew attributes with tasks."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            # Create mock tasks
            mock_task1 = MagicMock()
            mock_task1.description = "Research latest AI developments"

            mock_task2 = MagicMock()
            mock_task2.description = "Write a summary report"

            # Create mock crew
            mock_crew = MagicMock()
            mock_crew.tasks = [mock_task1, mock_task2]

            args = ()
            kwargs = {}

            attrs = instrumentor._extract_crew_attributes(mock_crew, args, kwargs)

            # Assert
            self.assertEqual(attrs["crewai.task_count"], 2)
            self.assertIn("Research latest AI developments", attrs["crewai.task.descriptions"])
            self.assertIn("Write a summary report", attrs["crewai.task.descriptions"])

    def test_extract_crew_attributes_with_inputs(self):
        """Test extraction of crew attributes with inputs."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            # Create mock crew
            mock_crew = MagicMock()

            # Test with dict inputs
            inputs = {"topic": "AI Agents", "format": "markdown"}
            args = (inputs,)
            kwargs = {}

            attrs = instrumentor._extract_crew_attributes(mock_crew, args, kwargs)

            # Assert
            self.assertEqual(attrs["crewai.inputs.keys"], ["topic", "format"])
            self.assertEqual(attrs["crewai.inputs.topic"], "AI Agents")
            self.assertEqual(attrs["crewai.inputs.format"], "markdown")

    def test_extract_crew_attributes_with_manager(self):
        """Test extraction of crew attributes with hierarchical manager."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            # Create mock manager agent
            mock_manager = MagicMock()
            mock_manager.role = "Project Manager"

            # Create mock crew
            mock_crew = MagicMock()
            mock_crew.manager_agent = mock_manager

            args = ()
            kwargs = {}

            attrs = instrumentor._extract_crew_attributes(mock_crew, args, kwargs)

            # Assert
            self.assertEqual(attrs["crewai.manager.role"], "Project Manager")

    def test_extract_response_attributes_string_result(self):
        """Test extraction of response attributes from string result."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            # Test with string result
            result = "This is the crew output"

            attrs = instrumentor._extract_response_attributes(result)

            # Assert
            self.assertEqual(attrs["crewai.output"], "This is the crew output")
            self.assertEqual(attrs["crewai.output_length"], 23)

    def test_extract_response_attributes_crew_output(self):
        """Test extraction of response attributes from CrewOutput object."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            # Create mock CrewOutput
            mock_result = MagicMock()
            mock_result.raw = "This is the raw output"

            # Create mock task outputs
            mock_task_output1 = MagicMock()
            mock_task_output1.raw = "Task 1 result"

            mock_task_output2 = MagicMock()
            mock_task_output2.raw = "Task 2 result"

            mock_result.tasks_output = [mock_task_output1, mock_task_output2]

            attrs = instrumentor._extract_response_attributes(mock_result)

            # Assert
            self.assertIn("crewai.output", attrs)
            self.assertEqual(attrs["crewai.tasks_completed"], 2)
            self.assertIn("Task 1 result", attrs["crewai.task_outputs"])
            self.assertIn("Task 2 result", attrs["crewai.task_outputs"])

    def test_extract_usage_when_available(self):
        """Test extraction of usage information when available."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            # Create mock result with usage
            mock_result = MagicMock()
            mock_result.token_usage = MagicMock()
            mock_result.token_usage.prompt_tokens = 100
            mock_result.token_usage.completion_tokens = 50
            mock_result.token_usage.total_tokens = 150

            usage = instrumentor._extract_usage(mock_result)

            # Assert
            self.assertIsNotNone(usage)
            self.assertEqual(usage["prompt_tokens"], 100)
            self.assertEqual(usage["completion_tokens"], 50)
            self.assertEqual(usage["total_tokens"], 150)

    def test_extract_usage_when_not_available(self):
        """Test extraction of usage information when not available."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            # Create mock result without usage
            mock_result = MagicMock()
            del mock_result.token_usage

            usage = instrumentor._extract_usage(mock_result)

            # Assert
            self.assertIsNone(usage)

    def test_extract_finish_reason(self):
        """Test extraction of finish reason from result."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            # Create mock result
            mock_result = "Some output"

            finish_reason = instrumentor._extract_finish_reason(mock_result)

            # Assert
            self.assertEqual(finish_reason, "completed")


if __name__ == "__main__":
    unittest.main()
