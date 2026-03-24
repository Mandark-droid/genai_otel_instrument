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

        # Create a real Crew class with all kickoff variants
        class MockCrew:
            def kickoff(self, inputs=None):
                return "crew_result"

            def kickoff_async(self, inputs=None):
                return "crew_result"

            async def akickoff(self, inputs=None):
                return "crew_result"

            def kickoff_for_each(self, inputs=None):
                return "crew_result"

            def kickoff_for_each_async(self, inputs=None):
                return "crew_result"

            async def akickoff_for_each(self, inputs=None):
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

        # Save original methods to verify they get wrapped
        original_kickoff = MockCrew.kickoff
        original_execute_sync = MockTask.execute_sync
        original_execute_task = MockAgent.execute_task

        with patch.dict("sys.modules", {"crewai": mock_crewai}):
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
            # Verify methods were wrapped (no longer the original functions)
            self.assertIsNot(MockCrew.kickoff, original_kickoff)
            self.assertIsNot(MockTask.execute_sync, original_execute_sync)
            self.assertIsNot(MockAgent.execute_task, original_execute_task)

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

        with patch.dict("sys.modules", {"crewai": mock_crewai}):
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

        with patch.dict("sys.modules", {"crewai": mock_crewai}):
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
            # Session ID should be set from crew.id
            self.assertEqual(attrs["session.id"], "crew_123")
            self.assertEqual(attrs["crewai.session.id"], "crew_123")

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

            # Create mock result with usage (mimics CrewAI UsageMetrics pydantic model)
            mock_result = MagicMock()
            mock_result.token_usage = MagicMock()
            mock_result.token_usage.prompt_tokens = 100
            mock_result.token_usage.completion_tokens = 50
            mock_result.token_usage.total_tokens = 150
            mock_result.token_usage.cached_prompt_tokens = 0

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

    def test_extract_crew_attributes_session_id_from_crew_id(self):
        """Test that session.id is set from crew instance ID."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            mock_crew = MagicMock()
            mock_crew.id = "crew-uuid-123"

            attrs = instrumentor._extract_crew_attributes(mock_crew, (), {})

            self.assertEqual(attrs["session.id"], "crew-uuid-123")
            self.assertEqual(attrs["crewai.session.id"], "crew-uuid-123")
            self.assertEqual(mock_crew._genai_otel_session_id, "crew-uuid-123")

    def test_extract_crew_attributes_session_id_from_inputs(self):
        """Test that session_id from inputs takes priority over crew ID."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            mock_crew = MagicMock()
            mock_crew.id = "crew-uuid-123"

            inputs = {"topic": "AI", "session_id": "app-session-456"}
            attrs = instrumentor._extract_crew_attributes(mock_crew, (inputs,), {})

            # App-provided session_id takes priority over crew ID
            self.assertEqual(attrs["session.id"], "app-session-456")
            self.assertEqual(attrs["crewai.session.id"], "app-session-456")

    def test_extract_crew_attributes_session_id_from_inputs_kwarg(self):
        """Test that session_id from inputs kwarg works."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            mock_crew = MagicMock()
            mock_crew.id = "crew-uuid-123"

            inputs = {"session.id": "dot-session-789"}
            attrs = instrumentor._extract_crew_attributes(mock_crew, (), {"inputs": inputs})

            self.assertEqual(attrs["session.id"], "dot-session-789")

    def test_extract_crew_attributes_session_id_from_extractor(self):
        """Test that session_id_extractor takes priority over crew ID."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()
            instrumentor.config = MagicMock()
            instrumentor.config.session_id_extractor = lambda inst, a, k: "extractor-session"

            mock_crew = MagicMock()
            mock_crew.id = "crew-uuid-123"

            attrs = instrumentor._extract_crew_attributes(mock_crew, (), {})

            # Extractor takes priority over crew ID
            self.assertEqual(attrs["session.id"], "extractor-session")

    def test_extract_crew_attributes_session_id_autogenerated(self):
        """Test that a UUID is auto-generated when no other source exists."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            mock_crew = MagicMock(spec=[])  # No attributes at all

            attrs = instrumentor._extract_crew_attributes(mock_crew, (), {})

            # Should have auto-generated UUID
            self.assertIn("session.id", attrs)
            self.assertIn("crewai.session.id", attrs)
            # Verify it's a valid UUID format
            import uuid

            uuid.UUID(attrs["session.id"])  # Will raise if not valid

    def test_extract_task_attributes_session_propagation_via_crew(self):
        """Test that session.id is propagated from crew to task spans via task.crew."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            mock_crew = MagicMock()
            mock_crew._genai_otel_session_id = "session-from-crew"

            mock_task = MagicMock()
            mock_task.description = "Test task"
            mock_task.crew = mock_crew

            attrs = instrumentor._extract_task_attributes(mock_task, (), {})

            self.assertEqual(attrs["session.id"], "session-from-crew")

    def test_extract_task_attributes_session_propagation_via_agent(self):
        """Test that session.id is propagated via task.agent.crew when task.crew is absent."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            mock_crew = MagicMock()
            mock_crew._genai_otel_session_id = "session-via-agent"

            mock_agent = MagicMock()
            mock_agent.crew = mock_crew
            mock_agent.role = "Researcher"

            # Task without direct crew attr but with agent.crew
            mock_task = MagicMock(spec=["description", "expected_output", "id", "agent"])
            mock_task.description = "Test task"
            mock_task.expected_output = "Expected"
            mock_task.id = "task-1"
            mock_task.agent = mock_agent

            attrs = instrumentor._extract_task_attributes(mock_task, (), {})

            self.assertEqual(attrs["session.id"], "session-via-agent")

    def test_extract_agent_attributes_session_propagation(self):
        """Test that session.id is propagated from crew to agent spans."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            # Create mock crew with session ID
            mock_crew = MagicMock()
            mock_crew._genai_otel_session_id = "session-from-crew"

            # Create mock agent with reference to crew
            mock_agent = MagicMock()
            mock_agent.role = "Researcher"
            mock_agent.crew = mock_crew

            attrs = instrumentor._extract_agent_attributes(mock_agent, (), {})

            self.assertEqual(attrs["session.id"], "session-from-crew")

    def test_extract_task_attributes_no_session_without_crew(self):
        """Test that task spans without crew reference don't set session.id."""
        with patch.dict("sys.modules", {"crewai": MagicMock()}):
            instrumentor = CrewAIInstrumentor()

            mock_task = MagicMock(spec=["description", "expected_output", "id"])
            mock_task.description = "Test task"
            mock_task.expected_output = "Expected"
            mock_task.id = "task-1"

            attrs = instrumentor._extract_task_attributes(mock_task, (), {})

            self.assertNotIn("session.id", attrs)


if __name__ == "__main__":
    unittest.main()
