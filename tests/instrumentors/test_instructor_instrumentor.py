import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.instructor_instrumentor import InstructorInstrumentor


class TestInstructorInstrumentor(unittest.TestCase):
    """Tests for InstructorInstrumentor"""

    @patch("genai_otel.instrumentors.instructor_instrumentor.logger")
    def test_init_with_instructor_available(self, mock_logger):
        """Test that __init__ detects Instructor availability."""
        mock_instructor = MagicMock()

        with patch.dict("sys.modules", {"instructor": mock_instructor}):
            instrumentor = InstructorInstrumentor()

            self.assertTrue(instrumentor._instructor_available)
            mock_logger.debug.assert_called_with(
                "Instructor framework detected and available for instrumentation"
            )

    @patch("genai_otel.instrumentors.instructor_instrumentor.logger")
    def test_init_with_instructor_not_available(self, mock_logger):
        """Test that __init__ handles missing Instructor."""
        with patch.dict("sys.modules", {"instructor": None}):
            instrumentor = InstructorInstrumentor()

            self.assertFalse(instrumentor._instructor_available)
            mock_logger.debug.assert_called_with(
                "Instructor not installed, instrumentation will be skipped"
            )

    @patch("genai_otel.instrumentors.instructor_instrumentor.logger")
    def test_instrument_when_instructor_not_available(self, mock_logger):
        """Test that instrument skips when Instructor is not available."""
        with patch.dict("sys.modules", {"instructor": None}):
            instrumentor = InstructorInstrumentor()
            config = MagicMock()

            instrumentor.instrument(config)

            mock_logger.debug.assert_any_call(
                "Skipping Instructor instrumentation - library not available"
            )

    @patch("genai_otel.instrumentors.instructor_instrumentor.logger")
    def test_instrument_wraps_methods(self, mock_logger):
        """Test that instrument wraps Instructor methods."""
        mock_instructor = MagicMock()
        mock_wrapt = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "instructor": mock_instructor,
                "instructor.client": mock_instructor.client,
                "instructor.retry": mock_instructor.retry,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = InstructorInstrumentor()
            config = MagicMock()

            # Call instrument
            instrumentor.instrument(config)

            # Assert wrapt.wrap_function_wrapper was called
            self.assertTrue(mock_wrapt.wrap_function_wrapper.called)
            self.assertEqual(instrumentor.config, config)
            self.assertTrue(instrumentor._instrumented)
            mock_logger.info.assert_called_with("Instructor instrumentation enabled")

    @patch("genai_otel.instrumentors.instructor_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_false(self, mock_logger):
        """Test that exceptions are logged when fail_on_error is False."""
        mock_instructor = MagicMock()
        mock_wrapt = MagicMock()
        mock_wrapt.wrap_function_wrapper.side_effect = RuntimeError("Wrap failed")

        with patch.dict("sys.modules", {"instructor": mock_instructor, "wrapt": mock_wrapt}):
            instrumentor = InstructorInstrumentor()
            config = MagicMock()
            config.fail_on_error = False

            # Should not raise
            instrumentor.instrument(config)

            mock_logger.error.assert_called_once()

    @patch("genai_otel.instrumentors.instructor_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_true(self, mock_logger):
        """Test that exceptions are raised when fail_on_error is True."""
        mock_instructor = MagicMock()
        mock_wrapt = MagicMock()
        mock_wrapt.wrap_function_wrapper.side_effect = RuntimeError("Wrap failed")

        with patch.dict("sys.modules", {"instructor": mock_instructor, "wrapt": mock_wrapt}):
            instrumentor = InstructorInstrumentor()
            config = MagicMock()
            config.fail_on_error = True

            # Should raise
            with self.assertRaises(RuntimeError) as context:
                instrumentor.instrument(config)

            self.assertEqual(str(context.exception), "Wrap failed")

    def test_extract_from_provider_attributes(self):
        """Test extraction of from_provider attributes."""
        mock_instructor = MagicMock()

        with patch.dict("sys.modules", {"instructor": mock_instructor}):
            instrumentor = InstructorInstrumentor()

            # Test with provider string
            args = ("openai/gpt-4",)
            kwargs = {"mode": "JSON"}

            attrs = instrumentor._extract_from_provider_attributes(args, kwargs)

            # Assert
            self.assertEqual(attrs["gen_ai.system"], "instructor")
            self.assertEqual(attrs["gen_ai.operation.name"], "from_provider")
            self.assertEqual(attrs["instructor.provider"], "openai/gpt-4")
            self.assertEqual(attrs["instructor.provider.name"], "openai")
            self.assertEqual(attrs["gen_ai.request.model"], "gpt-4")
            self.assertEqual(attrs["instructor.mode"], "JSON")

    def test_extract_patch_attributes(self):
        """Test extraction of patch attributes."""
        mock_instructor = MagicMock()

        with patch.dict("sys.modules", {"instructor": mock_instructor}):
            instrumentor = InstructorInstrumentor()

            # Create mock client
            mock_client = MagicMock()
            mock_client.__class__.__name__ = "OpenAI"

            args = (mock_client,)
            kwargs = {"mode": "TOOLS"}

            attrs = instrumentor._extract_patch_attributes(args, kwargs)

            # Assert
            self.assertEqual(attrs["gen_ai.system"], "instructor")
            self.assertEqual(attrs["gen_ai.operation.name"], "patch")
            self.assertEqual(attrs["instructor.client.type"], "OpenAI")
            self.assertEqual(attrs["instructor.mode"], "TOOLS")

    def test_extract_create_attributes_with_response_model(self):
        """Test extraction of create_with_completion attributes with response_model."""
        mock_instructor = MagicMock()

        with patch.dict("sys.modules", {"instructor": mock_instructor}):
            instrumentor = InstructorInstrumentor()

            # Create mock response model (Pydantic-like)
            mock_model = MagicMock()
            mock_model.__name__ = "UserProfile"
            mock_model.model_fields = {
                "name": MagicMock(),
                "age": MagicMock(),
                "email": MagicMock(),
            }

            # Create mock instance
            mock_instance = MagicMock()

            kwargs = {
                "response_model": mock_model,
                "max_retries": 3,
                "model": "gpt-4",
                "stream": False,
            }

            attrs = instrumentor._extract_create_attributes(mock_instance, kwargs)

            # Assert
            self.assertEqual(attrs["gen_ai.system"], "instructor")
            self.assertEqual(attrs["gen_ai.operation.name"], "create_with_completion")
            self.assertEqual(attrs["instructor.response_model.name"], "UserProfile")
            self.assertEqual(attrs["instructor.response_model.fields"], ["name", "age", "email"])
            self.assertEqual(attrs["instructor.response_model.fields_count"], 3)
            self.assertEqual(attrs["instructor.max_retries"], 3)
            self.assertEqual(attrs["gen_ai.request.model"], "gpt-4")
            self.assertFalse(attrs["instructor.stream"])

    def test_extract_create_attributes_with_streaming(self):
        """Test extraction of create_with_completion attributes with streaming."""
        mock_instructor = MagicMock()

        with patch.dict("sys.modules", {"instructor": mock_instructor}):
            instrumentor = InstructorInstrumentor()

            # Create mock partial model
            mock_model = MagicMock()
            mock_model.__name__ = "PartialResponse"
            mock_model.__origin__ = "Partial"  # Indicates Partial[Model]

            mock_instance = MagicMock()

            kwargs = {
                "response_model": mock_model,
                "stream": True,
                "validation_context": {"key": "value"},
            }

            attrs = instrumentor._extract_create_attributes(mock_instance, kwargs)

            # Assert
            self.assertTrue(attrs["instructor.response_model.is_partial"])
            self.assertTrue(attrs["instructor.stream"])
            self.assertTrue(attrs["instructor.has_validation_context"])

    def test_extract_retry_attributes(self):
        """Test extraction of retry attributes."""
        mock_instructor = MagicMock()

        with patch.dict("sys.modules", {"instructor": mock_instructor}):
            instrumentor = InstructorInstrumentor()

            kwargs = {
                "max_retries": 5,
                "context": {"validation_error": "field missing"},
            }

            attrs = instrumentor._extract_retry_attributes(kwargs)

            # Assert
            self.assertEqual(attrs["gen_ai.system"], "instructor")
            self.assertEqual(attrs["gen_ai.operation.name"], "retry")
            self.assertEqual(attrs["instructor.retry.max_attempts"], 5)
            self.assertTrue(attrs["instructor.retry.has_context"])

    def test_extract_create_response_attributes(self):
        """Test extraction of create_with_completion response attributes."""
        mock_instructor = MagicMock()

        with patch.dict("sys.modules", {"instructor": mock_instructor}):
            instrumentor = InstructorInstrumentor()

            # Create mock Pydantic model result
            mock_result = MagicMock()
            mock_result.__class__.__name__ = "UserProfile"
            mock_result.model_fields = {"name": MagicMock(), "age": MagicMock()}
            mock_result.model_dump.return_value = {
                "name": "John Doe",
                "age": 30,
                "email": "john@example.com",
            }

            attrs = instrumentor._extract_create_response_attributes(mock_result)

            # Assert
            self.assertEqual(attrs["instructor.response.type"], "UserProfile")
            self.assertEqual(attrs["instructor.response.fields_count"], 2)
            self.assertIn("name", attrs["instructor.response.fields"])
            self.assertEqual(attrs["instructor.response.name"], "John Doe")
            self.assertEqual(attrs["instructor.response.age"], "30")
            self.assertTrue(attrs["instructor.validation.success"])

    def test_extract_create_response_attributes_validation_failure(self):
        """Test extraction of response attributes when validation fails."""
        mock_instructor = MagicMock()

        with patch.dict("sys.modules", {"instructor": mock_instructor}):
            instrumentor = InstructorInstrumentor()

            # Create result that doesn't have model_dump (validation failed)
            mock_result = MagicMock(spec=[])

            attrs = instrumentor._extract_create_response_attributes(mock_result)

            # Assert
            self.assertFalse(attrs["instructor.validation.success"])

    def test_extract_usage_returns_none(self):
        """Test that _extract_usage returns None."""
        mock_instructor = MagicMock()

        with patch.dict("sys.modules", {"instructor": mock_instructor}):
            instrumentor = InstructorInstrumentor()

            # Token usage tracked by underlying LLM providers
            result = MagicMock()

            usage = instrumentor._extract_usage(result)

            self.assertIsNone(usage)

    def test_extract_finish_reason_completed(self):
        """Test extraction of finish reason for successful completion."""
        mock_instructor = MagicMock()

        with patch.dict("sys.modules", {"instructor": mock_instructor}):
            instrumentor = InstructorInstrumentor()

            # Create mock Pydantic result
            mock_result = MagicMock()
            mock_result.model_dump.return_value = {"field": "value"}

            finish_reason = instrumentor._extract_finish_reason(mock_result)

            # Assert
            self.assertEqual(finish_reason, "completed")

    def test_extract_finish_reason_from_raw_response(self):
        """Test extraction of finish reason from raw response."""
        mock_instructor = MagicMock()

        with patch.dict("sys.modules", {"instructor": mock_instructor}):
            instrumentor = InstructorInstrumentor()

            # Create mock result with raw response (no model_dump)
            mock_choice = MagicMock()
            mock_choice.finish_reason = "stop"

            mock_raw_response = MagicMock()
            mock_raw_response.choices = [mock_choice]

            mock_result = MagicMock(spec=["_raw_response"])
            mock_result._raw_response = mock_raw_response

            finish_reason = instrumentor._extract_finish_reason(mock_result)

            # Assert
            self.assertEqual(finish_reason, "stop")

    def test_extract_finish_reason_none(self):
        """Test extraction of finish reason returns None when not available."""
        mock_instructor = MagicMock()

        with patch.dict("sys.modules", {"instructor": mock_instructor}):
            instrumentor = InstructorInstrumentor()

            # Create mock result without finish reason
            mock_result = MagicMock(spec=[])

            finish_reason = instrumentor._extract_finish_reason(mock_result)

            # Assert
            self.assertIsNone(finish_reason)

    def test_wrap_from_provider_creates_span(self):
        """Test that _wrap_from_provider creates appropriate span."""
        mock_instructor = MagicMock()

        with patch.dict("sys.modules", {"instructor": mock_instructor}):
            instrumentor = InstructorInstrumentor()
            instrumentor.config = MagicMock()

            # Mock the create_span_wrapper
            mock_wrapper = MagicMock(return_value=lambda f: f)
            instrumentor.create_span_wrapper = mock_wrapper

            # Mock wrapped function
            mock_wrapped = MagicMock(return_value="result")

            # Call wrap
            result = instrumentor._wrap_from_provider(mock_wrapped, None, (), {})

            # Assert span wrapper was called
            mock_wrapper.assert_called_once()
            call_kwargs = mock_wrapper.call_args[1]
            self.assertEqual(call_kwargs["span_name"], "instructor.from_provider")

    def test_wrap_patch_creates_span(self):
        """Test that _wrap_patch creates appropriate span."""
        mock_instructor = MagicMock()

        with patch.dict("sys.modules", {"instructor": mock_instructor}):
            instrumentor = InstructorInstrumentor()
            instrumentor.config = MagicMock()

            # Mock the create_span_wrapper
            mock_wrapper = MagicMock(return_value=lambda f: f)
            instrumentor.create_span_wrapper = mock_wrapper

            # Mock wrapped function
            mock_wrapped = MagicMock(return_value="result")

            # Call wrap
            result = instrumentor._wrap_patch(mock_wrapped, None, (), {})

            # Assert span wrapper was called
            mock_wrapper.assert_called_once()
            call_kwargs = mock_wrapper.call_args[1]
            self.assertEqual(call_kwargs["span_name"], "instructor.patch")

    def test_wrap_create_with_completion_creates_span(self):
        """Test that _wrap_create_with_completion creates appropriate span."""
        mock_instructor = MagicMock()

        with patch.dict("sys.modules", {"instructor": mock_instructor}):
            instrumentor = InstructorInstrumentor()
            instrumentor.config = MagicMock()

            # Create mock instance
            mock_instance = MagicMock()

            # Mock the create_span_wrapper
            mock_wrapper = MagicMock(return_value=lambda f: f)
            instrumentor.create_span_wrapper = mock_wrapper

            # Mock wrapped function
            mock_wrapped = MagicMock(return_value="result")

            # Call wrap
            result = instrumentor._wrap_create_with_completion(mock_wrapped, mock_instance, (), {})

            # Assert span wrapper was called
            mock_wrapper.assert_called_once()
            call_kwargs = mock_wrapper.call_args[1]
            self.assertEqual(call_kwargs["span_name"], "instructor.create_with_completion")

    def test_wrap_retry_sync_creates_span(self):
        """Test that _wrap_retry_sync creates appropriate span."""
        mock_instructor = MagicMock()

        with patch.dict("sys.modules", {"instructor": mock_instructor}):
            instrumentor = InstructorInstrumentor()
            instrumentor.config = MagicMock()

            # Mock the create_span_wrapper
            mock_wrapper = MagicMock(return_value=lambda f: f)
            instrumentor.create_span_wrapper = mock_wrapper

            # Mock wrapped function
            mock_wrapped = MagicMock(return_value="result")

            # Call wrap
            result = instrumentor._wrap_retry_sync(mock_wrapped, None, (), {})

            # Assert span wrapper was called
            mock_wrapper.assert_called_once()
            call_kwargs = mock_wrapper.call_args[1]
            self.assertEqual(call_kwargs["span_name"], "instructor.retry")

    def test_extract_create_attributes_minimal(self):
        """Test extraction with minimal arguments."""
        mock_instructor = MagicMock()

        with patch.dict("sys.modules", {"instructor": mock_instructor}):
            instrumentor = InstructorInstrumentor()

            mock_instance = MagicMock()
            kwargs = {}

            attrs = instrumentor._extract_create_attributes(mock_instance, kwargs)

            # Assert basic attributes are present
            self.assertEqual(attrs["gen_ai.system"], "instructor")
            self.assertEqual(attrs["gen_ai.operation.name"], "create_with_completion")

    def test_extract_response_attributes_with_long_values(self):
        """Test that long response values are truncated."""
        mock_instructor = MagicMock()

        with patch.dict("sys.modules", {"instructor": mock_instructor}):
            instrumentor = InstructorInstrumentor()

            # Create mock result with long string value
            long_text = "x" * 300
            mock_result = MagicMock()
            mock_result.__class__.__name__ = "Response"
            mock_result.model_fields = {"text": MagicMock()}
            mock_result.model_dump.return_value = {"text": long_text}

            attrs = instrumentor._extract_create_response_attributes(mock_result)

            # Assert value is truncated to 200 chars
            self.assertEqual(len(attrs["instructor.response.text"]), 200)
            self.assertTrue(attrs["instructor.validation.success"])


if __name__ == "__main__":
    unittest.main()
