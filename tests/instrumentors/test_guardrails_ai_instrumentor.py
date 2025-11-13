import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.guardrails_ai_instrumentor import GuardrailsAIInstrumentor


class TestGuardrailsAIInstrumentor(unittest.TestCase):
    """Tests for GuardrailsAIInstrumentor"""

    @patch("genai_otel.instrumentors.guardrails_ai_instrumentor.logger")
    def test_init_with_guardrails_available(self, mock_logger):
        """Test that __init__ detects Guardrails AI availability."""
        mock_guardrails = MagicMock()

        with patch.dict("sys.modules", {"guardrails": mock_guardrails}):
            instrumentor = GuardrailsAIInstrumentor()

            self.assertTrue(instrumentor._guardrails_available)
            mock_logger.debug.assert_called_with(
                "Guardrails AI framework detected and available for instrumentation"
            )

    @patch("genai_otel.instrumentors.guardrails_ai_instrumentor.logger")
    def test_init_with_guardrails_not_available(self, mock_logger):
        """Test that __init__ handles missing Guardrails AI."""
        with patch.dict("sys.modules", {"guardrails": None}):
            instrumentor = GuardrailsAIInstrumentor()

            self.assertFalse(instrumentor._guardrails_available)
            mock_logger.debug.assert_called_with(
                "Guardrails AI not installed, instrumentation will be skipped"
            )

    @patch("genai_otel.instrumentors.guardrails_ai_instrumentor.logger")
    def test_instrument_when_guardrails_not_available(self, mock_logger):
        """Test that instrument skips when Guardrails AI is not available."""
        with patch.dict("sys.modules", {"guardrails": None}):
            instrumentor = GuardrailsAIInstrumentor()
            config = MagicMock()

            instrumentor.instrument(config)

            mock_logger.debug.assert_any_call(
                "Skipping Guardrails AI instrumentation - library not available"
            )

    def test_extract_guard_call_attributes(self):
        """Test extraction of guard.__call__ attributes."""
        mock_guardrails = MagicMock()

        with patch.dict("sys.modules", {"guardrails": mock_guardrails}):
            instrumentor = GuardrailsAIInstrumentor()

            # Create mock guard with validators
            mock_validator1 = MagicMock()
            mock_validator1.__class__.__name__ = "RegexMatch"
            mock_validator1.on_fail_descriptor = "reask"

            mock_validator2 = MagicMock()
            mock_validator2.__class__.__name__ = "ValidLength"
            mock_validator2.on_fail_descriptor = "fix"

            mock_guard = MagicMock()
            mock_guard._validators = [mock_validator1, mock_validator2]

            kwargs = {
                "num_reasks": 3,
                "llm_api": MagicMock(__name__="openai.create"),
                "metadata": {"key": "value"},
                "prompt_params": {"param": "value"},
                "full_schema_reask": True,
            }

            attrs = instrumentor._extract_guard_call_attributes(mock_guard, kwargs)

            # Assert
            self.assertEqual(attrs["gen_ai.system"], "guardrails")
            self.assertEqual(attrs["gen_ai.operation.name"], "guard.call")
            self.assertIn("RegexMatch", attrs["guardrails.validators"])
            self.assertIn("ValidLength", attrs["guardrails.validators"])
            self.assertEqual(attrs["guardrails.validators_count"], 2)
            self.assertEqual(attrs["guardrails.num_reasks"], 3)
            self.assertTrue(attrs["guardrails.has_metadata"])
            self.assertTrue(attrs["guardrails.has_prompt_params"])
            self.assertTrue(attrs["guardrails.full_schema_reask"])

    def test_extract_guard_call_response_attributes(self):
        """Test extraction of guard.__call__ response attributes."""
        mock_guardrails = MagicMock()

        with patch.dict("sys.modules", {"guardrails": mock_guardrails}):
            instrumentor = GuardrailsAIInstrumentor()

            # Create mock ValidationOutcome
            mock_result = MagicMock()
            mock_result.validation_passed = True
            mock_result.validated_output = "This is validated output"
            mock_result.reasks = []
            mock_result.error = None

            attrs = instrumentor._extract_guard_call_response_attributes(mock_result)

            # Assert
            self.assertTrue(attrs["guardrails.validation.passed"])
            self.assertEqual(attrs["guardrails.validated_output_length"], 24)
            self.assertEqual(attrs["guardrails.reasks_count"], 0)

    def test_extract_guard_validate_attributes(self):
        """Test extraction of guard.validate attributes."""
        mock_guardrails = MagicMock()

        with patch.dict("sys.modules", {"guardrails": mock_guardrails}):
            instrumentor = GuardrailsAIInstrumentor()

            mock_validator = MagicMock()
            mock_validator.__class__.__name__ = "ValidJSON"

            mock_guard = MagicMock()
            mock_guard._validators = [mock_validator]

            args = ('{"valid": "json"}',)
            kwargs = {}

            attrs = instrumentor._extract_guard_validate_attributes(mock_guard, args, kwargs)

            # Assert
            self.assertEqual(attrs["gen_ai.system"], "guardrails")
            self.assertEqual(attrs["gen_ai.operation.name"], "guard.validate")
            self.assertIn("ValidJSON", attrs["guardrails.validators"])
            self.assertEqual(attrs["guardrails.llm_output_length"], 17)

    def test_extract_finish_reason_validated(self):
        """Test extraction of finish reason for validated output."""
        mock_guardrails = MagicMock()

        with patch.dict("sys.modules", {"guardrails": mock_guardrails}):
            instrumentor = GuardrailsAIInstrumentor()

            mock_result = MagicMock()
            mock_result.validation_passed = True

            finish_reason = instrumentor._extract_finish_reason(mock_result)

            self.assertEqual(finish_reason, "validated")

    def test_extract_finish_reason_validation_failed(self):
        """Test extraction of finish reason for failed validation."""
        mock_guardrails = MagicMock()

        with patch.dict("sys.modules", {"guardrails": mock_guardrails}):
            instrumentor = GuardrailsAIInstrumentor()

            mock_result = MagicMock()
            mock_result.validation_passed = False
            mock_result.error = None

            finish_reason = instrumentor._extract_finish_reason(mock_result)

            self.assertEqual(finish_reason, "validation_failed")


if __name__ == "__main__":
    unittest.main()
