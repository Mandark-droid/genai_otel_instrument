import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from genai_otel.instrumentors.cometapi_instrumentor import CometAPIInstrumentor


class TestCometAPIInstrumentor(unittest.TestCase):
    """Tests for CometAPIInstrumentor"""

    @patch("genai_otel.instrumentors.cometapi_instrumentor.logger")
    def test_init_with_both_libraries_available(self, mock_logger):
        """Test that __init__ detects OpenAI/Anthropic library availability."""
        with patch.dict("sys.modules", {"openai": MagicMock(), "anthropic": MagicMock()}):
            instrumentor = CometAPIInstrumentor()

            self.assertTrue(instrumentor._cometapi_available)
            self.assertTrue(instrumentor._openai_available)
            self.assertTrue(instrumentor._anthropic_available)
            mock_logger.debug.assert_called_with(
                "OpenAI/Anthropic library detected, CometAPI instrumentation available"
            )

    def test_init_with_only_openai_available(self):
        """Test that CometAPI is available when only the OpenAI library is installed."""
        with patch.dict("sys.modules", {"openai": MagicMock(), "anthropic": None}):
            instrumentor = CometAPIInstrumentor()

            self.assertTrue(instrumentor._cometapi_available)
            self.assertTrue(instrumentor._openai_available)
            self.assertFalse(instrumentor._anthropic_available)

    def test_init_with_only_anthropic_available(self):
        """Test that CometAPI is available when only the Anthropic library is installed."""
        with patch.dict("sys.modules", {"openai": None, "anthropic": MagicMock()}):
            instrumentor = CometAPIInstrumentor()

            self.assertTrue(instrumentor._cometapi_available)
            self.assertFalse(instrumentor._openai_available)
            self.assertTrue(instrumentor._anthropic_available)

    @patch("genai_otel.instrumentors.cometapi_instrumentor.logger")
    def test_init_with_no_libraries_available(self, mock_logger):
        """Test that __init__ handles missing OpenAI and Anthropic libraries gracefully."""
        with patch.dict("sys.modules", {"openai": None, "anthropic": None}):
            instrumentor = CometAPIInstrumentor()

            self.assertFalse(instrumentor._cometapi_available)
            mock_logger.debug.assert_called_with(
                "Neither OpenAI nor Anthropic library installed, "
                "CometAPI instrumentation will be skipped"
            )

    @patch("genai_otel.instrumentors.cometapi_instrumentor.logger")
    def test_instrument_when_no_libraries_available(self, mock_logger):
        """Test that instrument skips when no supported library is available."""
        with patch.dict("sys.modules", {"openai": None, "anthropic": None}):
            instrumentor = CometAPIInstrumentor()
            config = MagicMock()

            instrumentor.instrument(config)

            mock_logger.debug.assert_any_call(
                "Skipping CometAPI instrumentation - library not available"
            )

    @patch("genai_otel.instrumentors.cometapi_instrumentor.logger")
    def test_instrument_with_both_libraries_available(self, mock_logger):
        """Test that instrument wraps both OpenAI and Anthropic client inits."""

        class MockOpenAI:
            def __init__(self):
                pass

        class MockAnthropic:
            def __init__(self):
                pass

        mock_openai = MagicMock()
        mock_openai.OpenAI = MockOpenAI
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic = MockAnthropic
        mock_wrapt = MagicMock()

        with patch.dict(
            "sys.modules",
            {"openai": mock_openai, "anthropic": mock_anthropic, "wrapt": mock_wrapt},
        ):
            instrumentor = CometAPIInstrumentor()
            config = MagicMock()

            instrumentor.instrument(config)

            self.assertEqual(instrumentor.config, config)
            self.assertTrue(instrumentor._instrumented)
            mock_logger.info.assert_called_with("CometAPI instrumentation enabled")
            # Both OpenAI.__init__ and Anthropic.__init__ should be wrapped
            self.assertEqual(mock_wrapt.FunctionWrapper.call_count, 2)

    def test_instrument_with_only_anthropic_available(self):
        """Test that instrument wraps only the Anthropic client when OpenAI is missing."""

        class MockAnthropic:
            def __init__(self):
                pass

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic = MockAnthropic
        mock_wrapt = MagicMock()

        with patch.dict(
            "sys.modules", {"openai": None, "anthropic": mock_anthropic, "wrapt": mock_wrapt}
        ):
            instrumentor = CometAPIInstrumentor()
            config = MagicMock()

            instrumentor.instrument(config)

            self.assertTrue(instrumentor._instrumented)
            mock_wrapt.FunctionWrapper.assert_called_once()

    @patch("genai_otel.instrumentors.cometapi_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_false(self, mock_logger):
        """Test that instrument handles exceptions gracefully when fail_on_error is False."""
        mock_openai = MagicMock()

        def mock_hasattr_side_effect(obj, name):
            if name == "OpenAI":
                raise RuntimeError("Test error")
            return True

        with patch.dict(
            "sys.modules",
            {"openai": mock_openai, "anthropic": MagicMock(), "wrapt": MagicMock()},
        ):
            with patch("builtins.hasattr", side_effect=mock_hasattr_side_effect):
                instrumentor = CometAPIInstrumentor()
                config = MagicMock()
                config.fail_on_error = False

                # Should not raise exception
                instrumentor.instrument(config)

                mock_logger.error.assert_called_once()

    def test_instrument_exception_with_fail_on_error_true(self):
        """Test that instrument raises exceptions when fail_on_error is True."""
        mock_openai = MagicMock()

        def mock_hasattr_side_effect(obj, name):
            if name == "OpenAI":
                raise RuntimeError("Test error")
            return True

        with patch.dict(
            "sys.modules",
            {"openai": mock_openai, "anthropic": MagicMock(), "wrapt": MagicMock()},
        ):
            with patch("builtins.hasattr", side_effect=mock_hasattr_side_effect):
                instrumentor = CometAPIInstrumentor()
                config = MagicMock()
                config.fail_on_error = True

                with self.assertRaises(RuntimeError):
                    instrumentor.instrument(config)

    def test_is_cometapi_client_with_cometapi_base_url(self):
        """Test that _is_cometapi_client detects CometAPI clients correctly."""
        with patch.dict("sys.modules", {"openai": MagicMock(), "anthropic": MagicMock()}):
            instrumentor = CometAPIInstrumentor()

            mock_client = MagicMock()
            mock_client.base_url = "https://api.cometapi.com"

            self.assertTrue(instrumentor._is_cometapi_client(mock_client))

    def test_is_cometapi_client_with_v1_base_url(self):
        """Test that _is_cometapi_client detects CometAPI clients with /v1 path."""
        with patch.dict("sys.modules", {"openai": MagicMock(), "anthropic": MagicMock()}):
            instrumentor = CometAPIInstrumentor()

            mock_client = MagicMock()
            mock_client.base_url = "https://api.cometapi.com/v1"

            self.assertTrue(instrumentor._is_cometapi_client(mock_client))

    def test_is_cometapi_client_with_non_cometapi_base_url(self):
        """Test that _is_cometapi_client returns False for non-CometAPI clients."""
        with patch.dict("sys.modules", {"openai": MagicMock(), "anthropic": MagicMock()}):
            instrumentor = CometAPIInstrumentor()

            mock_client = MagicMock()
            mock_client.base_url = "https://api.anthropic.com"

            self.assertFalse(instrumentor._is_cometapi_client(mock_client))

    def test_is_cometapi_client_without_base_url(self):
        """Test that _is_cometapi_client handles missing base_url."""
        with patch.dict("sys.modules", {"openai": MagicMock(), "anthropic": MagicMock()}):
            instrumentor = CometAPIInstrumentor()

            mock_client = MagicMock()
            del mock_client.base_url

            self.assertFalse(instrumentor._is_cometapi_client(mock_client))

    def test_instrument_openai_client(self):
        """Test that _instrument_openai_client wraps chat.completions.create."""
        with patch.dict("sys.modules", {"openai": MagicMock(), "anthropic": MagicMock()}):
            instrumentor = CometAPIInstrumentor()

            mock_client = MagicMock()
            original_create = MagicMock()
            mock_client.chat.completions.create = original_create

            mock_wrapper = MagicMock()
            mock_decorator = MagicMock(return_value=mock_wrapper)
            instrumentor.create_span_wrapper = MagicMock(return_value=mock_decorator)

            instrumentor._instrument_openai_client(mock_client)

            instrumentor.create_span_wrapper.assert_called_once_with(
                span_name="cometapi.chat.completion",
                extract_attributes=instrumentor._extract_cometapi_attributes,
            )
            mock_decorator.assert_called_once_with(original_create)
            self.assertEqual(mock_client.chat.completions.create, mock_wrapper)

    def test_instrument_anthropic_client(self):
        """Test that _instrument_anthropic_client wraps messages.create."""
        with patch.dict("sys.modules", {"openai": MagicMock(), "anthropic": MagicMock()}):
            instrumentor = CometAPIInstrumentor()

            mock_client = MagicMock()
            original_create = MagicMock()
            mock_client.messages.create = original_create

            mock_wrapper = MagicMock()
            mock_decorator = MagicMock(return_value=mock_wrapper)
            instrumentor.create_span_wrapper = MagicMock(return_value=mock_decorator)

            instrumentor._instrument_anthropic_client(mock_client)

            instrumentor.create_span_wrapper.assert_called_once_with(
                span_name="cometapi.messages.create",
                extract_attributes=instrumentor._extract_cometapi_attributes,
            )
            mock_decorator.assert_called_once_with(original_create)
            self.assertEqual(mock_client.messages.create, mock_wrapper)

    def test_extract_cometapi_attributes_with_messages(self):
        """Test that _extract_cometapi_attributes extracts attributes correctly."""
        with patch.dict("sys.modules", {"openai": MagicMock(), "anthropic": MagicMock()}):
            instrumentor = CometAPIInstrumentor()

            kwargs = {
                "model": "claude-sonnet-5",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "Hello, Claude"},
                    {"role": "assistant", "content": "Hello!"},
                    {"role": "user", "content": "Can you describe LLMs to me?"},
                ],
            }

            attrs = instrumentor._extract_cometapi_attributes(None, [], kwargs)

            self.assertEqual(attrs["gen_ai.system"], "cometapi")
            self.assertEqual(attrs["gen_ai.request.model"], "claude-sonnet-5")
            self.assertEqual(attrs["gen_ai.operation.name"], "chat")
            self.assertEqual(attrs["gen_ai.request.message_count"], 3)
            self.assertEqual(attrs["gen_ai.request.max_tokens"], 1024)
            self.assertIn("gen_ai.request.first_message", attrs)

    def test_extract_cometapi_attributes_without_messages(self):
        """Test that _extract_cometapi_attributes handles missing messages."""
        with patch.dict("sys.modules", {"openai": MagicMock(), "anthropic": MagicMock()}):
            instrumentor = CometAPIInstrumentor()

            kwargs = {"model": "gpt-5-mini"}

            attrs = instrumentor._extract_cometapi_attributes(None, [], kwargs)

            self.assertEqual(attrs["gen_ai.system"], "cometapi")
            self.assertEqual(attrs["gen_ai.request.model"], "gpt-5-mini")
            self.assertEqual(attrs["gen_ai.request.message_count"], 0)
            self.assertNotIn("gen_ai.request.first_message", attrs)

    def test_extract_usage_with_openai_style_response(self):
        """Test that _extract_usage extracts OpenAI-compatible token usage."""
        with patch.dict("sys.modules", {"openai": MagicMock(), "anthropic": MagicMock()}):
            instrumentor = CometAPIInstrumentor()

            usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150)
            mock_result = SimpleNamespace(usage=usage)

            extracted = instrumentor._extract_usage(mock_result)

            self.assertIsNotNone(extracted)
            self.assertEqual(extracted["prompt_tokens"], 100)
            self.assertEqual(extracted["completion_tokens"], 50)
            self.assertEqual(extracted["total_tokens"], 150)

    def test_extract_usage_with_anthropic_style_response(self):
        """Test that _extract_usage extracts Anthropic-compatible token usage."""
        with patch.dict("sys.modules", {"openai": MagicMock(), "anthropic": MagicMock()}):
            instrumentor = CometAPIInstrumentor()

            usage = SimpleNamespace(
                input_tokens=100,
                output_tokens=50,
                cache_read_input_tokens=10,
                cache_creation_input_tokens=5,
            )
            mock_result = SimpleNamespace(usage=usage)

            extracted = instrumentor._extract_usage(mock_result)

            self.assertIsNotNone(extracted)
            self.assertEqual(extracted["prompt_tokens"], 100)
            self.assertEqual(extracted["completion_tokens"], 50)
            self.assertEqual(extracted["total_tokens"], 150)
            self.assertEqual(extracted["cache_read_input_tokens"], 10)
            self.assertEqual(extracted["cache_creation_input_tokens"], 5)

    def test_extract_usage_without_usage(self):
        """Test that _extract_usage handles missing usage information."""
        with patch.dict("sys.modules", {"openai": MagicMock(), "anthropic": MagicMock()}):
            instrumentor = CometAPIInstrumentor()

            mock_result = SimpleNamespace()

            self.assertIsNone(instrumentor._extract_usage(mock_result))

    def test_extract_response_attributes_openai_style(self):
        """Test that _extract_response_attributes handles OpenAI-compatible responses."""
        with patch.dict("sys.modules", {"openai": MagicMock(), "anthropic": MagicMock()}):
            instrumentor = CometAPIInstrumentor()

            mock_result = SimpleNamespace(
                id="response-123",
                model="gpt-5-mini",
                choices=[SimpleNamespace(finish_reason="stop")],
            )

            attrs = instrumentor._extract_response_attributes(mock_result)

            self.assertEqual(attrs["gen_ai.response.id"], "response-123")
            self.assertEqual(attrs["gen_ai.response.model"], "gpt-5-mini")
            self.assertEqual(attrs["gen_ai.response.finish_reasons"], ["stop"])

    def test_extract_response_attributes_anthropic_style(self):
        """Test that _extract_response_attributes handles Anthropic-compatible responses."""
        with patch.dict("sys.modules", {"openai": MagicMock(), "anthropic": MagicMock()}):
            instrumentor = CometAPIInstrumentor()

            mock_result = SimpleNamespace(
                id="msg-123",
                model="claude-sonnet-5",
                stop_reason="end_turn",
            )

            attrs = instrumentor._extract_response_attributes(mock_result)

            self.assertEqual(attrs["gen_ai.response.id"], "msg-123")
            self.assertEqual(attrs["gen_ai.response.model"], "claude-sonnet-5")
            self.assertEqual(attrs["gen_ai.response.finish_reasons"], ["end_turn"])

    def test_extract_finish_reason_openai_style(self):
        """Test that _extract_finish_reason handles OpenAI-compatible responses."""
        with patch.dict("sys.modules", {"openai": MagicMock(), "anthropic": MagicMock()}):
            instrumentor = CometAPIInstrumentor()

            mock_result = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop")])

            self.assertEqual(instrumentor._extract_finish_reason(mock_result), "stop")

    def test_extract_finish_reason_anthropic_style(self):
        """Test that _extract_finish_reason handles Anthropic-compatible responses."""
        with patch.dict("sys.modules", {"openai": MagicMock(), "anthropic": MagicMock()}):
            instrumentor = CometAPIInstrumentor()

            mock_result = SimpleNamespace(stop_reason="end_turn")

            self.assertEqual(instrumentor._extract_finish_reason(mock_result), "end_turn")

    def test_extract_finish_reason_without_data(self):
        """Test that _extract_finish_reason handles responses without finish info."""
        with patch.dict("sys.modules", {"openai": MagicMock(), "anthropic": MagicMock()}):
            instrumentor = CometAPIInstrumentor()

            mock_result = SimpleNamespace(choices=[])

            self.assertIsNone(instrumentor._extract_finish_reason(mock_result))

    def test_add_content_events_anthropic_style(self):
        """Test that _add_content_events records Anthropic-style content blocks."""
        with patch.dict("sys.modules", {"openai": MagicMock(), "anthropic": MagicMock()}):
            instrumentor = CometAPIInstrumentor()

            mock_span = MagicMock()
            mock_result = SimpleNamespace(
                content=[SimpleNamespace(text="Hello! LLMs are large language models.")]
            )
            request_kwargs = {"messages": [{"role": "user", "content": "Describe LLMs"}]}

            instrumentor._add_content_events(mock_span, mock_result, request_kwargs)

            # One prompt event + one completion event
            self.assertEqual(mock_span.add_event.call_count, 2)
            mock_span.set_attribute.assert_called_once_with(
                "gen_ai.response", "Hello! LLMs are large language models."
            )

    def test_add_content_events_openai_style(self):
        """Test that _add_content_events records OpenAI-style choices."""
        with patch.dict("sys.modules", {"openai": MagicMock(), "anthropic": MagicMock()}):
            instrumentor = CometAPIInstrumentor()

            mock_span = MagicMock()
            mock_result = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Hi there!"))]
            )
            request_kwargs = {"messages": [{"role": "user", "content": "Hi"}]}

            instrumentor._add_content_events(mock_span, mock_result, request_kwargs)

            self.assertEqual(mock_span.add_event.call_count, 2)
            mock_span.set_attribute.assert_called_once_with("gen_ai.response", "Hi there!")


if __name__ == "__main__":
    unittest.main()
