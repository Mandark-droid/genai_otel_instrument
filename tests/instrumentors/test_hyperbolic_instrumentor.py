import unittest
from unittest.mock import MagicMock, Mock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.hyperbolic_instrumentor import HyperbolicInstrumentor


class TestHyperbolicInstrumentor(unittest.TestCase):
    """Tests for HyperbolicInstrumentor"""

    @patch("genai_otel.instrumentors.hyperbolic_instrumentor.logger")
    def test_init_with_requests_available(self, mock_logger):
        """Test that __init__ detects requests availability."""
        with patch.dict("sys.modules", {"requests": MagicMock()}):
            instrumentor = HyperbolicInstrumentor()

            self.assertTrue(instrumentor._requests_available)
            mock_logger.debug.assert_called_with(
                "Requests library detected, Hyperbolic instrumentation available"
            )

    @patch("genai_otel.instrumentors.hyperbolic_instrumentor.logger")
    def test_init_with_requests_not_available(self, mock_logger):
        """Test that __init__ handles missing requests gracefully."""
        with patch.dict("sys.modules", {"requests": None}):
            instrumentor = HyperbolicInstrumentor()

            self.assertFalse(instrumentor._requests_available)
            mock_logger.debug.assert_called_with(
                "Requests library not installed, Hyperbolic instrumentation skipped"
            )

    @patch("genai_otel.instrumentors.hyperbolic_instrumentor.logger")
    def test_instrument_with_requests_not_available(self, mock_logger):
        """Test that instrument skips when requests is not available."""
        with patch.dict("sys.modules", {"requests": None}):
            instrumentor = HyperbolicInstrumentor()
            config = OTelConfig()

            instrumentor.instrument(config)

            mock_logger.debug.assert_any_call(
                "Skipping Hyperbolic instrumentation - requests library not available"
            )

    @patch("genai_otel.instrumentors.hyperbolic_instrumentor.wrapt")
    @patch("genai_otel.instrumentors.hyperbolic_instrumentor.logger")
    def test_instrument_with_requests_available(self, mock_logger, mock_wrapt):
        """Test that instrument wraps requests.post when available."""
        # Create mock requests module
        mock_requests = MagicMock()
        mock_requests.post = MagicMock()

        with patch.dict("sys.modules", {"requests": mock_requests}):
            instrumentor = HyperbolicInstrumentor()
            config = OTelConfig()

            # Call instrument
            instrumentor.instrument(config)

            # Verify that instrumentation was set up
            self.assertTrue(instrumentor._instrumented)
            self.assertEqual(instrumentor.config, config)
            mock_logger.info.assert_called_with("Hyperbolic instrumentation enabled")

    def test_extract_request_attributes(self):
        """Test extraction of request attributes."""
        instrumentor = HyperbolicInstrumentor()

        request_data = {
            "model": "Qwen/Qwen3-Next-80B-A3B-Thinking",
            "messages": [{"role": "user", "content": "What can I do in SF?"}],
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": 507,
        }

        attrs = instrumentor._extract_request_attributes(request_data)

        self.assertEqual(attrs["gen_ai.system"], "hyperbolic")
        self.assertEqual(attrs["gen_ai.request.model"], "Qwen/Qwen3-Next-80B-A3B-Thinking")
        self.assertEqual(attrs["gen_ai.operation.name"], "chat")
        self.assertEqual(attrs["gen_ai.request.message_count"], 1)
        self.assertEqual(attrs["gen_ai.request.temperature"], 0.7)
        self.assertEqual(attrs["gen_ai.request.top_p"], 0.8)
        self.assertEqual(attrs["gen_ai.request.max_tokens"], 507)

    def test_extract_and_record_response(self):
        """Test extraction and recording of response data."""
        instrumentor = HyperbolicInstrumentor()
        instrumentor.config = OTelConfig(enable_cost_tracking=False)
        instrumentor.token_counter = MagicMock()

        # Create mock span
        mock_span = MagicMock()
        mock_span.attributes = {"gen_ai.request.model": "Qwen/Qwen3-Next-80B-A3B-Thinking"}

        # Create mock response data
        response_data = {
            "id": "resp-123",
            "model": "Qwen/Qwen3-Next-80B-A3B-Thinking",
            "choices": [{"finish_reason": "stop", "message": {"content": "Response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        instrumentor._extract_and_record_response(mock_span, response_data)

        # Verify span attributes were set
        mock_span.set_attribute.assert_any_call("gen_ai.response.id", "resp-123")
        mock_span.set_attribute.assert_any_call(
            "gen_ai.response.model", "Qwen/Qwen3-Next-80B-A3B-Thinking"
        )
        mock_span.set_attribute.assert_any_call("gen_ai.usage.input_tokens", 10)
        mock_span.set_attribute.assert_any_call("gen_ai.usage.output_tokens", 20)
        mock_span.set_attribute.assert_any_call("gen_ai.usage.total_tokens", 30)

        # Verify token metrics were recorded
        self.assertEqual(instrumentor.token_counter.add.call_count, 2)

    def test_extract_usage_returns_none(self):
        """Test that _extract_usage returns None (not used for HTTP instrumentation)."""
        instrumentor = HyperbolicInstrumentor()

        result = MagicMock()
        usage = instrumentor._extract_usage(result)

        self.assertIsNone(usage)

    def _record_with_cost(self, cost_counter):
        """Run a priced response through the recorder and return set attributes."""
        instrumentor = HyperbolicInstrumentor()
        instrumentor.config = OTelConfig(enable_cost_tracking=True)
        instrumentor.token_counter = MagicMock()
        instrumentor.cost_counter = cost_counter

        captured = {}
        mock_span = MagicMock()
        mock_span.attributes = {"gen_ai.request.model": "meta-llama/Llama-3.3-70B-Instruct"}
        mock_span.set_attribute.side_effect = lambda k, v: captured.__setitem__(k, v)

        instrumentor._extract_and_record_response(
            mock_span,
            {
                "id": "resp-1",
                "model": "meta-llama/Llama-3.3-70B-Instruct",
                "choices": [{"finish_reason": "stop", "message": {"content": "hi"}}],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 1000,
                    "total_tokens": 2000,
                },
            },
        )
        return captured

    def test_cost_emitted_under_the_standard_attribute(self):
        """Hyperbolic must report cost where every other instrumentor does.

        Anything aggregating spend across providers reads
        gen_ai.usage.cost.total; emitting only gen_ai.cost.amount made
        Hyperbolic spans invisible to those queries.
        """
        captured = self._record_with_cost(MagicMock())

        self.assertGreater(captured.get("gen_ai.usage.cost.total", 0), 0)
        # The legacy name stays so existing dashboards keep working.
        self.assertEqual(
            captured.get("gen_ai.cost.amount"), captured.get("gen_ai.usage.cost.total")
        )

    def test_cost_attribute_set_without_a_cost_counter(self):
        """A missing metric counter must not drop cost from the span too."""
        captured = self._record_with_cost(None)

        self.assertGreater(captured.get("gen_ai.usage.cost.total", 0), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
