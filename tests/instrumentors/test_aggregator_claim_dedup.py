"""Tests for aggregator base_url claims preventing duplicate instrumentation.

A client pointed at an aggregator (OpenRouter, CometAPI) must produce ONE span
per call - from the dedicated aggregator instrumentor - not an additional
nested span from the generic OpenAI/Anthropic instrumentor (which would also
double-count token and cost metrics).
"""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors import base as base_module
from genai_otel.instrumentors.base import find_base_url_claim, register_base_url_claim


def _openai_style_response():
    return SimpleNamespace(
        id="chatcmpl-123",
        model="gpt-5-mini",
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            completion_tokens_details=None,
        ),
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="hi", tool_calls=None),
            )
        ],
    )


def _anthropic_style_response():
    return SimpleNamespace(
        id="msg_123",
        model="claude-sonnet-5",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="hi")],
    )


def _make_fake_openai_module(response):
    """Build a fake `openai` module whose clients return `response`."""
    module = types.ModuleType("openai")

    class FakeCompletions:
        def __init__(self):
            def create(**kwargs):
                return response

            self.create = create

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, base_url=None, api_key=None):
            self.base_url = base_url
            self.api_key = api_key
            self.chat = FakeChat()

    module.OpenAI = FakeOpenAI
    return module


def _make_fake_anthropic_module(response):
    """Build a fake `anthropic` module whose clients return `response`."""
    module = types.ModuleType("anthropic")

    class FakeMessages:
        def __init__(self):
            def create(**kwargs):
                return response

            self.create = create

    class FakeAnthropic:
        def __init__(self, base_url=None, api_key=None):
            self.base_url = base_url
            self.api_key = api_key
            self.messages = FakeMessages()

    module.Anthropic = FakeAnthropic
    return module


def _mock_tracer():
    """A tracer whose start_span calls can be counted per instrumentor."""
    tracer = MagicMock()
    span = MagicMock()
    span.is_recording.return_value = True
    span.attributes = {}
    span.name = "span"
    tracer.start_span.return_value = span
    return tracer


class TestBaseUrlClaimRegistry(unittest.TestCase):
    """Unit tests for the claim registry itself."""

    def setUp(self):
        base_module._BASE_URL_CLAIMS.clear()

    def tearDown(self):
        base_module._BASE_URL_CLAIMS.clear()

    def test_no_claims_returns_none(self):
        self.assertIsNone(find_base_url_claim("https://api.cometapi.com/v1"))

    def test_none_base_url_returns_none(self):
        register_base_url_claim("cometapi.com", "cometapi")
        self.assertIsNone(find_base_url_claim(None))
        self.assertIsNone(find_base_url_claim(""))

    def test_claim_matches_substring_case_insensitive(self):
        register_base_url_claim("CometAPI.com", "cometapi")
        self.assertEqual(find_base_url_claim("https://API.COMETAPI.COM/v1"), "cometapi")

    def test_unclaimed_domain_returns_none(self):
        register_base_url_claim("cometapi.com", "cometapi")
        self.assertIsNone(find_base_url_claim("https://api.openai.com/v1"))


class TestAggregatorClaimDedup(unittest.TestCase):
    """A claimed client is instrumented ONLY by its dedicated instrumentor."""

    def setUp(self):
        base_module._BASE_URL_CLAIMS.clear()

    def tearDown(self):
        base_module._BASE_URL_CLAIMS.clear()

    def test_cometapi_client_single_span_via_openai_sdk(self):
        response = _openai_style_response()
        fake_openai = _make_fake_openai_module(response)
        with patch.dict(sys.modules, {"openai": fake_openai, "anthropic": None}):
            from genai_otel.instrumentors.cometapi_instrumentor import CometAPIInstrumentor
            from genai_otel.instrumentors.openai_instrumentor import OpenAIInstrumentor

            config = OTelConfig(service_name="test")
            generic = OpenAIInstrumentor()
            comet = CometAPIInstrumentor()
            generic.instrument(config)
            comet.instrument(config)

            generic.tracer = _mock_tracer()
            comet.tracer = _mock_tracer()

            client = fake_openai.OpenAI(base_url="https://api.cometapi.com/v1", api_key="k")
            result = client.chat.completions.create(
                model="gpt-5-mini", messages=[{"role": "user", "content": "hi"}]
            )

            self.assertIs(result, response)
            comet.tracer.start_span.assert_called_once()
            generic.tracer.start_span.assert_not_called()

    def test_plain_openai_client_still_instrumented_by_generic(self):
        response = _openai_style_response()
        fake_openai = _make_fake_openai_module(response)
        with patch.dict(sys.modules, {"openai": fake_openai, "anthropic": None}):
            from genai_otel.instrumentors.cometapi_instrumentor import CometAPIInstrumentor
            from genai_otel.instrumentors.openai_instrumentor import OpenAIInstrumentor

            config = OTelConfig(service_name="test")
            generic = OpenAIInstrumentor()
            comet = CometAPIInstrumentor()
            generic.instrument(config)
            comet.instrument(config)

            generic.tracer = _mock_tracer()
            comet.tracer = _mock_tracer()

            client = fake_openai.OpenAI(base_url="https://api.openai.com/v1", api_key="k")
            result = client.chat.completions.create(
                model="gpt-5-mini", messages=[{"role": "user", "content": "hi"}]
            )

            self.assertIs(result, response)
            generic.tracer.start_span.assert_called_once()
            comet.tracer.start_span.assert_not_called()

    def test_cometapi_client_falls_back_to_generic_when_cometapi_disabled(self):
        """No claim registered -> generic OpenAI instrumentor still traces the call."""
        response = _openai_style_response()
        fake_openai = _make_fake_openai_module(response)
        with patch.dict(sys.modules, {"openai": fake_openai, "anthropic": None}):
            from genai_otel.instrumentors.openai_instrumentor import OpenAIInstrumentor

            config = OTelConfig(service_name="test")
            generic = OpenAIInstrumentor()
            generic.instrument(config)
            generic.tracer = _mock_tracer()

            client = fake_openai.OpenAI(base_url="https://api.cometapi.com/v1", api_key="k")
            result = client.chat.completions.create(
                model="gpt-5-mini", messages=[{"role": "user", "content": "hi"}]
            )

            self.assertIs(result, response)
            generic.tracer.start_span.assert_called_once()

    def test_cometapi_client_single_span_via_anthropic_sdk(self):
        response = _anthropic_style_response()
        fake_anthropic = _make_fake_anthropic_module(response)
        with patch.dict(sys.modules, {"openai": None, "anthropic": fake_anthropic}):
            from genai_otel.instrumentors.anthropic_instrumentor import AnthropicInstrumentor
            from genai_otel.instrumentors.cometapi_instrumentor import CometAPIInstrumentor

            config = OTelConfig(service_name="test")
            generic = AnthropicInstrumentor()
            comet = CometAPIInstrumentor()
            generic.instrument(config)
            comet.instrument(config)

            generic.tracer = _mock_tracer()
            comet.tracer = _mock_tracer()

            client = fake_anthropic.Anthropic(base_url="https://api.cometapi.com", api_key="k")
            result = client.messages.create(
                model="claude-sonnet-5",
                max_tokens=100,
                messages=[{"role": "user", "content": "hi"}],
            )

            self.assertIs(result, response)
            comet.tracer.start_span.assert_called_once()
            generic.tracer.start_span.assert_not_called()

    def test_plain_anthropic_client_still_instrumented_by_generic(self):
        response = _anthropic_style_response()
        fake_anthropic = _make_fake_anthropic_module(response)
        with patch.dict(sys.modules, {"openai": None, "anthropic": fake_anthropic}):
            from genai_otel.instrumentors.anthropic_instrumentor import AnthropicInstrumentor
            from genai_otel.instrumentors.cometapi_instrumentor import CometAPIInstrumentor

            config = OTelConfig(service_name="test")
            generic = AnthropicInstrumentor()
            comet = CometAPIInstrumentor()
            generic.instrument(config)
            comet.instrument(config)

            generic.tracer = _mock_tracer()
            comet.tracer = _mock_tracer()

            client = fake_anthropic.Anthropic(base_url=None, api_key="k")
            result = client.messages.create(
                model="claude-sonnet-5",
                max_tokens=100,
                messages=[{"role": "user", "content": "hi"}],
            )

            self.assertIs(result, response)
            generic.tracer.start_span.assert_called_once()
            comet.tracer.start_span.assert_not_called()

    def test_openrouter_client_single_span(self):
        response = _openai_style_response()
        fake_openai = _make_fake_openai_module(response)
        with patch.dict(sys.modules, {"openai": fake_openai, "anthropic": None}):
            from genai_otel.instrumentors.openai_instrumentor import OpenAIInstrumentor
            from genai_otel.instrumentors.openrouter_instrumentor import OpenRouterInstrumentor

            config = OTelConfig(service_name="test")
            generic = OpenAIInstrumentor()
            router = OpenRouterInstrumentor()
            generic.instrument(config)
            router.instrument(config)

            generic.tracer = _mock_tracer()
            router.tracer = _mock_tracer()

            client = fake_openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key="k")
            result = client.chat.completions.create(
                model="meta-llama/llama-3-70b", messages=[{"role": "user", "content": "hi"}]
            )

            self.assertIs(result, response)
            router.tracer.start_span.assert_called_once()
            generic.tracer.start_span.assert_not_called()

    def test_instrument_registers_claim(self):
        """CometAPI/OpenRouter instrument() must register their domain claims."""
        fake_openai = _make_fake_openai_module(_openai_style_response())
        with patch.dict(sys.modules, {"openai": fake_openai, "anthropic": None}):
            from genai_otel.instrumentors.cometapi_instrumentor import CometAPIInstrumentor
            from genai_otel.instrumentors.openrouter_instrumentor import OpenRouterInstrumentor

            config = OTelConfig(service_name="test")
            CometAPIInstrumentor().instrument(config)
            OpenRouterInstrumentor().instrument(config)

            self.assertEqual(find_base_url_claim("https://api.cometapi.com/v1"), "cometapi")
            self.assertEqual(find_base_url_claim("https://openrouter.ai/api/v1"), "openrouter")


if __name__ == "__main__":
    unittest.main()
