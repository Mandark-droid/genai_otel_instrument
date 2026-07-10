"""Regression tests for provider-instrumentor hardening fixes.

Covers the client-breaking bugs fixed on the perf-security-hardening branch:

a. Groq   - wrapped __init__ must return None (not the instance) so constructing
            groq.Groq() does not raise "TypeError: __init__() should return None".
b. SambaNova - same __init__-returns-instance bug.
c. AWS Bedrock - invoke_model must be the span-wrapper applied to the original
            bound method, not the decorator factory itself (which raises TypeError
            when called).
d. Hyperbolic - CostCalculator.calculate_cost must be called with the real
            (model, usage_dict, call_type) signature, and a non-JSON / streaming
            response body (response.json() raising) must never fail the host's
            otherwise-successful HTTP call.

Each test constructs / calls against a mock SDK and asserts no TypeError and a
single execution of the underlying business call.
"""

import sys
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.aws_bedrock_instrumentor import AWSBedrockInstrumentor
from genai_otel.instrumentors.groq_instrumentor import GroqInstrumentor
from genai_otel.instrumentors.hyperbolic_instrumentor import HyperbolicInstrumentor
from genai_otel.instrumentors.sambanova_instrumentor import SambaNovaInstrumentor


def test_groq_client_construction_no_typeerror_single_exec():
    """a. Constructing an instrumented groq.Groq() must not raise TypeError."""
    counts = {"init": 0, "instrument_client": 0}

    class RealGroq:
        def __init__(self, *args, **kwargs):
            counts["init"] += 1
            self.chat = MagicMock()
            self.chat.completions = MagicMock()
            self.chat.completions.create = MagicMock()

    mock_groq = MagicMock()
    mock_groq.Groq = RealGroq

    with patch.dict(sys.modules, {"groq": mock_groq}):
        instrumentor = GroqInstrumentor()
        original_instrument_client = instrumentor._instrument_client

        def _spy(client):
            counts["instrument_client"] += 1
            return original_instrument_client(client)

        instrumentor._instrument_client = _spy
        instrumentor.instrument(OTelConfig())

        # Must not raise "TypeError: __init__() should return None".
        client = mock_groq.Groq()

        assert isinstance(client, RealGroq)
        assert counts["init"] == 1
        assert counts["instrument_client"] == 1


def test_groq_instrument_is_idempotent():
    """Double-wrap guard: calling instrument() twice does not re-wrap."""

    class RealGroq:
        def __init__(self, *args, **kwargs):
            self.chat = MagicMock()
            self.chat.completions = MagicMock()
            self.chat.completions.create = MagicMock()

    mock_groq = MagicMock()
    mock_groq.Groq = RealGroq

    with patch.dict(sys.modules, {"groq": mock_groq}):
        instrumentor = GroqInstrumentor()
        instrumentor.instrument(OTelConfig())
        wrapped_once = mock_groq.Groq.__init__
        instrumentor.instrument(OTelConfig())
        # Second call must be a no-op (same wrapped __init__, no stacking).
        assert mock_groq.Groq.__init__ is wrapped_once


def test_sambanova_client_construction_no_typeerror_single_exec():
    """b. Constructing an instrumented sambanova.SambaNova() must not raise TypeError."""
    counts = {"init": 0, "instrument_client": 0}

    class RealSambaNova:
        def __init__(self, *args, **kwargs):
            counts["init"] += 1
            self.chat = MagicMock()
            self.chat.completions = MagicMock()
            self.chat.completions.create = MagicMock()

    mock_sambanova = MagicMock()
    mock_sambanova.SambaNova = RealSambaNova

    with patch.dict(sys.modules, {"sambanova": mock_sambanova}):
        instrumentor = SambaNovaInstrumentor()
        original_instrument_client = instrumentor._instrument_client

        def _spy(client):
            counts["instrument_client"] += 1
            return original_instrument_client(client)

        instrumentor._instrument_client = _spy
        instrumentor.instrument(OTelConfig())

        client = mock_sambanova.SambaNova()

        assert isinstance(client, RealSambaNova)
        assert counts["init"] == 1
        assert counts["instrument_client"] == 1


def test_bedrock_invoke_model_callable_single_exec():
    """c. invoke_model must be callable (factory applied to original), single exec."""
    exec_counts = {"n": 0}

    def real_invoke_model(*args, **kwargs):
        exec_counts["n"] += 1
        return {"contentType": "application/json", "body": "{}"}

    mock_client = MagicMock()
    mock_client.invoke_model = real_invoke_model

    instrumentor = AWSBedrockInstrumentor()
    instrumentor.config = OTelConfig()
    instrumentor._instrumented = True

    instrumentor._instrument_bedrock_client(mock_client)

    # Must not raise TypeError (would if the decorator factory were assigned raw).
    result = mock_client.invoke_model(modelId="anthropic.claude-v2", body="{}")

    assert exec_counts["n"] == 1
    assert result == {"contentType": "application/json", "body": "{}"}


def test_hyperbolic_cost_call_uses_correct_signature():
    """d. calculate_cost must be called with (model, usage_dict, call_type)."""
    instrumentor = HyperbolicInstrumentor()
    instrumentor.config = OTelConfig(enable_cost_tracking=True)
    instrumentor.token_counter = MagicMock()
    instrumentor.cost_counter = MagicMock()

    span = MagicMock()
    span.attributes = {"gen_ai.request.model": "Qwen/Qwen3-Next-80B-A3B-Thinking"}

    response_data = {
        "id": "resp-1",
        "model": "Qwen/Qwen3-Next-80B-A3B-Thinking",
        "choices": [{"finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }

    # Previously raised TypeError: unexpected keyword argument 'model_name'.
    instrumentor._extract_and_record_response(span, response_data)

    span.set_attribute.assert_any_call("gen_ai.usage.prompt_tokens", 10)


def test_hyperbolic_non_json_response_does_not_break_host():
    """d. A streaming / non-JSON body must not fail the host's successful call."""

    def fake_post(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = ValueError("streaming body, not JSON")
        return resp

    mock_requests = MagicMock()
    mock_requests.post = fake_post

    with patch.dict(sys.modules, {"requests": mock_requests}):
        instrumentor = HyperbolicInstrumentor()
        instrumentor.instrument(OTelConfig())

        wrapped_post = mock_requests.post  # now the instrumented wrapper

        # response.json() raising inside telemetry must be swallowed; the host
        # still gets its response back.
        resp = wrapped_post(
            "https://api.hyperbolic.xyz/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )

        assert resp.status_code == 200
