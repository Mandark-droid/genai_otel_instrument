"""OpenAI Decisions (``client.decisions.create``) and Granite Switch decision spans.

Driven through the real ``openai`` SDK over an ``httpx.MockTransport``, so the
request and response shapes are the SDK's own.
"""

import asyncio
import json
import math
from types import SimpleNamespace

import pytest

openai = pytest.importorskip("openai")
httpx = pytest.importorskip("httpx")

if not hasattr(openai.OpenAI(api_key="x"), "decisions"):  # pragma: no cover - old openai
    pytest.skip("this openai release has no Decisions resource", allow_module_level=True)

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode  # noqa: E402

from genai_otel.config import EVALUATION_MODE_INLINE, OTelConfig  # noqa: E402
from genai_otel.instrumentors.base import BaseInstrumentor  # noqa: E402
from genai_otel.instrumentors.decision import IMAGE_OMITTED  # noqa: E402
from genai_otel.instrumentors.openai_instrumentor import (  # noqa: E402
    OpenAIInstrumentor,
    wrap_decisions_resource,
)
from genai_otel.instrumentors.openrouter_instrumentor import OpenRouterInstrumentor  # noqa: E402

TOOLS = ["purchaseProduct", "searchCatalog", "trackOrder", "refund", "escalate"]
ROUTER_QUESTION = {
    "type": "choice",
    "name": "next_tool",
    "instructions": "Which tool handles the message?",
    "choices": [{"value": tool} for tool in TOOLS],
}
USAGE = {
    "input_tokens": 403,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens": 0,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 403,
}
CHOICE_ANSWER = {
    "type": "choice",
    "name": "next_tool",
    "choice": "purchaseProduct",
    "confidence": 0.81,
    "probabilities": [
        {"value": "purchaseProduct", "probability": 0.85},
        {"value": "searchCatalog", "probability": 0.12},
        {"value": "trackOrder", "probability": 0.01},
        {"value": "refund", "probability": 0.01},
        {"value": "escalate", "probability": 0.01},
    ],
}

ANSWER_FIELDS = (
    "genai.decision.choice",
    "genai.decision.confidence",
    "genai.decision.min_confidence",
    "genai.decision.margin",
    "genai.decision.probability",
    "genai.decision.score",
    "genai.decision.question.names",
    "input.value",
    "output.value",
)


class Harness:
    """An instrumented OpenAI client whose transport answers with ``body``."""

    def __init__(self, capture=False, max_length=0, base_url=None, status=200, **config):
        self.exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.instrumentor = OpenAIInstrumentor()
        self.instrumentor.tracer = provider.get_tracer("test")
        self.instrumentor._instrumented = True
        self.instrumentor.config = OTelConfig(
            service_name="test",
            enable_content_capture=capture,
            content_max_length=max_length,
            **config,
        )
        self.base_url = base_url
        self.status = status
        self.body = {"answers": [CHOICE_ANSWER], "model": "gpt-6-luna", "usage": USAGE}
        self.requests = []

    def _handler(self, request):
        self.requests.append(request)
        return httpx.Response(self.status, json=self.body)

    def client(self):
        kwargs = {"api_key": "test", "max_retries": 0}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = openai.OpenAI(
            http_client=httpx.Client(transport=httpx.MockTransport(self._handler)), **kwargs
        )
        self.instrumentor._instrument_client(client)
        return client

    def async_client(self):
        kwargs = {"api_key": "test", "max_retries": 0}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = openai.AsyncOpenAI(
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self._handler)), **kwargs
        )
        self.instrumentor._instrument_async_client(client)
        return client

    def span(self):
        (span,) = self.exporter.get_finished_spans()
        return span

    def attrs(self):
        return dict(self.span().attributes)


def _decide(client, questions=None, **kwargs):
    return client.decisions.create(
        model="gpt-6-luna",
        input="I'd like to buy the blue trainers",
        questions=questions or [ROUTER_QUESTION],
        **kwargs,
    )


class TestSpanContract:
    def test_capture_off_proves_a_decision_and_records_nothing_about_it(self):
        h = Harness(capture=False)
        result = _decide(h.client())
        attrs = h.attrs()

        assert result.answers[0].choice == "purchaseProduct"
        assert h.span().name == "openai.decisions"
        assert attrs["openinference.span.kind"] == "DECISION"
        assert attrs["decision.system"] == "openai"
        assert attrs["decision.provider"] == "openai"
        assert attrs["decision.model_name"] == "gpt-6-luna"
        assert attrs["decision.token_count.input"] == 403
        assert attrs["decision.token_count.output"] == 0
        assert attrs["gen_ai.system"] == "openai"
        assert attrs["gen_ai.operation.name"] == "openai.decisions"
        assert attrs["gen_ai.request.type"] == "decision"
        assert attrs["gen_ai.request.model"] == "gpt-6-luna"
        assert attrs["gen_ai.usage.input_tokens"] == 403
        assert attrs["gen_ai.usage.output_tokens"] == 0
        assert attrs["gen_ai.usage.cost.total"] == pytest.approx(403 * 0.10 / 1_000_000)
        assert attrs["gen_ai.usage.cost.decision"] == attrs["gen_ai.usage.cost.total"]
        assert attrs["genai.decision.protocol"] == "openai_decisions"
        assert attrs["genai.decision.question_count"] == 1
        assert attrs["genai.decision.refusal_count"] == 0
        assert tuple(attrs["genai.decision.question.types"]) == ("choice",)
        assert attrs["genai.decision.option_count"] == 5
        for key in ANSWER_FIELDS:
            assert key not in attrs, key
        # The upstream spec forbids llm.* on DECISION spans.
        assert not any(key.startswith("llm.") for key in attrs)
        # The decision is nowhere in the span's text either.
        assert "purchaseProduct" not in json.dumps({k: str(v) for k, v in attrs.items()})
        assert h.span().status.status_code == StatusCode.OK

    def test_capture_on_records_the_decision(self):
        h = Harness(capture=True)
        _decide(h.client())
        attrs = h.attrs()

        assert tuple(attrs["genai.decision.question.names"]) == ("next_tool",)
        assert attrs["genai.decision.choice"] == "purchaseProduct"
        assert attrs["genai.decision.confidence"] == 0.81
        assert attrs["genai.decision.min_confidence"] == 0.81
        assert attrs["genai.decision.margin"] == pytest.approx(0.73)
        assert attrs["input.mime_type"] == "application/json"
        assert json.loads(attrs["input.value"])["input"] == "I'd like to buy the blue trainers"
        assert json.loads(attrs["output.value"])["answers"][0]["choice"] == "purchaseProduct"

    def test_capture_truncated(self):
        h = Harness(capture=True, max_length=40)
        _decide(h.client())
        attrs = h.attrs()

        assert len(attrs["input.value"]) == 40
        assert len(attrs["output.value"]) == 40
        # The summary attributes are what survive the cap.
        assert attrs["genai.decision.choice"] == "purchaseProduct"

    def test_images_are_omitted(self):
        h = Harness(capture=True)
        h.client().decisions.create(
            model="gpt-6-luna",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Is the parcel damaged?"},
                        {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0K"},
                    ],
                }
            ],
            questions=[{"type": "predicate", "name": "damaged", "instructions": "Damaged?"}],
        )
        value = h.attrs()["input.value"]
        assert "iVBORw0K" not in value
        assert IMAGE_OMITTED in value
        assert "Is the parcel damaged?" in value


class TestQuestionTypes:
    def test_predicate(self):
        h = Harness(capture=True)
        h.body["answers"] = [{"type": "predicate", "name": "fraud", "probability": 0.07}]
        _decide(h.client(), [{"type": "predicate", "name": "fraud", "instructions": "Fraud?"}])
        attrs = h.attrs()
        assert attrs["genai.decision.probability"] == 0.07
        assert tuple(attrs["genai.decision.question.types"]) == ("predicate",)
        assert "genai.decision.min_confidence" not in attrs
        assert "genai.decision.option_count" not in attrs

    def test_score(self):
        h = Harness(capture=True)
        h.body["answers"] = [
            {
                "type": "score",
                "name": "urgency",
                "score": 1.4,
                "confidence": 0.62,
                "probabilities": [
                    {"value": 0, "label": "low", "probability": 0.1},
                    {"value": 1, "label": "mid", "probability": 0.4},
                    {"value": 2, "label": "high", "probability": 0.5},
                ],
            }
        ]
        _decide(
            h.client(),
            [
                {
                    "type": "score",
                    "name": "urgency",
                    "instructions": "How urgent?",
                    "levels": [{"label": "low"}, {"label": "mid"}, {"label": "high"}],
                }
            ],
        )
        attrs = h.attrs()
        assert attrs["genai.decision.score"] == 1.4
        assert attrs["genai.decision.confidence"] == 0.62
        assert attrs["genai.decision.option_count"] == 3
        assert attrs["genai.decision.margin"] == pytest.approx(0.1)

    def test_refusal(self):
        h = Harness(capture=True)
        h.body["answers"] = [{"type": "refusal", "name": "next_tool"}]
        _decide(h.client())
        attrs = h.attrs()
        assert attrs["genai.decision.refusal_count"] == 1
        assert tuple(attrs["genai.decision.question.types"]) == ("refusal",)
        assert "genai.decision.choice" not in attrs

    def test_several_questions(self):
        h = Harness(capture=True)
        h.body["answers"] = [
            CHOICE_ANSWER,
            {"type": "predicate", "name": "fraud", "probability": 0.3},
            {"type": "refusal", "name": None},
        ]
        _decide(
            h.client(),
            [
                ROUTER_QUESTION,
                {"type": "predicate", "name": "fraud", "instructions": "Fraud?"},
                {"type": "predicate", "instructions": "Unnamed"},
            ],
        )
        attrs = h.attrs()
        assert attrs["genai.decision.question_count"] == 3
        assert attrs["genai.decision.refusal_count"] == 1
        assert tuple(attrs["genai.decision.question.types"]) == ("choice", "predicate", "refusal")
        assert tuple(attrs["genai.decision.question.names"]) == ("next_tool", "fraud")
        assert attrs["genai.decision.min_confidence"] == 0.81
        assert "genai.decision.choice" not in attrs
        assert "genai.decision.option_count" not in attrs


class TestUsageAndCost:
    def test_no_usage_means_no_tokens_and_no_cost(self):
        h = Harness()
        del h.body["usage"]
        _decide(h.client())
        attrs = h.attrs()
        assert not any("token" in key for key in attrs)
        assert not any(key.startswith("gen_ai.usage.cost") for key in attrs)

    def test_an_unpriced_model_gets_no_cost_never_chat_pricing(self):
        h = Harness(base_url="http://localhost:8000/v1")
        h.body["model"] = "my-local-decider-8b"
        h.client().decisions.create(
            model="my-local-decider-8b", input="x", questions=[ROUTER_QUESTION]
        )
        attrs = h.attrs()
        assert "gen_ai.usage.cost.total" not in attrs
        assert "gen_ai.usage.cost.decision" not in attrs
        assert attrs["gen_ai.usage.cost.pricing_source"] == "unpriced"
        assert "decision.provider" not in attrs

    def test_cached_input_is_free(self):
        h = Harness()
        h.body["usage"] = dict(
            USAGE, input_tokens_details={"cached_tokens": 400, "cache_write_tokens": 0}
        )
        _decide(h.client())
        assert h.attrs()["gen_ai.usage.cost.decision"] == pytest.approx(3 * 0.10 / 1_000_000)

    def test_custom_pricing_prices_an_on_prem_decision_model(self):
        h = Harness(
            base_url="http://localhost:8000/v1",
            custom_pricing_json=json.dumps({"decision": {"my-decider": {"promptPrice": 0.001}}}),
        )
        h.body["model"] = "my-decider"
        h.client().decisions.create(model="my-decider", input="x", questions=[ROUTER_QUESTION])
        # promptPrice is per 1,000 tokens, as everywhere in the pricing file.
        assert h.attrs()["gen_ai.usage.cost.decision"] == pytest.approx(403 / 1000 * 0.001)


class TestProvider:
    def test_from_the_env_var_for_an_unknown_host(self):
        h = Harness(base_url="http://vllm.internal:8000/v1", decision_provider="nebius")
        _decide(h.client())
        assert h.attrs()["decision.provider"] == "nebius"

    def test_recognised_host_wins_over_the_env_var(self):
        h = Harness(decision_provider="nebius")
        _decide(h.client())
        assert h.attrs()["decision.provider"] == "openai"

    def test_azure_host(self):
        h = Harness(base_url="https://myres.openai.azure.com/openai/v1")
        _decide(h.client())
        assert h.attrs()["decision.provider"] == "azure"


class TestCallPaths:
    def test_async(self):
        h = Harness(capture=True)

        async def run():
            client = h.async_client()
            return await client.decisions.create(
                model="gpt-6-luna", input="hello", questions=[ROUTER_QUESTION]
            )

        result = asyncio.run(run())
        attrs = h.attrs()
        assert result.model == "gpt-6-luna"
        assert attrs["openinference.span.kind"] == "DECISION"
        assert attrs["genai.decision.choice"] == "purchaseProduct"
        assert attrs["gen_ai.usage.cost.decision"] > 0

    def test_with_raw_response(self):
        h = Harness(capture=True)
        raw = h.client().decisions.with_raw_response.create(
            model="gpt-6-luna", input="hello", questions=[ROUTER_QUESTION]
        )
        assert raw.parse().answers[0].choice == "purchaseProduct"
        attrs = h.attrs()
        assert attrs["genai.decision.choice"] == "purchaseProduct"
        assert attrs["decision.token_count.input"] == 403

    def test_async_with_raw_response(self):
        h = Harness()

        async def run():
            client = h.async_client()
            raw = await client.decisions.with_raw_response.create(
                model="gpt-6-luna", input="hello", questions=[ROUTER_QUESTION]
            )
            return raw.parse()

        assert asyncio.run(run()).answers[0].type == "choice"
        assert h.attrs()["decision.token_count.input"] == 403

    def test_api_error_gives_status_error(self):
        h = Harness(status=500)
        with pytest.raises(openai.APIStatusError):
            _decide(h.client())
        span = h.span()
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes["openinference.span.kind"] == "DECISION"

    def test_a_client_without_decisions_is_not_wrapped(self):
        client = SimpleNamespace(base_url="https://api.openai.com/v1")
        assert wrap_decisions_resource(OpenAIInstrumentor(), client, "openai.decisions") is False
        client.decisions = SimpleNamespace(create=None)
        assert wrap_decisions_resource(OpenAIInstrumentor(), client, "openai.decisions") is False

    def test_an_unreadable_request_still_makes_the_call(self, monkeypatch):
        from genai_otel.instrumentors import openai_instrumentor as module

        h = Harness()
        client = h.client()
        monkeypatch.setattr(
            module.OpenAIDecisionsTracer,
            "decisions_request",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert _decide(client).answers
        assert h.exporter.get_finished_spans() == ()

    def test_openrouter_client(self):
        h = Harness(base_url="https://openrouter.ai/api/v1")
        router = OpenRouterInstrumentor()
        router.tracer = h.instrumentor.tracer
        router.config = h.instrumentor.config
        router._instrumented = True
        client = openai.OpenAI(
            api_key="test",
            base_url="https://openrouter.ai/api/v1",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(h._handler)),
        )
        router._instrument_client(client)
        h.body["model"] = "openai/gpt-6-luna-decisions"
        client.decisions.create(
            model="openai/gpt-6-luna-decisions", input="x", questions=[ROUTER_QUESTION]
        )
        span = h.span()
        assert span.name == "openrouter.decisions"
        assert span.attributes["decision.provider"] == "openrouter"
        assert span.attributes["gen_ai.usage.cost.decision"] == pytest.approx(403 * 0.1 / 1e6)
        assert str(h.requests[0].url) == "https://openrouter.ai/api/v1/decisions"


class TestMetrics:
    def test_decisions_are_their_own_dimension(self):
        from unittest.mock import MagicMock

        h = Harness()
        client = h.client()
        _decide(client)
        tracer = h.instrumentor._genai_decision_tracer
        tracer.inference_usage_input_counter = MagicMock()
        tracer.token_counter = MagicMock()
        tracer.latency_histogram = MagicMock()
        _decide(client)

        (call,) = tracer.inference_usage_input_counter.add.call_args_list
        assert call.args[0] == 403
        assert call.args[1]["gen_ai.operation.name"] == "openai.decisions"
        assert call.args[1]["gen_ai.request.model"] == "gpt-6-luna"
        token_call = tracer.token_counter.add.call_args_list[0]
        assert token_call.args[1]["operation"] == "openai.decisions"
        assert tracer.latency_histogram.record.call_args.args[1] == {
            "operation": "openai.decisions"
        }


class TestEvaluation:
    def test_input_is_screened_and_the_response_checks_say_why_they_did_not_run(self):
        seen = []

        def detect(text):
            seen.append(text)
            return SimpleNamespace(
                has_pii=False, entity_counts={}, entities=[], score=0.0, redacted_text=None
            )

        class Hallucination:
            def detect(self, *a, **k):  # pragma: no cover - must not run
                raise AssertionError("hallucination ran on a decision")

        BaseInstrumentor._pii_detector = SimpleNamespace(detect=detect)
        BaseInstrumentor._hallucination_detector = Hallucination()
        h = Harness(evaluation_mode=EVALUATION_MODE_INLINE)
        _decide(h.client())
        attrs = h.attrs()

        assert seen and "blue trainers" in seen[0]
        assert attrs["evaluation.skipped_reason"] == "decision_span"
        assert attrs["evaluation.pii.prompt.detected"] is False
        assert not any(key.startswith("evaluation.hallucination") for key in attrs)


class TestGraniteSwitch:
    def _chat_body(self, logprobs=True):
        top = [
            {"token": "A", "logprob": math.log(0.982), "bytes": None},
            {"token": "B", "logprob": math.log(0.0035), "bytes": None},
            {"token": "C", "logprob": math.log(0.014), "bytes": None},
        ]
        choice = {
            "index": 0,
            "finish_reason": "length",
            "message": {"role": "assistant", "content": "A"},
            "logprobs": (
                {"content": [{"token": "A", "logprob": -0.01, "bytes": None, "top_logprobs": top}]}
                if logprobs
                else None
            ),
        }
        return {
            "id": "c1",
            "object": "chat.completion",
            "created": 1,
            "model": "ibm-granite/granite-switch-4.1-8b",
            "choices": [choice],
            "usage": {"prompt_tokens": 61, "completion_tokens": 1, "total_tokens": 62},
        }

    def _call(self, h, adapter="decision", label_ids=(32, 33, 34)):
        extra_body = {"chat_template_kwargs": {"adapter_name": adapter}}
        if label_ids:
            extra_body["logprob_token_ids"] = list(label_ids)
        return h.client().chat.completions.create(
            model="ibm-granite/granite-switch-4.1-8b",
            messages=[
                {"role": "user", "content": "State:\nMy payouts failed."},
                {
                    "role": "user",
                    "content": "### DECISION\nOptions:\nA: Payouts\nB: Login\nC: Other",
                },
            ],
            extra_body=extra_body,
            max_completion_tokens=1,
            logprobs=True,
            top_logprobs=1,
        )

    def test_decision_adapter_is_a_decision_span(self):
        h = Harness(capture=True, base_url="http://localhost:8000/v1")
        h.body = self._chat_body()
        self._call(h)
        span = h.span()
        attrs = dict(span.attributes)

        assert span.name == "openai.chat.decision"
        assert attrs["openinference.span.kind"] == "DECISION"
        assert attrs["genai.decision.protocol"] == "granite_adapter"
        assert attrs["genai.decision.option_count"] == 3
        assert attrs["genai.decision.choice"] == "A"
        assert attrs["genai.decision.margin"] > 0.9
        assert attrs["decision.token_count.input"] == 61
        assert "decision.provider" not in attrs
        # Self-hosted: no list price, and never the chat table's size estimate.
        assert "gen_ai.usage.cost.total" not in attrs

    def test_without_logprobs_the_span_carries_counts_only(self):
        h = Harness(capture=True, base_url="http://localhost:8000/v1")
        h.body = self._chat_body(logprobs=False)
        self._call(h, label_ids=None)
        attrs = h.attrs()
        assert attrs["genai.decision.question_count"] == 1
        assert "genai.decision.choice" not in attrs
        assert "genai.decision.option_count" not in attrs

    def test_other_adapters_stay_chat_spans(self):
        h = Harness(base_url="http://localhost:8000/v1")
        h.body = self._chat_body()
        self._call(h, adapter="guardian-core", label_ids=None)
        span = h.span()
        assert span.name == "openai.chat.completion"
        assert span.attributes.get("openinference.span.kind") != "DECISION"
        assert span.attributes["gen_ai.operation.name"] == "chat"

    def test_async_decision_adapter(self):
        h = Harness(base_url="http://localhost:8000/v1")
        h.body = self._chat_body()

        async def run():
            client = h.async_client()
            return await client.chat.completions.create(
                model="ibm-granite/granite-switch-4.1-8b",
                messages=[{"role": "user", "content": "x"}],
                extra_body={"chat_template_kwargs": {"adapter_name": "decision"}},
            )

        asyncio.run(run())
        assert h.attrs()["genai.decision.protocol"] == "granite_adapter"

    def test_an_unreadable_granite_request_falls_back_to_chat(self, monkeypatch):
        from genai_otel.instrumentors import openai_instrumentor as module

        h = Harness(base_url="http://localhost:8000/v1")
        h.body = self._chat_body()
        monkeypatch.setattr(
            module.OpenAIDecisionsTracer,
            "granite_request",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        self._call(h)
        assert h.span().name == "openai.chat.completion"
