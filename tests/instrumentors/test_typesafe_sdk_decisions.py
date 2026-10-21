"""SystemOne DECISION spans through the real ``typesafe-sdk``, over a mock transport.

The same wrapper covers TypeSafe, LiquidAI (``base_url``) and self-hosted
``/v1/systemone`` servers, so each is exercised by pointing the client at it.
"""

import asyncio
import json

import pytest

typesafe_sdk = pytest.importorskip("typesafe_sdk")
httpx2 = pytest.importorskip("httpx2")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from genai_otel.config import OTelConfig  # noqa: E402
from genai_otel.instrumentors.typesafe_instrumentor import TypeSafeInstrumentor  # noqa: E402

RESPONSE = {
    "model": "jev-1.13.0-20260917",
    "answers": {"urgent": {"type": "noul", "noul": 0.91}},
    "usage": {"input_tokens": 342, "output_tokens": 31},
}
QUESTIONS = {"urgent": typesafe_sdk.Noul(instructions="Does this need attention today?")}


@pytest.fixture(autouse=True)
def unwrap_typesafe():
    """Undo the class-level wrap, so each test instruments a clean SDK."""
    classes = (typesafe_sdk.TypeSafeClient, typesafe_sdk.AsyncTypeSafeClient)
    originals = {cls: cls.__dict__["system_one"] for cls in classes}
    yield
    for cls, original in originals.items():
        setattr(cls, "system_one", original)


class Harness:
    def __init__(self, capture=False, legacy=False, base_url=None, body=None):
        self.exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.instrumentor = TypeSafeInstrumentor()
        self.instrumentor.tracer = provider.get_tracer("test")
        self.instrumentor.instrument(
            OTelConfig(
                enable_content_capture=capture,
                content_max_length=0,
                decision_legacy_llm_kind=legacy,
            )
        )
        assert self.instrumentor._instrumented
        self.base_url = base_url
        self.body = body or RESPONSE
        self.urls = []

    def _handler(self, request):
        self.urls.append(str(request.url))
        return httpx2.Response(200, json=self.body)

    def _kwargs(self):
        kwargs = {"api_key": "test", "compression": False}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return kwargs

    def client(self):
        return typesafe_sdk.TypeSafeClient(
            transport=httpx2.MockTransport(self._handler), **self._kwargs()
        )

    def async_client(self):
        return typesafe_sdk.AsyncTypeSafeClient(
            transport=httpx2.MockTransport(self._handler), **self._kwargs()
        )

    def attrs(self):
        (span,) = self.exporter.get_finished_spans()
        return dict(span.attributes)


def test_decision_span_with_dual_emitted_llm_attributes():
    h = Harness()
    result = h.client().system_one(state="The export page crashes", questions=QUESTIONS)
    attrs = h.attrs()

    assert result.nouls["urgent"].noul == 0.91
    assert attrs["openinference.span.kind"] == "DECISION"
    assert attrs["decision.system"] == "typesafe"
    assert attrs["decision.provider"] == "typesafe"
    assert attrs["decision.model_name"] == "jev-1.13.0-20260917"
    assert attrs["decision.token_count.input"] == 342
    assert attrs["decision.token_count.output"] == 31
    assert attrs["genai.decision.protocol"] == "systemone"
    assert tuple(attrs["genai.decision.question.types"]) == ("predicate",)
    assert attrs["genai.decision.question_count"] == 1
    assert attrs["gen_ai.operation.name"] == "typesafe.system_one"
    assert attrs["gen_ai.request.type"] == "decision"
    # Kept for every reader of pre-1.38 Jev spans, until 2.0.
    assert attrs["llm.request.model_name"] == "jev-latest"
    assert attrs["llm.response.model_name"] == "jev-1.13.0-20260917"
    assert attrs["llm.token_count.prompt"] == 342
    assert attrs["llm.token_count.completion"] == 31
    assert "llm.invocation_parameters" in attrs
    # Priced from the decision section: input only.
    assert attrs["gen_ai.usage.cost.decision"] == pytest.approx(342 * 0.042 / 1_000_000)
    assert attrs["gen_ai.usage.cost.total"] == attrs["gen_ai.usage.cost.decision"]
    # Capture is off: nothing about the answer.
    assert "genai.decision.probability" not in attrs
    assert "genai.decision.question.names" not in attrs
    assert "output.value" not in attrs


def test_positional_state_with_capture_on():
    h = Harness(capture=True)
    h.client().system_one("The export page crashes", QUESTIONS)
    attrs = h.attrs()
    assert attrs["genai.decision.probability"] == 0.91
    assert tuple(attrs["genai.decision.question.names"]) == ("urgent",)
    assert json.loads(attrs["input.value"])["state"] == "The export page crashes"
    assert json.loads(attrs["output.value"])["answers"]["urgent"]["noul"] == 0.91


def test_legacy_flag():
    h = Harness(legacy=True)
    h.client().system_one(state="x", questions=QUESTIONS)
    attrs = h.attrs()
    assert attrs["openinference.span.kind"] == "LLM"
    assert attrs["decision.system"] == "typesafe"
    assert attrs["genai.decision.protocol"] == "systemone"


def test_liquid_base_url_is_the_provider():
    h = Harness(base_url="https://api.liquid.ai/decisions", body=dict(RESPONSE, model="d1"))
    h.client().system_one(state="x", questions=QUESTIONS, model="d1")
    attrs = h.attrs()
    assert h.urls[0] == "https://api.liquid.ai/decisions/v1/systemone"
    assert attrs["decision.provider"] == "liquid"
    assert attrs["gen_ai.usage.cost.decision"] == pytest.approx(342 * 0.02 / 1_000_000)


def test_self_hosted_hub_model_has_no_provider_and_no_price():
    h = Harness(base_url="http://localhost:30000", body=dict(RESPONSE, model="autotrust/JEV-27B"))
    h.client().system_one(state="x", questions=QUESTIONS, model="autotrust/JEV-27B")
    attrs = h.attrs()
    assert "decision.provider" not in attrs
    assert "gen_ai.usage.cost.total" not in attrs
    assert attrs["gen_ai.usage.cost.pricing_source"] == "unpriced"


def test_choice_and_score_answers():
    body = {
        "model": "jev-1",
        "answers": {
            "team": {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.7,
                "probabilities": {"billing": 0.8, "login": 0.2},
            },
            "sev": {
                "type": "score",
                "score": 1.2,
                "confidence": 0.55,
                "legend": {"0": "low", "1": "mid", "2": "high"},
                "probabilities": {"0": 0.2, "1": 0.4, "2": 0.4},
            },
        },
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }
    h = Harness(capture=True, body=body)
    h.client().system_one(
        state="x",
        questions={
            "team": typesafe_sdk.Choice(criteria={"billing": None, "login": None}),
            "sev": typesafe_sdk.Score(criteria=["low", "mid", "high"]),
        },
    )
    attrs = h.attrs()
    assert tuple(attrs["genai.decision.question.types"]) == ("choice", "score")
    assert attrs["genai.decision.min_confidence"] == 0.55
    assert "genai.decision.choice" not in attrs


def test_async():
    h = Harness(capture=True)

    async def run():
        return await h.async_client().system_one(state="x", questions=QUESTIONS)

    asyncio.run(run())
    attrs = h.attrs()
    assert attrs["openinference.span.kind"] == "DECISION"
    assert attrs["genai.decision.probability"] == 0.91
