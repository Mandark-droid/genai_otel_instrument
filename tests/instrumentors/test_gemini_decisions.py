"""Gemini Decisions API (``gemini-decision-1``): SDK path, REST path, usage, errors, pricing.

The SDK path is driven through a fake client so no ``google-genai`` install and no
network are needed; the REST path goes through ``httpx`` / ``requests`` fakes.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

httpx = pytest.importorskip("httpx")
requests = pytest.importorskip("requests")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode  # noqa: E402

from genai_otel.config import OTelConfig  # noqa: E402
from genai_otel.cost_calculator import CostCalculator  # noqa: E402
from genai_otel.instrumentors import decision_http  # noqa: E402
from genai_otel.instrumentors.decision import (  # noqa: E402
    PROTOCOL_GEMINI_DECISIONS,
    gemini_decision_usage,
    gemini_error_code,
    provider_for_base_url,
    usage_of,
)
from genai_otel.instrumentors.decision_http import (  # noqa: E402
    DecisionHTTPInstrumentor,
    match_decision_path,
)
from genai_otel.instrumentors.google_ai_instrumentor import GoogleAIInstrumentor  # noqa: E402

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/decisions"

QUESTIONS = {
    "is_urgent": {"type": "noul", "instructions": "Is the message urgent?"},
    "topic": {
        "type": "choice",
        "instructions": "What is it about?",
        "criteria": {"billing": "Money", "tech": "Technical", "other": "Anything else"},
    },
    "tone": {
        "type": "score",
        "instructions": "How angry is the writer?",
        "criteria": ["calm", "annoyed", "angry", "furious"],
    },
}
ANSWERS = {
    "is_urgent": {"noul": 0.91, "confidence": 0.8},
    "topic": {
        "choice": "billing",
        "confidence": 0.7,
        "probabilities": {"billing": 0.7, "tech": 0.2, "other": 0.1},
    },
    "tone": {
        "score": 1.6,
        "legend": ["calm", "annoyed", "angry", "furious"],
        "confidence": 0.6,
        "probabilities": {"calm": 0.1, "annoyed": 0.5, "angry": 0.3, "furious": 0.1},
    },
}
USAGE = {
    "total_input_tokens": 1000,
    "input_tokens_by_modality": [
        {"modality": "TEXT", "tokens": 100},
        {"modality": "IMAGE", "tokens": 900},
    ],
    "total_output_tokens": 0,
    "total_tokens": 1000,
}
RESPONSE = {"model": "gemini-decision-1", "answers": ANSWERS, "usage": USAGE}
BODY = {"model": "gemini-decision-1", "input": "My card was charged twice", "questions": QUESTIONS}
SAFETY_ERROR = {"error": {"code": "safety", "message": "blocked"}}

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


_MAPS = ("answers", "probabilities")


def ns(value):
    """A response as attribute-access objects, the way the SDK returns it.

    ``answers`` and ``probabilities`` stay real mappings (name -> object), as in the SDK.
    """
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in _MAPS and isinstance(item, dict):
                out[key] = {k: ns(v) for k, v in item.items()}
            else:
                out[key] = ns(item)
        return SimpleNamespace(**out)
    if isinstance(value, list):
        return [ns(v) for v in value]
    return value


class FakeAPIError(Exception):
    def __init__(self, details):
        super().__init__("400 INVALID_ARGUMENT")
        self.code = 400
        self.details = details


class FakeDecisions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return ns(self.response)


class FakeAsyncDecisions(FakeDecisions):
    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return ns(self.response)


class Harness:
    def __init__(self, capture=False, **config):
        self.exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.instrumentor = GoogleAIInstrumentor()
        self.instrumentor.tracer = provider.get_tracer("test")
        self.instrumentor._instrumented = True
        self.instrumentor.config = OTelConfig(
            service_name="test", enable_content_capture=capture, content_max_length=0, **config
        )

    def client(self, response=RESPONSE, error=None, aio=False):
        client = SimpleNamespace(decisions=FakeDecisions(response, error))
        if aio:
            client.aio = SimpleNamespace(decisions=FakeAsyncDecisions(response, error))
        self.instrumentor._instrument_client(client)
        return client

    def span(self):
        (span,) = self.exporter.get_finished_spans()
        return span

    def attrs(self):
        return dict(self.span().attributes)


def _decide(client, questions=None):
    return client.decisions.create(
        model="gemini-decision-1",
        input="My card was charged twice",
        questions=questions or QUESTIONS,
    )


class TestSdkPath:
    def test_capture_off_records_shape_tokens_and_cost_only(self):
        h = Harness(capture=False)
        result = _decide(h.client())
        attrs = h.attrs()

        assert result.answers["is_urgent"].noul == 0.91
        assert h.span().name == "google.genai.decisions"
        assert attrs["openinference.span.kind"] == "DECISION"
        assert attrs["decision.system"] == "google"
        assert attrs["decision.provider"] == "google"
        assert attrs["gen_ai.system"] == "google"
        assert attrs["gen_ai.operation.name"] == "google.genai.decisions"
        assert attrs["gen_ai.request.type"] == "decision"
        assert attrs["decision.model_name"] == "gemini-decision-1"
        assert attrs["genai.decision.protocol"] == PROTOCOL_GEMINI_DECISIONS
        assert attrs["genai.decision.question_count"] == 3
        assert set(attrs["genai.decision.question.types"]) == {"predicate", "choice", "score"}
        assert attrs["genai.decision.refusal_count"] == 0
        assert attrs["decision.token_count.input"] == 1000
        assert attrs["decision.token_count.output"] == 0
        assert attrs["gen_ai.usage.input_tokens"] == 1000
        # Per-modality counts are counts, not content: present with capture off.
        assert attrs["decision.token_count.input.text"] == 100
        assert attrs["decision.token_count.input.image"] == 900
        assert attrs["gen_ai.usage.cost.total"] == pytest.approx(1000 * 0.10 / 1_000_000)
        for key in ANSWER_FIELDS:
            assert key not in attrs, key
        blob = json.dumps({k: str(v) for k, v in attrs.items()})
        assert "billing" not in blob and "is_urgent" not in blob
        assert h.span().status.status_code == StatusCode.OK

    def test_capture_on_records_all_three_question_types(self):
        h = Harness(capture=True)
        _decide(h.client())
        attrs = h.attrs()

        assert set(attrs["genai.decision.question.names"]) == {"is_urgent", "topic", "tone"}
        assert attrs["genai.decision.min_confidence"] == 0.6
        assert attrs["input.mime_type"] == "application/json"
        assert json.loads(attrs["input.value"])["questions"]["topic"]["type"] == "choice"
        assert "output.value" in attrs
        assert attrs["decision.token_count.input.image"] == 900

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("is_urgent", {"genai.decision.probability": 0.91}),
            (
                "topic",
                {
                    "genai.decision.choice": "billing",
                    "genai.decision.confidence": 0.7,
                    "genai.decision.option_count": 3,
                },
            ),
            ("tone", {"genai.decision.score": 1.6, "genai.decision.option_count": 4}),
        ],
    )
    def test_each_question_type_alone(self, name, expected):
        response = {**RESPONSE, "answers": {name: ANSWERS[name]}}
        content_keys = [k for k in expected if k != "genai.decision.option_count"]

        off = Harness(capture=False)
        _decide(off.client(response), {name: QUESTIONS[name]})
        for key in content_keys:
            assert key not in off.attrs(), key

        on = Harness(capture=True)
        _decide(on.client(response), {name: QUESTIONS[name]})
        for key, value in expected.items():
            assert on.attrs()[key] == pytest.approx(value), key

    def test_safety_400_is_an_error_span_with_the_code(self):
        h = Harness(capture=True)
        with pytest.raises(FakeAPIError):
            _decide(h.client(error=FakeAPIError(SAFETY_ERROR)))
        span = h.span()
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes["error.type"] == "safety"
        # Counts come from the request; there are no answers to record.
        assert span.attributes["genai.decision.question_count"] == 3
        assert "genai.decision.choice" not in span.attributes
        assert "genai.decision.refusal_count" not in span.attributes

    def test_other_exceptions_keep_no_error_code(self):
        h = Harness()
        with pytest.raises(RuntimeError):
            _decide(h.client(error=RuntimeError("boom")))
        assert h.span().status.status_code == StatusCode.ERROR
        assert "error.type" not in h.span().attributes

    def test_async_variant_when_present(self):
        h = Harness(capture=False)
        client = h.client(aio=True)
        result = asyncio.run(
            client.aio.decisions.create(model="gemini-decision-1", input="hi", questions=QUESTIONS)
        )
        assert result.answers["topic"].choice == "billing"
        assert h.span().name == "google.genai.aio.decisions"
        assert h.attrs()["decision.token_count.input.image"] == 900

    def test_async_safety_error(self):
        h = Harness()
        client = h.client(aio=True)
        client.aio.decisions.error = FakeAPIError({"error": {"code": "blocklist"}})
        with pytest.raises(FakeAPIError):
            asyncio.run(
                client.aio.decisions.create(model="gemini-decision-1", input="x", questions={})
            )
        assert h.span().attributes["error.type"] == "blocklist"
        assert h.span().status.status_code == StatusCode.ERROR

    def test_noop_without_decisions(self):
        h = Harness()
        client = SimpleNamespace(models=SimpleNamespace())  # an older google-genai
        h.instrumentor._instrument_client(client)
        assert not hasattr(client, "decisions")
        assert h.exporter.get_finished_spans() == ()

    def test_noop_with_decisions_but_no_create(self):
        h = Harness()
        client = SimpleNamespace(decisions=SimpleNamespace())
        h.instrumentor._instrument_client(client)
        assert not hasattr(client.decisions, "create")


# --------------------------------------------------------------------------- #
# Usage and parsers
# --------------------------------------------------------------------------- #


class TestUsage:
    def test_gemini_usage_reads_totals_and_modalities(self):
        usage = gemini_decision_usage(USAGE)
        assert usage.input_tokens == 1000
        assert usage.output_tokens == 0
        assert usage.input_tokens_by_modality == {"text": 100, "image": 900}

    def test_usage_of_dispatches_on_the_gemini_spelling(self):
        assert usage_of(USAGE).input_tokens == 1000
        # The other spellings are unchanged.
        assert usage_of({"input_tokens": 5, "output_tokens": 0}).input_tokens == 5
        assert usage_of({"prompt_tokens": 7}).input_tokens == 7

    def test_modality_may_be_an_enum(self):
        usage = gemini_decision_usage(
            {
                "total_input_tokens": 3,
                "input_tokens_by_modality": [
                    {"modality": SimpleNamespace(value="AUDIO"), "tokens": 3}
                ],
            }
        )
        assert usage.input_tokens_by_modality == {"audio": 3}

    def test_unreported_is_none_not_zero(self):
        usage = gemini_decision_usage({})
        assert usage.input_tokens is None and usage.output_tokens is None

    def test_error_code_helper(self):
        assert gemini_error_code(SAFETY_ERROR) == "safety"
        assert gemini_error_code({"error": {"code": 400}}) is None
        assert gemini_error_code({"error": {"code": "400"}}) is None
        assert gemini_error_code(None) is None

    def test_provider_host(self):
        assert provider_for_base_url(GEMINI_URL) == "google"


# --------------------------------------------------------------------------- #
# REST path
# --------------------------------------------------------------------------- #


@pytest.fixture
def matcher():
    originals = {
        (requests.sessions.Session, "send"): requests.sessions.Session.__dict__["send"],
        (httpx.Client, "send"): httpx.Client.__dict__["send"],
        (httpx.AsyncClient, "send"): httpx.AsyncClient.__dict__["send"],
    }
    try:
        import httpx2

        originals[(httpx2.Client, "send")] = httpx2.Client.__dict__["send"]
        originals[(httpx2.AsyncClient, "send")] = httpx2.AsyncClient.__dict__["send"]
    except ImportError:  # pragma: no cover - httpx2 is optional
        pass
    decision_http._WRAPPED.clear()
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    def install(**config):
        instrumentor = DecisionHTTPInstrumentor()
        instrumentor.tracer = provider.get_tracer("test")
        instrumentor.instrument(OTelConfig(content_max_length=0, **config))
        return instrumentor

    install.exporter = exporter
    install.tracer = provider.get_tracer("test")
    yield install
    for (cls, name), original in originals.items():
        setattr(cls, name, original)
    decision_http._WRAPPED.clear()


def _httpx_client(body, status=200):
    return httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(status, json=body)))


class TestRestMatching:
    @pytest.mark.parametrize(
        "url,protocol",
        [
            (GEMINI_URL, PROTOCOL_GEMINI_DECISIONS),
            (GEMINI_URL + "/", PROTOCOL_GEMINI_DECISIONS),
            ("https://generativelanguage.googleapis.com/v1/decisions", PROTOCOL_GEMINI_DECISIONS),
            (
                "https://GenerativeLanguage.googleapis.com/v1beta/decisions",
                PROTOCOL_GEMINI_DECISIONS,
            ),
            # The host gate: the path alone is not Gemini.
            ("https://example.com/v1beta/decisions", None),
            ("http://localhost:8000/v1beta/decisions", None),
            # OpenAI's behaviour is unchanged.
            ("https://api.openai.com/v1/decisions", "openai_decisions"),
            ("https://example.com/v1/decisions", "openai_decisions"),
            # Other paths on the Gemini host are not decisions.
            ("https://generativelanguage.googleapis.com/v1beta/models/x:generateContent", None),
        ],
    )
    def test_paths(self, url, protocol):
        assert match_decision_path("POST", url) == protocol

    def test_only_post(self):
        assert match_decision_path("GET", GEMINI_URL) is None


class TestRestSpans:
    def test_httpx_capture_off(self, matcher):
        matcher(enable_content_capture=False)
        with _httpx_client(RESPONSE) as client:
            client.post(GEMINI_URL, json=BODY)
        (span,) = matcher.exporter.get_finished_spans()
        attrs = dict(span.attributes)
        assert span.name == "decision.http.google"
        assert attrs["openinference.span.kind"] == "DECISION"
        assert attrs["genai.decision.protocol"] == PROTOCOL_GEMINI_DECISIONS
        assert attrs["decision.system"] == "google"
        assert attrs["decision.provider"] == "google"
        assert attrs["gen_ai.operation.name"] == "google.genai.decisions"
        assert attrs["genai.decision.question_count"] == 3
        assert attrs["decision.token_count.input"] == 1000
        assert attrs["decision.token_count.input.text"] == 100
        assert attrs["decision.token_count.input.image"] == 900
        assert attrs["gen_ai.usage.cost.total"] == pytest.approx(1000 * 0.10 / 1_000_000)
        for key in ANSWER_FIELDS:
            assert key not in attrs, key

    def test_httpx_capture_on(self, matcher):
        matcher(enable_content_capture=True)
        with _httpx_client(RESPONSE) as client:
            client.post(GEMINI_URL, json=BODY)
        attrs = dict(matcher.exporter.get_finished_spans()[0].attributes)
        assert set(attrs["genai.decision.question.names"]) == {"is_urgent", "topic", "tone"}
        assert attrs["genai.decision.min_confidence"] == 0.6

    def test_requests_path(self, matcher):
        matcher()

        class Adapter(requests.adapters.BaseAdapter):
            def send(self, request, **kwargs):
                response = requests.Response()
                response.status_code = 200
                response._content = json.dumps(RESPONSE).encode()
                response.request = request
                response.url = request.url
                return response

            def close(self):
                pass

        session = requests.Session()
        session.mount("https://", Adapter())
        session.post(GEMINI_URL, json=BODY)
        (span,) = matcher.exporter.get_finished_spans()
        assert span.name == "decision.http.google"

    def test_async_httpx(self, matcher):
        matcher()

        async def go():
            transport = httpx.MockTransport(lambda r: httpx.Response(200, json=RESPONSE))
            async with httpx.AsyncClient(transport=transport) as client:
                await client.post(GEMINI_URL, json=BODY)

        asyncio.run(go())
        (span,) = matcher.exporter.get_finished_spans()
        assert span.attributes["decision.token_count.input.image"] == 900

    def test_400_safety_is_an_error_span(self, matcher):
        matcher(enable_content_capture=True)
        with _httpx_client(SAFETY_ERROR, status=400) as client:
            client.post(GEMINI_URL, json=BODY)
        (span,) = matcher.exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes["error.type"] == "safety"
        assert span.attributes["genai.decision.question_count"] == 3
        assert "genai.decision.choice" not in span.attributes

    def test_400_without_a_named_code_falls_back_to_the_status(self, matcher):
        matcher()
        with _httpx_client({"error": {"code": 400, "status": "INVALID_ARGUMENT"}}, 400) as client:
            client.post(GEMINI_URL, json=BODY)
        (span,) = matcher.exporter.get_finished_spans()
        assert span.attributes["error.type"] == "400"

    def test_other_host_is_not_gemini(self, matcher):
        matcher()
        with _httpx_client(RESPONSE) as client:
            client.post("https://example.com/v1beta/decisions", json=BODY)
        assert matcher.exporter.get_finished_spans() == ()

    def test_openai_v1_decisions_unchanged(self, matcher):
        matcher()
        body = {
            "model": "gpt-6-luna",
            "input": "hello",
            "questions": [{"type": "predicate", "name": "spam", "instructions": "Spam?"}],
        }
        response = {
            "model": "gpt-6-luna",
            "answers": [{"type": "predicate", "name": "spam", "probability": 0.02}],
            "usage": {"input_tokens": 50, "output_tokens": 0},
        }
        with _httpx_client(response) as client:
            client.post("https://example.com/v1/decisions", json=body)
        (span,) = matcher.exporter.get_finished_spans()
        assert span.name == "decision.http.openai"
        assert span.attributes["genai.decision.protocol"] == "openai_decisions"
        assert "decision.token_count.input.text" not in span.attributes

    def test_sdk_call_over_http_is_one_span(self, matcher):
        """The SDK path opens the span; the HTTP matcher beneath it leaves it alone."""
        matcher()
        h = Harness()
        h.instrumentor.tracer = matcher.tracer  # one exporter for both layers
        http_client = _httpx_client(RESPONSE)

        class HttpDecisions:
            def create(self, **kwargs):
                return ns(http_client.post(GEMINI_URL, json=BODY).json())

        client = SimpleNamespace(decisions=HttpDecisions())
        h.instrumentor._instrument_client(client)
        _decide(client)
        http_client.close()
        (span,) = matcher.exporter.get_finished_spans()
        assert span.name == "google.genai.decisions"


# --------------------------------------------------------------------------- #
# Pricing
# --------------------------------------------------------------------------- #


class TestPricing:
    @pytest.fixture(scope="class")
    def calc(self):
        return CostCalculator()

    def test_decision_price_is_input_only(self, calc):
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
        assert calc.calculate_cost("gemini-decision-1", usage, "decision") == pytest.approx(0.10)
        assert calc.pricing_source("gemini-decision-1", "decision") == "table"

    def test_dated_ids_resolve(self, calc):
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0}
        cost = calc.calculate_cost("gemini-decision-1-2026-11", usage, "decision")
        assert cost == pytest.approx(0.10)

    def test_it_is_not_in_the_chat_table(self, calc):
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0}
        assert "gemini-decision-1" not in calc.pricing_data.get("chat", {})
        assert calc.calculate_cost("gemini-decision-1", usage, "chat") == 0.0
