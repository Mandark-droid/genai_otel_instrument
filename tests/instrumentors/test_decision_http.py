"""The HTTP path matcher: decision endpoints called with plain requests / httpx."""

import asyncio
import json

import pytest

requests = pytest.importorskip("requests")
httpx = pytest.importorskip("httpx")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode  # noqa: E402

from genai_otel.config import OTelConfig  # noqa: E402
from genai_otel.instrumentors import decision_http  # noqa: E402
from genai_otel.instrumentors.decision_http import (  # noqa: E402
    DecisionHTTPInstrumentor,
    match_decision_path,
)

SYSTEMONE_BODY = {
    "model": "d1",
    "state": "Photo of a parcel",
    "questions": {"damaged": {"type": "noul", "instructions": "Is it damaged?"}},
}
SYSTEMONE_RESPONSE = {
    "model": "d1",
    "answers": {"damaged": {"type": "noul", "noul": 0.12}},
    "usage": {"input_tokens": 900, "output_tokens": 0},
}
OPENAI_BODY = {
    "model": "gpt-6-luna",
    "input": "hello",
    "questions": [{"type": "predicate", "name": "spam", "instructions": "Spam?"}],
}
OPENAI_RESPONSE = {
    "model": "gpt-6-luna",
    "answers": [{"type": "predicate", "name": "spam", "probability": 0.02}],
    "usage": {"input_tokens": 50, "output_tokens": 0},
}


@pytest.fixture
def matcher():
    """A matcher wrapped onto requests and httpx for one test, then removed."""
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


class _Adapter(requests.adapters.BaseAdapter):
    def __init__(self, status=200, body=None, raw_text=None):
        super().__init__()
        self.status = status
        self.body = body
        self.raw_text = raw_text

    def send(self, request, **kwargs):
        response = requests.Response()
        response.status_code = self.status
        response._content = (
            self.raw_text.encode() if self.raw_text is not None else json.dumps(self.body).encode()
        )
        response.request = request
        response.url = request.url
        return response

    def close(self):
        pass


def _session(adapter):
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _spans(matcher):
    return matcher.exporter.get_finished_spans()


class TestPathMatching:
    @pytest.mark.parametrize(
        "url,protocol",
        [
            ("https://api.liquid.ai/decisions/v1/systemone", "systemone"),
            ("http://localhost:30000/v1/systemone", "systemone"),
            ("http://localhost:30000/v1/systemone/", "systemone"),
            ("https://api.openai.com/v1/decisions", "openai_decisions"),
            ("https://openrouter.ai/api/v1/decisions", "openai_decisions"),
            ("http://localhost:4318/v1/traces", None),
            ("https://api.openai.com/v1/chat/completions", None),
            (None, None),
        ],
    )
    def test_post_paths(self, url, protocol):
        assert match_decision_path("POST", url) == protocol

    def test_only_post(self):
        assert match_decision_path("GET", "https://api.openai.com/v1/decisions") is None
        assert match_decision_path(None, "https://api.openai.com/v1/decisions") is None


class TestRequests:
    def test_liquid_image_example_shape(self, matcher):
        matcher(enable_content_capture=False)
        session = _session(_Adapter(body=SYSTEMONE_RESPONSE))
        response = session.post("https://api.liquid.ai/decisions/v1/systemone", json=SYSTEMONE_BODY)

        assert response.json()["answers"]["damaged"]["noul"] == 0.12
        (span,) = _spans(matcher)
        attrs = dict(span.attributes)
        assert span.name == "decision.http.systemone"
        assert attrs["openinference.span.kind"] == "DECISION"
        assert attrs["decision.provider"] == "liquid"
        assert attrs["decision.system"] == "typesafe"
        assert attrs["genai.decision.protocol"] == "systemone"
        assert attrs["decision.token_count.input"] == 900
        assert attrs["gen_ai.usage.cost.decision"] == pytest.approx(900 * 0.02 / 1e6)
        assert tuple(attrs["genai.decision.question.types"]) == ("predicate",)
        assert "genai.decision.probability" not in attrs
        assert span.status.status_code == StatusCode.OK

    def test_capture_on(self, matcher):
        matcher(enable_content_capture=True)
        _session(_Adapter(body=SYSTEMONE_RESPONSE)).post(
            "https://api.liquid.ai/decisions/v1/systemone", json=SYSTEMONE_BODY
        )
        (span,) = _spans(matcher)
        assert span.attributes["genai.decision.probability"] == 0.12
        assert json.loads(span.attributes["input.value"])["state"] == "Photo of a parcel"

    def test_http_error_is_status_error(self, matcher):
        matcher()
        response = _session(_Adapter(status=429, body={"error": "slow down"})).post(
            "http://localhost:30000/v1/systemone", json=SYSTEMONE_BODY
        )
        assert response.status_code == 429
        (span,) = _spans(matcher)
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes["error.type"] == "429"
        assert "decision.token_count.input" not in span.attributes

    def test_a_transport_exception_is_recorded_and_raised(self, matcher):
        matcher()

        class Failing(_Adapter):
            def send(self, request, **kwargs):
                raise requests.ConnectionError("refused")

        with pytest.raises(requests.ConnectionError):
            _session(Failing()).post("http://localhost:30000/v1/systemone", json=SYSTEMONE_BODY)
        (span,) = _spans(matcher)
        assert span.status.status_code == StatusCode.ERROR

    def test_a_non_json_body_still_makes_a_decision_span(self, matcher):
        matcher()
        _session(_Adapter(raw_text="not json")).post(
            "http://localhost:30000/v1/systemone", data=b"\xff\xfe"
        )
        (span,) = _spans(matcher)
        assert span.attributes["genai.decision.question_count"] == 0
        assert "decision.token_count.input" not in span.attributes

    def test_other_traffic_passes_through_untouched(self, matcher):
        matcher()
        _session(_Adapter(body={"ok": True})).post("http://localhost:4318/v1/traces", data=b"x")
        assert _spans(matcher) == ()

    def test_disabled(self, matcher):
        instrumentor = matcher(decision_http_matcher=False)
        assert instrumentor._instrumented is False
        _session(_Adapter(body=SYSTEMONE_RESPONSE)).post(
            "http://localhost:30000/v1/systemone", json=SYSTEMONE_BODY
        )
        assert _spans(matcher) == ()

    def test_instrument_is_idempotent(self, matcher):
        matcher()
        matcher()
        _session(_Adapter(body=SYSTEMONE_RESPONSE)).post(
            "http://localhost:30000/v1/systemone", json=SYSTEMONE_BODY
        )
        assert len(_spans(matcher)) == 1

    def test_an_unread_stream_is_not_consumed(self, matcher):
        matcher(enable_content_capture=True)

        class Streaming(_Adapter):
            def send(self, request, **kwargs):
                response = super().send(request, **kwargs)
                response._content = False
                response.raw = None
                return response

        _session(Streaming(body=SYSTEMONE_RESPONSE)).post(
            "http://localhost:30000/v1/systemone", json=SYSTEMONE_BODY, stream=True
        )
        (span,) = _spans(matcher)
        assert "genai.decision.probability" not in span.attributes


class TestHttpx:
    def _client(self, body, status=200):
        return httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(status, json=body))
        )

    def test_openai_decisions_over_plain_httpx(self, matcher):
        matcher(enable_content_capture=True)
        self._client(OPENAI_RESPONSE).post("https://api.openai.com/v1/decisions", json=OPENAI_BODY)
        (span,) = _spans(matcher)
        attrs = dict(span.attributes)
        assert span.name == "decision.http.openai"
        assert attrs["decision.system"] == "openai"
        assert attrs["genai.decision.protocol"] == "openai_decisions"
        assert attrs["genai.decision.probability"] == 0.02
        assert attrs["gen_ai.usage.cost.decision"] == pytest.approx(50 * 0.1 / 1e6)

    def test_async(self, matcher):
        matcher()

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json=SYSTEMONE_RESPONSE)
                )
            ) as client:
                return await client.post("http://localhost:30000/v1/systemone", json=SYSTEMONE_BODY)

        assert asyncio.run(run()).status_code == 200
        (span,) = _spans(matcher)
        assert span.attributes["decision.token_count.input"] == 900

    def test_async_other_traffic_passes_through(self, matcher):
        matcher()

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))
            ) as client:
                return await client.get("http://localhost:30000/v1/systemone")

        asyncio.run(run())
        assert _spans(matcher) == ()

    def test_sync_other_traffic_passes_through(self, matcher):
        matcher()
        self._client({}).post("https://api.openai.com/v1/chat/completions", json={})
        assert _spans(matcher) == ()

    def test_an_unread_streamed_response(self, matcher):
        matcher(enable_content_capture=True)
        chunks = [json.dumps(SYSTEMONE_RESPONSE).encode()]
        client = httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=iter(chunks)))
        )
        with client.stream(
            "POST", "http://localhost:30000/v1/systemone", json=SYSTEMONE_BODY
        ) as response:
            # Still the caller's to read: the matcher did not consume it.
            assert json.loads(response.read())["model"] == "d1"
        (span,) = _spans(matcher)
        assert "genai.decision.probability" not in span.attributes

    def test_one_span_inside_an_sdk_decision_span(self, matcher):
        """The OpenAI SDK sends decisions.create through httpx: one decision, one span."""
        openai = pytest.importorskip("openai")
        if not hasattr(openai.OpenAI(api_key="x"), "decisions"):  # pragma: no cover
            pytest.skip("this openai release has no Decisions resource")
        from genai_otel.instrumentors.openai_instrumentor import OpenAIInstrumentor

        matcher()
        instrumentor = OpenAIInstrumentor()
        instrumentor.tracer = matcher.tracer
        instrumentor.config = OTelConfig()
        client = openai.OpenAI(
            api_key="x", http_client=self._client(OPENAI_RESPONSE), max_retries=0
        )
        instrumentor._instrument_client(client)
        client.decisions.create(
            model="gpt-6-luna", input="hello", questions=OPENAI_BODY["questions"]
        )
        (span,) = _spans(matcher)
        assert span.name == "openai.decisions"

    def test_the_sdk_path_is_caught_when_the_sdk_is_not_instrumented(self, matcher):
        openai = pytest.importorskip("openai")
        if not hasattr(openai.OpenAI(api_key="x"), "decisions"):  # pragma: no cover
            pytest.skip("this openai release has no Decisions resource")
        matcher()
        client = openai.OpenAI(
            api_key="x", http_client=self._client(OPENAI_RESPONSE), max_retries=0
        )
        client.decisions.create(
            model="gpt-6-luna", input="hello", questions=OPENAI_BODY["questions"]
        )
        (span,) = _spans(matcher)
        assert span.name == "decision.http.openai"
        assert span.attributes["decision.provider"] == "openai"


def test_json_or_none():
    assert decision_http._json_or_none(None) is None
    assert decision_http._json_or_none(b'{"a": 1}') == {"a": 1}
    assert decision_http._json_or_none("[1]") == [1]
    assert decision_http._json_or_none(b"\xff") is None
    assert decision_http._json_or_none(object()) is None


def test_eval_kwargs_and_request_without_a_body():
    instrumentor = DecisionHTTPInstrumentor()
    instrumentor.config = OTelConfig()
    assert instrumentor._eval_kwargs(None) == {}
    request = instrumentor._http_request("systemone", "http://h/v1/systemone", None)
    assert request.payload is None and request.questions == []


def test_a_missing_library_is_skipped():
    instrumentor = DecisionHTTPInstrumentor()
    instrumentor._wrap("no_such_module_here", "Client.send", lambda *a: None)
    assert "no_such_module_here.Client.send" not in decision_http._WRAPPED


def test_legacy_kind_applies_to_systemone_over_http(matcher):
    matcher(decision_legacy_llm_kind=True)
    _session(_Adapter(body=SYSTEMONE_RESPONSE)).post(
        "http://localhost:30000/v1/systemone", json=SYSTEMONE_BODY
    )
    (span,) = _spans(matcher)
    assert span.attributes["openinference.span.kind"] == "LLM"
    assert span.attributes["genai.decision.protocol"] == "systemone"


class TestHttpx2:
    """openai >= 3.3 and typesafe-sdk send through httpx2."""

    def test_plain_httpx2(self, matcher):
        httpx2 = pytest.importorskip("httpx2")
        matcher()
        client = httpx2.Client(
            transport=httpx2.MockTransport(lambda r: httpx2.Response(200, json=SYSTEMONE_RESPONSE))
        )
        client.post("http://localhost:30000/v1/systemone", json=SYSTEMONE_BODY)
        (span,) = _spans(matcher)
        assert span.attributes["decision.token_count.input"] == 900

    def test_openai_default_client_is_one_span(self, matcher):
        httpx2 = pytest.importorskip("httpx2")
        openai = pytest.importorskip("openai")
        if not hasattr(openai.OpenAI(api_key="x"), "decisions"):  # pragma: no cover
            pytest.skip("this openai release has no Decisions resource")
        from genai_otel.instrumentors.openai_instrumentor import OpenAIInstrumentor

        matcher()
        instrumentor = OpenAIInstrumentor()
        instrumentor.tracer = matcher.tracer
        instrumentor.config = OTelConfig()
        client = openai.OpenAI(
            api_key="x",
            max_retries=0,
            http_client=httpx2.Client(
                transport=httpx2.MockTransport(lambda r: httpx2.Response(200, json=OPENAI_RESPONSE))
            ),
        )
        instrumentor._instrument_client(client)
        client.decisions.create(model="gpt-6-luna", input="hi", questions=OPENAI_BODY["questions"])
        (span,) = _spans(matcher)
        assert span.name == "openai.decisions"
