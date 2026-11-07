"""Fast/slow routing telemetry (issue #49): router headers, requested vs served, links."""

import asyncio
import json

import pytest
from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import genai_otel
from genai_otel import decision as manual
from genai_otel import routing
from genai_otel.config import OTelConfig
from genai_otel.routing import (
    DecisionLinkSpanProcessor,
    escalation_from,
    last_decision,
    parse_header_allowlist,
    record_response_headers,
    served_by_another_model,
)

QUESTIONS = {"escalate": {"type": "noul", "description": "Needs the large model?"}}


@pytest.fixture(autouse=True)
def empty_context():
    token = otel_context.attach(otel_context.Context())
    yield
    otel_context.detach(token)


# --------------------------------------------------------------------------- headers


def test_allowlist_lowercases_dedupes_and_refuses_credentials():
    allow = parse_header_allowlist(
        "X-LiteLLM-Model-Id, authorization,x-litellm-model-id", ["Set-Cookie", "x-auth-token"]
    )
    assert allow == ("x-litellm-model-id",)
    # Not every header with "token" or "key" in it is a credential.
    assert parse_header_allowlist("x-ratelimit-remaining-tokens,x-litellm-key-spend") == (
        "x-ratelimit-remaining-tokens",
        "x-litellm-key-spend",
    )
    assert parse_header_allowlist("", None) == ()


def test_record_caps_values_and_keeps_repeated_headers():
    class Span:
        def __init__(self):
            self.attributes = {}

        def set_attribute(self, key, value):
            self.attributes[key] = value

    class Headers:
        """httpx.Headers' shape: get_list for repeated headers."""

        _items = [("x-a", "1"), ("x-a", "2"), ("x-b", "y" * 1000)]

        def get_list(self, name):
            return [v for k, v in self._items if k == name]

    span = Span()
    headers = Headers()
    record_response_headers(span, headers, ("x-a", "x-b", "x-missing"))
    assert span.attributes["http.response.header.x-a"] == ["1", "2"]
    assert len(span.attributes["http.response.header.x-b"][0]) == routing.MAX_HEADER_VALUE
    assert "http.response.header.x-missing" not in span.attributes


CHAT = {
    "id": "c1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-5.5",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}
ROUTER_HEADERS = {
    "x-litellm-model-id": "deploy-7",
    "x-litellm-model-group": "auto",
    "authorization": "Bearer should-never-appear",
}


def _openai_harness(allow="", is_async=False, body=CHAT, monkeypatch=None):
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    from genai_otel.instrumentors.openai_instrumentor import OpenAIInstrumentor

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(DecisionLinkSpanProcessor(infer=False))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    inst = OpenAIInstrumentor()
    inst.tracer = provider.get_tracer("genai_otel.test")
    if is_async and monkeypatch is not None:
        # The async wrapper takes its tracer from the global provider.
        from opentelemetry import trace as trace_api

        monkeypatch.setattr(
            trace_api, "get_tracer", lambda name, *a, **k: provider.get_tracer(name)
        )
    inst._instrumented = True
    inst.config = OTelConfig(service_name="test", capture_response_headers=allow)

    def handler(request):
        return httpx.Response(200, headers=ROUTER_HEADERS, content=json.dumps(body).encode())

    if monkeypatch is not None:
        # An earlier test may have left OpenAI clients instrumented globally; build this
        # client unwrapped, so only this harness's instrumentor traces it.
        for cls in (openai.OpenAI, openai.AsyncOpenAI):
            init = cls.__dict__.get("__init__")
            while hasattr(init, "__wrapped__"):
                init = init.__wrapped__
            monkeypatch.setattr(cls, "__init__", init)

    transport = httpx.MockTransport(handler)
    if is_async:
        client = openai.AsyncOpenAI(
            api_key="t", max_retries=0, http_client=httpx.AsyncClient(transport=transport)
        )
        inst._instrument_async_client(client)
    else:
        client = openai.OpenAI(
            api_key="t", max_retries=0, http_client=httpx.Client(transport=transport)
        )
        inst._instrument_client(client)
    return client, exporter, provider


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_router_headers_on_the_chat_span_only_when_listed(is_async, monkeypatch):
    allow = "x-litellm-model-id,x-litellm-model-group,authorization"
    client, exporter, _ = _openai_harness(allow, is_async, monkeypatch=monkeypatch)
    messages = [{"role": "user", "content": "hi"}]
    if is_async:
        asyncio.run(client.chat.completions.create(model="auto", messages=messages))
    else:
        client.chat.completions.create(model="auto", messages=messages)
    (span,) = exporter.get_finished_spans()
    attrs = dict(span.attributes)
    assert list(attrs["http.response.header.x-litellm-model-id"]) == ["deploy-7"]
    assert list(attrs["http.response.header.x-litellm-model-group"]) == ["auto"]
    assert "http.response.header.authorization" not in attrs
    assert not any("should-never-appear" in str(v) for v in attrs.values())


def test_no_headers_by_default(monkeypatch):
    client, exporter, _ = _openai_harness("", monkeypatch=monkeypatch)
    client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "hi"}])
    (span,) = exporter.get_finished_spans()
    assert not any(k.startswith("http.response.header.") for k in span.attributes)


def test_raw_response_variant_is_covered(monkeypatch):
    client, exporter, _ = _openai_harness("x-litellm-model-id", monkeypatch=monkeypatch)
    raw = client.chat.completions.with_raw_response.create(
        model="auto", messages=[{"role": "user", "content": "hi"}]
    )
    raw.parse()
    spans = [s for s in exporter.get_finished_spans() if s.name == "openai.chat.completion"]
    assert list(spans[0].attributes["http.response.header.x-litellm-model-id"]) == ["deploy-7"]


def test_hook_ignores_spans_that_are_not_our_model_calls(monkeypatch):
    client, exporter, provider = _openai_harness("x-litellm-model-id", monkeypatch=monkeypatch)
    app_tracer = provider.get_tracer("my_app")
    with app_tracer.start_as_current_span("app.request"):
        client._client.get("https://api.openai.com/v1/models")
    (span,) = exporter.get_finished_spans()
    assert not any(k.startswith("http.response.header.") for k in span.attributes)


# --------------------------------------------------------------------------- served model


@pytest.mark.parametrize(
    "requested,served,flag",
    [
        ("auto", "gpt-5.5", True),
        ("gpt-4o", "gpt-4o-2024-08-06", False),
        ("openai/gpt-4o", "gpt-4o-2024-08-06", False),
        ("gpt-4o", "gpt-4o", False),
        ("my-group", None, False),
    ],
)
def test_served_by_another_model(requested, served, flag):
    assert served_by_another_model(requested, served) is flag


def test_alias_served_as_another_model_is_explicit(monkeypatch):
    client, exporter, _ = _openai_harness("", monkeypatch=monkeypatch)
    client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "hi"}])
    (span,) = exporter.get_finished_spans()
    assert span.attributes["gen_ai.router.requested_model"] == "auto"
    assert span.attributes["gen_ai.response.model"] == "gpt-5.5"


# --------------------------------------------------------------------------- links


@pytest.fixture
def links(monkeypatch):
    def make(infer):
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(DecisionLinkSpanProcessor(infer=infer))
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        monkeypatch.setattr(manual, "_INSTRUMENTOR", None)
        monkeypatch.setattr(manual, "_CONFIG", None)
        manual.configure(OTelConfig())
        manual._instrumentor().tracer = provider.get_tracer("genai_otel.test")
        tracer = provider.get_tracer("genai_otel.test")

        def model_call(name="llm.call", operation="chat"):
            with tracer.start_as_current_span(
                name, attributes={"gen_ai.operation.name": operation}
            ):
                pass

        return exporter, model_call, tracer

    return make


def _named(exporter, name):
    return [s for s in exporter.get_finished_spans() if s.name == name]


def test_explicit_links_go_on_model_calls_in_the_block_only(links):
    exporter, model_call, tracer = links(infer=False)
    with genai_otel.decision_span(model="router", questions=QUESTIONS) as d:
        d.record({"escalate": {"type": "noul", "probability": 0.71}})
    assert d.span_context is not None
    with escalation_from(d):
        model_call("big.1")
        with tracer.start_as_current_span(
            "agent", attributes={"gen_ai.operation.name": "agent.run"}
        ):
            model_call("big.2")
    model_call("after")

    (decision,) = _named(exporter, "decision")
    for name in ("big.1", "big.2"):
        (span,) = _named(exporter, name)
        assert [l.context.span_id for l in span.links] == [decision.context.span_id]
        assert span.attributes["genai.decision.link.inferred"] is False
        assert span.attributes["genai.decision.link.span_id"] == f"{decision.context.span_id:016x}"
        # A link, not a parent change.
        assert span.parent is None or span.parent.span_id != decision.context.span_id
    (agent,) = _named(exporter, "agent")
    assert not agent.links
    (after,) = _named(exporter, "after")
    assert not after.links


def test_last_decision_is_the_handle_for_sdk_decisions(links):
    exporter, model_call, _ = links(infer=False)
    with genai_otel.decision_span(model="router", questions=QUESTIONS) as d:
        d.record({"escalate": {"type": "noul", "probability": 0.9}})
    assert last_decision() == d.span_context
    with escalation_from(last_decision()):
        model_call("big")
    (span,) = _named(exporter, "big")
    assert span.links and span.links[0].context == d.span_context


def test_inferred_link_takes_the_next_model_call_only(links):
    exporter, model_call, _ = links(infer=True)
    with genai_otel.decision_span(model="router", questions=QUESTIONS) as d:
        d.record({"escalate": {"type": "noul", "probability": 0.9}})
    model_call("first")
    model_call("second")
    (first,) = _named(exporter, "first")
    (second,) = _named(exporter, "second")
    assert first.links[0].context == d.span_context
    assert first.links[0].attributes["genai.link.inferred"] is True
    assert first.attributes["genai.decision.link.inferred"] is True
    assert not second.links


def test_no_inference_unless_enabled(links):
    exporter, model_call, _ = links(infer=False)
    with genai_otel.decision_span(model="router", questions=QUESTIONS) as d:
        d.record({"escalate": {"type": "noul", "probability": 0.9}})
    model_call("next")
    (span,) = _named(exporter, "next")
    assert not span.links


def test_concurrent_tasks_never_link_to_each_others_decisions(links):
    exporter, model_call, _ = links(infer=True)

    async def task(i):
        await asyncio.sleep(0.001 * (i % 5))
        with genai_otel.decision_span(
            model="router", questions=QUESTIONS, name=f"decision-{i}"
        ) as d:
            d.record({"escalate": {"type": "noul", "probability": 0.5}})
        await asyncio.sleep(0.001 * ((i * 7) % 5))
        model_call(f"call-{i}")

    async def main():
        await asyncio.gather(*(task(i) for i in range(20)))

    asyncio.run(main())
    for i in range(20):
        (decision,) = _named(exporter, f"decision-{i}")
        (call,) = _named(exporter, f"call-{i}")
        assert [l.context.span_id for l in call.links] == [decision.context.span_id], i


def test_gathered_calls_after_one_decision_get_one_link(links):
    exporter, model_call, _ = links(infer=True)

    async def call(name):
        model_call(name)

    async def main():
        with genai_otel.decision_span(model="router", questions=QUESTIONS) as d:
            d.record({"escalate": {"type": "noul", "probability": 0.9}})
        await asyncio.gather(call("a"), call("b"))

    asyncio.run(main())
    linked = [s for s in exporter.get_finished_spans() if s.name in ("a", "b") and s.links]
    assert len(linked) == 1
