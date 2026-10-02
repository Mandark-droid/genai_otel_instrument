"""One model call, one usage record.

A framework span (LangChain chat model and similar) and the provider span beneath it
used to carry the same tokens and the same cost, and both recorded the token and cost
metrics, so anything summing spans or reading the counters saw one request twice.
Reproduced with LangChain 1.4.6 over the OpenAI SDK: ``langchain.chat_model.invoke``
153 tokens / 8.67e-05 and its child ``openai.chat.completion`` 153 tokens / 8.67e-05.

The provider span is the one that keeps the usage. The enclosing span keeps it only
when nothing beneath it recorded any.
"""

import asyncio
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.base import BaseInstrumentor

USAGE = {"prompt_tokens": 17, "completion_tokens": 136, "total_tokens": 153}
RECORDED_BY = "gen_ai.usage.recorded_by"


class _Instrumentor(BaseInstrumentor):
    def instrument(self, config):
        self.config = config

    def _extract_usage(self, result):
        return result.get("usage") if isinstance(result, dict) else None


@pytest.fixture
def harness():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    def make():
        inst = _Instrumentor()
        inst.config = OTelConfig()
        inst._instrumented = True
        inst.tracer = tracer
        inst.token_counter = MagicMock()
        inst.cost_counter = MagicMock()
        return inst

    framework, provider_inst = make(), make()
    # one pair of counters, as in a real process where the instruments are shared
    provider_inst.token_counter = framework.token_counter
    provider_inst.cost_counter = framework.cost_counter

    def spans():
        return {s.name: s for s in exporter.get_finished_spans()}

    return framework, provider_inst, spans


def _model_attr(instance, args, kwargs):
    return {"gen_ai.request.model": kwargs.get("model", "gpt-4o")}


def _tokens_recorded(inst) -> int:
    return sum(c.args[0] for c in inst.token_counter.add.call_args_list)


def _attrs(span):
    return dict(span.attributes or {})


def _has_usage(span) -> bool:
    a = _attrs(span)
    return any(
        k in a
        for k in (
            "gen_ai.usage.total_tokens",
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.cost.total",
        )
    )


def test_framework_over_instrumented_provider_records_usage_once(harness):
    framework, provider, spans = harness

    provider_call = provider.create_span_wrapper(
        "openai.chat.completion", extract_attributes=_model_attr
    )(lambda **kw: {"usage": dict(USAGE)})
    framework_call = framework.create_span_wrapper(
        "langchain.chat_model.invoke", extract_attributes=_model_attr
    )(
        # the framework returns its own object carrying the same usage
        lambda **kw: {"usage": dict(provider_call(model="gpt-4o")["usage"])}
    )
    framework_call(model="gpt-4o")

    got = spans()
    child, parent = got["openai.chat.completion"], got["langchain.chat_model.invoke"]
    assert child.parent.span_id == parent.context.span_id

    assert _attrs(child)["gen_ai.usage.total_tokens"] == 153
    assert _attrs(child)["gen_ai.usage.cost.total"] > 0
    assert RECORDED_BY not in _attrs(child)

    assert not _has_usage(parent), _attrs(parent)
    assert _attrs(parent)[RECORDED_BY] == "descendant"

    assert _tokens_recorded(framework) == 153  # 17 + 136, once
    assert framework.cost_counter.add.call_count == 1


def test_framework_over_uninstrumented_provider_keeps_its_usage(harness):
    framework, _provider, spans = harness

    framework_call = framework.create_span_wrapper(
        "langchain.chat_model.invoke", extract_attributes=_model_attr
    )(lambda **kw: {"usage": dict(USAGE)})
    framework_call(model="gpt-4o")

    parent = spans()["langchain.chat_model.invoke"]
    assert _attrs(parent)["gen_ai.usage.total_tokens"] == 153
    assert _attrs(parent)["gen_ai.usage.cost.total"] > 0
    assert RECORDED_BY not in _attrs(parent)
    assert _tokens_recorded(framework) == 153


def test_two_provider_calls_under_one_framework_span(harness):
    framework, provider, spans = harness
    provider_call = provider.create_span_wrapper(
        "openai.chat.completion", extract_attributes=_model_attr
    )(lambda **kw: {"usage": dict(USAGE)})

    def run(**kw):
        provider_call(model="gpt-4o")
        provider_call(model="gpt-4o")
        return {"usage": {"prompt_tokens": 34, "completion_tokens": 272, "total_tokens": 306}}

    framework.create_span_wrapper("langgraph.graph.invoke", extract_attributes=_model_attr)(run)(
        model="gpt-4o"
    )

    parent = spans()["langgraph.graph.invoke"]
    assert not _has_usage(parent)
    assert _tokens_recorded(framework) == 306  # two calls, each once
    assert framework.cost_counter.add.call_count == 2


def test_provider_error_without_usage_leaves_framework_usage(harness):
    framework, provider, spans = harness

    def failing(**kw):
        raise RuntimeError("provider down")

    provider_call = provider.create_span_wrapper(
        "openai.chat.completion", extract_attributes=_model_attr
    )(failing)

    def run(**kw):
        with pytest.raises(RuntimeError):
            provider_call(model="gpt-4o")
        return {"usage": dict(USAGE)}  # the framework answered from elsewhere

    framework.create_span_wrapper("langchain.chat_model.invoke", extract_attributes=_model_attr)(
        run
    )(model="gpt-4o")

    parent = spans()["langchain.chat_model.invoke"]
    assert _attrs(parent)["gen_ai.usage.total_tokens"] == 153
    assert RECORDED_BY not in _attrs(parent)


def test_sibling_calls_do_not_silence_each_other(harness):
    """A usage record under one framework span must not leak into the next one."""
    framework, provider, spans = harness
    provider_call = provider.create_span_wrapper(
        "openai.chat.completion", extract_attributes=_model_attr
    )(lambda **kw: {"usage": dict(USAGE)})
    first = framework.create_span_wrapper("first", extract_attributes=_model_attr)(
        lambda **kw: {"usage": dict(provider_call(model="gpt-4o")["usage"])}
    )
    second = framework.create_span_wrapper("second", extract_attributes=_model_attr)(
        lambda **kw: {"usage": dict(USAGE)}
    )

    first(model="gpt-4o")
    second(model="gpt-4o")

    got = spans()
    assert not _has_usage(got["first"])
    assert _attrs(got["second"])["gen_ai.usage.total_tokens"] == 153


def test_async_framework_over_async_provider_records_usage_once(harness):
    framework, provider, spans = harness

    async def provider_fn(**kw):
        return {"usage": dict(USAGE)}

    provider_call = provider.create_span_wrapper(
        "openai.chat.completion", extract_attributes=_model_attr
    )(provider_fn)

    async def framework_fn(**kw):
        return {"usage": dict((await provider_call(model="gpt-4o"))["usage"])}

    framework_call = framework.create_span_wrapper(
        "langchain.chat_model.ainvoke", extract_attributes=_model_attr
    )(framework_fn)
    asyncio.run(framework_call(model="gpt-4o"))

    got = spans()
    assert _attrs(got["openai.chat.completion"])["gen_ai.usage.total_tokens"] == 153
    assert not _has_usage(got["langchain.chat_model.ainvoke"])
    assert _tokens_recorded(framework) == 153


def test_streamed_provider_under_framework_span(harness):
    framework, provider, spans = harness

    def provider_stream(**kw):
        yield {"delta": "a"}
        yield {"usage": dict(USAGE)}

    provider_call = provider.create_span_wrapper(
        "openai.chat.completion", extract_attributes=_model_attr
    )(provider_stream)

    def run(**kw):
        list(provider_call(model="gpt-4o", stream=True))
        return {"usage": dict(USAGE)}

    framework.create_span_wrapper("langchain.chat_model.invoke", extract_attributes=_model_attr)(
        run
    )(model="gpt-4o")

    got = spans()
    assert _attrs(got["openai.chat.completion"])["gen_ai.usage.total_tokens"] == 153
    assert not _has_usage(got["langchain.chat_model.invoke"])
    assert _tokens_recorded(framework) == 153


def test_framework_span_opened_outside_the_base_wrapper():
    """LangChain's instrumentor opens its span itself and only then records metrics.

    Such a span never passes through ``create_span_wrapper``; the span processor is what
    gives it a holder and links the provider span's holder to it.
    """
    import time

    from genai_otel.instrumentors.base import UsageHolderSpanProcessor

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(UsageHolderSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    inst = _Instrumentor()
    inst.config = OTelConfig()
    inst._instrumented = True
    inst.tracer = tracer
    inst.token_counter = MagicMock()
    inst.cost_counter = MagicMock()

    provider_call = inst.create_span_wrapper(
        "openai.chat.completion", extract_attributes=_model_attr
    )(lambda **kw: {"usage": dict(USAGE)})

    with tracer.start_as_current_span(
        "langchain.chat_model.invoke", attributes={"gen_ai.request.model": "gpt-4o"}
    ) as span:
        start = time.time()
        with tracer.start_as_current_span("application.step"):  # a span nobody instruments
            result = provider_call(model="gpt-4o")
        inst._record_result_metrics(span, {"usage": dict(result["usage"])}, start, {})

    got = {s.name: s for s in exporter.get_finished_spans()}
    assert _attrs(got["openai.chat.completion"])["gen_ai.usage.total_tokens"] == 153
    assert not _has_usage(got["langchain.chat_model.invoke"])
    assert _attrs(got["langchain.chat_model.invoke"])[RECORDED_BY] == "descendant"
    assert _tokens_recorded(inst) == 153
    assert inst.cost_counter.add.call_count == 1


def test_holders_are_released_when_spans_end():
    from genai_otel.instrumentors import base

    provider = TracerProvider()
    provider.add_span_processor(base.UsageHolderSpanProcessor())
    tracer = provider.get_tracer("test")
    before = len(base._USAGE_HOLDERS)
    with tracer.start_as_current_span("a"):
        with tracer.start_as_current_span("b"):
            assert len(base._USAGE_HOLDERS) == before + 2
    assert len(base._USAGE_HOLDERS) == before
