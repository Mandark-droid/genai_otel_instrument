from types import SimpleNamespace

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.antigravity_instrumentor import AntigravityInstrumentor


class Text:
    def __init__(self, text):
        self.text = text


class FakeResponse:
    def __init__(self, chunks):
        self._chunk_stream = self._chunks(chunks)
        self.usage_metadata = SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=5,
            thoughts_token_count=2,
            cached_content_token_count=3,
            total_token_count=19,
        )

    @staticmethod
    async def _chunks(chunks):
        for chunk in chunks:
            assert trace.get_current_span().get_span_context().is_valid
            yield chunk


@pytest.mark.asyncio
async def test_chat_span_covers_lazy_stream_and_records_usage_and_opt_in_content():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = AntigravityInstrumentor()
    instrumentor.tracer = provider.get_tracer("test.antigravity")
    instrumentor._setup_config(
        OTelConfig(
            enabled_instrumentors=["antigravity"],
            enable_content_capture=True,
            content_max_length=7,
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
        )
    )
    instrumentor._instrumented = True

    async def chat(prompt):
        assert prompt == "Say hello"
        return FakeResponse([Text("Hello "), Text("world!")])

    agent = SimpleNamespace(_config=SimpleNamespace(model="gemini-3-flash"))
    response = await instrumentor._wrap_chat(chat, agent, ("Say hello",), {})
    assert not exporter.get_finished_spans()

    chunks = [chunk async for chunk in response._chunk_stream]
    assert len(chunks) == 2

    span = exporter.get_finished_spans()[0]
    assert span.name == "antigravity.agent.chat"
    assert span.attributes["gen_ai.system"] == "antigravity"
    assert span.attributes["gen_ai.request.model"] == "gemini-3-flash"
    assert span.attributes["input.value"] == '"Say he'
    assert span.attributes["output.value"] == "Hello w"
    assert span.attributes["gen_ai.usage.input_tokens"] == 12
    assert span.attributes["gen_ai.usage.output_tokens"] == 7
    assert span.attributes["gen_ai.usage.reasoning_tokens"] == 2
    assert span.attributes["gen_ai.usage.cache_read_input_tokens"] == 3
    assert span.attributes["gen_ai.usage.total_tokens"] == 19
    provider.shutdown()


def test_usage_metadata_normalizes_antigravity_token_names():
    attrs = AntigravityInstrumentor._usage_attributes(
        {"prompt_token_count": 8, "candidates_token_count": 4, "thoughts_token_count": 3}
    )

    assert attrs == {
        "gen_ai.usage.input_tokens": 8,
        "gen_ai.usage.output_tokens": 7,
        "gen_ai.usage.reasoning_tokens": 3,
        "gen_ai.usage.total_tokens": 15,
    }
