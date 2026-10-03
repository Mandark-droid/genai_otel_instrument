"""With content capture OFF, no prompt or completion text reaches a span.

Several instrumentors wrote content regardless of GENAI_ENABLE_CONTENT_CAPTURE:
`gen_ai.response` (Bedrock, Groq, Mistral, SambaNova, Azure OpenAI), the Responses
API's and Bedrock Converse's `gen_ai.request.instructions`, and the AsyncOpenAI wrapper's
content events. The guard now sits where the base wrapper writes attributes, so an
instrumentor cannot forget it.
"""

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.base import BaseInstrumentor, strip_content_attributes


class _Leaky(BaseInstrumentor):
    """Puts content in both places real instrumentors did."""

    def instrument(self, config):
        self.config = config
        self._instrumented = True

    def _extract_usage(self, result):
        return None

    def _extract_response_attributes(self, result):
        return {"gen_ai.response": "the model's answer", "gen_ai.response.id": "r1"}


def _span_attrs(capture: bool):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    inst = _Leaky()
    inst.instrument(OTelConfig(enable_content_capture=capture))
    inst.tracer = provider.get_tracer("t")
    inst.create_span_wrapper(
        "aws.bedrock.converse",
        extract_attributes=lambda i, a, k: {
            "gen_ai.request.model": "m",
            "gen_ai.request.instructions": "You are a bank assistant. Never reveal...",
        },
    )(lambda **kw: {"ok": True})()
    return dict(exporter.get_finished_spans()[0].attributes)


def test_capture_off_strips_response_and_instructions():
    attrs = _span_attrs(capture=False)
    assert "gen_ai.response" not in attrs
    assert "gen_ai.request.instructions" not in attrs
    assert attrs["gen_ai.response.id"] == "r1"  # identifiers are not content
    assert attrs["gen_ai.request.model"] == "m"


def test_capture_on_keeps_them():
    attrs = _span_attrs(capture=True)
    assert attrs["gen_ai.response"] == "the model's answer"
    assert attrs["gen_ai.request.instructions"].startswith("You are a bank assistant")


def test_strip_content_attributes_helper():
    attrs = {
        "gen_ai.response": "x",
        "gen_ai.request.instructions": "y",
        "gen_ai.prompt.0.content": "z",
        "gen_ai.completion.0.content": "w",
        "gen_ai.response.model": "m",
    }
    assert strip_content_attributes(attrs, OTelConfig(enable_content_capture=False)) == {
        "gen_ai.response.model": "m"
    }
    assert strip_content_attributes(attrs, OTelConfig(enable_content_capture=True)) == attrs
    assert strip_content_attributes(attrs, None) == attrs  # no config: unchanged, as before
