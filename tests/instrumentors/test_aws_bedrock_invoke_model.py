"""invoke_model records tokens, and the caller still reads the body; the streaming
sibling keeps its span open until the stream ends.

boto3 returns invoke_model's body as a botocore StreamingBody - readable ONCE. The
instrumentor passed it straight to json.loads, which raised and was swallowed, so a real
call recorded no tokens and no cost (only tests with a str body ever passed). And
invoke_model_with_response_stream has no `stream=True` argument, so the generic wrapper
closed its span the moment the call returned: near-zero latency, no tokens.
"""

import io
import json
from typing import Optional
from unittest.mock import MagicMock

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_ON

from genai_otel.instrumentors.aws_bedrock_instrumentor import AWSBedrockInstrumentor


class OneShotBody:
    """Behaves like botocore's StreamingBody: the bytes can be read once."""

    def __init__(self, data: bytes) -> None:
        self._raw = io.BytesIO(data)

    def read(self, amt=None):
        return self._raw.read() if amt is None else self._raw.read(amt)


class FakeClient:
    pass


def _instrumentor():
    i = AWSBedrockInstrumentor()
    i.config = MagicMock(content_max_length=0, enable_content_capture=True)
    i._instrumented = True
    exporter = InMemorySpanExporter()
    provider = TracerProvider(sampler=ALWAYS_ON)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    i.tracer = provider.get_tracer(__name__)
    return i, exporter


ANTHROPIC_BODY = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Paris."}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 14, "output_tokens": 3},
}


def _response(body: dict, headers: Optional[dict] = None) -> dict:
    return {
        "body": OneShotBody(json.dumps(body).encode()),
        "contentType": "application/json",
        "ResponseMetadata": {"HTTPStatusCode": 200, "HTTPHeaders": headers or {}},
    }


def _invoke(i, response):
    client = FakeClient()
    client.invoke_model = lambda **kw: response
    i._instrument_bedrock_client(client)
    return client.invoke_model(
        modelId="anthropic.claude-sonnet-4",
        body=json.dumps({"messages": [{"role": "user", "content": "Capital of France?"}]}),
    )


def test_tokens_come_from_bedrocks_headers() -> None:
    i, exporter = _instrumentor()
    _invoke(
        i,
        _response(
            {"outputText": "x"},
            {"x-amzn-bedrock-input-token-count": "21", "x-amzn-bedrock-output-token-count": "5"},
        ),
    )
    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs.get("gen_ai.usage.input_tokens") == 21
    assert attrs.get("gen_ai.usage.output_tokens") == 5


def test_tokens_come_from_an_anthropic_body_without_headers() -> None:
    i, exporter = _instrumentor()
    _invoke(i, _response(ANTHROPIC_BODY))
    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs.get("gen_ai.usage.input_tokens") == 14
    assert attrs.get("gen_ai.usage.output_tokens") == 3


def test_the_caller_still_reads_the_whole_body() -> None:
    i, _ = _instrumentor()
    result = _invoke(i, _response(ANTHROPIC_BODY))
    assert json.loads(result["body"].read()) == ANTHROPIC_BODY


def test_the_response_text_is_captured_from_a_real_body() -> None:
    i, exporter = _instrumentor()
    _invoke(i, _response(ANTHROPIC_BODY))
    assert exporter.get_finished_spans()[0].attributes.get("gen_ai.response") == "Paris."


# --- invoke_model_with_response_stream ----------------------------------------------


def _chunk(payload: dict) -> dict:
    return {"chunk": {"bytes": json.dumps(payload).encode()}}


STREAM = [
    _chunk({"type": "message_start", "message": {"usage": {"input_tokens": 14}}}),
    _chunk({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Paris."}}),
    _chunk({"type": "message_delta", "usage": {"output_tokens": 3}}),
    _chunk(
        {
            "type": "message_stop",
            "amazon-bedrock-invocationMetrics": {
                "inputTokenCount": 14,
                "outputTokenCount": 3,
                "invocationLatency": 400,
            },
        }
    ),
]


def _stream_client(i, events):
    client = FakeClient()
    client.invoke_model_with_response_stream = lambda **kw: {
        "body": iter(events),
        "contentType": "application/json",
    }
    i._instrument_bedrock_client(client)
    return client


def test_stream_span_stays_open_until_the_stream_is_read() -> None:
    i, exporter = _instrumentor()
    client = _stream_client(i, STREAM)
    result = client.invoke_model_with_response_stream(
        modelId="anthropic.claude-sonnet-4", body="{}"
    )
    assert exporter.get_finished_spans() == (), "span closed before the stream was read"
    assert list(result["body"]) == STREAM, "wrapping must not alter the events"
    (span,) = exporter.get_finished_spans()
    assert span.name == "aws.bedrock.invoke_model_with_response_stream"


def test_stream_tokens_come_from_the_invocation_metrics() -> None:
    i, exporter = _instrumentor()
    list(
        _stream_client(i, STREAM).invoke_model_with_response_stream(
            modelId="anthropic.claude-sonnet-4", body="{}"
        )["body"]
    )
    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs.get("gen_ai.usage.input_tokens") == 14
    assert attrs.get("gen_ai.usage.output_tokens") == 3
