"""Bedrock's Converse API must produce a span, not silence (issue #27).

`aws_bedrock_instrumentor` wrapped `invoke_model` and nothing else. Bedrock's
Converse API (`converse` / `converse_stream`) -- the unified API AWS points
callers at, and the practical path for every non-Anthropic model -- produced
**no span at all**. So did `invoke_model_with_response_stream`, meaning even the
covered call went dark as soon as a caller streamed.

Converse differs from `invoke_model` in every field the instrumentor reads:

    request   `messages[].content[]` typed blocks; `system` is a top-level
              parameter, not a message role; `inferenceConfig.{maxTokens,...}`
    response  a plain dict with `output.message.content[]` and `stopReason`
              -- no `body`, no `contentType`
    usage     `usage.{inputTokens,outputTokens,totalTokens}` -- camelCase, and
              at the top level rather than inside a JSON body

The last one is the dangerous one: `_extract_usage` keyed on `contentType`, so a
Converse response returned None and the span was priced at zero rather than
flagged as having no usage.
"""

from unittest.mock import MagicMock

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF, ALWAYS_ON

from genai_otel.instrumentors.aws_bedrock_instrumentor import AWSBedrockInstrumentor


class FakeClient:
    """Stand-in for a boto3 bedrock-runtime client.

    A MagicMock answers hasattr for everything, which would make the
    "is this method wrapped" assertions meaningless.
    """

    def __init__(self, *methods):
        for name in methods:
            setattr(self, name, self._make(name))

    @staticmethod
    def _make(name):
        def _call(**kwargs):
            return {"called": name}

        return _call


def _converse_response():
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "It is sunny in SF."},
                    {"toolUse": {"toolUseId": "tu_1", "name": "get_weather", "input": {"c": "SF"}}},
                ],
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 100, "outputTokens": 50, "totalTokens": 150},
        "metrics": {"latencyMs": 1234},
    }


def _invoke_model_response():
    """The existing shape -- must keep working."""
    return {
        "contentType": "application/json",
        "body": '{"content": [{"text": "hi"}], "usage": {"inputTokens": 7, "outputTokens": 3}}',
    }


class TestConverseMethodsAreWrapped:
    def test_converse_is_wrapped(self):
        i = AWSBedrockInstrumentor()
        i.config = MagicMock(content_max_length=0, enable_content_capture=True)
        i._instrumented = True
        c = FakeClient("converse")
        original = c.converse

        i._instrument_bedrock_client(c)

        assert c.converse is not original, "converse left uninstrumented"

    def test_converse_stream_is_wrapped(self):
        i = AWSBedrockInstrumentor()
        i.config = MagicMock(content_max_length=0, enable_content_capture=True)
        i._instrumented = True
        c = FakeClient("converse_stream")
        original = c.converse_stream

        i._instrument_bedrock_client(c)

        assert c.converse_stream is not original

    def test_invoke_model_with_response_stream_is_wrapped(self):
        i = AWSBedrockInstrumentor()
        i.config = MagicMock(content_max_length=0, enable_content_capture=True)
        i._instrumented = True
        c = FakeClient("invoke_model_with_response_stream")
        original = c.invoke_model_with_response_stream

        i._instrument_bedrock_client(c)

        assert c.invoke_model_with_response_stream is not original

    def test_invoke_model_still_wrapped(self):
        """Regression: the one method that already worked."""
        i = AWSBedrockInstrumentor()
        i.config = MagicMock(content_max_length=0, enable_content_capture=True)
        i._instrumented = True
        c = FakeClient("invoke_model")
        original = c.invoke_model

        i._instrument_bedrock_client(c)

        assert c.invoke_model is not original

    def test_client_missing_converse_does_not_raise(self):
        """An older botocore has no converse; that must not break invoke_model."""
        i = AWSBedrockInstrumentor()
        i.config = MagicMock(content_max_length=0, enable_content_capture=True)
        i._instrumented = True
        c = FakeClient("invoke_model")

        i._instrument_bedrock_client(c)  # must not raise


class TestConverseRequestAttributes:
    def _attrs(self, **kwargs):
        return AWSBedrockInstrumentor()._extract_converse_attributes(None, (), kwargs)

    def test_core_attributes(self):
        a = self._attrs(
            modelId="meta.llama3-70b-instruct-v1:0",
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
        )

        assert a["gen_ai.system"] == "aws_bedrock"
        assert a["gen_ai.request.model"] == "meta.llama3-70b-instruct-v1:0"
        assert a["gen_ai.operation.name"] == "chat"
        assert a["gen_ai.request.message_count"] == 1

    def test_system_is_a_top_level_parameter_not_a_message(self):
        """Converse takes `system` separately; it is not a role in messages[]."""
        a = self._attrs(
            modelId="m",
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
            system=[{"text": "Be terse."}],
        )

        assert a["gen_ai.request.instructions"] == "Be terse."
        assert a["gen_ai.request.message_count"] == 1, "system must not inflate the message count"

    def test_inference_config_maps_onto_semconv(self):
        a = self._attrs(
            modelId="m",
            messages=[],
            inferenceConfig={
                "maxTokens": 256,
                "temperature": 0.2,
                "topP": 0.9,
                "stopSequences": ["END"],
            },
        )

        assert a["gen_ai.request.max_tokens"] == 256
        assert a["gen_ai.request.temperature"] == 0.2
        assert a["gen_ai.request.top_p"] == 0.9
        assert a["gen_ai.request.stop_sequences"] == ["END"]

    def test_tool_config_is_serialized(self):
        a = self._attrs(
            modelId="m",
            messages=[],
            toolConfig={"tools": [{"toolSpec": {"name": "get_weather"}}]},
        )

        assert "llm.tools" in a


class TestConverseUsage:
    def test_camelcase_usage_is_mapped(self):
        u = AWSBedrockInstrumentor()._extract_usage(_converse_response())

        assert u is not None, "Converse usage went unread, so the span is priced at zero"
        assert u["prompt_tokens"] == 100
        assert u["completion_tokens"] == 50
        assert u["total_tokens"] == 150

    def test_usage_from_trailing_stream_metadata_event(self):
        """converse_stream reports tokens only in the final `metadata` event."""
        event = {"metadata": {"usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15}}}

        u = AWSBedrockInstrumentor()._extract_usage(event)

        assert u["prompt_tokens"] == 11
        assert u["completion_tokens"] == 4
        assert u["total_tokens"] == 15

    def test_invoke_model_usage_still_works(self):
        """Regression: usage inside a JSON body."""
        u = AWSBedrockInstrumentor()._extract_usage(_invoke_model_response())

        assert u["prompt_tokens"] == 7
        assert u["completion_tokens"] == 3

    def test_non_usage_event_returns_none(self):
        assert AWSBedrockInstrumentor()._extract_usage({"contentBlockDelta": {}}) is None


class TestConverseResponseAttributes:
    def test_completion_text_extracted_from_output_message(self):
        a = AWSBedrockInstrumentor()._extract_response_attributes(_converse_response())

        assert a["gen_ai.response"] == "It is sunny in SF."

    def test_stop_reason_recorded(self):
        a = AWSBedrockInstrumentor()._extract_response_attributes(_converse_response())

        assert a["gen_ai.response.finish_reasons"] == ["end_turn"]

    def test_tool_use_blocks_extracted(self):
        a = AWSBedrockInstrumentor()._extract_response_attributes(_converse_response())

        names = [v for k, v in a.items() if k.endswith(".tool_call.function.name")]
        ids = [v for k, v in a.items() if k.endswith(".tool_call.id")]

        assert names == ["get_weather"]
        assert ids == ["tu_1"]

    def test_invoke_model_response_still_extracted(self):
        """Regression: the body-parsing path."""
        a = AWSBedrockInstrumentor()._extract_response_attributes(_invoke_model_response())

        assert a["gen_ai.response"] == "hi"

    def test_finish_reason_from_stop_reason(self):
        assert AWSBedrockInstrumentor()._extract_finish_reason(_converse_response()) == "end_turn"


class TestConverseStreamKeepsSpanOpen:
    """converse_stream returns immediately; generation happens during iteration.

    A span closed on return would report near-zero latency and no tokens, so the
    returned event stream is wrapped and the span closes when it is exhausted.
    """

    EVENTS = [
        {"contentBlockDelta": {"delta": {"text": "It is "}}},
        {"contentBlockDelta": {"delta": {"text": "sunny."}}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15}}},
    ]

    @staticmethod
    def _instrumentor(recording=True):
        i = AWSBedrockInstrumentor()
        i.config = MagicMock(content_max_length=0, enable_content_capture=True)
        i._instrumented = True
        exporter = InMemorySpanExporter()
        provider = TracerProvider(sampler=ALWAYS_ON if recording else ALWAYS_OFF)
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        i.tracer = provider.get_tracer(__name__)
        return i, exporter

    def test_stream_is_wrapped_and_yields_the_same_events(self):
        i, _ = self._instrumentor()
        raw = iter(self.EVENTS)
        client = FakeClient()
        client.converse_stream = lambda **kw: {"stream": raw, "ResponseMetadata": {}}
        i._instrument_bedrock_client(client)

        result = client.converse_stream(
            modelId="m", messages=[{"role": "user", "content": [{"text": "hi"}]}]
        )

        assert "stream" in result
        # Identity, not just equality: an unwrapped stream would still yield the
        # right events, so equality alone cannot tell instrumented from not.
        assert result["stream"] is not raw, "event stream was not wrapped; span closes early"
        assert list(result["stream"]) == self.EVENTS, "wrapping must not alter the events"

    def test_span_stays_open_until_the_stream_is_exhausted(self):
        i, exporter = self._instrumentor()
        client = FakeClient()
        client.converse_stream = lambda **kw: {"stream": iter(self.EVENTS)}
        i._instrument_bedrock_client(client)

        result = client.converse_stream(modelId="m", messages=[])
        assert exporter.get_finished_spans() == (), "span closed before the stream was read"

        list(result["stream"])

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "aws.bedrock.converse_stream"

    def test_usage_from_the_trailing_metadata_event_lands_on_the_span(self):
        i, exporter = self._instrumentor()
        client = FakeClient()
        client.converse_stream = lambda **kw: {"stream": iter(self.EVENTS)}
        i._instrument_bedrock_client(client)

        list(client.converse_stream(modelId="m", messages=[])["stream"])

        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs.get("gen_ai.usage.input_tokens") == 11
        assert attrs.get("gen_ai.usage.output_tokens") == 4

    def test_sampled_out_span_does_not_raise(self):
        """A NonRecordingSpan has no `.name`; measuring one would raise on every call."""
        i, _ = self._instrumentor(recording=False)
        raw = iter(self.EVENTS)
        client = FakeClient()
        client.converse_stream = lambda **kw: {"stream": raw}
        i._instrument_bedrock_client(client)

        result = client.converse_stream(modelId="m", messages=[])

        assert list(result["stream"]) == self.EVENTS

    def test_response_without_a_stream_key_does_not_hang(self):
        i, _ = self._instrumentor()
        client = FakeClient()
        client.converse_stream = lambda **kw: {"ResponseMetadata": {}}
        i._instrument_bedrock_client(client)

        result = client.converse_stream(modelId="m", messages=[])

        assert result == {"ResponseMetadata": {}}
