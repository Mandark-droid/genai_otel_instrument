"""Detectors and block policies on AWS Bedrock.

``invoke_model`` carries its prompt inside a JSON ``body`` whose shape depends on
the model family, and its answer inside a JSON response body. The detectors read
neither, so an ``invoke_model`` call was not evaluated and a block policy let it
through. ``converse`` was evaluated, but without its ``system`` prompt.
"""

import io
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genai_otel.config import OTelConfig
from genai_otel.exceptions import PolicyViolationError
from genai_otel.instrumentors.aws_bedrock_instrumentor import AWSBedrockInstrumentor
from genai_otel.instrumentors.base import BaseInstrumentor, drain_evaluations

CARD = "4111 1111 1111 1111"
USER_TEXT = "My card number is {0}.".format(CARD)
SYSTEM_TEXT = "You are terse."

# Request bodies by model family, as invoke_model takes them.
BODIES = {
    "claude": (
        "anthropic.claude-3-haiku-20240307-v1:0",
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 16,
            "system": SYSTEM_TEXT,
            "messages": [{"role": "user", "content": [{"type": "text", "text": USER_TEXT}]}],
        },
    ),
    "claude_string_content": (
        "anthropic.claude-3-haiku-20240307-v1:0",
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 16,
            "system": SYSTEM_TEXT,
            "messages": [{"role": "user", "content": USER_TEXT}],
        },
    ),
    "nova": (
        "amazon.nova-micro-v1:0",
        {
            "system": [{"text": SYSTEM_TEXT}],
            "messages": [{"role": "user", "content": [{"text": USER_TEXT}]}],
            "inferenceConfig": {"maxTokens": 16},
        },
    ),
    "titan_text": (
        "amazon.titan-text-express-v1",
        {"inputText": USER_TEXT, "textGenerationConfig": {"maxTokenCount": 16}},
    ),
    "llama": ("meta.llama3-8b-instruct-v1:0", {"prompt": USER_TEXT, "max_gen_len": 16}),
    "cohere_command_r": (
        "cohere.command-r-v1:0",
        {"message": USER_TEXT, "preamble": SYSTEM_TEXT, "max_tokens": 16},
    ),
}
WITH_SYSTEM = {"claude", "claude_string_content", "nova", "cohere_command_r"}

CONVERSE_KWARGS = {
    "modelId": "amazon.nova-micro-v1:0",
    "system": [{"text": SYSTEM_TEXT}],
    "messages": [{"role": "user", "content": [{"text": USER_TEXT}]}],
}


class _PIIDetector:
    """Stands in for the PII detector: flags text that contains the card number."""

    def __init__(self):
        self.seen = []

    def detect(self, text):
        self.seen.append(text)
        found = CARD in text
        return SimpleNamespace(
            has_pii=found,
            entity_counts={"CREDIT_CARD": 1} if found else {},
            entities=[CARD] if found else [],
            score=0.9 if found else 0.0,
            redacted_text=None,
            blocked=False,
        )


@pytest.fixture
def detector():
    cleared = {
        "_pii_detector": None,
        "_toxicity_detector": None,
        "_bias_detector": None,
        "_prompt_injection_detector": None,
        "_restricted_topics_detector": None,
        "_hallucination_detector": None,
        "_blocking_active": False,
        "_blocking_policies": frozenset(),
    }
    saved = {name: getattr(BaseInstrumentor, name) for name in cleared}
    for name, value in cleared.items():
        setattr(BaseInstrumentor, name, value)
    BaseInstrumentor._pii_detector = _PIIDetector()
    yield BaseInstrumentor._pii_detector
    for name, value in saved.items():
        setattr(BaseInstrumentor, name, value)


def _block_on_pii():
    BaseInstrumentor._blocking_policies = frozenset({"pii"})
    BaseInstrumentor._blocking_active = True


class _Client:
    """A bedrock-runtime client's four generation calls, recording what reaches AWS."""

    def __init__(self):
        self.calls = []

    def invoke_model(self, **kwargs):
        self.calls.append(("invoke_model", kwargs))
        answer = {"content": [{"type": "text", "text": "Noted."}]}
        return {"body": io.BytesIO(json.dumps(answer).encode("utf-8")), "ResponseMetadata": {}}

    def invoke_model_with_response_stream(self, **kwargs):
        self.calls.append(("invoke_model_with_response_stream", kwargs))
        return {"body": iter(())}

    def converse(self, **kwargs):
        self.calls.append(("converse", kwargs))
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": "Noted."}]}},
            "usage": {"inputTokens": 12, "outputTokens": 2, "totalTokens": 14},
            "stopReason": "end_turn",
        }

    def converse_stream(self, **kwargs):
        self.calls.append(("converse_stream", kwargs))
        return {"stream": iter(())}


@pytest.fixture
def bedrock():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with patch.dict("sys.modules", {"boto3": MagicMock()}):
        instrumentor = AWSBedrockInstrumentor()
    instrumentor.tracer = provider.get_tracer("test")
    instrumentor.config = OTelConfig(service_name="t")
    instrumentor._instrumented = True
    client = _Client()
    instrumentor._instrument_bedrock_client(client)

    def spans():
        # The detectors run after the call has returned, and the span is
        # exported when they are done.
        assert drain_evaluations(30)
        return exporter.get_finished_spans()

    return SimpleNamespace(client=client, spans=spans, inst=instrumentor)


def _invoke_kwargs(family, as_bytes=False):
    model_id, body = BODIES[family]
    text = json.dumps(body)
    return {"modelId": model_id, "body": text.encode("utf-8") if as_bytes else text}


@pytest.mark.parametrize("family", sorted(BODIES))
def test_invoke_model_prompt_is_evaluated(family, detector, bedrock):
    bedrock.client.invoke_model(**_invoke_kwargs(family))

    (span,) = bedrock.spans()
    assert span.attributes["evaluation.pii.prompt.detected"] is True
    evaluated = "\n".join(detector.seen)
    assert CARD in evaluated
    if family in WITH_SYSTEM:
        assert SYSTEM_TEXT in evaluated, "the system prompt was not evaluated"


@pytest.mark.parametrize("family", sorted(BODIES))
@pytest.mark.parametrize("as_bytes", [False, True], ids=["str_body", "bytes_body"])
def test_invoke_model_block_policy_stops_the_request(family, as_bytes, detector, bedrock):
    _block_on_pii()

    with pytest.raises(PolicyViolationError):
        bedrock.client.invoke_model(**_invoke_kwargs(family, as_bytes))

    assert bedrock.client.calls == [], "the request reached AWS despite the block policy"
    (span,) = bedrock.spans()
    assert span.attributes["evaluation.blocked"] is True
    assert span.status.status_code.name == "ERROR"


@pytest.mark.parametrize("family", sorted(BODIES))
def test_invoke_model_stream_block_policy_stops_the_request(family, detector, bedrock):
    _block_on_pii()

    with pytest.raises(PolicyViolationError):
        bedrock.client.invoke_model_with_response_stream(**_invoke_kwargs(family))

    assert bedrock.client.calls == []
    (span,) = bedrock.spans()
    assert span.attributes["evaluation.blocked"] is True
    assert span.status.status_code.name == "ERROR"


def test_converse_evaluates_the_system_prompt_too(detector, bedrock):
    bedrock.client.converse(**CONVERSE_KWARGS)

    (span,) = bedrock.spans()
    assert span.attributes["evaluation.pii.prompt.detected"] is True
    evaluated = "\n".join(detector.seen)
    assert CARD in evaluated and SYSTEM_TEXT in evaluated


@pytest.mark.parametrize("method", ["converse", "converse_stream"])
def test_converse_block_policy_stops_the_request(method, detector, bedrock):
    _block_on_pii()

    with pytest.raises(PolicyViolationError):
        getattr(bedrock.client, method)(**CONVERSE_KWARGS)

    assert bedrock.client.calls == []
    (span,) = bedrock.spans()
    assert span.attributes["evaluation.blocked"] is True
    assert span.status.status_code.name == "ERROR"


def test_block_policy_holds_when_the_only_pii_is_in_the_system_prompt(detector, bedrock):
    _block_on_pii()
    kwargs = {
        "modelId": "amazon.nova-micro-v1:0",
        "system": [{"text": "The customer's card is {0}.".format(CARD)}],
        "messages": [{"role": "user", "content": [{"text": "What do you know about me?"}]}],
    }

    with pytest.raises(PolicyViolationError):
        bedrock.client.converse(**kwargs)

    assert bedrock.client.calls == []


def test_invoke_model_block_policy_holds_when_the_span_is_sampled_out(detector):
    """Whether a request is blocked must not depend on trace sampling."""
    from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

    with patch.dict("sys.modules", {"boto3": MagicMock()}):
        instrumentor = AWSBedrockInstrumentor()
    instrumentor.tracer = TracerProvider(sampler=ALWAYS_OFF).get_tracer("test")
    instrumentor.config = OTelConfig(service_name="t")
    instrumentor._instrumented = True
    client = _Client()
    instrumentor._instrument_bedrock_client(client)
    _block_on_pii()

    with pytest.raises(PolicyViolationError):
        client.invoke_model(**_invoke_kwargs("claude"))

    assert client.calls == []


def test_clean_requests_reach_aws_exactly_once(detector, bedrock):
    _block_on_pii()
    clean = {
        "modelId": "amazon.nova-micro-v1:0",
        "messages": [{"role": "user", "content": [{"text": "What is the capital of France?"}]}],
    }
    body = json.dumps({"prompt": "What is the capital of France?"})

    bedrock.client.converse(**clean)
    bedrock.client.converse_stream(**clean)
    bedrock.client.invoke_model(modelId="meta.llama3-8b-instruct-v1:0", body=body)
    bedrock.client.invoke_model_with_response_stream(
        modelId="meta.llama3-8b-instruct-v1:0", body=body
    )

    assert [name for name, _ in bedrock.client.calls] == [
        "converse",
        "converse_stream",
        "invoke_model",
        "invoke_model_with_response_stream",
    ]


def test_embedding_input_is_not_treated_as_a_prompt(detector, bedrock):
    """An embeddings body is a document being indexed, not a prompt."""
    _block_on_pii()
    body = json.dumps({"inputText": USER_TEXT})

    bedrock.client.invoke_model(modelId="amazon.titan-embed-text-v2:0", body=body)

    assert len(bedrock.client.calls) == 1


def test_a_body_that_is_not_json_does_not_break_the_call(detector, bedrock):
    _block_on_pii()

    bedrock.client.invoke_model(modelId="some.model", body="not json at all")

    assert len(bedrock.client.calls) == 1


RESPONSE_BODIES = {
    "claude": {"content": [{"type": "text", "text": "Noted. "}, {"type": "text", "text": "Done."}]},
    "nova": {"output": {"message": {"content": [{"text": "Noted. Done."}]}}},
    "titan_text": {"results": [{"outputText": "Noted. Done."}]},
    "llama": {"generation": "Noted. Done."},
    "mistral": {"outputs": [{"text": "Noted. Done."}]},
    "cohere_command_r": {"text": "Noted. Done."},
    "cohere_command": {"generations": [{"text": "Noted. Done."}]},
}


@pytest.mark.parametrize("family", sorted(RESPONSE_BODIES))
def test_invoke_model_answer_is_read_for_response_checks(family, bedrock):
    data = json.dumps(RESPONSE_BODIES[family]).encode("utf-8")
    result = {"body": SimpleNamespace(_genai_otel_bytes=data)}

    assert bedrock.inst._extract_response_for_eval(result) == "Noted. Done."


def test_converse_answer_is_read_for_response_checks(bedrock):
    result = {"output": {"message": {"content": [{"text": "Noted. "}, {"text": "Done."}]}}}

    assert bedrock.inst._extract_response_for_eval(result) == "Noted. Done."


def test_response_checks_see_the_bedrock_answer(detector, bedrock):
    """Hallucination detection needs the answer; it is read from the result, so it
    does not depend on content capture being on."""
    seen = []

    def detect(response, **kwargs):
        seen.append(response)
        return SimpleNamespace(
            has_hallucination=False,
            hallucination_score=0.0,
            hallucination_indicators=[],
            citation_count=0,
            hedge_words_count=0,
            factual_claim_count=0,
            unsupported_claims=[],
        )

    BaseInstrumentor._hallucination_detector = SimpleNamespace(detect=detect)

    bedrock.client.converse(**CONVERSE_KWARGS)
    bedrock.client.invoke_model(**_invoke_kwargs("claude"))
    assert drain_evaluations(30)

    assert seen == ["Noted.", "Noted."]
