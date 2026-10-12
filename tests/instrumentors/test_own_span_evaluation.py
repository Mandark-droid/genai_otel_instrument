"""Detectors and block policies apply on every generation path, not only the shared wrapper.

Several instrumentors build their own spans. Those paths never called the
evaluation hooks, so a call through them carried no ``evaluation.*`` attributes and
a block policy let the request through. The OpenAI Responses API had the same
problem for a different reason: its prompt is ``input=`` / ``instructions=``, which
the hooks did not read.
"""

import asyncio
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genai_otel.config import OTelConfig
from genai_otel.exceptions import PolicyViolationError
from genai_otel.instrumentors.anthropic_instrumentor import AnthropicInstrumentor
from genai_otel.instrumentors.aws_bedrock_instrumentor import AWSBedrockInstrumentor
from genai_otel.instrumentors.azure_openai_instrumentor import AzureOpenAIInstrumentor
from genai_otel.instrumentors.base import BaseInstrumentor, drain_evaluations
from genai_otel.instrumentors.groq_instrumentor import GroqInstrumentor
from genai_otel.instrumentors.hyperbolic_instrumentor import HyperbolicInstrumentor
from genai_otel.instrumentors.litellm_latency_instrumentor import LiteLLMLatencyInstrumentor
from genai_otel.instrumentors.llamacpp_instrumentor import LlamaCppInstrumentor
from genai_otel.instrumentors.mistralai_instrumentor import MistralAIInstrumentor
from genai_otel.instrumentors.openai_instrumentor import OpenAIInstrumentor
from genai_otel.instrumentors.vllm_instrumentor import VLLMInstrumentor

CARD = "4111 1111 1111 1111"
MESSAGES = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "My card number is {0}.".format(CARD)},
]


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


def _chat_result():
    message = SimpleNamespace(content="Noted.", role="assistant", tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
    usage = SimpleNamespace(prompt_tokens=12, completion_tokens=2, total_tokens=14)
    return SimpleNamespace(id="chatcmpl-1", model="m", choices=[choice], usage=usage)


@pytest.fixture
def detector():
    """A PII detector in the class-level slot, with every other slot cleared."""
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


@pytest.fixture
def tracing():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    def spans():
        # The detectors run after the call has returned, and the span is
        # exported when they are done.
        assert drain_evaluations(30)
        return exporter.get_finished_spans()

    return provider.get_tracer("test"), spans


def _ready(instrumentor, tracer):
    instrumentor.tracer = tracer
    instrumentor.config = OTelConfig(service_name="t")
    instrumentor._instrumented = True
    return instrumentor


# Each driver wires one instrumentor around a fake provider call and returns a
# function that makes that call. ``calls`` records what reached the provider.


def _groq(tracer, calls):
    with patch.dict("sys.modules", {"groq": MagicMock()}):
        instrumentor = _ready(GroqInstrumentor(), tracer)

    def create(*args, **kwargs):
        calls.append(kwargs)
        return _chat_result()

    completions = SimpleNamespace(create=create)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    instrumentor._instrument_client(client)
    return lambda: client.chat.completions.create(model="m", messages=MESSAGES)


def _azure_openai(tracer, calls):
    class OpenAIClient:
        def complete(self, *args, **kwargs):
            calls.append(kwargs)
            return _chat_result()

    module = types.ModuleType("azure.ai.openai")
    module.OpenAIClient = OpenAIClient
    modules = {"azure": MagicMock(), "azure.ai": MagicMock(), "azure.ai.openai": module}
    with patch.dict("sys.modules", modules):
        instrumentor = AzureOpenAIInstrumentor()
        instrumentor.tracer = tracer
        instrumentor.instrument(OTelConfig(service_name="t"))
    client = OpenAIClient()
    return lambda: client.complete(model="m", messages=MESSAGES)


def _mistral(tracer, calls):
    with patch.dict("sys.modules", {"mistralai": MagicMock()}):
        instrumentor = _ready(MistralAIInstrumentor(), tracer)

    def complete(*args, **kwargs):
        calls.append(kwargs)
        return _chat_result()

    kwargs = {"model": "m", "messages": MESSAGES}
    return lambda: instrumentor._wrap_chat_complete(complete, None, (), dict(kwargs))


def _mistral_stream(tracer, calls):
    with patch.dict("sys.modules", {"mistralai": MagicMock()}):
        instrumentor = _ready(MistralAIInstrumentor(), tracer)

    def stream(*args, **kwargs):
        calls.append(kwargs)
        return iter(())

    kwargs = {"model": "m", "messages": MESSAGES}
    return lambda: instrumentor._wrap_chat_stream(stream, None, (), dict(kwargs))


def _hyperbolic(tracer, calls):
    body = {
        "id": "x",
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": "Noted."}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
    }

    def post(url, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status_code=200, json=lambda: body)

    fake_requests = types.ModuleType("requests")
    fake_requests.post = post
    with patch.dict("sys.modules", {"requests": fake_requests}):
        instrumentor = HyperbolicInstrumentor()
        instrumentor.tracer = tracer
        instrumentor.instrument(OTelConfig(service_name="t"))
    wrapped_post = fake_requests.post
    url = HyperbolicInstrumentor.HYPERBOLIC_API_BASE + "/v1/chat/completions"
    return lambda: wrapped_post(url, json={"model": "m", "messages": MESSAGES})


def _litellm(tracer, calls):
    with patch.dict("sys.modules", {"litellm": MagicMock()}):
        instrumentor = _ready(LiteLLMLatencyInstrumentor(), tracer)

    def completion(*args, **kwargs):
        calls.append(kwargs)
        return _chat_result()

    wrapped = instrumentor._wrap_sync(completion, "litellm.completion")
    return lambda: wrapped(model="m", messages=MESSAGES)


def _litellm_async(tracer, calls):
    with patch.dict("sys.modules", {"litellm": MagicMock()}):
        instrumentor = _ready(LiteLLMLatencyInstrumentor(), tracer)

    async def acompletion(*args, **kwargs):
        calls.append(kwargs)
        return _chat_result()

    wrapped = instrumentor._wrap_async(acompletion, "litellm.acompletion")
    return lambda: asyncio.run(wrapped(model="m", messages=MESSAGES))


def _anthropic_stream(tracer, calls):
    with patch.dict("sys.modules", {"anthropic": MagicMock()}):
        instrumentor = _ready(AnthropicInstrumentor(), tracer)

    def stream(*args, **kwargs):
        calls.append(kwargs)
        return MagicMock()

    client = SimpleNamespace(messages=SimpleNamespace(stream=stream))
    instrumentor._instrument_stream_method(client)
    return lambda: client.messages.stream(model="m", messages=MESSAGES, max_tokens=8)


def _llamacpp_stream(tracer, calls):
    with patch.dict("sys.modules", {"llama_cpp": MagicMock()}):
        instrumentor = _ready(LlamaCppInstrumentor(), tracer)

    def create_chat_completion(self, *args, **kwargs):
        calls.append(kwargs)
        return iter(())

    def extractor(instance, args, kwargs):
        return {"gen_ai.system": "llamacpp"}

    class Llama:
        pass

    Llama.create_chat_completion = instrumentor._stream_aware(
        create_chat_completion, "llamacpp.chat", extractor
    )
    model = Llama()
    return lambda: list(model.create_chat_completion(messages=MESSAGES, stream=True))


def _vllm_generate(tracer, calls):
    with patch.dict("sys.modules", {"vllm": MagicMock()}):
        instrumentor = _ready(VLLMInstrumentor(), tracer)

    async def generate(self, prompt, *args, **kwargs):
        calls.append(prompt)
        yield "token"

    class Engine:
        pass

    Engine.generate = instrumentor._async_generate_wrapper(generate)
    engine = Engine()

    async def consume():
        return [item async for item in engine.generate("My card number is {0}.".format(CARD))]

    return lambda: asyncio.run(consume())


def _bedrock_converse_stream(tracer, calls):
    with patch.dict("sys.modules", {"boto3": MagicMock()}):
        instrumentor = _ready(AWSBedrockInstrumentor(), tracer)

    def converse_stream(*args, **kwargs):
        calls.append(kwargs)
        return {"stream": iter(())}

    wrapped = instrumentor._wrap_converse_stream(converse_stream)
    messages = [{"role": "user", "content": [{"text": "My card number is {0}.".format(CARD)}]}]
    return lambda: wrapped(modelId="m", messages=messages)


NON_STREAMED = [_groq, _azure_openai, _mistral, _hyperbolic, _litellm, _litellm_async]
EVERY_PATH = NON_STREAMED + [
    _mistral_stream,
    _anthropic_stream,
    _llamacpp_stream,
    _vllm_generate,
    _bedrock_converse_stream,
]


@pytest.mark.parametrize("driver", NON_STREAMED, ids=lambda d: d.__name__.strip("_"))
def test_call_is_evaluated(driver, detector, tracing):
    tracer, spans = tracing
    calls = []

    driver(tracer, calls)()

    assert len(calls) == 1
    (span,) = spans()
    assert span.attributes["evaluation.pii.prompt.detected"] is True
    # The user's message is what gets evaluated, not only the first one.
    assert any(CARD in text for text in detector.seen)


@pytest.mark.parametrize("driver", EVERY_PATH, ids=lambda d: d.__name__.strip("_"))
def test_block_policy_stops_the_request(driver, detector, tracing):
    tracer, spans = tracing
    calls = []
    _block_on_pii()

    with pytest.raises(PolicyViolationError):
        driver(tracer, calls)()

    assert calls == [], "the request reached the provider despite the block policy"
    (span,) = spans()
    assert span.attributes["evaluation.blocked"] is True
    assert span.status.status_code.name == "ERROR"


@pytest.mark.parametrize("driver", EVERY_PATH, ids=lambda d: d.__name__.strip("_"))
def test_provider_is_called_exactly_once_when_nothing_blocks(driver, detector, tracing):
    """A block policy that does not fire must not cost a second provider call."""
    tracer, _ = tracing
    calls = []
    BaseInstrumentor._pii_detector = SimpleNamespace(
        detect=lambda text: SimpleNamespace(
            has_pii=False,
            entity_counts={},
            entities=[],
            score=0.0,
            redacted_text=None,
            blocked=False,
        )
    )
    _block_on_pii()

    driver(tracer, calls)()

    assert len(calls) == 1


@pytest.mark.parametrize("driver", NON_STREAMED, ids=lambda d: d.__name__.strip("_"))
def test_failing_detector_does_not_fail_the_call(driver, detector, tracing):
    tracer, _ = tracing
    calls = []

    def broken(text):
        raise RuntimeError("model not loaded")

    BaseInstrumentor._pii_detector = SimpleNamespace(detect=broken)

    driver(tracer, calls)()

    assert len(calls) == 1


# --- OpenAI Responses API: the prompt is `input` / `instructions`, the answer `output_text`


def _responses_result():
    usage = SimpleNamespace(input_tokens=12, output_tokens=2, total_tokens=14)
    return SimpleNamespace(id="resp_1", model="gpt-5.6", output_text="Noted.", usage=usage)


def _openai_responses(tracer, calls, input_value):
    instrumentor = _ready(OpenAIInstrumentor(), tracer)

    def create(*args, **kwargs):
        calls.append(kwargs)
        return _responses_result()

    wrapped = instrumentor.create_span_wrapper(
        "openai.responses", extract_attributes=instrumentor._extract_responses_attributes
    )(create)
    return lambda: wrapped(model="gpt-5.6", input=input_value, instructions="Be terse.")


def _openai_responses_async(tracer, calls, input_value):
    instrumentor = _ready(OpenAIInstrumentor(), tracer)

    async def create(*args, **kwargs):
        calls.append(kwargs)
        return _responses_result()

    with patch("opentelemetry.trace.get_tracer", return_value=tracer):
        wrapped = instrumentor._create_async_span_wrapper(
            span_name="openai.responses",
            extract_attributes=instrumentor._extract_responses_attributes,
        )(create)

    def call():
        with patch("opentelemetry.trace.get_tracer", return_value=tracer):
            return asyncio.run(
                wrapped(model="gpt-5.6", input=input_value, instructions="Be terse.")
            )

    return call


RESPONSES_INPUTS = {
    "string": "My card number is {0}.".format(CARD),
    "items": [{"role": "user", "content": "My card number is {0}.".format(CARD)}],
    "parts": [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "My card number is {0}.".format(CARD)}],
        }
    ],
}


@pytest.mark.parametrize("shape", sorted(RESPONSES_INPUTS))
@pytest.mark.parametrize("driver", [_openai_responses, _openai_responses_async])
def test_responses_call_is_evaluated(driver, shape, detector, tracing):
    tracer, spans = tracing
    calls = []

    driver(tracer, calls, RESPONSES_INPUTS[shape])()

    (span,) = spans()
    assert span.attributes["evaluation.pii.prompt.detected"] is True
    assert any(CARD in text and "Be terse." in text for text in detector.seen)


@pytest.mark.parametrize("shape", sorted(RESPONSES_INPUTS))
@pytest.mark.parametrize("driver", [_openai_responses, _openai_responses_async])
def test_responses_block_policy_stops_the_request(driver, shape, detector, tracing):
    tracer, spans = tracing
    calls = []
    _block_on_pii()

    with pytest.raises(PolicyViolationError):
        driver(tracer, calls, RESPONSES_INPUTS[shape])()

    assert calls == []
    (span,) = spans()
    assert span.attributes["evaluation.blocked"] is True


def test_responses_block_policy_holds_when_the_span_is_sampled_out(detector):
    """Whether a request is blocked must not depend on trace sampling."""
    from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

    tracer = TracerProvider(sampler=ALWAYS_OFF).get_tracer("test")
    calls = []
    _block_on_pii()

    with pytest.raises(PolicyViolationError):
        _openai_responses(tracer, calls, RESPONSES_INPUTS["string"])()

    assert calls == []


def test_embedding_input_is_not_treated_as_a_prompt(detector, tracing):
    """`input=` on an embeddings call is a document being indexed, not a prompt.
    Block policies on prompts did not apply to it before and still do not."""
    tracer, _ = tracing
    calls = []
    _block_on_pii()
    instrumentor = _ready(OpenAIInstrumentor(), tracer)

    def create(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=[], model="text-embedding-3-small", usage=None)

    wrapped = instrumentor.create_span_wrapper(
        "openai.embeddings", extract_attributes=instrumentor._extract_embedding_attributes
    )(create)
    wrapped(model="text-embedding-3-small", input="My card number is {0}.".format(CARD))

    assert len(calls) == 1
