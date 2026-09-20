import json
from types import SimpleNamespace

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.typesafe_instrumentor import TypeSafeInstrumentor


def _instrumentor(capture=True):
    instrumentor = TypeSafeInstrumentor()
    instrumentor._setup_config(
        OTelConfig(
            enabled_instrumentors=["typesafe"],
            enable_content_capture=capture,
            content_max_length=0,
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
        )
    )
    return instrumentor


def test_request_attributes_capture_structured_state_and_questions():
    instrumentor = _instrumentor()
    instance = SimpleNamespace(_config=SimpleNamespace(default_model="jev-1.13.0"))
    questions = {"urgent": SimpleNamespace(type="noul", instructions="Is this urgent?")}

    attrs = instrumentor._extract_request_attributes(
        instance,
        (),
        {"state": {"ticket": "The export page crashes"}, "questions": questions},
    )

    assert attrs["gen_ai.system"] == "typesafe"
    assert attrs["gen_ai.request.model"] == "jev-1.13.0"
    assert attrs["openinference.span.kind"] == "LLM"
    assert json.loads(attrs["input.value"])["state"]["ticket"] == "The export page crashes"
    assert json.loads(attrs["gen_ai.request.first_message"])["role"] == "user"


def test_response_attributes_and_usage_support_sdk_structs():
    instrumentor = _instrumentor()
    response = SimpleNamespace(
        model="jev-1.13.0-20260917",
        answers={"urgent": SimpleNamespace(noul=0.91)},
        usage=SimpleNamespace(input_tokens=342, output_tokens=31),
    )

    attrs = instrumentor._extract_response_attributes(response)
    usage = instrumentor._extract_usage(response)

    assert attrs["llm.response.model_name"] == "jev-1.13.0-20260917"
    assert attrs["llm.token_count.prompt"] == 342
    assert attrs["llm.token_count.completion"] == 31
    assert json.loads(attrs["output.value"])["answers"]["urgent"]["noul"] == 0.91
    assert usage == {"prompt_tokens": 342, "completion_tokens": 31, "total_tokens": 373}


def test_request_and_response_capture_is_opt_in():
    instrumentor = _instrumentor(capture=False)
    attrs = instrumentor._extract_request_attributes(None, ("state", {}), {})
    response_attrs = instrumentor._extract_response_attributes(
        {"model": "jev-latest", "answers": {}, "usage": {}}
    )

    assert "input.value" not in attrs
    assert "gen_ai.request.first_message" not in attrs
    assert "output.value" not in response_attrs


def test_prompt_for_evaluation_uses_state_and_questions():
    prompt = _instrumentor()._extract_prompt_for_eval(
        {"state": {"email": "alice@example.com"}, "questions": {"pii": "detect"}}
    )

    assert "alice@example.com" in prompt
    assert "pii" in prompt
