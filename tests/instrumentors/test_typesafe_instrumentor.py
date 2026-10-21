import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from genai_otel.config import OTelConfig
from genai_otel.exceptions import PolicyViolationError
from genai_otel.instrumentors.base import BaseInstrumentor
from genai_otel.instrumentors.typesafe_instrumentor import TypeSafeInstrumentor


def _instrumentor(capture=True, **config):
    instrumentor = TypeSafeInstrumentor()
    instrumentor._setup_config(
        OTelConfig(
            enabled_instrumentors=["typesafe"],
            enable_content_capture=capture,
            content_max_length=0,
            enable_gpu_metrics=False,
            enable_mcp_instrumentation=False,
            **config,
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
    # A decision span since 1.38, with the pre-1.38 llm.* attributes kept.
    assert attrs["openinference.span.kind"] == "DECISION"
    assert attrs["gen_ai.request.type"] == "decision"
    assert attrs["decision.system"] == "typesafe"
    assert attrs["llm.request.model_name"] == "jev-1.13.0"
    assert json.loads(attrs["input.value"])["state"]["ticket"] == "The export page crashes"
    assert json.loads(attrs["gen_ai.request.first_message"])["role"] == "user"


def test_legacy_flag_keeps_the_llm_kind():
    attrs = _instrumentor(decision_legacy_llm_kind=True)._extract_request_attributes(
        None, ("state", {}), {}
    )
    assert attrs["openinference.span.kind"] == "LLM"
    assert attrs["genai.decision.protocol"] == "systemone"


def test_invocation_parameters_keep_extra_body():
    attrs = _instrumentor()._extract_request_attributes(
        None, (), {"state": "s", "questions": {}, "extra_body": {"seed": 1}}
    )
    assert json.loads(attrs["llm.invocation_parameters"]) == {
        "model": "jev-latest",
        "extra_body": {"seed": 1},
    }


def test_provider_comes_from_the_client_base_url():
    instance = SimpleNamespace(_config=SimpleNamespace(base_url="https://api.liquid.ai/decisions"))
    attrs = _instrumentor()._extract_request_attributes(instance, ("s", {}), {})
    assert attrs["decision.provider"] == "liquid"
    attrs = _instrumentor()._extract_request_attributes(
        SimpleNamespace(base_url="http://localhost:30000"), ("s", {}), {}
    )
    assert "decision.provider" not in attrs


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
    assert json.loads(attrs["gen_ai.response"])["urgent"]["noul"] == 0.91
    assert usage == {"prompt_tokens": 342, "completion_tokens": 31, "total_tokens": 373}


def test_noul_is_recorded_as_predicate_and_answered_from_an_untagged_struct():
    instrumentor = _instrumentor()
    request = instrumentor._decision_request(
        None, (), {"state": "s", "questions": {"urgent": {"type": "noul"}}}
    )
    attrs = instrumentor._decision_response_attributes(
        request,
        SimpleNamespace(
            model="jev-1",
            answers={"urgent": SimpleNamespace(noul=0.91)},
            usage=SimpleNamespace(input_tokens=1, output_tokens=0),
        ),
    )
    assert attrs["genai.decision.question.types"] == ["predicate"]
    assert attrs["genai.decision.probability"] == 0.91
    assert json.loads(attrs["output.value"])["answers"]["urgent"]["noul"] == 0.91


def test_request_and_response_capture_is_opt_in():
    instrumentor = _instrumentor(capture=False)
    attrs = instrumentor._extract_request_attributes(None, ("state", {}), {})
    response_attrs = instrumentor._extract_response_attributes(
        {"model": "jev-latest", "answers": {}, "usage": {}}
    )

    assert "input.value" not in attrs
    assert "gen_ai.request.first_message" not in attrs
    assert "output.value" not in response_attrs
    assert "gen_ai.response" not in response_attrs


def test_prompt_for_evaluation_uses_state_and_questions():
    prompt = _instrumentor()._extract_prompt_for_eval(
        {"state": {"email": "alice@example.com"}, "questions": {"pii": "detect"}}
    )

    assert "alice@example.com" in prompt
    assert "pii" in prompt


def test_prompt_for_evaluation_accepts_the_operation_the_wrapper_passes():
    prompt = _instrumentor()._extract_prompt_for_eval(
        {"state": {"email": "alice@example.com"}, "questions": {"pii": "detect"}},
        "typesafe.system_one",
    )

    assert "alice@example.com" in prompt


def test_prompt_for_evaluation_without_state_falls_back():
    assert _instrumentor()._extract_prompt_for_eval({"prompt": "hi"}) == "hi"


def test_not_instrumented_without_the_sdk():
    instrumentor = TypeSafeInstrumentor()
    instrumentor._typesafe_available = False
    instrumentor.instrument(OTelConfig())
    assert instrumentor._instrumented is False


def test_block_policy_stops_a_typesafe_request():
    """The pre-call check reads the prompt the way the shared wrapper asks for it."""
    seen = []

    def detect(text):
        seen.append(text)
        return SimpleNamespace(has_pii=True, entity_counts={"EMAIL_ADDRESS": 1}, score=0.9)

    BaseInstrumentor._pii_detector = SimpleNamespace(detect=detect)
    BaseInstrumentor._blocking_policies = frozenset({"pii"})
    BaseInstrumentor._blocking_active = True

    with pytest.raises(PolicyViolationError):
        _instrumentor()._enforce_precall_policy(
            MagicMock(),
            {"state": {"email": "alice@example.com"}, "questions": {"pii": "detect"}},
            "typesafe.system_one",
        )

    assert len(seen) == 1 and "alice@example.com" in seen[0]


def test_instrument_wraps_once_and_tolerates_a_missing_async_client(monkeypatch):
    import sys
    import types

    calls = []
    fake = types.ModuleType("wrapt_fake")

    def wrap(module, name, wrapper):
        calls.append(name)
        if name.startswith("Async"):
            raise AttributeError(name)

    fake.wrap_function_wrapper = wrap
    monkeypatch.setitem(sys.modules, "wrapt", fake)
    instrumentor = TypeSafeInstrumentor()
    instrumentor._typesafe_available = True
    instrumentor.instrument(OTelConfig())
    instrumentor.instrument(OTelConfig())
    assert calls == ["TypeSafeClient.system_one", "AsyncTypeSafeClient.system_one"]
    assert instrumentor._instrumented is True


def test_instrument_failure_raises_only_under_fail_on_error(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("wrapt_fake")
    fake.wrap_function_wrapper = lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setitem(sys.modules, "wrapt", fake)
    instrumentor = TypeSafeInstrumentor()
    instrumentor._typesafe_available = True
    instrumentor.instrument(OTelConfig())
    assert instrumentor._instrumented is False
    with pytest.raises(RuntimeError):
        instrumentor.instrument(OTelConfig(fail_on_error=True))
