"""The decision normaliser: both protocols reduce to one shape and one attribute set."""

import json
from types import SimpleNamespace

import pytest

from genai_otel.instrumentors import decision as d

OPENAI_QUESTIONS = [
    {
        "type": "choice",
        "name": "next_tool",
        "instructions": "Pick the tool",
        "choices": [{"value": "search"}, {"value": "refund"}, {"value": "escalate"}],
    }
]
OPENAI_ANSWERS = [
    {
        "type": "choice",
        "name": "next_tool",
        "choice": "refund",
        "confidence": 0.81,
        "probabilities": [
            {"value": "refund", "probability": 0.85},
            {"value": "search", "probability": 0.12},
            {"value": "escalate", "probability": 0.03},
        ],
    }
]
SYSTEMONE_QUESTIONS = {
    "next_tool": {
        "type": "choice",
        "instructions": "Pick the tool",
        "criteria": {"search": None, "refund": None, "escalate": None},
    }
}
SYSTEMONE_ANSWERS = {
    "next_tool": {
        "type": "choice",
        "choice": "refund",
        "confidence": 0.81,
        "probabilities": {"refund": 0.85, "search": 0.12, "escalate": 0.03},
    }
}


def _request(protocol, questions, model="m"):
    return d.DecisionRequest(
        protocol=protocol,
        system="openai",
        operation="op",
        model=model,
        provider=None,
        questions=questions,
    )


def _summary(attrs):
    return {k: v for k, v in attrs.items() if k.startswith("genai.decision.")}


class TestBothProtocolsAgree:
    def test_openai_and_systemone_shapes_produce_the_same_summary(self):
        oq = d.openai_questions(OPENAI_QUESTIONS)
        sq = d.systemone_questions(SYSTEMONE_QUESTIONS)
        o = d.response_attributes(
            _request(d.PROTOCOL_OPENAI_DECISIONS, oq),
            d.openai_answers(OPENAI_ANSWERS, oq),
            d.DecisionUsage(input_tokens=10, output_tokens=0),
            "m",
            capture=True,
        )
        s = d.response_attributes(
            _request(d.PROTOCOL_SYSTEMONE, sq),
            d.systemone_answers({"answers": SYSTEMONE_ANSWERS}, sq),
            d.DecisionUsage(input_tokens=10, output_tokens=0),
            "m",
            capture=True,
        )

        assert _summary(o) == _summary(s)
        assert o["genai.decision.choice"] == "refund"
        assert o["genai.decision.option_count"] == 3
        assert o["genai.decision.margin"] == pytest.approx(0.73)


class TestQuestions:
    def test_noul_is_recorded_as_predicate(self):
        (q,) = d.systemone_questions({"urgent": {"type": "noul", "instructions": "Urgent?"}})
        assert q.type == "predicate" and q.option_count is None

    def test_score_options_are_its_levels(self):
        (q,) = d.openai_questions([{"type": "score", "levels": [{"label": "a"}, {"label": "b"}]}])
        assert q.option_count == 2

    def test_systemone_score_options_are_its_criteria(self):
        (q,) = d.systemone_questions({"s": {"type": "score", "criteria": ["low", "mid", "high"]}})
        assert q.option_count == 3

    def test_questions_as_sdk_objects(self):
        (q,) = d.systemone_questions({"x": SimpleNamespace(type="choice", criteria={"a": 1})})
        assert q.type == "choice" and q.option_count == 1

    def test_a_generator_of_questions_is_not_consumed(self):
        questions = (q for q in OPENAI_QUESTIONS)
        assert d.openai_questions(questions) == []
        assert next(questions)["name"] == "next_tool"

    def test_questions_of_picks_the_shape(self):
        assert d.questions_of(SYSTEMONE_QUESTIONS)[0].name == "next_tool"
        assert d.questions_of(OPENAI_QUESTIONS)[0].name == "next_tool"
        assert d.systemone_questions(["not", "a", "mapping"]) == []

    def test_unknown_type(self):
        (q,) = d.openai_questions([{}])
        assert q.type == "unknown"


class TestAnswers:
    def test_predicate_probability(self):
        (a,) = d.openai_answers([{"type": "predicate", "probability": 0.9}], [])
        assert a.type == "predicate" and a.probability == 0.9

    def test_refusal(self):
        (a,) = d.openai_answers([{"type": "refusal", "name": None}], [])
        assert a.type == "refusal"

    def test_answer_name_falls_back_to_the_question(self):
        qs = d.openai_questions([{"type": "predicate", "name": "fraud"}])
        (a,) = d.openai_answers([{"type": "predicate", "probability": 0.2}], qs)
        assert a.name == "fraud"

    def test_boolean_choice_value_is_text(self):
        (a,) = d.openai_answers([{"type": "choice", "choice": True, "confidence": 1.0}], [])
        assert a.choice == "true"

    def test_untagged_systemone_answers_are_inferred(self):
        answers = d.systemone_answers(
            {
                "answers": {
                    "urgent": SimpleNamespace(noul=0.91),
                    "team": SimpleNamespace(choice="billing", confidence=0.6, probabilities={}),
                    "sev": SimpleNamespace(score=1.7, confidence=0.5, probabilities={"0": 0.5}),
                    "dog": SimpleNamespace(box={"top": 0}, confidence=0.4),
                }
            },
            [],
        )
        types = {a.name: a.type for a in answers}
        assert types == {"urgent": "predicate", "team": "choice", "sev": "score", "dog": "box"}

    def test_per_type_maps_are_read_when_there_is_no_answers_map(self):
        answers = d.systemone_answers(
            {"nouls": {"urgent": 0.91}, "choices": {"team": {"choice": "a", "confidence": 0.7}}},
            d.systemone_questions({"team": {"type": "choice"}, "urgent": {"type": "noul"}}),
        )
        assert [a.name for a in answers] == ["team", "urgent"]
        assert answers[1].probability == 0.91
        assert answers[0].choice == "a"

    def test_unknown_answer(self):
        (a,) = d.systemone_answers({"answers": {"x": {"foo": 1}}}, [])
        assert a.type == "unknown"

    def test_granite_softmax_over_the_label_logprobs(self):
        import math

        top = [
            SimpleNamespace(token="A", logprob=math.log(0.6)),
            SimpleNamespace(token="B", logprob=math.log(0.3)),
            SimpleNamespace(token="C", logprob=math.log(0.1)),
        ]
        result = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    logprobs=SimpleNamespace(content=[SimpleNamespace(top_logprobs=top)])
                )
            ]
        )
        (a,) = d.granite_answers(result, 3)
        assert a.choice == "A"
        assert sum(a.probabilities) == pytest.approx(1.0)
        assert a.probabilities[0] == pytest.approx(0.6)

    def test_granite_without_logprobs_has_no_distribution(self):
        (a,) = d.granite_answers(SimpleNamespace(choices=[SimpleNamespace(logprobs=None)]), None)
        assert a.choice is None and a.probabilities == []

    def test_granite_keeps_only_the_labels(self):
        top = [SimpleNamespace(token=t, logprob=-i) for i, t in enumerate("ABCD")]
        result = {"choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}]}
        (a,) = d.granite_answers(result, 2)
        assert len(a.probabilities) == 2


class TestSummary:
    def _attrs(self, answers, questions=None, capture=True):
        questions = questions if questions is not None else []
        return d.response_attributes(
            _request(d.PROTOCOL_OPENAI_DECISIONS, questions),
            answers,
            d.DecisionUsage(),
            None,
            capture=capture,
        )

    def test_min_confidence_ignores_answers_without_a_confidence(self):
        attrs = self._attrs(
            [
                d.DecisionAnswer(name="a", type="predicate", probability=0.1),
                d.DecisionAnswer(name="b", type="choice", confidence=0.4),
                d.DecisionAnswer(name="c", type="score", confidence=0.9),
                d.DecisionAnswer(name="d", type="choice"),
                d.DecisionAnswer(name="e", type="box", confidence=0.01),
            ]
        )
        assert attrs["genai.decision.min_confidence"] == 0.4

    def test_no_confidence_at_all_means_no_min_confidence(self):
        attrs = self._attrs([d.DecisionAnswer(name="a", type="predicate", probability=0.1)])
        assert "genai.decision.min_confidence" not in attrs

    def test_margin_needs_two_probabilities(self):
        attrs = self._attrs([d.DecisionAnswer(name="a", type="choice", probabilities=[1.0])])
        assert "genai.decision.margin" not in attrs
        assert attrs["genai.decision.option_count"] == 1

    def test_single_question_scalars(self):
        attrs = self._attrs([d.DecisionAnswer(name="s", type="score", score=1.7, confidence=0.5)])
        assert attrs["genai.decision.score"] == 1.7
        attrs = self._attrs([d.DecisionAnswer(name="p", type="predicate", probability=0.2)])
        assert attrs["genai.decision.probability"] == 0.2

    def test_several_questions_carry_no_single_answer_scalars(self):
        attrs = self._attrs(
            [
                d.DecisionAnswer(name="a", type="choice", choice="x", confidence=0.5),
                d.DecisionAnswer(name="b", type="refusal"),
            ]
        )
        assert "genai.decision.choice" not in attrs
        assert attrs["genai.decision.question_count"] == 2
        assert attrs["genai.decision.refusal_count"] == 1
        assert attrs["genai.decision.question.types"] == ["choice", "refusal"]

    def test_a_refused_question_is_recorded_as_a_refusal(self):
        questions = d.openai_questions([{"type": "predicate", "name": "q"}])
        attrs = self._attrs([d.DecisionAnswer(name="q", type="refusal")], questions)
        assert attrs["genai.decision.question.types"] == ["refusal"]
        assert "genai.decision.probability" not in attrs

    def test_capture_off_records_nothing_about_the_answer(self):
        questions = d.openai_questions(OPENAI_QUESTIONS)
        attrs = self._attrs(d.openai_answers(OPENAI_ANSWERS, questions), questions, capture=False)
        for key in (
            "genai.decision.choice",
            "genai.decision.confidence",
            "genai.decision.min_confidence",
            "genai.decision.margin",
            "genai.decision.question.names",
            "output.value",
        ):
            assert key not in attrs
        assert attrs["genai.decision.option_count"] == 3
        assert attrs["genai.decision.question.types"] == ["choice"]

    def test_tokens_only_when_reported(self):
        attrs = d.response_attributes(
            _request(d.PROTOCOL_OPENAI_DECISIONS, []), [], d.DecisionUsage(), None, False
        )
        assert not any("token" in k for k in attrs)
        attrs = d.response_attributes(
            _request(d.PROTOCOL_OPENAI_DECISIONS, []),
            [],
            d.DecisionUsage(input_tokens=5, output_tokens=0),
            None,
            False,
        )
        assert attrs["decision.token_count.output"] == 0
        assert attrs["gen_ai.usage.input_tokens"] == 5

    def test_model_names_only_split_when_they_differ(self):
        attrs = d.response_attributes(_request("p", [], "m"), [], d.DecisionUsage(), "m", False)
        assert attrs["decision.model_name"] == "m"
        assert "decision.request.model_name" not in attrs

    def test_request_attributes(self):
        request = _request(d.PROTOCOL_SYSTEMONE, d.systemone_questions(SYSTEMONE_QUESTIONS))
        request.provider = "liquid"
        attrs = d.request_attributes(request)
        assert attrs["openinference.span.kind"] == "DECISION"
        assert attrs["decision.provider"] == "liquid"
        assert attrs["genai.decision.option_count"] == 3
        request.legacy_llm_kind = True
        assert d.request_attributes(request)["openinference.span.kind"] == "LLM"
        assert d.request_list_attributes(request, capture=False) == {
            "genai.decision.question.types": ["choice"]
        }
        assert d.request_list_attributes(request, capture=True)[
            "genai.decision.question.names"
        ] == ["next_tool"]


class TestUsage:
    def test_both_spellings(self):
        assert d.usage_of({"input_tokens": 3, "output_tokens": 1}).input_tokens == 3
        usage = d.usage_of(
            SimpleNamespace(
                prompt_tokens=7,
                completion_tokens=2,
                prompt_tokens_details=SimpleNamespace(cached_tokens=4),
            )
        )
        assert (usage.input_tokens, usage.output_tokens, usage.cached_input_tokens) == (7, 2, 4)

    def test_absent_and_invalid(self):
        assert not d.usage_of(None).reported
        assert d.usage_of({"input_tokens": True, "output_tokens": -1}).reported is False
        assert d.usage_of({"input_tokens": float("nan")}).input_tokens is None


class TestProvider:
    @pytest.mark.parametrize(
        "url,provider",
        [
            ("https://api.openai.com/v1", "openai"),
            ("https://openrouter.ai/api/v1", "openrouter"),
            ("https://api.liquid.ai/decisions", "liquid"),
            ("https://api.typesafe.ai", "typesafe"),
            ("https://api.cloudflare.com/client/v4", "cloudflare"),
            ("https://my-res.openai.azure.com/openai", "azure"),
            ("http://localhost:8000/v1", None),
            (None, None),
        ],
    )
    def test_from_the_host(self, url, provider):
        assert d.provider_for_base_url(url) == provider

    def test_configured_names_an_unknown_host_only(self):
        assert d.provider_for_base_url("http://vllm:8000/v1", "nebius") == "nebius"
        assert d.provider_for_base_url("https://api.openai.com/v1", "nebius") == "openai"
        assert d.provider_for_base_url("http://vllm:8000", "  ") is None


class TestContent:
    def test_images_are_omitted(self):
        payload = {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Is this a cat?"},
                        {"type": "input_image", "image_url": "https://x/cat.png", "detail": "low"},
                    ],
                }
            ],
            "state": "data:image/png;base64,AAAA",
            "blob": b"\x89PNG",
        }
        text = d.payload_json(payload)
        assert "cat.png" not in text and "AAAA" not in text
        loaded = json.loads(text)
        assert loaded["input"][0]["content"][1] == {
            "type": "input_image",
            "image_url": d.IMAGE_OMITTED,
            "detail": "low",
        }
        assert loaded["state"] == d.IMAGE_OMITTED
        assert loaded["blob"] == d.IMAGE_OMITTED
        assert "Is this a cat?" in text

    def test_truncated(self):
        assert len(d.payload_json({"a": "x" * 100}, 20)) == 20

    def test_sdk_objects(self):
        class Image:
            pass

        class Model:
            def model_dump(self):
                return {"k": 1}

        class Plain:
            def __init__(self):
                self.visible = 1
                self._hidden = 2

        assert d.to_jsonable(Image()) == d.IMAGE_OMITTED
        assert d.to_jsonable(Model()) == {"k": 1}
        assert d.to_jsonable(Plain()) == {"visible": 1}
        assert d.to_jsonable(d.DecisionUsage(input_tokens=1))["input_tokens"] == 1
        generator = (i for i in range(3))
        assert isinstance(d.to_jsonable(generator), str)
        assert next(generator) == 0

    def test_a_model_dump_that_raises_falls_through(self):
        class Broken:
            def model_dump(self):
                raise RuntimeError

            def dict(self):
                return {"ok": True}

        assert d.to_jsonable(Broken()) == {"ok": True}

    def test_struct_without_msgspec_or_dict(self):
        assert d.to_jsonable(object()).startswith("<object")


class TestBaseInstrumentorPieces:
    def _instrumentor(self):
        class Plain(d.BaseDecisionInstrumentor):
            def instrument(self, config):
                self.config = config

        return Plain()

    def test_usage_with_cache_write(self):
        usage = self._instrumentor()._extract_usage(
            {
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 0,
                    "input_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 3},
                }
            }
        )
        assert usage["cache_read_input_tokens"] == 2
        assert usage["cache_creation_input_tokens"] == 3

    def test_without_a_config_nothing_is_captured(self):
        instrumentor = self._instrumentor()
        assert instrumentor.config is None
        assert instrumentor._capture() is False
        assert instrumentor._max_length() == 0
        assert instrumentor._configured_provider() is None

    def test_eval_prompt_shapes(self):
        instrumentor = self._instrumentor()
        assert instrumentor._extract_prompt_for_eval(None) is None
        assert instrumentor._extract_prompt_for_eval({"input": "hello"}) == "hello"
        assert instrumentor._extract_prompt_for_eval({"messages": [{"content": "hi"}]}) == "hi"
        assert "s1" in instrumentor._extract_prompt_for_eval({"state": "s1"})

    def test_no_payload_no_content(self):
        assert d.request_content_attributes(_request("p", []), 0) == {}
        assert d._label(None) is None
