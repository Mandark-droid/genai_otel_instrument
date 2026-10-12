"""What the toxicity detector loads, and when.

Detoxify brings torch and transformers with it. A process that does not use the
local toxicity model must not pay for them, and one that does can choose a
cheaper form of the model.

No model is loaded here: stand-ins play Detoxify and torch.
"""

import json
import logging
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from genai_otel.evaluation import toxicity_detector
from genai_otel.evaluation.config import ToxicityConfig
from genai_otel.evaluation.toxicity_detector import ToxicityDetector

SCORES = {
    "toxicity": 0.9,
    "severe_toxicity": 0.1,
    "obscene": 0.1,
    "threat": 0.1,
    "insult": 0.8,
    "identity_attack": 0.1,
}

# Reports every attempt to import one of the heavy packages, and refuses it.
_IMPORT_TRAP = """
    import importlib.abc
    import json
    import sys

    attempted = []

    class _Trap(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in ("torch", "detoxify", "transformers", "googleapiclient"):
                attempted.append(name)
                raise ImportError("refused by the test: " + name)
            return None

    sys.meta_path.insert(0, _Trap())
"""


def _run(code):
    """Run ``code`` in a fresh interpreter and return the imports it attempted."""
    script = textwrap.dedent(_IMPORT_TRAP) + textwrap.dedent(code)
    script += "\nprint('ATTEMPTED=' + json.dumps(sorted(set(attempted))))\n"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
        stdin=subprocess.DEVNULL,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("ATTEMPTED=")][-1]
    return json.loads(line[len("ATTEMPTED=") :])


def _heavy(attempted):
    return [name for name in attempted if name.split(".")[0] in ("torch", "detoxify")]


class TestNothingHeavyIsImportedUnlessNeeded:
    def test_importing_the_detectors_imports_no_model_library(self):
        attempted = _run("""
            import genai_otel.evaluation.toxicity_detector
            import genai_otel.evaluation.span_processor
            """)
        assert attempted == []

    def test_regex_detectors_alone_do_not_import_torch(self):
        attempted = _run("""
            from genai_otel.evaluation.config import (
                BiasConfig,
                HallucinationConfig,
                PromptInjectionConfig,
                RestrictedTopicsConfig,
            )
            from genai_otel.evaluation.span_processor import EvaluationSpanProcessor

            processor = EvaluationSpanProcessor(
                bias_config=BiasConfig(enabled=True),
                prompt_injection_config=PromptInjectionConfig(enabled=True),
                restricted_topics_config=RestrictedTopicsConfig(enabled=True),
                hallucination_config=HallucinationConfig(enabled=True),
            )
            assert processor.bias_detector.detect("Women are always bad drivers.").has_bias
            """)
        assert attempted == []

    def test_instrument_with_a_regex_detector_does_not_import_torch(self):
        attempted = _run("""
            import os

            # instrument() reads a .env file when python-dotenv is installed. What a
            # developer keeps in theirs must not decide what this test enables.
            sys.modules["dotenv"] = None

            os.environ["GENAI_ENABLED_INSTRUMENTORS"] = "openai"
            os.environ["GENAI_ENABLE_GPU_METRICS"] = "false"
            os.environ["GENAI_ENABLE_MCP_INSTRUMENTATION"] = "false"
            os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = ""
            os.environ["GENAI_ENABLE_PII_DETECTION"] = "false"
            os.environ["GENAI_ENABLE_TOXICITY_DETECTION"] = "false"
            os.environ["GENAI_ENABLE_PROMPT_INJECTION_DETECTION"] = "true"
            os.environ["GENAI_ENABLE_BIAS_DETECTION"] = "true"

            import genai_otel

            genai_otel.instrument()

            from genai_otel.instrumentors.base import BaseInstrumentor

            assert BaseInstrumentor._bias_detector is not None
            assert BaseInstrumentor._prompt_injection_detector is not None
            assert BaseInstrumentor._toxicity_detector is None
            """)
        assert _heavy(attempted) == []

    def test_the_local_toxicity_model_is_what_imports_it(self):
        """The control: the trap does see the import when the model is asked for."""
        attempted = _run("""
            from genai_otel.evaluation.config import ToxicityConfig
            from genai_otel.evaluation.toxicity_detector import ToxicityDetector

            detector = ToxicityDetector(ToxicityConfig(enabled=True, use_local_model=True))
            assert not detector.is_available()
            """)
        assert attempted == ["detoxify"]

    def test_toxicity_without_the_local_model_does_not_import_it(self):
        attempted = _run("""
            from genai_otel.evaluation.config import ToxicityConfig
            from genai_otel.evaluation.toxicity_detector import ToxicityDetector

            ToxicityDetector(ToxicityConfig(enabled=True, use_local_model=False))
            """)
        assert attempted == []

    def test_the_perspective_client_library_is_imported_only_for_perspective(self):
        attempted = _run("""
            from genai_otel.evaluation.config import ToxicityConfig
            from genai_otel.evaluation.toxicity_detector import ToxicityDetector

            ToxicityDetector(
                ToxicityConfig(
                    enabled=True,
                    use_local_model=False,
                    use_perspective_api=True,
                    perspective_api_key="not-a-real-key",
                )
            )
            """)
        assert attempted == ["googleapiclient"]


def test_a_missing_detoxify_leaves_the_detector_unavailable(caplog):
    with patch.dict(sys.modules, {"detoxify": None}):
        with patch.object(toxicity_detector, "Detoxify", toxicity_detector._NOT_IMPORTED):
            with caplog.at_level(logging.WARNING, logger=toxicity_detector.__name__):
                detector = ToxicityDetector(ToxicityConfig(enabled=True))
    assert detector.is_available() is False
    assert "pip install detoxify" in caplog.text


def _fake_torch(**overrides):
    torch = MagicMock()
    torch.nn.Linear = type("Linear", (), {})
    torch.qint8 = "qint8"
    for name, value in overrides.items():
        setattr(torch, name, value)
    return torch


def _loaded(config, torch):
    """A detector that loaded a stand-in model with a stand-in torch."""
    model = SimpleNamespace(model="float32 weights", predict=Mock(return_value=dict(SCORES)))
    with patch.dict(sys.modules, {"torch": torch}):
        with patch.object(toxicity_detector, "Detoxify", Mock(return_value=model)):
            detector = ToxicityDetector(config)
    return detector, model


class TestQuantisedModel:
    def test_the_model_is_left_as_it_is_by_default(self):
        torch = _fake_torch()
        detector, model = _loaded(ToxicityConfig(enabled=True), torch)

        assert detector.is_available()
        assert model.model == "float32 weights"
        torch.quantization.quantize_dynamic.assert_not_called()

    def test_linear_layers_become_eight_bit_when_asked(self):
        torch = _fake_torch()
        torch.quantization.quantize_dynamic.return_value = "int8 weights"

        detector, model = _loaded(ToxicityConfig(enabled=True, quantize=True), torch)

        assert detector.is_available()
        assert model.model == "int8 weights"
        torch.quantization.quantize_dynamic.assert_called_once_with(
            "float32 weights", {torch.nn.Linear}, dtype="qint8"
        )

    def test_a_torch_without_the_api_keeps_the_model_and_says_so_once(self, caplog):
        torch = _fake_torch(quantization=SimpleNamespace())  # no quantize_dynamic

        with patch.object(toxicity_detector, "_QUANTIZE_FAILURE_LOGGED", False):
            with caplog.at_level(logging.WARNING, logger=toxicity_detector.__name__):
                detector, model = _loaded(ToxicityConfig(enabled=True, quantize=True), torch)
                _loaded(ToxicityConfig(enabled=True, quantize=True), torch)

        assert detector.is_available()
        assert model.model == "float32 weights"
        warnings = [r for r in caplog.records if "GENAI_TOXICITY_QUANTIZE" in r.getMessage()]
        assert len(warnings) == 1

    def test_a_quantised_model_that_cannot_run_is_put_back(self, caplog):
        """Some builds of torch quantise without complaint and fail at the first
        prediction. A detector left like that would answer "not toxic" for good."""
        torch = _fake_torch()
        torch.quantization.quantize_dynamic.return_value = "int8 weights"
        model = SimpleNamespace(model="float32 weights")

        def predict(text):
            if model.model == "int8 weights":
                raise RuntimeError("no quantized engine")
            return dict(SCORES)

        model.predict = predict
        with patch.dict(sys.modules, {"torch": torch}):
            with patch.object(toxicity_detector, "Detoxify", Mock(return_value=model)):
                with patch.object(toxicity_detector, "_QUANTIZE_FAILURE_LOGGED", False):
                    with caplog.at_level(logging.WARNING, logger=toxicity_detector.__name__):
                        detector = ToxicityDetector(ToxicityConfig(enabled=True, quantize=True))

        assert model.model == "float32 weights"
        assert detector.detect("You are stupid and worthless").is_toxic
        assert "GENAI_TOXICITY_QUANTIZE" in caplog.text


class TestTorchThreads:
    def test_the_thread_count_is_not_touched_by_default(self):
        torch = _fake_torch()
        _loaded(ToxicityConfig(enabled=True), torch)
        torch.set_num_threads.assert_not_called()

    def test_the_thread_count_is_set_when_asked(self):
        torch = _fake_torch()
        _loaded(ToxicityConfig(enabled=True, torch_threads=4), torch)
        torch.set_num_threads.assert_called_once_with(4)

    def test_it_is_not_set_when_the_local_model_is_not_used(self):
        torch = _fake_torch()
        with patch.dict(sys.modules, {"torch": torch}):
            ToxicityDetector(ToxicityConfig(enabled=True, use_local_model=False, torch_threads=4))
        torch.set_num_threads.assert_not_called()


def _detector(scores=None, **config):
    with patch.object(ToxicityDetector, "_check_availability"):
        detector = ToxicityDetector(ToxicityConfig(enabled=True, **config))
    detector._detoxify_model = Mock()
    detector._detoxify_model.predict.return_value = dict(scores or SCORES)
    detector._detoxify_available = True
    return detector


class TestResultCache:
    def test_text_already_scored_is_not_scored_again(self):
        detector = _detector()

        first = detector.detect("You are stupid and worthless")
        second = detector.detect("You are stupid and worthless")

        assert detector._detoxify_model.predict.call_count == 1
        assert second.is_toxic is True
        assert second.scores == first.scores
        assert second.toxic_categories == first.toxic_categories
        assert second.original_text == "You are stupid and worthless"

    def test_the_cache_does_not_hold_on_to_the_text(self):
        detector = _detector()
        text = "A long internal memo that nobody else should keep a copy of."
        detector.detect(text)
        assert len(detector._cache) == 1
        assert text not in repr(detector._cache._entries)

    def test_a_failed_scoring_is_not_remembered(self):
        detector = _detector()
        detector._detoxify_model.predict.side_effect = [RuntimeError("out of memory"), SCORES]

        assert detector.detect("You are stupid and worthless").is_toxic is False
        assert detector.detect("You are stupid and worthless").is_toxic is True

    def test_a_changed_threshold_is_not_answered_from_the_cache(self):
        detector = _detector()
        assert detector.detect("You are stupid and worthless").is_toxic is True
        detector.config.threshold = 0.95
        assert detector.detect("You are stupid and worthless").is_toxic is False
        assert detector._detoxify_model.predict.call_count == 2

    def test_the_cache_can_be_turned_off(self):
        detector = _detector(cache_size=0)
        detector.detect("Same text.")
        detector.detect("Same text.")
        assert detector._detoxify_model.predict.call_count == 2


class TestFallbackResultsAndTheCache:
    """When Perspective cannot answer, the local model does. That answer is the
    local model's: remembered as Perspective's, it would go on being given after
    Perspective had come back."""

    @staticmethod
    def _both_backends(perspective_answers):
        detector = _detector(use_perspective_api=True, perspective_api_key="not-a-real-key")
        detector._perspective_available = True
        detector._egress_allowed = True
        detector._perspective_client = MagicMock()
        execute = detector._perspective_client.comments.return_value.analyze.return_value.execute
        execute.side_effect = perspective_answers
        return detector, execute

    @staticmethod
    def _perspective_says(score):
        names = ("TOXICITY", "SEVERE_TOXICITY", "IDENTITY_ATTACK", "INSULT", "PROFANITY", "THREAT")
        return {"attributeScores": {name: {"summaryScore": {"value": score}} for name in names}}

    def test_perspective_is_asked_again_once_it_is_back(self):
        detector, execute = self._both_backends(
            [RuntimeError("service unavailable"), self._perspective_says(0.01)]
        )
        text = "You are stupid and worthless"

        while_down = detector.detect(text)
        once_back = detector.detect(text)

        assert while_down.is_toxic is True  # the local model's verdict
        assert execute.call_count == 2
        assert once_back.is_toxic is False  # Perspective's
        assert once_back.scores["toxicity"] == 0.01

    def test_the_local_answer_is_still_remembered_as_the_local_model_s(self):
        detector, execute = self._both_backends([RuntimeError("down"), RuntimeError("down")])
        text = "You are stupid and worthless"

        first = detector.detect(text)
        second = detector.detect(text)

        assert execute.call_count == 2
        assert detector._detoxify_model.predict.call_count == 1
        assert second.scores == first.scores

    def test_an_answer_from_perspective_is_remembered(self):
        detector, execute = self._both_backends([self._perspective_says(0.9)])
        text = "You are stupid and worthless"

        detector.detect(text)
        again = detector.detect(text)

        assert execute.call_count == 1
        assert again.is_toxic is True
        detector._detoxify_model.predict.assert_not_called()

    def test_what_a_caller_does_to_its_result_does_not_reach_the_next_caller(self):
        detector = _detector()
        text = "You are stupid and worthless"

        first = detector.detect(text)
        first.scores.clear()
        first.toxic_categories.clear()
        second = detector.detect(text)
        second.scores.clear()
        third = detector.detect(text)

        assert detector._detoxify_model.predict.call_count == 1
        assert third.is_toxic is True
        assert third.scores["toxicity"] == 0.9
        assert sorted(third.toxic_categories) == ["insult", "toxicity"]
