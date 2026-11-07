"""With content capture off, no instrumentor puts prompt, input or output text on a span.

Framework instrumentors write text under their own keys (``google_adk.input_message``,
``haystack.generator.prompt``, ``crewai.inputs.<name>``...). Several never consulted the
content setting, so with capture off -- the default -- those keys reached the backend
anyway. Each instrumentor now declares its content keys and the base wrapper strips
them; this file holds the rule in place.

The static check: any attribute whose value is produced by ``_cap_content(...)`` is
text, so its key must be declared content by the instrumentor that writes it. A new
instrumentor (or a new key) that forgets the declaration fails here.
"""

import ast
import importlib
import inspect
import os
import pathlib
import re
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import genai_otel.instrumentors as instrumentors_pkg
from genai_otel.config import OTelConfig
from genai_otel.instrumentors.base import (
    BaseInstrumentor,
    is_content_attribute,
    strip_content_attributes,
)

_PKG_DIR = pathlib.Path(instrumentors_pkg.__file__).parent

# Text attributes written by str(...)[:n] rather than _cap_content, found by review.
_KNOWN_TEXT_KEYS = {
    "bedrock_agents_instrumentor.py": [
        "bedrock.agent.input_text",
        "bedrock.agent.prompt_attributes",
        "bedrock.retrieval.query",
        "bedrock.rag.input_text",
        "bedrock.rag.output_text",
    ],
    "crewai_instrumentor.py": [
        "crewai.task.description",
        "crewai.task.expected_output",
        "crewai.agent.goal",
        "crewai.agent.backstory",
        "crewai.output.json",
        "crewai.output.pydantic",
    ],
    "haystack_instrumentor.py": ["haystack.retriever.filters"],
    "sarvam_instrumentor.py": ["sarvam.stt.prompt"],
}


def _is_cap_call(node) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name == "_cap_content":
                return True
    return False


def _sample_key(node) -> str:
    """A concrete key for a literal or an f-string (each placeholder becomes ``x``)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant) else "x" for v in node.values)
    raise AssertionError(f"unrecognised key expression: {ast.unparse(node)}")


def _captured_keys(path: pathlib.Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keys = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and _is_cap_call(n.value):
            for t in n.targets:
                if isinstance(t, ast.Subscript):
                    keys.add(_sample_key(t.slice))
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in ("set_attribute", "_set_content_attribute")
        ):
            args = n.args[1:] if n.func.attr == "_set_content_attribute" else n.args
            if len(args) >= 2 and _is_cap_call(args[1]):
                keys.add(_sample_key(args[0]))
    return keys


def _instrumentor_classes(module):
    return [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, BaseInstrumentor)
        and obj is not BaseInstrumentor
        and obj.__module__ == module.__name__
    ]


def _cases():
    for path in sorted(_PKG_DIR.glob("*.py")):
        keys = _captured_keys(path) | set(_KNOWN_TEXT_KEYS.get(path.name, []))
        if keys:
            yield pytest.param(path, sorted(keys), id=path.stem)


@pytest.mark.parametrize("path,keys", list(_cases()))
def test_every_captured_text_key_is_declared_content(path, keys):
    module = importlib.import_module(f"genai_otel.instrumentors.{path.stem}")
    classes = _instrumentor_classes(module)
    assert classes, f"{path.name} writes text but defines no instrumentor"
    undeclared = [k for k in keys if not any(is_content_attribute(k, c) for c in classes)]
    assert not undeclared, (
        f"{path.name} writes text under {undeclared} without declaring them in "
        "CONTENT_ATTRIBUTES / CONTENT_ATTRIBUTE_PREFIXES / CONTENT_ATTRIBUTE_PATTERNS, "
        "so they would reach spans with content capture off"
    )


def test_structural_keys_under_content_prefixes_are_kept():
    from genai_otel.instrumentors.crewai_instrumentor import CrewAIInstrumentor
    from genai_otel.instrumentors.haystack_instrumentor import HaystackInstrumentor
    from genai_otel.instrumentors.instructor_instrumentor import InstructorInstrumentor
    from genai_otel.instrumentors.langgraph_instrumentor import LangGraphInstrumentor

    assert is_content_attribute("crewai.inputs.topic", CrewAIInstrumentor)
    assert not is_content_attribute("crewai.inputs.keys", CrewAIInstrumentor)
    assert is_content_attribute("langgraph.output.messages", LangGraphInstrumentor)
    assert not is_content_attribute("langgraph.output.keys", LangGraphInstrumentor)
    assert is_content_attribute("haystack.output.llm.first_reply", HaystackInstrumentor)
    assert not is_content_attribute("haystack.output.llm.replies.count", HaystackInstrumentor)
    assert is_content_attribute("instructor.response.name", InstructorInstrumentor)
    assert not is_content_attribute("instructor.response.fields_count", InstructorInstrumentor)


class _Declaring(BaseInstrumentor):
    CONTENT_ATTRIBUTES = frozenset({"fw.prompt"})
    CONTENT_ATTRIBUTE_PREFIXES = ("fw.input.",)
    NON_CONTENT_ATTRIBUTES = frozenset({"fw.input.keys"})

    def instrument(self, config):
        self.config = config

    def _extract_usage(self, result):
        return None

    def _extract_response_attributes(self, result):
        return {"fw.output_len": 3, "output.value": "secret answer", "fw.prompt": "again"}


def test_strip_drops_declared_keys_only_with_capture_off():
    attrs = {
        "fw.prompt": "secret",
        "fw.input.city": "secret",
        "fw.input.keys": ["city"],
        "fw.count": 2,
        "gen_ai.prompt.0.content": "secret",
        "input.value": "secret",
    }
    off = strip_content_attributes(attrs, OTelConfig(enable_content_capture=False), _Declaring)
    assert off == {"fw.input.keys": ["city"], "fw.count": 2}
    on = strip_content_attributes(attrs, OTelConfig(enable_content_capture=True), _Declaring)
    assert on == attrs
    # Two-argument callers are unchanged: only the global keys go.
    legacy = strip_content_attributes(attrs, OTelConfig(enable_content_capture=False))
    assert "fw.prompt" in legacy and "input.value" not in legacy


@pytest.mark.parametrize("capture", [False, True])
def test_base_wrapper_strips_declared_keys_at_start_and_end(capture):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    inst = _Declaring()
    inst.config = OTelConfig(enable_content_capture=capture)
    inst._instrumented = True
    inst.tracer = provider.get_tracer("test")

    def extract(_instance, _args, _kwargs):
        return {"fw.prompt": "secret question", "fw.input.city": "Pune", "fw.input.keys": "city"}

    token = otel_context.attach(otel_context.Context())
    try:
        inst.create_span_wrapper("fw.run", extract_attributes=extract)(lambda: "ok")()
    finally:
        otel_context.detach(token)

    (span,) = exporter.get_finished_spans()
    attrs = dict(span.attributes)
    assert attrs["fw.input.keys"] == "city"
    assert attrs["fw.output_len"] == 3
    for key in ("fw.prompt", "fw.input.city", "output.value"):
        assert (key in attrs) is capture, key


def test_set_content_attribute_respects_capture_and_limit():
    span = MagicMock()
    inst = _Declaring()
    inst.config = OTelConfig(enable_content_capture=False)
    inst._set_content_attribute(span, "fw.prompt", "secret")
    span.set_attribute.assert_not_called()

    inst.config = OTelConfig(enable_content_capture=True, content_max_length=4)
    inst._set_content_attribute(span, "fw.prompt", "secret")
    span.set_attribute.assert_called_once_with("fw.prompt", "secr")


# --------------------------------------------------------------------------- ADK G-4


def _adk_instrumentor():
    from genai_otel.instrumentors.google_adk_instrumentor import GoogleADKInstrumentor

    class Runner:
        def run_async(self, **kwargs):
            return None

    class InMemoryRunner(Runner):
        def run_debug(self, message):
            return None

    runners = MagicMock(Runner=Runner, InMemoryRunner=InMemoryRunner)
    modules = {"google": MagicMock(), "google.adk": MagicMock(), "google.adk.runners": runners}
    return GoogleADKInstrumentor, modules


@pytest.mark.parametrize(
    "capture,preset,expected",
    [
        (False, None, "false"),
        (False, "true", "true"),  # the application's value wins
        (True, None, None),
    ],
)
def test_adk_content_switch_follows_ours_unless_set(capture, preset, expected):
    cls, modules = _adk_instrumentor()
    env = {} if preset is None else {"ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS": preset}
    with patch.dict("sys.modules", modules), patch.dict(os.environ, env, clear=False):
        os.environ.pop("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", None)
        os.environ.update(env)
        inst = cls()
        inst.instrument(OTelConfig(enable_content_capture=capture))
        assert os.environ.get("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS") == expected


def test_pattern_declarations_are_compiled():
    for path in sorted(_PKG_DIR.glob("*_instrumentor.py")):
        module = importlib.import_module(f"genai_otel.instrumentors.{path.stem}")
        for cls in _instrumentor_classes(module):
            for p in cls.CONTENT_ATTRIBUTE_PATTERNS:
                assert isinstance(p, re.Pattern), f"{cls.__name__}: {p!r} is not compiled"
