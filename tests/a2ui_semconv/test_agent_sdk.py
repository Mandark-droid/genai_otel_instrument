"""A2UI agent-SDK instrumentation, driven through the real ``a2ui-agent-sdk`` processors.

No network. ``a2ui.a2a.parts`` belongs to ``a2ui-a2a``, which pins ``a2a-sdk`` 0.3 and so
cannot sit beside the 1.x the A2A tests need; its ``parse_content_to_parts`` is covered with
a stand-in module that has the same name, logger and log line.
"""

import asyncio
import functools
import json
import logging
import subprocess
import sys
import textwrap
import types

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

pytest.importorskip("a2ui", reason="a2ui-agent-sdk is not installed")
pytest.importorskip("a2ui.rpc.processor", reason="a2ui-agent-sdk is older than 0.11")

from a2ui.core.basic_catalog import BasicCatalog  # noqa: E402
from a2ui.inference_formats.direct_json import DirectJsonFormatFactory  # noqa: E402
from a2ui.processor import A2uiRequestProcessor  # noqa: E402
from a2ui.rpc.processor import RendererToAgentMessageProcessor  # noqa: E402

from genai_otel.a2ui_semconv import A2UIInstrumentor  # noqa: E402
from genai_otel.a2ui_semconv import instrumentor as a2ui_module  # noqa: E402
from genai_otel.config import OTelConfig  # noqa: E402

CATALOG = BasicCatalog("1.0")
CATALOG_ID = CATALOG.catalog_id
SECRET = "jane.doe@example.test"


def _async(func):
    @functools.wraps(func)
    def runner(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return runner


def good_messages(surface="s1", text="hello"):
    return [
        {"version": "1.0", "createSurface": {"surfaceId": surface, "catalogId": CATALOG_ID}},
        {
            "version": "1.0",
            "updateComponents": {
                "surfaceId": surface,
                "components": [{"id": "root", "component": "Text", "text": text}],
            },
        },
    ]


def llm_response(messages):
    return "Here you go.\n<a2ui-json>\n" + json.dumps(messages) + "\n</a2ui-json>"


BAD_RESPONSE = '<a2ui-json>\n[{"version": "1.0", "nonsense": {"x": "%s"}}]\n</a2ui-json>' % SECRET


def processor():
    return A2uiRequestProcessor([CATALOG], format_factory=DirectJsonFormatFactory())


@pytest.fixture
def exporter():
    return InMemorySpanExporter()


@pytest.fixture
def provider(exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


@pytest.fixture
def make_instrumentor(provider):
    made = []

    def build(content=False, enabled=True):
        config = OTelConfig(
            service_name="a2ui-test", enable_content_capture=content, a2ui_enabled=enabled
        )
        inst = A2UIInstrumentor()
        inst.tracer = provider.get_tracer("test-a2ui")
        inst.instrument(config)
        made.append(inst)
        return inst

    yield build
    for inst in made:
        inst.uninstrument()


def spans(exporter, name=None):
    found = exporter.get_finished_spans()
    return [s for s in found if name is None or s.name == name]


def one(exporter, name):
    found = spans(exporter, name)
    assert len(found) == 1, [s.name for s in exporter.get_finished_spans()]
    return found[0]


# ---- parse_response ----------------------------------------------------------------


def test_parse_response_valid(make_instrumentor, exporter):
    make_instrumentor()
    parts = processor().parse_response(llm_response(good_messages()))
    assert len(parts) == 2  # the wrapper returns what the SDK returned
    span = one(exporter, "a2ui.parse_response")
    a = dict(span.attributes)
    assert span.kind == SpanKind.INTERNAL
    assert span.status.status_code == StatusCode.UNSET
    assert a["a2ui.validation.outcome"] == "valid"
    assert a["a2ui.protocol.version"] == "1.0"
    assert a["a2ui.agent_message.count"] == 2
    assert a["a2ui.message.types"] == ("create_surface", "update_components")
    assert a["a2ui.surface.id"] == "s1"
    assert a["a2ui.catalog.id"] == CATALOG_ID
    assert a["a2ui.component.count"] == 1
    assert "a2ui.renderer_message.count" not in a


def test_parse_response_invalid_is_an_error_span_and_still_raises(make_instrumentor, exporter):
    make_instrumentor()
    with pytest.raises(Exception) as caught:
        processor().parse_response(BAD_RESPONSE)
    span = one(exporter, "a2ui.parse_response")
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == type(caught.value).__name__
    assert span.attributes["a2ui.validation.outcome"] == "invalid"


def test_parse_response_error_text_is_kept_only_with_content_capture(make_instrumentor, exporter):
    inst = make_instrumentor(content=False)
    with pytest.raises(Exception):
        processor().parse_response(BAD_RESPONSE)
    off = one(exporter, "a2ui.parse_response")
    assert SECRET not in json.dumps(
        [dict(off.attributes), off.status.description, [dict(e.attributes) for e in off.events]],
        default=str,
    )
    inst.uninstrument()
    exporter.clear()

    make_instrumentor(content=True)
    with pytest.raises(Exception) as caught:
        processor().parse_response(BAD_RESPONSE)
    on = one(exporter, "a2ui.parse_response")
    assert on.status.status_code == StatusCode.ERROR
    assert on.attributes["error.type"] == type(caught.value).__name__
    assert on.status.description  # the text is there, because capture is on


def test_parse_response_is_a_child_of_the_llm_span(make_instrumentor, exporter, provider):
    make_instrumentor()
    tracer = provider.get_tracer("test-app")
    with tracer.start_as_current_span("chat model") as llm:
        processor().parse_response(llm_response(good_messages()))
        with pytest.raises(Exception):
            processor().parse_response(BAD_RESPONSE)
    llm_id = llm.get_span_context().span_id
    children = spans(exporter, "a2ui.parse_response")
    assert len(children) == 2
    assert {c.parent.span_id for c in children} == {llm_id}


def test_content_off_and_on(make_instrumentor, exporter):
    inst = make_instrumentor(content=False)
    processor().parse_response(llm_response(good_messages(text=SECRET)))
    off = dict(one(exporter, "a2ui.parse_response").attributes)
    assert not any(k in off for k in ("a2ui.messages", "a2ui.component.types", "a2ui.data_model"))
    assert SECRET not in json.dumps(off, default=str)
    inst.uninstrument()
    exporter.clear()

    make_instrumentor(content=True)
    processor().parse_response(llm_response(good_messages(text=SECRET)))
    on = dict(one(exporter, "a2ui.parse_response").attributes)
    assert on["a2ui.component.types"] == ("Text",)
    assert on["a2ui.messages"]


# ---- open_stream -------------------------------------------------------------------


def stream_chunks(text, n=7):
    return [text[i : i + n] for i in range(0, len(text), n)]


def test_open_stream_span_ends_when_the_stream_does_and_carries_totals(make_instrumentor, exporter):
    make_instrumentor()
    stream = processor().open_stream()
    assert spans(exporter, "a2ui.parse_stream") == []  # open: not finished yet
    for chunk in stream_chunks(llm_response(good_messages())):
        stream.feed(chunk)
    assert spans(exporter, "a2ui.parse_stream") == []
    stream.close()
    span = one(exporter, "a2ui.parse_stream")
    a = dict(span.attributes)
    assert span.status.status_code == StatusCode.UNSET
    assert a["a2ui.agent_message.count"] == 2
    assert a["a2ui.message.types"] == ("create_surface", "update_components")
    assert a["a2ui.surface.count"] == 1
    assert a["a2ui.validation.outcome"] == "valid"
    # Closing again changes nothing.
    stream.close()
    assert len(spans(exporter, "a2ui.parse_stream")) == 1


def test_open_stream_is_a_child_of_the_llm_span_but_not_current(
    make_instrumentor, exporter, provider
):
    from opentelemetry import trace

    make_instrumentor()
    tracer = provider.get_tracer("test-app")
    with tracer.start_as_current_span("chat model") as llm:
        stream = processor().open_stream()
        # The caller's code between chunks must not run under our span.
        assert trace.get_current_span() is llm
        for chunk in stream_chunks(llm_response(good_messages())):
            stream.feed(chunk)
        stream.close()
    assert one(exporter, "a2ui.parse_stream").parent.span_id == llm.get_span_context().span_id


def test_open_stream_error_when_a_block_is_rejected(make_instrumentor, exporter):
    make_instrumentor()
    stream = processor().open_stream()
    with pytest.raises(Exception) as caught:
        for chunk in stream_chunks(BAD_RESPONSE):
            stream.feed(chunk)
        stream.close()
    span = one(exporter, "a2ui.parse_stream")
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == type(caught.value).__name__
    assert span.attributes["a2ui.validation.outcome"] == "invalid"


def test_open_stream_reading_past_a_rejected_block_is_still_an_error_span(
    make_instrumentor, exporter
):
    make_instrumentor()
    stream = processor().open_stream(after_rejected_block="continue")
    for chunk in stream_chunks(BAD_RESPONSE):
        stream.feed(chunk)
    stream.close()
    span = one(exporter, "a2ui.parse_stream")
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["a2ui.validation.outcome"] == "invalid"
    assert "error.type" in span.attributes


def test_a_stream_never_closed_still_ends_its_span_when_collected(make_instrumentor, exporter):
    import gc

    make_instrumentor()
    stream = processor().open_stream()
    stream.feed(llm_response(good_messages())[:20])
    del stream
    gc.collect()
    assert len(spans(exporter, "a2ui.parse_stream")) == 1


# ---- process_message ---------------------------------------------------------------


def action(name="buy", **extra):
    return {
        "version": "1.0",
        "action": {
            "name": name,
            "surfaceId": "s1",
            "sourceComponentId": "btn",
            "timestamp": "2026-01-01T00:00:00Z",
            "context": {"email": SECRET},
            **extra,
        },
    }


@_async
async def test_process_action(make_instrumentor, exporter):
    make_instrumentor()
    seen = []
    rp = RendererToAgentMessageProcessor([CATALOG], on_action=seen.append)
    result = await rp.process_message(action())
    assert result.handled and seen
    span = one(exporter, "a2ui.process action")
    a = dict(span.attributes)
    assert span.kind == SpanKind.INTERNAL
    assert a["a2ui.renderer_message.count"] == 1
    assert a["a2ui.agent_message.count"] == 0  # the reply count
    assert a["a2ui.message.types"] == ("action",)
    assert a["a2ui.action.name"] == "buy"
    assert a["a2ui.surface.id"] == "s1"
    # The action's name is an identifier; its context is content.
    assert "a2ui.action.context" not in a and SECRET not in json.dumps(a, default=str)


@_async
async def test_process_action_context_only_with_content_capture(make_instrumentor, exporter):
    make_instrumentor(content=True)
    rp = RendererToAgentMessageProcessor([CATALOG], on_action=lambda m: True)
    await rp.process_message(action())
    a = dict(one(exporter, "a2ui.process action").attributes)
    assert json.loads(a["a2ui.action.context"]) == [{"email": SECRET}]


@_async
async def test_process_call_agent_function_records_function_and_reply(make_instrumentor, exporter):
    make_instrumentor()
    rp = RendererToAgentMessageProcessor([CATALOG])
    result = await rp.process_message(
        {
            "version": "1.0",
            "callAgentFunction": {
                "surfaceId": "s1",
                "functionCallId": "c1",
                "callFunction": {"@call": "missing_function", "catalogId": CATALOG_ID},
            },
        }
    )
    assert len(result.replies) == 1
    span = one(exporter, "a2ui.process call_agent_function")
    a = dict(span.attributes)
    assert a["a2ui.function.name"] == "missing_function"
    assert a["a2ui.agent_message.count"] == 1  # one agentFunctionResponse
    assert a["a2ui.error.codes"] == ("UNKNOWN_FUNCTION",)
    assert a["a2ui.error.count"] == 1
    # A function error does not make the processing span ERROR: the call succeeded.
    assert span.status.status_code == StatusCode.UNSET
    assert [e.name for e in span.events] == ["a2ui.error"]
    assert span.events[0].attributes["a2ui.error.code"] == "UNKNOWN_FUNCTION"


@_async
async def test_process_renderer_error_message_by_code(make_instrumentor, exporter):
    make_instrumentor(content=False)
    rp = RendererToAgentMessageProcessor([CATALOG], on_error=lambda m: True)
    await rp.process_message(
        {
            "version": "1.0",
            "error": {
                "code": "VALIDATION_FAILED",
                "surfaceId": "s1",
                "path": "/components/0/text",
                "message": f"value {SECRET} is not allowed",
            },
        }
    )
    span = one(exporter, "a2ui.process error")
    a = dict(span.attributes)
    assert a["a2ui.error.codes"] == ("VALIDATION_FAILED",)
    assert a["a2ui.error.count"] == 1
    assert span.status.status_code == StatusCode.UNSET  # the renderer's error is data here
    assert span.events[0].attributes == {
        "a2ui.error.code": "VALIDATION_FAILED",
        "a2ui.surface.id": "s1",
    }
    # Path and message are content: a validation message can quote user data.
    assert "a2ui.error.paths" not in a and "a2ui.error.messages" not in a
    assert SECRET not in json.dumps([a, [dict(e.attributes) for e in span.events]], default=str)


@_async
async def test_process_error_path_and_message_with_content_capture(make_instrumentor, exporter):
    make_instrumentor(content=True)
    rp = RendererToAgentMessageProcessor([CATALOG], on_error=lambda m: True)
    await rp.process_message(
        {
            "version": "1.0",
            "error": {
                "code": "UNALLOWED_CHILD",
                "surfaceId": "s1",
                "path": "/components/2",
                "message": "child not allowed",
            },
        }
    )
    a = dict(one(exporter, "a2ui.process error").attributes)
    assert a["a2ui.error.paths"] == ("/components/2",)
    assert a["a2ui.error.messages"] == ("child not allowed",)


@_async
async def test_process_an_invalid_message_is_an_error_span_and_still_raises(
    make_instrumentor, exporter
):
    make_instrumentor()
    rp = RendererToAgentMessageProcessor([CATALOG], on_action=lambda m: True)
    with pytest.raises(Exception) as caught:
        await rp.process_message({"version": "1.0", "nope": {}})
    span = one(exporter, "a2ui.process unknown")
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == type(caught.value).__name__
    assert span.attributes["a2ui.validation.outcome"] == "invalid"


@_async
async def test_process_accepts_schema_models(make_instrumentor, exporter):
    from a2ui.core.schema.v1_0 import A2uiRendererActionMessage

    make_instrumentor()
    model = A2uiRendererActionMessage.model_validate(action())
    rp = RendererToAgentMessageProcessor([CATALOG], on_action=lambda m: True)
    await rp.process_message(model)
    assert dict(one(exporter, "a2ui.process action").attributes)["a2ui.action.name"] == "buy"


# ---- parse_content_to_parts (stand-in for a2ui-a2a's module) ------------------------


@pytest.fixture
def fake_parts_module(monkeypatch):
    package = types.ModuleType("a2ui.a2a")
    package.__path__ = []
    module = types.ModuleType("a2ui.a2a.parts")
    log = logging.getLogger("a2ui.a2a.parts")

    def parse_content_to_parts(
        content, parser=None, fallback_text=None, version=None, *, state=None
    ):
        parts = []
        try:
            if "BAD" in content:
                raise ValueError(f"cannot parse {SECRET}")
            parts.append(("text", content))
        except Exception as e:  # noqa: BLE001 - the real one swallows exactly like this
            log.warning("Failed to parse A2UI response: %s", type(e).__name__)
            log.debug("A2UI response parse error: %s", e)
        if not parts and fallback_text:
            parts.append(("text", fallback_text))
        return parts

    module.parse_content_to_parts = parse_content_to_parts
    package.parts = module
    monkeypatch.setitem(sys.modules, "a2ui.a2a", package)
    monkeypatch.setitem(sys.modules, "a2ui.a2a.parts", module)
    return module


def test_swallowed_parse_failure_is_recorded_without_changing_the_result_or_the_log(
    make_instrumentor, exporter, provider, fake_parts_module, caplog
):
    original = fake_parts_module.parse_content_to_parts
    make_instrumentor()
    assert fake_parts_module.parse_content_to_parts is not original
    tracer = provider.get_tracer("test-app")
    with caplog.at_level(logging.WARNING, logger="a2ui.a2a.parts"):
        with tracer.start_as_current_span("a2a server span"):
            result = fake_parts_module.parse_content_to_parts("BAD", None, fallback_text="sorry")
    assert result == [("text", "sorry")]  # the SDK's return value, untouched
    assert [r.getMessage() for r in caplog.records] == ["Failed to parse A2UI response: ValueError"]
    span = one(exporter, "a2a server span")
    assert span.attributes["a2ui.validation.outcome"] == "invalid"
    assert [e.name for e in span.events] == ["a2ui.validation_failed"]
    assert span.events[0].attributes["error.type"] == "ValueError"
    assert SECRET not in json.dumps(
        [dict(span.attributes), [dict(e.attributes) for e in span.events]], default=str
    )


def test_a_good_parse_leaves_the_current_span_alone(
    make_instrumentor, exporter, provider, fake_parts_module
):
    make_instrumentor()
    with provider.get_tracer("t").start_as_current_span("outer"):
        assert fake_parts_module.parse_content_to_parts("fine") == [("text", "fine")]
    assert "a2ui.validation.outcome" not in dict(one(exporter, "outer").attributes)


def test_the_log_filter_is_inert_outside_the_wrapper(make_instrumentor, fake_parts_module, caplog):
    make_instrumentor()
    with caplog.at_level(logging.WARNING, logger="a2ui.a2a.parts"):
        logging.getLogger("a2ui.a2a.parts").warning("Failed to parse A2UI response: %s", "X")
    assert len(caplog.records) == 1  # and nothing raised


def test_uninstrument_restores_the_parts_function_and_the_filter(
    make_instrumentor, fake_parts_module
):
    original = fake_parts_module.parse_content_to_parts
    inst = make_instrumentor()
    assert logging.getLogger("a2ui.a2a.parts").filters
    inst.uninstrument()
    assert fake_parts_module.parse_content_to_parts is original
    assert not logging.getLogger("a2ui.a2a.parts").filters


# ---- switches and lifecycle --------------------------------------------------------


def test_disabled_by_config_does_nothing(make_instrumentor, exporter):
    inst = make_instrumentor(enabled=False)
    assert inst._patched == []
    processor().parse_response(llm_response(good_messages()))
    assert spans(exporter) == []


def test_instrument_is_idempotent_and_uninstrument_restores(make_instrumentor):
    before = dict(vars(A2uiRequestProcessor))
    rpc_before = dict(vars(RendererToAgentMessageProcessor))
    first = make_instrumentor()
    assert A2uiRequestProcessor.parse_response is not before["parse_response"]
    count = len(first._patched)
    second = make_instrumentor()
    assert second._patched == []  # already wrapped; nothing wrapped twice
    first.uninstrument()
    assert dict(vars(A2uiRequestProcessor)) == before
    assert dict(vars(RendererToAgentMessageProcessor)) == rpc_before
    assert count >= 3


def test_wrapped_methods_keep_their_names(make_instrumentor):
    make_instrumentor()
    assert A2uiRequestProcessor.parse_response.__name__ == "parse_response"
    assert A2uiRequestProcessor.open_stream.__name__ == "open_stream"
    assert RendererToAgentMessageProcessor.process_message.__name__ == "process_message"


def test_registered_and_on_by_default():
    from genai_otel.auto_instrument import INSTRUMENTORS
    from genai_otel.config import DEFAULT_INSTRUMENTORS

    assert INSTRUMENTORS["a2ui"] is A2UIInstrumentor
    assert "a2ui" in DEFAULT_INSTRUMENTORS


def test_enabled_setting_reads_its_environment_variable(monkeypatch):
    monkeypatch.setenv("GENAI_A2UI_ENABLED", "false")
    assert OTelConfig(service_name="t").a2ui_enabled is False
    monkeypatch.delenv("GENAI_A2UI_ENABLED")
    assert OTelConfig(service_name="t").a2ui_enabled is True


# ---- no-op without the package -----------------------------------------------------


def test_noop_when_the_sdk_is_older_than_0_11(monkeypatch):
    monkeypatch.setattr(a2ui_module, "_sdk_version", lambda: (0, 10))
    before = dict(vars(A2uiRequestProcessor))
    inst = A2UIInstrumentor()
    inst.instrument(OTelConfig(service_name="t"))
    assert dict(vars(A2uiRequestProcessor)) == before
    assert inst._patched == []
    assert A2UIInstrumentor.available() is False


def test_a_newer_sdk_is_supported(monkeypatch):
    monkeypatch.setattr(a2ui_module, "_sdk_version", lambda: (0, 12))
    assert a2ui_module.sdk_supported() is True
    monkeypatch.setattr(a2ui_module, "_sdk_version", lambda: (1, 0))
    assert a2ui_module.sdk_supported() is True


def test_noop_when_the_sdk_is_missing(monkeypatch):
    import importlib.util as util

    real = util.find_spec
    monkeypatch.setattr(
        a2ui_module.importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "a2ui" else real(name, *a, **k),
    )
    inst = A2UIInstrumentor()
    inst.instrument(OTelConfig(service_name="t"))
    assert inst._patched == []
    assert A2UIInstrumentor.available() is False


def _fresh_interpreter(code):
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip().splitlines()[-1]


def test_import_and_instrument_without_the_sdk_in_a_fresh_interpreter():
    code = """
        import sys
        sys.modules["a2ui"] = None   # makes "import a2ui" raise ImportError, as if not installed
        from genai_otel.a2ui_semconv import A2UIInstrumentor
        from genai_otel.config import OTelConfig
        inst = A2UIInstrumentor()
        inst.instrument(OTelConfig(service_name="t"))
        print("noop", inst._patched == [])
    """
    assert _fresh_interpreter(code) == "noop True"
