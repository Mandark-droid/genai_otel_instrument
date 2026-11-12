"""A2UI parts inside A2A spans, driven through the real ``a2a-sdk`` 1.x over an in-process app.

No network and no sockets: the client's httpx transport hands each request to a Starlette
app that serves the SDK's real request handler, in an empty ``contextvars`` context so the
two sides are as separate as two processes.
"""

import asyncio
import contextvars
import functools
import json
import sys

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

pytest.importorskip("a2a", reason="a2a-sdk is not installed")
httpx = pytest.importorskip("httpx")
pytest.importorskip("starlette", reason="a2a-sdk[http-server] is not installed")

from a2a.client.client import ClientCallContext, ClientConfig  # noqa: E402
from a2a.client.client_factory import ClientFactory  # noqa: E402
from a2a.helpers.proto_helpers import new_task  # noqa: E402
from a2a.server.agent_execution.agent_executor import AgentExecutor  # noqa: E402
from a2a.server.request_handlers.default_request_handler_v2 import (  # noqa: E402
    DefaultRequestHandlerV2,
)
from a2a.server.routes import create_jsonrpc_routes  # noqa: E402
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore  # noqa: E402
from a2a.server.tasks.task_updater import TaskUpdater  # noqa: E402
from a2a.types.a2a_pb2 import (  # noqa: E402
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    Message,
    Part,
    Role,
    SendMessageRequest,
    TaskState,
)
from google.protobuf import json_format  # noqa: E402
from google.protobuf.struct_pb2 import Value  # noqa: E402
from starlette.applications import Starlette  # noqa: E402

from genai_otel.a2a_semconv import A2AInstrumentor  # noqa: E402
from genai_otel.a2ui_semconv.attributes import (  # noqa: E402
    DEPRECATED_MIME_TYPE,
    EXTENSION_URI,
    MIME_TYPE,
)
from genai_otel.config import OTelConfig  # noqa: E402

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="the in-process rig needs asyncio.create_task(context=)"
)

AGENT_URL = "http://agent-b.test:8443/a2a"
CATALOG = "https://example.test/catalog.json"
SECRET_URL = "https://cdn.test/passport-scan.png?token=abc"
SECRET_TEXT = "Jane Doe, card 4111111111111111"


def _async(func):
    @functools.wraps(func)
    def runner(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return runner


def _value(obj):
    return json_format.ParseDict(obj, Value())


def a2ui_part(payload, mime=MIME_TYPE, where="metadata"):
    """An A2A data part carrying A2UI, its media type in metadata or in ``media_type``."""
    if where == "metadata":
        return Part(data=_value(payload), metadata={"mimeType": mime})
    return Part(data=_value(payload), media_type=mime)


def create(surface="s1", version="1.0", catalog=CATALOG):
    return {"version": version, "createSurface": {"surfaceId": surface, "catalogId": catalog}}


def update(surface="s1", version="1.0", **component_extra):
    return {
        "version": version,
        "updateComponents": {
            "surfaceId": surface,
            "components": [
                {"id": "root", "component": "Image", **component_extra},
                {"id": "t", "component": "Text", "text": SECRET_TEXT},
            ],
        },
    }


ACTION = {
    "version": "1.0",
    "action": {"name": "approve", "surfaceId": "s1", "sourceComponentId": "b", "context": {"n": 1}},
}


class _Executor(AgentExecutor):
    """Replies with A2UI parts, shaped by the first word of the user's text."""

    SCENARIOS = {
        # one message per part (the Python agent SDK's shape), current media type
        "one": lambda: [a2ui_part(create()), a2ui_part(update(url=SECRET_URL))],
        # a list in one part (the specification's shape), deprecated media type, v0.9,
        # the media type on the part itself
        "list": lambda: [
            a2ui_part(
                [create(version="v0.9"), update(version="v0.9")],
                mime=DEPRECATED_MIME_TYPE,
                where="media_type",
            )
        ],
        "two": lambda: [a2ui_part(create("s1")), a2ui_part(create("s2", catalog="https://other"))],
        "v08": lambda: [
            a2ui_part({"beginRendering": {"surfaceId": "s1", "root": "r"}}, DEPRECATED_MIME_TYPE)
        ],
        "mixed": lambda: [a2ui_part(create(version="v0.9"), DEPRECATED_MIME_TYPE)],
        "plain": lambda: [Part(data=_value({"order": 1}))],
    }

    async def execute(self, context, event_queue):
        text = context.message.parts[0].text if context.message.parts else ""
        scenario = text.split()[0] if text else "plain"
        task = new_task(
            context.task_id,
            context.context_id,
            TaskState.TASK_STATE_SUBMITTED,
            history=[context.message],
        )
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.start_work()
        await updater.add_artifact(self.SCENARIOS[scenario](), artifact_id="ui-1")
        await updater.complete()

    async def cancel(self, context, event_queue):
        await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()


def _card():
    return AgentCard(
        name="ui-agent",
        description="Sends UI",
        version="1.0.0",
        supported_interfaces=[
            AgentInterface(url=AGENT_URL, protocol_binding="JSONRPC", protocol_version="1.0")
        ],
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
    )


class _IsolatedASGITransport(httpx.AsyncBaseTransport):
    def __init__(self, app):
        self._inner = httpx.ASGITransport(app=app)

    async def handle_async_request(self, request):
        task = asyncio.get_running_loop().create_task(
            self._inner.handle_async_request(request), context=contextvars.Context()
        )
        return await task

    async def aclose(self):
        await self._inner.aclose()


class Rig:
    def __init__(self, streaming=True):
        self.card = _card()
        handler = DefaultRequestHandlerV2(_Executor(), InMemoryTaskStore(), self.card)
        transport = _IsolatedASGITransport(
            Starlette(routes=create_jsonrpc_routes(handler, rpc_url="/a2a"))
        )
        self.http = httpx.AsyncClient(transport=transport, base_url="http://agent-b.test:8443")
        config = ClientConfig(
            streaming=streaming, httpx_client=self.http, supported_protocol_bindings=["JSONRPC"]
        )
        self.client = ClientFactory(config).create(self.card)

    async def close(self):
        await self.client.close()
        await self.http.aclose()


def _request(scenario, extra_parts=(), metadata=None):
    message = Message(
        message_id="m-1",
        role=Role.ROLE_USER,
        parts=[Part(text=scenario), *extra_parts],
    )
    if metadata:
        message.metadata.update(metadata)
    return SendMessageRequest(message=message)


async def _drain(stream):
    return [event async for event in stream]


class _NoTracing:
    def get_tracer(self, *args, **kwargs):
        return self

    def start_as_current_span(self, *args, **kwargs):
        import contextlib

        @contextlib.contextmanager
        def noop():
            class _Span:
                def __getattr__(self, name):
                    return lambda *a, **k: None

            yield _Span()

        return noop()


@pytest.fixture(autouse=True)
def sdk_spans_off(monkeypatch):
    import a2a.utils.telemetry as sdk_telemetry

    monkeypatch.setattr(sdk_telemetry, "trace", _NoTracing())


@pytest.fixture
def exporter():
    return InMemorySpanExporter()


@pytest.fixture
def provider(exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


@pytest.fixture
def make_instrumentor(provider, monkeypatch):
    made = []

    def build(content=False, a2ui_enabled=True):
        monkeypatch.setenv("OTEL_INSTRUMENTATION_A2A_SDK_ENABLED", "false")
        config = OTelConfig(
            service_name="a2ui-a2a-test",
            enable_content_capture=content,
            a2ui_enabled=a2ui_enabled,
            content_max_length=100_000,
        )
        inst = A2AInstrumentor()
        inst.tracer = provider.get_tracer("test-a2a")
        inst.instrument(config)
        made.append(inst)
        return inst

    yield build
    for inst in made:
        inst.uninstrument()


def _kind(exporter, kind):
    found = [s for s in exporter.get_finished_spans() if s.kind == kind]
    assert len(found) == 1, [s.name for s in exporter.get_finished_spans()]
    return found[0]


def _a2ui(span):
    return {k: v for k, v in span.attributes.items() if k.startswith("a2ui.")}


async def _send(request, streaming=True, context=None):
    rig = Rig(streaming=streaming)
    try:
        await _drain(rig.client.send_message(request, context=context))
    finally:
        await rig.close()


# ---- both shapes, both media types, both sides --------------------------------------


@_async
async def test_one_message_per_part_current_media_type(make_instrumentor, exporter):
    make_instrumentor()
    await _send(_request("one"))
    client = _kind(exporter, SpanKind.CLIENT)
    a = _a2ui(client)
    assert a["a2ui.protocol.version"] == "1.0"
    assert a["a2ui.agent_message.count"] == 2  # response side
    assert "a2ui.renderer_message.count" not in a
    assert a["a2ui.message.types"] == ("create_surface", "update_components")
    assert a["a2ui.surface.id"] == "s1"
    assert a["a2ui.catalog.id"] == CATALOG
    assert a["a2ui.component.count"] == 2
    # The existing A2A counts are unchanged: A2UI parts are still data parts.
    assert client.attributes["a2a.response.data_part_count"] == 2
    assert client.attributes["a2a.artifact.count"] == 1


@_async
async def test_a_list_in_one_part_deprecated_type_on_the_part_itself(make_instrumentor, exporter):
    make_instrumentor()
    await _send(_request("list"))
    client = _kind(exporter, SpanKind.CLIENT)
    a = _a2ui(client)
    assert a["a2ui.protocol.version"] == "0.9"
    assert a["a2ui.agent_message.count"] == 2
    assert client.attributes["a2a.response.data_part_count"] == 1  # one part, two messages
    assert a["a2ui.component.count"] == 2


@_async
async def test_server_span_reads_the_same_response(make_instrumentor, exporter):
    make_instrumentor()
    await _send(_request("one"))
    server = _kind(exporter, SpanKind.SERVER)
    assert _a2ui(server)["a2ui.agent_message.count"] == 2


@_async
async def test_request_side_is_the_renderer_and_streaming_events_accumulate(
    make_instrumentor, exporter
):
    make_instrumentor()
    request = _request(
        "one",
        extra_parts=[a2ui_part(ACTION)],
        metadata={
            "a2uiRendererCapabilities": {"1.0": {"supportedCatalogIds": [CATALOG, "https://b"]}}
        },
    )
    await _send(request, streaming=True)
    a = _a2ui(_kind(exporter, SpanKind.CLIENT))
    assert a["a2ui.renderer_message.count"] == 1
    assert a["a2ui.agent_message.count"] == 2
    assert a["a2ui.action.name"] == "approve"
    assert a["a2ui.message.types"] == ("action", "create_surface", "update_components")
    assert a["a2ui.renderer.supported_catalog_ids"] == ("https://b", CATALOG)


@_async
async def test_non_streaming_call_reads_the_final_task(make_instrumentor, exporter):
    make_instrumentor()
    await _send(_request("one"), streaming=False)
    assert _a2ui(_kind(exporter, SpanKind.CLIENT))["a2ui.agent_message.count"] == 2


@_async
async def test_the_older_capabilities_key_is_read_too(make_instrumentor, exporter):
    make_instrumentor()
    await _send(
        _request(
            "plain", metadata={"a2uiClientCapabilities": {"v0.9": {"supportedCatalogIds": ["c"]}}}
        )
    )
    a = _a2ui(_kind(exporter, SpanKind.CLIENT))
    assert a == {"a2ui.renderer.supported_catalog_ids": ("c",)}


@_async
async def test_extension_header_marks_both_spans(make_instrumentor, exporter):
    make_instrumentor()
    context = ClientCallContext(service_parameters={"A2A-Extensions": EXTENSION_URI})
    await _send(_request("one"), context=context)
    assert _a2ui(_kind(exporter, SpanKind.CLIENT))["a2ui.extension.active"] is True
    assert _a2ui(_kind(exporter, SpanKind.SERVER))["a2ui.extension.active"] is True


@_async
async def test_the_older_extension_header_name_is_read(make_instrumentor, exporter):
    make_instrumentor()
    context = ClientCallContext(
        service_parameters={"X-A2A-Extensions": "https://other.test/x," + EXTENSION_URI}
    )
    await _send(_request("plain"), context=context)
    assert _a2ui(_kind(exporter, SpanKind.CLIENT)) == {"a2ui.extension.active": True}


@_async
async def test_a_call_without_a2ui_gets_no_a2ui_attributes(make_instrumentor, exporter):
    make_instrumentor()
    context = ClientCallContext(service_parameters={"A2A-Extensions": "https://other.test/x"})
    await _send(_request("plain"), context=context)
    for span in exporter.get_finished_spans():
        assert _a2ui(span) == {}
    client = _kind(exporter, SpanKind.CLIENT)
    assert client.attributes["a2a.response.data_part_count"] == 1


# ---- multi-value omission and v0.8 --------------------------------------------------


@_async
async def test_several_surfaces_and_catalogs_omit_the_single_valued_keys(
    make_instrumentor, exporter
):
    make_instrumentor()
    await _send(_request("two"))
    a = _a2ui(_kind(exporter, SpanKind.CLIENT))
    assert "a2ui.surface.id" not in a and a["a2ui.surface.count"] == 2
    assert "a2ui.catalog.id" not in a and a["a2ui.catalog.ids"] == (CATALOG, "https://other")


@_async
async def test_mixed_protocol_versions_across_request_and_response_omit_the_version(
    make_instrumentor, exporter
):
    make_instrumentor()
    await _send(_request("mixed", extra_parts=[a2ui_part(ACTION)]))  # v1.0 action, v0.9 reply
    a = _a2ui(_kind(exporter, SpanKind.CLIENT))
    assert "a2ui.protocol.version" not in a
    assert a["a2ui.renderer_message.count"] == 1 and a["a2ui.agent_message.count"] == 1


@_async
async def test_v08_is_counted_but_not_parsed(make_instrumentor, exporter):
    make_instrumentor(content=True)
    await _send(_request("v08"))
    a = _a2ui(_kind(exporter, SpanKind.CLIENT))
    assert a == {"a2ui.protocol.version": "0.8", "a2ui.agent_message.count": 1}


# ---- content ------------------------------------------------------------------------


def _all_text(exporter):
    return json.dumps([dict(s.attributes) for s in exporter.get_finished_spans()], default=str)


@_async
async def test_content_off_records_no_a2ui_payload(make_instrumentor, exporter):
    make_instrumentor(content=False)
    await _send(_request("one", extra_parts=[a2ui_part(ACTION)]))
    client = _kind(exporter, SpanKind.CLIENT)
    for key in ("a2ui.messages", "a2ui.action.context", "a2ui.component.types", "a2ui.data_model"):
        assert key not in client.attributes
    text = _all_text(exporter)
    assert SECRET_TEXT not in text and SECRET_URL not in text


@_async
async def test_content_on_records_messages_but_never_media_urls(make_instrumentor, exporter):
    make_instrumentor(content=True)
    await _send(_request("one", extra_parts=[a2ui_part(ACTION)]))
    client = _kind(exporter, SpanKind.CLIENT)
    assert SECRET_TEXT in client.attributes["a2ui.messages"]
    assert client.attributes["a2ui.component.types"] == ("Image", "Text")
    assert json.loads(client.attributes["a2ui.action.context"]) == [{"n": 1}]
    assert SECRET_URL not in _all_text(exporter)  # not in a2ui.* and not in a2a.*.data either


# ---- errors and switches ------------------------------------------------------------


@_async
async def test_a_renderer_error_does_not_make_the_a2a_span_error(make_instrumentor, exporter):
    make_instrumentor()
    error = {
        "version": "1.0",
        "error": {"code": "VALIDATION_FAILED", "surfaceId": "s1", "path": "/c/0", "message": "no"},
    }
    await _send(_request("one", extra_parts=[a2ui_part(error)]))
    client = _kind(exporter, SpanKind.CLIENT)
    assert client.status.status_code != StatusCode.ERROR
    assert "error.type" not in client.attributes
    a = _a2ui(client)
    assert a["a2ui.error.codes"] == ("VALIDATION_FAILED",) and a["a2ui.error.count"] == 1
    events = [e for e in client.events if e.name == "a2ui.error"]
    assert len(events) == 1
    assert dict(events[0].attributes) == {
        "a2ui.error.code": "VALIDATION_FAILED",
        "a2ui.surface.id": "s1",
    }


@_async
async def test_disabled_by_config_leaves_a2a_untouched(make_instrumentor, exporter):
    make_instrumentor(a2ui_enabled=False)
    context = ClientCallContext(service_parameters={"A2A-Extensions": EXTENSION_URI})
    await _send(_request("one", extra_parts=[a2ui_part(ACTION)]), context=context)
    for span in exporter.get_finished_spans():
        assert _a2ui(span) == {}
    client = _kind(exporter, SpanKind.CLIENT)
    assert client.attributes["a2a.method.name"]
    assert client.attributes["a2a.response.data_part_count"] == 2
    assert client.attributes["a2a.message.data_part_count"] == 1


@_async
async def test_disabled_keeps_the_a2a_data_attribute_as_it_was(make_instrumentor, exporter):
    make_instrumentor(content=True, a2ui_enabled=False)
    await _send(_request("one"))
    client = _kind(exporter, SpanKind.CLIENT)
    # With A2UI off the A2A data attribute is the SDK data as before, URL included: it is
    # the A2UI reader that redacts, and it is off.
    assert SECRET_URL in client.attributes["a2a.response.data"]
