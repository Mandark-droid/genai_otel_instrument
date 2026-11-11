"""A2A instrumentation, driven through the real ``a2a-sdk`` 1.x over an in-process app.

No network and no sockets: the client's httpx transport hands each request straight to a
Starlette app that serves the SDK's real request handler. The transport runs the app in a
fresh ``contextvars`` context, because that is what a separate process looks like: the
only thing that can join the two sides into one trace is the propagated header.
"""

import asyncio
import contextvars
import functools
import json
import logging
import subprocess
import sys
import textwrap

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

pytest.importorskip("a2a", reason="a2a-sdk is not installed")
httpx = pytest.importorskip("httpx")
pytest.importorskip("starlette", reason="a2a-sdk[http-server] is not installed")

from a2a.client.client import ClientConfig  # noqa: E402
from a2a.client.client_factory import ClientFactory  # noqa: E402
from a2a.client.errors import A2AClientError  # noqa: E402
from a2a.helpers.proto_helpers import (  # noqa: E402
    new_data_part,
    new_raw_part,
    new_task,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution.agent_executor import AgentExecutor  # noqa: E402
from a2a.server.request_handlers.default_request_handler_v2 import (  # noqa: E402
    DefaultRequestHandlerV2,
)
from a2a.server.routes import create_jsonrpc_routes, create_rest_routes  # noqa: E402
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore  # noqa: E402
from a2a.server.tasks.task_updater import TaskUpdater  # noqa: E402
from a2a.types.a2a_pb2 import (  # noqa: E402
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    CancelTaskRequest,
    GetTaskRequest,
    Message,
    Part,
    Role,
    SendMessageRequest,
    TaskState,
)
from starlette.applications import Starlette  # noqa: E402

from genai_otel.a2a_semconv import A2AInstrumentor  # noqa: E402
from genai_otel.a2a_semconv import instrumentor as a2a_module  # noqa: E402
from genai_otel.config import OTelConfig  # noqa: E402

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="the in-process rig needs asyncio.create_task(context=)"
)


def _async(func):
    """Run an async test with asyncio.run (pytest-asyncio is not a dependency of the suite)."""

    @functools.wraps(func)
    def runner(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return runner


AGENT_URL = "http://agent-b.test:8443/a2a"
SECRET_TEXT = "my card number is 4111111111111111"
REPLY_TEXT = "refund approved for order 8841"


# ---- the rig ------------------------------------------------------------------------


class _Executor(AgentExecutor):
    """Answers by what the user message says: ``fail`` / ``reject`` / ``file`` / default."""

    async def execute(self, context, event_queue):
        text = context.message.parts[0].text if context.message.parts else ""
        task = new_task(
            context.task_id,
            context.context_id,
            TaskState.TASK_STATE_SUBMITTED,
            history=[context.message],
        )
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.start_work()
        if text.startswith("fail"):
            await updater.failed()
        elif text.startswith("reject"):
            await updater.reject()
        else:
            parts = [new_text_part(REPLY_TEXT), new_data_part({"order": 8841})]
            if text.startswith("file"):
                parts.append(new_raw_part(b"%PDF-secret", media_type="application/pdf"))
            await updater.add_artifact(parts, artifact_id="art-1")
            await updater.complete()

    async def cancel(self, context, event_queue):
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()


def _card(url=AGENT_URL, binding="JSONRPC"):
    return AgentCard(
        name="refund-agent",
        description="Handles refund requests",
        version="2.1.0",
        supported_interfaces=[
            AgentInterface(url=url, protocol_binding=binding, protocol_version="1.0")
        ],
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(id="s1", name="refund-policy", description="Policy lookups"),
            AgentSkill(id="s2", name="order-lookup", description="Find an order"),
        ],
    )


class _IsolatedASGITransport(httpx.AsyncBaseTransport):
    """Run the app the way another process would: in an empty context, with no current span."""

    def __init__(self, app):
        self._inner = httpx.ASGITransport(app=app)
        self.seen_headers = []

    async def handle_async_request(self, request):
        self.seen_headers.append(dict(request.headers))
        task = asyncio.get_running_loop().create_task(
            self._inner.handle_async_request(request), context=contextvars.Context()
        )
        return await task

    async def aclose(self):
        await self._inner.aclose()


class Rig:
    def __init__(self, binding="JSONRPC", streaming=True):
        self.card = _card(binding=binding)
        self.handler = DefaultRequestHandlerV2(_Executor(), InMemoryTaskStore(), self.card)
        if binding == "HTTP+JSON":
            routes = create_rest_routes(self.handler)
        else:
            routes = create_jsonrpc_routes(self.handler, rpc_url="/a2a")
        self.transport = _IsolatedASGITransport(Starlette(routes=routes))
        self.http = httpx.AsyncClient(transport=self.transport, base_url="http://agent-b.test:8443")
        config = ClientConfig(
            streaming=streaming, httpx_client=self.http, supported_protocol_bindings=[binding]
        )
        self.client = ClientFactory(config).create(self.card)

    async def close(self):
        await self.client.close()
        await self.http.aclose()


def _send(text, **message_fields):
    return SendMessageRequest(
        message=Message(
            message_id="m-1",
            role=Role.ROLE_USER,
            parts=[Part(text=text)],
            **message_fields,
        )
    )


async def _drain(stream):
    return [event async for event in stream]


# ---- fixtures -----------------------------------------------------------------------


class _NoTracing:
    """What ``a2a.utils.telemetry.trace`` is when the SDK was imported with
    ``OTEL_INSTRUMENTATION_A2A_SDK_ENABLED=false``, which is how we run it by default."""

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
    """The test process imports a2a before any of this runs, so switch its spans off here."""
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

    def build(content=False, propagate=True):
        monkeypatch.setenv("OTEL_INSTRUMENTATION_A2A_SDK_ENABLED", "false")
        config = OTelConfig(
            service_name="a2a-test", enable_content_capture=content, a2a_propagate=propagate
        )
        inst = A2AInstrumentor()
        inst.tracer = provider.get_tracer("test-a2a")
        inst.instrument(config)
        made.append(inst)
        return inst

    yield build
    for inst in made:
        inst.uninstrument()


@pytest.fixture
def tracer(provider):
    return provider.get_tracer("test-app")


def _by_kind(exporter, kind):
    return [s for s in exporter.get_finished_spans() if s.kind == kind]


def _one(spans):
    assert len(spans) == 1, [s.name for s in spans]
    return spans[0]


# ---- client -------------------------------------------------------------------------


@_async
async def test_client_send_makes_a_client_span(make_instrumentor, exporter):
    make_instrumentor()
    rig = Rig(streaming=False)
    try:
        events = await _drain(rig.client.send_message(_send("hello", context_id="ctx-9")))
    finally:
        await rig.close()
    assert events
    span = _one(_by_kind(exporter, SpanKind.CLIENT))
    a = span.attributes
    assert span.name == "SendMessage refund-agent"
    assert a["a2a.method.name"] == "SendMessage"
    assert a["a2a.protocol.binding"] == "jsonrpc"
    assert a["a2a.protocol.version"] == "1.0"
    assert a["a2a.message.id"] == "m-1"
    assert a["gen_ai.operation.name"] == "invoke_agent"
    assert a["gen_ai.agent.name"] == "refund-agent"
    assert a["gen_ai.agent.description"] == "Handles refund requests"
    assert a["gen_ai.agent.version"] == "2.1.0"
    assert a["a2a.agent.skill_count"] == 2
    assert a["server.address"] == "agent-b.test"
    assert a["server.port"] == 8443
    assert a["gen_ai.conversation.id"] == "ctx-9"
    assert a["a2a.task.state"] == "completed"
    assert a["a2a.task.id"]
    assert a["a2a.message.text_part_count"] == 1
    assert a["a2a.message.file_part_count"] == 0
    assert a["a2a.artifact.count"] == 1
    assert a["a2a.response.text_part_count"] == 1
    assert a["a2a.response.data_part_count"] == 1
    assert span.status.status_code != StatusCode.ERROR


@_async
async def test_client_streaming_span_ends_when_the_stream_ends(make_instrumentor, exporter):
    make_instrumentor()
    rig = Rig(streaming=True)
    try:
        stream = rig.client.send_message(_send("hello"))
        first = await stream.__anext__()
        assert first is not None
        # Mid-stream: nothing has been exported yet.
        assert not _by_kind(exporter, SpanKind.CLIENT)
        async for _ in stream:
            pass
    finally:
        await rig.close()
    span = _one(_by_kind(exporter, SpanKind.CLIENT))
    assert span.name == "SendStreamingMessage refund-agent"
    assert span.attributes["a2a.method.name"] == "SendStreamingMessage"
    assert span.attributes["a2a.artifact.count"] == 1
    # The task went submitted -> working -> completed: the FINAL state is recorded.
    assert span.attributes["a2a.task.state"] == "completed"
    assert span.attributes["a2a.task.id"]


@_async
async def test_abandoned_stream_still_ends_the_span(make_instrumentor, exporter):
    make_instrumentor()
    rig = Rig(streaming=True)
    try:
        stream = rig.client.send_message(_send("hello"))
        await stream.__anext__()
        await stream.aclose()
    finally:
        await rig.close()
    span = _one(_by_kind(exporter, SpanKind.CLIENT))
    assert span.status.status_code != StatusCode.ERROR


@_async
async def test_client_get_and_cancel(make_instrumentor, exporter):
    make_instrumentor()
    rig = Rig(streaming=False)
    try:
        events = await _drain(rig.client.send_message(_send("hello")))
        task_id = events[-1].task.id
        got = await rig.client.get_task(GetTaskRequest(id=task_id))
        assert got.id == task_id
        with pytest.raises(Exception):
            await rig.client.cancel_task(CancelTaskRequest(id=task_id))
    finally:
        await rig.close()
    client_spans = _by_kind(exporter, SpanKind.CLIENT)
    names = {s.attributes["a2a.method.name"]: s for s in client_spans}
    assert set(names) == {"SendMessage", "GetTask", "CancelTask"}
    assert names["GetTask"].attributes["a2a.task.id"] == task_id
    assert names["GetTask"].attributes["a2a.task.state"] == "completed"
    # Cancelling a finished task is refused by the agent: that is a failed call.
    cancel = names["CancelTask"]
    assert cancel.status.status_code == StatusCode.ERROR
    assert cancel.attributes["error.type"]


# ---- propagation ---------------------------------------------------------------------


@_async
async def test_trace_context_round_trip_one_trace_across_both_agents(
    make_instrumentor, exporter, tracer
):
    make_instrumentor()
    rig = Rig(streaming=False)
    try:
        with tracer.start_as_current_span("caller-agent-turn") as root:
            await _drain(rig.client.send_message(_send("hello")))
    finally:
        await rig.close()

    client = _one(_by_kind(exporter, SpanKind.CLIENT))
    server = _one(_by_kind(exporter, SpanKind.SERVER))
    trace_id = root.get_span_context().trace_id
    assert client.context.trace_id == trace_id
    assert server.context.trace_id == trace_id
    assert server.parent is not None
    assert server.parent.span_id == client.context.span_id

    sent = rig.transport.seen_headers[-1]
    assert sent["traceparent"].split("-")[1] == f"{trace_id:032x}"
    assert sent["traceparent"].split("-")[2] == f"{client.context.span_id:016x}"


@_async
async def test_propagation_can_be_turned_off(make_instrumentor, exporter, tracer):
    make_instrumentor(propagate=False)
    rig = Rig(streaming=False)
    try:
        with tracer.start_as_current_span("caller-agent-turn"):
            await _drain(rig.client.send_message(_send("hello")))
    finally:
        await rig.close()
    client = _one(_by_kind(exporter, SpanKind.CLIENT))
    server = _one(_by_kind(exporter, SpanKind.SERVER))
    assert "traceparent" not in rig.transport.seen_headers[-1]
    assert server.context.trace_id != client.context.trace_id


@_async
async def test_baggage_follows_the_session_baggage_policy(make_instrumentor, tracer):
    from opentelemetry import baggage
    from opentelemetry import context as otel_context

    make_instrumentor()
    rig = Rig(streaming=False)
    try:
        ctx = baggage.set_baggage("tenant.note", "gold")
        token = otel_context.attach(ctx)
        try:
            with tracer.start_as_current_span("caller"):
                await _drain(rig.client.send_message(_send("hello")))
        finally:
            otel_context.detach(token)
    finally:
        await rig.close()
    assert "tenant.note=gold" in rig.transport.seen_headers[-1]["baggage"]


@_async
async def test_callers_context_object_is_not_edited(make_instrumentor, tracer):
    from a2a.client.client import ClientCallContext

    make_instrumentor()
    rig = Rig(streaming=False)
    mine = ClientCallContext(service_parameters={"X-Mine": "1"})
    try:
        with tracer.start_as_current_span("caller"):
            await _drain(rig.client.send_message(_send("hello"), context=mine))
    finally:
        await rig.close()
    assert mine.service_parameters == {"X-Mine": "1"}
    assert rig.transport.seen_headers[-1]["x-mine"] == "1"


# ---- server --------------------------------------------------------------------------


@_async
async def test_standalone_server_span(make_instrumentor, exporter):
    make_instrumentor()
    rig = Rig(streaming=False)
    try:
        await _drain(rig.client.send_message(_send("hello")))
    finally:
        await rig.close()
    server = _one(_by_kind(exporter, SpanKind.SERVER))
    a = server.attributes
    assert server.name == "SendMessage refund-agent"
    assert a["a2a.method.name"] == "SendMessage"
    assert a["a2a.protocol.binding"] == "jsonrpc"
    assert a["gen_ai.agent.name"] == "refund-agent"
    assert a["a2a.task.state"] == "completed"
    assert a["a2a.artifact.count"] == 1
    assert a["a2a.message.text_part_count"] == 1
    assert "gen_ai.operation.name" not in a  # that edge belongs to the client span


@_async
async def test_server_streaming_span(make_instrumentor, exporter):
    make_instrumentor()
    rig = Rig(streaming=True)
    try:
        await _drain(rig.client.send_message(_send("hello")))
    finally:
        await rig.close()
    server = _one(_by_kind(exporter, SpanKind.SERVER))
    assert server.attributes["a2a.method.name"] == "SendStreamingMessage"
    assert server.attributes["a2a.artifact.count"] == 1


@_async
async def test_existing_transport_span_is_enriched_not_nested(
    make_instrumentor, exporter, provider
):
    """With an HTTP server span current (as the ASGI instrumentation leaves it), the
    A2A attributes go on that span, it is renamed, and no ``a2a.server`` span opens."""
    make_instrumentor()
    rig = Rig(streaming=False)
    http_tracer = provider.get_tracer("fake-asgi")

    class _WithTransportSpan(_IsolatedASGITransport):
        async def handle_async_request(self, request):
            async def run():
                with http_tracer.start_as_current_span("POST /a2a", kind=SpanKind.SERVER):
                    return await self._inner.handle_async_request(request)

            return await asyncio.get_running_loop().create_task(
                run(), context=contextvars.Context()
            )

    rig.transport = _WithTransportSpan(Starlette(routes=create_jsonrpc_routes(rig.handler, "/a2a")))
    rig.http = httpx.AsyncClient(transport=rig.transport, base_url="http://agent-b.test:8443")
    rig.client = ClientFactory(
        ClientConfig(
            streaming=False, httpx_client=rig.http, supported_protocol_bindings=["JSONRPC"]
        )
    ).create(rig.card)
    try:
        await _drain(rig.client.send_message(_send("hello")))
    finally:
        await rig.close()
    server = _one(_by_kind(exporter, SpanKind.SERVER))
    assert server.name == "SendMessage refund-agent"  # renamed from "POST /a2a"
    assert server.attributes["a2a.method.name"] == "SendMessage"
    assert server.attributes["a2a.task.state"] == "completed"


@_async
async def test_rest_binding_is_recorded(make_instrumentor, exporter):
    make_instrumentor()
    rig = Rig(binding="HTTP+JSON", streaming=False)
    try:
        await _drain(rig.client.send_message(_send("hello")))
    finally:
        await rig.close()
    assert (
        _one(_by_kind(exporter, SpanKind.CLIENT)).attributes["a2a.protocol.binding"] == "http_json"
    )
    assert (
        _one(_by_kind(exporter, SpanKind.SERVER)).attributes["a2a.protocol.binding"] == "http_json"
    )


# ---- content (A2A-D4) ----------------------------------------------------------------


async def _send_with_file(rig):
    request = _send("file please " + SECRET_TEXT)
    request.message.parts.append(
        Part(raw=b"\x00secret-bytes", media_type="image/png", filename="passport.png")
    )
    request.message.parts.append(Part(data=_value({"card": "4111"})))
    await _drain(rig.client.send_message(request))


def _value(obj):
    from google.protobuf import json_format
    from google.protobuf.struct_pb2 import Value

    return json_format.ParseDict(obj, Value())


def _all_attribute_text(exporter):
    return json.dumps(
        [dict(s.attributes) for s in exporter.get_finished_spans()], default=str, sort_keys=True
    )


@_async
async def test_content_off_records_counts_and_ids_but_no_text(make_instrumentor, exporter):
    make_instrumentor(content=False)
    rig = Rig(streaming=False)
    try:
        await _send_with_file(rig)
    finally:
        await rig.close()
    blob = _all_attribute_text(exporter)
    for secret in (
        SECRET_TEXT,
        REPLY_TEXT,
        "4111",
        "secret-bytes",
        "passport",
        "refund-policy",
        "order-lookup",
    ):
        assert secret not in blob, secret
    client = _one(_by_kind(exporter, SpanKind.CLIENT)).attributes
    assert client["a2a.message.text_part_count"] == 1
    assert client["a2a.message.file_part_count"] == 1
    assert client["a2a.message.data_part_count"] == 1
    assert client["a2a.message.file_media_types"] == ("image/png",)
    assert client["a2a.response.file_part_count"] == 1
    assert client["a2a.response.file_media_types"] == ("application/pdf",)
    for key in (
        "a2a.message.text",
        "a2a.message.data",
        "a2a.response.text",
        "a2a.agent.skill_names",
    ):
        assert key not in client


@_async
async def test_content_on_records_text_data_and_skill_names_but_never_files(
    make_instrumentor, exporter
):
    make_instrumentor(content=True)
    rig = Rig(streaming=False)
    try:
        await _send_with_file(rig)
    finally:
        await rig.close()
    client = _one(_by_kind(exporter, SpanKind.CLIENT)).attributes
    assert SECRET_TEXT in client["a2a.message.text"]
    assert "4111" in client["a2a.message.data"]
    assert REPLY_TEXT in client["a2a.response.text"]
    assert "order" in client["a2a.response.data"]
    assert client["a2a.agent.skill_names"] == ("refund-policy", "order-lookup")
    blob = _all_attribute_text(exporter)
    assert "secret-bytes" not in blob
    assert "%PDF-secret" not in blob
    assert "passport" not in blob  # the file name


# ---- errors (A2A-D5) -----------------------------------------------------------------


@pytest.mark.parametrize("text,state", [("fail now", "failed"), ("reject now", "rejected")])
@_async
async def test_failed_or_rejected_task_is_not_an_error_span(
    make_instrumentor, exporter, text, state
):
    make_instrumentor()
    rig = Rig(streaming=False)
    try:
        await _drain(rig.client.send_message(_send(text)))
    finally:
        await rig.close()
    for span in exporter.get_finished_spans():
        if span.kind in (SpanKind.CLIENT, SpanKind.SERVER):
            assert span.status.status_code != StatusCode.ERROR, span.name
            assert "error.type" not in span.attributes
    assert _one(_by_kind(exporter, SpanKind.CLIENT)).attributes["a2a.task.state"] == state


@_async
async def test_a_raised_a2a_error_marks_both_spans_error(make_instrumentor, exporter):
    make_instrumentor()
    rig = Rig(streaming=False)
    try:
        with pytest.raises(Exception) as caught:
            await rig.client.get_task(GetTaskRequest(id="no-such-task"))
    finally:
        await rig.close()
    client = _one(_by_kind(exporter, SpanKind.CLIENT))
    server = _one(_by_kind(exporter, SpanKind.SERVER))
    assert client.status.status_code == StatusCode.ERROR
    assert client.attributes["error.type"] == type(caught.value).__name__
    assert server.status.status_code == StatusCode.ERROR
    assert server.attributes["error.type"] == "TaskNotFoundError"


@pytest.mark.parametrize("content", [False, True])
def test_error_text_is_kept_only_with_content_capture(make_instrumentor, exporter, content):
    from genai_otel.a2a_semconv.extract import Observation
    from genai_otel.a2a_semconv.instrumentor import _CallScope

    inst = make_instrumentor(content=content)
    secret = "no task for card 4111111111111111"
    span = inst.tracer.start_span("x", kind=SpanKind.CLIENT)
    scope = _CallScope(inst, span, True, Observation(content), Observation(content))
    scope.fail(ValueError(secret))
    scope.finish()
    done = _one(exporter.get_finished_spans())
    blob = json.dumps(
        [done.status.description] + [dict(e.attributes) for e in done.events], default=str
    )
    assert done.status.status_code == StatusCode.ERROR
    assert done.attributes["error.type"] == "ValueError"
    if content:
        assert secret in blob
    else:
        assert secret not in blob
        assert done.status.description == "ValueError"
        assert [e.attributes["exception.type"] for e in done.events] == ["ValueError"]


def test_error_type_for_the_sdks_generic_client_error():
    from genai_otel.a2a_semconv.extract import error_type

    assert error_type(A2AClientError("HTTP Error 503: Service Unavailable")) == "503"
    assert error_type(A2AClientError("gRPC Error UNAVAILABLE: down")) == "UNAVAILABLE"
    assert error_type(A2AClientError("something else")) == "A2AClientError"
    assert error_type(ValueError("x")) == "ValueError"


# ---- the SDK's own spans (A2A-D2) ----------------------------------------------------


def _fresh_interpreter(code, env_extra=None):
    import os

    env = {k: v for k, v in os.environ.items() if k != "OTEL_INSTRUMENTATION_A2A_SDK_ENABLED"}
    env.update(env_extra or {})
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


_BEFORE_IMPORT = """
    import os, sys
    from genai_otel.a2a_semconv import A2AInstrumentor
    from genai_otel.config import OTelConfig
    assert "a2a.utils.telemetry" not in sys.modules
    A2AInstrumentor().instrument(OTelConfig(service_name="t"))
    import a2a.utils.telemetry as t
    print(os.environ.get("OTEL_INSTRUMENTATION_A2A_SDK_ENABLED"), t.otel_enabled)
"""


def test_sdk_spans_are_switched_off_when_the_user_did_not_choose():
    assert _fresh_interpreter(_BEFORE_IMPORT) == "false False"


def test_a_users_choice_for_sdk_spans_is_kept():
    out = _fresh_interpreter(_BEFORE_IMPORT, {"OTEL_INSTRUMENTATION_A2A_SDK_ENABLED": "true"})
    assert out == "true True"


def test_sdk_already_imported_cannot_be_switched_off_and_says_so_once(monkeypatch, caplog):
    import a2a.utils.telemetry as sdk_telemetry

    monkeypatch.delenv("OTEL_INSTRUMENTATION_A2A_SDK_ENABLED", raising=False)
    monkeypatch.setattr(sdk_telemetry, "otel_enabled", True)
    inst = A2AInstrumentor()
    with caplog.at_level(logging.INFO, logger=a2a_module.logger.name):
        inst._manage_sdk_telemetry()
    messages = [r.getMessage() for r in caplog.records if "stay on beside ours" in r.getMessage()]
    assert len(messages) == 1
    import os

    assert "OTEL_INSTRUMENTATION_A2A_SDK_ENABLED" not in os.environ  # no pretending


def test_uninstrument_restores_the_environment_default(monkeypatch):
    import os

    monkeypatch.delenv("OTEL_INSTRUMENTATION_A2A_SDK_ENABLED", raising=False)
    monkeypatch.delitem(sys.modules, "a2a.utils.telemetry")
    inst = A2AInstrumentor()
    inst.config = OTelConfig(service_name="t")
    inst._manage_sdk_telemetry()
    assert os.environ["OTEL_INSTRUMENTATION_A2A_SDK_ENABLED"] == "false"
    inst.uninstrument()
    assert "OTEL_INSTRUMENTATION_A2A_SDK_ENABLED" not in os.environ
    import a2a.utils.telemetry  # noqa: F401  # put the module back for later tests


def test_the_sdks_own_server_spans_are_never_mistaken_for_the_transport_span():
    from genai_otel.a2a_semconv.instrumentor import _is_sdk_span

    class _Scope:
        name = "a2a-python-sdk"

    class _Span:
        instrumentation_scope = _Scope()

    class _Other:
        instrumentation_scope = None

    assert _is_sdk_span(_Span()) is True
    assert _is_sdk_span(_Other()) is False


# ---- gRPC (no server: the SDK hands the handler a ServicerContext, so use a stand-in) ----


class _FakeGrpcContext:
    def __init__(self, metadata):
        self._metadata = metadata

    def invocation_metadata(self):
        return self._metadata


def _grpc_call(metadata):
    from a2a.server.context import ServerCallContext

    return ServerCallContext(state={"grpc_context": _FakeGrpcContext(metadata)}, tenant="t-7")


class _FakeHandler:
    _agent_card = _card(binding="GRPC")


def test_grpc_server_reads_trace_context_from_metadata(make_instrumentor, exporter):
    inst = make_instrumentor()
    parent = f"00-{'ab' * 16}-{'cd' * 8}-01"
    scope = inst._begin_server(
        _FakeHandler(),
        "on_get_task",
        "GetTask",
        GetTaskRequest(id="t-1"),
        _grpc_call([("traceparent", parent), ("a2a-extensions", "x")]),
    )
    scope.finish()
    span = _one(exporter.get_finished_spans())
    assert span.kind == SpanKind.SERVER
    assert f"{span.context.trace_id:032x}" == "ab" * 16
    assert f"{span.parent.span_id:016x}" == "cd" * 8
    assert span.attributes["a2a.protocol.binding"] == "grpc"
    assert span.attributes["a2a.tenant"] == "t-7"
    assert span.attributes["a2a.task.id"] == "t-1"


def test_grpc_transport_span_is_enriched_but_not_renamed(make_instrumentor, exporter, tracer):
    inst = make_instrumentor()
    with tracer.start_as_current_span("/a2a.v1.A2AService/GetTask", kind=SpanKind.SERVER):
        scope = inst._begin_server(
            _FakeHandler(), "on_get_task", "GetTask", GetTaskRequest(id="t-1"), _grpc_call([])
        )
        assert scope.owned is False
        scope.finish()
    span = _one(exporter.get_finished_spans())
    assert span.name == "/a2a.v1.A2AService/GetTask"
    assert span.attributes["a2a.method.name"] == "GetTask"
    assert span.attributes["a2a.protocol.binding"] == "grpc"


def test_grpc_server_ignores_context_when_propagation_is_off(make_instrumentor, exporter):
    inst = make_instrumentor(propagate=False)
    scope = inst._begin_server(
        _FakeHandler(),
        "on_get_task",
        "GetTask",
        GetTaskRequest(id="t-1"),
        _grpc_call([("traceparent", f"00-{'ab' * 16}-{'cd' * 8}-01")]),
    )
    scope.finish()
    assert f"{_one(exporter.get_finished_spans()).context.trace_id:032x}" != "ab" * 16


def test_client_binding_comes_from_the_transport_class():
    from genai_otel.a2a_semconv.extract import binding_from_transport

    class GrpcTransport: ...

    class JsonRpcTransport: ...

    class RestTransport: ...

    class TenantTransportDecorator:
        def __init__(self, base):
            self._base = base

    assert binding_from_transport(GrpcTransport()) == "grpc"
    assert binding_from_transport(JsonRpcTransport()) == "jsonrpc"
    assert binding_from_transport(RestTransport()) == "http_json"
    assert binding_from_transport(TenantTransportDecorator(GrpcTransport())) == "grpc"
    assert binding_from_transport(object()) is None


def test_interceptor_puts_trace_context_in_service_parameters(make_instrumentor, tracer):
    """The same ``service_parameters`` become HTTP headers and gRPC metadata in the SDK."""
    from a2a.client.interceptors import BeforeArgs

    inst = make_instrumentor()
    interceptor = inst._make_interceptor()
    with tracer.start_as_current_span("c") as span:
        args = BeforeArgs(input=None, method="get_task", agent_card=_card(), context=None)
        asyncio.run(interceptor.before(args))
    params = args.context.service_parameters
    assert params["traceparent"].split("-")[2] == f"{span.get_span_context().span_id:016x}"


# ---- lifecycle ------------------------------------------------------------------------


def test_instrument_is_idempotent_and_uninstrument_restores(make_instrumentor):
    from a2a.client.base_client import BaseClient

    original = BaseClient.__dict__["get_task"]
    first = make_instrumentor()
    wrapped = BaseClient.__dict__["get_task"]
    assert wrapped is not original
    second = A2AInstrumentor()
    second.instrument(OTelConfig(service_name="t"))
    assert BaseClient.__dict__["get_task"] is wrapped  # not wrapped twice
    first.uninstrument()
    assert BaseClient.__dict__["get_task"] is original


def test_both_request_handler_generations_are_wrapped(make_instrumentor):
    from a2a.server.request_handlers.default_request_handler import LegacyRequestHandler

    original_legacy = LegacyRequestHandler.__dict__["on_get_task"]
    original_v2 = DefaultRequestHandlerV2.__dict__["on_message_send_stream"]
    make_instrumentor()
    assert LegacyRequestHandler.__dict__["on_get_task"] is not original_legacy
    assert DefaultRequestHandlerV2.__dict__["on_message_send_stream"] is not original_v2


def test_wrapped_methods_keep_their_names(make_instrumentor):
    from a2a.client.base_client import BaseClient

    make_instrumentor()
    assert BaseClient.send_message.__name__ == "send_message"
    assert BaseClient.get_task.__name__ == "get_task"


def test_registered_and_on_by_default():
    from genai_otel.auto_instrument import INSTRUMENTORS
    from genai_otel.config import DEFAULT_INSTRUMENTORS

    assert INSTRUMENTORS["a2a"] is A2AInstrumentor
    assert "a2a" in DEFAULT_INSTRUMENTORS


def test_propagate_setting_reads_its_environment_variable(monkeypatch):
    monkeypatch.setenv("GENAI_A2A_PROPAGATE", "false")
    assert OTelConfig(service_name="t").a2a_propagate is False
    monkeypatch.delenv("GENAI_A2A_PROPAGATE")
    assert OTelConfig(service_name="t").a2a_propagate is True


# ---- no-op without the SDK -------------------------------------------------------------


def test_noop_when_the_sdk_is_older_than_1_0(monkeypatch):
    from a2a.client.base_client import BaseClient

    monkeypatch.setattr(a2a_module, "_sdk_version", lambda: "0.3.25")
    before = dict(vars(BaseClient))
    inst = A2AInstrumentor()
    inst.instrument(OTelConfig(service_name="t"))
    assert dict(vars(BaseClient)) == before
    assert inst._patched == []


def test_noop_when_the_sdk_is_missing(monkeypatch):
    import importlib.util as util

    real = util.find_spec
    monkeypatch.setattr(
        a2a_module.importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "a2a" else real(name, *a, **k),
    )
    inst = A2AInstrumentor()
    inst.instrument(OTelConfig(service_name="t"))
    assert inst._patched == []
    assert A2AInstrumentor.available() is False


def test_import_and_instrument_without_the_sdk_in_a_fresh_interpreter():
    code = """
        import sys
        sys.modules["a2a"] = None   # makes "import a2a" raise ImportError, as if not installed
        from genai_otel.a2a_semconv import A2AInstrumentor
        from genai_otel.config import OTelConfig
        inst = A2AInstrumentor()
        inst.instrument(OTelConfig(service_name="t"))
        print("noop", inst._patched == [])
    """
    assert _fresh_interpreter(code) == "noop True"
