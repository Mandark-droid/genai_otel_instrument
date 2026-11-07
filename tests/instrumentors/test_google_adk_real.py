"""Google ADK traced end to end: a real Runner, a scripted model, no network.

Runs when ``google-adk`` is installed (skipped otherwise). A root agent hands off to a
refunds agent over two turns. Covers:

* the generator fix: a caller that breaks out on the final response (the usual
  pattern in ADK samples) no longer merges later turns into one trace;
* the run span's outcome: the agent the run started at, the agent that answered,
  the handoffs and, with content capture on, the answer;
* content: with capture off, the prompt reaches no span, ours or ADK's.
"""

import asyncio
import gc
import json
import logging
from typing import AsyncGenerator

import pytest

pytest.importorskip("google.adk")

from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.models.base_llm import BaseLlm  # noqa: E402
from google.adk.models.llm_request import LlmRequest  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402
from opentelemetry import context as otel_context  # noqa: E402
from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from genai_otel.config import OTelConfig  # noqa: E402
from genai_otel.instrumentors.google_adk_instrumentor import GoogleADKInstrumentor  # noqa: E402

SECRET = "CARD-4111-1111-1111-1111"
RUN = "google_adk.runner.run"


class ScriptedLlm(BaseLlm):
    """Returns the next scripted response on each call."""

    script: list = []
    calls: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        kind, value = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if kind == "transfer":
            part = types.Part(
                function_call=types.FunctionCall(
                    name="transfer_to_agent", args={"agent_name": value}
                )
            )
        else:
            part = types.Part(text=value)
        usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100, candidates_token_count=20, total_token_count=120
        )
        yield LlmResponse(content=types.Content(role="model", parts=[part]), usage_metadata=usage)


class _DetachFailures(logging.Handler):
    count = 0

    def emit(self, record):
        if "Failed to detach context" in record.getMessage():
            type(self).count += 1


_EXPORTER = InMemorySpanExporter()
_INSTRUMENTOR = None


@pytest.fixture(scope="module")
def instrumentor():
    global _INSTRUMENTOR
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))
    inst = GoogleADKInstrumentor()
    inst.instrument(OTelConfig(enable_content_capture=True))
    inst.tracer = provider.get_tracer("genai_otel.test")
    _INSTRUMENTOR = inst
    logging.getLogger("opentelemetry.context").addHandler(_DetachFailures())
    return inst


@pytest.fixture(autouse=True)
def fresh(instrumentor, monkeypatch):
    _EXPORTER.clear()
    _DetachFailures.count = 0
    monkeypatch.delenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", raising=False)
    token = otel_context.attach(otel_context.Context())
    yield
    otel_context.detach(token)


def _runner():
    refunds = LlmAgent(
        name="refunds_agent",
        model=ScriptedLlm(
            model="fake-refunds",
            script=[("text", "Refund issued for order 42."), ("text", "Anything else?")],
        ),
        instruction="You handle refunds.",
        description="Handles refunds",
    )
    root = LlmAgent(
        name="support_agent",
        model=ScriptedLlm(model="fake-root", script=[("transfer", "refunds_agent")]),
        instruction="Route the customer.",
        sub_agents=[refunds],
    )
    return InMemoryRunner(agent=root, app_name="repro")


def _msg(text):
    return types.Content(role="user", parts=[types.Part(text=text)])


async def _turn(runner, sid, text, break_on_final=False):
    authors = []
    async for event in runner.run_async(user_id="u1", session_id=sid, new_message=_msg(text)):
        authors.append(event.author)
        if break_on_final and event.is_final_response():
            break
    return authors


def _runs():
    return sorted(
        (s for s in _EXPORTER.get_finished_spans() if s.name == RUN),
        key=lambda s: s.start_time,
    )


def _two_turns(break_on_final=False, first=f"My item arrived damaged. Card {SECRET}"):
    async def main():
        runner = _runner()
        session = await runner.session_service.create_session(app_name="repro", user_id="u1")
        await _turn(runner, session.id, first, break_on_final)
        assert not trace.get_current_span().get_span_context().is_valid
        gc.collect()
        await asyncio.sleep(0.05)
        await _turn(runner, session.id, "Thanks", break_on_final)
        assert not trace.get_current_span().get_span_context().is_valid
        gc.collect()
        await asyncio.sleep(0.05)
        await _turn(runner, session.id, "Bye")

    asyncio.run(main())
    gc.collect()


def test_handoff_turns_are_separate_traces_with_their_outcome(instrumentor):
    _two_turns()
    runs = _runs()
    assert len(runs) == 3
    assert len({s.context.trace_id for s in runs}) == 3
    turn1, turn2, _ = runs

    assert turn1.attributes["gen_ai.agent.name"] == "support_agent"
    assert turn1.attributes["google_adk.root_agent.name"] == "support_agent"
    assert turn1.attributes["google_adk.final_agent.name"] == "refunds_agent"
    assert list(turn1.attributes["google_adk.transfers"]) == ["support_agent>refunds_agent"]
    assert turn1.attributes["google_adk.transfer_count"] == 1
    assert turn1.attributes["google_adk.output"] == "Refund issued for order 42."
    assert turn1.attributes["google_adk.invocation_id"]

    # ADK starts the next turn at the agent that answered last.
    assert turn2.attributes["gen_ai.agent.name"] == "refunds_agent"
    assert turn2.attributes["google_adk.root_agent.name"] == "support_agent"
    assert turn2.attributes["google_adk.transfer_count"] == 0

    # Everything ADK traced for a turn sits under that turn's run span.
    by_id = {s.context.span_id: s for s in _EXPORTER.get_finished_spans()}
    for span in _EXPORTER.get_finished_spans():
        if span.name == RUN:
            assert span.parent is None
            continue
        root = span
        while root.parent is not None and root.parent.span_id in by_id:
            root = by_id[root.parent.span_id]
        assert root.name == RUN, f"{span.name} is not under a run span"

    # No token counts on the run span: ADK's model spans carry them.
    for run in runs:
        assert not any(k.startswith("gen_ai.usage.") for k in run.attributes)
    assert _DetachFailures.count == 0


def test_breaking_out_on_the_final_response_keeps_turns_apart(instrumentor):
    _two_turns(break_on_final=True)
    runs = _runs()
    assert len(runs) == 3
    assert len({s.context.trace_id for s in runs}) == 3, "turns merged into one trace"
    assert all(s.parent is None for s in runs)
    assert [bool(s.attributes.get("genai.stream.closed_early")) for s in runs] == [
        True,
        True,
        False,
    ]
    assert _DetachFailures.count == 0


@pytest.mark.parametrize("schema_v1", [False, True])
def test_content_capture_off_puts_the_prompt_on_no_span(instrumentor, monkeypatch, schema_v1):
    monkeypatch.setattr(instrumentor, "config", OTelConfig(enable_content_capture=False))
    if schema_v1:
        monkeypatch.setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", "1")
    # What instrument() does with capture off when the app has not set it.
    monkeypatch.setenv("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", "false")
    _two_turns()
    spans = _EXPORTER.get_finished_spans()
    assert spans
    leaks = [
        (s.name, k)
        for s in spans
        for k, v in (s.attributes or {}).items()
        if SECRET in json.dumps(v, default=str)
    ]
    leaks += [
        (s.name, e.name)
        for s in spans
        for e in s.events
        if SECRET in json.dumps(dict(e.attributes or {}), default=str)
    ]
    assert not leaks
    for run in _runs():
        assert "google_adk.input_message" not in run.attributes
        assert "google_adk.output" not in run.attributes
        # Structural outcome stays.
        assert "google_adk.final_agent.name" in run.attributes


def test_content_capture_on_keeps_input_and_output_capped(instrumentor, monkeypatch):
    monkeypatch.setattr(
        instrumentor, "config", OTelConfig(enable_content_capture=True, content_max_length=10)
    )
    _two_turns()
    turn1 = _runs()[0]
    assert turn1.attributes["google_adk.input_message"] == "My item ar"
    assert turn1.attributes["google_adk.output"] == "Refund iss"


def test_gemini_tokens_are_counted_once_under_adks_model_span(instrumentor):
    """A real google-genai client under ADK, against a local stub of generateContent.

    Our Gemini span must sit inside ADK's model-call span, so a backend that counts
    the innermost span's usage counts ours (which carries cost) once, and the run
    span must carry no usage of its own.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    pytest.importorskip("google.genai")
    from google import genai
    from google.adk.models.google_llm import Gemini

    from genai_otel.instrumentors.google_ai_instrumentor import GoogleAIInstrumentor

    body = json.dumps(
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "Refund issued."}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 11,
                "candidatesTokenCount": 3,
                "totalTokenCount": 14,
            },
            "modelVersion": "gemini-2.5-flash",
        }
    ).encode()

    class Stub(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server API
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        gemini_inst = GoogleAIInstrumentor()
        gemini_inst.instrument(OTelConfig(enable_content_capture=False))
        gemini_inst.tracer = instrumentor.tracer
        client = genai.Client(
            api_key="test-key",
            http_options={"base_url": f"http://127.0.0.1:{server.server_port}"},
        )
        agent = LlmAgent(
            name="solo_agent",
            model=Gemini(model="gemini-2.5-flash", client=client),
            instruction="Answer.",
        )

        async def main():
            runner = InMemoryRunner(agent=agent, app_name="repro")
            session = await runner.session_service.create_session(app_name="repro", user_id="u1")
            await _turn(runner, session.id, "refund please")

        asyncio.run(main())
    finally:
        server.shutdown()

    spans = _EXPORTER.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}
    ours = [s for s in spans if s.name.startswith("google.genai.")]
    assert len(ours) == 1, [s.name for s in spans]
    (ours,) = ours
    assert ours.attributes.get("gen_ai.usage.input_tokens") == 11
    ancestors = []
    node = ours
    while node.parent is not None and node.parent.span_id in by_id:
        node = by_id[node.parent.span_id]
        ancestors.append(node.name)
    assert any(n.startswith(("generate_content", "call_llm")) for n in ancestors), ancestors
    assert ancestors[-1] == RUN
    (run,) = _runs()
    assert not any(k.startswith("gen_ai.usage.") for k in run.attributes)


def test_run_debug_gets_the_same_outcome(instrumentor):
    async def main():
        runner = _runner()
        return await runner.run_debug("refund please", quiet=True)

    events = asyncio.run(main())
    assert events
    (span,) = [s for s in _EXPORTER.get_finished_spans() if s.name == "google_adk.runner.run_debug"]
    assert span.attributes["google_adk.final_agent.name"] == "refunds_agent"
    assert span.attributes["google_adk.transfer_count"] == 1
