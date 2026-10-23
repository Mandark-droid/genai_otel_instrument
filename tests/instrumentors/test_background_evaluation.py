"""Detectors run after the call has returned, on a worker thread.

A detector is CPU work that grows with the prompt. Run inline it is added to
every call's latency. In the default mode the wrapper hands the still-open span
to a worker and returns; the worker runs the detectors and then ends the span as
of the moment the call returned. The span processors and the exporter therefore
see the same span they always did, with the same attributes.

The stand-in detectors here are held and released by the test, so nothing below
depends on how long anything takes.
"""

import asyncio
import json
import logging
import subprocess
import sys
import textwrap
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genai_otel.config import OTelConfig
from genai_otel.exceptions import PolicyViolationError
from genai_otel.instrumentors import base
from genai_otel.instrumentors.azure_openai_instrumentor import AzureOpenAIInstrumentor
from genai_otel.instrumentors.base import BaseInstrumentor
from genai_otel.instrumentors.groq_instrumentor import GroqInstrumentor
from genai_otel.instrumentors.litellm_latency_instrumentor import LiteLLMLatencyInstrumentor
from genai_otel.instrumentors.openai_instrumentor import OpenAIInstrumentor

CARD = "4111 1111 1111 1111"
WAIT = 30  # seconds a test waits for something that should happen at once


def _messages():
    return [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "My card number is {0}.".format(CARD)},
    ]


def _chat_result():
    message = SimpleNamespace(content="Noted.", role="assistant", tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
    usage = SimpleNamespace(prompt_tokens=12, completion_tokens=2, total_tokens=14)
    return SimpleNamespace(id="chatcmpl-1", model="m", choices=[choice], usage=usage)


class _Detector:
    """Stands in for the PII detector. Flags the card number.

    With ``held=True`` every call waits inside ``detect`` until the test calls
    ``release()``: that is how a test keeps the workers busy for as long as it
    needs to look at what the caller sees meanwhile.
    """

    def __init__(self, held=False):
        self._state = threading.Condition()
        self._released = not held
        self.started = 0
        self.seen = []
        self.threads = []

    def detect(self, text):
        with self._state:
            self.seen.append(text)
            self.threads.append(threading.get_ident())
            self.started += 1
            self._state.notify_all()
            assert self._state.wait_for(lambda: self._released, timeout=WAIT), "never released"
        found = CARD in text
        return SimpleNamespace(
            has_pii=found,
            entity_counts={"CREDIT_CARD": 1} if found else {},
            entities=[CARD] if found else [],
            score=0.9 if found else 0.0,
            redacted_text=None,
            blocked=False,
        )

    def wait_until_started(self, count):
        with self._state:
            assert self._state.wait_for(lambda: self.started >= count, timeout=WAIT)

    def release(self):
        with self._state:
            self._released = True
            self._state.notify_all()


class _Instrumentor(BaseInstrumentor):
    def instrument(self, config):
        self.config = config

    def _extract_usage(self, result):
        return None


def _attributes(instance, args, kwargs):
    return {"gen_ai.system": "test", "gen_ai.operation.name": "chat"}


@pytest.fixture
def detectors():
    """Detector slots are class-level: start from none configured, and put back
    whatever was there. Any detector a test left held is released first, so no
    worker is still inside one when the next test starts."""
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
    installed = []

    def install(held=False):
        detector = _Detector(held=held)
        installed.append(detector)
        BaseInstrumentor._pii_detector = detector
        return detector

    yield install
    for detector in installed:
        detector.release()
    assert base.drain_evaluations(WAIT)
    for name, value in saved.items():
        setattr(BaseInstrumentor, name, value)


@pytest.fixture
def tracing():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter.get_finished_spans


def _instrumentor(tracer, **config):
    instrumentor = _Instrumentor()
    instrumentor.config = OTelConfig(service_name="t", **config)
    instrumentor._instrumented = True
    instrumentor.tracer = tracer
    return instrumentor


def _wrapped(tracer, calls=None, **config):
    """A chat call through the shared span wrapper, with ``config`` applied."""
    instrumentor = _instrumentor(tracer, **config)

    def create(*args, **kwargs):
        if calls is not None:
            calls.append(kwargs)
        return _chat_result()

    return instrumentor.create_span_wrapper("test.chat", _attributes)(create)


# --- the caller does not wait for the detectors --------------------------------


def test_the_call_returns_while_the_detectors_are_still_running(detectors, tracing):
    tracer, spans = tracing
    detector = detectors(held=True)
    call = _wrapped(tracer)

    result = call(model="m", messages=_messages())

    # The caller has its response. The detector has not answered yet.
    assert result.choices[0].message.content == "Noted."
    detector.wait_until_started(1)
    assert spans() == ()

    detector.release()
    assert base.drain_evaluations(WAIT)
    (span,) = spans()
    assert span.attributes["evaluation.pii.prompt.detected"] is True
    assert span.attributes["evaluation.pii.prompt.entity_count"] == 1
    assert span.attributes["evaluation.completed"] is True
    assert "evaluation.skipped" not in span.attributes


def test_the_detectors_run_on_another_thread(detectors, tracing):
    tracer, _ = tracing
    detector = detectors()

    _wrapped(tracer)(model="m", messages=_messages())
    assert base.drain_evaluations(WAIT)

    assert detector.threads and threading.get_ident() not in detector.threads


def test_the_span_lasts_as_long_as_the_call_not_as_long_as_the_detectors(detectors, tracing):
    tracer, spans = tracing
    detector = detectors(held=True)

    _wrapped(tracer)(model="m", messages=_messages())
    returned_at = time.time_ns()
    detector.wait_until_started(1)
    held_until = time.time_ns()
    detector.release()
    assert base.drain_evaluations(WAIT)

    (span,) = spans()
    assert span.start_time <= span.end_time <= returned_at <= held_until
    assert span.status.status_code.name == "OK"


def test_the_detectors_see_the_request_that_was_sent(detectors, tracing):
    """An application appends the answer and the next question to the same list as
    soon as the call returns. That must not change what is evaluated."""
    tracer, _ = tracing
    blocker = detectors(held=True)
    call = _wrapped(tracer)
    # Keep every worker busy, so that the call under test waits in the queue.
    for _ in range(base._background_worker_count()):
        call(model="m", messages=[{"role": "user", "content": "filler"}])
    blocker.wait_until_started(base._background_worker_count())
    messages = _messages()

    call(model="m", messages=messages)
    messages.append({"role": "user", "content": "And my passport number is X1234567."})
    messages[1]["content"] = "changed after the call"
    blocker.release()
    assert base.drain_evaluations(WAIT)

    assert blocker.seen[-1] == "You are terse.\nMy card number is {0}.".format(CARD)


def test_inline_mode_runs_the_detectors_before_the_call_returns(detectors, tracing):
    tracer, spans = tracing
    detector = detectors()

    _wrapped(tracer, evaluation_mode="inline")(model="m", messages=_messages())

    # No waiting for a worker: the span has ended and carries the result.
    (span,) = spans()
    assert span.attributes["evaluation.pii.prompt.detected"] is True
    assert detector.threads == [threading.get_ident()]


def test_without_detectors_nothing_is_handed_to_a_worker(detectors, tracing):
    """The common case stays as it was: no queue, no thread, no extra attribute."""
    tracer, spans = tracing
    queue = MagicMock()

    with patch.object(base, "_EVALUATION_QUEUE", queue):
        _wrapped(tracer)(model="m", messages=_messages())

    assert queue.mock_calls == []
    (span,) = spans()
    assert not [key for key in span.attributes if key.startswith("evaluation.")]


def test_a_detector_that_fails_still_lets_the_span_end(detectors, tracing):
    tracer, spans = tracing
    detectors()

    def broken(text):
        raise RuntimeError("model not loaded")

    BaseInstrumentor._pii_detector = SimpleNamespace(detect=broken)

    result = _wrapped(tracer)(model="m", messages=_messages())
    assert base.drain_evaluations(WAIT)

    assert result.choices[0].message.content == "Noted."
    (span,) = spans()
    assert span.status.status_code.name == "OK"


def test_a_failed_call_ends_its_span_at_once(detectors, tracing):
    tracer, spans = tracing
    detector = detectors(held=True)
    instrumentor = _instrumentor(tracer)

    def create(*args, **kwargs):
        raise ConnectionError("no route to host")

    with pytest.raises(ConnectionError):
        instrumentor.create_span_wrapper("test.chat", _attributes)(create)(
            model="m", messages=_messages()
        )

    (span,) = spans()
    assert span.status.status_code.name == "ERROR"
    assert detector.started == 0


def test_a_blocked_request_is_still_refused_before_it_is_sent(detectors, tracing):
    """The block check cannot move to a worker: it has to answer first."""
    tracer, spans = tracing
    detector = detectors()
    BaseInstrumentor._blocking_policies = frozenset({"pii"})
    BaseInstrumentor._blocking_active = True
    calls = []

    with pytest.raises(PolicyViolationError):
        _wrapped(tracer, calls)(model="m", messages=_messages())

    assert calls == []
    assert detector.threads == [threading.get_ident()]
    (span,) = spans()
    assert span.attributes["evaluation.blocked"] is True
    assert span.attributes["evaluation.pii.outcome"] == "completed"
    assert span.attributes["evaluation.content.outcomes"] == ("omitted", "rejected")


# --- when the detectors cannot keep up -----------------------------------------


def _fill(call, detector, waiting):
    """Occupy every worker, then leave ``waiting`` more spans in the queue."""
    workers = base._background_worker_count()
    for _ in range(workers):
        call(model="m", messages=_messages())
    detector.wait_until_started(workers)
    for _ in range(waiting):
        call(model="m", messages=_messages())
    return workers + waiting


def test_past_the_limit_a_span_is_exported_at_once_and_says_why(detectors, tracing):
    tracer, spans = tracing
    detector = detectors(held=True)
    call = _wrapped(tracer, evaluation_queue_size=2)
    accepted = _fill(call, detector, waiting=2)
    assert spans() == ()

    result = call(model="m", messages=_messages())

    # The caller was not held up and the span was not held back.
    assert result.choices[0].message.content == "Noted."
    (span,) = spans()
    assert span.attributes["evaluation.skipped"] == "overloaded"
    assert "evaluation.pii.prompt.detected" not in span.attributes
    assert "evaluation.completed" not in span.attributes
    assert span.status.status_code.name == "OK"

    detector.release()
    assert base.drain_evaluations(WAIT)
    evaluated = [s for s in spans() if s.attributes.get("evaluation.completed")]
    assert len(evaluated) == accepted
    assert len(spans()) == accepted + 1


def test_falling_behind_is_logged_once_not_once_per_span(detectors, tracing, caplog):
    tracer, spans = tracing
    detector = detectors(held=True)
    call = _wrapped(tracer, evaluation_queue_size=1)
    _fill(call, detector, waiting=1)

    # Start from "nothing logged or counted yet", whatever earlier tests left.
    queue = base._EVALUATION_QUEUE
    with patch.object(queue, "_overload_logged_at", None), patch.object(queue, "_overloaded", 0):
        with caplog.at_level(logging.WARNING, logger=base.__name__):
            for _ in range(25):
                call(model="m", messages=_messages())
            lines = [r for r in caplog.records if "evaluation.skipped" in r.getMessage()]
            assert len(lines) == 1

            # What was not reported yet is not lost: it is reported at shutdown.
            detector.release()
            base.finish_pending_evaluations(WAIT)
            lines = [r for r in caplog.records if "evaluation.skipped" in r.getMessage()]
            assert len(lines) == 2
            assert "24" in lines[1].getMessage()

    assert sum(1 for s in spans() if s.attributes.get("evaluation.skipped") == "overloaded") == 25


# --- shutdown ---------------------------------------------------------------


def test_at_shutdown_what_has_not_been_evaluated_is_exported_and_says_so(
    detectors, tracing, caplog
):
    tracer, spans = tracing
    detector = detectors(held=True)
    call = _wrapped(tracer, evaluation_queue_size=10)
    accepted = _fill(call, detector, waiting=3)
    assert spans() == ()

    with caplog.at_level(logging.WARNING):
        ended = base.finish_pending_evaluations(timeout=0.05)

        assert ended == accepted
        assert len(spans()) == accepted
        assert {s.attributes["evaluation.skipped"] for s in spans()} == {"shutdown"}

        # The detectors that were in the middle of a span answer afterwards. The
        # span has gone; nothing may be written to it and it must not end twice.
        detector.release()
        assert base.drain_evaluations(WAIT)

    assert len(spans()) == accepted
    assert not [s for s in spans() if "evaluation.pii.prompt.detected" in s.attributes]
    assert not [r for r in caplog.records if "ended span" in r.getMessage()]


def test_shutdown_waits_for_detectors_that_finish_in_time(detectors, tracing):
    tracer, spans = tracing
    detectors()
    call = _wrapped(tracer)
    for _ in range(5):
        call(model="m", messages=_messages())

    assert base.finish_pending_evaluations(timeout=WAIT) == 0

    assert len(spans()) == 5
    assert all(s.attributes["evaluation.pii.prompt.detected"] is True for s in spans())
    assert not [s for s in spans() if "evaluation.skipped" in s.attributes]


def test_the_queue_works_again_after_a_shutdown_drain(detectors, tracing):
    """uninstrument() followed by instrument() must not leave evaluation dead."""
    tracer, spans = tracing
    detectors()
    call = _wrapped(tracer)
    call(model="m", messages=_messages())
    base.finish_pending_evaluations(timeout=WAIT)

    call(model="m", messages=_messages())
    assert base.drain_evaluations(WAIT)

    assert len(spans()) == 2
    assert all(s.attributes.get("evaluation.completed") for s in spans())


def test_drain_reports_whether_everything_finished(detectors, tracing):
    tracer, _ = tracing
    detector = detectors(held=True)
    _wrapped(tracer)(model="m", messages=_messages())
    detector.wait_until_started(1)

    assert base.drain_evaluations(timeout=0.01) is False
    detector.release()
    assert base.drain_evaluations(WAIT) is True


def test_the_exit_hook_runs_before_tracer_providers_registered_so_far():
    """Exit hooks run last-registered first, and a tracer provider registers its
    shutdown when it is created. Registering again moves ours in front of it."""
    with patch.object(base, "atexit") as atexit:
        base.register_evaluation_exit_hook()
        base.register_evaluation_exit_hook()

    names = [call[0] for call in atexit.mock_calls]
    assert names == ["unregister", "register", "unregister", "register"]
    assert {call.args[0] for call in atexit.mock_calls} == {base._finish_evaluations_at_exit}


_EXIT_SCRIPT = """
    import json
    import sys
    import threading
    from types import SimpleNamespace

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult

    from genai_otel.config import OTelConfig
    from genai_otel.instrumentors.base import BaseInstrumentor

    mode, calls = sys.argv[1], int(sys.argv[2])


    class Printer(SpanExporter):
        def export(self, spans):
            for span in spans:
                attributes = dict(span.attributes)
                print(
                    "SPAN="
                    + json.dumps(
                        {
                            "skipped": attributes.get("evaluation.skipped"),
                            "completed": attributes.get("evaluation.completed"),
                            "detected": attributes.get("evaluation.pii.prompt.detected"),
                        }
                    )
                )
            sys.stdout.flush()
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass


    never = threading.Event()


    class Detector:
        def detect(self, text):
            if mode == "stuck":
                never.wait()
            return SimpleNamespace(
                has_pii=True,
                entity_counts={"CREDIT_CARD": 1},
                entities=["x"],
                score=0.9,
                redacted_text=None,
                blocked=False,
            )


    class Instrumentor(BaseInstrumentor):
        def instrument(self, config):
            self.config = config

        def _extract_usage(self, result):
            return None


    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(Printer()))
    BaseInstrumentor._pii_detector = Detector()
    instrumentor = Instrumentor()
    instrumentor.config = OTelConfig(service_name="t", evaluation_shutdown_timeout=0.2)
    instrumentor._instrumented = True
    instrumentor.tracer = provider.get_tracer("exit")
    call = instrumentor.create_span_wrapper(
        "test.chat", lambda i, a, k: {"gen_ai.operation.name": "chat"}
    )(lambda **kwargs: SimpleNamespace(choices=[]))
    for _ in range(calls):
        call(model="m", messages=[{"role": "user", "content": "hello"}])
    # The process ends here, with spans still waiting for their detectors.
"""


def _spans_exported_by_a_process_that_exits(mode, calls):
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_EXIT_SCRIPT), mode, str(calls)],
        capture_output=True,
        text=True,
        timeout=180,
        stdin=subprocess.DEVNULL,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return [
        json.loads(line[len("SPAN=") :])
        for line in proc.stdout.splitlines()
        if line.startswith("SPAN=")
    ]


def test_a_process_that_exits_right_after_a_burst_exports_every_span():
    spans = _spans_exported_by_a_process_that_exits("quick", calls=40)

    assert len(spans) == 40
    assert all(span["completed"] and span["detected"] for span in spans)
    assert not [span for span in spans if span["skipped"]]


def test_a_process_whose_detectors_never_finish_still_exits_and_exports_every_span():
    spans = _spans_exported_by_a_process_that_exits("stuck", calls=40)

    assert len(spans) == 40
    assert {span["skipped"] for span in spans} == {"shutdown"}


# --- async callers ----------------------------------------------------------


def test_a_coroutine_call_returns_without_waiting_for_the_detectors(detectors, tracing):
    tracer, spans = tracing
    detector = detectors(held=True)
    instrumentor = _instrumentor(tracer)

    async def create(*args, **kwargs):
        return _chat_result()

    call = instrumentor.create_span_wrapper("test.chat", _attributes)(create)

    async def main():
        result = await call(model="m", messages=_messages())
        return result, threading.get_ident()

    result, loop_thread = asyncio.run(main())

    assert result.choices[0].message.content == "Noted."
    detector.wait_until_started(1)
    assert spans() == ()
    detector.release()
    assert base.drain_evaluations(WAIT)
    (span,) = spans()
    assert span.attributes["evaluation.pii.prompt.detected"] is True
    assert loop_thread not in detector.threads


def _async_openai(tracer, **config):
    instrumentor = OpenAIInstrumentor()
    instrumentor.config = OTelConfig(service_name="t", **config)

    async def create(*args, **kwargs):
        return _chat_result()

    with patch("opentelemetry.trace.get_tracer", return_value=tracer):
        wrapped = instrumentor._create_async_span_wrapper(
            span_name="openai.chat.completion",
            extract_attributes=instrumentor._extract_openai_attributes,
        )(create)

    def call():
        async def main():
            result = await wrapped(model="gpt-4o-mini", messages=_messages())
            return result, threading.get_ident()

        with patch("opentelemetry.trace.get_tracer", return_value=tracer):
            return asyncio.run(main())

    return call


def test_async_openai_returns_without_waiting_for_the_detectors(detectors, tracing):
    tracer, spans = tracing
    detector = detectors(held=True)

    with patch.object(base, "_off_the_event_loop", new=MagicMock()) as awaited:
        result, loop_thread = _async_openai(tracer)()

    assert result.choices[0].message.content == "Noted."
    # Handed over, not awaited: the coroutine is finished while the detector runs.
    awaited.assert_not_called()
    detector.wait_until_started(1)
    assert spans() == ()
    detector.release()
    assert base.drain_evaluations(WAIT)
    (span,) = spans()
    assert span.attributes["evaluation.pii.prompt.detected"] is True
    assert span.status.status_code.name == "OK"
    assert loop_thread not in detector.threads


def test_async_openai_inline_mode_waits_but_keeps_the_event_loop_free(detectors, tracing):
    tracer, spans = tracing
    detector = detectors()

    _, loop_thread = _async_openai(tracer, evaluation_mode="inline")()

    (span,) = spans()
    assert span.attributes["evaluation.pii.prompt.detected"] is True
    assert detector.threads and loop_thread not in detector.threads


# --- the wrappers that build their own span -------------------------------------


def _groq(tracer, config):
    with patch.dict("sys.modules", {"groq": MagicMock()}):
        instrumentor = GroqInstrumentor()
    instrumentor.tracer = tracer
    instrumentor.config = OTelConfig(service_name="t", **config)
    instrumentor._instrumented = True
    completions = SimpleNamespace(create=lambda *args, **kwargs: _chat_result())
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    instrumentor._instrument_client(client)
    return lambda: client.chat.completions.create(model="m", messages=_messages())


def _azure_openai(tracer, config):
    import types

    class OpenAIClient:
        def complete(self, *args, **kwargs):
            return _chat_result()

    module = types.ModuleType("azure.ai.openai")
    module.OpenAIClient = OpenAIClient
    modules = {"azure": MagicMock(), "azure.ai": MagicMock(), "azure.ai.openai": module}
    with patch.dict("sys.modules", modules):
        instrumentor = AzureOpenAIInstrumentor()
        instrumentor.tracer = tracer
        instrumentor.instrument(OTelConfig(service_name="t", **config))
    client = OpenAIClient()
    return lambda: client.complete(model="m", messages=_messages())


def _litellm(tracer, config):
    with patch.dict("sys.modules", {"litellm": MagicMock()}):
        instrumentor = LiteLLMLatencyInstrumentor()
    instrumentor.tracer = tracer
    instrumentor.config = OTelConfig(service_name="t", **config)
    instrumentor._instrumented = True
    wrapped = instrumentor._wrap_sync(lambda *a, **k: _chat_result(), "litellm.completion")
    return lambda: wrapped(model="m", messages=_messages())


def _litellm_async(tracer, config):
    with patch.dict("sys.modules", {"litellm": MagicMock()}):
        instrumentor = LiteLLMLatencyInstrumentor()
    instrumentor.tracer = tracer
    instrumentor.config = OTelConfig(service_name="t", **config)
    instrumentor._instrumented = True

    async def acompletion(*args, **kwargs):
        return _chat_result()

    wrapped = instrumentor._wrap_async(acompletion, "litellm.acompletion")
    return lambda: asyncio.run(wrapped(model="m", messages=_messages()))


OWN_SPAN = [_groq, _azure_openai, _litellm, _litellm_async]


@pytest.mark.parametrize("driver", OWN_SPAN, ids=lambda d: d.__name__.strip("_"))
def test_own_span_wrapper_returns_without_waiting_for_the_detectors(driver, detectors, tracing):
    tracer, spans = tracing
    detector = detectors(held=True)

    result = driver(tracer, {})()

    assert result.choices[0].message.content == "Noted."
    detector.wait_until_started(1)
    assert spans() == ()
    detector.release()
    assert base.drain_evaluations(WAIT)
    (span,) = spans()
    assert span.attributes["evaluation.pii.prompt.detected"] is True
    assert span.attributes["evaluation.completed"] is True
    assert threading.get_ident() not in detector.threads


@pytest.mark.parametrize("driver", OWN_SPAN, ids=lambda d: d.__name__.strip("_"))
def test_own_span_wrapper_inline_mode_has_the_result_when_the_call_returns(
    driver, detectors, tracing
):
    tracer, spans = tracing
    detectors()

    driver(tracer, {"evaluation_mode": "inline"})()

    (span,) = spans()
    assert span.attributes["evaluation.pii.prompt.detected"] is True


@pytest.mark.parametrize("driver", OWN_SPAN, ids=lambda d: d.__name__.strip("_"))
def test_own_span_wrapper_without_detectors_ends_its_span_itself(driver, detectors, tracing):
    tracer, spans = tracing
    queue = MagicMock()

    with patch.object(base, "_EVALUATION_QUEUE", queue):
        driver(tracer, {})()

    assert queue.mock_calls == []
    assert len(spans()) == 1


class _SignallingExporter(InMemorySpanExporter):
    """An in-memory exporter a test can wait on for a number of spans."""

    def __init__(self):
        super().__init__()
        self._arrived = threading.Condition()

    def export(self, spans):
        result = super().export(spans)
        with self._arrived:
            self._arrived.notify_all()
        return result

    def wait_for(self, count):
        with self._arrived:
            assert self._arrived.wait_for(
                lambda: len(self.get_finished_spans()) >= count, timeout=WAIT
            )


def test_a_detector_in_the_middle_of_a_span_at_shutdown_is_allowed_to_finish(detectors):
    """A detector abandoned inside a native library while the interpreter shuts
    down can abort the process, and with it every span not yet exported. So a
    span whose detectors have started is waited for; only the ones nobody has
    started on are given up at once."""
    exporter = _SignallingExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    detector = detectors(held=True)
    call = _wrapped(provider.get_tracer("test"), evaluation_queue_size=10)
    in_flight = _fill(call, detector, waiting=3) - 3
    outcome = {}

    def finish():
        outcome["ended"] = base._EVALUATION_QUEUE.finish(0.05, in_flight_timeout=WAIT)

    finisher = threading.Thread(target=finish)
    finisher.start()
    # Time is up for the three no detector had started on: they are exported now.
    exporter.wait_for(3)
    assert {s.attributes["evaluation.skipped"] for s in exporter.get_finished_spans()} == {
        "shutdown"
    }
    # The detectors that were in the middle of a span finish it.
    detector.release()
    finisher.join(WAIT)
    assert not finisher.is_alive()

    assert outcome["ended"] == 3
    spans = exporter.get_finished_spans()
    assert len(spans) == 3 + in_flight
    evaluated = [s for s in spans if "evaluation.skipped" not in s.attributes]
    assert len(evaluated) == in_flight
    assert all(s.attributes["evaluation.completed"] is True for s in evaluated)
    assert all(s.attributes["evaluation.pii.prompt.detected"] is True for s in evaluated)
