"""Spans waiting for their detectors are dealt with before telemetry is shut down.

In the default evaluation mode a span is exported when its detectors are done,
which may be after the call returned. Anything that shuts down or flushes the
tracer provider has to let those spans end first: a span that ends after its
provider has shut down is dropped.
"""

import signal
import threading
import time
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import genai_otel.auto_instrument as auto_instrument
from genai_otel.auto_instrument import (
    _install_sigterm_flush_handler,
    flush_telemetry,
    setup_auto_instrumentation,
    uninstrument,
)
from genai_otel.config import OTelConfig
from genai_otel.instrumentors import base


@pytest.fixture
def providers(monkeypatch):
    """Fake active providers; ``events`` records the order things happen in."""
    events = []
    tracer, meter = MagicMock(), MagicMock()
    tracer.force_flush.side_effect = lambda *a: events.append("flush traces") or True
    meter.force_flush.side_effect = lambda *a: events.append("flush metrics") or True
    tracer.shutdown.side_effect = lambda *a: events.append("shut down traces")
    meter.shutdown.side_effect = lambda *a: events.append("shut down metrics")
    monkeypatch.setattr(auto_instrument, "_active_tracer_provider", tracer)
    monkeypatch.setattr(auto_instrument, "_active_meter_provider", meter)
    monkeypatch.setattr(auto_instrument, "_active_gpu_collector", None)
    return events


@pytest.fixture
def restore_sigterm():
    original = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, original)


def test_uninstrument_finishes_pending_evaluations_before_shutting_the_provider_down(
    providers, monkeypatch
):
    monkeypatch.setattr(auto_instrument, "_INSTRUMENTATION_INITIALIZED", True)

    def finish(timeout=None):
        providers.append("finish evaluations (timeout %r)" % (timeout,))
        return 0

    with patch.object(base, "finish_pending_evaluations", side_effect=finish):
        uninstrument()

    assert providers == [
        "finish evaluations (timeout None)",
        "shut down traces",
        "shut down metrics",
    ]


def test_uninstrument_shuts_down_even_if_finishing_evaluations_fails(providers, monkeypatch):
    monkeypatch.setattr(auto_instrument, "_INSTRUMENTATION_INITIALIZED", True)

    with patch.object(base, "finish_pending_evaluations", side_effect=RuntimeError("boom")):
        uninstrument()

    assert providers == ["shut down traces", "shut down metrics"]
    assert auto_instrument._INSTRUMENTATION_INITIALIZED is False


def test_flush_waits_for_pending_evaluations_first(providers):
    def wait(timeout=None):
        providers.append("wait for evaluations (%.1f s)" % timeout)
        return True

    with patch.object(base, "wait_for_pending_evaluations", side_effect=wait):
        assert flush_telemetry(2.0) is True

    assert providers == ["wait for evaluations (2.0 s)", "flush traces", "flush metrics"]


def test_flush_says_so_when_evaluations_did_not_finish_in_time(providers):
    """Their spans are not exported yet, so not everything was flushed."""
    with patch.object(base, "wait_for_pending_evaluations", return_value=False):
        assert flush_telemetry(0.5) is False

    assert providers == ["flush traces", "flush metrics"]


def test_flush_goes_ahead_if_waiting_for_evaluations_fails(providers):
    with patch.object(base, "wait_for_pending_evaluations", side_effect=RuntimeError("boom")):
        assert flush_telemetry(0.5) is False

    assert providers == ["flush traces", "flush metrics"]


def test_flush_does_not_wait_for_calls_made_after_it(providers):
    """On a service that keeps making calls the queue is never empty. A flush
    that waited for it to be would use up its whole timeout every time."""
    with patch.object(base, "drain_evaluations", side_effect=AssertionError("waits for all")):
        assert flush_telemetry(2.0) is True


def test_sigterm_exports_what_the_detectors_have_not_finished(providers, restore_sigterm):
    """A terminated process runs no exit hooks. What is still waiting for its
    detectors has to be ended by the handler, or it is lost."""
    cfg = OTelConfig(service_name="t")
    cfg.sigterm_flush_timeout = 1.5
    signal.signal(signal.SIGTERM, lambda s, f: providers.append("previous handler"))
    assert _install_sigterm_flush_handler(cfg) is True
    handler = signal.getsignal(signal.SIGTERM)

    waits = []

    def finish(timeout=None, in_flight_timeout=None):
        waits.append(timeout + in_flight_timeout)
        providers.append("finish evaluations")
        return 3

    with patch.object(base, "finish_pending_evaluations", side_effect=finish):
        handler(signal.SIGTERM, None)

    assert providers == [
        "finish evaluations",
        "flush traces",
        "flush metrics",
        "previous handler",
    ]
    # Everything the detectors are given fits inside the handler's timeout.
    assert 1.0 < waits[0] < 1.5


def test_sigterm_finishes_evaluations_on_another_thread(providers, restore_sigterm):
    """The handler runs on the main thread, wherever that thread happened to be."""
    cfg = OTelConfig(service_name="t")
    signal.signal(signal.SIGTERM, lambda s, f: None)
    assert _install_sigterm_flush_handler(cfg) is True
    handler = signal.getsignal(signal.SIGTERM)
    threads = []

    with patch.object(
        base,
        "finish_pending_evaluations",
        side_effect=lambda *waits: threads.append(threading.current_thread()) or 0,
    ):
        handler(signal.SIGTERM, None)

    assert len(threads) == 1
    assert threads[0] is not threading.current_thread()


def test_sigterm_while_the_main_thread_is_inside_the_queue_does_not_hang(
    providers, restore_sigterm, monkeypatch
):
    """A signal that arrives while the main thread is handing a span to the queue
    finds the queue's lock taken, by the very thread the handler runs on."""
    cfg = OTelConfig(service_name="t")
    cfg.sigterm_flush_timeout = 30.0
    signal.signal(signal.SIGTERM, lambda s, f: providers.append("previous handler"))
    assert _install_sigterm_flush_handler(cfg) is True
    handler = signal.getsignal(signal.SIGTERM)
    queue = base._EvaluationQueue()
    monkeypatch.setattr(base, "_EVALUATION_QUEUE", queue)
    monkeypatch.setattr(base, "_QUEUE_REACHABLE_WITHIN", 0.05)
    finished = threading.Event()

    with patch.object(
        base, "finish_pending_evaluations", side_effect=lambda *waits: finished.set() or 0
    ):
        with queue._lock:
            started = time.monotonic()
            handler(signal.SIGTERM, None)
            elapsed = time.monotonic() - started
        # It still exported what had ended, and let the application's handler run.
        assert providers == ["flush traces", "flush metrics", "previous handler"]
        assert elapsed < 10
        # With the main thread out of the queue again, the finishing goes ahead.
        assert finished.wait(30)


def test_instrument_registers_the_exit_hook_after_creating_the_tracer_provider(monkeypatch):
    """Exit hooks run last-registered first. Registered after the provider, the
    hook that ends pending spans runs before the provider's own shutdown."""
    monkeypatch.setattr(auto_instrument, "_INSTRUMENTATION_INITIALIZED", False)
    monkeypatch.setattr(auto_instrument, "_active_tracer_provider", None)
    monkeypatch.setattr(auto_instrument, "_active_meter_provider", None)
    events = MagicMock()

    with ExitStack() as stack:
        provider_class = stack.enter_context(patch.object(auto_instrument, "TracerProvider"))
        stack.enter_context(patch.object(auto_instrument, "trace"))
        stack.enter_context(patch.object(auto_instrument, "metrics"))
        stack.enter_context(patch.object(auto_instrument, "Resource"))
        stack.enter_context(patch("genai_otel.auto_instrument._check_openinference"))
        register = stack.enter_context(patch.object(base, "register_evaluation_exit_hook"))
        events.attach_mock(provider_class, "create_provider")
        events.attach_mock(register, "register_exit_hook")

        setup_auto_instrumentation(
            OTelConfig(
                service_name="t",
                endpoint="",
                enable_gpu_metrics=False,
                enable_mcp_instrumentation=False,
                enabled_instrumentors=[],
                # Pinned, not inherited from the shell: a detector enabled here
                # would be registered for every test that runs afterwards.
                enable_pii_detection=False,
                enable_toxicity_detection=False,
                enable_bias_detection=False,
                enable_prompt_injection_detection=False,
                enable_restricted_topics=False,
                enable_hallucination_detection=False,
            )
        )
        monkeypatch.setattr(auto_instrument, "_INSTRUMENTATION_INITIALIZED", False)

    order = [name for name, _, _ in events.mock_calls if "." not in name]
    assert order == ["create_provider", "register_exit_hook"]


# --- the evaluation span processor, flushed and shut down with its provider -----


class _HeldDetector:
    """Stands in for the PII detector; every call waits until the test releases it."""

    def __init__(self, held):
        self._state = threading.Condition()
        self._released = not held
        self.started = 0

    def detect(self, text):
        with self._state:
            self.started += 1
            self._state.notify_all()
            assert self._state.wait_for(lambda: self._released, timeout=30), "never released"
        return SimpleNamespace(
            has_pii=False,
            entity_counts={},
            entities=[],
            score=0.0,
            redacted_text=None,
            blocked=False,
        )

    def wait_until_started(self):
        with self._state:
            assert self._state.wait_for(lambda: self.started >= 1, timeout=30)

    def release(self):
        with self._state:
            self._released = True
            self._state.notify_all()


class _Instrumentor(base.BaseInstrumentor):
    def instrument(self, config):
        self.config = config

    def _extract_usage(self, result):
        return None


@pytest.fixture
def pipeline():
    """A provider set up the way instrument() sets one up: the evaluation
    processor first, the exporting processor after it."""
    from genai_otel.evaluation.config import BiasConfig
    from genai_otel.evaluation.span_processor import EvaluationSpanProcessor

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(EvaluationSpanProcessor(bias_config=BiasConfig(enabled=True)))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    detectors = []

    def call(held):
        detector = _HeldDetector(held)
        detectors.append(detector)
        base.BaseInstrumentor._bias_detector = None
        base.BaseInstrumentor._pii_detector = detector
        instrumentor = _Instrumentor()
        instrumentor.config = OTelConfig(service_name="t", evaluation_shutdown_timeout=0.05)
        instrumentor._instrumented = True
        instrumentor.tracer = provider.get_tracer("test")
        wrapped = instrumentor.create_span_wrapper(
            "test.chat", lambda i, a, k: {"gen_ai.operation.name": "chat"}
        )(lambda **kwargs: SimpleNamespace(choices=[]))
        wrapped(model="m", messages=[{"role": "user", "content": "hello"}])
        return detector

    yield SimpleNamespace(provider=provider, call=call, spans=exporter.get_finished_spans)
    for detector in detectors:
        detector.release()
    base.drain_evaluations(30)


def test_flushing_the_provider_waits_for_spans_still_being_evaluated(pipeline):
    """Applications flush their provider at the end of a request or of a serverless
    invocation. That flush has to include the spans the detectors are still on."""
    pipeline.call(held=False)

    assert pipeline.provider.force_flush(30000) is True

    (span,) = pipeline.spans()
    assert span.attributes["evaluation.completed"] is True


def test_flushing_the_provider_does_not_wait_for_spans_that_come_later(pipeline):
    """A request handler flushes at the end of its request. The spans of requests
    that other threads make meanwhile are theirs to flush."""
    pipeline.call(held=False)
    assert base.drain_evaluations(30)
    waited = base._EVALUATION_QUEUE.drain

    def traffic_goes_on(timeout=None, upto=None):
        pipeline.call(held=True)  # another request, made while the flush is waiting
        return waited(timeout, upto)

    with patch.object(base._EVALUATION_QUEUE, "drain", side_effect=traffic_goes_on):
        started = time.monotonic()
        assert pipeline.provider.force_flush(30000) is True
        elapsed = time.monotonic() - started

    assert elapsed < 10  # half of the timeout is 15 s
    assert len(pipeline.spans()) == 1


def test_a_flush_does_not_wait_for_ever_for_a_detector_that_is_stuck(pipeline):
    """It gives up within its timeout and still lets the exporter flush: one slow
    detector must not hold back every other span."""
    detector = pipeline.call(held=True)
    detector.wait_until_started()

    assert pipeline.provider.force_flush(50) is True
    assert pipeline.spans() == ()


def test_shutting_the_provider_down_exports_what_is_still_being_evaluated(pipeline):
    """Whoever shuts the provider down, and in whatever order exit hooks run."""
    detector = pipeline.call(held=True)
    detector.wait_until_started()

    pipeline.provider.shutdown()

    (span,) = pipeline.spans()
    assert span.attributes["evaluation.skipped"] == "shutdown"
