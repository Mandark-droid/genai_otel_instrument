"""The queue that holds spans for their detectors: what it keeps, and how it lets go.

Stand-in work items are held and released by the test, so nothing below depends
on how long anything takes.
"""

import asyncio
import os
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genai_otel.config import OTelConfig
from genai_otel.instrumentors import base
from genai_otel.instrumentors.base import BaseInstrumentor

WAIT = 30  # seconds a test waits for something that should happen at once
FILLER = "The statement is generated a few days before the end of each billing cycle. "


class _Work:
    """Stands in for a span under evaluation. ``held`` work stays with its
    detector until the test releases it."""

    def __init__(self, held=False):
        self._release = threading.Event()
        if not held:
            self._release.set()
        self.started = threading.Event()
        self.over = threading.Event()
        self._lock = threading.Lock()
        self.endings = []

    def evaluate(self):
        self.started.set()
        assert self._release.wait(WAIT), "never released"
        self.end()

    def end(self, skipped=None):
        with self._lock:
            if self.endings:
                return False
            self.endings.append(skipped)
        self.over.set()
        return True

    def release(self):
        self._release.set()


@pytest.fixture
def queue():
    """A queue of its own, so that no test here waits for another test's spans."""
    made = base._EvaluationQueue()
    work = []

    def submit(held=False):
        item = _Work(held)
        work.append(item)
        assert made.submit(item, limit=100)
        return item

    made.add = submit
    yield made
    for item in work:
        item.release()
    assert made.drain(WAIT)


def _in_thread(function, *args):
    """Run ``function`` on a thread; returns the thread and a dict that gets its outcome."""
    outcome = {}

    def run():
        try:
            outcome["result"] = function(*args)
        except BaseException as e:  # noqa: BLE001 - the test looks at it
            outcome["error"] = e

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, outcome


# --- a detector that is in the middle of a text when the process ends ------------


def test_a_span_being_evaluated_is_waited_for_even_when_the_timeout_is_zero(queue):
    """A timeout of zero says how long spans may wait for a detector, not that a
    thread may be left inside one: at interpreter exit that aborts the process."""
    workers = base._background_worker_count()
    in_flight = [queue.add(held=True) for _ in range(workers)]
    for item in in_flight:
        assert item.started.wait(WAIT)
    waiting = queue.add()

    thread, outcome = _in_thread(queue.finish, 0)
    # Nothing has started on this one, so it is ended at once...
    assert waiting.over.wait(WAIT)
    # ...and only now do the detectors come back. They are still in time.
    for item in in_flight:
        item.release()
    thread.join(WAIT)

    assert outcome == {"result": 1}
    assert waiting.endings == ["shutdown"]
    assert [item.endings for item in in_flight] == [[None]] * workers


def test_the_wait_for_a_span_being_evaluated_is_at_least_the_floor(queue, monkeypatch):
    waits = []
    drain = queue.drain

    def recorded(timeout=None, upto=None):
        waits.append(timeout)
        return drain(0, upto)

    monkeypatch.setattr(queue, "drain", recorded)
    item = queue.add(held=True)
    assert item.started.wait(WAIT)

    queue.finish(0)
    assert waits == [0, base._IN_FLIGHT_GRACE]

    del waits[:]
    queue.add(held=True)
    queue.finish(base._IN_FLIGHT_GRACE + 10)
    assert waits == [base._IN_FLIGHT_GRACE + 10, base._IN_FLIGHT_GRACE + 10]

    del waits[:]
    queue.add(held=True)
    queue.finish(7, in_flight_timeout=0.5)  # a caller that says how long is taken at its word
    assert waits == [7, 0.5]


# --- a detector that never comes back ---------------------------------------------


def test_a_detector_that_never_returns_is_waited_for_once(queue):
    stuck = queue.add(held=True)
    assert stuck.started.wait(WAIT)

    assert queue.finish(0, in_flight_timeout=0) == 1
    assert stuck.endings == ["shutdown"]

    # Its span has been exported. Nothing waits for it a second time.
    started = time.monotonic()
    assert queue.drain(WAIT) is True
    assert queue.finish(WAIT) == 0
    assert time.monotonic() - started < WAIT / 2


def test_a_worker_lost_to_a_stuck_detector_is_replaced(queue):
    """Otherwise two stuck detectors would leave every later span unevaluated."""
    workers = base._background_worker_count()
    stuck = [queue.add(held=True) for _ in range(workers)]
    for item in stuck:
        assert item.started.wait(WAIT)
    queue.finish(0, in_flight_timeout=0)

    later = queue.add()

    assert later.over.wait(WAIT)
    assert later.endings == [None]


def test_a_stuck_detector_that_does_come_back_does_not_add_a_worker(queue):
    workers = base._background_worker_count()
    stuck = [queue.add(held=True) for _ in range(workers)]
    for item in stuck:
        assert item.started.wait(WAIT)
    queue.finish(0, in_flight_timeout=0)
    replacement = queue.add()
    assert replacement.over.wait(WAIT)

    for item in stuck:
        item.release()
    # A worker that was written off stops when its detector returns; the count
    # of workers the queue knows about is what it was before any got stuck.
    with queue._lock:
        assert queue._idle.wait_for(lambda: not queue._abandoned, timeout=WAIT)
        assert queue._workers <= workers


# --- waits that nobody can wait for -----------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [(3, 3.0), (0, 0.0), (-1, 0.0), (float("nan"), 0.0), (float("inf"), threading.TIMEOUT_MAX)],
)
def test_any_number_is_turned_into_a_wait_a_thread_can_be_asked_for(value, expected):
    assert base._waitable(value) == expected


def test_waiting_for_ever_is_possible(queue):
    """An unbounded number of seconds used to raise instead of waiting."""
    item = queue.add(held=True)
    assert item.started.wait(WAIT)

    thread, outcome = _in_thread(queue.drain, float("inf"))
    item.release()
    thread.join(WAIT)

    assert outcome == {"result": True}


def test_a_wait_that_is_not_a_number_returns_at_once(queue):
    item = queue.add(held=True)
    assert item.started.wait(WAIT)
    assert queue.drain(float("nan")) is False
    assert queue.drain(-5) is False


# --- waiting for what is pending, not for what comes later -------------------------


def test_waiting_for_what_is_pending_does_not_wait_for_what_comes_later(queue):
    """A flush is about the spans that exist when it is called. On a busy service
    there is always a later one, and a wait for all of them never ends."""
    earlier = queue.add(held=True)
    assert earlier.started.wait(WAIT)
    pending_now = queue.submitted
    later = queue.add(held=True)

    thread, outcome = _in_thread(queue.drain, WAIT, pending_now)
    earlier.release()
    thread.join(WAIT)

    assert outcome == {"result": True}
    assert later.endings == []


def test_a_wait_for_what_is_pending_still_waits_for_it(queue):
    item = queue.add(held=True)
    assert item.started.wait(WAIT)
    assert queue.drain(0, upto=queue.submitted) is False


def test_the_public_wait_takes_its_mark_when_it_is_called(monkeypatch):
    queue = base._EvaluationQueue()
    monkeypatch.setattr(base, "_EVALUATION_QUEUE", queue)
    earlier, later = _Work(held=True), _Work(held=True)
    queue.submit(earlier, limit=100)
    assert earlier.started.wait(WAIT)
    try:
        assert base.wait_for_pending_evaluations(0) is False
        earlier.release()
        assert earlier.over.wait(WAIT)
        queue.submit(later, limit=100)
        assert later.started.wait(WAIT)
        # ``later`` was queued before this call, so this call waits for it...
        assert base.wait_for_pending_evaluations(0) is False
    finally:
        earlier.release()
        later.release()
    assert queue.drain(WAIT)
    # ...and with nothing pending there is nothing to wait for.
    assert base.wait_for_pending_evaluations(0) is True


# --- the exit hook ------------------------------------------------------------------


@pytest.fixture
def exit_hook(monkeypatch):
    """The exit hook, with a queue of its own and a recording tracer provider."""
    queue = base._EvaluationQueue()
    monkeypatch.setattr(base, "_EVALUATION_QUEUE", queue)
    events = []
    provider = MagicMock()
    provider.force_flush.side_effect = lambda *a, **k: events.append("flush the provider")
    monkeypatch.setattr(base.trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(
        base, "finish_pending_evaluations", lambda *a: events.append("finish evaluations") or 0
    )
    return SimpleNamespace(queue=queue, events=events, provider=provider)


def test_at_exit_what_has_ended_is_exported_before_anything_else_can_go_wrong(exit_hook):
    """The exit hooks that run after this one include those of the libraries the
    detectors are built on. One of them aborts the process if a detector is
    still inside it, and whatever had not been exported by then was lost."""
    exit_hook.queue.submit(_Work(), limit=100)
    assert exit_hook.queue.drain(WAIT)

    base._finish_evaluations_at_exit()

    assert exit_hook.events == ["finish evaluations", "flush the provider"]


def test_at_exit_nothing_is_done_for_a_process_that_never_queued_a_span(exit_hook):
    base._finish_evaluations_at_exit()
    assert exit_hook.events == []


def test_at_exit_the_provider_is_flushed_even_if_finishing_fails(exit_hook, monkeypatch):
    exit_hook.queue.submit(_Work(), limit=100)
    assert exit_hook.queue.drain(WAIT)

    def fail(*args):
        raise RuntimeError("boom")

    monkeypatch.setattr(base, "finish_pending_evaluations", fail)

    base._finish_evaluations_at_exit()

    assert exit_hook.events == ["flush the provider"]


def test_at_exit_a_provider_that_cannot_flush_is_left_alone(exit_hook, monkeypatch):
    exit_hook.queue.submit(_Work(), limit=100)
    assert exit_hook.queue.drain(WAIT)
    monkeypatch.setattr(base.trace, "get_tracer_provider", lambda: object())
    base._finish_evaluations_at_exit()  # does not raise

    exit_hook.provider.force_flush.side_effect = RuntimeError("collector down")
    monkeypatch.setattr(base.trace, "get_tracer_provider", lambda: exit_hook.provider)
    base._finish_evaluations_at_exit()  # does not raise either


# --- from a signal handler ------------------------------------------------------------


@pytest.fixture
def own_queue(monkeypatch):
    queue = base._EvaluationQueue()
    monkeypatch.setattr(base, "_EVALUATION_QUEUE", queue)
    return queue


def test_finishing_from_a_signal_handler_finishes_and_says_so(own_queue, monkeypatch):
    asked = []
    monkeypatch.setattr(base, "finish_pending_evaluations", lambda *waits: asked.append(waits) or 0)

    assert base.finish_pending_evaluations_in_signal_handler(1.5) is True

    # Two waits, for what is waiting and for what is under way, that together
    # fit inside the handler's own timeout.
    ((waiting, under_way),) = asked
    assert waiting == under_way
    assert 1.0 < waiting + under_way < 1.5


def test_a_span_whose_detector_is_stuck_is_ended_before_the_handler_goes_on(own_queue):
    """The handler flushes next. A span that had not ended by then was not in
    that flush, and a process killed by the signal has no later one."""
    stuck = _Work(held=True)
    own_queue.submit(stuck, limit=100)
    assert stuck.started.wait(WAIT)
    try:
        assert base.finish_pending_evaluations_in_signal_handler(3.0) is True
        assert stuck.endings == ["shutdown"]
    finally:
        stuck.release()


def test_finishing_from_a_signal_handler_is_not_done_on_the_thread_it_interrupted(
    own_queue, monkeypatch
):
    threads = []
    monkeypatch.setattr(
        base,
        "finish_pending_evaluations",
        lambda *waits: threads.append(threading.current_thread()) or 0,
    )

    base.finish_pending_evaluations_in_signal_handler(1.5)

    assert threads and threads[0] is not threading.current_thread()


def test_a_handler_that_interrupted_the_queue_itself_does_not_wait_for_itself(
    own_queue, monkeypatch
):
    """A signal handler runs on the main thread, between two of its instructions.
    If that thread was handing a span to the queue it holds the queue's lock,
    and finishing on the same thread waited for that lock for ever."""
    finished = threading.Event()
    monkeypatch.setattr(base, "finish_pending_evaluations", lambda *waits: finished.set() or 0)
    monkeypatch.setattr(base, "_QUEUE_REACHABLE_WITHIN", 0.05)

    with own_queue._lock:  # this thread is inside the queue, as an interrupted submit is
        started = time.monotonic()
        result = base.finish_pending_evaluations_in_signal_handler(WAIT)
        elapsed = time.monotonic() - started

    assert result is False
    assert elapsed < WAIT / 2
    # Once the interrupted thread is out of the queue the finishing goes ahead.
    assert finished.wait(WAIT)


def test_a_slow_finish_does_not_hold_the_handler_past_its_timeout(own_queue, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(base, "finish_pending_evaluations", lambda *waits: release.wait(WAIT))

    started = time.monotonic()
    result = base.finish_pending_evaluations_in_signal_handler(0.05)
    elapsed = time.monotonic() - started
    release.set()

    assert result is False
    assert elapsed < WAIT / 2


# --- a forked child -------------------------------------------------------------------


def test_a_forked_child_gets_a_detector_pool_of_its_own(monkeypatch):
    """The child has none of the parent's threads. Work given to the parent's
    pool there is never picked up, and the async caller waits for ever."""
    asyncio.run(base._off_the_event_loop(lambda: None))
    inherited = base._EVALUATION_EXECUTOR
    assert inherited is not None
    monkeypatch.setattr(base, "_EVALUATION_EXECUTOR", inherited)
    monkeypatch.setattr(base, "_EVALUATION_EXECUTOR_LOCK", base._EVALUATION_EXECUTOR_LOCK)
    held = base._EVALUATION_EXECUTOR_LOCK
    held.acquire()  # as if another thread had held it at the moment of the fork
    try:
        base._after_fork_in_child()

        assert base._EVALUATION_EXECUTOR is None
        assert base._EVALUATION_EXECUTOR_LOCK is not held
        fresh = base._evaluation_executor()
        assert fresh is not inherited
        assert asyncio.run(base._off_the_event_loop(lambda: "ran")) == "ran"
        fresh.shutdown(wait=True)
    finally:
        held.release()


def test_a_forked_child_gets_a_queue_of_its_own(monkeypatch):
    queue = base._EvaluationQueue()
    monkeypatch.setattr(base, "_EVALUATION_QUEUE", queue)
    monkeypatch.setattr(base, "_EVALUATION_EXECUTOR", base._EVALUATION_EXECUTOR)
    monkeypatch.setattr(base, "_EVALUATION_EXECUTOR_LOCK", base._EVALUATION_EXECUTOR_LOCK)
    item = _Work(held=True)
    queue.submit(item, limit=100)
    assert item.started.wait(WAIT)
    lock = queue._lock
    try:
        base._after_fork_in_child()
        assert queue._lock is not lock
        assert queue.drain(0) is True  # the parent's spans are the parent's
    finally:
        item.release()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork()")
def test_an_async_detector_run_returns_in_a_forked_child():
    assert asyncio.run(base._off_the_event_loop(lambda: "parent")) == "parent"
    pid = os.fork()
    if pid == 0:
        try:
            answer = asyncio.run(
                asyncio.wait_for(base._off_the_event_loop(lambda: "child"), timeout=10)
            )
            os._exit(0 if answer == "child" else 1)
        except BaseException:  # noqa: BLE001
            os._exit(2)
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


# --- what a queued span keeps ------------------------------------------------------------


class _Held:
    """Stands in for a detector; every call waits until the test releases it."""

    def __init__(self, answer):
        self._state = threading.Condition()
        self._released = False
        self.started = 0
        self.calls = []
        self._answer = answer

    def detect(self, text, **kwargs):
        with self._state:
            self.calls.append((text, kwargs))
            self.started += 1
            self._state.notify_all()
            assert self._state.wait_for(lambda: self._released, timeout=WAIT), "never released"
        return self._answer

    def wait_until_started(self):
        with self._state:
            assert self._state.wait_for(lambda: self.started >= 1, timeout=WAIT)

    def release(self):
        with self._state:
            self._released = True
            self._state.notify_all()


_NOT_TOXIC = SimpleNamespace(is_toxic=False, max_score=0.0, toxic_categories=[], scores={})
_NO_PII = SimpleNamespace(
    has_pii=False,
    entity_counts={},
    entities=[],
    score=0.0,
    redacted_text=None,
    blocked=False,
    model_analyzed_length=None,
)
_NO_BIAS = SimpleNamespace(
    has_bias=False, max_score=0.0, detected_biases=[], bias_scores={}, patterns_matched={}
)
_NO_HALLUCINATION = SimpleNamespace(
    has_hallucination=False,
    hallucination_score=0.0,
    citation_count=0,
    hedge_words_count=0,
    factual_claim_count=0,
    hallucination_indicators=[],
    unsupported_claims=[],
)


class _Instrumentor(BaseInstrumentor):
    def instrument(self, config):
        self.config = config

    def _extract_usage(self, result):
        return None


@pytest.fixture
def queued(monkeypatch):
    """Make one call whose detector is held, and look at what the queue holds for it."""
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
    for name, value in cleared.items():
        monkeypatch.setattr(BaseInstrumentor, name, value)
    queue = base._EvaluationQueue()
    monkeypatch.setattr(base, "_EVALUATION_QUEUE", queue)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    held = []

    def call(slot, answer, prompt, response, **attributes):
        detector = _Held(answer)
        for name, value in attributes.items():
            setattr(detector, name, value)
        held.append(detector)
        setattr(BaseInstrumentor, slot, detector)
        instrumentor = _Instrumentor()
        instrumentor.config = OTelConfig(service_name="t")
        instrumentor._instrumented = True
        instrumentor.tracer = provider.get_tracer("test")
        message = SimpleNamespace(content=response, role="assistant", tool_calls=None)
        result = SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)
        wrapped = instrumentor.create_span_wrapper(
            "test.chat", lambda i, a, k: {"gen_ai.operation.name": "chat"}
        )(lambda **kwargs: result)
        wrapped(model="m", messages=[{"role": "user", "content": prompt}])
        detector.wait_until_started()
        (pending,) = queue._running
        return pending

    def finish():
        for detector in held:
            detector.release()
        assert queue.drain(WAIT)
        return exporter.get_finished_spans()[-1]

    call.finish = finish
    yield call
    for detector in held:
        detector.release()
    assert queue.drain(WAIT)


LONG_PROMPT = "Start of a long conversation. " + FILLER * 60
LONG_RESPONSE = "In 2019 exactly 97.3% agreed. " + FILLER * 60


def test_a_queued_span_keeps_the_part_of_the_prompt_its_detector_reads(queued):
    """A full queue of whole prompts and responses is a great deal of memory
    held for text that no detector was going to look at."""
    pending = queued("_toxicity_detector", _NOT_TOXIC, LONG_PROMPT, LONG_RESPONSE)

    # Toxicity alone is configured, and it reads the last 500 characters.
    assert pending.prompt == LONG_PROMPT[-500:]
    assert pending.response is None

    span = queued.finish()
    # The span still says how long the prompt was and how much of it was read.
    assert span.attributes["evaluation.truncated"] is True
    assert span.attributes["evaluation.prompt.length"] == len(LONG_PROMPT)
    assert span.attributes["evaluation.toxicity.prompt.analyzed_length"] == 500
    assert "evaluation.prompt.analyzed_length" not in span.attributes
    assert "evaluation.response.length" not in span.attributes


def test_a_queued_span_keeps_the_largest_part_any_of_its_detectors_reads(queued, monkeypatch):
    """With toxicity next to a detector that reads more, the larger part is kept
    and each of the two is given its own."""
    reader = _Held(_NO_BIAS)
    reader.release()
    monkeypatch.setattr(BaseInstrumentor, "_bias_detector", reader)
    pending = queued("_toxicity_detector", _NOT_TOXIC, LONG_PROMPT, LONG_RESPONSE)

    assert pending.prompt == LONG_PROMPT[-1500:]

    span = queued.finish()
    assert [text for text, _ in BaseInstrumentor._toxicity_detector.calls] == [LONG_PROMPT[-500:]]
    assert [text for text, _ in reader.calls] == [LONG_PROMPT[-1500:]]
    assert span.attributes["evaluation.prompt.analyzed_length"] == 1500
    assert span.attributes["evaluation.toxicity.prompt.analyzed_length"] == 500


def test_a_queued_span_keeps_the_part_of_the_response_its_detector_reads(queued):
    pending = queued("_hallucination_detector", _NO_HALLUCINATION, LONG_PROMPT, LONG_RESPONSE)

    assert pending.response == LONG_RESPONSE[:1500]
    assert pending.prompt == LONG_PROMPT[-1500:]

    span = queued.finish()
    assert span.attributes["evaluation.response.length"] == len(LONG_RESPONSE)
    assert span.attributes["evaluation.response.analyzed_length"] == 1500
    assert span.attributes["evaluation.prompt.length"] == len(LONG_PROMPT)
    assert span.attributes["evaluation.prompt.analyzed_length"] == 1500


def test_a_queued_span_keeps_the_whole_prompt_for_a_detector_that_reads_all_of_it(queued):
    """The PII patterns cover what the model does not read."""
    pending = queued("_pii_detector", _NO_PII, LONG_PROMPT, LONG_RESPONSE, accepts_model_chars=True)

    assert pending.prompt == LONG_PROMPT
    assert pending.response is None

    queued.finish()
    (detector,) = [BaseInstrumentor._pii_detector]
    assert detector.calls == [(LONG_PROMPT, {"model_chars": 1500})]


def test_a_queued_span_keeps_the_whole_prompt_for_a_redacting_detector(queued):
    pending = queued("_pii_detector", _NO_PII, LONG_PROMPT, LONG_RESPONSE, reads_whole_text=True)
    assert pending.prompt == LONG_PROMPT


def test_a_pii_detector_that_only_takes_text_gets_the_part_and_no_more_is_kept(queued):
    pending = queued("_pii_detector", _NO_PII, LONG_PROMPT, LONG_RESPONSE)
    assert pending.prompt == LONG_PROMPT[-1500:]


def test_a_queued_span_keeps_the_whole_prompt_for_a_block_policy_that_reads_all_of_it(
    queued, monkeypatch
):
    monkeypatch.setattr(BaseInstrumentor, "_blocking_policies", frozenset({"bias"}))
    pending = queued("_bias_detector", _NO_BIAS, LONG_PROMPT, LONG_RESPONSE)

    assert pending.prompt == LONG_PROMPT

    span = queued.finish()
    assert span.attributes["evaluation.bias.prompt.analyzed_length"] == len(LONG_PROMPT)


def test_short_texts_are_kept_as_they_are(queued):
    pending = queued("_hallucination_detector", _NO_HALLUCINATION, "Short question?", "Yes.")

    assert pending.prompt == "Short question?"
    assert pending.response == "Yes."

    span = queued.finish()
    assert "evaluation.truncated" not in span.attributes
