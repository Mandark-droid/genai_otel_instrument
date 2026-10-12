"""Base classes for OpenTelemetry instrumentors for GenAI libraries and tools.

This module defines the `BaseInstrumentor` abstract base class, which provides
common functionality and a standardized interface for instrumenting various
Generative AI (GenAI) libraries and Model Context Protocol (MCP) tools.
It includes methods for creating OpenTelemetry spans, recording metrics,
and handling configuration and cost calculation.
"""

import ast
import asyncio
import atexit
import collections
import contextvars
import functools
import inspect
import json
import logging
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import wrapt
from opentelemetry import context as otel_context
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

from .. import agent_budget
from ..config import (
    DEFAULT_EVALUATION_MAX_CHARS,
    DEFAULT_EVALUATION_QUEUE_SIZE,
    DEFAULT_EVALUATION_SHUTDOWN_TIMEOUT,
    DEFAULT_TOXICITY_MAX_CHARS,
    EVALUATION_MODE_INLINE,
    OTelConfig,
)
from ..cost_calculator import CostCalculator
from ..exceptions import PolicyViolationError
from ..semconv import SemanticConvention as SC
from ..semconv import genai_semconv_modes, genai_tier_opted_in
from ..server_metrics import get_server_metrics


def _unwrap_api_response(result: Any) -> Any:
    """Return the parsed model when handed an OpenAI raw-response wrapper.

    LiteLLM calls the OpenAI SDK in raw-response mode so it can read rate-limit
    headers, so ``chat.completions.create`` hands back
    ``openai._legacy_response.LegacyAPIResponse`` rather than a
    ``ChatCompletion``. That wrapper exposes no ``.usage`` -- the model is only
    reachable through ``.parse()`` -- so every ``hasattr(result, "usage")`` check
    was False and usage, cost and finish-reason were dropped without a word.

    The failure was silent and therefore expensive: spans were created, request
    attributes were correct, and only the numbers were missing. Where a gateway
    is the standard inference entry point that zeroes cost dashboards and spend
    attribution for every service behind it -- and a zeroed cost reads as "this
    was free" rather than "this was not measured".

    Degrades to the original object if the wrapper cannot be parsed:
    instrumentation must never raise into the caller. Streaming is unaffected,
    since the only caller runs on the non-streaming branch, so no stream is
    consumed here. ``LegacyAPIResponse.parse()`` memoises, so this does not
    re-deserialise on repeated access.

    See https://github.com/Mandark-droid/genai_otel_instrument/issues/10
    """
    if result is None or hasattr(result, "usage"):
        return result
    parse = getattr(result, "parse", None)
    if not callable(parse):
        return result
    try:
        parsed = parse()
    except Exception:  # noqa: BLE001 - telemetry must never break the call
        return result
    return parsed if parsed is not None else result


# Import histogram bucket definitions
try:
    from genai_otel.metrics import _GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS
except ImportError:
    # Fallback buckets if import fails
    _GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS = [
        0.01,
        0.02,
        0.04,
        0.08,
        0.16,
        0.32,
        0.64,
        1.28,
        2.56,
        5.12,
        10.24,
        20.48,
        40.96,
        81.92,
    ]

logger = logging.getLogger(__name__)
# Global flag to track if shared metrics have been created
_SHARED_METRICS_CREATED = False
_SHARED_METRICS_LOCK = threading.Lock()

# Shared default CostCalculator so the ~150 KB pricing JSON is parsed once for the
# whole process instead of once per instrumentor instance (previously ~29 parses,
# ~14 MB of duplicate dicts). Instrumentors that supply custom pricing get their
# own instance in _setup_config().
_DEFAULT_COST_CALCULATOR = None
_DEFAULT_COST_CALCULATOR_LOCK = threading.Lock()


def _get_default_cost_calculator() -> "CostCalculator":
    """Return the process-wide shared CostCalculator, creating it once."""
    global _DEFAULT_COST_CALCULATOR
    if _DEFAULT_COST_CALCULATOR is None:
        with _DEFAULT_COST_CALCULATOR_LOCK:
            if _DEFAULT_COST_CALCULATOR is None:
                _DEFAULT_COST_CALCULATOR = CostCalculator()
    return _DEFAULT_COST_CALCULATOR


# Attribute keys that carry prompt or completion TEXT. With content capture off none of
# them may reach a span. Several instrumentors used to write some of them regardless
# (gen_ai.response in Bedrock, Groq, Mistral, SambaNova and Azure OpenAI; the Responses
# API's and Bedrock Converse's instructions). The guard sits where the base wrapper writes
# attributes, so a new instrumentor cannot forget it.
_CONTENT_KEYS = frozenset({"gen_ai.response", "gen_ai.request.instructions", "gen_ai.prompt"})
_CONTENT_PREFIXES = ("gen_ai.prompt.", "gen_ai.completion.")


def strip_content_attributes(
    attrs: Dict[str, Any], config: Optional["OTelConfig"]
) -> Dict[str, Any]:
    """Drop prompt / completion text from ``attrs`` unless content capture is on.

    Without a config (direct use, unit tests) attributes pass unchanged, as before.
    """
    if not config or getattr(config, "enable_content_capture", False):
        return attrs
    return {
        k: v
        for k, v in attrs.items()
        if k not in _CONTENT_KEYS and not k.startswith(_CONTENT_PREFIXES)
    }


def content_capture_on(config: Optional["OTelConfig"]) -> bool:
    """Whether prompt / completion text may be put on a span.

    Without a config (direct use, unit tests) it may, as in
    :func:`strip_content_attributes`.
    """
    return config is None or bool(getattr(config, "enable_content_capture", False))


# Worker threads for detectors run on behalf of async callers. A pool of its own
# rather than the event loop's default executor: applications use that one, and so
# does name resolution, and a queue of detector runs must not hold them up.
_EVALUATION_EXECUTOR: Optional[ThreadPoolExecutor] = None
_EVALUATION_EXECUTOR_LOCK = threading.Lock()
_EVALUATION_WORKERS = 4


def _evaluation_executor() -> ThreadPoolExecutor:
    """The detector worker pool, created the first time a detector needs it."""
    global _EVALUATION_EXECUTOR
    if _EVALUATION_EXECUTOR is None:
        with _EVALUATION_EXECUTOR_LOCK:
            if _EVALUATION_EXECUTOR is None:
                _EVALUATION_EXECUTOR = ThreadPoolExecutor(
                    max_workers=min(_EVALUATION_WORKERS, os.cpu_count() or 1),
                    thread_name_prefix="genai-otel-eval",
                )
    return _EVALUATION_EXECUTOR


async def _off_the_event_loop(function, *args):
    """Run ``function`` on a detector worker thread and wait for it.

    A detector is CPU work: tens to hundreds of milliseconds on a long prompt.
    Run inside a coroutine it would stall every other task on the loop for that
    long. The caller's context goes with it, so the span stays current there.
    """
    loop = asyncio.get_running_loop()
    context = contextvars.copy_context()
    return await loop.run_in_executor(
        _evaluation_executor(), functools.partial(context.run, function, *args)
    )


# ---------------------------------------------------------------------------
# Detectors after the call, off the caller's thread.
#
# The detectors that annotate a span do not change what the caller gets back,
# so the caller has no reason to wait for them. A wrapper that has finished a
# call hands the still-open span to the queue below and returns. A worker runs
# the detectors, writes their attributes and ends the span, with the end time
# taken when the call returned: the span measures the call, and every span
# processor and exporter sees it complete, as when the detectors ran inline.
#
# Threads of its own rather than the executor above. An executor finishes
# everything queued on it before the interpreter runs a single exit hook, with
# no way to say "enough", and refuses new work once the main thread has
# returned. Here the wait at exit is bounded and whatever is left is exported
# unevaluated instead of being lost.
# ---------------------------------------------------------------------------

# Seconds between two "detectors are not keeping up" log lines.
_OVERLOAD_LOG_INTERVAL = 60.0

# Threads that run detectors in the background. Few, on purpose. The Python
# parts of a detector hold the interpreter lock, so each thread busy with one is
# another thread the application's own threads have to take turns with, and the
# pattern-based detectors, which are Python throughout, gain nothing from more.
_BACKGROUND_EVALUATION_WORKERS = 2


def _background_worker_count() -> int:
    """How many threads may run detectors in the background at once."""
    return min(_BACKGROUND_EVALUATION_WORKERS, os.cpu_count() or 1)


class _SpanUnderEvaluation:
    """A span whose call has returned and whose detectors have not finished.

    The detector pass writes its attributes through this object rather than
    straight to the span. That is what lets the span be ended from elsewhere,
    at shutdown, while a detector is still inside it: the pass goes quiet
    instead of writing to a span that has already been exported.

    ``prompt`` and ``response`` are what the detectors will read and no more;
    the lengths are those of the texts they were taken from. See
    :meth:`BaseInstrumentor._texts_to_keep`.
    """

    __slots__ = (
        "instrumentor",
        "span",
        "prompt",
        "response",
        "prompt_length",
        "response_length",
        "end_time",
        "sequence",
        "_lock",
        "_ended",
    )

    def __init__(
        self,
        instrumentor,
        span,
        prompt,
        response,
        end_time: int,
        prompt_length: Optional[int] = None,
        response_length: Optional[int] = None,
    ):
        self.instrumentor = instrumentor
        self.span = span
        self.prompt = prompt
        self.response = response
        self.prompt_length = prompt_length
        self.response_length = response_length
        self.end_time = end_time
        # Its place in the order spans were queued in; set by the queue.
        self.sequence = 0
        self._lock = threading.Lock()
        self._ended = False

    def set_attribute(self, key: str, value: Any) -> None:
        with self._lock:
            if not self._ended:
                self.span.set_attribute(key, value)

    def end(self, skipped: Optional[str] = None) -> bool:
        """End the span as of the moment its call returned.

        ``skipped`` says why it ends unevaluated. Returns False when the span
        had been ended already, so that it is exported exactly once whoever
        gets here first.
        """
        with self._lock:
            if self._ended:
                return False
            self._ended = True
        try:
            if skipped:
                self.span.set_attribute("evaluation.skipped", skipped)
        finally:
            self.span.end(end_time=self.end_time)
        return True

    def evaluate(self) -> None:
        """Run the detectors, then end the span. Never raises."""
        token = None
        try:
            # Current here as it is when detectors run inline, so that anything
            # a detector logs or traces is attributed to this call.
            token = otel_context.attach(trace.set_span_in_context(self.span))
            self.instrumentor._evaluate_texts(
                self, self.prompt, self.response, self.prompt_length, self.response_length
            )
        except Exception as e:  # noqa: BLE001 - the span must end whatever a detector does
            logger.warning(
                "Failed to run evaluation checks for span '%s': %s",
                getattr(self.span, "name", "?"),
                e,
            )
        finally:
            try:
                if token is not None:
                    otel_context.detach(token)
            finally:
                self.end()


class _SpanForDetectors:
    """A span as the detectors see it while an async caller waits for them.

    The caller awaits a worker thread. Cancelled, the caller stops waiting and
    the thread does not stop working: what it went on to write landed on a span
    the caller had ended by then. Told to :meth:`stop`, the detectors' writes go
    nowhere instead.
    """

    __slots__ = ("_span", "_lock", "_stopped")

    def __init__(self, span):
        self._span = span
        self._lock = threading.Lock()
        self._stopped = False

    @property
    def attributes(self):
        return self._span.attributes

    @property
    def name(self):
        return getattr(self._span, "name", "?")

    def is_recording(self) -> bool:
        return self._span.is_recording()

    def set_attribute(self, key: str, value: Any) -> None:
        with self._lock:
            if not self._stopped:
                self._span.set_attribute(key, value)

    def stop(self) -> None:
        with self._lock:
            self._stopped = True


# Shortest time a detector that is in the middle of a text is given to finish it
# when the queue is being emptied, however short the configured timeout. At
# interpreter exit a thread left inside a detector's native code is not
# interrupted: the exit hook of the library it is in aborts the process. One
# pass over one text is bounded by the limit on analysed text and takes a small
# fraction of this on any machine, so the floor costs nothing when all is well
# and is the difference between an exit and a crash when a pass is under way.
_IN_FLIGHT_GRACE = 2.0

# How long a signal handler waits to see whether the queue can be reached at
# all before concluding that the thread it interrupted is the one holding it.
_QUEUE_REACHABLE_WITHIN = 0.25

# The part of a signal handler's timeout that each of its two waits for the
# detectors gets. A little under half, so that both are over before the
# handler stops waiting for them.
_SIGNAL_SHARE = 0.45


def _waitable(seconds: Any) -> float:
    """``seconds`` as a wait a thread can be asked for: 0 up to the platform's limit.

    A wait of infinity raises instead of waiting, and one that is not a number
    or is negative has no meaning; both used to leave spans that were never
    ended. Not-a-number and negative become 0.
    """
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return 0.0
    if not seconds > 0:  # also true of not-a-number
        return 0.0
    return min(seconds, threading.TIMEOUT_MAX)


class _EvaluationQueue:
    """Spans waiting for their detectors, and the threads that run them."""

    def __init__(self):
        self._reset()

    def _reset(self) -> None:
        self._lock = threading.Lock()
        self._work = threading.Condition(self._lock)
        self._idle = threading.Condition(self._lock)
        self._waiting: Deque[_SpanUnderEvaluation] = collections.deque()
        self._running: Set[_SpanUnderEvaluation] = set()
        # Spans ended from under a detector that did not come back. Nothing
        # waits for them any more, and their threads are not counted as workers.
        self._abandoned: Set[_SpanUnderEvaluation] = set()
        self._workers = 0
        self._free_workers = 0
        self._overloaded = 0
        self._overload_logged_at: Optional[float] = None
        self._exit_hook_registered = False
        # How many spans have been queued so far. Each takes the count as its
        # sequence number, which is how a wait can be for "everything queued up
        # to now" on a queue that never stops being added to.
        self.submitted = 0
        self.shutdown_timeout = DEFAULT_EVALUATION_SHUTDOWN_TIMEOUT

    def has_room(self, limit: int) -> bool:
        """Whether another span may be queued. A hint: :meth:`submit` decides.

        ``limit`` spans may wait on top of the ones the workers can be running.
        Counted that way, a span queued a moment before a free worker picks it
        up does not take a place in the line.
        """
        return len(self._waiting) + len(self._running) < limit + _background_worker_count()

    def submit(self, pending: _SpanUnderEvaluation, limit: int) -> bool:
        """Queue a span. False when the line is full; never blocks."""
        with self._lock:
            if not self.has_room(limit):
                return False
            if not self._exit_hook_registered:
                self._exit_hook_registered = True
                register_evaluation_exit_hook()
            self.submitted += 1
            pending.sequence = self.submitted
            self._waiting.append(pending)
            if self._free_workers:
                # Taken off the count here, not by the worker when it wakes: a
                # second span queued before it does must not count on it too.
                self._free_workers -= 1
                self._work.notify()
            elif self._workers < _background_worker_count():
                try:
                    self._start_worker()
                except RuntimeError:  # the process cannot start another thread
                    if not self._workers:
                        self._waiting.remove(pending)
                        return False
        return True

    def _start_worker(self) -> None:
        thread = threading.Thread(
            target=self._work_loop,
            name="genai-otel-evaluation-%d" % self._workers,
            # A worker must never keep the process alive. What it has not done
            # by exit is dealt with by the exit hook.
            daemon=True,
        )
        thread.start()
        self._workers += 1

    def _work_loop(self) -> None:
        while True:
            with self._lock:
                while not self._waiting:
                    self._free_workers += 1
                    self._work.wait()
                pending = self._waiting.popleft()
                self._running.add(pending)
            try:
                pending.evaluate()
            except BaseException as e:  # noqa: BLE001 - a worker that dies takes the queue with it
                logger.debug("Evaluation worker error: %s", e)
            finally:
                with self._lock:
                    self._running.discard(pending)
                    written_off = pending in self._abandoned
                    self._abandoned.discard(pending)
                    # Whoever is waiting decides whether this was what it was
                    # waiting for: a flush waits for some spans, not for all.
                    self._idle.notify_all()
            if written_off:
                # This thread was given up on while its detector was stuck, and
                # another has taken its place.
                return

    def _acquire(self, deadline: Optional[float]) -> bool:
        """Take the lock, by ``deadline`` at the latest. False if it could not be had."""
        if deadline is None:
            return self._lock.acquire()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return self._lock.acquire(False)
        return self._lock.acquire(timeout=remaining)

    def _pending(self, upto: Optional[int]) -> bool:
        """Whether a span is waiting or running. With ``upto``, one queued no later than that."""
        if upto is None:
            return bool(self._waiting or self._running)
        # Spans wait in the order they were queued in.
        if self._waiting and self._waiting[0].sequence <= upto:
            return True
        return any(pending.sequence <= upto for pending in self._running)

    def drain(self, timeout: Optional[float] = None, upto: Optional[int] = None) -> bool:
        """Wait until nothing is waiting or running. False if ``timeout`` ran out first.

        With ``upto``, a sequence number, only spans queued up to then are
        waited for. Spans a stuck detector was holding when :meth:`finish` gave
        up on them are not waited for at all.

        The timeout covers getting to the queue as well as waiting on it, so a
        caller that cannot get to it (a signal handler that interrupted a thread
        which was inside it) comes back too.
        """
        deadline = None if timeout is None else time.monotonic() + _waitable(timeout)
        if not self._acquire(deadline):
            return False
        try:
            while self._pending(upto):
                if deadline is None:
                    self._idle.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle.wait(remaining)
        finally:
            self._lock.release()
        return True

    def finish(self, timeout: float, in_flight_timeout: Optional[float] = None) -> int:
        """Give the detectors ``timeout`` seconds, then end what is left unevaluated.

        Spans no detector has started on are ended when the time is up. A span a
        detector is in the middle of is waited for, for up to ``in_flight_timeout``
        more, and ended unevaluated only then. By default that is as long again
        and never less than a floor: see ``_IN_FLIGHT_GRACE``.

        The second wait is not a courtesy. At interpreter exit this runs before
        the exit hooks of the libraries the detectors are built on, and some of
        those tear their native state down there. A detector still inside one
        at that point does not get an exception: the process is aborted. One
        pass over one text is bounded, so letting it complete is cheap next to
        that.

        A detector that has not come back by then is given up on. Its span is
        ended from under it, nothing waits for it again, and its thread no
        longer counts as a worker, so that a detector which never returns costs
        one wait and not one at every later flush and shutdown.

        Returns how many spans were ended without their detectors having finished.
        """
        timeout = _waitable(timeout)
        if in_flight_timeout is None:
            in_flight_timeout = max(timeout, _IN_FLIGHT_GRACE)
        drained = self.drain(timeout)
        left: List[_SpanUnderEvaluation] = []
        with self._lock:
            if not drained:
                left = list(self._waiting)
                self._waiting.clear()
            unreported, self._overloaded = self._overloaded, 0
        ended = sum(1 for pending in left if pending.end(skipped="shutdown"))
        if not drained and not self.drain(in_flight_timeout):
            with self._lock:
                stuck = list(self._running)
                self._running.difference_update(stuck)
                self._abandoned.update(stuck)
                self._workers -= len(stuck)
                self._idle.notify_all()
            # Ended here all the same, from under the detector; see
            # _SpanUnderEvaluation.
            ended += sum(1 for pending in stuck if pending.end(skipped="shutdown"))
        if unreported:
            self._log_overload(unreported)
        if ended:
            logger.warning(
                "%d span(s) were exported without evaluation, marked "
                'evaluation.skipped="shutdown": the detectors had not got to them '
                "%.1f s after shutdown began (GENAI_EVALUATION_SHUTDOWN_TIMEOUT).",
                ended,
                timeout,
            )
        return ended

    def note_overload(self) -> None:
        """Count a span that could not wait. Logs at intervals, never once per span."""
        now = time.monotonic()
        with self._lock:
            self._overloaded += 1
            last = self._overload_logged_at
            if last is not None and now - last < _OVERLOAD_LOG_INTERVAL:
                return
            count, self._overloaded = self._overloaded, 0
            self._overload_logged_at = now
        self._log_overload(count)

    @staticmethod
    def _log_overload(count: int) -> None:
        logger.warning(
            "Evaluation detectors are not keeping up with the request rate: %d span(s) "
            'were exported without evaluation, marked evaluation.skipped="overloaded". '
            "Tracing is not affected. Raise GENAI_EVALUATION_QUEUE_SIZE, lower "
            "GENAI_EVALUATION_MAX_CHARS or enable fewer detectors.",
            count,
        )


_EVALUATION_QUEUE = _EvaluationQueue()


def _after_fork_in_child() -> None:
    """Start the child of a fork with a queue and a detector pool of its own.

    The child inherits both and none of their threads, and possibly a lock that
    some other thread held at the moment of the fork. Work handed to the
    inherited pool is never picked up: an async caller with a block policy
    waited on it for ever.
    """
    global _EVALUATION_EXECUTOR, _EVALUATION_EXECUTOR_LOCK
    _EVALUATION_QUEUE._reset()
    _EVALUATION_EXECUTOR = None
    _EVALUATION_EXECUTOR_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


def drain_evaluations(timeout: Optional[float] = None) -> bool:
    """Wait until no span is waiting for its detectors or in the middle of them.

    For code that reads a span's ``evaluation.*`` attributes straight after a
    call: in the default evaluation mode the detectors run after the call has
    returned, and the span is exported when they are done. Returns False if
    ``timeout`` (seconds) ran out first.

    On a service that keeps making calls there is always another span, and this
    comes back only when ``timeout`` runs out. To flush, use
    :func:`wait_for_pending_evaluations`.
    """
    return _EVALUATION_QUEUE.drain(timeout)


def wait_for_pending_evaluations(timeout: Optional[float] = None) -> bool:
    """Wait for the spans that are with their detectors now, not for later ones.

    What a flush needs: the spans of calls that have already returned are
    exported when their detectors are done, and calls made by other threads
    while this waits are none of its business. Returns False if ``timeout``
    (seconds) ran out first.
    """
    queue = _EVALUATION_QUEUE
    return queue.drain(timeout, upto=queue.submitted)


def finish_pending_evaluations(
    timeout: Optional[float] = None, in_flight_timeout: Optional[float] = None
) -> int:
    """Wait for the detectors, then export whatever they have not got to.

    Spans no detector has started on after ``timeout`` seconds (by default the
    configured shutdown timeout) are ended without evaluation and marked
    ``evaluation.skipped="shutdown"``. A detector that is in the middle of a
    span gets ``in_flight_timeout`` more to finish it: by default as long
    again, and never less than two seconds. Call this before a tracer provider
    is shut down: a span that ends afterwards is dropped. Returns how many
    spans were ended unevaluated.
    """
    if timeout is None:
        timeout = _EVALUATION_QUEUE.shutdown_timeout
    return _EVALUATION_QUEUE.finish(timeout, in_flight_timeout)


def finish_pending_evaluations_in_signal_handler(timeout: float) -> bool:
    """:func:`finish_pending_evaluations` for a signal handler. True if it finished.

    A signal handler runs on the main thread, between two of its instructions.
    If that thread was handing a span to the queue it holds the queue's lock,
    and finishing on the same thread would wait for that lock for ever. So the
    work is done on a helper thread and waited for, for ``timeout`` seconds at
    most. A helper that cannot even get to the queue is waiting for the thread
    this handler interrupted, which goes on only when the handler returns; the
    handler then stops waiting at once and the helper finishes afterwards.

    The detectors get a little under half of ``timeout`` to work through what
    is waiting, and one that is in the middle of a text as much again. Both
    waits then fit inside ``timeout``, so that the span of a detector which
    does not come back has been ended, and can be exported, when the handler
    goes on to flush.
    """
    reached = threading.Event()
    timeout = _waitable(timeout)
    share = timeout * _SIGNAL_SHARE

    def finish() -> None:
        try:
            queue = _EVALUATION_QUEUE
            with queue._lock:
                pass
            reached.set()
            finish_pending_evaluations(share, share)
        except Exception as e:  # noqa: BLE001
            logger.debug("Finishing pending evaluations failed: %s", e)

    helper = threading.Thread(target=finish, name="genai-otel-evaluation-finish", daemon=True)
    helper.start()
    if not reached.wait(min(timeout, _QUEUE_REACHABLE_WITHIN)):
        return False
    helper.join(timeout)
    return not helper.is_alive()


def _flush_tracer_provider() -> None:
    """Have the tracer provider in use export what it holds, if it can."""
    flush = getattr(trace.get_tracer_provider(), "force_flush", None)
    if callable(flush):
        flush()


def _finish_evaluations_at_exit() -> None:
    if not _EVALUATION_QUEUE.submitted:
        # Nothing was ever handed to a worker: exit as if this hook were not here.
        return
    try:
        finish_pending_evaluations()
    except Exception as e:  # noqa: BLE001 - nothing may stop the interpreter from exiting
        logger.debug("Could not finish pending evaluations at exit: %s", e)
    try:
        # Exported now, not when the provider's own exit hook runs. The hooks in
        # between belong to the libraries the detectors are built on, and one
        # of them aborts the process if a detector that was given up on above
        # is still inside it. What has ended by now must not depend on that.
        _flush_tracer_provider()
    except Exception as e:  # noqa: BLE001
        logger.debug("Could not flush the tracer provider at exit: %s", e)


def register_evaluation_exit_hook() -> None:
    """Make pending evaluations finish before tracer providers shut down at exit.

    Exit hooks run last-registered first, and a tracer provider registers its
    shutdown when it is created. Registering again, after a provider has been
    created, moves this hook in front of that provider's.
    """
    atexit.unregister(_finish_evaluations_at_exit)
    atexit.register(_finish_evaluations_at_exit)


_CUSTOM_COST_CALCULATORS: Dict[str, "CostCalculator"] = {}


def _get_custom_cost_calculator(custom_pricing_json: str) -> "CostCalculator":
    """One calculator per distinct custom-pricing document, shared by every instrumentor."""
    calc = _CUSTOM_COST_CALCULATORS.get(custom_pricing_json)
    if calc is None:
        with _DEFAULT_COST_CALCULATOR_LOCK:
            calc = _CUSTOM_COST_CALCULATORS.get(custom_pricing_json)
            if calc is None:
                calc = CostCalculator(custom_pricing_json=custom_pricing_json)
                _CUSTOM_COST_CALCULATORS[custom_pricing_json] = calc
    return calc


# base_url domains claimed by dedicated aggregator instrumentors (OpenRouter,
# CometAPI). Generic SDK instrumentors (OpenAI, Anthropic) consult this before
# instrumenting a client: a client whose base_url is claimed gets its span AND
# metrics from the dedicated instrumentor only. Without this, an aggregator
# call was wrapped twice (instance wrapper over class wrapper), producing a
# duplicate nested span and double-counted token/cost metrics.
_BASE_URL_CLAIMS: Dict[str, str] = {}
_BASE_URL_CLAIMS_LOCK = threading.Lock()


def register_base_url_claim(domain_substring: str, instrumentor_name: str) -> None:
    """Claim clients whose base_url contains ``domain_substring`` for a dedicated instrumentor.

    Called by aggregator instrumentors when they activate. Registration is
    idempotent; claims apply only for the lifetime of the process.
    """
    with _BASE_URL_CLAIMS_LOCK:
        _BASE_URL_CLAIMS[domain_substring.lower()] = instrumentor_name


# Attribute paths walked to find the SDK client's base URL, in order. Wrapped
# methods are bound to a sub-resource (e.g. `openai.resources.Completions`),
# not the client, so the client is reached through `_client` -- twice for SDKs
# that wrap a transport client of their own.
_BASE_URL_PATHS = (
    ("base_url",),
    ("_client", "base_url"),
    ("_client", "_client", "base_url"),
    ("_client", "_base_url"),
)


def _request_parameter_attributes(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Map common request kwargs onto the conventions' request attributes.

    Keyed on the OpenAI-compatible parameter names, which the large majority of
    SDKs and every OpenAI-compatible gateway reuse. Only parameters the caller
    actually passed are recorded: an absent attribute means "not set by the
    caller", whereas materialising provider defaults would report a temperature
    or top_k the application never chose.
    """
    attrs: Dict[str, Any] = {}
    if not isinstance(kwargs, dict):
        return attrs

    seed = kwargs.get("seed")
    if isinstance(seed, int) and not isinstance(seed, bool):
        attrs[SC.GEN_AI_REQUEST_SEED] = seed

    if "stream" in kwargs:
        attrs[SC.GEN_AI_REQUEST_STREAM] = bool(kwargs.get("stream"))

    top_k = kwargs.get("top_k")
    if isinstance(top_k, (int, float)) and not isinstance(top_k, bool):
        attrs[SC.GEN_AI_REQUEST_TOP_K] = top_k

    # `n` is the OpenAI spelling; `candidate_count` is Google's.
    for key in ("n", "candidate_count"):
        count = kwargs.get(key)
        if isinstance(count, int) and not isinstance(count, bool) and count > 0:
            attrs[SC.GEN_AI_REQUEST_CHOICE_COUNT] = count
            break

    # `gen_ai.output.type` is about the shape the caller asked for, so it is
    # derived from the request rather than from what came back.
    response_format = kwargs.get("response_format")
    if response_format is not None:
        fmt = response_format
        if isinstance(fmt, dict):
            fmt = fmt.get("type")
        fmt = str(getattr(fmt, "value", fmt) or "").lower()
        if fmt.startswith("json"):
            attrs[SC.GEN_AI_OUTPUT_TYPE] = "json"
        elif fmt == "text":
            attrs[SC.GEN_AI_OUTPUT_TYPE] = "text"

    return attrs


def _server_attributes(instance: Any) -> Dict[str, Any]:
    """Derive `server.address` / `server.port` from an SDK client's base URL.

    Returns an empty dict when no base URL can be observed. That is the honest
    answer for SDKs that do not expose one: an absent attribute reads as
    "endpoint unknown", whereas defaulting to the provider's public host would
    silently misattribute self-hosted, proxied and gateway traffic to the vendor.
    """
    for path in _BASE_URL_PATHS:
        target = instance
        for step in path:
            target = getattr(target, step, None)
            if target is None:
                break
        if target is None:
            continue
        try:
            parsed = urlparse(str(target))
        except Exception:  # pragma: no cover - urlparse is near-total
            continue
        if not parsed.hostname:
            continue
        attrs: Dict[str, Any] = {SC.SERVER_ADDRESS: parsed.hostname}
        port = None
        try:
            port = parsed.port
        except ValueError:
            # Malformed port in the URL; record the host without inventing one.
            port = None
        if port is None:
            port = {"https": 443, "http": 80}.get(parsed.scheme)
        if port is not None:
            attrs[SC.SERVER_PORT] = int(port)
        return attrs
    return {}


def find_base_url_claim(base_url: Any) -> Optional[str]:
    """Return the name of the dedicated instrumentor claiming ``base_url``, or None."""
    if not base_url:
        return None
    url = str(base_url).lower()
    for domain, name in _BASE_URL_CLAIMS.items():
        if domain in url:
            return name
    return None


class _StreamTiming:
    """Latency bookkeeping for one streamed response.

    Shared by the sync and async stream wrappers so both report identical
    attributes. ``ttft`` stays None until the first chunk actually arrives,
    which is what lets the finalizer tell "not streamed / never yielded" apart
    from "first token was fast".
    """

    __slots__ = (
        "start_time",
        "ttft",
        "last_chunk_time",
        "chunk_count",
        "last_chunk",
        "usage",
        "emit_measurements",
        "started",
        "tbt_sum",
        "tbt_count",
        "tbt_per_chunk",
        "usage_in_events",
    )

    def __init__(self, start_time: float, emit_measurements: bool = True):
        self.start_time = start_time
        self.ttft: Optional[float] = None
        self.last_chunk_time = start_time
        self.chunk_count = 0
        self.last_chunk: Any = None
        # Usage assembled across chunks by a provider that splits it over
        # several events. Stays None for providers that report it on the final
        # chunk, and the finalizer then reads that chunk exactly as before.
        self.usage: Optional[Dict[str, Any]] = None
        # False on an outer span whose inner provider span already recorded
        # this request. The outer span still spans the call - it is the parent -
        # but publishing the same numbers twice would double-count one request.
        self.emit_measurements = emit_measurements
        # True once the measuring generator has begun to run. A stream closed
        # before that has no pending yield for its span to be finalized at.
        self.started = False
        # Time between chunks, added up as the stream is read and recorded as
        # one sample, the mean, when it ends. A metric sample is far more work
        # than an addition, and it used to be made for every chunk.
        self.tbt_sum = 0.0
        self.tbt_count = 0
        # Whether a sample is recorded for every chunk after all (the "full"
        # metrics profile). None until the second chunk asks.
        self.tbt_per_chunk: Optional[bool] = None
        # Whether the provider reports usage on events of their own, which its
        # accumulator then has to watch for. None until it has looked at the
        # first chunk: a stream does not change kind part-way through.
        self.usage_in_events: Optional[bool] = None


def _is_sampled_out(span) -> bool:
    """True for a span the sampler dropped. Anything that cannot say is treated as live."""
    is_recording = getattr(span, "is_recording", None)
    return callable(is_recording) and is_recording() is False


def _span_name(span) -> str:
    """The name of ``span`` for a metric label. A span the sampler dropped has none."""
    return getattr(span, "name", None) or "unknown"


class _MeasuredStream(wrapt.ObjectProxy):
    """The SDK's stream, with its iteration measured and nothing else changed.

    A transparent proxy rather than a replacement iterator. Callers use a stream
    as more than an iterable: langchain-openai reads one inside
    ``with stream as response:``, and others reach for ``close()`` and
    ``response``. A bare generator in its place breaks the caller's application,
    not just a span.
    """

    def __init__(self, wrapped, measured, timing, finish_unread):
        super().__init__(wrapped)
        # ObjectProxy reserves the `_self_` prefix for attributes belonging to
        # the proxy rather than to the wrapped object.
        self._self_measured = measured
        self._self_timing = timing
        self._self_finish_unread = finish_unread

    def __iter__(self):
        # The generator itself, so a `for` loop pays nothing per chunk for
        # going through the proxy.
        return self._self_measured

    def __next__(self):
        return next(self._self_measured)

    def __enter__(self):
        enter = getattr(self.__wrapped__, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._self_end_measurement()
        leave = getattr(self.__wrapped__, "__exit__", None)
        if leave is not None:
            return leave(exc_type, exc_value, traceback)
        return None

    def close(self):
        self._self_end_measurement()
        close = getattr(self.__wrapped__, "close", None)
        if close is not None:
            return close()
        return None

    def _self_end_measurement(self):
        """End the span of a stream the caller is done with, read or not.

        Never raises: the caller is closing a connection, and bookkeeping that
        fails here (or a generator another thread is still inside) must not stop
        the SDK's own close from running.
        """
        timing = self._self_timing
        started, timing.started = timing.started, True
        try:
            # Raises GeneratorExit at the pending yield, which finalizes the span.
            # A no-op on a stream that already ran to its end or failed.
            self._self_measured.close()
            if not started:
                # A generator that never ran has no pending yield to finalize at.
                self._self_finish_unread()
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not end stream measurement: %s", e)


class _MeasuredAsyncStream(wrapt.ObjectProxy):
    """Async counterpart of :class:`_MeasuredStream`.

    Pydantic AI reads an OpenAI stream inside ``async with response:``.
    """

    def __init__(self, wrapped, measured, timing, finish_unread):
        super().__init__(wrapped)
        self._self_measured = measured
        self._self_timing = timing
        self._self_finish_unread = finish_unread

    def __aiter__(self):
        return self._self_measured

    def __anext__(self):
        return self._self_measured.__anext__()

    async def __aenter__(self):
        enter = getattr(self.__wrapped__, "__aenter__", None)
        if enter is not None:
            await enter()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self._self_end_measurement()
        leave = getattr(self.__wrapped__, "__aexit__", None)
        if leave is not None:
            return await leave(exc_type, exc_value, traceback)
        return None

    async def close(self):
        await self._self_end_measurement()
        await self._self_close_wrapped("close", "aclose")

    async def aclose(self):
        await self._self_end_measurement()
        await self._self_close_wrapped("aclose", "close")

    async def _self_close_wrapped(self, *names):
        """Close the SDK stream by whichever of ``names`` it has, first one first.

        SDK streams spell it ``close()`` (openai) and async generators
        ``aclose()``; a caller reaching for either name must close the stream.
        """
        for name in names:
            method = getattr(self.__wrapped__, name, None)
            if method is not None:
                closing = method()
                if inspect.isawaitable(closing):
                    await closing
                return

    async def _self_end_measurement(self):
        """End the span of a stream the caller is done with, read or not. Never raises."""
        timing = self._self_timing
        started, timing.started = timing.started, True
        try:
            await self._self_measured.aclose()
            if not started:
                self._self_finish_unread()
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not end stream measurement: %s", e)


# Set for the duration of a call by a wrapper that sits *outside* a provider SDK
# we already instrument -- litellm routes to the OpenAI SDK, for instance. The
# inner wrapper appends to it when it takes over measurement, so the outer one
# can tell whether the request is already accounted for. A mutable holder rather
# than a plain flag so the inner scope's write is visible to the outer one
# without needing the context to propagate back out.
_INNER_MEASUREMENT: contextvars.ContextVar = contextvars.ContextVar(
    "genai_otel_inner_measurement", default=None
)


def inner_measurement_scope():
    """Open a scope recording whether an inner instrumented span measured.

    Reuses an enclosing scope when one is already open. litellm.acompletion
    calls litellm.completion internally (run_in_executor over a copied
    context), so two of our wrappers can be live for a single request.
    Shadowing the holder there would hide the provider span's note from the
    outer wrapper and let both publish the same tokens and cost.
    """
    existing = _INNER_MEASUREMENT.get()
    if existing is not None:
        return existing, None
    holder: List[bool] = []
    token = _INNER_MEASUREMENT.set(holder)
    return holder, token


def close_inner_measurement_scope(token) -> None:
    if token is None:  # reused an enclosing scope; its owner resets it
        return
    try:
        _INNER_MEASUREMENT.reset(token)
    except (ValueError, LookupError):  # different context; nothing to reset
        pass


def _note_inner_measurement() -> None:
    holder = _INNER_MEASUREMENT.get()
    if holder is not None:
        holder.append(True)


# ---------------------------------------------------------------------------
# One model call, one usage record.
#
# A framework span (a LangChain chat model, a graph run) and the provider span
# beneath it both see the same usage: the framework hands back an object that
# carries the provider's token counts. Both used to write the tokens and the
# cost and to record the token and cost metrics, so a single request was
# reported twice by anything that sums spans or reads the counters.
#
# Each span the base wrapper opens owns a holder that travels in the OTel
# context alongside the span, so it is parented exactly as the span is and
# survives every async path the span does. Recording usage on a span marks the
# holders of the spans enclosing it. An enclosing span that finds its holder
# marked leaves the usage to the span that already recorded it.
#
# The holder is also pinned to the span object, because a stream is finalised
# after the wrapper has detached its context.
# ---------------------------------------------------------------------------
_USAGE_HOLDER_KEY = otel_context.create_key("genai_otel_usage_holder")
_USAGE_HOLDER_ATTR = "_genai_otel_usage_holder"
USAGE_RECORDED_BY = "gen_ai.usage.recorded_by"


class _UsageHolder:
    __slots__ = ("parent", "descendant_recorded")

    def __init__(self, parent: Optional["_UsageHolder"]):
        self.parent = parent
        self.descendant_recorded = False


# Holders for spans this module did not open itself. Several framework
# instrumentors start their span directly (``tracer.start_as_current_span``) and
# only then call ``_record_result_metrics``; a span in between may come from the
# application or from other instrumentation. ``UsageHolderSpanProcessor`` gives
# every span a holder, linked to its parent's by span id, so the chain is
# complete whoever opened the spans. Without the processor (a tracer provider
# this library did not set up) only spans opened by the base wrapper have one.
_USAGE_HOLDERS: Dict[int, "_UsageHolder"] = {}
_USAGE_HOLDERS_LOCK = threading.Lock()
_USAGE_HOLDERS_MAX = 100_000  # spans that never end must not grow this forever


def _span_id(span) -> Optional[int]:
    try:
        span_id = span.get_span_context().span_id
        return span_id if isinstance(span_id, int) and span_id else None
    except Exception:  # noqa: BLE001
        return None


class UsageHolderSpanProcessor(SpanProcessor):
    """Give every span a usage holder linked to its parent's."""

    def on_start(self, span, parent_context=None) -> None:
        try:
            span_id = _span_id(span)
            if span_id is None:
                return
            parent_id = _span_id(trace.get_current_span(parent_context))
            with _USAGE_HOLDERS_LOCK:
                if len(_USAGE_HOLDERS) >= _USAGE_HOLDERS_MAX:
                    _USAGE_HOLDERS.clear()
                _USAGE_HOLDERS[span_id] = _UsageHolder(
                    _USAGE_HOLDERS.get(parent_id) if parent_id is not None else None
                )
        except Exception:  # noqa: BLE001 - bookkeeping must never break a span
            pass

    def on_end(self, span) -> None:
        try:
            span_id = _span_id(span)
            if span_id is not None:
                with _USAGE_HOLDERS_LOCK:
                    _USAGE_HOLDERS.pop(span_id, None)
        except Exception:  # noqa: BLE001
            pass

    def shutdown(self) -> None:
        with _USAGE_HOLDERS_LOCK:
            _USAGE_HOLDERS.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _usage_holder_of(span) -> Optional["_UsageHolder"]:
    holder = getattr(span, _USAGE_HOLDER_ATTR, None)
    if isinstance(holder, _UsageHolder):  # isinstance: a mock span answers any getattr
        return holder
    span_id = _span_id(span)
    if span_id is None:
        return None
    holder = _USAGE_HOLDERS.get(span_id)
    return holder if isinstance(holder, _UsageHolder) else None


def _context_with_usage_holder(span, ctx):
    """Return ``ctx`` carrying the usage holder for ``span``."""
    try:
        holder = _usage_holder_of(span)  # the processor's, when it is installed
        if holder is None:
            parent = otel_context.get_value(_USAGE_HOLDER_KEY)
            holder = _UsageHolder(parent if isinstance(parent, _UsageHolder) else None)
        try:
            setattr(span, _USAGE_HOLDER_ATTR, holder)
        except Exception:  # noqa: BLE001 - a span type that refuses attributes
            pass
        return otel_context.set_value(_USAGE_HOLDER_KEY, holder, ctx)
    except Exception:  # noqa: BLE001 - never let bookkeeping break a call
        return ctx


def _descendant_recorded_usage(span) -> bool:
    holder = _usage_holder_of(span)
    return bool(holder is not None and holder.descendant_recorded)


def _mark_usage_recorded(span) -> None:
    """Tell every enclosing span that this request is accounted for."""
    holder = _usage_holder_of(span)
    seen = 0
    ancestor = holder.parent if holder is not None else None
    while ancestor is not None and seen < 64:  # bounded: a cycle must not hang a call
        ancestor.descendant_recorded = True
        ancestor = ancestor.parent
        seen += 1


def _leave_usage_to_descendant(span) -> None:
    try:
        span.set_attribute(USAGE_RECORDED_BY, "descendant")
    except Exception:  # noqa: BLE001
        pass


class BaseInstrumentor(ABC):  # pylint: disable=R0902
    """Abstract base class for all LLM library instrumentors.

    Provides common functionality for setting up OpenTelemetry spans, metrics,
    and handling common instrumentation patterns.
    """

    # Multimodal: subclasses that support content-part detection set this to one of
    # "openai" | "anthropic" | "google" | "groq". Default None disables media capture.
    MEDIA_PROVIDER: Optional[str] = None

    # Class-level shared metrics (created once, shared by all instances)
    _shared_request_counter = None
    _shared_token_counter = None
    _shared_latency_histogram = None
    _shared_cost_counter = None
    _shared_error_counter = None
    # Granular cost counters (Phase 3.2)
    _shared_prompt_cost_counter = None
    _shared_completion_cost_counter = None
    _shared_reasoning_cost_counter = None
    _shared_cache_read_cost_counter = None
    _shared_cache_write_cost_counter = None
    # Agent budget governance (semantic-conventions-genai#425)
    _shared_token_budget_utilization_histogram = None
    # Streaming metrics (Phase 3.4)
    _shared_ttft_histogram = None
    _shared_tbt_histogram = None
    # Same measurements under the upstream semconv names (issue #21)
    _shared_time_to_first_token_histogram = None
    _shared_time_per_output_token_histogram = None
    # Token distribution histograms
    _shared_prompt_tokens_histogram = None
    _shared_completion_tokens_histogram = None
    # Finish reason tracking counters
    _shared_request_finish_counter = None
    _shared_request_success_counter = None
    _shared_request_failure_counter = None

    # Evaluation detectors (set by EvaluationSpanProcessor)
    _pii_detector = None
    _toxicity_detector = None
    _bias_detector = None
    _prompt_injection_detector = None
    _restricted_topics_detector = None
    _hallucination_detector = None
    _evaluation_lock = threading.Lock()

    # Pre-call BLOCK policy. Set by EvaluationSpanProcessor to the set of policy
    # names configured with block_on_detection / PII block mode. When empty
    # (_blocking_active False) the hot path skips pre-call evaluation entirely,
    # so non-blocking deployments pay zero extra cost.
    _blocking_active = False
    _blocking_policies = frozenset()
    # Detectors that only match patterns. A block decision of theirs reads the
    # whole prompt whatever the limit on analysed text: they are cheap, and a
    # limit would let whoever writes the prompt hide text from the policy by
    # padding it.
    _WHOLE_PROMPT_BLOCK_POLICIES = frozenset({"prompt_injection", "restricted_topics", "bias"})

    def __init__(self):
        """Initializes the instrumentor with OpenTelemetry tracers, meters, and common metrics."""
        self.tracer = trace.get_tracer(__name__)
        self.meter = metrics.get_meter(__name__)
        self.config: Optional[OTelConfig] = None
        # None = resolve from self.config on every use (see the cost_calculator property).
        self._cost_calculator_override: Optional[CostCalculator] = None
        self._instrumented = False

        # Use shared metrics to avoid duplicate warnings
        self._ensure_shared_metrics_created()

        # Reference the shared metrics
        self.request_counter = self._shared_request_counter
        self.token_counter = self._shared_token_counter
        self.latency_histogram = self._shared_latency_histogram
        self.cost_counter = self._shared_cost_counter
        self.error_counter = self._shared_error_counter
        # Granular cost counters (Phase 3.2)
        self.prompt_cost_counter = self._shared_prompt_cost_counter
        self.completion_cost_counter = self._shared_completion_cost_counter
        self.reasoning_cost_counter = self._shared_reasoning_cost_counter
        self.cache_read_cost_counter = self._shared_cache_read_cost_counter
        self.cache_write_cost_counter = self._shared_cache_write_cost_counter
        # Agent budget governance
        self.token_budget_utilization_histogram = self._shared_token_budget_utilization_histogram
        # Streaming metrics
        self.ttft_histogram = self._shared_ttft_histogram
        self.tbt_histogram = self._shared_tbt_histogram
        self.time_to_first_token_histogram = self._shared_time_to_first_token_histogram
        self.time_per_output_token_histogram = self._shared_time_per_output_token_histogram
        # Token distribution histograms
        self.prompt_tokens_histogram = self._shared_prompt_tokens_histogram
        self.completion_tokens_histogram = self._shared_completion_tokens_histogram
        # Finish reason tracking counters
        self.request_finish_counter = self._shared_request_finish_counter
        self.request_success_counter = self._shared_request_success_counter
        self.request_failure_counter = self._shared_request_failure_counter

    @classmethod
    def _ensure_shared_metrics_created(cls):
        """Ensure shared metrics are created only once across all instrumentor instances."""
        # The instruments are stored on BaseInstrumentor, never on `cls`. Assigning to
        # `cls` put them on whichever SUBCLASS happened to be constructed first; every
        # other instrumentor class then read the base class's None and recorded no
        # token, cost or latency metrics at all.
        global _SHARED_METRICS_CREATED

        with _SHARED_METRICS_LOCK:
            if _SHARED_METRICS_CREATED:
                return

            try:
                meter = metrics.get_meter(__name__)

                # Create shared metrics once using semantic conventions
                BaseInstrumentor._shared_request_counter = meter.create_counter(
                    SC.GEN_AI_REQUESTS, description="Number of GenAI requests"
                )
                BaseInstrumentor._shared_token_counter = meter.create_counter(
                    SC.GEN_AI_CLIENT_TOKEN_USAGE, description="Token usage for GenAI operations"
                )
                # Note: Histogram buckets should be configured via Views in MeterProvider
                # The advisory parameter is provided as a hint but Views take precedence
                BaseInstrumentor._shared_latency_histogram = meter.create_histogram(
                    SC.GEN_AI_CLIENT_OPERATION_DURATION,
                    description="GenAI client operation duration",
                    unit="s",
                )
                BaseInstrumentor._shared_cost_counter = meter.create_counter(
                    SC.GEN_AI_USAGE_COST, description="Cost of GenAI operations", unit="USD"
                )
                # Granular cost counters (Phase 3.2)
                BaseInstrumentor._shared_prompt_cost_counter = meter.create_counter(
                    "gen_ai.usage.cost.prompt", description="Prompt tokens cost", unit="USD"
                )
                BaseInstrumentor._shared_completion_cost_counter = meter.create_counter(
                    "gen_ai.usage.cost.completion", description="Completion tokens cost", unit="USD"
                )
                BaseInstrumentor._shared_reasoning_cost_counter = meter.create_counter(
                    "gen_ai.usage.cost.reasoning",
                    description="Reasoning tokens cost (o1 models)",
                    unit="USD",
                )
                BaseInstrumentor._shared_cache_read_cost_counter = meter.create_counter(
                    "gen_ai.usage.cost.cache_read",
                    description="Cache read cost (Anthropic)",
                    unit="USD",
                )
                BaseInstrumentor._shared_cache_write_cost_counter = meter.create_counter(
                    "gen_ai.usage.cost.cache_write",
                    description="Cache write cost (Anthropic)",
                    unit="USD",
                )
                BaseInstrumentor._shared_error_counter = meter.create_counter(
                    "gen_ai.client.errors", description="Number of GenAI client errors"
                )
                # Streaming metrics (Phase 3.4)
                # Note: Buckets should be configured via Views in MeterProvider
                BaseInstrumentor._shared_token_budget_utilization_histogram = (
                    meter.create_histogram(
                        SC.GEN_AI_INVOKE_AGENT_TOKEN_BUDGET_UTILIZATION,
                        description=(
                            "Fraction of an agent invocation's configured token budget "
                            "that it consumed"
                        ),
                        unit="1",
                    )
                )
                BaseInstrumentor._shared_ttft_histogram = meter.create_histogram(
                    SC.GEN_AI_SERVER_TTFT,
                    description="Time to first token in seconds",
                    unit="s",
                )
                BaseInstrumentor._shared_tbt_histogram = meter.create_histogram(
                    SC.GEN_AI_SERVER_TBT,
                    description="Time between tokens in seconds",
                    unit="s",
                )
                # Upstream semconv spellings of the same two ideas. Recorded
                # alongside the abbreviations above so consumers reading either
                # name get data (issue #21).
                BaseInstrumentor._shared_time_to_first_token_histogram = meter.create_histogram(
                    SC.GEN_AI_SERVER_TIME_TO_FIRST_TOKEN,
                    description="Time to first token in seconds",
                    unit="s",
                )
                BaseInstrumentor._shared_time_per_output_token_histogram = meter.create_histogram(
                    SC.GEN_AI_SERVER_TIME_PER_OUTPUT_TOKEN,
                    description="Time per output token after the first, in seconds",
                    unit="s",
                )
                # Token distribution histograms
                BaseInstrumentor._shared_prompt_tokens_histogram = meter.create_histogram(
                    "gen_ai.client.token.usage.prompt",
                    description="Distribution of prompt tokens per request",
                    unit="tokens",
                )
                BaseInstrumentor._shared_completion_tokens_histogram = meter.create_histogram(
                    "gen_ai.client.token.usage.completion",
                    description="Distribution of completion tokens per request",
                    unit="tokens",
                )
                # Finish reason tracking counters
                BaseInstrumentor._shared_request_finish_counter = meter.create_counter(
                    "gen_ai.server.request.finish",
                    description="Number of finished requests by finish reason",
                )
                BaseInstrumentor._shared_request_success_counter = meter.create_counter(
                    "gen_ai.server.request.success",
                    description="Number of successfully completed requests",
                )
                BaseInstrumentor._shared_request_failure_counter = meter.create_counter(
                    "gen_ai.server.request.failure",
                    description="Number of failed requests",
                )

                _SHARED_METRICS_CREATED = True
                logger.debug("Shared metrics created successfully")

            except Exception as e:
                logger.error("Failed to create shared metrics: %s", e, exc_info=True)
                # Create dummy metrics that do nothing to avoid crashes
                BaseInstrumentor._shared_request_counter = None
                BaseInstrumentor._shared_token_counter = None
                BaseInstrumentor._shared_latency_histogram = None
                BaseInstrumentor._shared_cost_counter = None
                BaseInstrumentor._shared_prompt_cost_counter = None
                BaseInstrumentor._shared_completion_cost_counter = None
                BaseInstrumentor._shared_reasoning_cost_counter = None
                BaseInstrumentor._shared_cache_read_cost_counter = None
                BaseInstrumentor._shared_cache_write_cost_counter = None
                BaseInstrumentor._shared_error_counter = None
                BaseInstrumentor._shared_token_budget_utilization_histogram = None
                BaseInstrumentor._shared_ttft_histogram = None
                BaseInstrumentor._shared_tbt_histogram = None
                BaseInstrumentor._shared_time_to_first_token_histogram = None
                BaseInstrumentor._shared_time_per_output_token_histogram = None
                BaseInstrumentor._shared_prompt_tokens_histogram = None
                BaseInstrumentor._shared_completion_tokens_histogram = None
                BaseInstrumentor._shared_request_finish_counter = None
                BaseInstrumentor._shared_request_success_counter = None
                BaseInstrumentor._shared_request_failure_counter = None

    @property
    def cost_calculator(self) -> "CostCalculator":
        """The calculator for this instrumentor's CURRENT configuration.

        Custom pricing used to be applied only by ``_setup_config()``, which 38 of the 42
        instrumentors never call (their ``instrument()`` assigns ``self.config`` directly),
        so ``GENAI_CUSTOM_PRICING_JSON`` was silently ignored for OpenAI, Anthropic and most
        others. Resolving here makes every instrumentor honour it. A calculator assigned
        explicitly (tests, ``_setup_config``) still wins.
        """
        if self._cost_calculator_override is not None:
            return self._cost_calculator_override
        custom = getattr(getattr(self, "config", None), "custom_pricing_json", None)
        if custom:
            return _get_custom_cost_calculator(custom)
        return _get_default_cost_calculator()

    @cost_calculator.setter
    def cost_calculator(self, value: "CostCalculator") -> None:
        self._cost_calculator_override = value

    def _setup_config(self, config: OTelConfig):
        """Set up configuration and reinitialize cost calculator with custom pricing if provided.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        self.config = config
        # Reinitialize cost calculator with custom pricing if provided
        if config.custom_pricing_json:
            self.cost_calculator = CostCalculator(custom_pricing_json=config.custom_pricing_json)
            logger.info("Cost calculator reinitialized with custom pricing")

    @abstractmethod
    def instrument(self, config: OTelConfig):
        """Abstract method to implement library-specific instrumentation.

        Implementers should call self._setup_config(config) at the beginning of this method
        to ensure custom pricing is loaded.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """

    def _build_first_message(self, messages) -> Optional[str]:
        """Build a parseable first_message attribute from a messages list.

        Respects ``enable_content_capture`` and ``content_max_length`` config.
        Truncates the *content* value before serialising to JSON.

        When config is not set (e.g. during unit tests), defaults to capturing
        with a 200-char content limit.

        Returns:
            A JSON string like ``'{"role": "user", "content": "..."}'`` or
            ``None`` if content capture is disabled or messages is empty.
        """
        if not messages:
            return None

        config = getattr(self, "config", None)
        # When config is set, respect enable_content_capture flag.
        # When config is not set, default to capturing (backward-compat).
        if config and not config.enable_content_capture:
            return None

        msg = messages[0]
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))
        else:
            role = "user"
            content = str(msg)

        max_len = config.content_max_length if config else 200
        if max_len > 0:
            content = content[:max_len]

        return json.dumps({"role": role, "content": content})

    def _emit_media_attributes(self, span, request_kwargs: dict, result: Any) -> None:
        """Emit multimodal content-part attributes on the span.

        Walks the request messages (and, when available, the result) using the
        media detector for the configured ``MEDIA_PROVIDER``, then offloads any
        binary parts via the configured store. Each part contributes a small
        set of ``gen_ai.prompt.{n}.content.{m}.*`` attributes.

        Subclasses MAY override ``_extract_completion_parts`` to surface
        provider-specific completion-side multimodal output (e.g. image
        generation responses).
        """
        # Lazy import to avoid pulling media deps when feature is unused.
        from ..media import detect_parts, get_store, offload_part  # noqa: WPS433
        from ..media.canonical import build_canonical_messages  # noqa: WPS433

        provider = self.MEDIA_PROVIDER
        if provider is None:
            return

        store = None
        try:
            store = get_store(self.config)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to initialize media store: %s", e)
            store = None

        # Trace ID for keying uploads
        trace_id_hex = ""
        try:
            ctx = span.get_span_context()
            if ctx and ctx.trace_id:
                trace_id_hex = f"{ctx.trace_id:032x}"
        except Exception:  # noqa: BLE001
            pass

        # Prompt side: walk messages
        messages = (request_kwargs or {}).get("messages") or []
        # Gemini uses "contents" rather than "messages"
        if not messages and provider == "google":
            messages = (request_kwargs or {}).get("contents") or []

        prompt_messages_for_canonical = []  # [(role, [ContentPart])] for canonical emit
        for n, msg in enumerate(messages or []):
            role = "user"
            content: Any = msg
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content")
                if provider == "google" and content is None:
                    content = msg.get("parts")

            # Bound untrusted inline-media decoding to the configured per-request
            # cap (defends against decompression/size bombs on the hot path).
            _media_cap = getattr(self.config, "media_max_bytes", None) if self.config else None
            parts = (
                detect_parts(provider, content, max_decoded_bytes=_media_cap)
                if _media_cap
                else detect_parts(provider, content)
            )
            if not parts:
                continue
            span.set_attribute(SC.GEN_AI_PROMPT_ROLE.format(n=n), role)
            for m, part in enumerate(parts):
                try:
                    offload_part(part, config=self.config, store=store, trace_id=trace_id_hex)
                except Exception as e:  # noqa: BLE001
                    logger.warning("offload_part failed: %s", e)
                self._set_part_attrs(span, "prompt", n, m, part)
            prompt_messages_for_canonical.append((role, parts))

        # Completion side: best-effort. Subclasses can override.
        try:
            completion_parts = self._extract_completion_parts(result)
        except Exception:  # noqa: BLE001
            completion_parts = None
        completion_messages_for_canonical = []
        if completion_parts:
            span.set_attribute(SC.GEN_AI_COMPLETION_ROLE.format(n=0), "assistant")
            for m, part in enumerate(completion_parts):
                try:
                    offload_part(part, config=self.config, store=store, trace_id=trace_id_hex)
                except Exception as e:  # noqa: BLE001
                    logger.warning("offload_part (completion) failed: %s", e)
                self._set_part_attrs(span, "completion", 0, m, part)
            completion_messages_for_canonical.append(("assistant", completion_parts))

        # Dual-emission: also write OTel-canonical gen_ai.input.messages /
        # gen_ai.output.messages JSON when the user has opted into the new
        # GenAI semconv stability tier ("gen_ai" or "gen_ai/dup").
        # Canonical schema: gen-ai-input-messages.json / gen-ai-output-messages.json
        # in open-telemetry/semantic-conventions.
        # Gated on the GenAI tier opt-in, NOT on genai_semconv_modes: this payload
        # carries message CONTENT, so an explicit opt-out must be honoured rather
        # than defaulted to the safe-for-naming value. See genai_tier_opted_in.
        emit_canonical = genai_tier_opted_in(
            self.config.semconv_stability_opt_in if self.config else None
        )
        if emit_canonical and (prompt_messages_for_canonical or completion_messages_for_canonical):
            try:
                import json as _json  # noqa: WPS433

                # Cap canonical text content to the configured length (0/None =
                # unlimited for full audit). offload_part already caps part.text;
                # this is belt-and-suspenders for the canonical emission.
                _text_cap = getattr(self.config, "content_max_length", 0) if self.config else 0
                _text_cap = _text_cap or None

                if prompt_messages_for_canonical:
                    canonical_input = build_canonical_messages(
                        prompt_messages_for_canonical, max_length=_text_cap
                    )
                    span.set_attribute("gen_ai.input.messages", _json.dumps(canonical_input))
                if completion_messages_for_canonical:
                    canonical_output = build_canonical_messages(
                        completion_messages_for_canonical, max_length=_text_cap
                    )
                    span.set_attribute("gen_ai.output.messages", _json.dumps(canonical_output))
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to emit canonical gen_ai.input/output.messages: %s", e)

    def _set_part_attrs(self, span, side: str, n: int, m: int, part) -> None:
        """Set the small fixed attribute set for a single content part."""
        if side == "prompt":
            tname = SC.GEN_AI_PROMPT_CONTENT_TYPE.format(n=n, m=m)
            ttext = SC.GEN_AI_PROMPT_CONTENT_TEXT.format(n=n, m=m)
            turi = SC.GEN_AI_PROMPT_CONTENT_MEDIA_URI.format(n=n, m=m)
            tmime = SC.GEN_AI_PROMPT_CONTENT_MEDIA_MIME.format(n=n, m=m)
            tsize = SC.GEN_AI_PROMPT_CONTENT_MEDIA_BYTES.format(n=n, m=m)
            tsource = SC.GEN_AI_PROMPT_CONTENT_MEDIA_SOURCE.format(n=n, m=m)
        else:
            tname = SC.GEN_AI_COMPLETION_CONTENT_TYPE.format(n=n, m=m)
            ttext = SC.GEN_AI_COMPLETION_CONTENT_TEXT.format(n=n, m=m)
            turi = SC.GEN_AI_COMPLETION_CONTENT_MEDIA_URI.format(n=n, m=m)
            tmime = SC.GEN_AI_COMPLETION_CONTENT_MEDIA_MIME.format(n=n, m=m)
            tsize = SC.GEN_AI_COMPLETION_CONTENT_MEDIA_BYTES.format(n=n, m=m)
            tsource = SC.GEN_AI_COMPLETION_CONTENT_MEDIA_SOURCE.format(n=n, m=m)

        span.set_attribute(tname, part.type)
        if part.type == "text" and part.text is not None:
            # Apply the same content cap as text capture
            max_len = (
                self.config.content_max_length
                if self.config and self.config.content_max_length
                else 0
            )
            text = part.text[:max_len] if max_len > 0 else part.text
            span.set_attribute(ttext, text)
        else:
            if part.media_uri:
                span.set_attribute(turi, part.media_uri)
            if part.media_mime_type:
                span.set_attribute(tmime, part.media_mime_type)
            if part.media_byte_size is not None:
                span.set_attribute(tsize, int(part.media_byte_size))
            if part.media_source:
                span.set_attribute(tsource, part.media_source)
            stripped = part.extra.get("stripped_reason") if part.extra else None
            if stripped:
                span.set_attribute(SC.GEN_AI_MEDIA_STRIPPED_REASON, stripped)

    def _extract_completion_parts(self, result):  # noqa: D401
        """Override in subclasses to return List[ContentPart] from a result."""
        return None

    # ------------------------------------------------------------------
    # Speech media: payloads that are audio files and bytes, not chat messages.
    # ------------------------------------------------------------------
    def _speech_media_enabled(self) -> bool:
        """Audio is captured only when media capture is on (it is off by default)."""
        return bool(self.config) and getattr(self.config, "media_capture_mode", "off") != "off"

    def _speech_text_enabled(self) -> bool:
        """Transcripts and spoken text are content: captured only with content capture on."""
        return bool(self.config) and bool(getattr(self.config, "enable_content_capture", False))

    def _media_max_bytes(self) -> int:
        try:
            return int(
                getattr(self.config, "media_max_bytes", 10 * 1024 * 1024) or 10 * 1024 * 1024
            )
        except (TypeError, ValueError):
            return 10 * 1024 * 1024

    def _emit_speech_parts(self, span, side: str, role: str, parts: list) -> None:
        """Record one speech message (audio and/or text) as content-part attributes.

        For the speech SDKs, whose request or response is an audio file rather than a
        chat message (speech-to-text in, text-to-speech out). Audio goes through
        ``offload_part`` -- the same size cap, modality allow-list, redactor and store
        as every other media part -- and only when media capture is on; text only with
        content capture on. Never raises: capture must not fail the traced call.
        """
        try:
            kept = [
                p
                for p in parts
                if p is not None
                and (
                    (p.type == "text" and self._speech_text_enabled())
                    or (p.type != "text" and self._speech_media_enabled())
                )
            ]
            if not kept:
                return
            from ..media import (  # noqa: WPS433 - lazy, as in _emit_media_attributes
                get_store,
                offload_part,
            )

            store = None
            if (
                any(p.type != "text" for p in kept)
                and getattr(self.config, "media_capture_mode", "off") == "full"
            ):
                try:
                    store = get_store(self.config)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to initialize media store: %s", e)
            trace_id_hex = ""
            try:
                ctx = span.get_span_context()
                if ctx and ctx.trace_id:
                    trace_id_hex = f"{ctx.trace_id:032x}"
            except Exception:  # noqa: BLE001
                pass
            role_key = SC.GEN_AI_PROMPT_ROLE if side == "prompt" else SC.GEN_AI_COMPLETION_ROLE
            span.set_attribute(role_key.format(n=0), role)
            for m, part in enumerate(kept):
                try:
                    offload_part(part, config=self.config, store=store, trace_id=trace_id_hex)
                except Exception as e:  # noqa: BLE001
                    logger.warning("offload_part (speech) failed: %s", e)
                declared = part.extra.get("declared_size") if part.extra else None
                if part.extra and part.extra.get("stripped_reason") == "size_exceeded" and declared:
                    part.media_byte_size = int(declared)
                self._set_part_attrs(span, side, 0, m, part)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to emit speech media for span: %s", e)

    def create_span_wrapper(
        self, span_name: str, extract_attributes: Optional[Callable[[Any, Any, Any], Dict]] = None
    ) -> Callable:
        """Create a decorator that instruments a function with an OpenTelemetry span."""

        @wrapt.decorator
        def wrapper(wrapped, instance, args, kwargs):
            # If instrumentation failed during initialization, just call the original function.
            if not self._instrumented:
                logger.debug("Instrumentation not active, calling %s directly", span_name)
                return wrapped(*args, **kwargs)

            # Tracks whether the wrapped (business) call has been invoked. The
            # outer fallback below must NEVER re-invoke it once it has run, or an
            # errored/sampled-out call would execute twice (double API spend and
            # duplicated side effects).
            invoked = False

            try:
                # Start a new span
                initial_attributes = {}
                if extract_attributes:
                    try:
                        extracted_attrs = extract_attributes(instance, args, kwargs)
                        extracted_attrs = self._with_provider_aliases(extracted_attrs)
                        extracted_attrs = strip_content_attributes(extracted_attrs, self.config)
                        for key, value in extracted_attrs.items():
                            if isinstance(value, (str, int, float, bool)):
                                initial_attributes[key] = value
                            else:
                                initial_attributes[key] = str(value)
                    except Exception as e:
                        logger.warning(
                            "Failed to extract attributes for span '%s': %s", span_name, e
                        )

                # Endpoint identity, derived centrally so every instrumentor
                # using this wrapper gets it. Never overwrites a value an
                # instrumentor set itself -- it knows its SDK better than the
                # generic walk does.
                try:
                    for _sk, _sv in _server_attributes(instance).items():
                        initial_attributes.setdefault(_sk, _sv)
                    for _rk, _rv in _request_parameter_attributes(kwargs).items():
                        initial_attributes.setdefault(_rk, _rv)
                except Exception as e:  # pragma: no cover - defensive
                    logger.debug("Could not derive span attributes for '%s': %s", span_name, e)

                # Check if this is a streaming request before creating the span
                is_streaming = kwargs.get("stream", False)

                # Start the span and activate it as the current span in context
                # so nested calls (e.g. tool calls within an LLM call) inherit
                # this span as their parent for proper trace hierarchy
                span = self.tracer.start_span(span_name, attributes=initial_attributes)

                # Fast path for sampled-out spans. When trace sampling drops a
                # span the tracer returns a NonRecordingSpan, which has no `.name`
                # or `.attributes`; the full recording path below touched those and
                # raised, and the outer fallback then re-invoked the wrapped call
                # (double execution). Here we skip all extraction/record work (it
                # would be discarded anyway) and record only the request count and
                # latency, keeping client metrics independent of trace sampling.
                if not span.is_recording():
                    ctx = trace.set_span_in_context(span)
                    token = otel_context.attach(ctx)
                    fast_start = time.time()

                    def _count_sampled_out():
                        try:
                            if self.request_counter:
                                self.request_counter.add(1, {"operation": span_name})
                            if self.latency_histogram:
                                self.latency_histogram.record(
                                    time.time() - fast_start, {"operation": span_name}
                                )
                        except Exception:  # pragma: no cover - metrics must never break the call
                            pass

                    deferred = False
                    try:
                        if BaseInstrumentor._blocking_active:
                            self._enforce_precall_policy(
                                span, kwargs, initial_attributes.get("gen_ai.operation.name")
                            )
                        invoked = True
                        result = wrapped(*args, **kwargs)
                        if inspect.iscoroutine(result):
                            # The work happens where the coroutine is awaited. Ended
                            # here, the span left that work with nothing to inherit
                            # the sampling decision from: its children became roots
                            # and were sampled on their own, as partial traces.
                            deferred = True

                            async def _sampled_out():
                                run_token = otel_context.attach(trace.set_span_in_context(span))
                                try:
                                    value = await result
                                    _count_sampled_out()
                                    return value
                                finally:
                                    otel_context.detach(run_token)
                                    span.end()

                            return _sampled_out()
                        _count_sampled_out()
                        return result
                    finally:
                        if not deferred:
                            span.end()
                        otel_context.detach(token)

                ctx = _context_with_usage_holder(span, trace.set_span_in_context(span))
                token = otel_context.attach(ctx)
                # Begin accumulating budget consumption if this is an agent
                # invocation. No-op for ordinary LLM spans.
                self._begin_agent_budget(span, initial_attributes)
                start_time = time.time()

                # Increment server metrics: running requests counter
                server_metrics = get_server_metrics()
                if server_metrics:
                    server_metrics.increment_requests_running()
                    logger.debug(f"Incremented running requests for {span_name}")

                # Extract session and user context (Phase 4.1)
                if self.config:
                    if self.config.session_id_extractor:
                        try:
                            session_id = self.config.session_id_extractor(instance, args, kwargs)
                            if session_id:
                                span.set_attribute("session.id", session_id)
                                # Cross-framework conversation correlation per upstream
                                # proposal semantic-conventions-genai#145.
                                span.set_attribute("gen_ai.conversation.id", session_id)
                                logger.debug("Set session.id: %s", session_id)
                        except Exception as e:
                            logger.debug("Failed to extract session ID: %s", e)

                    if self.config.user_id_extractor:
                        try:
                            user_id = self.config.user_id_extractor(instance, args, kwargs)
                            if user_id:
                                span.set_attribute("user.id", user_id)
                                logger.debug("Set user.id: %s", user_id)
                        except Exception as e:
                            logger.debug("Failed to extract user ID: %s", e)

                try:
                    # Pre-call block policy (gated; zero cost unless a blocking
                    # detector is configured). Raises PolicyViolationError to
                    # prevent the LLM call entirely when a policy triggers.
                    if BaseInstrumentor._blocking_active:
                        self._enforce_precall_policy(
                            span, kwargs, initial_attributes.get("gen_ai.operation.name")
                        )
                    # Call the original function
                    invoked = True
                    result = wrapped(*args, **kwargs)

                    # Handle async functions: if the result is a coroutine,
                    # keep the span open and context attached until the
                    # coroutine completes so nested async calls inherit this
                    # span as their parent.
                    if inspect.iscoroutine(result):
                        # Nothing has run yet: the work happens wherever this
                        # coroutine is awaited, and under asyncio.run, gather
                        # or create_task that is another task with its own
                        # context. Left attached here, the span would stay the
                        # caller's current span for good - the detach would
                        # run in the other context and fail - and a second
                        # call made before the first is awaited would nest
                        # under it. So the caller's context is put back now,
                        # and the span is made current again where it runs.
                        budgeted = getattr(span, "_genai_budget_token", None) is not None
                        if budgeted:
                            self._abandon_agent_budget(span)
                        otel_context.detach(token)

                        async def _async_traced():
                            handed_to_stream = False
                            evaluate_later = False
                            actual_result = None
                            # Built here rather than reused from the call, so what
                            # the caller put in its context before awaiting is seen.
                            run_token = otel_context.attach(
                                _context_with_usage_holder(span, trace.set_span_in_context(span))
                            )
                            if budgeted:
                                self._begin_agent_budget(span, initial_attributes)
                            try:
                                actual_result = await result

                                # An async streaming call resolves to the stream
                                # object, not to the answer: the model has produced
                                # nothing yet. Closing the span here would time the
                                # handshake instead of the generation and drop TTFT,
                                # tokens and cost entirely, which is why async
                                # clients reported no streaming latency at all.
                                if is_streaming and hasattr(actual_result, "__aiter__"):
                                    handed_to_stream = True
                                    if self.request_counter:
                                        self.request_counter.add(1, {"operation": span.name})
                                    return self._wrap_async_streaming_response(
                                        actual_result,
                                        span,
                                        start_time,
                                        kwargs.get(
                                            "model",
                                            initial_attributes.get(
                                                "gen_ai.request.model", "unknown"
                                            ),
                                        ),
                                    )

                                if self.request_counter:
                                    self.request_counter.add(1, {"operation": span.name})

                                try:
                                    self._record_result_metrics(
                                        span, actual_result, start_time, kwargs
                                    )
                                except Exception as e:
                                    logger.warning(
                                        "Failed to record metrics for span '%s': %s",
                                        span_name,
                                        e,
                                    )

                                # Detectors: here, or on a worker thread that is
                                # handed the span once this coroutine is done
                                # with it (see the `finally` below).
                                evaluate_later = self._evaluate_or_defer(
                                    span, args, kwargs, actual_result
                                )

                                span.set_status(Status(StatusCode.OK))
                                return actual_result

                            except Exception as e:
                                try:
                                    if self.error_counter:
                                        self.error_counter.add(
                                            1,
                                            {
                                                "operation": span_name,
                                                "error_type": type(e).__name__,
                                            },
                                        )
                                except Exception:
                                    pass
                                span.set_status(Status(StatusCode.ERROR, str(e)))
                                span.record_exception(e)
                                raise

                            finally:
                                if budgeted:
                                    self._finish_agent_budget(span)
                                otel_context.detach(run_token)
                                # When the stream wrapper took over it owns the
                                # span's end and the running-requests decrement.
                                if not handed_to_stream:
                                    if evaluate_later:
                                        self._end_span_after_evaluation(
                                            span, args, kwargs, actual_result
                                        )
                                    else:
                                        span.end()
                                    server_metrics = get_server_metrics()
                                    if server_metrics:
                                        server_metrics.decrement_requests_running()

                        return _async_traced()

                    # Handle streaming responses (Phase 3.4). This has to run
                    # before the generic generator branches below: a provider
                    # that returns a bare generator for stream=True (Ollama)
                    # used to fall into them and be timed but never measured --
                    # no TTFT, no end-of-stream usage, no cost.
                    if is_streaming:
                        stream_model = kwargs.get(
                            "model", initial_attributes.get("gen_ai.request.model", "unknown")
                        )
                        logger.debug("Detected streaming response for model: %s", stream_model)
                        handled, value = self._install_stream_measurement(
                            span, result, start_time, stream_model, kwargs, sync_call=True
                        )
                        if handled:
                            if self.request_counter:
                                self.request_counter.add(1, {"operation": span.name})
                            # The stream is read after this call returns, so a
                            # budget frame opened for it here would never close.
                            self._abandon_agent_budget(span)
                            # Detach context before handing the span over; the
                            # stream wrapper owns its lifecycle from here.
                            otel_context.detach(token)
                            return value

                    # Handle async generators (e.g. astream(), run_stream())
                    # These return immediately but the actual work happens during
                    # iteration, so keep the span open until iteration completes.
                    if inspect.isasyncgen(result):
                        # Iterated later, possibly elsewhere: see the stream case.
                        self._abandon_agent_budget(span)

                        async def _async_gen_traced():
                            try:
                                async for item in result:
                                    yield item
                                if self.request_counter:
                                    self.request_counter.add(1, {"operation": span.name})
                                span.set_status(Status(StatusCode.OK))
                            except Exception as e:
                                try:
                                    if self.error_counter:
                                        self.error_counter.add(
                                            1,
                                            {
                                                "operation": span_name,
                                                "error_type": type(e).__name__,
                                            },
                                        )
                                except Exception:
                                    pass
                                span.set_status(Status(StatusCode.ERROR, str(e)))
                                span.record_exception(e)
                                raise
                            finally:
                                span.end()
                                otel_context.detach(token)
                                server_metrics = get_server_metrics()
                                if server_metrics:
                                    server_metrics.decrement_requests_running()

                        return _async_gen_traced()

                    # Handle sync generators (e.g. stream())
                    if inspect.isgenerator(result):
                        self._abandon_agent_budget(span)

                        def _gen_traced():
                            try:
                                yield from result
                                if self.request_counter:
                                    self.request_counter.add(1, {"operation": span.name})
                                span.set_status(Status(StatusCode.OK))
                            except Exception as e:
                                try:
                                    if self.error_counter:
                                        self.error_counter.add(
                                            1,
                                            {
                                                "operation": span_name,
                                                "error_type": type(e).__name__,
                                            },
                                        )
                                except Exception:
                                    pass
                                span.set_status(Status(StatusCode.ERROR, str(e)))
                                span.record_exception(e)
                                raise
                            finally:
                                span.end()
                                otel_context.detach(token)
                                server_metrics = get_server_metrics()
                                if server_metrics:
                                    server_metrics.decrement_requests_running()

                        return _gen_traced()

                    if self.request_counter:
                        self.request_counter.add(1, {"operation": span.name})

                    # Non-streaming: record metrics and close span normally
                    try:
                        self._record_result_metrics(span, result, start_time, kwargs)
                    except Exception as e:
                        logger.warning("Failed to record metrics for span '%s': %s", span_name, e)

                    # Run evaluation checks BEFORE ending the span: here, or on
                    # a worker thread that is handed the span below and ends it.
                    evaluate_later = self._evaluate_or_defer(span, args, kwargs, result)

                    self._finish_agent_budget(span)

                    # Set span status to OK on successful execution
                    span.set_status(Status(StatusCode.OK))
                    if evaluate_later:
                        # Last, after everything this thread writes on the span:
                        # the worker may end it at any moment from here on.
                        self._end_span_after_evaluation(span, args, kwargs, result)
                    else:
                        span.end()

                    # Detach context after span is ended
                    otel_context.detach(token)

                    # Decrement server metrics: running requests counter
                    server_metrics = get_server_metrics()
                    if server_metrics:
                        server_metrics.decrement_requests_running()
                        logger.debug(f"Decremented running requests for {span_name}")

                    return result

                except Exception as e:
                    # Handle exceptions during the wrapped function execution
                    try:
                        if self.error_counter:
                            self.error_counter.add(
                                1, {"operation": span_name, "error_type": type(e).__name__}
                            )
                    except Exception:
                        pass

                    self._finish_agent_budget(span)

                    # Set span status to ERROR and record the exception
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    span.end()

                    # Detach context after span is ended (error path)
                    otel_context.detach(token)

                    # Decrement server metrics: running requests counter (error path)
                    server_metrics = get_server_metrics()
                    if server_metrics:
                        server_metrics.decrement_requests_running()
                        logger.debug(f"Decremented running requests for {span_name} (error)")

                    raise

            except Exception as e:
                if invoked or isinstance(e, PolicyViolationError):
                    # Either the wrapped call already executed (a failure here is
                    # in our instrumentation, not the business call), or this is a
                    # deliberate pre-call policy block. Never (re-)invoke the
                    # wrapped function - propagate the original exception.
                    raise
                # Failure happened during span setup before the call ran: fall back
                # to calling the wrapped function directly so instrumentation never
                # breaks the host application.
                logger.error("Span creation failed for '%s': %s", span_name, e, exc_info=True)
                return wrapped(*args, **kwargs)

        return wrapper

    def _with_provider_aliases(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """Return ``attrs`` with the model provider under both accepted spellings.

        ``gen_ai.system`` was renamed to ``gen_ai.provider.name``. Roughly 29
        instrumentors write the superseded spelling as a raw string literal in
        their own ``_extract_*_attributes``, so mirroring it here -- on the two
        paths every one of those dicts flows through -- fixes all of them without
        editing any, and keeps the naming policy in one place. The same reason
        token emission was centralised in :meth:`_set_token_usage_attributes`.

        This is the provider half of the interop problem 1.9.0 fixed for tokens:
        a backend reading the current conventions sees no provider at all, which
        is not a cosmetic gap -- consumers routinely treat a missing provider as
        "not a GenAI span" and drop the record rather than showing it unlabelled.

        Never overwrites a value the instrumentor set deliberately, and returns
        the dict unchanged when neither spelling is present, so non-GenAI spans
        are untouched. Copies before mutating -- the caller's dict may be reused.
        """
        current = attrs.get(SC.GEN_AI_PROVIDER_NAME)
        superseded = attrs.get(SC.GEN_AI_SYSTEM)
        if current == superseded:  # both absent, or already consistent
            return attrs

        _, emit_superseded = genai_semconv_modes(
            self.config.semconv_stability_opt_in if self.config else None
        )

        mirrored = dict(attrs)
        if current is None:
            mirrored[SC.GEN_AI_PROVIDER_NAME] = superseded
        elif superseded is None and emit_superseded:
            mirrored[SC.GEN_AI_SYSTEM] = current
        return mirrored

    def _set_token_usage_attributes(
        self,
        span,
        prompt_tokens: Any = None,
        completion_tokens: Any = None,
    ) -> None:
        """Set token-usage span attributes per the OTel GenAI semantic conventions.

        ``gen_ai.usage.input_tokens`` / ``gen_ai.usage.output_tokens`` are the
        current convention: semantic-conventions v1.27.0 renamed
        ``gen_ai.usage.prompt_tokens`` -> ``gen_ai.usage.input_tokens`` and
        ``gen_ai.usage.completion_tokens`` -> ``gen_ai.usage.output_tokens``
        (open-telemetry/semantic-conventions#1200). They are therefore always
        emitted.

        The superseded names are emitted only when the operator opts into dual
        emission with ``OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai/dup``, which is
        what that flag means upstream.

        This matters for interoperability, not just tidiness: a backend that
        consumes the current conventions reads zero tokens when only the
        superseded names are present. Arize AX, for one, maps
        ``gen_ai.usage.{input,output}_tokens`` onto ``llm.token_count.*`` and
        derives cost from them.
        """
        _, emit_superseded = genai_semconv_modes(
            self.config.semconv_stability_opt_in if self.config else None
        )

        if isinstance(prompt_tokens, (int, float)) and prompt_tokens > 0:
            span.set_attribute(SC.GEN_AI_USAGE_INPUT_TOKENS, int(prompt_tokens))
            if emit_superseded:
                span.set_attribute(SC.GEN_AI_USAGE_PROMPT_TOKENS, int(prompt_tokens))

        if isinstance(completion_tokens, (int, float)) and completion_tokens > 0:
            span.set_attribute(SC.GEN_AI_USAGE_OUTPUT_TOKENS, int(completion_tokens))
            if emit_superseded:
                span.set_attribute(SC.GEN_AI_USAGE_COMPLETION_TOKENS, int(completion_tokens))

    def add_embedding_request_attributes(
        self, attrs: Dict[str, Any], dimensions: Any = None, encoding_format: Any = None
    ) -> Dict[str, Any]:
        """Record embeddings request shape under the registry spellings.

        The conventions name these `gen_ai.embeddings.dimension.count` and
        `gen_ai.request.encoding_formats` (an array). This library shipped
        `gen_ai.request.dimensions` and a singular `encoding_format`, which no
        consumer following the conventions looks for; both are kept under the
        dup policy so existing dashboards keep resolving.
        """
        # Attribute extraction runs at call time, which can precede `config`
        # being attached to the instrumentor, so read it defensively rather
        # than assuming the attribute exists.
        _cfg = getattr(self, "config", None)
        _, emit_superseded = genai_semconv_modes(_cfg.semconv_stability_opt_in if _cfg else None)
        if dimensions is not None:
            attrs[SC.GEN_AI_EMBEDDINGS_DIMENSION_COUNT] = dimensions
            if emit_superseded:
                attrs[SC.GEN_AI_REQUEST_DIMENSIONS] = dimensions
        if encoding_format is not None:
            formats = (
                list(encoding_format)
                if isinstance(encoding_format, (list, tuple))
                else [encoding_format]
            )
            attrs[SC.GEN_AI_REQUEST_ENCODING_FORMATS] = [str(f) for f in formats]
            if emit_superseded:
                attrs[SC.GEN_AI_REQUEST_ENCODING_FORMAT] = encoding_format
        return attrs

    # Operation names that denote an agent invocation rather than a single
    # model call. Budget accounting is scoped to these so an ordinary LLM span
    # pays nothing for the feature.
    _AGENT_OPERATION_MARKERS = (
        "agent",
        "runner",
        "graph",
        "workflow",
        "crew",
        "conversation",
        "task.execution",
    )

    @classmethod
    def _is_agent_invocation(cls, attrs: Dict[str, Any]) -> bool:
        operation = str(attrs.get("gen_ai.operation.name") or "").lower()
        if not operation:
            return False
        return any(marker in operation for marker in cls._AGENT_OPERATION_MARKERS)

    def _begin_agent_budget(self, span, attrs: Dict[str, Any]) -> None:
        """Start budget accounting for an agent invocation.

        Consumption is accumulated in a context-local frame rather than summed
        from child spans afterwards, because head sampling drops exactly those
        children -- a sampled trace would otherwise report a runaway agent as
        having consumed nothing.
        """
        try:
            if not self._is_agent_invocation(attrs):
                return
            token = agent_budget.push_frame()
            if token is not None:
                span._genai_budget_token = token
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("Could not begin agent budget accounting: %s", e)

    def _abandon_agent_budget(self, span) -> None:
        """Drop an accounting frame that was opened in a context the work will not run in."""
        try:
            token = getattr(span, "_genai_budget_token", None)
            if token is None:
                return
            span._genai_budget_token = None
            agent_budget.pop_frame(token)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("Could not abandon agent budget accounting: %s", e)

    def _finish_agent_budget(self, span) -> None:
        """Emit budget consumption and utilization, then end the accounting frame."""
        try:
            token = getattr(span, "_genai_budget_token", None)
            if token is None:
                return
            try:
                span._genai_budget_token = None
            except Exception:  # pragma: no cover - defensive
                pass

            frame = agent_budget.pop_frame(token)
            consumed = agent_budget.consumption_attributes(frame)
            for key, value in consumed.items():
                span.set_attribute(key, value)

            attrs = getattr(span, "attributes", None) or {}
            ratio = agent_budget.utilization(
                consumed.get(SC.GEN_AI_AGENT_TOKEN_BUDGET_CONSUMED),
                attrs.get(SC.GEN_AI_AGENT_TOKEN_BUDGET),
            )
            if ratio is not None and self.token_budget_utilization_histogram:
                self.token_budget_utilization_histogram.record(
                    ratio,
                    {
                        "gen_ai.operation.name": str(attrs.get("gen_ai.operation.name", "")),
                        "gen_ai.agent.name": str(attrs.get("gen_ai.agent.name", "")),
                    },
                )
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("Could not finish agent budget accounting: %s", e)

    def _set_modality_token_attributes(self, span, usage: Dict[str, Any]) -> None:
        """Emit `gen_ai.usage.<modality>.*` token counts when a provider reports them.

        Each value is a subset of the corresponding total: a consumer that sums
        the modality attributes alongside `gen_ai.usage.input_tokens` double
        counts. Absent modalities are omitted rather than emitted as zero -- a
        zero would claim the provider reported no audio tokens, when in practice
        it usually means the provider does not break usage down at all.
        """
        if not isinstance(usage, dict):
            return
        for modality in SC.TOKEN_MODALITIES:
            for key, template in (
                (
                    f"{modality}_input_tokens",
                    SC.GEN_AI_USAGE_MODALITY_INPUT_TOKENS,
                ),
                (
                    f"{modality}_output_tokens",
                    SC.GEN_AI_USAGE_MODALITY_OUTPUT_TOKENS,
                ),
                (
                    f"{modality}_cache_read_input_tokens",
                    SC.GEN_AI_USAGE_MODALITY_CACHE_READ_INPUT_TOKENS,
                ),
            ):
                value = usage.get(key)
                if isinstance(value, (int, float)) and value > 0:
                    span.set_attribute(template.format(modality=modality), int(value))

    def _record_result_metrics(self, span, result, start_time: float, request_kwargs: dict = None):
        """Record metrics derived from the function result and execution time.

        Args:
            span: The OpenTelemetry span to record metrics on.
            result: The result from the wrapped function.
            start_time: The time when the function started executing.
            request_kwargs: The original request kwargs (for content capture).
        """
        # A raw-response wrapper hides the model behind .parse(); unwrap before
        # any extraction runs, or usage, cost and finish-reason are all silently
        # dropped. This is the LiteLLM path, so it affects every caller routed
        # through an LLM gateway. See _unwrap_api_response.
        result = _unwrap_api_response(result)

        # Resolve metric-verbosity flags once. Full per-request detail is always
        # written to span attributes below regardless of these flags; they only
        # gate the extra aggregated-metric instruments recorded on the hot path.
        cfg = self.config
        _profile_full = bool(cfg) and getattr(cfg, "metrics_profile", "standard") == "full"
        _rec_token_hist = _profile_full or (
            bool(cfg) and getattr(cfg, "record_token_histograms", False)
        )
        _rec_granular = _profile_full or (
            bool(cfg) and getattr(cfg, "record_granular_cost_metrics", False)
        )
        _rec_finish = _profile_full or (bool(cfg) and getattr(cfg, "record_finish_metrics", False))

        # A NonRecordingSpan (sampled out, or an API-default tracer used by a
        # custom async/sync wrapper) exposes neither .name nor .attributes; read
        # them defensively so this method never raises on such spans.
        _name = getattr(span, "name", None)
        _sname = _name or "unknown"
        _sattrs = getattr(span, "attributes", None) or {}

        # Record latency
        try:
            duration = time.time() - start_time
            if self.latency_histogram:
                self.latency_histogram.record(duration, {"operation": _sname})
        except Exception as e:
            logger.warning("Failed to record latency for span '%s': %s", _sname, e)

        if _name is None and _is_sampled_out(span):
            # Everything below is written to the span or derived from what is
            # on it, and a span the sampler dropped keeps nothing. Wrappers
            # that build their own span call this without looking, and it used
            # to raise out of a call the provider had already answered. A span
            # with a name is a recording one, so only a nameless span is asked.
            return

        # Extract and set response attributes if available
        try:
            if hasattr(self, "_extract_response_attributes"):
                response_attrs = self._extract_response_attributes(result)
                if response_attrs and isinstance(response_attrs, dict):
                    response_attrs = self._with_provider_aliases(response_attrs)
                    response_attrs = strip_content_attributes(response_attrs, self.config)
                    for key, value in response_attrs.items():
                        if isinstance(value, (str, int, float, bool)):
                            span.set_attribute(key, value)
                        elif isinstance(value, list):
                            # For arrays like finish_reasons
                            span.set_attribute(key, value)
                        else:
                            span.set_attribute(key, str(value))
        except Exception as e:
            logger.warning("Failed to extract response attributes for span '%s': %s", _sname, e)

        # Add content events if content capture is enabled
        try:
            if (
                hasattr(self, "_add_content_events")
                and self.config
                and self.config.enable_content_capture
            ):
                self._add_content_events(span, result, request_kwargs or {})
            if (
                self.config
                and self.config.enable_content_capture
                and getattr(span, "attributes", {}).get("gen_ai.request.type") == "embedding"
                and not getattr(span, "attributes", {}).get("embedding.text")
            ):
                self._add_generic_embedding_content(span, result, request_kwargs or {})
        except Exception as e:
            logger.warning("Failed to add content events for span '%s': %s", _sname, e)

        # Multimodal content-part attributes (gated on MEDIA_PROVIDER + capture mode).
        # Additive: when MEDIA_PROVIDER is None or capture_mode == "off", this is a no-op.
        try:
            if (
                self.MEDIA_PROVIDER
                and self.config
                and getattr(self.config, "media_capture_mode", "off") != "off"
            ):
                self._emit_media_attributes(span, request_kwargs or {}, result)
        except Exception as e:
            logger.warning("Failed to emit media attributes for span '%s': %s", _sname, e)

        # Extract and record token usage and cost. If the provider response
        # lacks usage data (common for Ollama multimodal and several HF
        # transformers pipelines), fall back to a per-instrumentor estimate
        # based on the request + result text/tokens. The estimate path tags
        # the span with `gen_ai.usage.token_count_estimated=true` so downstream
        # consumers can distinguish exact from estimated token counts.
        try:
            usage = self._extract_usage(result)
            estimated = False
            if not usage:
                est = self._estimate_usage(result, request_kwargs or {})
                if est:
                    usage = est
                    estimated = True
            if usage and isinstance(usage, dict) and _descendant_recorded_usage(span):
                # A span beneath this one already recorded this request's tokens
                # and cost (a framework over an instrumented provider SDK).
                # Writing them again here would report one request twice.
                _leave_usage_to_descendant(span)
                usage = None
            if usage and isinstance(usage, dict):
                # This span is accounting for the request. A wrapper further out
                # (litellm around a provider SDK, a framework around either) reads
                # this and stays silent, so one request is never billed twice.
                _note_inner_measurement()
                _mark_usage_recorded(span)
                if estimated:
                    try:
                        span.set_attribute("gen_ai.usage.token_count_estimated", True)
                    except Exception:  # noqa: BLE001
                        pass
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)

                # Record token counts if available and positive. Which attribute
                # names are emitted (current vs superseded) is decided centrally
                # by _set_token_usage_attributes.
                self._set_token_usage_attributes(
                    span,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

                # Record prompt tokens
                if isinstance(prompt_tokens, (int, float)) and prompt_tokens > 0:
                    # Record counter metric if available
                    if self.token_counter:
                        self.token_counter.add(
                            prompt_tokens, {"token_type": "prompt", "operation": _sname}
                        )
                    # Record histogram for distribution analysis (opt-in)
                    if _rec_token_hist and self.prompt_tokens_histogram:
                        model = _sattrs.get("gen_ai.request.model", "unknown")
                        self.prompt_tokens_histogram.record(
                            int(prompt_tokens), {"model": str(model), "operation": _sname}
                        )

                # Record completion tokens
                if isinstance(completion_tokens, (int, float)) and completion_tokens > 0:
                    # Record counter metric if available
                    if self.token_counter:
                        self.token_counter.add(
                            completion_tokens, {"token_type": "completion", "operation": _sname}
                        )
                    # Record histogram for distribution analysis (opt-in)
                    if _rec_token_hist and self.completion_tokens_histogram:
                        model = _sattrs.get("gen_ai.request.model", "unknown")
                        self.completion_tokens_histogram.record(
                            int(completion_tokens), {"model": str(model), "operation": _sname}
                        )

                # Record total tokens
                if isinstance(total_tokens, (int, float)) and total_tokens > 0:
                    span.set_attribute("gen_ai.usage.total_tokens", int(total_tokens))
                    # Attribute this call to the enclosing agent invocation's
                    # budget, if there is one. No-op for a plain LLM call.
                    agent_budget.record_inference(int(total_tokens))

                # Record detailed token-usage breakdowns when the provider
                # supplies them. Attribute names align with upstream
                # `semantic-conventions-genai#76` (detailed token usage:
                # cache + reasoning).
                #   - Anthropic populates cache_read_input_tokens /
                #     cache_creation_input_tokens at top level of usage_dict.
                #   - OpenAI populates completion_tokens_details.reasoning_tokens
                #     (o1/o3-style models) and prompt_tokens_details.cached_tokens
                #     (prompt caching).
                cache_read = usage.get("cache_read_input_tokens")
                if isinstance(cache_read, (int, float)) and cache_read > 0:
                    span.set_attribute(SC.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, int(cache_read))
                cache_creation = usage.get("cache_creation_input_tokens")
                if isinstance(cache_creation, (int, float)) and cache_creation > 0:
                    # `cache_write` is the current spelling (renamed upstream in
                    # semantic-conventions-genai#440); `cache_creation` is the
                    # superseded one and follows the same dual-emission policy as
                    # prompt/completion tokens.
                    _, _emit_superseded_cache = genai_semconv_modes(
                        self.config.semconv_stability_opt_in if self.config else None
                    )
                    span.set_attribute(
                        SC.GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS, int(cache_creation)
                    )
                    if _emit_superseded_cache:
                        span.set_attribute(
                            SC.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS, int(cache_creation)
                        )
                completion_details = usage.get("completion_tokens_details")
                if isinstance(completion_details, dict):
                    reasoning = completion_details.get("reasoning_tokens")
                    if isinstance(reasoning, (int, float)) and reasoning > 0:
                        # Upstream-standardised name (in registry at Development
                        # stability since open-telemetry/semantic-conventions#3194,
                        # migrated to semantic-conventions-genai). The value
                        # is also included in `gen_ai.usage.output_tokens`.
                        span.set_attribute(SC.GEN_AI_USAGE_REASONING_TOKENS, int(reasoning))

                # Per-modality token breakdown (semantic-conventions-genai#440).
                # Instrumentors normalise provider-specific shapes into flat
                # `<modality>_{input,output}_tokens` /
                # `<modality>_cache_read_input_tokens` keys on usage_dict, the
                # same way they already do for cache and reasoning tokens, so
                # this stays a single emission point.
                self._set_modality_token_attributes(span, usage)

                # Calculate and record cost if enabled and applicable
                if self.config and self.config.enable_cost_tracking:
                    try:
                        model = _sattrs.get("gen_ai.request.model", "unknown")
                        # Assuming 'chat' as a default call_type for generic base instrumentor tests.
                        # Specific instrumentors will provide the actual call_type.
                        call_type = _sattrs.get("gen_ai.request.type", "chat")

                        # Use granular cost calculation for chat requests
                        if call_type == "chat":
                            costs = self.cost_calculator.calculate_granular_cost(
                                model, usage, call_type
                            )
                            total_cost = costs["total"]

                            # Record total cost (always) and set span attribute.
                            if total_cost > 0:
                                if self.cost_counter:
                                    self.cost_counter.add(total_cost, {"model": str(model)})
                                # Always set span attributes (needed for cost tracking)
                                span.set_attribute("gen_ai.usage.cost.total", total_cost)

                            # Say where the price came from. Without this a 0.0
                            # cost is indistinguishable from a genuinely free
                            # call, so unpriced models silently read as free
                            # rather than as unmeasured.
                            span.set_attribute(
                                "gen_ai.usage.cost.pricing_source",
                                self.cost_calculator.pricing_source(model, call_type),
                            )

                            # Flag models the provider has announced the end of.
                            # These bill normally right up until withdrawal, so
                            # nothing else in the telemetry distinguishes them -
                            # the only warning is a vendor changelog. Emitted
                            # only when set, so active models stay unaffected.
                            _dep = self.cost_calculator.deprecation(model, call_type)
                            if _dep:
                                span.set_attribute("gen_ai.request.model.deprecated", True)
                                span.set_attribute("gen_ai.request.model.deprecation_note", _dep)

                            # Granular costs: span attributes are ALWAYS set (needed for
                            # audit / explainability); the per-breakdown metric counters
                            # are opt-in (_rec_granular) to keep the hot path lean.
                            if costs["prompt"] > 0:
                                if _rec_granular and self.prompt_cost_counter:
                                    self.prompt_cost_counter.add(
                                        costs["prompt"], {"model": str(model)}
                                    )
                                span.set_attribute("gen_ai.usage.cost.prompt", costs["prompt"])

                            if costs["completion"] > 0:
                                if _rec_granular and self.completion_cost_counter:
                                    self.completion_cost_counter.add(
                                        costs["completion"], {"model": str(model)}
                                    )
                                span.set_attribute(
                                    "gen_ai.usage.cost.completion", costs["completion"]
                                )

                            if costs["reasoning"] > 0:
                                if _rec_granular and self.reasoning_cost_counter:
                                    self.reasoning_cost_counter.add(
                                        costs["reasoning"], {"model": str(model)}
                                    )
                                span.set_attribute(
                                    "gen_ai.usage.cost.reasoning", costs["reasoning"]
                                )

                            if costs["cache_read"] > 0:
                                if _rec_granular and self.cache_read_cost_counter:
                                    self.cache_read_cost_counter.add(
                                        costs["cache_read"], {"model": str(model)}
                                    )
                                span.set_attribute(
                                    "gen_ai.usage.cost.cache_read", costs["cache_read"]
                                )

                            if costs["cache_write"] > 0:
                                if _rec_granular and self.cache_write_cost_counter:
                                    self.cache_write_cost_counter.add(
                                        costs["cache_write"], {"model": str(model)}
                                    )
                                span.set_attribute(
                                    "gen_ai.usage.cost.cache_write", costs["cache_write"]
                                )
                        else:
                            # For non-chat requests (image, audio, embedding,
                            # speech_to_text, ...), use simple cost calculation
                            # AND surface the result on the span so downstream
                            # tooling can read `gen_ai.usage.cost.total` for
                            # all call types, not only chat.
                            cost = self.cost_calculator.calculate_cost(model, usage, call_type)
                            if cost and cost > 0:
                                if self.cost_counter:
                                    self.cost_counter.add(cost, {"model": str(model)})
                                span.set_attribute("gen_ai.usage.cost.total", cost)
                    except Exception as e:
                        logger.warning("Failed to calculate cost for span '%s': %s", _sname, e)

        except Exception as e:
            logger.warning("Failed to extract or record usage metrics for span '%s': %s", _sname, e)

        # Extract and record finish reason if available (for request outcome tracking)
        try:
            if hasattr(self, "_extract_finish_reason"):
                finish_reason = self._extract_finish_reason(result)
                if finish_reason:
                    # Set span attribute (always - needed for audit).
                    # The conventions define an array (`finish_reasons`); the
                    # singular is this library's older spelling and follows the
                    # dup policy. A consumer reading the plural name found
                    # nothing here, even though several instrumentors already
                    # emitted it -- so the library disagreed with itself.
                    _, _emit_superseded_fr = genai_semconv_modes(
                        self.config.semconv_stability_opt_in if self.config else None
                    )
                    # An instrumentor that already extracted a real array keeps
                    # it; this only fills in from the singular value.
                    if SC.GEN_AI_RESPONSE_FINISH_REASONS not in (
                        getattr(span, "attributes", None) or {}
                    ):
                        span.set_attribute(SC.GEN_AI_RESPONSE_FINISH_REASONS, [str(finish_reason)])
                    if _emit_superseded_fr:
                        span.set_attribute(SC.GEN_AI_RESPONSE_FINISH_REASON, finish_reason)

                    # Finish/success/failure metric counters are opt-in (_rec_finish).
                    if _rec_finish:
                        model = _sattrs.get("gen_ai.request.model", "unknown")

                        # Record finish reason counter
                        if self.request_finish_counter:
                            self.request_finish_counter.add(
                                1, {"finish_reason": finish_reason, "model": str(model)}
                            )

                        # Track success vs failure based on finish reason
                        # Success: stop, length, end_turn, etc.
                        # Failure: error, content_filter, timeout, etc.
                        success_reasons = {"stop", "length", "end_turn", "max_tokens"}
                        failure_reasons = {"error", "content_filter", "timeout", "rate_limit"}

                        if finish_reason in success_reasons:
                            if self.request_success_counter:
                                self.request_success_counter.add(1, {"model": str(model)})
                        elif finish_reason in failure_reasons:
                            if self.request_failure_counter:
                                self.request_failure_counter.add(
                                    1, {"finish_reason": finish_reason, "model": str(model)}
                                )
        except Exception as e:
            logger.debug("Failed to extract or record finish reason for span '%s': %s", _sname, e)

    @staticmethod
    def _responses_prompt_text(kwargs) -> Optional[str]:
        """The prompt of an OpenAI Responses call: ``instructions`` plus ``input``.

        ``input`` is a string, or a list of items whose ``content`` is a string
        or a list of parts such as ``{"type": "input_text", "text": ...}``.
        """
        parts: List[str] = []
        instructions = kwargs.get("instructions")
        if isinstance(instructions, str):
            parts.append(instructions)
        raw_input = kwargs.get("input")
        if isinstance(raw_input, str):
            parts.append(raw_input)
        elif isinstance(raw_input, list):
            for item in raw_input:
                content = (
                    item.get("content")
                    if isinstance(item, dict)
                    else getattr(item, "content", None)
                )
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        text = (
                            part.get("text")
                            if isinstance(part, dict)
                            else getattr(part, "text", None)
                        )
                        if isinstance(text, str):
                            parts.append(text)
        joined = "\n".join(p for p in parts if p)
        return joined or None

    @staticmethod
    def _extract_prompt_for_eval(kwargs, operation: Optional[str] = None) -> Optional[str]:
        """Best-effort prompt text extraction for pre-call policy evaluation.

        Reads the full ``messages`` list (so USER content is evaluated, not just
        a system/first message) and falls back to a plain ``prompt`` kwarg.

        ``operation`` is the call's ``gen_ai.operation.name`` when the caller
        knows it. A chat call that carries its prompt as ``input`` /
        ``instructions`` (the OpenAI Responses API) is read only when it is
        ``"chat"``: an embeddings call passes ``input`` too, and that is a
        document being indexed, not a prompt.
        """
        msgs = kwargs.get("messages")
        if msgs and isinstance(msgs, list):
            parts: List[str] = []
            for m in msgs:
                content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    parts.extend(blk.get("text", "") for blk in content if isinstance(blk, dict))
            joined = "\n".join(p for p in parts if p)
            if joined:
                return joined
        prompt = kwargs.get("prompt")
        if isinstance(prompt, str):
            return prompt
        if operation == "chat":
            return BaseInstrumentor._responses_prompt_text(kwargs)
        return None

    def _extract_response_for_eval(self, result) -> Optional[str]:
        """The model's answer, for the detectors that check a response.

        Read from the result itself rather than from span attributes, so it does
        not depend on content capture being on. Chat Completions shape and the
        OpenAI Responses API here; providers with another shape override this.
        """
        choices = getattr(result, "choices", None)
        if choices:
            first = choices[0]
            if hasattr(first, "message"):
                return first.message.content
            if hasattr(first, "text"):
                return first.text
        output_text = getattr(result, "output_text", None)
        if isinstance(output_text, str) and output_text:
            return output_text
        return None

    def _apply_block_policies(self, span, kwargs, operation: Optional[str] = None) -> None:
        """Pre-call block policies, for a wrapper that builds its own span.

        Raises ``PolicyViolationError``; the caller must let it reach the
        application and must not make the provider call. Costs one attribute
        read when no block policy is configured.
        """
        if BaseInstrumentor._blocking_active:
            self._enforce_precall_policy(span, kwargs, operation)

    def _block_or_end(self, span, kwargs, operation: Optional[str] = None) -> None:
        """Block policies for a span with no error handler around it yet.

        A blocked request ends ``span`` as an error before the exception leaves,
        so the caller's wrapper does not have to. For paths that open the span
        and hand it to a stream.
        """
        try:
            self._apply_block_policies(span, kwargs, operation)
        except PolicyViolationError as e:
            try:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.end()
            except Exception:  # noqa: BLE001 - the block must reach the caller regardless
                pass
            raise

    def _apply_detectors(self, span, args, kwargs, result) -> None:
        """Post-call detectors, for a wrapper that builds its own span. Never raises."""
        try:
            self._run_evaluation_checks(span, args, kwargs, result)
        except Exception as e:
            logger.warning(
                "Failed to run evaluation checks for span '%s': %s", getattr(span, "name", "?"), e
            )

    def _enforce_precall_policy(self, span, kwargs, operation: Optional[str] = None) -> None:
        """Evaluate the prompt against block-mode detectors BEFORE the LLM call.

        Runs only the policies the operator configured with block_on_detection /
        PII block mode (``_blocking_policies``). If any triggers, records the
        block on the span and raises ``PolicyViolationError`` so the wrapped call
        is never executed. Gated by ``_blocking_active`` in the caller, so the
        default (non-blocking) hot path never reaches this method.
        """
        policies = BaseInstrumentor._blocking_policies
        if not policies:
            return
        prompt = self._extract_prompt_for_eval(kwargs, operation)
        if not prompt:
            return

        # Each detector is shown what the post-call pass will show it, so that
        # pass can take the answer from the detector's cache. A request
        # decided on part of its prompt says so, blocked or not.
        limit = self._evaluation_max_chars()
        window = self._prompt_window(prompt, limit)
        cut = False

        # (policy name, detector, truthy result attribute)
        checks = (
            ("pii", BaseInstrumentor._pii_detector, "has_pii"),
            ("toxicity", BaseInstrumentor._toxicity_detector, "is_toxic"),
            ("prompt_injection", BaseInstrumentor._prompt_injection_detector, "is_injection"),
            (
                "restricted_topics",
                BaseInstrumentor._restricted_topics_detector,
                "has_restricted_topic",
            ),
            ("bias", BaseInstrumentor._bias_detector, "has_bias"),
        )
        triggered: List[str] = []
        for name, detector, flag in checks:
            if name not in policies or detector is None:
                continue
            try:
                if name == "pii":
                    cut = cut or (window is not prompt and not self._pii_reads_everything())
                    result = self._detect_pii(span, detector, prompt, window, limit)
                elif name == "toxicity":
                    # The part the later pass will read too, so that pass
                    # takes this answer from the detector's cache.
                    text = self._toxicity_text(span, window, limit, len(prompt))
                    cut = cut or (window is not prompt and self._toxicity_reads_the_window(limit))
                    result = detector.detect(text)
                else:
                    text = self._policy_text(span, name, prompt, window)
                    cut = cut or text is not prompt
                    result = detector.detect(text)
            except Exception as e:  # noqa: BLE001 - a detector fault must not block the call
                logger.warning("Pre-call %s policy check failed: %s", name, e)
                continue
            if getattr(result, flag, False):
                try:
                    span.set_attribute(f"evaluation.{name}.prompt.blocked", True)
                except Exception:  # noqa: BLE001
                    pass
                triggered.append(name)

        if cut:
            self._mark_truncated(span, "prompt", len(prompt), len(window))

        if triggered:
            try:
                span.set_attribute("evaluation.blocked", True)
                span.set_attribute("evaluation.blocked_reasons", triggered)
            except Exception:  # noqa: BLE001
                pass
            raise PolicyViolationError(
                "Request blocked by evaluation policy: " + ", ".join(triggered),
                policy=triggered[0],
                details={"reasons": triggered},
            )

    @staticmethod
    def _evaluation_active() -> bool:
        """Whether any detector is configured."""
        return bool(
            BaseInstrumentor._pii_detector
            or BaseInstrumentor._toxicity_detector
            or BaseInstrumentor._bias_detector
            or BaseInstrumentor._prompt_injection_detector
            or BaseInstrumentor._restricted_topics_detector
            or BaseInstrumentor._hallucination_detector
        )

    def _evaluation_in_background(self) -> bool:
        """Whether detectors run on a worker thread once a call has returned.

        That is the default. Without a config (direct use, unit tests) the
        default applies.
        """
        return getattr(self.config, "evaluation_mode", None) != EVALUATION_MODE_INLINE

    def _evaluate_or_defer(self, span, args, kwargs, result) -> bool:
        """Run the post-call detectors on ``span`` now, unless a worker thread is to.

        Returns True when they are left to a worker. The caller then writes
        whatever else it has to say on the span and, instead of ending it,
        hands it over with :meth:`_end_span_after_evaluation`.

        With no detector configured this is the six attribute reads the call
        has always cost, and nothing is handed anywhere. Never raises.
        """
        if not (
            BaseInstrumentor._pii_detector
            or BaseInstrumentor._toxicity_detector
            or BaseInstrumentor._bias_detector
            or BaseInstrumentor._prompt_injection_detector
            or BaseInstrumentor._restricted_topics_detector
            or BaseInstrumentor._hallucination_detector
        ):
            return False
        if _is_sampled_out(span):
            # Nothing a detector found could be written anywhere.
            return False
        if getattr(self.config, "evaluation_mode", None) != EVALUATION_MODE_INLINE:
            return True
        try:
            self._run_evaluation_checks(span, args, kwargs, result)
        except Exception as e:
            logger.warning(
                "Failed to run evaluation checks for span '%s': %s", getattr(span, "name", "?"), e
            )
        return False

    def _evaluation_setting(self, name: str, default: Any, kind: Any) -> Any:
        """An evaluation setting from the config, or its default when there is none.

        ``kind`` is the type, or tuple of types, a usable value has. Without a
        config (direct use, unit tests), or with one that predates the setting,
        the default applies.
        """
        value = getattr(self.config, name, None)
        if isinstance(value, bool) or not isinstance(value, kind):
            return default
        return value

    def _evaluation_max_chars(self) -> int:
        """Characters of a prompt or response the detectors read. 0 = all of it."""
        return max(
            0, self._evaluation_setting("evaluation_max_chars", DEFAULT_EVALUATION_MAX_CHARS, int)
        )

    def _toxicity_max_chars(self, limit: int) -> int:
        """Characters of a prompt the toxicity detector reads. 0 = all of it.

        The local toxicity model is the slowest detector by far, so it has a
        limit of its own. That limit applies when it is the smaller of the two:
        ``limit`` is the one every detector is under, and no limit there (0)
        leaves the toxicity limit as it is. No toxicity limit (0) leaves the
        general one.
        """
        own = max(
            0, self._evaluation_setting("toxicity_max_chars", DEFAULT_TOXICITY_MAX_CHARS, int)
        )
        if own and limit:
            return min(own, limit)
        return own or limit

    def _toxicity_reads_the_window(self, limit: int) -> bool:
        """Whether toxicity reads what the other detectors read, not less."""
        return self._toxicity_max_chars(limit) == limit

    def _toxicity_text(self, span, window, limit: int, prompt_length: Optional[int]):
        """The part of a prompt the toxicity detector reads: the end of ``window``.

        ``window`` is the part every detector may read. The span says so
        whenever what toxicity read is less than the whole prompt, which can be
        the case when nothing else was cut: a prompt longer than the toxicity
        limit and shorter than the general one.
        """
        text = self._prompt_window(window, self._toxicity_max_chars(limit))
        if isinstance(text, str) and prompt_length is not None and len(text) < prompt_length:
            try:
                span.set_attribute("evaluation.truncated", True)
                span.set_attribute("evaluation.prompt.length", prompt_length)
                span.set_attribute("evaluation.toxicity.prompt.analyzed_length", len(text))
            except Exception:  # noqa: BLE001
                pass
        return text

    @staticmethod
    def _only_toxicity_reads_prompts() -> bool:
        """Whether the toxicity detector is the only one configured that reads a prompt."""
        return bool(BaseInstrumentor._toxicity_detector) and not (
            BaseInstrumentor._pii_detector
            or BaseInstrumentor._bias_detector
            or BaseInstrumentor._prompt_injection_detector
            or BaseInstrumentor._restricted_topics_detector
            # Reads the response, with the prompt as its context.
            or BaseInstrumentor._hallucination_detector
        )

    @staticmethod
    def _prompt_window(text, limit: int):
        """The part of a prompt the detectors read: its end, the newest turns."""
        if limit and isinstance(text, str) and len(text) > limit:
            return text[-limit:]
        return text

    @staticmethod
    def _response_window(text, limit: int):
        """The part of a response the detectors read: its start."""
        if limit and isinstance(text, str) and len(text) > limit:
            return text[:limit]
        return text

    @staticmethod
    def _mark_truncated(span, side: str, length: int, analyzed_length: int) -> None:
        """Say on the span that the detectors read only part of ``side``."""
        try:
            span.set_attribute("evaluation.truncated", True)
            span.set_attribute("evaluation.%s.length" % side, length)
            span.set_attribute("evaluation.%s.analyzed_length" % side, analyzed_length)
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _reads_whole_prompt(name: str) -> bool:
        """Whether detector ``name`` is exempt from the limit on analysed text."""
        return (
            name in BaseInstrumentor._WHOLE_PROMPT_BLOCK_POLICIES
            and name in BaseInstrumentor._blocking_policies
        )

    @staticmethod
    def _pii_reads_everything() -> bool:
        """Whether the PII detector reads a whole text whatever the limit.

        In redact mode it does: see ``PIIDetector.reads_whole_text``. Nothing
        was cut for it then, and a span must not say that something was.
        """
        return getattr(BaseInstrumentor._pii_detector, "reads_whole_text", False) is True

    @staticmethod
    def _whole_prompt_wanted() -> bool:
        """Whether a configured detector reads more of a prompt than its last part."""
        pii = BaseInstrumentor._pii_detector
        if pii is not None and (
            getattr(pii, "accepts_model_chars", False) is True
            or getattr(pii, "reads_whole_text", False) is True
        ):
            return True
        return bool(
            (BaseInstrumentor._bias_detector and BaseInstrumentor._reads_whole_prompt("bias"))
            or (
                BaseInstrumentor._prompt_injection_detector
                and BaseInstrumentor._reads_whole_prompt("prompt_injection")
            )
            or (
                BaseInstrumentor._restricted_topics_detector
                and BaseInstrumentor._reads_whole_prompt("restricted_topics")
            )
        )

    def _texts_to_keep(self, prompt, response) -> Tuple[Any, Any]:
        """What of a call's texts the configured detectors are going to read.

        A span waits in the queue with its texts. Kept whole, a full queue of
        long prompts and responses is memory held for text that nothing will
        look at: the detectors read the end of a prompt and the start of a
        response. So that is what is kept: the largest part of the prompt that
        any configured detector reads, which is all of it only when one of
        them reads all of it, and the response only when one reads responses
        at all.
        """
        limit = self._evaluation_max_chars()
        if not self._whole_prompt_wanted():
            if self._only_toxicity_reads_prompts():
                prompt = self._prompt_window(prompt, self._toxicity_max_chars(limit))
            else:
                prompt = self._prompt_window(prompt, limit)
        if BaseInstrumentor._hallucination_detector:
            response = self._response_window(response, limit)
        else:
            response = None
        return prompt, response

    @staticmethod
    def _policy_text(span, name: str, prompt: str, window: str) -> str:
        """The part of ``prompt`` that detector ``name`` reads.

        ``window``, unless the detector is exempt from the limit; the span
        then says that this detector read everything.
        """
        if window is prompt or not BaseInstrumentor._reads_whole_prompt(name):
            return window
        try:
            span.set_attribute("evaluation.%s.prompt.analyzed_length" % name, len(prompt))
        except Exception:  # noqa: BLE001
            pass
        return prompt

    @staticmethod
    def _detect_pii(span, detector, prompt: str, window: str, limit: int):
        """Look for PII in a prompt, of which ``window`` is the part detectors read.

        A detector that can limit its own model is given the whole prompt and
        the limit instead: its patterns then cover the part its model does not
        read, so a card number or an address early in a long prompt is found.
        One that reads everything whatever the limit (redact mode) is given the
        whole prompt and no limit.
        """
        if window is prompt:
            return detector.detect(window)
        if getattr(detector, "reads_whole_text", False) is True:
            return detector.detect(prompt)
        if getattr(detector, "accepts_model_chars", False) is not True:
            return detector.detect(window)
        result = detector.detect(prompt, model_chars=limit)
        model_length = getattr(result, "model_analyzed_length", None)
        if isinstance(model_length, int) and not isinstance(model_length, bool):
            try:
                span.set_attribute("evaluation.pii.prompt.model_analyzed_length", model_length)
                span.set_attribute("evaluation.pii.prompt.pattern_analyzed_length", len(prompt))
            except Exception:  # noqa: BLE001
                pass
        return result

    def _end_span_after_evaluation(self, span, args, kwargs, result) -> None:
        """Hand ``span`` to the worker threads, which run the detectors and end it.

        For a wrapper that was told to by :meth:`_evaluate_or_defer` and has
        written everything else it has to say on the span. The span is no
        longer the caller's when this returns: it has ended, or will without
        the caller. Either way it ends as of now, so that it measures the call
        and not the detectors. Never raises and never waits.
        """
        end_time = time.time_ns()
        if _is_sampled_out(span):
            # A span the sampler dropped has nothing to annotate and nowhere to
            # read the request from.
            span.end(end_time=end_time)
            return
        try:
            limit = max(
                1,
                self._evaluation_setting(
                    "evaluation_queue_size", DEFAULT_EVALUATION_QUEUE_SIZE, int
                ),
            )
            if _EVALUATION_QUEUE.has_room(limit):
                # Read here, on the caller's thread. A caller is free to add
                # the answer and its next question to the same message list as
                # soon as the call returns; the detectors must see the request
                # that was sent.
                prompt, response = self._evaluation_texts(span, kwargs, result)
                kept_prompt, kept_response = self._texts_to_keep(prompt, response)
                _EVALUATION_QUEUE.shutdown_timeout = self._evaluation_setting(
                    "evaluation_shutdown_timeout",
                    DEFAULT_EVALUATION_SHUTDOWN_TIMEOUT,
                    (int, float),
                )
                pending = _SpanUnderEvaluation(
                    self,
                    span,
                    kept_prompt,
                    kept_response,
                    end_time,
                    len(prompt) if isinstance(prompt, str) else None,
                    len(response) if isinstance(response, str) else None,
                )
                if _EVALUATION_QUEUE.submit(pending, limit):
                    return
            # The detectors are behind. The span is exported now, unevaluated
            # and saying so: tracing does not wait for them.
            span.set_attribute("evaluation.skipped", "overloaded")
            _EVALUATION_QUEUE.note_overload()
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Failed to queue evaluation for span '%s': %s", getattr(span, "name", "?"), e
            )
        span.end(end_time=end_time)

    async def _enforce_precall_policy_async(
        self, span, kwargs, operation: Optional[str] = None
    ) -> None:
        """:meth:`_enforce_precall_policy` for an async caller, off the event loop.

        Raises ``PolicyViolationError`` exactly as the sync form does. Costs
        nothing when no block policy is configured.
        """
        if not BaseInstrumentor._blocking_policies:
            return
        await _off_the_event_loop(self._enforce_precall_policy, span, kwargs, operation)

    async def _run_evaluation_checks_async(self, span, args, kwargs, result) -> None:
        """:meth:`_run_evaluation_checks` for an async caller, off the event loop.

        Costs nothing when no detector is configured.

        A caller that is cancelled while it waits gets ``CancelledError`` as
        usual, and the detectors, which a worker thread is still running, stop
        writing to the span: the caller is about to end it or hand it on. A
        caller that still wants the span evaluated hands it to
        :meth:`_end_span_after_evaluation` then.
        """
        if not self._evaluation_active() or _is_sampled_out(span):
            return
        watched = _SpanForDetectors(span)
        try:
            await _off_the_event_loop(self._run_evaluation_checks, watched, args, kwargs, result)
        except asyncio.CancelledError:
            watched.stop()
            raise

    def _run_evaluation_checks(self, span, args, kwargs, result):
        """Run all evaluation checks (PII, toxicity, bias, prompt injection, etc.) before ending the span.

        This method extracts prompt and response from span attributes and runs
        evaluation detectors if they are configured. Attributes are added to the
        span BEFORE it ends, ensuring they appear in exported traces.

        Args:
            span: The active span (mutable, not yet ended)
            args: Original function arguments
            kwargs: Original function keyword arguments
            result: The LLM response object
        """
        if (
            not BaseInstrumentor._pii_detector
            and not BaseInstrumentor._toxicity_detector
            and not BaseInstrumentor._bias_detector
            and not BaseInstrumentor._prompt_injection_detector
            and not BaseInstrumentor._restricted_topics_detector
            and not BaseInstrumentor._hallucination_detector
        ):
            return  # No detectors configured
        if _is_sampled_out(span):
            return  # Nothing a detector found could be written anywhere

        try:
            prompt, response = self._evaluation_texts(span, kwargs, result)
            self._evaluate_texts(span, prompt, response)
        except Exception as e:
            logger.warning(f"Evaluation checks failed: {e}", exc_info=True)

    def _evaluation_texts(self, span, kwargs, result) -> Tuple[Optional[str], Optional[str]]:
        """The prompt and the response of a finished call, for the detectors.

        Read on the thread that made the call, before the span is handed to a
        worker: the request's message list belongs to the caller, who may
        change it as soon as the call has returned.

        Nothing of either text is logged, at any level: a log is not where a
        prompt belongs, whatever the content-capture setting says.
        """
        # Extract prompt and response from span attributes
        attrs = dict(getattr(span, "attributes", None) or {})

        # The prompt, read the same way the pre-call policy reads it: the FULL
        # message list, so PII/toxicity/bias in USER messages (not only the
        # system/first message) is evaluated. Providers that carry the prompt
        # differently override _extract_prompt_for_eval.
        prompt = None
        try:
            prompt = self._extract_prompt_for_eval(kwargs, attrs.get("gen_ai.operation.name"))
        except Exception as e:  # noqa: BLE001 - fall back to the span attribute
            logger.debug("Failed to extract prompt for evaluation: %s", type(e).__name__)
        if prompt is None and "gen_ai.request.first_message" in attrs:
            value = attrs["gen_ai.request.first_message"]
            if isinstance(value, str):
                # Try json.loads first (preferred format)
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, dict) and "content" in parsed:
                        prompt = parsed["content"]
                except (json.JSONDecodeError, ValueError):
                    # Fallback: try ast.literal_eval for legacy Python dict strings
                    try:
                        parsed = ast.literal_eval(value)
                        if isinstance(parsed, dict) and "content" in parsed:
                            prompt = parsed["content"]
                    except (ValueError, SyntaxError):
                        pass
                if prompt is None:
                    # Truncated string - extract content with regex fallback
                    match = re.search(r'"content":\s*"(.*)', value)
                    if not match:
                        match = re.search(r"'content':\s*'(.*)", value)
                    if match:
                        prompt = match.group(1).rstrip("'\"}")
                    else:
                        # Last resort: use the raw value as prompt
                        prompt = value
                logger.debug(
                    "Prompt for evaluation read from gen_ai.request.first_message: %s characters",
                    len(prompt) if isinstance(prompt, str) else "no",
                )

        # Extract response - try to get from result object
        response = None
        try:
            response = self._extract_response_for_eval(result)
        except Exception as e:  # noqa: BLE001 - fall back to the span attribute
            logger.debug("Failed to extract response for evaluation: %s", type(e).__name__)
        if response is None:
            # Structured decision SDKs (for example TypeSafe) do not expose
            # chat-style choices. Their instrumentors may preserve a JSON
            # response on the span when content capture is enabled.
            response = attrs.get("gen_ai.response") or attrs.get("output.value")

        return prompt, response

    def _evaluate_texts(
        self,
        span,
        prompt,
        response,
        prompt_length: Optional[int] = None,
        response_length: Optional[int] = None,
    ) -> None:
        """Run the configured detectors on a call's texts and annotate ``span``.

        ``span`` only has to take ``set_attribute``: on a worker thread it is
        the span as held by the evaluation queue, not the span itself.

        ``prompt`` and ``response`` may be the parts of longer texts that the
        evaluation queue kept (see :meth:`_texts_to_keep`); the lengths are then
        those of the texts they came from, so that the span says how much was
        read of how much.
        """
        try:
            # Past the limit the detectors read the end of the prompt and the
            # start of the response. The span says so whenever that cut
            # something a detector was going to read.
            limit = self._evaluation_max_chars()
            # All of the prompt that there is to read here.
            whole_prompt = prompt
            prompt = self._prompt_window(prompt, limit)
            if prompt_length is None and isinstance(whole_prompt, str):
                prompt_length = len(whole_prompt)
            prompt_cut = (
                isinstance(prompt, str)
                and prompt_length is not None
                and len(prompt) < prompt_length
            )
            reads_response = bool(BaseInstrumentor._hallucination_detector and response)
            whole = self._reads_whole_prompt
            if prompt_cut and (
                reads_response
                or (BaseInstrumentor._pii_detector and not self._pii_reads_everything())
                # Toxicity says what it read under its own name; it counts
                # here only when that is this same part and not less.
                or (BaseInstrumentor._toxicity_detector and self._toxicity_reads_the_window(limit))
                or (BaseInstrumentor._bias_detector and not whole("bias"))
                or (BaseInstrumentor._prompt_injection_detector and not whole("prompt_injection"))
                or (BaseInstrumentor._restricted_topics_detector and not whole("restricted_topics"))
            ):
                self._mark_truncated(span, "prompt", prompt_length, len(prompt))
            if reads_response:
                whole_response = response
                response = self._response_window(response, limit)
                if response_length is None and isinstance(whole_response, str):
                    response_length = len(whole_response)
                if (
                    isinstance(response, str)
                    and response_length is not None
                    and len(response) < response_length
                ):
                    self._mark_truncated(span, "response", response_length, len(response))

            # Run PII detection
            if BaseInstrumentor._pii_detector and prompt:
                try:
                    logger.info("Running PII detection on a prompt of %d characters", len(prompt))
                    pii_result = self._detect_pii(
                        span, BaseInstrumentor._pii_detector, whole_prompt, prompt, limit
                    )
                    logger.info(
                        f"PII detection result: has_pii={pii_result.has_pii}, entities={list(pii_result.entity_counts.keys()) if pii_result.has_pii else []}"
                    )
                    if pii_result.has_pii:
                        span.set_attribute("evaluation.pii.prompt.detected", True)
                        span.set_attribute(
                            "evaluation.pii.prompt.entity_count", len(pii_result.entities)
                        )
                        span.set_attribute(
                            "evaluation.pii.prompt.entity_types",
                            list(pii_result.entity_counts.keys()),
                        )
                        span.set_attribute("evaluation.pii.prompt.score", pii_result.score)

                        # Add entity counts by type
                        for entity_type, count in pii_result.entity_counts.items():
                            span.set_attribute(
                                f"evaluation.pii.prompt.{entity_type.lower()}_count", count
                            )

                        if pii_result.redacted_text:
                            span.set_attribute(
                                "evaluation.pii.prompt.redacted", pii_result.redacted_text
                            )
                        if pii_result.blocked:
                            span.set_attribute("evaluation.pii.prompt.blocked", True)
                    else:
                        span.set_attribute("evaluation.pii.prompt.detected", False)
                except Exception as e:
                    logger.warning(f"PII detection failed: {e}", exc_info=True)

            # Run toxicity detection
            if BaseInstrumentor._toxicity_detector and prompt:
                try:
                    toxicity_result = BaseInstrumentor._toxicity_detector.detect(
                        self._toxicity_text(span, prompt, limit, prompt_length)
                    )
                    if toxicity_result.is_toxic:
                        span.set_attribute("evaluation.toxicity.prompt.detected", True)
                        span.set_attribute(
                            "evaluation.toxicity.prompt.max_score", toxicity_result.max_score
                        )
                        span.set_attribute(
                            "evaluation.toxicity.prompt.categories",
                            toxicity_result.toxic_categories,
                        )

                        # Add individual category scores
                        for category, score in toxicity_result.scores.items():
                            span.set_attribute(
                                f"evaluation.toxicity.prompt.{category}_score", score
                            )
                    else:
                        span.set_attribute("evaluation.toxicity.prompt.detected", False)
                except Exception as e:
                    logger.warning(f"Toxicity detection failed: {e}", exc_info=True)

            # Run bias detection
            if BaseInstrumentor._bias_detector and prompt:
                try:
                    logger.info("Running bias detection on a prompt of %d characters", len(prompt))
                    bias_result = BaseInstrumentor._bias_detector.detect(
                        self._policy_text(span, "bias", whole_prompt, prompt)
                    )
                    logger.info(
                        f"Bias detection result: has_bias={bias_result.has_bias}, max_score={bias_result.max_score}, detected_biases={bias_result.detected_biases}"
                    )
                    if bias_result.has_bias:
                        span.set_attribute("evaluation.bias.prompt.detected", True)
                        span.set_attribute(
                            "evaluation.bias.prompt.max_score", bias_result.max_score
                        )
                        span.set_attribute(
                            "evaluation.bias.prompt.detected_biases",
                            bias_result.detected_biases,
                        )

                        # Add individual bias type scores
                        for bias_type, score in bias_result.bias_scores.items():
                            if score > 0:
                                span.set_attribute(
                                    f"evaluation.bias.prompt.{bias_type}_score", score
                                )

                        # Add patterns matched (limit to first 5 per type)
                        for bias_type, patterns in bias_result.patterns_matched.items():
                            if patterns:
                                span.set_attribute(
                                    f"evaluation.bias.prompt.{bias_type}_patterns",
                                    patterns[:5],
                                )
                    else:
                        span.set_attribute("evaluation.bias.prompt.detected", False)
                except Exception as e:
                    logger.warning(f"Bias detection failed: {e}", exc_info=True)

            # Run prompt injection detection
            if BaseInstrumentor._prompt_injection_detector and prompt:
                try:
                    injection_result = BaseInstrumentor._prompt_injection_detector.detect(
                        self._policy_text(span, "prompt_injection", whole_prompt, prompt)
                    )
                    if injection_result.is_injection:
                        span.set_attribute("evaluation.prompt_injection.detected", True)
                        span.set_attribute(
                            "evaluation.prompt_injection.score", injection_result.injection_score
                        )
                        span.set_attribute(
                            "evaluation.prompt_injection.types", injection_result.injection_types
                        )

                        # Add patterns matched (limit to first 5 per type)
                        for inj_type, patterns in injection_result.patterns_matched.items():
                            if patterns:
                                span.set_attribute(
                                    f"evaluation.prompt_injection.{inj_type}_patterns",
                                    patterns[:5],
                                )
                    else:
                        span.set_attribute("evaluation.prompt_injection.detected", False)
                except Exception as e:
                    logger.warning(f"Prompt injection detection failed: {e}", exc_info=True)

            # Run restricted topics detection
            if BaseInstrumentor._restricted_topics_detector and prompt:
                try:
                    topics_result = BaseInstrumentor._restricted_topics_detector.detect(
                        self._policy_text(span, "restricted_topics", whole_prompt, prompt)
                    )
                    if topics_result.has_restricted_topic:
                        span.set_attribute("evaluation.restricted_topics.prompt.detected", True)
                        span.set_attribute(
                            "evaluation.restricted_topics.prompt.max_score", topics_result.max_score
                        )
                        span.set_attribute(
                            "evaluation.restricted_topics.prompt.topics",
                            topics_result.detected_topics,
                        )

                        # Add individual topic scores
                        for topic, score in topics_result.topic_scores.items():
                            if score > 0:
                                span.set_attribute(
                                    f"evaluation.restricted_topics.prompt.{topic}_score", score
                                )
                    else:
                        span.set_attribute("evaluation.restricted_topics.prompt.detected", False)
                except Exception as e:
                    logger.warning(f"Restricted topics detection failed: {e}", exc_info=True)

            # Run hallucination detection (requires response)
            if BaseInstrumentor._hallucination_detector and response:
                try:
                    # Use prompt as context if available
                    hallucination_result = BaseInstrumentor._hallucination_detector.detect(
                        response, context=prompt
                    )
                    span.set_attribute(
                        "evaluation.hallucination.response.detected",
                        hallucination_result.has_hallucination,
                    )
                    span.set_attribute(
                        "evaluation.hallucination.response.score",
                        hallucination_result.hallucination_score,
                    )
                    span.set_attribute(
                        "evaluation.hallucination.response.citations",
                        hallucination_result.citation_count,
                    )
                    span.set_attribute(
                        "evaluation.hallucination.response.hedge_words",
                        hallucination_result.hedge_words_count,
                    )
                    span.set_attribute(
                        "evaluation.hallucination.response.claims",
                        hallucination_result.factual_claim_count,
                    )

                    if hallucination_result.has_hallucination:
                        span.set_attribute(
                            "evaluation.hallucination.response.indicators",
                            hallucination_result.hallucination_indicators,
                        )
                        if hallucination_result.unsupported_claims:
                            span.set_attribute(
                                "evaluation.hallucination.response.unsupported_claims",
                                hallucination_result.unsupported_claims[:3],
                            )
                except Exception as e:
                    logger.warning(f"Hallucination detection failed: {e}", exc_info=True)

            # Mark evaluation as done so the downstream EvaluationSpanProcessor /
            # EvaluationEnrichingSpanExporter can skip re-running the (expensive)
            # detector suite on the same span - detectors previously ran up to 3x
            # per request (inline here, in on_end, and in the exporter).
            try:
                span.set_attribute("evaluation.completed", True)
            except Exception:  # noqa: BLE001
                pass

        except Exception as e:
            logger.warning(f"Evaluation checks failed: {e}", exc_info=True)

    def _wrap_streaming_response(
        self, stream, span, start_time: float, model: str, emit_measurements: bool = True
    ):
        """Wrap a streaming response to capture TTFT, TPOT and end-of-stream usage.

        Hands back the stream itself as far as the caller can tell - still a
        context manager if it was one, with its ``close()`` and ``response`` -
        with iteration measuring time to first token (TTFT) and time between
        tokens (TBT). The span is finalized when the stream completes, errors,
        or is closed or abandoned.

        Args:
            stream: The streaming response iterator
            span: The OpenTelemetry span for this request
            start_time: Request start time (for TTFT calculation)
            model: Model name/identifier for metric attributes

        Returns:
            The stream, proxied so that iterating it is measured
        """
        if _is_sampled_out(span):
            # A sampled-out span has no name or attributes to measure onto; the
            # finalizer raised on the caller's first chunk. Nothing to record.
            self._end_unmeasured_stream(span)
            return stream

        timing = _StreamTiming(start_time, emit_measurements)
        return _MeasuredStream(
            stream,
            self._measured_chunks(stream, span, timing, model),
            timing,
            lambda: self._finalize_stream(span, timing, model),
        )

    def _end_unmeasured_stream(self, span) -> None:
        """End a span whose stream is handed back as it came."""
        span.end()
        server_metrics = get_server_metrics()
        if server_metrics:
            server_metrics.decrement_requests_running()

    def _measured_chunks(self, stream, span, timing: "_StreamTiming", model: str):
        """Yield a stream's chunks, observing each one and finalizing the span."""
        timing.started = True
        try:
            for chunk in stream:
                self._observe_stream_chunk(span, timing, chunk, model)
                yield chunk
        except GeneratorExit:
            # Consumer walked away mid-stream (a `break`, or a timeout). Close
            # the span with what was actually measured instead of leaking it.
            self._finalize_stream(span, timing, model)
            raise
        except Exception as e:
            self._fail_measured_stream(span, timing, model, e)
            raise
        except BaseException:
            # KeyboardInterrupt and the like are not errors of the stream, but
            # the span still has to end.
            self._finalize_stream(span, timing, model)
            raise

        self._finalize_stream(span, timing, model)

    def _wrap_async_streaming_response(
        self, stream, span, start_time: float, model: str, emit_measurements: bool = True
    ):
        """Async counterpart of :meth:`_wrap_streaming_response`.

        Async SDK calls resolve to the stream object rather than to the answer,
        so awaiting the call tells us nothing about the model's latency -- the
        generation happens while the caller iterates. Wrapping that iteration
        is the only place TTFT, TPOT and the final usage chunk are observable
        on an async client.

        Args:
            stream: The async streaming response iterator
            span: The OpenTelemetry span for this request
            start_time: Request start time (for TTFT calculation)
            model: Model name/identifier for metric attributes

        Returns:
            The stream, proxied so that iterating it is measured
        """
        if _is_sampled_out(span):
            self._end_unmeasured_stream(span)
            return stream

        timing = _StreamTiming(start_time, emit_measurements)
        return _MeasuredAsyncStream(
            stream,
            self._measured_chunks_async(stream, span, timing, model),
            timing,
            lambda: self._finalize_stream(span, timing, model),
        )

    async def _measured_chunks_async(self, stream, span, timing: "_StreamTiming", model: str):
        """Async counterpart of :meth:`_measured_chunks`."""
        timing.started = True
        try:
            async for chunk in stream:
                self._observe_stream_chunk(span, timing, chunk, model)
                yield chunk
        except GeneratorExit:
            # See the sync wrapper: an abandoned stream still gets its span
            # closed with the measurements taken so far.
            self._finalize_stream(span, timing, model)
            raise
        except Exception as e:
            self._fail_measured_stream(span, timing, model, e)
            raise
        except BaseException:
            # A consumer cancelled while waiting for a chunk (client disconnect,
            # a timeout) raises CancelledError, which is not an Exception.
            self._finalize_stream(span, timing, model)
            raise

        self._finalize_stream(span, timing, model)

    def _observe_stream_chunk(self, span, timing: "_StreamTiming", chunk, model: str) -> None:
        """Record first-token / inter-token latency for a single streamed chunk."""
        current_time = time.time()
        timing.chunk_count += 1

        if timing.ttft is None:
            timing.ttft = current_time - timing.start_time
            if timing.emit_measurements:
                self._record_time_to_first_token(span, timing.ttft, model)
        elif timing.emit_measurements:
            tbt = current_time - timing.last_chunk_time
            timing.tbt_sum += tbt
            timing.tbt_count += 1
            if timing.tbt_per_chunk is None:
                timing.tbt_per_chunk = self._metrics_profile_full()
            if timing.tbt_per_chunk and self.tbt_histogram:
                self.tbt_histogram.record(tbt, {"model": model, "operation": _span_name(span)})

        timing.last_chunk_time = current_time
        timing.last_chunk = chunk  # Keep track of last chunk for usage extraction

        # Give the provider a chance to fold usage it reports mid-stream. A
        # failure here must never break the caller's iteration -- a missing
        # token count is a reporting gap, a raised exception is an outage.
        try:
            self._accumulate_stream_usage(timing, chunk)
        except Exception as e:  # noqa: BLE001 - measurement must not break the stream
            logger.debug("Failed to accumulate streamed usage: %s", e)

    def _metrics_profile_full(self) -> bool:
        """Whether the ``full`` metrics profile is on (see ``OTelConfig.metrics_profile``)."""
        cfg = self.config
        return bool(cfg) and getattr(cfg, "metrics_profile", "standard") == "full"

    def _record_mean_time_between_tokens(self, span, timing: "_StreamTiming", model: str) -> None:
        """Record a finished stream's mean time between chunks, as one sample.

        Nothing to record for a stream that sampled every chunk as it went, for
        one whose measurements belong to an inner span, or for one that never
        got a second chunk.
        """
        if not timing.emit_measurements or timing.tbt_per_chunk or not timing.tbt_count:
            return
        if self.tbt_histogram:
            self.tbt_histogram.record(
                timing.tbt_sum / timing.tbt_count, {"model": model, "operation": _span_name(span)}
            )

    def _accumulate_stream_usage(self, timing: "_StreamTiming", chunk) -> None:
        """Fold usage reported part-way through a stream. Default: do nothing.

        Most providers repeat the full usage on the final chunk, so the
        finalizer can simply read that one. Some split it across events and
        finish with an event carrying none at all -- Anthropic sends input
        tokens on ``message_start``, output tokens on ``message_delta`` and
        then a bare ``message_stop`` -- and for those, reading only the last
        chunk finds nothing and the span is left with no tokens and no cost.

        An instrumentor that overrides this sets ``timing.usage`` to the
        usage dict assembled so far; the finalizer prefers it over the last
        chunk. Leaving ``timing.usage`` None keeps the original behaviour
        exactly, which is why this is opt-in per provider.
        """

    def _measure_parsed_stream(
        self, span, parsed, start_time: float, model: str, emit_measurements: bool = True
    ):
        """Wrap whatever a raw response's ``parse()`` handed back."""
        if hasattr(parsed, "__aiter__"):
            return self._wrap_async_streaming_response(
                parsed, span, start_time, model, emit_measurements
            )
        if hasattr(parsed, "__iter__") and not isinstance(parsed, (str, bytes, dict, list, tuple)):
            return self._wrap_streaming_response(parsed, span, start_time, model, emit_measurements)

        # Not a stream after all. Close the span rather than leak it, and leave
        # off the streaming attributes -- nothing was measured.
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.OK))
        span.end()
        server_metrics = get_server_metrics()
        if server_metrics:
            server_metrics.decrement_requests_running()
        return parsed

    def _hook_raw_response_stream(
        self, span, result, start_time: float, model: str, emit_measurements: bool = True
    ) -> bool:
        """Measure a stream that only exists once ``.parse()`` is called.

        A caller that wants the provider's response headers -- litellm does, to
        read rate limits -- reaches the OpenAI SDK through
        ``with_raw_response.create``. That returns a LegacyAPIResponse (sync
        ``parse()``) or AsyncAPIResponse (``async parse()``) rather than the
        stream, so at await time there is nothing iterable to detect and
        streaming looks exactly like a buffered call. Deferring to ``parse()``
        is the only point where the stream becomes observable.

        Returns True when the hook is installed, in which case the caller
        returns the raw response unchanged and must NOT end the span.
        """
        parse = getattr(result, "parse", None)
        if not callable(parse):
            return False

        if inspect.iscoroutinefunction(parse):

            async def hooked_parse(*args, **kwargs):
                return self._measure_parsed_stream(
                    span, await parse(*args, **kwargs), start_time, model, emit_measurements
                )

        else:

            def hooked_parse(*args, **kwargs):
                return self._measure_parsed_stream(
                    span, parse(*args, **kwargs), start_time, model, emit_measurements
                )

        try:
            result.parse = hooked_parse
        except (AttributeError, TypeError):
            # Frozen or slot-based response object; nothing to hook.
            return False
        return True

    def _install_stream_measurement(
        self,
        span,
        result,
        start_time: float,
        model: str,
        kwargs,
        emit_measurements: bool = True,
        sync_call: bool = False,
    ):
        """Arrange for a streamed call to be measured. Returns ``(handled, value)``.

        Instrumentors that build their own spans instead of going through
        :meth:`create_span_wrapper` call this straight after invoking the SDK,
        so every provider reports TTFT/TPOT the same way. When ``handled`` is
        True the caller must return ``value`` and must NOT end the span -- the
        stream wrapper owns it from that point and ends it when iteration
        finishes.

        ``emit_measurements=False`` keeps the span open for the whole stream --
        so its duration is right and it parents the inner span -- while leaving
        the numbers to an inner provider span that already recorded them.
        """
        if not kwargs.get("stream", False):
            return False, None

        handled = False
        # Some stream objects are iterable both ways (litellm's is). One that came
        # back from a synchronous call will be read with a plain `for`.
        if sync_call and hasattr(result, "__next__") and hasattr(result, "__aiter__"):
            value = self._wrap_streaming_response(
                result, span, start_time, model, emit_measurements
            )
            handled = True
        elif hasattr(result, "__aiter__"):
            value = self._wrap_async_streaming_response(
                result, span, start_time, model, emit_measurements
            )
            handled = True
        # A buffered response is an object, not an iterator. Strings, dicts and
        # lists are excluded deliberately: iterating them would "measure"
        # characters or keys and invent a TTFT out of nothing.
        elif hasattr(result, "__iter__") and not isinstance(
            result, (str, bytes, dict, list, tuple)
        ):
            value = self._wrap_streaming_response(
                result, span, start_time, model, emit_measurements
            )
            handled = True
        elif self._hook_raw_response_stream(span, result, start_time, model, emit_measurements):
            value = result
            handled = True

        if not handled:
            return False, None

        if emit_measurements:
            # Tell any wrapper further out (litellm around a provider SDK, a
            # framework around either) that this request is already accounted
            # for. Marked when the stream is handed over, not when it ends: the
            # enclosing call may return before the stream has been read.
            _note_inner_measurement()
            _mark_usage_recorded(span)
        return True, value

    def _record_time_to_first_token(self, span, ttft: float, model: str) -> None:
        """Publish a measured TTFT under both attribute spellings.

        The upstream semconv name is what downstream platforms look for;
        ``gen_ai.server.ttft`` is this library's older abbreviation and stays
        for consumers already reading it. Every streaming path goes through
        here so all providers report the same names.
        """
        span.set_attribute(SC.GEN_AI_SERVER_TIME_TO_FIRST_TOKEN, ttft)
        span.set_attribute(SC.GEN_AI_SERVER_TTFT, ttft)
        metric_attrs = {"model": model, "operation": _span_name(span)}
        if self.ttft_histogram:
            self.ttft_histogram.record(ttft, metric_attrs)
        if self.time_to_first_token_histogram:
            self.time_to_first_token_histogram.record(ttft, metric_attrs)
        logger.debug("TTFT for %s: %.3fs", model, ttft)

    def _record_time_per_output_token(
        self, span, ttft: Optional[float], duration: float, output_tokens, model: str
    ) -> None:
        """Emit TPOT, or record why it could not be measured.

        TPOT is generation speed with the first token's wait removed, so it
        needs both a measured TTFT and a real output-token count. When the
        provider sends no usage we emit nothing at all: a zero is
        indistinguishable from an instant response once averaged, and dividing
        by the chunk count instead would be wrong by whatever the provider's
        chunking happens to be. An absent attribute can be reported as "not
        measured"; a fabricated number cannot be challenged.
        """
        if ttft is None:
            return

        if not isinstance(output_tokens, (int, float)) or output_tokens <= 0:
            span.set_attribute(
                SC.GEN_AI_STREAMING_TPOT_UNAVAILABLE_REASON, "output_token_count_unavailable"
            )
            return

        # Per OTel GenAI conventions: the first token is excluded, since its
        # cost is already reported as TTFT.
        tpot = max(duration - ttft, 0.0) / max(int(output_tokens) - 1, 1)
        span.set_attribute(SC.GEN_AI_SERVER_TIME_PER_OUTPUT_TOKEN, tpot)
        if self.time_per_output_token_histogram:
            self.time_per_output_token_histogram.record(
                tpot, {"model": model, "operation": _span_name(span)}
            )

    def _fail_measured_stream(
        self, span, timing: "_StreamTiming", model: str, error: Exception
    ) -> None:
        """Close the span of a stream that raised, keeping what was measured of it.

        The chunks that did arrive were timed. Their time between chunks used to
        reach the histogram as they came; it must not be lost now that it is
        recorded at the end.
        """
        try:
            self._record_mean_time_between_tokens(span, timing, model)
        except Exception as e:  # noqa: BLE001 - the stream's own error is the one to report
            logger.debug("Failed to record time between tokens: %s", e)
        self._fail_stream(span, error)

    def _fail_stream(self, span, error: Exception) -> None:
        """Close a span whose stream raised part-way through iteration."""
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.ERROR, str(error)))
        span.record_exception(error)
        span.end()  # Close the span even on error

        # Decrement server metrics: running requests counter (streaming error)
        server_metrics = get_server_metrics()
        if server_metrics:
            server_metrics.decrement_requests_running()
            logger.debug("Decremented running requests (streaming error)")

        if self.error_counter:
            self.error_counter.add(
                1, {"operation": _span_name(span), "error_type": type(error).__name__}
            )
        logger.warning("Error in streaming wrapper: %s", error)

    def _finalize_stream(self, span, timing: "_StreamTiming", model: str) -> None:
        """Record duration, latency, usage and cost once a stream is exhausted."""
        from opentelemetry.trace import Status, StatusCode

        duration = time.time() - timing.start_time

        if not timing.emit_measurements:
            # An inner provider span recorded this request. This span exists to
            # parent it and to cover the true duration of the call; repeating
            # the tokens, cost and latency here would count one request twice.
            span.set_status(Status(StatusCode.OK))
            span.end()
            server_metrics = get_server_metrics()
            if server_metrics:
                server_metrics.decrement_requests_running()
            logger.debug(
                "Streaming completed in %.3fs; measurements left to the inner span", duration
            )
            return

        if self.latency_histogram:
            self.latency_histogram.record(duration, {"operation": _span_name(span)})
        self._record_mean_time_between_tokens(span, timing, model)
        span.set_attribute("gen_ai.streaming.token_count", timing.chunk_count)

        # Output-token count for TPOT. Stays None unless the provider actually
        # reported usage, so TPOT is omitted rather than guessed.
        output_tokens = None
        last_chunk = timing.last_chunk

        # Extract usage and calculate cost. Most providers repeat the full
        # usage on the final chunk, so that chunk is enough. A provider that
        # splits usage across events accumulates it via
        # _accumulate_stream_usage instead, and what it assembled wins -- the
        # final chunk in that case carries nothing to read.
        try:
            usage = timing.usage
            if usage is None and last_chunk is not None:
                usage = self._extract_usage(last_chunk)
            if usage and _descendant_recorded_usage(span):
                _leave_usage_to_descendant(span)
                usage = None
                last_chunk = None
            if usage is not None or last_chunk is not None:
                if usage and isinstance(usage, dict):
                    # Record token usage metrics and calculate cost
                    # This will set span attributes and record cost metrics
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                    output_tokens = completion_tokens

                    # Record token counts. Attribute naming (current vs
                    # superseded) is decided centrally by
                    # _set_token_usage_attributes.
                    self._set_token_usage_attributes(
                        span,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )

                    if isinstance(prompt_tokens, (int, float)) and prompt_tokens > 0:
                        if self.token_counter:
                            self.token_counter.add(
                                prompt_tokens,
                                {"token_type": "prompt", "operation": _span_name(span)},
                            )
                        # Record histogram for distribution analysis
                        if self.prompt_tokens_histogram:
                            self.prompt_tokens_histogram.record(
                                int(prompt_tokens), {"model": model, "operation": _span_name(span)}
                            )

                    if isinstance(completion_tokens, (int, float)) and completion_tokens > 0:
                        if self.token_counter:
                            self.token_counter.add(
                                completion_tokens,
                                {"token_type": "completion", "operation": _span_name(span)},
                            )
                        # Record histogram for distribution analysis
                        if self.completion_tokens_histogram:
                            self.completion_tokens_histogram.record(
                                int(completion_tokens),
                                {"model": model, "operation": _span_name(span)},
                            )

                    if isinstance(total_tokens, (int, float)) and total_tokens > 0:
                        span.set_attribute("gen_ai.usage.total_tokens", int(total_tokens))

                    # Calculate and record cost if enabled
                    if self.config and self.config.enable_cost_tracking:
                        try:
                            # Get call_type from span attributes or default to "chat"
                            call_type = (getattr(span, "attributes", None) or {}).get(
                                "gen_ai.request.type", "chat"
                            )

                            # Use granular cost calculation for chat requests
                            if call_type == "chat":
                                costs = self.cost_calculator.calculate_granular_cost(
                                    model, usage, call_type
                                )
                                total_cost = costs["total"]

                                # Record total cost
                                if total_cost > 0:
                                    if self.cost_counter:
                                        self.cost_counter.add(total_cost, {"model": str(model)})
                                    span.set_attribute("gen_ai.usage.cost.total", total_cost)
                                    logger.debug(f"Streaming cost: {total_cost} USD")

                                # See the non-streaming path: a 0.0 cost has
                                # to be distinguishable from an unpriced one.
                                span.set_attribute(
                                    "gen_ai.usage.cost.pricing_source",
                                    self.cost_calculator.pricing_source(model, call_type),
                                )

                                # Record granular costs
                                if costs["prompt"] > 0:
                                    if self.prompt_cost_counter:
                                        self.prompt_cost_counter.add(
                                            costs["prompt"], {"model": str(model)}
                                        )
                                    span.set_attribute("gen_ai.usage.cost.prompt", costs["prompt"])

                                if costs["completion"] > 0:
                                    if self.completion_cost_counter:
                                        self.completion_cost_counter.add(
                                            costs["completion"], {"model": str(model)}
                                        )
                                    span.set_attribute(
                                        "gen_ai.usage.cost.completion", costs["completion"]
                                    )

                                if costs["reasoning"] > 0:
                                    if self.reasoning_cost_counter:
                                        self.reasoning_cost_counter.add(
                                            costs["reasoning"], {"model": str(model)}
                                        )
                                    span.set_attribute(
                                        "gen_ai.usage.cost.reasoning", costs["reasoning"]
                                    )

                                if costs["cache_read"] > 0:
                                    if self.cache_read_cost_counter:
                                        self.cache_read_cost_counter.add(
                                            costs["cache_read"], {"model": str(model)}
                                        )
                                    span.set_attribute(
                                        "gen_ai.usage.cost.cache_read", costs["cache_read"]
                                    )

                                if costs["cache_write"] > 0:
                                    if self.cache_write_cost_counter:
                                        self.cache_write_cost_counter.add(
                                            costs["cache_write"], {"model": str(model)}
                                        )
                                    span.set_attribute(
                                        "gen_ai.usage.cost.cache_write", costs["cache_write"]
                                    )
                            else:
                                # For non-chat requests, use simple cost calculation
                                cost = self.cost_calculator.calculate_cost(model, usage, call_type)
                                if cost and cost > 0:
                                    if self.cost_counter:
                                        self.cost_counter.add(cost, {"model": str(model)})
                                    span.set_attribute("gen_ai.usage.cost.total", cost)
                        except Exception as e:
                            logger.warning("Failed to calculate cost for streaming response: %s", e)
                else:
                    logger.debug("No usage information found in streaming response")
        except Exception as e:
            logger.warning("Failed to extract usage from streaming response: %s", e)

        self._record_time_per_output_token(span, timing.ttft, duration, output_tokens, model)

        span.set_status(Status(StatusCode.OK))
        span.end()  # Close the span when streaming completes

        # Decrement server metrics: running requests counter (streaming success)
        server_metrics = get_server_metrics()
        if server_metrics:
            server_metrics.decrement_requests_running()
            logger.debug("Decremented running requests (streaming success)")

        logger.debug("Streaming completed: %d chunks in %.3fs", timing.chunk_count, duration)

    # Phase 4.2: RAG/Embedding Helper Methods
    def _add_generic_embedding_content(self, span, result, request_kwargs: dict):
        """Capture bounded embedding text for providers without custom content hooks."""
        value = request_kwargs.get("input")
        if value is None:
            value = request_kwargs.get(
                "texts", request_kwargs.get("text", request_kwargs.get("prompt"))
            )
        if isinstance(value, dict):
            # Replicate's `run(model, input={...})` nests the actual text one
            # level deeper than every other provider's flat `input=`/`texts=`
            # kwarg, since `input` there is the whole model payload.
            value = value.get("text") or value.get("texts") or value.get("prompt")
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
        if value is None or not isinstance(value, str):
            return
        attributes = getattr(span, "attributes", {}) or {}
        model = attributes.get("gen_ai.request.model", "unknown")
        vector = None
        try:
            data = result.get("data") if isinstance(result, dict) else getattr(result, "data", None)
            if data:
                first = data[0]
                vector = (
                    first.get("embedding")
                    if isinstance(first, dict)
                    else getattr(first, "embedding", None)
                )
            if vector is None:
                vector = (
                    result.get("embedding")
                    if isinstance(result, dict)
                    else getattr(result, "embedding", None)
                )
        except (IndexError, AttributeError, TypeError):
            vector = None
        self.add_embedding_attributes(span, str(model), value, vector=vector)

    def add_embedding_attributes(
        self, span, model: str, input_text: str, vector: Optional[List[float]] = None
    ):
        """Add embedding-specific attributes to a span.

        Args:
            span: The OpenTelemetry span
            model: The embedding model name
            input_text: The text being embedded (will be truncated to 500 chars)
            vector: Optional embedding vector (use with caution - can be large!)
        """
        span.set_attribute("embedding.model_name", model)
        span.set_attribute("embedding.text", input_text[:500])  # Truncate to avoid large spans

        if vector and self.config and getattr(self.config, "capture_embedding_vectors", False):
            # Only capture vectors if explicitly enabled (they can be very large)
            span.set_attribute("embedding.vector", json.dumps(vector))
            span.set_attribute("embedding.vector.dimension", len(vector))

    def add_retrieval_attributes(
        self,
        span,
        documents: List[Dict[str, Any]],
        query: Optional[str] = None,
        max_docs: int = 5,
    ):
        """Add retrieval/RAG-specific attributes to a span.

        Args:
            span: The OpenTelemetry span
            documents: List of retrieved documents. Each dict should have:
                - id: Document identifier
                - score: Relevance score
                - content: Document content
                - metadata: Optional metadata dict
            query: Optional query string
            max_docs: Maximum number of documents to include in attributes (default: 5)
        """
        if query:
            span.set_attribute("retrieval.query", query[:500])  # Truncate

        # Limit to first N documents to avoid attribute explosion
        for i, doc in enumerate(documents[:max_docs]):
            prefix = f"retrieval.documents.{i}.document"

            if "id" in doc:
                span.set_attribute(f"{prefix}.id", str(doc["id"]))
            if "score" in doc:
                span.set_attribute(f"{prefix}.score", float(doc["score"]))
            if "content" in doc:
                # Truncate content to avoid large attributes
                content = str(doc["content"])[:500]
                span.set_attribute(f"{prefix}.content", content)

            # Add metadata if present
            if "metadata" in doc and isinstance(doc["metadata"], dict):
                for key, value in doc["metadata"].items():
                    # Flatten metadata, limit key names to avoid explosion
                    safe_key = str(key)[:50]  # Limit key length
                    safe_value = str(value)[:200]  # Limit value length
                    span.set_attribute(f"{prefix}.metadata.{safe_key}", safe_value)

        span.set_attribute("retrieval.document_count", len(documents))

    def add_retrieval_quality_attributes(self, span, **kwargs):
        """Add optional, additive quality signals to a retrieval span.

        Supported keyword arguments are ``embedding_model``,
        ``index_embedding_model``, ``embedding_dim``, ``score_floor``,
        ``distance``, ``scores`` (an iterable of numeric result scores),
        ``score_max``, ``score_min``, ``score_mean``, ``score_margin``,
        ``corpus_version``, ``context_tokens_est``, ``context_truncated``,
        and ``answer_refused``.  The helper never emits ``top_k`` or result
        count; callers should continue using ``db.vector.top_k`` and
        ``retrieval.document_count`` for those values.
        """
        embedding_model = kwargs.get("embedding_model")
        index_embedding_model = kwargs.get("index_embedding_model")
        if embedding_model is not None:
            span.set_attribute("rag.embedding.model", str(embedding_model))
        if index_embedding_model is not None:
            span.set_attribute("rag.embedding.index_model", str(index_embedding_model))
        if embedding_model is not None and index_embedding_model is not None:
            span.set_attribute(
                "rag.embedding.model_match", str(embedding_model) == str(index_embedding_model)
            )

        scalar_attributes = {
            "embedding_dim": "rag.embedding.dim",
            "score_floor": "rag.search.score_floor",
            "distance": "rag.search.distance",
            "score_max": "rag.result.score_max",
            "score_min": "rag.result.score_min",
            "score_mean": "rag.result.score_mean",
            "score_margin": "rag.result.score_margin",
            "corpus_version": "rag.corpus.version",
            "context_tokens_est": "rag.context.tokens_est",
            "context_truncated": "rag.context.truncated",
            "answer_refused": "rag.answer.refused",
        }
        for argument, attribute in scalar_attributes.items():
            value = kwargs.get(argument)
            if value is None:
                continue
            if argument in {"embedding_dim", "context_tokens_est"}:
                value = int(value)
            elif argument in {
                "score_floor",
                "score_max",
                "score_min",
                "score_mean",
                "score_margin",
            }:
                value = float(value)
            elif argument in {"context_truncated", "answer_refused"}:
                value = bool(value)
            else:
                value = str(value)
            span.set_attribute(attribute, value)

        scores = kwargs.get("scores")
        if scores is not None:
            try:
                numeric_scores = [float(score) for score in scores]
                if numeric_scores:
                    derived = {
                        "rag.result.score_max": max(numeric_scores),
                        "rag.result.score_min": min(numeric_scores),
                        "rag.result.score_mean": sum(numeric_scores) / len(numeric_scores),
                    }
                    if len(numeric_scores) >= 2:
                        ordered = sorted(numeric_scores, reverse=True)
                        derived["rag.result.score_margin"] = ordered[0] - ordered[1]
                    explicit_arguments = {
                        "rag.result.score_max": "score_max",
                        "rag.result.score_min": "score_min",
                        "rag.result.score_mean": "score_mean",
                        "rag.result.score_margin": "score_margin",
                    }
                    for attribute, value in derived.items():
                        if kwargs.get(explicit_arguments[attribute]) is None:
                            span.set_attribute(attribute, value)
            except (TypeError, ValueError):
                logger.debug("Could not derive retrieval score statistics from %r", scores)

    @staticmethod
    def record_degradation(
        span,
        component: str,
        from_value: str,
        to_value: str,
        reason: str,
        recoverable: bool,
    ):
        """Record a non-error capability downgrade on an active span."""
        span.add_event(
            "gen_ai.degraded",
            attributes={
                "gen_ai.degraded.component": str(component),
                "gen_ai.degraded.from": str(from_value),
                "gen_ai.degraded.to": str(to_value),
                "gen_ai.degraded.reason": str(reason),
                "gen_ai.degraded.recoverable": bool(recoverable),
            },
        )

    @abstractmethod
    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Abstract method to extract token usage information from a function result.

        Subclasses must implement this to parse the specific library's response object
        and return a dictionary containing 'prompt_tokens', 'completion_tokens',
        and optionally 'total_tokens'.

        Args:
            result: The return value of the instrumented function.

        Returns:
            Optional[Dict[str, int]]: A dictionary with token counts, or None if usage cannot be extracted.
        """

    def _estimate_usage(self, result: Any, request_kwargs: dict) -> Optional[Dict[str, int]]:
        """Best-effort token estimate for providers whose response omits usage.

        Default returns None (no estimate). Subclasses MAY override this to
        approximate token counts from request/response text length, audio
        duration, image dimensions, etc., when the canonical `_extract_usage`
        path returns None. The base recorder will set
        `gen_ai.usage.token_count_estimated=true` on the span when this path
        is used, so consumers can distinguish exact from estimated counts.

        Args:
            result: The return value of the instrumented function.
            request_kwargs: The original request kwargs.

        Returns:
            Optional[Dict[str, int]]: Estimated token counts, or None.
        """
        return None
