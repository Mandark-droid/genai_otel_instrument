"""Decision spans for in-process decision models and rules engines.

A decision model run in-process (a ModernBERT router, a ``transformers``
checkpoint, ``joint_schema_model.systemone``) makes no network call for this
library to wrap. ``decision_span`` records the call by hand, as the same
OpenInference ``DECISION`` span the instrumented call paths emit, under the same
content-capture rule::

    from genai_otel import decision_span

    questions = {"next_tool": {"type": "choice", "criteria": {"search": None, "refund": None}}}
    with decision_span(model="laya-router-421m", questions=questions, state=message) as d:
        result = router(message)
        d.record({"next_tool": {"type": "choice", "choice": result.label,
                                "confidence": result.confidence,
                                "probabilities": result.probabilities}})

``questions`` and ``answers`` may be in either protocol's shape: a SystemOne
mapping of names to questions / answers, or an OpenAI Decisions list. ``usage``
(``{"input_tokens": ..., "output_tokens": ...}``) is optional, and when it is
left out the span records no tokens rather than zero.

As a decorator, the function's return value is recorded when it is a response
(it has ``answers``) or the answers themselves (a list or a mapping)::

    @decision_span(model="clef-flash", questions=QUESTIONS)
    def classify(state): ...

With content capture off the span still says that a decision happened, by
which model, how many questions of which types and how long it took; the
answers, confidences and question names are recorded only with content
capture on (``GENAI_ENABLE_CONTENT_CAPTURE=true``).
"""

import functools
import inspect
import threading
from collections.abc import Mapping
from typing import Any, Callable, Optional

from .config import OTelConfig
from .instrumentors.decision import (
    PROTOCOL_MANUAL,
    BaseDecisionInstrumentor,
    DecisionRecorder,
    DecisionRequest,
    _get,
    questions_of,
)

__all__ = ["decision_span", "DecisionRecorder"]

_LOCK = threading.Lock()
_CONFIG: Optional[OTelConfig] = None
_INSTRUMENTOR: Optional["_ManualDecisionInstrumentor"] = None


class _ManualDecisionInstrumentor(BaseDecisionInstrumentor):
    """Writes the spans ``decision_span`` opens. Not an SDK instrumentor."""

    def instrument(self, config: OTelConfig) -> None:
        self.config = config
        self._instrumented = True


def configure(config: OTelConfig) -> None:
    """Use ``config`` for manual decision spans. Called by ``instrument()``."""
    global _CONFIG
    with _LOCK:
        _CONFIG = config
        if _INSTRUMENTOR is not None:
            _INSTRUMENTOR.instrument(config)


def _instrumentor() -> _ManualDecisionInstrumentor:
    global _INSTRUMENTOR
    with _LOCK:
        if _INSTRUMENTOR is None:
            _INSTRUMENTOR = _ManualDecisionInstrumentor()
            # Before instrument() has run, the settings come from the environment,
            # so content capture is still whatever the operator configured.
            _INSTRUMENTOR.instrument(_CONFIG if _CONFIG is not None else OTelConfig())
        return _INSTRUMENTOR


class decision_span:  # pylint: disable=invalid-name - used like a function
    """Context manager and decorator that records one decision as a DECISION span.

    Args:
        model: The decision model's name.
        questions: The questions asked, as a SystemOne mapping or an OpenAI list.
        provider: Who hosts the model, when anyone does. Omit for in-process.
        state: The input the model decided on. Recorded only with content capture on.
        system: The API shape, ``decision.system``. Defaults to ``"custom"``.
        name: The span name. Defaults to ``"decision"``.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        questions: Any = None,
        provider: Optional[str] = None,
        state: Any = None,
        system: str = "custom",
        name: str = "decision",
    ):
        self.model = model
        self.questions = questions
        self.provider = provider
        self.state = state
        self.system = system
        self.name = name
        self._context = None

    def _request(self) -> DecisionRequest:
        return DecisionRequest(
            protocol=PROTOCOL_MANUAL,
            system=self.system,
            operation="decision",
            model=self.model,
            provider=self.provider,
            questions=questions_of(self.questions),
            payload={"state": self.state, "model": self.model, "questions": self.questions},
        )

    def _eval_kwargs(self) -> dict:
        return {"state": self.state, "questions": self.questions} if self.state is not None else {}

    def __enter__(self) -> DecisionRecorder:
        self._context = _instrumentor()._run_decision_span(
            self.name, self._request(), self._eval_kwargs()
        )
        return self._context.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        context, self._context = self._context, None
        return context.__exit__(exc_type, exc_value, traceback)

    def _copy(self) -> "decision_span":
        return decision_span(
            model=self.model,
            questions=self.questions,
            provider=self.provider,
            state=self.state,
            system=self.system,
            name=self.name,
        )

    @staticmethod
    def _record_return(recorder: DecisionRecorder, result: Any) -> None:
        if recorder.has_result:
            return  # the function recorded it itself
        if _get(result, "answers") is not None:
            recorder.record_response(result)
        elif isinstance(result, (list, tuple, Mapping)):
            recorder.record(result)

    def __call__(self, func: Callable) -> Callable:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self._copy() as recorder:
                    result = await func(*args, **kwargs)
                    self._record_return(recorder, result)
                    return result

            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self._copy() as recorder:
                result = func(*args, **kwargs)
                self._record_return(recorder, result)
                return result

        return wrapper
