"""Per-invocation budget governance for agent spans.

Agent frameworks all expose a cap on how much work one invocation may do --
turns, iterations, recursion depth, LLM calls, sometimes tokens -- but the
GenAI conventions have no way to record it. `semantic-conventions-genai#425
<https://github.com/open-telemetry/semantic-conventions-genai/issues/425>`_
proposes four attributes and a utilization metric to close that gap; this
module implements them.

The value is in the pairing. An ``invoke_agent`` span already tells you an
agent ran and how long it took. With the cap *and* the consumption on the same
span, "which agents run near their limit" becomes one query instead of a manual
sum over child spans -- and, critically, it survives head sampling, which drops
the child inference spans that manual sum depends on.

Consumption is accumulated through a context-local stack rather than by reading
child spans back, for that same reason: a sampled-out child still records its
tokens here.

Known limitation: a call made on a thread that does not carry the context (a
bare ``threading.Thread``, rather than an asyncio task or a
``contextvars.copy_context().run`` worker) is not counted. This is consistent
with tracing itself -- a framework that spawns threads without propagating
context also loses the parent/child span relationship, so that call would not
appear beneath the agent span either way.
"""

import contextvars
import logging
import threading
from typing import Any, Dict, Optional

from .semconv import SemanticConvention as SC

logger = logging.getLogger(__name__)


class _BudgetFrame:
    """Consumption accumulated for one in-flight agent invocation."""

    __slots__ = ("tokens", "iterations", "_lock")

    def __init__(self) -> None:
        self.tokens = 0
        self.iterations = 0
        # Agent frameworks fan out for parallel tool and model calls. A worker
        # that carries the context (an asyncio task, or a thread started via
        # contextvars.copy_context().run) sees this same frame *object*, so the
        # counters are genuinely contended. A bare threading.Thread inherits no
        # context and its tokens are not counted -- see the module docstring.
        self._lock = threading.Lock()

    def add_tokens(self, count: int) -> None:
        with self._lock:
            self.tokens += count

    def add_iteration(self) -> None:
        with self._lock:
            self.iterations += 1


# Stack of in-flight agent invocations. A tuple, so that pushing inside a
# nested agent leaves the outer frame untouched for other tasks sharing the
# parent context. Nested agents each accumulate their own consumption; a
# supervisor delegating to sub-agents therefore reports its own direct usage,
# not its children's, which is what its own budget governs.
_BUDGET_STACK: contextvars.ContextVar = contextvars.ContextVar(
    "genai_otel_agent_budget_stack", default=()
)


def push_frame() -> Optional[object]:
    """Begin accumulating for a new agent invocation. Returns a reset token."""
    try:
        stack = _BUDGET_STACK.get()
        return _BUDGET_STACK.set(stack + (_BudgetFrame(),))
    except Exception:  # pragma: no cover - contextvars is near-total
        return None


def pop_frame(token: Optional[object]) -> Optional[_BudgetFrame]:
    """Finish the innermost invocation and return its accumulated frame."""
    if token is None:
        return None
    try:
        stack = _BUDGET_STACK.get()
        frame = stack[-1] if stack else None
        _BUDGET_STACK.reset(token)  # type: ignore[arg-type]
        return frame
    except Exception:  # pragma: no cover - defensive
        return None


def record_inference(tokens: int) -> None:
    """Attribute one inference call's tokens to the enclosing agent invocation.

    A no-op outside an agent span, which is the common case for a plain LLM
    call, so this stays cheap on the hot path.
    """
    if not isinstance(tokens, int) or tokens < 0:
        return
    try:
        stack = _BUDGET_STACK.get()
    except Exception:  # pragma: no cover - defensive
        return
    if not stack:
        return
    frame = stack[-1]
    frame.add_tokens(tokens)
    frame.add_iteration()


# ---------------------------------------------------------------------------
# Budget extraction
# ---------------------------------------------------------------------------

# How each framework spells its per-invocation caps. Sourced from the mapping
# table in semantic-conventions-genai#425 plus the SDK signatures. Iteration
# budgets are far more common than token budgets: most frameworks bound the
# loop, not the spend.
_ITERATION_BUDGET_KEYS = (
    "max_turns",  # OpenAI Agents SDK, AutoGen
    "max_iterations",  # LangChain
    "max_iter",  # CrewAI
    "recursion_limit",  # LangGraph
    "max_llm_calls",  # Google ADK
    "max_rounds",  # AutoGen group chat
)

_TOKEN_BUDGET_KEYS = (
    "max_tokens",  # CrewAI aggregate cap
    "token_budget",
    "max_total_tokens",
)


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = int(value)
    return value if value > 0 else None


def _lookup(sources: Any, keys: tuple) -> Optional[int]:
    """First positive integer found under ``keys`` across mappings/objects."""
    for source in sources:
        if source is None:
            continue
        for key in keys:
            if isinstance(source, dict):
                found = _positive_int(source.get(key))
            else:
                found = _positive_int(getattr(source, key, None))
            if found is not None:
                return found
    return None


def extract_budget_attributes(*sources: Any) -> Dict[str, int]:
    """Read configured budgets from any mix of kwargs dicts and agent objects.

    Frameworks put the cap in different places -- a kwarg on the run call, a
    field on the agent, an entry in a config dict -- so callers pass every
    plausible source and the first match wins.

    A framework's own cap is recorded as configured. It is never synthesised:
    #425 states explicitly that instrumentations MUST NOT derive a token budget
    by multiplying an iteration limit by a per-call ``max_tokens``, because
    those are different things and the product is a number nobody configured.
    """
    attrs: Dict[str, int] = {}

    iterations = _lookup(sources, _ITERATION_BUDGET_KEYS)
    if iterations is not None:
        attrs[SC.GEN_AI_AGENT_ITERATION_BUDGET] = iterations

    tokens = _lookup(sources, _TOKEN_BUDGET_KEYS)
    if tokens is not None:
        attrs[SC.GEN_AI_AGENT_TOKEN_BUDGET] = tokens

    return attrs


def consumption_attributes(frame: Optional[_BudgetFrame]) -> Dict[str, int]:
    """Consumption recorded during one agent invocation."""
    if frame is None:
        return {}
    attrs: Dict[str, int] = {}
    if frame.tokens > 0:
        attrs[SC.GEN_AI_AGENT_TOKEN_BUDGET_CONSUMED] = frame.tokens
    if frame.iterations > 0:
        attrs[SC.GEN_AI_AGENT_ITERATION_BUDGET_CONSUMED] = frame.iterations
    return attrs


def utilization(consumed: Optional[int], budget: Optional[int]) -> Optional[float]:
    """Consumed / budget, or None when the ratio would be meaningless.

    A zero or absent budget yields None rather than a division error or a
    misleading 0.0 -- "no budget configured" is not "zero percent used".
    """
    if not isinstance(consumed, int) or not isinstance(budget, int):
        return None
    if budget <= 0:
        return None
    return consumed / budget
