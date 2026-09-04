"""Tests for per-invocation agent budget governance.

Implements semantic-conventions-genai#425. The interesting property is that
consumption is accumulated in a context-local frame rather than summed from
child spans: head sampling drops those children, so a sampled trace would
otherwise report a runaway agent as having consumed nothing.
"""

import contextvars
import threading
from types import SimpleNamespace

import pytest

from genai_otel.agent_budget import (
    consumption_attributes,
    extract_budget_attributes,
    pop_frame,
    push_frame,
    record_inference,
    utilization,
)


class TestBudgetExtraction:
    @pytest.mark.parametrize(
        "key,framework",
        [
            ("max_turns", "OpenAI Agents SDK / AutoGen"),
            ("max_iterations", "LangChain"),
            ("max_iter", "CrewAI"),
            ("recursion_limit", "LangGraph"),
            ("max_llm_calls", "Google ADK"),
        ],
    )
    def test_iteration_budget_from_each_framework_spelling(self, key, framework):
        attrs = extract_budget_attributes({key: 12})
        assert attrs["gen_ai.agent.iteration_budget"] == 12, framework

    def test_token_budget(self):
        attrs = extract_budget_attributes({"max_tokens": 4000})
        assert attrs["gen_ai.agent.token_budget"] == 4000

    def test_reads_from_objects_as_well_as_dicts(self):
        """CrewAI puts max_iter on the Agent, not in kwargs."""
        agent = SimpleNamespace(max_iter=5)
        assert extract_budget_attributes({}, agent)["gen_ai.agent.iteration_budget"] == 5

    def test_first_source_wins(self):
        """An explicit call kwarg overrides the agent's configured default."""
        agent = SimpleNamespace(max_iter=5)
        attrs = extract_budget_attributes({"max_iter": 20}, agent)
        assert attrs["gen_ai.agent.iteration_budget"] == 20

    def test_none_and_nonpositive_ignored(self):
        assert extract_budget_attributes({"max_turns": None}) == {}
        assert extract_budget_attributes({"max_turns": 0}) == {}
        assert extract_budget_attributes({"max_turns": -1}) == {}

    def test_bool_is_not_a_budget(self):
        assert extract_budget_attributes({"max_turns": True}) == {}

    def test_no_budget_is_synthesised_from_an_iteration_limit(self):
        """#425: MUST NOT derive a token budget by multiplying limits.

        max_tokens-per-call times max_turns is a number nobody configured.
        """
        attrs = extract_budget_attributes({"max_turns": 10})
        assert "gen_ai.agent.token_budget" not in attrs

    def test_missing_sources_are_safe(self):
        assert extract_budget_attributes(None, {}, SimpleNamespace()) == {}


class TestConsumptionAccounting:
    def test_tokens_and_iterations_accumulate(self):
        token = push_frame()
        try:
            record_inference(100)
            record_inference(250)
        finally:
            frame = pop_frame(token)

        attrs = consumption_attributes(frame)
        assert attrs["gen_ai.agent.token_budget.consumed"] == 350
        assert attrs["gen_ai.agent.iteration_budget.consumed"] == 2

    def test_outside_an_agent_is_a_noop(self):
        """A plain LLM call must not pay for this feature or crash."""
        record_inference(100)  # no frame pushed

    def test_nested_agents_account_separately(self):
        """A supervisor reports its own direct usage, not its children's.

        Its budget governs its own loop, so folding a delegated sub-agent's
        tokens into it would make the utilization ratio meaningless.
        """
        outer = push_frame()
        record_inference(10)

        inner = push_frame()
        record_inference(500)
        inner_frame = pop_frame(inner)

        record_inference(20)
        outer_frame = pop_frame(outer)

        assert consumption_attributes(inner_frame)["gen_ai.agent.token_budget.consumed"] == 500
        assert consumption_attributes(outer_frame)["gen_ai.agent.token_budget.consumed"] == 30

    def test_concurrent_inference_calls_are_not_lost(self):
        """Threads that carry the context share one frame, so counts must not race.

        Frameworks fan out for parallel tool and model calls. A worker started
        with `contextvars.copy_context().run(...)` sees the same frame *object*
        as its parent, so all 800 increments must land.
        """
        token = push_frame()
        errors = []

        def worker():
            try:
                for _ in range(100):
                    record_inference(1)
            except Exception as e:  # pragma: no cover
                errors.append(e)

        # The context must be copied on THIS thread; calling copy_context()
        # inside the worker would copy the worker's own empty context.
        threads = [
            threading.Thread(target=contextvars.copy_context().run, args=(worker,))
            for _ in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        frame = pop_frame(token)
        assert not errors
        assert consumption_attributes(frame)["gen_ai.agent.token_budget.consumed"] == 800

    def test_threads_without_the_context_are_not_counted(self):
        """A bare thread does not inherit contextvars, so its tokens are missed.

        This is the documented limitation, and it is consistent with tracing
        itself: a framework that spawns threads without propagating context
        also loses the parent/child span relationship, so such a call would not
        appear under the agent span either way.
        """
        token = push_frame()
        thread = threading.Thread(target=lambda: record_inference(500))
        thread.start()
        thread.join()
        frame = pop_frame(token)

        assert consumption_attributes(frame) == {}

    def test_negative_and_non_int_ignored(self):
        token = push_frame()
        record_inference(-5)
        record_inference("many")  # type: ignore[arg-type]
        frame = pop_frame(token)
        assert consumption_attributes(frame) == {}

    def test_pop_without_token_is_safe(self):
        assert pop_frame(None) is None
        assert consumption_attributes(None) == {}


class TestUtilization:
    def test_ratio(self):
        assert utilization(950, 1000) == pytest.approx(0.95)

    def test_absent_budget_is_not_zero_percent(self):
        """ "No budget configured" and "zero percent used" are different facts."""
        assert utilization(100, None) is None
        assert utilization(100, 0) is None

    def test_absent_consumption(self):
        assert utilization(None, 1000) is None

    def test_over_budget_is_reported_not_clamped(self):
        """An agent that blew its cap is exactly what alerting looks for."""
        assert utilization(1500, 1000) == pytest.approx(1.5)
