"""Every instrumentor records metrics, not only the first one constructed.

`_ensure_shared_metrics_created` stored the instruments with `cls._shared_* = ...`, so
they landed on whichever SUBCLASS happened to be constructed first; every other
instrumentor class read the base class's `None` and recorded nothing. In an application
using OpenAI and Anthropic, only one provider produced token, cost and latency metrics.
"""

import genai_otel.instrumentors.base as base
from genai_otel.instrumentors.base import BaseInstrumentor


class _First(BaseInstrumentor):
    def instrument(self, config):
        self.config = config

    def _extract_usage(self, result):
        return None


class _Second(BaseInstrumentor):
    def instrument(self, config):
        self.config = config

    def _extract_usage(self, result):
        return None


def test_second_instrumentor_class_gets_the_shared_instruments():
    for name in [n for n in vars(BaseInstrumentor) if n.startswith("_shared_")]:
        setattr(BaseInstrumentor, name, None)
    for cls in (_First, _Second):
        for name in [n for n in vars(cls) if n.startswith("_shared_")]:
            delattr(cls, name)
    base._SHARED_METRICS_CREATED = False

    first, second = _First(), _Second()

    for attr in (
        "request_counter",
        "token_counter",
        "latency_histogram",
        "cost_counter",
        "error_counter",
    ):
        assert getattr(first, attr) is not None, attr
        assert getattr(second, attr) is not None, f"{attr} missing on the second instrumentor class"
        assert getattr(second, attr) is getattr(
            first, attr
        ), f"{attr} must be the same shared instrument"
