"""GENAI_CUSTOM_PRICING_JSON applies to every instrumentor.

Custom pricing was applied only by `_setup_config()`, which 38 of the 42 instrumentors
never call: their `instrument()` assigns `self.config` directly. OpenAI and Anthropic
were among them, so a custom-priced model through either was reported as unpriced.
"""

import json
from unittest.mock import MagicMock

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.base import BaseInstrumentor

CUSTOM = json.dumps({"chat": {"my-private-model": {"promptPrice": 1.0, "completionPrice": 2.0}}})


class _AssignsConfigDirectly(BaseInstrumentor):
    """Shaped like most real instrumentors: instrument() just stores the config."""

    def instrument(self, config):
        self.config = config
        self._instrumented = True

    def _extract_usage(self, result):
        return result["usage"]


def _run(config):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    inst = _AssignsConfigDirectly()
    inst.instrument(config)
    inst.tracer = provider.get_tracer("t")
    inst.cost_counter = MagicMock()
    call = inst.create_span_wrapper(
        "openai.chat.completion",
        extract_attributes=lambda i, a, k: {"gen_ai.request.model": "my-private-model"},
    )(
        lambda **kw: {
            "usage": {"prompt_tokens": 1000, "completion_tokens": 1000, "total_tokens": 2000}
        }
    )
    call()
    return dict(exporter.get_finished_spans()[0].attributes)


def test_custom_pricing_is_used_without_setup_config():
    attrs = _run(OTelConfig(custom_pricing_json=CUSTOM))
    assert attrs.get("gen_ai.usage.cost.total") == 3.0  # 1k x $1/1k + 1k x $2/1k
    assert attrs.get("gen_ai.usage.cost.pricing_source") == "table"


def test_without_custom_pricing_the_model_stays_unpriced():
    attrs = _run(OTelConfig())
    assert "gen_ai.usage.cost.total" not in attrs
