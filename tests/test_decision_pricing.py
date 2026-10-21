"""Decision models get their own cost calculation, and never fall back to chat pricing."""

import json
from unittest.mock import MagicMock

import pytest

from genai_otel.cost_calculator import CostCalculator
from genai_otel.cost_enriching_exporter import CostEnrichingSpanExporter
from genai_otel.cost_enrichment_processor import CostEnrichmentSpanProcessor


@pytest.fixture(scope="module")
def calc():
    return CostCalculator()


USAGE = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000, "total_tokens": 2_000_000}


class TestShippedDecisionSection:
    @pytest.mark.parametrize(
        "model,per_million_input",
        [
            ("gpt-6-luna", 0.10),
            ("openai/gpt-6-luna-decisions", 0.10),
            ("d1", 0.02),
            ("d1-ultra", 0.042),
        ],
    )
    def test_input_only_prices_from_the_spec(self, calc, model, per_million_input):
        # Output is free for every verified decision provider.
        assert calc.calculate_cost(model, USAGE, "decision") == pytest.approx(per_million_input)
        assert calc.pricing_source(model, "decision") == "table"

    def test_decision_pricing_is_not_the_chat_price_of_the_same_name(self, calc):
        chat = calc.calculate_cost("gpt-6-luna", USAGE, "chat")
        decision = calc.calculate_cost("gpt-6-luna", USAGE, "decision")
        assert chat > decision == pytest.approx(0.10)

    @pytest.mark.parametrize("model", ["clef", "clef-flash", "clef-omni", "Cloudflare/clef-flash"])
    def test_cloudflare_clef_is_not_priced_yet(self, calc, model):
        assert calc.pricing_source(model, "decision") == "unpriced"
        assert calc.calculate_cost(model, USAGE, "decision") == 0.0

    @pytest.mark.parametrize(
        "model,key",
        [
            ("gpt-6-luna-2026-10-01", "gpt-6-luna"),
            ("d1-ultra-0925", "d1-ultra"),
            ("liquid/d1", "d1"),
            ("LiquidAI/D1-Ultra", "d1-ultra"),
            ("GPT-6-LUNA", "gpt-6-luna"),
        ],
    )
    def test_dated_and_routed_variants(self, calc, model, key):
        assert calc._normalize_model_name(model, "decision") == key

    @pytest.mark.parametrize(
        "model",
        [
            "ibm-granite/granite-switch-4.1-8b",
            "my-decider-8b",
            "d10",
            "",
        ],
    )
    def test_self_hosted_and_lookalikes_are_unpriced(self, calc, model):
        assert calc.pricing_source(model, "decision") == "unpriced"

    def test_no_fallback_to_chat_or_to_a_size_estimate(self, calc):
        # A chat lookup would estimate an 8b model from its name.
        assert calc.pricing_source("my-decider-8b", "chat") == "estimated"
        assert calc.pricing_source("my-decider-8b", "decision") == "unpriced"
        assert calc._section_for("decision") == "decision"

    def test_no_decision_section_loaded_still_never_reads_chat(self):
        calc = CostCalculator()
        calc.pricing_data = {"chat": {"gpt-6-luna": {"promptPrice": 1.0, "completionPrice": 1.0}}}
        assert calc._section_for("decision") == "decision"
        assert calc.pricing_source("gpt-6-luna", "decision") == "unpriced"
        assert calc.calculate_cost("gpt-6-luna", USAGE, "decision") == 0.0

    def test_cache_reads_are_free_where_the_entry_says_so(self, calc):
        usage = {"prompt_tokens": 1000, "cache_read_input_tokens": 900, "completion_tokens": 0}
        assert calc.calculate_cost("gpt-6-luna", usage, "decision") == pytest.approx(
            100 / 1000 * 0.0001
        )
        # An entry without a cache price bills cached input as input.
        assert calc.calculate_cost("d1", usage, "decision") == pytest.approx(1000 / 1000 * 2e-05)

    def test_cache_write_priced_by_the_entry(self, calc):
        usage = {"prompt_tokens": 1000, "cache_creation_input_tokens": 1000}
        assert calc.calculate_cost("gpt-6-luna", usage, "decision") == 0.0

    def test_a_non_dict_entry(self):
        calc = CostCalculator()
        calc.pricing_data = {"decision": {"odd": 0.5}}
        assert calc.calculate_cost("odd", USAGE, "decision") == 0.0

    def test_granular_cost_for_decision_is_total_only(self, calc):
        costs = calc.calculate_granular_cost("gpt-6-luna", USAGE, "decision")
        assert costs["total"] == pytest.approx(0.10)
        assert costs["prompt"] == 0.0


class TestCustomPricing:
    def test_a_decision_section_is_accepted(self):
        calc = CostCalculator(
            custom_pricing_json=json.dumps(
                {"decision": {"on-prem-decider": {"promptPrice": 0.00005, "completionPrice": 0}}}
            )
        )
        assert calc.calculate_cost("on-prem-decider", USAGE, "decision") == pytest.approx(0.05)
        assert calc.pricing_source("on-prem-decider", "chat") == "unpriced"

    def test_an_unknown_category_is_still_refused(self, caplog):
        calc = CostCalculator(custom_pricing_json=json.dumps({"rerank": {"m": {"promptPrice": 1}}}))
        assert "rerank" not in calc.pricing_data
        assert "decision" in caplog.text


def _span(attributes, name="s"):
    span = MagicMock()
    span.name = name
    span.attributes = attributes
    span.context = span.parent = span.status = span.resource = None
    span.events = span.links = ()
    span.kind = None
    span.start_time = span.end_time = 0
    span.instrumentation_scope = None
    return span


class TestEnrichersNeverChatPriceADecision:
    def _exporter(self):
        exporter = CostEnrichingSpanExporter(MagicMock())
        exporter.cost_calculator = CostCalculator()
        return exporter

    def test_an_unpriced_decision_span_stays_unpriced(self):
        attrs = {
            "openinference.span.kind": "DECISION",
            "gen_ai.request.model": "ibm-granite/granite-switch-4.1-8b",
            "gen_ai.usage.input_tokens": 500,
        }
        result = self._exporter()._enrich_span(_span(attrs))
        assert "gen_ai.usage.cost.total" not in result.attributes

    def test_an_openinference_only_decision_span_is_priced_from_decision_fields(self):
        attrs = {
            "openinference.span.kind": "DECISION",
            "decision.model_name": "gpt-6-luna",
            "decision.token_count.input": 403,
            "decision.token_count.output": 0,
        }
        enriched = self._exporter()._enrich_span(_span(attrs)).attributes
        assert enriched["gen_ai.usage.cost.total"] == pytest.approx(403 * 0.1 / 1e6)
        assert enriched["gen_ai.usage.cost.decision"] == enriched["gen_ai.usage.cost.total"]

    def test_the_processor_hook_reads_decision_spans_too(self, caplog):
        processor = CostEnrichmentSpanProcessor(CostCalculator())
        caplog.set_level("INFO")
        processor.on_end(
            _span(
                {
                    "openinference.span.kind": "DECISION",
                    "decision.model_name": "gpt-6-luna",
                    "decision.token_count.input": 403,
                }
            )
        )
        assert "gpt-6-luna" in caplog.text
        caplog.clear()
        processor.on_end(
            _span(
                {
                    "openinference.span.kind": "DECISION",
                    "gen_ai.request.model": "my-decider-8b",
                    "gen_ai.usage.input_tokens": 100,
                }
            )
        )
        assert "Cost calculated" not in caplog.text
