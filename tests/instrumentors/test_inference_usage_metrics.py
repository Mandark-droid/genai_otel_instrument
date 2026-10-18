"""Token usage under the current GenAI metric conventions.

semantic-conventions-genai#374 removed the ``gen_ai.client.token.usage`` histogram and the
``gen_ai.token.type`` attribute. Token usage is now five counters broken down by
``gen_ai.token.modality`` plus two per-operation histograms with no modality breakdown.
semantic-conventions-genai#602 adds that the counters carry only known, strictly positive
contributions: a zero or missing count is no measurement at all.

The library's own ``gen_ai.client.token.usage`` counter is still recorded alongside the
new instruments, because existing consumers read it.

These tests read what a real MeterProvider exported, not calls on a mock, so the
instrument names, types and units are the ones a backend receives.
"""

import time
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import Histogram, InMemoryMetricReader, Sum

import genai_otel.instrumentors.base as base
from genai_otel.config import OTelConfig
from genai_otel.instrumentors.base import BaseInstrumentor, _StreamTiming

INPUT = "gen_ai.client.inference.usage.input_tokens"
OUTPUT = "gen_ai.client.inference.usage.output_tokens"
CACHE_READ = "gen_ai.client.inference.usage.cache_read.input_tokens"
CACHE_WRITE = "gen_ai.client.inference.usage.cache_write.input_tokens"
REASONING = "gen_ai.client.inference.usage.reasoning.output_tokens"
OP_INPUT = "gen_ai.client.inference.operation.input_tokens"
OP_OUTPUT = "gen_ai.client.inference.operation.output_tokens"
COUNTERS = (INPUT, OUTPUT, CACHE_READ, CACHE_WRITE, REASONING)
HISTOGRAMS = (OP_INPUT, OP_OUTPUT)
LEGACY = "gen_ai.client.token.usage"


class _Instrumentor(BaseInstrumentor):
    def instrument(self, config):
        self.config = config

    def _extract_usage(self, result):
        return result.get("usage")


class _EstimatingInstrumentor(_Instrumentor):
    def _estimate_usage(self, result, request_kwargs):
        return {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}


class _Span:
    """The parts of a span the recorder touches, with attributes that can be read back."""

    def __init__(self, attributes=None, name="chat gpt-4o"):
        self.name = name
        self.attributes = dict(attributes or {})

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_status(self, *args, **kwargs):
        pass

    def end(self, *args, **kwargs):
        pass

    def is_recording(self):
        return True


CHAT_ATTRIBUTES = {
    "gen_ai.operation.name": "chat",
    "gen_ai.provider.name": "openai",
    "gen_ai.request.model": "gpt-4o",
    "gen_ai.response.model": "gpt-4o-2024-08-06",
    "server.address": "api.openai.com",
    "server.port": 443,
}


def _reset_shared_metrics():
    for name in [n for n in vars(BaseInstrumentor) if n.startswith("_shared_")]:
        setattr(BaseInstrumentor, name, None)
    base._SHARED_METRICS_CREATED = False


@pytest.fixture
def reader(monkeypatch):
    """Shared instruments created on a private MeterProvider whose output we can read."""
    _reset_shared_metrics()
    metric_reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[metric_reader])
    monkeypatch.setattr(
        base.metrics, "get_meter", lambda *args, **kwargs: provider.get_meter("test")
    )
    yield metric_reader
    _reset_shared_metrics()


def _instrumentor(cls=_Instrumentor):
    inst = cls()
    inst.instrument(OTelConfig(enable_cost_tracking=False))
    return inst


def _metrics(metric_reader):
    data = metric_reader.get_metrics_data()
    found = {}
    if data is None:
        return found
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                found[metric.name] = metric
    return found


def _points(metrics, name):
    metric = metrics.get(name)
    return list(metric.data.data_points) if metric else []


def _by_modality(metrics, name):
    return {p.attributes["gen_ai.token.modality"]: p.value for p in _points(metrics, name)}


def _record(metric_reader, usage, attributes=CHAT_ATTRIBUTES, cls=_Instrumentor):
    inst = _instrumentor(cls)
    inst._record_result_metrics(_Span(attributes), {"usage": usage}, time.time() - 0.1)
    return _metrics(metric_reader)


def test_counters_and_histograms_carry_the_upstream_names_types_and_units(reader):
    metrics = _record(reader, {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})

    for name in (INPUT, OUTPUT):
        metric = metrics[name]
        assert isinstance(metric.data, Sum), name
        assert metric.data.is_monotonic, name
        assert metric.unit == "{token}", name
    for name in HISTOGRAMS:
        metric = metrics[name]
        assert isinstance(metric.data, Histogram), name
        assert metric.unit == "{token}", name


def test_plain_usage_is_counted_under_unknown_modality(reader):
    metrics = _record(reader, {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})

    assert _by_modality(metrics, INPUT) == {"unknown": 100}
    assert _by_modality(metrics, OUTPUT) == {"unknown": 50}


def test_counter_attributes_are_the_metric_attributes_plus_modality(reader):
    metrics = _record(reader, {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})

    (point,) = _points(metrics, INPUT)
    assert dict(point.attributes) == {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "openai",
        "gen_ai.request.model": "gpt-4o",
        "gen_ai.response.model": "gpt-4o-2024-08-06",
        "server.address": "api.openai.com",
        "server.port": 443,
        "gen_ai.token.modality": "unknown",
    }


def test_histograms_record_one_total_per_operation_without_modality(reader):
    metrics = _record(
        reader,
        {
            "prompt_tokens": 300,
            "completion_tokens": 50,
            "total_tokens": 350,
            "text_input_tokens": 100,
            "image_input_tokens": 200,
        },
    )

    (point,) = _points(metrics, OP_INPUT)
    assert point.count == 1
    assert point.sum == 300
    assert "gen_ai.token.modality" not in point.attributes
    assert point.attributes["gen_ai.operation.name"] == "chat"
    (point,) = _points(metrics, OP_OUTPUT)
    assert point.count == 1
    assert point.sum == 50


def test_reported_modalities_partition_the_total(reader):
    metrics = _record(
        reader,
        {
            "prompt_tokens": 300,
            "completion_tokens": 50,
            "total_tokens": 350,
            "text_input_tokens": 100,
            "image_input_tokens": 200,
            "audio_output_tokens": 20,
        },
    )

    # Fully broken down: nothing is left over for `unknown`.
    assert _by_modality(metrics, INPUT) == {"text": 100, "image": 200}
    # Partly broken down: the rest is counted, under `unknown`, so the sum is the total.
    assert _by_modality(metrics, OUTPUT) == {"audio": 20, "unknown": 30}


def test_breakdown_larger_than_the_total_is_not_trusted(reader):
    metrics = _record(
        reader,
        {
            "prompt_tokens": 100,
            "completion_tokens": 0,
            "total_tokens": 100,
            "text_input_tokens": 150,
        },
    )

    assert _by_modality(metrics, INPUT) == {"unknown": 100}


def test_cache_reads_are_a_subset_of_input_when_the_provider_includes_them(reader):
    # OpenAI and Gemini count cached prompt tokens inside the prompt total.
    metrics = _record(
        reader,
        {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "cache_read_input_tokens": 80,
        },
    )

    assert _by_modality(metrics, INPUT) == {"unknown": 100}
    assert _by_modality(metrics, CACHE_READ) == {"unknown": 80}
    assert _points(metrics, CACHE_WRITE) == []


def test_cache_read_modality_breakdown_is_used(reader):
    metrics = _record(
        reader,
        {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "cache_read_input_tokens": 80,
            "audio_cache_read_input_tokens": 30,
        },
    )

    assert _by_modality(metrics, CACHE_READ) == {"audio": 30, "unknown": 50}


def test_cache_tokens_are_added_to_input_when_the_provider_excludes_them(reader):
    # Anthropic's input_tokens counts only uncached input; the conventions define the
    # input counter as including cached tokens, with cache reads and writes as subsets.
    metrics = _record(
        reader,
        {
            "prompt_tokens": 10,
            "completion_tokens": 40,
            "total_tokens": 50,
            "cache_read_input_tokens": 300,
            "cache_creation_input_tokens": 90,
            "prompt_tokens_excludes_cache": True,
        },
        attributes={**CHAT_ATTRIBUTES, "gen_ai.provider.name": "anthropic"},
    )

    assert _by_modality(metrics, INPUT) == {"unknown": 400}
    assert _by_modality(metrics, CACHE_READ) == {"unknown": 300}
    assert _by_modality(metrics, CACHE_WRITE) == {"unknown": 90}
    (point,) = _points(metrics, OP_INPUT)
    assert point.sum == 400


def test_reasoning_tokens_are_a_subset_of_output(reader):
    metrics = _record(
        reader,
        {
            "prompt_tokens": 20,
            "completion_tokens": 100,
            "total_tokens": 120,
            "completion_tokens_details": {"reasoning_tokens": 64},
        },
    )

    assert _by_modality(metrics, OUTPUT) == {"unknown": 100}
    assert _by_modality(metrics, REASONING) == {"unknown": 64}


def test_zero_counts_produce_no_counter_measurement(reader):
    # semantic-conventions-genai#602: zero and missing counts are not measurements.
    metrics = _record(
        reader,
        {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    )

    for name in COUNTERS:
        assert _points(metrics, name) == [], name


def test_absent_cache_and_reasoning_produce_no_measurement(reader):
    metrics = _record(reader, {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})

    for name in (CACHE_READ, CACHE_WRITE, REASONING):
        assert _points(metrics, name) == [], name


def test_a_reported_zero_is_still_one_operation_on_the_histograms(reader):
    # The histograms describe operations, so a known zero is a real observation.
    metrics = _record(reader, {"prompt_tokens": 25, "completion_tokens": 0, "total_tokens": 25})

    assert _points(metrics, OUTPUT) == []
    (point,) = _points(metrics, OP_OUTPUT)
    assert point.count == 1
    assert point.sum == 0


def test_no_usage_records_nothing_on_the_new_instruments(reader):
    metrics = _record(reader, None)

    for name in COUNTERS + HISTOGRAMS:
        assert _points(metrics, name) == [], name


def test_estimated_usage_is_kept_off_the_standard_instruments(reader):
    # The conventions allow offline token counting only when the user enables it. The
    # estimate path is automatic, so its counts stay on the span (flagged as estimated)
    # and on this library's own counter.
    metrics = _record(reader, None, cls=_EstimatingInstrumentor)

    for name in COUNTERS + HISTOGRAMS:
        assert _points(metrics, name) == [], name
    assert {p.value for p in _points(metrics, LEGACY)} == {12, 8}


def test_embeddings_are_not_inference_operations(reader):
    # #374 leaves embeddings to a dedicated instrument still to be defined.
    metrics = _record(
        reader,
        {"prompt_tokens": 30, "completion_tokens": 0, "total_tokens": 30},
        attributes={**CHAT_ATTRIBUTES, "gen_ai.operation.name": "embeddings"},
    )

    for name in COUNTERS + HISTOGRAMS:
        assert _points(metrics, name) == [], name
    assert [p.value for p in _points(metrics, LEGACY)] == [30]


def test_an_agent_run_is_counted_but_is_not_one_inference_operation(reader):
    # A framework span that accounts several model calls (no inner provider span did) adds
    # its tokens to the totals, but its total is not the size of any single inference.
    metrics = _record(
        reader,
        {"prompt_tokens": 900, "completion_tokens": 300, "total_tokens": 1200},
        attributes={**CHAT_ATTRIBUTES, "gen_ai.operation.name": "invoke_agent"},
    )

    assert _by_modality(metrics, INPUT) == {"unknown": 900}
    assert _by_modality(metrics, OUTPUT) == {"unknown": 300}
    for name in HISTOGRAMS:
        assert _points(metrics, name) == [], name


def test_superseded_provider_spelling_is_read_when_the_current_one_is_absent(reader):
    attributes = {k: v for k, v in CHAT_ATTRIBUTES.items() if k != "gen_ai.provider.name"}
    attributes["gen_ai.system"] = "groq"

    metrics = _record(reader, {"prompt_tokens": 5, "completion_tokens": 5}, attributes=attributes)

    (point,) = _points(metrics, INPUT)
    assert point.attributes["gen_ai.provider.name"] == "groq"


def test_missing_attributes_are_left_out_rather_than_invented(reader):
    metrics = _record(
        reader,
        {"prompt_tokens": 5, "completion_tokens": 5},
        attributes={"gen_ai.operation.name": "chat", "gen_ai.provider.name": "ollama"},
    )

    (point,) = _points(metrics, INPUT)
    assert dict(point.attributes) == {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "ollama",
        "gen_ai.token.modality": "unknown",
    }


def test_legacy_token_usage_counter_is_still_recorded(reader):
    metrics = _record(reader, {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})

    by_type = {p.attributes["token_type"]: p.value for p in _points(metrics, LEGACY)}
    assert by_type == {"prompt": 100, "completion": 50}


def test_streamed_usage_is_recorded_on_the_new_instruments(reader):
    inst = _instrumentor()
    timing = _StreamTiming(time.time() - 0.2)
    timing.usage = {
        "prompt_tokens": 40,
        "completion_tokens": 60,
        "total_tokens": 100,
        "cache_read_input_tokens": 16,
    }

    inst._finalize_stream(_Span(CHAT_ATTRIBUTES), timing, "gpt-4o")
    metrics = _metrics(reader)

    assert _by_modality(metrics, INPUT) == {"unknown": 40}
    assert _by_modality(metrics, OUTPUT) == {"unknown": 60}
    assert _by_modality(metrics, CACHE_READ) == {"unknown": 16}
    assert [p.sum for p in _points(metrics, OP_INPUT)] == [40]


@pytest.mark.parametrize("name", HISTOGRAMS)
def test_operation_histograms_use_token_buckets(monkeypatch, name):
    """The default buckets are seconds-sized; token counts need the token boundaries."""
    import genai_otel.auto_instrument as auto_instrument
    from genai_otel.metrics import _GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS

    monkeypatch.setattr(auto_instrument.metrics, "set_meter_provider", lambda provider: None)
    metric_reader = InMemoryMetricReader()
    provider = auto_instrument._setup_meter_provider(None, metric_reader)

    provider.get_meter("test").create_histogram(name, unit="{token}").record(300)

    (point,) = _points(_metrics(metric_reader), name)
    assert list(point.explicit_bounds) == [float(b) for b in _GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS]


def test_anthropic_usage_says_its_input_excludes_cache():
    from genai_otel.instrumentors.anthropic_instrumentor import AnthropicInstrumentor

    usage = AnthropicInstrumentor()._extract_usage(
        SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=40,
                cache_read_input_tokens=300,
                cache_creation_input_tokens=90,
            )
        )
    )

    assert usage["prompt_tokens_excludes_cache"] is True
    # The span attribute and the cost inputs keep their meaning.
    assert usage["prompt_tokens"] == 10
    assert usage["total_tokens"] == 50


def test_anthropic_stream_counts_cache_in_input(reader):
    """Usage folded over a streamed message still adds the cache to the input counter."""
    from genai_otel.instrumentors.anthropic_instrumentor import AnthropicInstrumentor

    inst = AnthropicInstrumentor()
    inst.config = OTelConfig(enable_cost_tracking=False)
    timing = _StreamTiming(time.time() - 0.2)
    for event in (
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=10,
                    output_tokens=1,
                    cache_read_input_tokens=300,
                    cache_creation_input_tokens=90,
                )
            ),
        ),
        SimpleNamespace(type="message_delta", usage=SimpleNamespace(output_tokens=40)),
    ):
        inst._accumulate_stream_usage(timing, event)

    inst._finalize_stream(
        _Span({**CHAT_ATTRIBUTES, "gen_ai.provider.name": "anthropic"}), timing, "claude"
    )
    metrics = _metrics(reader)

    assert _by_modality(metrics, INPUT) == {"unknown": 400}
    assert _by_modality(metrics, OUTPUT) == {"unknown": 40}
    assert _by_modality(metrics, CACHE_WRITE) == {"unknown": 90}
