"""Tests for server metrics (KV cache, request queue, etc.)."""

import pytest
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from genai_otel.server_metrics import (
    ServerMetricsCollector,
    get_server_metrics,
    initialize_server_metrics,
)


@pytest.fixture
def meter_provider():
    """Create a MeterProvider with in-memory metric reader for testing."""
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    return provider, reader


def test_server_metrics_initialization(meter_provider):
    """Test that ServerMetricsCollector initializes correctly."""
    provider, reader = meter_provider
    meter = provider.get_meter("test")

    collector = ServerMetricsCollector(meter)

    assert collector is not None
    assert collector.meter == meter
    assert collector._kv_cache_usage == {}
    assert collector._num_requests_running == 0
    assert collector._num_requests_waiting == 0
    assert collector._num_requests_max == 0


def test_set_kv_cache_usage(meter_provider):
    """Test setting KV cache usage for different models."""
    provider, reader = meter_provider
    meter = provider.get_meter("test")
    collector = ServerMetricsCollector(meter)

    # Set cache usage for multiple models
    collector.set_kv_cache_usage("gpt-4", 75.5)
    collector.set_kv_cache_usage("llama-2-7b", 50.0)

    # Force metric collection
    metrics_data = reader.get_metrics_data()

    # Find the KV cache metric
    kv_cache_found = False
    for resource_metric in metrics_data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name == "gen_ai.server.kv_cache.usage":
                    kv_cache_found = True
                    # Check that we have data points for both models
                    data_points = list(metric.data.data_points)
                    assert len(data_points) == 2

                    # Extract values by model name
                    values = {dp.attributes.get("model"): dp.value for dp in data_points}
                    assert values.get("gpt-4") == 75.5
                    assert values.get("llama-2-7b") == 50.0

    assert kv_cache_found, "KV cache metric not found"


def test_kv_cache_bounds(meter_provider):
    """Test that KV cache usage is clamped to 0-100 range."""
    provider, reader = meter_provider
    meter = provider.get_meter("test")
    collector = ServerMetricsCollector(meter)

    # Test upper bound
    collector.set_kv_cache_usage("model1", 150.0)
    assert collector._kv_cache_usage["model1"] == 100.0

    # Test lower bound
    collector.set_kv_cache_usage("model2", -50.0)
    assert collector._kv_cache_usage["model2"] == 0.0


def test_request_queue_metrics(meter_provider):
    """Test request queue and concurrency metrics."""
    provider, reader = meter_provider
    meter = provider.get_meter("test")
    collector = ServerMetricsCollector(meter)

    # Set request metrics
    collector.set_requests_running(5)
    collector.set_requests_waiting(10)
    collector.set_requests_max(20)

    # Force metric collection
    metrics_data = reader.get_metrics_data()

    # Check metrics
    metric_values = {}
    for resource_metric in metrics_data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name.startswith("gen_ai.server.requests"):
                    data_points = list(metric.data.data_points)
                    if data_points:
                        metric_values[metric.name] = data_points[0].value

    assert metric_values.get("gen_ai.server.requests.running") == 5
    assert metric_values.get("gen_ai.server.requests.waiting") == 10
    assert metric_values.get("gen_ai.server.requests.max") == 20


def test_increment_decrement_running_requests(meter_provider):
    """Test incrementing and decrementing running requests."""
    provider, reader = meter_provider
    meter = provider.get_meter("test")
    collector = ServerMetricsCollector(meter)

    # Start from 0
    assert collector._num_requests_running == 0

    # Increment
    collector.increment_requests_running()
    assert collector._num_requests_running == 1

    collector.increment_requests_running(3)
    assert collector._num_requests_running == 4

    # Decrement
    collector.decrement_requests_running()
    assert collector._num_requests_running == 3

    collector.decrement_requests_running(2)
    assert collector._num_requests_running == 1

    # Should not go negative
    collector.decrement_requests_running(10)
    assert collector._num_requests_running == 0


def test_increment_decrement_waiting_requests(meter_provider):
    """Test incrementing and decrementing waiting requests."""
    provider, reader = meter_provider
    meter = provider.get_meter("test")
    collector = ServerMetricsCollector(meter)

    # Start from 0
    assert collector._num_requests_waiting == 0

    # Increment
    collector.increment_requests_waiting()
    assert collector._num_requests_waiting == 1

    collector.increment_requests_waiting(5)
    assert collector._num_requests_waiting == 6

    # Decrement
    collector.decrement_requests_waiting(3)
    assert collector._num_requests_waiting == 3

    # Should not go negative
    collector.decrement_requests_waiting(10)
    assert collector._num_requests_waiting == 0


def test_thread_safety(meter_provider):
    """Test that metrics are thread-safe using concurrent updates."""
    import threading

    provider, reader = meter_provider
    meter = provider.get_meter("test")
    collector = ServerMetricsCollector(meter)

    def increment_worker():
        for _ in range(100):
            collector.increment_requests_running()

    def decrement_worker():
        for _ in range(50):
            collector.decrement_requests_running()

    threads = []
    # Start 5 increment threads and 5 decrement threads
    for _ in range(5):
        t1 = threading.Thread(target=increment_worker)
        t2 = threading.Thread(target=decrement_worker)
        threads.extend([t1, t2])
        t1.start()
        t2.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # Expected: 5 * 100 increments - 5 * 50 decrements = 250
    assert collector._num_requests_running == 250


def test_global_server_metrics_initialization():
    """Test global server metrics singleton initialization."""
    # Reset global state
    import genai_otel.server_metrics as sm
    from genai_otel.server_metrics import _global_server_metrics

    sm._global_server_metrics = None

    # Should be None initially
    assert get_server_metrics() is None

    # Initialize
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    meter = provider.get_meter("test")

    collector = initialize_server_metrics(meter)

    # Should now return the instance
    assert get_server_metrics() is not None
    assert get_server_metrics() == collector

    # Second call should return same instance
    collector2 = initialize_server_metrics(meter)
    assert collector2 == collector


def test_observable_gauge_callbacks(meter_provider):
    """Test that observable gauge callbacks work correctly."""
    provider, reader = meter_provider
    meter = provider.get_meter("test")
    collector = ServerMetricsCollector(meter)

    # Set various metrics
    collector.set_kv_cache_usage("model-a", 80.0)
    collector.set_kv_cache_usage("model-b", 60.0)
    collector.set_requests_running(3)
    collector.set_requests_waiting(7)
    collector.set_requests_max(15)

    # Force metric collection
    metrics_data = reader.get_metrics_data()

    # Verify all metrics are present
    metric_names = set()
    for resource_metric in metrics_data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                metric_names.add(metric.name)

    expected_metrics = {
        "gen_ai.server.kv_cache.usage",
        "gen_ai.server.requests.running",
        "gen_ai.server.requests.waiting",
        "gen_ai.server.requests.max",
    }

    assert expected_metrics.issubset(
        metric_names
    ), f"Missing metrics: {expected_metrics - metric_names}"
