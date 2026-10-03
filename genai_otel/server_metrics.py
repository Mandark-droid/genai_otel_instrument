"""Module for collecting server-side LLM serving metrics.

This module provides metrics for LLM serving engines like vLLM, TGI, etc.
It includes KV cache metrics, request queue metrics, and other server-level observability.
"""

import logging
import threading
from typing import Callable, Dict, List, Optional

# Whose queue a sample describes (the cross-SDK contract, SDK_ADDITIONS section 12).
# "engine": a poller read it from the serving engine; "client": the in-flight count of
# this library's own calls. TraceVerse stores the two apart (migration 177).
ENGINE = "engine"
CLIENT = "client"
_SOURCE_KEY = "server_metrics_source"

from opentelemetry.metrics import Meter, ObservableGauge, Observation

logger = logging.getLogger(__name__)


class ServerMetricsCollector:
    """Collects and reports server-side LLM serving metrics.

    This class provides metrics for:
    - KV cache usage (per model)
    - Request queue depth and concurrency
    - Maximum request capacity

    These metrics can be populated by:
    1. Manual instrumentation (user calls set_* methods)
    2. Auto-instrumentation of serving frameworks (vLLM, TGI, etc.)
    """

    def __init__(self, meter: Meter):
        """Initialize the ServerMetricsCollector.

        Args:
            meter: The OpenTelemetry meter to use for recording metrics.
        """
        self.meter = meter
        self._lock = threading.Lock()

        # Storage for metric values
        self._kv_cache_usage: Dict[str, float] = {}  # model_name -> usage_percentage
        self._num_requests_running = 0
        self._num_requests_waiting = 0
        self._num_requests_max = 0
        # True once an application sets the running count itself: the value is then no
        # longer only this library's in-flight count, so it is not labelled "client".
        self._running_set_by_app = False
        # What engine pollers read, kept apart from the values above so neither
        # overwrites the other and each sample can say where it came from.
        self._engine_kv_cache_usage: Dict[str, float] = {}
        self._engine_requests: Dict[str, int] = {}

        # Create observable gauges
        self.kv_cache_gauge = self.meter.create_observable_gauge(
            "gen_ai.server.kv_cache.usage",
            callbacks=[self._observe_kv_cache],
            description="GPU KV-cache usage percentage (0-100)",
            unit="%",
        )

        self.requests_running_gauge = self.meter.create_observable_gauge(
            "gen_ai.server.requests.running",
            callbacks=[self._observe_requests_running],
            description="Number of requests currently executing",
        )

        self.requests_waiting_gauge = self.meter.create_observable_gauge(
            "gen_ai.server.requests.waiting",
            callbacks=[self._observe_requests_waiting],
            description="Number of requests waiting in queue",
        )

        self.requests_max_gauge = self.meter.create_observable_gauge(
            "gen_ai.server.requests.max",
            callbacks=[self._observe_requests_max],
            description="Maximum concurrent request capacity",
        )

        logger.info("Server metrics collector initialized")

    def _observe_kv_cache(self, options) -> list:
        """Observable callback for KV cache usage."""
        with self._lock:
            observations = []
            for model_name, usage in self._kv_cache_usage.items():
                observations.append(Observation(value=usage, attributes={"model": model_name}))
            for model_name, usage in self._engine_kv_cache_usage.items():
                observations.append(
                    Observation(value=usage, attributes={"model": model_name, _SOURCE_KEY: ENGINE})
                )
            return observations

    def _engine_observation(self, name: str) -> List[Observation]:
        if name not in self._engine_requests:
            return []
        return [Observation(value=self._engine_requests[name], attributes={_SOURCE_KEY: ENGINE})]

    def _observe_requests_running(self, options) -> list:
        """Observable callback for running requests."""
        with self._lock:
            attributes = {} if self._running_set_by_app else {_SOURCE_KEY: CLIENT}
            return [Observation(value=self._num_requests_running, attributes=attributes)] + (
                self._engine_observation("running")
            )

    def _observe_requests_waiting(self, options) -> list:
        """Observable callback for waiting requests."""
        with self._lock:
            return [Observation(value=self._num_requests_waiting)] + self._engine_observation(
                "waiting"
            )

    def _observe_requests_max(self, options) -> list:
        """Observable callback for max requests."""
        with self._lock:
            return [Observation(value=self._num_requests_max)] + self._engine_observation("max")

    # Public API for manual instrumentation

    def set_kv_cache_usage(
        self, model_name: str, usage_percent: float, source: Optional[str] = None
    ):
        """Set KV cache usage for a specific model.

        Args:
            model_name: Name of the model
            usage_percent: Cache usage as percentage (0-100)
            source: "engine" when a poller read it from the serving engine; the sample is
                then labelled server_metrics_source="engine" and kept apart from values
                the application sets.
        """
        value = min(100.0, max(0.0, usage_percent))
        with self._lock:
            if source == ENGINE:
                self._engine_kv_cache_usage[model_name] = value
            else:
                self._kv_cache_usage[model_name] = value

    def replace_engine_kv_cache_usage(self, usage_by_model: Dict[str, float]):
        """Replace every engine KV-cache entry with what the engine reports now.

        A model the engine has unloaded disappears instead of keeping its last value
        (SDK_ADDITIONS section 12).
        """
        with self._lock:
            self._engine_kv_cache_usage = {
                model: min(100.0, max(0.0, usage)) for model, usage in usage_by_model.items()
            }

    def _set_engine(self, name: str, count: int) -> None:
        with self._lock:
            self._engine_requests[name] = max(0, count)

    def set_requests_running(self, count: int, source: Optional[str] = None):
        """Set number of currently running requests.

        Args:
            count: Number of active requests
            source: "engine" for a poller's reading of the engine's queue.
        """
        if source == ENGINE:
            self._set_engine("running", count)
            return
        with self._lock:
            self._num_requests_running = max(0, count)
            self._running_set_by_app = True

    def set_requests_waiting(self, count: int, source: Optional[str] = None):
        """Set number of requests waiting in queue.

        Args:
            count: Number of queued requests
            source: "engine" for a poller's reading of the engine's queue.
        """
        if source == ENGINE:
            self._set_engine("waiting", count)
            return
        with self._lock:
            self._num_requests_waiting = max(0, count)

    def set_requests_max(self, count: int, source: Optional[str] = None):
        """Set maximum concurrent request capacity.

        Args:
            count: Maximum request capacity
            source: "engine" for a poller's reading of the engine's capacity.
        """
        if source == ENGINE:
            self._set_engine("max", count)
            return
        with self._lock:
            self._num_requests_max = max(0, count)

    def increment_requests_running(self, delta: int = 1):
        """Increment running requests counter.

        Args:
            delta: Amount to increment by (default: 1)
        """
        with self._lock:
            self._num_requests_running = max(0, self._num_requests_running + delta)

    def decrement_requests_running(self, delta: int = 1):
        """Decrement running requests counter.

        Args:
            delta: Amount to decrement by (default: 1)
        """
        with self._lock:
            self._num_requests_running = max(0, self._num_requests_running - delta)

    def increment_requests_waiting(self, delta: int = 1):
        """Increment waiting requests counter.

        Args:
            delta: Amount to increment by (default: 1)
        """
        with self._lock:
            self._num_requests_waiting = max(0, self._num_requests_waiting + delta)

    def decrement_requests_waiting(self, delta: int = 1):
        """Decrement waiting requests counter.

        Args:
            delta: Amount to decrement by (default: 1)
        """
        with self._lock:
            self._num_requests_waiting = max(0, self._num_requests_waiting - delta)


# Global instance that can be accessed by users
_global_server_metrics: Optional[ServerMetricsCollector] = None


def get_server_metrics() -> Optional[ServerMetricsCollector]:
    """Get the global server metrics collector instance.

    Returns:
        ServerMetricsCollector instance or None if not initialized
    """
    return _global_server_metrics


def initialize_server_metrics(meter: Meter) -> ServerMetricsCollector:
    """Initialize the global server metrics collector.

    Args:
        meter: OpenTelemetry meter instance

    Returns:
        Initialized ServerMetricsCollector instance
    """
    global _global_server_metrics
    if _global_server_metrics is None:
        _global_server_metrics = ServerMetricsCollector(meter)
        logger.info("Global server metrics collector initialized")
    return _global_server_metrics
