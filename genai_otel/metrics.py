# isort: skip_file
import logging
import os
from typing import Any, Dict, Optional, Tuple

from opentelemetry import metrics
from opentelemetry.metrics import Meter  # Import Meter here
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics._internal.export import MetricExporter
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import (
    DEPLOYMENT_ENVIRONMENT,
    SERVICE_NAME,
    TELEMETRY_SDK_NAME,
    Resource,
)

logger = logging.getLogger(__name__)

# Correct the import for OTLP Metric Exporter
if os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL") == "grpc":
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (  # noqa: E402
        OTLPMetricExporter,
    )
else:
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (  # noqa: E402
        OTLPMetricExporter,
    )

# Global variables to hold the MeterProvider and Meter
_meter_provider: Optional[MeterProvider] = None
_meter: Optional[Meter] = None


def get_meter() -> Meter:
    """
    Returns the globally configured Meter.
    """
    return metrics.get_meter(__name__)


def get_meter_provider() -> MeterProvider:
    """
    Returns the globally configured MeterProvider.
    """
    return metrics.get_meter_provider()


_DB_CLIENT_OPERATION_DURATION_BUCKETS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]

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

_GEN_AI_SERVER_TBT = [
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
    2.5,
]

_GEN_AI_SERVER_TFTT = [
    0.001,
    0.005,
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
]

_GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS = [
    1,
    4,
    16,
    64,
    256,
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    4194304,
    16777216,
    67108864,
]

# MCP-specific bucket boundaries for performance and size metrics
_MCP_CLIENT_OPERATION_DURATION_BUCKETS = [
    0.001,  # 1ms
    0.005,  # 5ms
    0.01,  # 10ms
    0.05,  # 50ms
    0.1,  # 100ms
    0.5,  # 500ms
    1.0,  # 1s
    2.0,  # 2s
    5.0,  # 5s
    10.0,  # 10s
]

_MCP_PAYLOAD_SIZE_BUCKETS = [
    100,  # 100 bytes
    500,  # 500 bytes
    1024,  # 1KB
    5120,  # 5KB
    10240,  # 10KB
    51200,  # 50KB
    102400,  # 100KB
    512000,  # 500KB
    1048576,  # 1MB
    5242880,  # 5MB
]
