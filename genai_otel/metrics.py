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

# Correct the import for OTLP Metric Exporter
if os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL") == "grpc":
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
else:
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter,
    )

from opentelemetry.sdk.resources import (
    DEPLOYMENT_ENVIRONMENT,
    SERVICE_NAME,
    TELEMETRY_SDK_NAME,
    Resource,
)

logger = logging.getLogger(__name__)

# Global variables to hold the MeterProvider and Meter
_meter_provider: Optional[MeterProvider] = None
_meter: Optional[Meter] = None


def setup_meter(
    app_name: str,
    env: str,
    otlp_endpoint: Optional[str] = None,
    otlp_headers: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], Optional[Meter]]:
    """
    Sets up the OpenTelemetry MeterProvider and Meter.

    Args:
        app_name: The name of the application.
        env: The environment (e.g., 'dev', 'prod').
        otlp_endpoint: The OTLP endpoint URL. If empty, Console exporter is used.
        otlp_headers: Optional headers for the OTLP request.

    Returns:
        A tuple containing:
        - A dictionary with metrics configuration.
        - The configured Meter object, or None if initialization fails.
    """
    global _meter_provider, _meter

    # Initialize Resource
    resource_attributes = {
        SERVICE_NAME: app_name,
        DEPLOYMENT_ENVIRONMENT: env,
        TELEMETRY_SDK_NAME: "openlit",
    }
    resource = Resource(attributes=resource_attributes)

    exporter: Optional[MetricExporter] = None
    if otlp_endpoint:
        try:
            # Basic validation for otlp_endpoint (can be expanded)
            if not isinstance(otlp_endpoint, str) or not otlp_endpoint.startswith(
                ("http://", "https://")
            ):
                logger.error("Invalid OTLP endpoint format: %s", otlp_endpoint)
                return {"app_name": app_name, "env": env}, None

            # Basic validation for otlp_headers
            if otlp_headers is not None and not isinstance(otlp_headers, dict):
                logger.error("Invalid OTLP headers format. Must be a dictionary.")
                return {"app_name": app_name, "env": env}, None

            exporter = OTLPMetricExporter(endpoint=otlp_endpoint, headers=otlp_headers)
            logger.info("Configured OTLP metric exporter for endpoint: %s", otlp_endpoint)
        except Exception as e:
            logger.error("Failed to configure OTLP metric exporter: %s", e, exc_info=True)
            return {"app_name": app_name, "env": env}, None
    else:
        exporter = ConsoleMetricExporter()
        logger.info("Configured Console metric exporter.")

    if exporter is None:
        logger.error("No metric exporter could be configured. MeterProvider cannot be initialized.")
        return {"app_name": app_name, "env": env}, None

    # Initialize the MeterProvider with a PeriodicExportingMetricReader
    try:
        metric_reader = PeriodicExportingMetricReader(exporter)
        _meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(_meter_provider)  # Set the global provider
        _meter = _meter_provider.get_meter(__name__)  # Get the meter for this module
        logger.info("MeterProvider and Meter initialized successfully.")
        return {"app_name": app_name, "env": env}, _meter
    except Exception as e:
        logger.error("Failed to initialize MeterProvider: %s", e, exc_info=True)
        return {"app_name": app_name, "env": env}, None


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
