"""Configuration management for the GenAI OpenTelemetry instrumentation library.

This module defines the `OTelConfig` dataclass, which encapsulates all configurable
parameters for the OpenTelemetry setup, including service name, exporter endpoint,
enablement flags for various features (GPU metrics, cost tracking, MCP instrumentation),
and error handling behavior. Configuration values are primarily loaded from
environment variables, with sensible defaults provided.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class OTelConfig:
    """Configuration for OpenTelemetry instrumentation.

    Loads settings from environment variables with sensible defaults.
    """

    service_name: str = field(default_factory=lambda: os.getenv("OTEL_SERVICE_NAME", "genai-app"))
    endpoint: str = field(
        default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    )
    enable_gpu_metrics: bool = field(
        default_factory=lambda: os.getenv("GENAI_ENABLE_GPU_METRICS", "true").lower() == "true"
    )
    enable_cost_tracking: bool = field(
        default_factory=lambda: os.getenv("GENAI_ENABLE_COST_TRACKING", "true").lower() == "true"
    )
    enable_mcp_instrumentation: bool = field(
        default_factory=lambda: os.getenv("GENAI_ENABLE_MCP_INSTRUMENTATION", "true").lower()
        == "true"
    )
    # Add fail_on_error configuration
    fail_on_error: bool = field(
        default_factory=lambda: os.getenv("GENAI_FAIL_ON_ERROR", "false").lower() == "true"
    )
    headers: Optional[Dict[str, str]] = None

    enable_co2_tracking: bool = field(
        default_factory=lambda: os.getenv("GENAI_ENABLE_CO2_TRACKING", "false").lower() == "true"
    )
    carbon_intensity: float = field(
        default_factory=lambda: float(os.getenv("GENAI_CARBON_INTENSITY", "475.0"))
    )  # gCO2e/kWh

    def __post_init__(self):
        """Post-initialization hook to parse headers from environment variable."""
        if self.enable_co2_tracking and self.carbon_intensity <= 0:
            raise ValueError("Carbon intensity must be positive")
        if self.headers is None:
            headers_str = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
            if headers_str:
                try:
                    self.headers = dict(h.split("=") for h in headers_str.split(","))
                except ValueError:
                    logger.error(
                        "Failed to parse OTEL_EXPORTER_OTLP_HEADERS: '%s'. Expected format 'key1=value1,key2=value2'.",
                        headers_str,
                    )
