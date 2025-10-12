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
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Default list of instrumentors to enable if not specified by the user.
# This maintains the "instrument everything available" behavior.
DEFAULT_INSTRUMENTORS = [
    "openai",
    "anthropic",
    "google.generativeai",
    "boto3",
    "azure.ai.openai",
    "cohere",
    "mistralai",
    "together",
    "groq",
    "ollama",
    "vertexai",
    "replicate",
    "anyscale",
    "langchain",
    "llama_index",
    "transformers",
]


def _get_enabled_instrumentors() -> List[str]:
    """
    Gets the list of enabled instrumentors from the environment variable.
    Defaults to all supported instrumentors if the variable is not set.
    """
    enabled_str = os.getenv("GENAI_ENABLED_INSTRUMENTORS")
    if enabled_str:
        return [s.strip() for s in enabled_str.split(",")]
    return DEFAULT_INSTRUMENTORS


@dataclass
class OTelConfig:
    """Configuration for OpenTelemetry instrumentation.

    Loads settings from environment variables with sensible defaults.
    """

    service_name: str = field(default_factory=lambda: os.getenv("OTEL_SERVICE_NAME", "genai-app"))
    endpoint: str = field(
        default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    )
    enabled_instrumentors: List[str] = field(default_factory=_get_enabled_instrumentors)
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
