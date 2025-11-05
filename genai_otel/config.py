"""Configuration management for the GenAI OpenTelemetry instrumentation library.

This module defines the `OTelConfig` dataclass, which encapsulates all configurable
parameters for the OpenTelemetry setup, including service name, exporter endpoint,
enablement flags for various features (GPU metrics, cost tracking, MCP instrumentation),
and error handling behavior. Configuration values are primarily loaded from
environment variables, with sensible defaults provided.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Default list of instrumentors to enable if not specified by the user.
# This maintains the "instrument everything available" behavior.
# Note: "mcp" is excluded by default because it requires the 'mcp' library (>= 1.6.0)
# which is a specialized dependency for Model Context Protocol servers/clients.
# Users can enable it by setting GENAI_ENABLED_INSTRUMENTORS="...,mcp" if needed.
#
# Note: "smolagents" and "litellm" OpenInference instrumentors require Python >= 3.10
# They are only added to the default list if Python version is compatible.
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

# Add OpenInference instrumentors only for Python >= 3.10
# IMPORTANT: Order matters! Load in this specific sequence:
# 1. smolagents - instruments the agent framework
# 2. litellm - instruments the LLM calls made by agents
if sys.version_info >= (3, 10):
    DEFAULT_INSTRUMENTORS.extend(["smolagents", "litellm"])


def _get_enabled_instrumentors() -> List[str]:
    """
    Gets the list of enabled instrumentors from the environment variable.
    Defaults to all supported instrumentors if the variable is not set.
    """
    enabled_str = os.getenv("GENAI_ENABLED_INSTRUMENTORS")
    if enabled_str:
        return [s.strip() for s in enabled_str.split(",")]
    return DEFAULT_INSTRUMENTORS


def _get_exporter_timeout() -> int:
    """
    Gets the OTLP exporter timeout from environment variable.
    Returns default of 60 seconds if not set or invalid.
    """
    timeout_str = os.getenv("OTEL_EXPORTER_OTLP_TIMEOUT", "60")
    try:
        return int(timeout_str)
    except ValueError:
        logger.warning(
            f"Invalid timeout value '{timeout_str}' in OTEL_EXPORTER_OTLP_TIMEOUT. "
            f"Using default of 60 seconds."
        )
        return 60


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
    enable_http_instrumentation: bool = field(
        default_factory=lambda: os.getenv("GENAI_ENABLE_HTTP_INSTRUMENTATION", "false").lower()
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
    exporter_timeout: int = field(default_factory=_get_exporter_timeout)
    carbon_intensity: float = field(
        default_factory=lambda: float(os.getenv("GENAI_CARBON_INTENSITY", "475.0"))
    )  # gCO2e/kWh

    power_cost_per_kwh: float = field(
        default_factory=lambda: float(os.getenv("GENAI_POWER_COST_PER_KWH", "0.12"))
    )  # USD per kWh - electricity cost for power consumption tracking

    gpu_collection_interval: int = field(
        default_factory=lambda: int(os.getenv("GENAI_GPU_COLLECTION_INTERVAL", "5"))
    )  # seconds - how often to collect GPU metrics and CO2 emissions

    # OpenTelemetry semantic convention stability opt-in
    # Supports "gen_ai" for new conventions, "gen_ai/dup" for dual emission
    semconv_stability_opt_in: str = field(
        default_factory=lambda: os.getenv("OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai")
    )

    # Enable content capture as span events
    # WARNING: May capture sensitive data. Use with caution.
    enable_content_capture: bool = field(
        default_factory=lambda: os.getenv("GENAI_ENABLE_CONTENT_CAPTURE", "false").lower() == "true"
    )

    # Custom pricing configuration for models not in llm_pricing.json
    # Format: JSON string with same structure as llm_pricing.json
    # Example: {"chat": {"custom-model": {"promptPrice": 0.001, "completionPrice": 0.002}}}
    custom_pricing_json: Optional[str] = field(
        default_factory=lambda: os.getenv("GENAI_CUSTOM_PRICING_JSON")
    )

    # Session and user tracking (Phase 4.1)
    # Optional callable functions to extract session_id and user_id from requests
    # Signature: (instance, args, kwargs) -> Optional[str]
    # Example: lambda instance, args, kwargs: kwargs.get("metadata", {}).get("session_id")
    session_id_extractor: Optional[Callable[[Any, Tuple, Dict], Optional[str]]] = None
    user_id_extractor: Optional[Callable[[Any, Tuple, Dict], Optional[str]]] = None


import os

from opentelemetry import trace
from opentelemetry.sdk.resources import (  # noqa: F401
    DEPLOYMENT_ENVIRONMENT,
    SERVICE_NAME,
    TELEMETRY_SDK_NAME,
    Resource,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

if os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL") == "grpc":
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
else:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


def setup_tracing(
    config: "OTelConfig",  # Use OTelConfig from this module
    tracer_name: str,
    disable_batch: bool = False,
):
    """
    Sets up tracing with OpenTelemetry.
    Initializes the tracer provider and configures the span processor and exporter.
    """

    try:
        # Disable Haystack Auto Tracing
        os.environ["HAYSTACK_AUTO_TRACE_ENABLED"] = "false"

        # Create a resource with the service name attribute.
        resource = Resource.create(
            attributes={
                SERVICE_NAME: config.service_name,
                DEPLOYMENT_ENVIRONMENT: os.getenv("ENVIRONMENT", "dev"),
                TELEMETRY_SDK_NAME: "genai_otel_instrument",
            }
        )

        # Initialize the TracerProvider with the created resource.
        trace.set_tracer_provider(TracerProvider(resource=resource))

        # Configure the span exporter and processor based on whether the endpoint is effectively set.
        if config.endpoint:
            span_exporter = OTLPSpanExporter(headers=config.headers)
            span_processor = (
                BatchSpanProcessor(span_exporter)
                if not disable_batch
                else SimpleSpanProcessor(span_exporter)
            )
        else:
            span_exporter = ConsoleSpanExporter()
            span_processor = SimpleSpanProcessor(span_exporter)

        trace.get_tracer_provider().add_span_processor(span_processor)

        return trace.get_tracer(tracer_name)

    except Exception as e:
        logger.error("Failed to initialize OpenTelemetry: %s", e, exc_info=True)
        return None
