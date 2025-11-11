"""Top-level package for GenAI OpenTelemetry Auto-Instrumentation.

This package provides a comprehensive solution for automatically instrumenting
Generative AI (GenAI) and Large Language Model (LLM) applications with OpenTelemetry.
It supports various LLM providers, frameworks, and common data stores (MCP tools).
"""

import logging
import os
import warnings

import httpx

# Suppress known third-party library warnings that we cannot control
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")
warnings.filterwarnings("ignore", message=".*validate_default.*", module="pydantic")
warnings.filterwarnings("ignore", message=".*NumPy module was reloaded.*", module="replicate")

from .__version__ import __version__

# Package metadata (from pyproject.toml)
__author__ = "Kshitij Thakkar"
__email__ = "kshitijthakkar@rocketmail.com"
__license__ = "AGPL-3.0-or-later"

# Re-exporting key components for easier access
from .auto_instrument import setup_auto_instrumentation  # Restoring direct import
from .config import OTelConfig
from .cost_calculator import CostCalculator
from .gpu_metrics import GPUMetricsCollector

# Import instrumentors conditionally to avoid errors if dependencies aren't installed
from .instrumentors import (
    AnthropicInstrumentor,
    AnyscaleInstrumentor,
    AWSBedrockInstrumentor,
    AzureOpenAIInstrumentor,
    CohereInstrumentor,
    GoogleAIInstrumentor,
    GroqInstrumentor,
    HuggingFaceInstrumentor,
    LangChainInstrumentor,
    LlamaIndexInstrumentor,
    MistralAIInstrumentor,
    OllamaInstrumentor,
    OpenAIInstrumentor,
    ReplicateInstrumentor,
    TogetherAIInstrumentor,
    VertexAIInstrumentor,
)
from .mcp_instrumentors.manager import MCPInstrumentorManager
from .server_metrics import ServerMetricsCollector, get_server_metrics

logger = logging.getLogger(__name__)


def instrument(**kwargs):
    """Public function to initialize and start auto-instrumentation.

    Loads configuration from environment variables or provided keyword arguments,
    then sets up OpenTelemetry tracing and metrics.

    Args:
        **kwargs: Configuration parameters that can override environment variables.
                  See OTelConfig for available parameters (e.g., service_name, endpoint).

    Example:
        >>> from genai_otel import instrument
        >>> instrument(service_name="my-app", endpoint="http://localhost:4318")

    Environment Variables:
        OTEL_SERVICE_NAME: Name of the service (default: "genai-app")
        OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint (default: "http://localhost:4318")
        GENAI_ENABLE_GPU_METRICS: Enable GPU metrics (default: "true")
        GENAI_ENABLE_COST_TRACKING: Enable cost tracking (default: "true")
        GENAI_ENABLE_MCP_INSTRUMENTATION: Enable MCP instrumentation (default: "true")
        GENAI_FAIL_ON_ERROR: Fail if instrumentation errors occur (default: "false")
        OTEL_EXPORTER_OTLP_HEADERS: OTLP headers in format "key1=val1,key2=val2"
        GENAI_LOG_LEVEL: Logging level (default: "INFO")
        GENAI_LOG_FILE: Log file path (optional)
    """
    try:
        # Create config object, allowing kwargs to override env vars
        config = OTelConfig(**kwargs)
        setup_auto_instrumentation(config)
        logger.info("GenAI OpenTelemetry instrumentation initialized successfully")
    except Exception as e:
        # Log the error and potentially re-raise based on fail_on_error
        logger.error("Failed to initialize instrumentation: %s", e, exc_info=True)
        fail_on_error = kwargs.get(
            "fail_on_error", os.getenv("GENAI_FAIL_ON_ERROR", "false").lower() == "true"
        )
        if fail_on_error:
            raise


__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # Core functions
    "instrument",
    "setup_auto_instrumentation",  # Re-added to __all__
    # Configuration
    "OTelConfig",
    # Utilities
    "CostCalculator",
    "GPUMetricsCollector",
    "ServerMetricsCollector",
    "get_server_metrics",
    # Instrumentors
    "OpenAIInstrumentor",
    "AnthropicInstrumentor",
    "GoogleAIInstrumentor",
    "AWSBedrockInstrumentor",
    "AzureOpenAIInstrumentor",
    "CohereInstrumentor",
    "MistralAIInstrumentor",
    "TogetherAIInstrumentor",
    "GroqInstrumentor",
    "LangChainInstrumentor",
    "LlamaIndexInstrumentor",
    "HuggingFaceInstrumentor",
    "OllamaInstrumentor",
    "VertexAIInstrumentor",
    "ReplicateInstrumentor",
    "AnyscaleInstrumentor",
    # MCP Manager
    "MCPInstrumentorManager",
]
