"""Top-level package for GenAI OpenTelemetry Auto-Instrumentation.

This package provides a comprehensive solution for automatically instrumenting
Generative AI (GenAI) and Large Language Model (LLM) applications with OpenTelemetry.
It supports various LLM providers, frameworks, and common data stores (MCP tools).

Heavy imports (instrumentors, OTel SDK, GPU metrics) are deferred until first access
to keep ``import genai_otel`` fast.
"""

import importlib
import logging
import os
import warnings

# Suppress known third-party library warnings that we cannot control
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")
warnings.filterwarnings("ignore", message=".*validate_default.*", module="pydantic")
warnings.filterwarnings("ignore", message=".*NumPy module was reloaded.*", module="replicate")

from .__version__ import __version__

# Package metadata (from pyproject.toml)
__author__ = "Kshitij Thakkar"
__email__ = "kshitijthakkar@rocketmail.com"
__license__ = "Apache-2.0"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loading registry: maps public name -> (module_path, attribute_name)
# ---------------------------------------------------------------------------
_LAZY_IMPORTS = {
    # Core functions
    "setup_auto_instrumentation": (".auto_instrument", "setup_auto_instrumentation"),
    "uninstrument": (".auto_instrument", "uninstrument"),
    # Configuration
    "OTelConfig": (".config", "OTelConfig"),
    # Utilities
    "CostCalculator": (".cost_calculator", "CostCalculator"),
    "GPUMetricsCollector": (".gpu_metrics", "GPUMetricsCollector"),
    "ServerMetricsCollector": (".server_metrics", "ServerMetricsCollector"),
    "get_server_metrics": (".server_metrics", "get_server_metrics"),
    # Instrumentors
    "OpenAIInstrumentor": (".instrumentors", "OpenAIInstrumentor"),
    "AnthropicInstrumentor": (".instrumentors", "AnthropicInstrumentor"),
    "GoogleAIInstrumentor": (".instrumentors", "GoogleAIInstrumentor"),
    "AWSBedrockInstrumentor": (".instrumentors", "AWSBedrockInstrumentor"),
    "AzureOpenAIInstrumentor": (".instrumentors", "AzureOpenAIInstrumentor"),
    "CohereInstrumentor": (".instrumentors", "CohereInstrumentor"),
    "MistralAIInstrumentor": (".instrumentors", "MistralAIInstrumentor"),
    "TogetherAIInstrumentor": (".instrumentors", "TogetherAIInstrumentor"),
    "GroqInstrumentor": (".instrumentors", "GroqInstrumentor"),
    "LangChainInstrumentor": (".instrumentors", "LangChainInstrumentor"),
    "LlamaIndexInstrumentor": (".instrumentors", "LlamaIndexInstrumentor"),
    "HuggingFaceInstrumentor": (".instrumentors", "HuggingFaceInstrumentor"),
    "OllamaInstrumentor": (".instrumentors", "OllamaInstrumentor"),
    "VertexAIInstrumentor": (".instrumentors", "VertexAIInstrumentor"),
    "ReplicateInstrumentor": (".instrumentors", "ReplicateInstrumentor"),
    "AnyscaleInstrumentor": (".instrumentors", "AnyscaleInstrumentor"),
    "SarvamAIInstrumentor": (".instrumentors", "SarvamAIInstrumentor"),
    # MCP Manager
    "MCPInstrumentorManager": (".mcp_instrumentors.manager", "MCPInstrumentorManager"),
    # Tracing utilities
    "trace_operation": (".tracing", "trace_operation"),
}


def __getattr__(name):
    """Lazily import heavy modules on first attribute access."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, __package__)
        value = getattr(module, attr_name)
        # Cache on the module so __getattr__ is not called again
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    # Auto-load .env file if python-dotenv is available
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    # Import from sub-modules directly (patchable via "genai_otel.config.OTelConfig" etc.)
    from . import auto_instrument as _auto_mod
    from . import config as _config_mod

    try:
        # Create config object, allowing kwargs to override env vars
        config = _config_mod.OTelConfig(**kwargs)
        _auto_mod.setup_auto_instrumentation(config)
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
    "uninstrument",
    "setup_auto_instrumentation",
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
    "SarvamAIInstrumentor",
    # MCP Manager
    "MCPInstrumentorManager",
    # Tracing utilities
    "trace_operation",
]
