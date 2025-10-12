"""Module for setting up OpenTelemetry auto-instrumentation for GenAI applications."""

import logging
import sys

from opentelemetry import metrics

from .config import OTelConfig
from .gpu_metrics import GPUMetricsCollector
from .mcp_instrumentors import MCPInstrumentorManager
from .otel_setup import configure_opentelemetry

# Import instrumentors - fix the import path based on your actual structure
try:
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
except ImportError:
    # Fallback for testing or if instrumentors are in different structure
    from genai_otel.instrumentors import (
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

logger = logging.getLogger(__name__)

# Defines the available instrumentors. This is now at the module level for easier mocking in tests.
INSTRUMENTORS = {
    "openai": OpenAIInstrumentor,
    "anthropic": AnthropicInstrumentor,
    "google.generativeai": GoogleAIInstrumentor,
    "boto3": AWSBedrockInstrumentor,
    "azure.ai.openai": AzureOpenAIInstrumentor,
    "cohere": CohereInstrumentor,
    "mistralai": MistralAIInstrumentor,
    "together": TogetherAIInstrumentor,
    "groq": GroqInstrumentor,
    "ollama": OllamaInstrumentor,
    "vertexai": VertexAIInstrumentor,
    "replicate": ReplicateInstrumentor,
    "anyscale": AnyscaleInstrumentor,
    "langchain": LangChainInstrumentor,
    "llama_index": LlamaIndexInstrumentor,
    "transformers": HuggingFaceInstrumentor,
}


def setup_auto_instrumentation(config: OTelConfig):
    """
    Set up OpenTelemetry with auto-instrumentation for LLM frameworks and MCP tools.
    """
    logger.info("Starting auto-instrumentation setup...")

    # Configure OpenTelemetry SDK (TracerProvider, MeterProvider, etc.)
    configure_opentelemetry(config)

    # Auto-instrument LLM libraries based on the configuration
    for name in config.enabled_instrumentors:
        if name in INSTRUMENTORS:
            try:
                instrumentor_class = INSTRUMENTORS[name]
                instrumentor = instrumentor_class()
                instrumentor.instrument(config=config)
                logger.info(f"{name} instrumentation enabled")
            except Exception as e:
                logger.error(f"Failed to instrument {name}: {e}", exc_info=True)
                if config.fail_on_error:
                    raise
        else:
            logger.warning(f"Unknown instrumentor '{name}' requested.")

    # Auto-instrument MCP tools (databases, APIs, etc.)
    if config.enable_mcp_instrumentation:
        try:
            mcp_manager = MCPInstrumentorManager(config)
            mcp_manager.instrument_all(config.fail_on_error)  # This is the correct method name
            logger.info("MCP tools instrumentation enabled and set up.")
        except Exception as e:
            logger.error(
                f"Failed to set up MCP tools instrumentation: {e}", exc_info=True
            )  # Fixed f-string
            if config.fail_on_error:
                raise

    # Start GPU metrics collection if enabled
    if config.enable_gpu_metrics:
        try:
            meter_provider = metrics.get_meter_provider()
            gpu_collector = GPUMetricsCollector(meter_provider.get_meter("genai.gpu"), config)
            gpu_collector.start()
            logger.info("GPU metrics collection started.")
        except Exception as e:
            logger.error(f"Failed to start GPU metrics collection: {e}", exc_info=True)
            if config.fail_on_error:
                raise

    logger.info("Auto-instrumentation setup complete")
