"""Module for setting up OpenTelemetry auto-instrumentation for GenAI applications."""

import logging
import sys

from openinference.instrumentation.litellm import LiteLLMInstrumentor
from openinference.instrumentation.mcp import MCPInstrumentor
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from .config import OTelConfig
from .gpu_metrics import GPUMetricsCollector
from .mcp_instrumentors import MCPInstrumentorManager

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
    "smolagents": SmolagentsInstrumentor,
    "mcp": MCPInstrumentor,
    "litellm": LiteLLMInstrumentor,
}


def setup_auto_instrumentation(config: OTelConfig):
    """
    Set up OpenTelemetry with auto-instrumentation for LLM frameworks and MCP tools.

    Args:
        config: OTelConfig instance with configuration parameters.
    """
    logger.info("Starting auto-instrumentation setup...")

    # Configure OpenTelemetry SDK (TracerProvider, MeterProvider, etc.)
    import os

    service_instance_id = os.getenv("OTEL_SERVICE_INSTANCE_ID")
    environment = os.getenv("OTEL_ENVIRONMENT")
    resource_attributes = {"service.name": config.service_name}
    if service_instance_id:
        resource_attributes["service.instance.id"] = service_instance_id
    if environment:
        resource_attributes["environment"] = environment
    resource = Resource.create(resource_attributes)

    # Configure Tracing
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    set_global_textmap(TraceContextTextMapPropagator())

    logger.debug(f"OTelConfig endpoint: {config.endpoint}")
    if config.endpoint:
        import requests

        # Create a requests session that is not instrumented
        uninstrumented_session = requests.Session()

        # Suppress instrumentation on the session
        from opentelemetry.instrumentation.requests import RequestsInstrumentor

        RequestsInstrumentor.uninstrument_session(uninstrumented_session)

        # Convert timeout to float safely
        timeout_str = os.getenv("OTEL_EXPORTER_OTLP_TIMEOUT", "10.0")
        try:
            timeout = float(timeout_str)
        except (ValueError, TypeError):
            logger.warning(f"Invalid timeout value '{timeout_str}', using default 10.0")
            timeout = 10.0

        span_exporter = OTLPSpanExporter(
            endpoint=config.endpoint,
            headers=config.headers,
            timeout=timeout,
            session=uninstrumented_session,
        )
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        logger.info(f"OpenTelemetry tracing configured with OTLP endpoint: {config.endpoint}")

        # Configure Metrics
        metric_exporter = OTLPMetricExporter(
            endpoint=config.endpoint,
            headers=config.headers,
            timeout=timeout,
            session=uninstrumented_session,
        )
        metric_reader = PeriodicExportingMetricReader(exporter=metric_exporter)
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        logger.info("OpenTelemetry metrics configured with OTLP exporter")
    else:
        # Configure Console Exporters if no OTLP endpoint is set
        span_exporter = ConsoleSpanExporter()
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        logger.info("No OTLP endpoint configured, traces will be exported to console.")

        metric_exporter = ConsoleMetricExporter()
        metric_reader = PeriodicExportingMetricReader(exporter=metric_exporter)
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        logger.info("No OTLP endpoint configured, metrics will be exported to console.")

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
            mcp_manager.instrument_all(config.fail_on_error)
            logger.info("MCP tools instrumentation enabled and set up.")
        except Exception as e:
            logger.error(f"Failed to set up MCP tools instrumentation: {e}", exc_info=True)
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


def instrument(**kwargs):
    """
    Convenience wrapper for setup_auto_instrumentation that accepts kwargs.

    Set up OpenTelemetry with auto-instrumentation for LLM frameworks and MCP tools.

    Args:
        **kwargs: Keyword arguments to configure OTelConfig. These will override
                  environment variables.

    Example:
        >>> instrument(service_name="my-app", endpoint="http://localhost:4318")
    """
    # Load configuration from environment variables or use provided kwargs
    config = OTelConfig(**kwargs)

    # Call the main setup function
    setup_auto_instrumentation(config)
