import sys
import logging
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes

from .instrumentors import (
    OpenAIInstrumentor,
    AnthropicInstrumentor,
    GoogleAIInstrumentor,
    AWSBedrockInstrumentor,
    AzureOpenAIInstrumentor,
    CohereInstrumentor,
    MistralAIInstrumentor,
    TogetherAIInstrumentor,
    GroqInstrumentor,
    LangChainInstrumentor,
    LlamaIndexInstrumentor,
    HuggingFaceInstrumentor,
    OllamaInstrumentor,
    VertexAIInstrumentor,
    ReplicateInstrumentor,
    AnyscaleInstrumentor,
)
from .mcp_instrumentors import MCPInstrumentorManager
from .gpu_metrics import GPUMetricsCollector
from .config import OTelConfig

logger = logging.getLogger(__name__)


def setup_auto_instrumentation(config: OTelConfig):
    """Set up OpenTelemetry with auto-instrumentation for LLM frameworks and MCP tools"""

    # Create resource
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: config.service_name,
        ResourceAttributes.SERVICE_VERSION: "1.0.0",
        "telemetry.sdk.language": "python",
    })

    # Set up tracing
    tracer_provider = TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(
        endpoint=f"{config.endpoint}/v1/traces",
        headers=config.headers or {}
    )
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)

    # Set up metrics
    metric_exporter = OTLPMetricExporter(
        endpoint=f"{config.endpoint}/v1/metrics",
        headers=config.headers or {}
    )
    metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=30000)
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    # Auto-instrument LLM libraries
    _auto_instrument_llm_libraries(config)

    # Auto-instrument MCP tools (databases, APIs, message queues, vector DBs)
    if config.enable_mcp_instrumentation:
        _auto_instrument_mcp_tools(config)

    # Start GPU metrics collection if enabled
    if config.enable_gpu_metrics:
        gpu_collector = GPUMetricsCollector(meter_provider.get_meter("genai.gpu"))
        gpu_collector.start()

    logger.info("Auto-instrumentation setup complete")


def _auto_instrument_llm_libraries(config: OTelConfig):
    """Automatically instrument detected LLM libraries"""
    instrumentors = []

    # OpenAI
    if "openai" in sys.modules or _try_import("openai"):
        instrumentors.append(OpenAIInstrumentor())
        logger.info("OpenAI instrumentation enabled")

    # Anthropic (Claude)
    if "anthropic" in sys.modules or _try_import("anthropic"):
        instrumentors.append(AnthropicInstrumentor())
        logger.info("Anthropic instrumentation enabled")

    # Google AI (Gemini)
    if "google.generativeai" in sys.modules or _try_import("google.generativeai"):
        instrumentors.append(GoogleAIInstrumentor())
        logger.info("Google AI instrumentation enabled")

    # AWS Bedrock
    if "boto3" in sys.modules or _try_import("boto3"):
        instrumentors.append(AWSBedrockInstrumentor())
        logger.info("AWS Bedrock instrumentation enabled")

    # Azure OpenAI
    if "azure.ai.openai" in sys.modules or _try_import("azure.ai.openai"):
        instrumentors.append(AzureOpenAIInstrumentor())
        logger.info("Azure OpenAI instrumentation enabled")

    # Cohere
    if "cohere" in sys.modules or _try_import("cohere"):
        instrumentors.append(CohereInstrumentor())
        logger.info("Cohere instrumentation enabled")

    # Mistral AI
    if "mistralai" in sys.modules or _try_import("mistralai"):
        instrumentors.append(MistralAIInstrumentor())
        logger.info("Mistral AI instrumentation enabled")

    # Together AI
    if "together" in sys.modules or _try_import("together"):
        instrumentors.append(TogetherAIInstrumentor())
        logger.info("Together AI instrumentation enabled")

    # Groq
    if "groq" in sys.modules or _try_import("groq"):
        instrumentors.append(GroqInstrumentor())
        logger.info("Groq instrumentation enabled")

    # Ollama
    if "ollama" in sys.modules or _try_import("ollama"):
        instrumentors.append(OllamaInstrumentor())
        logger.info("Ollama instrumentation enabled")

    # Vertex AI
    if "vertexai" in sys.modules or _try_import("vertexai"):
        instrumentors.append(VertexAIInstrumentor())
        logger.info("Vertex AI instrumentation enabled")

    # Replicate
    if "replicate" in sys.modules or _try_import("replicate"):
        instrumentors.append(ReplicateInstrumentor())
        logger.info("Replicate instrumentation enabled")

    # Anyscale
    if "anyscale" in sys.modules or _try_import("anyscale"):
        instrumentors.append(AnyscaleInstrumentor())
        logger.info("Anyscale instrumentation enabled")

    # LangChain
    if "langchain" in sys.modules or _try_import("langchain"):
        instrumentors.append(LangChainInstrumentor())
        logger.info("LangChain instrumentation enabled")

    # LlamaIndex
    if "llama_index" in sys.modules or _try_import("llama_index"):
        instrumentors.append(LlamaIndexInstrumentor())
        logger.info("LlamaIndex instrumentation enabled")

    # HuggingFace
    if "transformers" in sys.modules or _try_import("transformers"):
        instrumentors.append(HuggingFaceInstrumentor())
        logger.info("HuggingFace instrumentation enabled")

    # Instrument all detected libraries
    for instrumentor in instrumentors:
        try:
            instrumentor.instrument(config=config)
        except Exception as e:
            logger.error(f"Failed to instrument {instrumentor.__class__.__name__}: {e}")


def _auto_instrument_mcp_tools(config: OTelConfig):
    """Automatically instrument MCP tools (databases, APIs, message queues, vector DBs)"""
    logger.info("Setting up MCP tool instrumentation")
    mcp_manager = MCPInstrumentorManager(config)
    mcp_manager.instrument_all()


def _try_import(module_name: str) -> bool:
    """Try to import a module and return success status"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False