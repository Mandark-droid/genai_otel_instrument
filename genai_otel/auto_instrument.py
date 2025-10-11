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
import os

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


def setup_logging(config: OTelConfig):
    """Configure logging based on environment variables.

    Sets up logging handlers and formatters based on GENAI_LOG_LEVEL and GENAI_LOG_FILE.
    If GENAI_LOG_FILE is set, logs are written to that file; otherwise, they are logged to the console.
    The logging level is determined by GENAI_LOG_LEVEL (defaults to INFO).
    """
    log_level_str = os.getenv("GENAI_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_file = os.getenv("GENAI_LOG_FILE")

    # Basic configuration: set level and format
    logging_kwargs = {
        "level": log_level,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    }
    if log_file:
        # Ensure the directory for the log file exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir)
                logger.info(f"Created log directory: {log_dir}")
            except OSError as e:
                logger.error(f"Failed to create log directory {log_dir}: {e}", exc_info=True)
                # Fallback to console logging if directory creation fails
                log_file = None
        
        if log_file: # Only add filename if directory creation was successful or not needed
            logging_kwargs["filename"] = log_file
            logging_kwargs["filemode"] = "a" # Append mode

    # Only configure if no handlers are already set up to avoid re-configuration issues
    # This is important if the logging system is configured elsewhere.
    if not logging.getLogger().handlers:
        logging.basicConfig(**logging_kwargs)
        logger.info(f"Logging configured. Level: {log_level_str}, File: {log_file or 'console'}")
    else:
        # If handlers exist, try to update the level of the root logger
        # Note: This might not affect all existing handlers if they have fixed levels.
        logging.getLogger().setLevel(log_level)
        logger.info(f"Logging level updated. New level: {log_level_str}")


def setup_auto_instrumentation(config: OTelConfig):
    """Set up OpenTelemetry with auto-instrumentation for LLM frameworks and MCP tools.

    This function initializes the OpenTelemetry tracer and meter providers, 
    configures exporters, and then attempts to instrument various LLM libraries 
    and MCP tools based on the provided configuration.

    Args:
        config (OTelConfig): The OpenTelemetry configuration object.
    """
    setup_logging(config) # Call logging setup first to ensure logs from here are captured.

    logger.info("Starting auto-instrumentation setup...")

    # Create resource
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: config.service_name,
        ResourceAttributes.SERVICE_VERSION: "1.0.0", # Consider making this configurable or dynamic
        "telemetry.sdk.language": "python",
    })

    # Set up tracing
    try:
        tracer_provider = TracerProvider(resource=resource)
        span_exporter = OTLPSpanExporter(
            endpoint=f"{config.endpoint}/v1/traces",
            headers=config.headers or {}
        )
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(tracer_provider)
        logger.info("OpenTelemetry tracing configured.")
    except Exception as e:
        logger.error(f"Failed to configure OpenTelemetry tracing: {e}", exc_info=True)
        if config.fail_on_error:
            raise

    # Set up metrics
    try:
        metric_exporter = OTLPMetricExporter(
            endpoint=f"{config.endpoint}/v1/metrics",
            headers=config.headers or {}
        )
        metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=30000)
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        logger.info("OpenTelemetry metrics configured.")
    except Exception as e:
        logger.error(f"Failed to configure OpenTelemetry metrics: {e}", exc_info=True)
        if config.fail_on_error:
            raise

    # Auto-instrument LLM libraries
    _auto_instrument_llm_libraries(config)

    # Auto-instrument MCP tools (databases, APIs, message queues, vector DBs)
    if config.enable_mcp_instrumentation:
        try:
            _auto_instrument_mcp_tools(config)
            logger.info("MCP tools instrumentation enabled and set up.")
        except Exception as e:
            logger.error(f"Failed to set up MCP tools instrumentation: {e}", exc_info=True)
            if config.fail_on_error:
                raise

    # Start GPU metrics collection if enabled
    if config.enable_gpu_metrics:
        try:
            gpu_collector = GPUMetricsCollector(meter_provider.get_meter("genai.gpu"))
            gpu_collector.start()
            logger.info("GPU metrics collection started.")
        except Exception as e:
            logger.error(f"Failed to start GPU metrics collection: {e}", exc_info=True)
            if config.fail_on_error:
                raise

    logger.info("Auto-instrumentation setup complete")


def _auto_instrument_llm_libraries(config: OTelConfig):
    """Automatically instrument detected LLM libraries.

    Checks for the presence of various LLM libraries in sys.modules or attempts to import them.
    If found, corresponding instrumentors are instantiated and their `instrument` method is called.
    Handles ImportErrors and other exceptions during instrumentation, respecting `config.fail_on_error`.
    """
    instrumentors = []

    # List of libraries and their corresponding instrumentor classes
    # This can be extended easily.
    llm_library_map = {
        "openai": OpenAIInstrumentor,
        "anthropic": AnthropicInstrumentor,
        "google.generativeai": GoogleAIInstrumentor,
        "boto3": AWSBedrockInstrumentor, # boto3 is used for AWS services like Bedrock
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
        "transformers": HuggingFaceInstrumentor, # HuggingFace often uses transformers
    }

    for module_name, InstrumentorClass in llm_library_map.items():
        if module_name in sys.modules or _try_import(module_name):
            try:
                instrumentor = InstrumentorClass()
                instrumentor.instrument(config=config)
                logger.info(f"{module_name} instrumentation enabled")
                instrumentors.append(instrumentor) # Keep track if needed, though not used here
            except ImportError as e:
                logger.debug("Skipping instrumentation for %s due to missing dependency: %s", module_name, e)
                if config.fail_on_error:
                    raise
            except Exception as e:
                logger.error("Failed to instrument %s: %s", module_name, e, exc_info=True)
                if config.fail_on_error:
                    raise


def _auto_instrument_mcp_tools(config: OTelConfig):
    """Automatically instrument MCP tools (databases, APIs, message queues, vector DBs).

    Initializes and runs the MCPInstrumentorManager if MCP instrumentation is enabled.
    """
    # MCPInstrumentorManager is responsible for instrumenting various MCP tools.
    # It should handle its own error logging and potentially respect fail_on_error.
    mcp_manager = MCPInstrumentorManager(config)
    mcp_manager.instrument_all()


def _try_import(module_name: str) -> bool:
    """Try to import a module and return success status.

    Args:
        module_name (str): The name of the module to import.

    Returns:
        bool: True if the module was imported successfully, False otherwise.
    """
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False
