"""Module for setting up OpenTelemetry auto-instrumentation for GenAI applications."""

# isort: skip_file

import logging
import sys

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter as GrpcOTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GrpcOTLPSpanExporter,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentelemetry.sdk.metrics.view import View
from opentelemetry.sdk.metrics._internal.aggregation import ExplicitBucketHistogramAggregation
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from .config import OTelConfig
from .cost_calculator import CostCalculator
from .cost_enrichment_processor import CostEnrichmentSpanProcessor
from .cost_enriching_exporter import CostEnrichingSpanExporter
from .evaluation.config import (
    BiasConfig,
    HallucinationConfig,
    PIIConfig,
    PIIEntityType,
    PIIMode,
    PromptInjectionConfig,
    RestrictedTopicsConfig,
    ToxicityConfig,
)
from .evaluation.span_processor import EvaluationSpanProcessor
from .gpu_metrics import GPUMetricsCollector
from .mcp_instrumentors import MCPInstrumentorManager
from .server_metrics import initialize_server_metrics
from .metrics import (
    _GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS,
    _GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS,
    _GEN_AI_SERVER_TBT,
    _GEN_AI_SERVER_TFTT,
    _MCP_CLIENT_OPERATION_DURATION_BUCKETS,
    _MCP_PAYLOAD_SIZE_BUCKETS,
)

# Import semantic conventions
try:
    from openlit.semcov import SemanticConvention as SC
except ImportError:
    # Fallback if openlit not available
    class SC:
        GEN_AI_CLIENT_OPERATION_DURATION = "gen_ai.client.operation.duration"
        GEN_AI_SERVER_TTFT = "gen_ai.server.ttft"
        GEN_AI_SERVER_TBT = "gen_ai.server.tbt"


# Import instrumentors - fix the import path based on your actual structure
try:
    from .instrumentors import (
        AnthropicInstrumentor,
        AnyscaleInstrumentor,
        AutoGenInstrumentor,
        AWSBedrockInstrumentor,
        AzureOpenAIInstrumentor,
        BedrockAgentsInstrumentor,
        CohereInstrumentor,
        CrewAIInstrumentor,
        DSPyInstrumentor,
        GoogleAIInstrumentor,
        GroqInstrumentor,
        GuardrailsAIInstrumentor,
        HaystackInstrumentor,
        HuggingFaceInstrumentor,
        HyperbolicInstrumentor,
        InstructorInstrumentor,
        LangChainInstrumentor,
        LangGraphInstrumentor,
        LlamaIndexInstrumentor,
        MistralAIInstrumentor,
        OllamaInstrumentor,
        OpenAIInstrumentor,
        OpenAIAgentsInstrumentor,
        PydanticAIInstrumentor,
        ReplicateInstrumentor,
        SambaNovaInstrumentor,
        TogetherAIInstrumentor,
        VertexAIInstrumentor,
    )
except ImportError:
    # Fallback for testing or if instrumentors are in different structure
    from genai_otel.instrumentors import (
        AnthropicInstrumentor,
        AnyscaleInstrumentor,
        AutoGenInstrumentor,
        AWSBedrockInstrumentor,
        AzureOpenAIInstrumentor,
        BedrockAgentsInstrumentor,
        CohereInstrumentor,
        CrewAIInstrumentor,
        DSPyInstrumentor,
        GoogleAIInstrumentor,
        GroqInstrumentor,
        GuardrailsAIInstrumentor,
        HaystackInstrumentor,
        HuggingFaceInstrumentor,
        HyperbolicInstrumentor,
        InstructorInstrumentor,
        LangChainInstrumentor,
        LangGraphInstrumentor,
        LlamaIndexInstrumentor,
        MistralAIInstrumentor,
        OllamaInstrumentor,
        OpenAIInstrumentor,
        OpenAIAgentsInstrumentor,
        PydanticAIInstrumentor,
        ReplicateInstrumentor,
        SambaNovaInstrumentor,
        TogetherAIInstrumentor,
        VertexAIInstrumentor,
    )

logger = logging.getLogger(__name__)

# Optional OpenInference instrumentors (requires Python >= 3.10)
try:
    from openinference.instrumentation.litellm import LiteLLMInstrumentor  # noqa: E402
    from openinference.instrumentation.mcp import MCPInstrumentor  # noqa: E402
    from openinference.instrumentation.smolagents import (  # noqa: E402
        SmolagentsInstrumentor,
    )

    OPENINFERENCE_AVAILABLE = True
except ImportError:
    LiteLLMInstrumentor = None
    MCPInstrumentor = None
    SmolagentsInstrumentor = None
    OPENINFERENCE_AVAILABLE = False

# Defines the available instrumentors. This is now at the module level for easier mocking in tests.
INSTRUMENTORS = {
    "openai": OpenAIInstrumentor,
    "agents": OpenAIAgentsInstrumentor,  # OpenAI Agents SDK
    "anthropic": AnthropicInstrumentor,
    "google.generativeai": GoogleAIInstrumentor,
    "boto3": AWSBedrockInstrumentor,
    "azure.ai.openai": AzureOpenAIInstrumentor,
    "autogen": AutoGenInstrumentor,  # AutoGen multi-agent framework
    "bedrock_agents": BedrockAgentsInstrumentor,  # AWS Bedrock Agents
    "cohere": CohereInstrumentor,
    "crewai": CrewAIInstrumentor,  # CrewAI multi-agent framework
    "dspy": DSPyInstrumentor,  # DSPy declarative LM programming framework
    "mistralai": MistralAIInstrumentor,
    "together": TogetherAIInstrumentor,
    "groq": GroqInstrumentor,
    "guardrails": GuardrailsAIInstrumentor,  # Guardrails AI validation framework
    "haystack": HaystackInstrumentor,  # Haystack NLP pipeline framework
    "instructor": InstructorInstrumentor,  # Instructor structured output extraction
    "ollama": OllamaInstrumentor,
    "vertexai": VertexAIInstrumentor,
    "replicate": ReplicateInstrumentor,
    "anyscale": AnyscaleInstrumentor,
    "sambanova": SambaNovaInstrumentor,
    "hyperbolic": HyperbolicInstrumentor,
    "langchain": LangChainInstrumentor,
    "langgraph": LangGraphInstrumentor,  # LangGraph stateful workflow framework
    "llama_index": LlamaIndexInstrumentor,
    "pydantic_ai": PydanticAIInstrumentor,  # Pydantic AI type-safe agent framework
    "transformers": HuggingFaceInstrumentor,
}

# Add OpenInference instrumentors if available (requires Python >= 3.10)
# IMPORTANT: Order matters! Load in this specific sequence:
# 1. smolagents - instruments the agent framework
# 2. litellm - instruments LLM calls made by agents
# 3. mcp - instruments Model Context Protocol tools
if OPENINFERENCE_AVAILABLE:
    INSTRUMENTORS.update(
        {
            "smolagents": SmolagentsInstrumentor,
            "litellm": LiteLLMInstrumentor,
            "mcp": MCPInstrumentor,
        }
    )


# Global list to store OTLP exporter sessions that should not be instrumented
_OTLP_EXPORTER_SESSIONS = []


def setup_auto_instrumentation(config: OTelConfig):
    """
    Set up OpenTelemetry with auto-instrumentation for LLM frameworks and MCP tools.

    Args:
        config: OTelConfig instance with configuration parameters.
    """
    global _OTLP_EXPORTER_SESSIONS
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

    # Add cost enrichment processor for custom instrumentors (OpenAI, Ollama, etc.)
    # These instrumentors set cost attributes directly, so processor is mainly for logging
    # Also attempts to enrich OpenInference spans (smolagents, litellm, mcp), though
    # the processor can't modify ReadableSpan - the exporter below handles that
    cost_calculator = None
    if config.enable_cost_tracking:
        try:
            cost_calculator = CostCalculator()
            cost_processor = CostEnrichmentSpanProcessor(cost_calculator)
            tracer_provider.add_span_processor(cost_processor)
            logger.info("Cost enrichment processor added")
        except Exception as e:
            logger.warning(f"Failed to add cost enrichment processor: {e}", exc_info=True)

    # Add evaluation and safety span processor (v0.2.0)
    if any(
        [
            config.enable_pii_detection,
            config.enable_toxicity_detection,
            config.enable_bias_detection,
            config.enable_prompt_injection_detection,
            config.enable_restricted_topics,
            config.enable_hallucination_detection,
        ]
    ):
        try:
            # Build PII config from OTelConfig
            pii_config = None
            if config.enable_pii_detection:
                pii_config = PIIConfig(
                    enabled=True,
                    mode=PIIMode(config.pii_mode),
                    threshold=config.pii_threshold,
                    gdpr_mode=config.pii_gdpr_mode,
                    hipaa_mode=config.pii_hipaa_mode,
                    pci_dss_mode=config.pii_pci_dss_mode,
                )

            # Build Toxicity config
            toxicity_config = None
            if config.enable_toxicity_detection:
                toxicity_config = ToxicityConfig(
                    enabled=True,
                    threshold=config.toxicity_threshold,
                    use_perspective_api=config.toxicity_use_perspective_api,
                    perspective_api_key=config.toxicity_perspective_api_key,
                )

            # Build Bias config
            bias_config = None
            if config.enable_bias_detection:
                bias_config = BiasConfig(
                    enabled=True,
                    threshold=config.bias_threshold,
                )

            # Build Prompt Injection config
            prompt_injection_config = None
            if config.enable_prompt_injection_detection:
                prompt_injection_config = PromptInjectionConfig(
                    enabled=True,
                    threshold=config.prompt_injection_threshold,
                )

            # Build Restricted Topics config
            restricted_topics_config = None
            if config.enable_restricted_topics:
                restricted_topics_config = RestrictedTopicsConfig(
                    enabled=True,
                    threshold=config.restricted_topics_threshold,
                )

            # Build Hallucination config
            hallucination_config = None
            if config.enable_hallucination_detection:
                hallucination_config = HallucinationConfig(
                    enabled=True,
                    threshold=config.hallucination_threshold,
                )

            # Create and add evaluation processor
            evaluation_processor = EvaluationSpanProcessor(
                pii_config=pii_config,
                toxicity_config=toxicity_config,
                bias_config=bias_config,
                prompt_injection_config=prompt_injection_config,
                restricted_topics_config=restricted_topics_config,
                hallucination_config=hallucination_config,
            )
            tracer_provider.add_span_processor(evaluation_processor)
            logger.info("Evaluation and safety span processor added")
        except Exception as e:
            logger.warning(f"Failed to add evaluation span processor: {e}", exc_info=True)

    logger.debug(f"OTelConfig endpoint: {config.endpoint}")
    if config.endpoint:
        # Use timeout from config (already validated as int)
        timeout = config.exporter_timeout

        # CRITICAL FIX: Set endpoint in environment variable so exporters can append correct paths
        # The exporters only call _append_trace_path() when reading from env vars
        from urllib.parse import urlparse

        # Set the base endpoint in environment variable
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = config.endpoint

        parsed = urlparse(config.endpoint)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        # Build list of URLs to exclude from instrumentation
        excluded_urls = [
            base_url,
            config.endpoint,
            f"{base_url}/v1/traces",
            f"{base_url}/v1/metrics",
            config.endpoint.rstrip("/") + "/v1/traces",
            config.endpoint.rstrip("/") + "/v1/metrics",
        ]

        # Add to environment variable (comma-separated)
        existing = os.environ.get("OTEL_PYTHON_REQUESTS_EXCLUDED_URLS", "")
        if existing:
            excluded_urls.append(existing)
        os.environ["OTEL_PYTHON_REQUESTS_EXCLUDED_URLS"] = ",".join(excluded_urls)
        logger.info(f"Excluded OTLP endpoints from instrumentation: {base_url}")

        # Set timeout in environment variable as integer string (OTLP exporters expect int)
        os.environ["OTEL_EXPORTER_OTLP_TIMEOUT"] = str(timeout)

        # Detect OTLP protocol (grpc or http)
        # Priority: OTEL_EXPORTER_OTLP_PROTOCOL env var > port-based detection > default (http)
        otlp_protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "").lower()
        if not otlp_protocol:
            # Detect from port: 4317 = gRPC, 4318 = HTTP
            from urllib.parse import urlparse

            parsed_endpoint = urlparse(base_url)
            if parsed_endpoint.port == 4317:
                otlp_protocol = "grpc"
            else:
                otlp_protocol = "http"

        # Create exporters based on protocol
        if otlp_protocol == "grpc":
            span_exporter = GrpcOTLPSpanExporter(
                headers=config.headers,
            )
            metric_exporter = GrpcOTLPMetricExporter(
                headers=config.headers,
            )
            logger.info(f"Using OTLP gRPC protocol (endpoint: {base_url})")
        else:
            # Default to HTTP
            span_exporter = OTLPSpanExporter(
                headers=config.headers,
            )
            metric_exporter = OTLPMetricExporter(
                headers=config.headers,
            )
            logger.info(f"Using OTLP HTTP protocol (endpoint: {base_url})")

        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        logger.info(
            f"OpenTelemetry tracing configured with OTLP endpoint: {span_exporter._endpoint}"
        )

        # Configure Metrics with Views for histogram buckets
        metric_reader = PeriodicExportingMetricReader(exporter=metric_exporter)

        # Create Views to configure histogram buckets for GenAI operation duration
        duration_view = View(
            instrument_name=SC.GEN_AI_CLIENT_OPERATION_DURATION,
            aggregation=ExplicitBucketHistogramAggregation(
                boundaries=_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS
            ),
        )

        # Create Views for MCP metrics histograms
        mcp_duration_view = View(
            instrument_name="mcp.client.operation.duration",
            aggregation=ExplicitBucketHistogramAggregation(
                boundaries=_MCP_CLIENT_OPERATION_DURATION_BUCKETS
            ),
        )

        mcp_request_size_view = View(
            instrument_name="mcp.request.size",
            aggregation=ExplicitBucketHistogramAggregation(boundaries=_MCP_PAYLOAD_SIZE_BUCKETS),
        )

        mcp_response_size_view = View(
            instrument_name="mcp.response.size",
            aggregation=ExplicitBucketHistogramAggregation(boundaries=_MCP_PAYLOAD_SIZE_BUCKETS),
        )

        # Create Views for streaming metrics (Phase 3.4)
        ttft_view = View(
            instrument_name=SC.GEN_AI_SERVER_TTFT,
            aggregation=ExplicitBucketHistogramAggregation(boundaries=_GEN_AI_SERVER_TFTT),
        )

        tbt_view = View(
            instrument_name=SC.GEN_AI_SERVER_TBT,
            aggregation=ExplicitBucketHistogramAggregation(boundaries=_GEN_AI_SERVER_TBT),
        )

        # Create Views for token distribution histograms
        prompt_tokens_view = View(
            instrument_name="gen_ai.client.token.usage.prompt",
            aggregation=ExplicitBucketHistogramAggregation(
                boundaries=_GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS
            ),
        )

        completion_tokens_view = View(
            instrument_name="gen_ai.client.token.usage.completion",
            aggregation=ExplicitBucketHistogramAggregation(
                boundaries=_GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS
            ),
        )

        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader],
            views=[
                duration_view,
                mcp_duration_view,
                mcp_request_size_view,
                mcp_response_size_view,
                ttft_view,
                tbt_view,
                prompt_tokens_view,
                completion_tokens_view,
            ],
        )
        metrics.set_meter_provider(meter_provider)
        logger.info(
            f"OpenTelemetry metrics configured with OTLP endpoint: {metric_exporter._endpoint}"
        )
    else:
        # Configure Console Exporters if no OTLP endpoint is set
        span_exporter = ConsoleSpanExporter()
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        logger.info("No OTLP endpoint configured, traces will be exported to console.")

        metric_exporter = ConsoleMetricExporter()
        metric_reader = PeriodicExportingMetricReader(exporter=metric_exporter)

        # Create Views to configure histogram buckets (same as OTLP path)
        duration_view = View(
            instrument_name=SC.GEN_AI_CLIENT_OPERATION_DURATION,
            aggregation=ExplicitBucketHistogramAggregation(
                boundaries=_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS
            ),
        )

        # Create Views for MCP metrics histograms
        mcp_duration_view = View(
            instrument_name="mcp.client.operation.duration",
            aggregation=ExplicitBucketHistogramAggregation(
                boundaries=_MCP_CLIENT_OPERATION_DURATION_BUCKETS
            ),
        )

        mcp_request_size_view = View(
            instrument_name="mcp.request.size",
            aggregation=ExplicitBucketHistogramAggregation(boundaries=_MCP_PAYLOAD_SIZE_BUCKETS),
        )

        mcp_response_size_view = View(
            instrument_name="mcp.response.size",
            aggregation=ExplicitBucketHistogramAggregation(boundaries=_MCP_PAYLOAD_SIZE_BUCKETS),
        )

        # Create Views for streaming metrics (Phase 3.4)
        ttft_view = View(
            instrument_name=SC.GEN_AI_SERVER_TTFT,
            aggregation=ExplicitBucketHistogramAggregation(boundaries=_GEN_AI_SERVER_TFTT),
        )

        tbt_view = View(
            instrument_name=SC.GEN_AI_SERVER_TBT,
            aggregation=ExplicitBucketHistogramAggregation(boundaries=_GEN_AI_SERVER_TBT),
        )

        # Create Views for token distribution histograms
        prompt_tokens_view = View(
            instrument_name="gen_ai.client.token.usage.prompt",
            aggregation=ExplicitBucketHistogramAggregation(
                boundaries=_GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS
            ),
        )

        completion_tokens_view = View(
            instrument_name="gen_ai.client.token.usage.completion",
            aggregation=ExplicitBucketHistogramAggregation(
                boundaries=_GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS
            ),
        )

        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader],
            views=[
                duration_view,
                mcp_duration_view,
                mcp_request_size_view,
                mcp_response_size_view,
                ttft_view,
                tbt_view,
                prompt_tokens_view,
                completion_tokens_view,
            ],
        )
        metrics.set_meter_provider(meter_provider)
        logger.info("No OTLP endpoint configured, metrics will be exported to console.")

    # OpenInference instrumentors that use different API (no config parameter)
    # Only include if OpenInference is available (Python >= 3.10)
    OPENINFERENCE_INSTRUMENTORS = (
        {"smolagents", "mcp", "litellm"} if OPENINFERENCE_AVAILABLE else set()
    )

    # Auto-instrument LLM libraries based on the configuration
    for name in config.enabled_instrumentors:
        if name in INSTRUMENTORS:
            try:
                instrumentor_class = INSTRUMENTORS[name]
                instrumentor = instrumentor_class()

                # OpenInference instrumentors don't take config parameter
                if name in OPENINFERENCE_INSTRUMENTORS:
                    instrumentor.instrument()
                else:
                    instrumentor.instrument(config=config)

                logger.info(f"{name} instrumentation enabled")
            except Exception as e:
                logger.error(f"Failed to instrument {name}: {e}", exc_info=True)
                if config.fail_on_error:
                    raise
        else:
            logger.warning(f"Unknown instrumentor '{name}' requested.")

    # Auto-instrument MCP tools (databases, APIs, etc.)
    # NOTE: OTLP endpoints are excluded via OTEL_PYTHON_REQUESTS_EXCLUDED_URLS set above
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
            gpu_collector = GPUMetricsCollector(
                meter_provider.get_meter("genai.gpu"),
                config,
                interval=config.gpu_collection_interval,
            )
            gpu_collector.start()
            logger.info(
                f"GPU metrics collection started (interval: {config.gpu_collection_interval}s)."
            )
        except Exception as e:
            logger.error(f"Failed to start GPU metrics collection: {e}", exc_info=True)
            if config.fail_on_error:
                raise

    # Initialize server metrics collector (KV cache, request queue, etc.)
    try:
        meter_provider = metrics.get_meter_provider()
        initialize_server_metrics(meter_provider.get_meter("genai.server"))
        logger.info("Server metrics collector initialized (KV cache, request queue)")
    except Exception as e:
        logger.error(f"Failed to initialize server metrics: {e}", exc_info=True)
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
