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

from .config import EVALUATION_MODE_INLINE, OTelConfig
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
from .evaluation_enriching_exporter import EvaluationEnrichingSpanExporter
from .gpu_metrics import GPUMetricsCollector
from .litellm_span_enrichment_processor import LiteLLMSpanEnrichmentProcessor
from .mcp_instrumentors import MCPInstrumentorManager
from .mcp_span_enrichment_processor import MCPSpanEnrichmentProcessor
from .resource import build_resource
from .server_metrics import initialize_server_metrics
from .smolagents_span_enrichment_processor import SmolagentsSpanEnrichmentProcessor
from .metrics import (
    _GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS,
    _GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS,
    _GEN_AI_SERVER_TBT,
    _GEN_AI_SERVER_TFTT,
    _MCP_CLIENT_OPERATION_DURATION_BUCKETS,
    _MCP_PAYLOAD_SIZE_BUCKETS,
)

from .semconv import SemanticConvention as SC

# Import instrumentors - fix the import path based on your actual structure
try:
    from .instrumentors import (
        AnthropicInstrumentor,
        AntigravityInstrumentor,
        AnyscaleInstrumentor,
        AutoGenAgentChatInstrumentor,
        AutoGenInstrumentor,
        AWSBedrockInstrumentor,
        AzureAIInferenceInstrumentor,
        AzureOpenAIInstrumentor,
        BedrockAgentsInstrumentor,
        CohereInstrumentor,
        CometAPIInstrumentor,
        CrewAIInstrumentor,
        DatabricksInstrumentor,
        DSPyInstrumentor,
        ElevenLabsInstrumentor,
        GoogleADKInstrumentor,
        GoogleAIInstrumentor,
        GroqInstrumentor,
        GuardrailsAIInstrumentor,
        HaystackInstrumentor,
        HuggingFaceInstrumentor,
        HyperbolicInstrumentor,
        InstructorInstrumentor,
        LangChainInstrumentor,
        LangGraphInstrumentor,
        LiteLLMLatencyInstrumentor,
        LiquidAudioInstrumentor,
        LlamaCppInstrumentor,
        LlamaIndexInstrumentor,
        MistralAIInstrumentor,
        OllamaInstrumentor,
        OpenAIInstrumentor,
        OpenAIAgentsInstrumentor,
        OpenRouterInstrumentor,
        PydanticAIInstrumentor,
        VLLMInstrumentor,
        XAIInstrumentor,
        ReplicateInstrumentor,
        SambaNovaInstrumentor,
        SarvamAIInstrumentor,
        SentenceTransformersInstrumentor,
        StrandsHarnessInstrumentor,
        TogetherAIInstrumentor,
        TypeSafeInstrumentor,
        VertexAIInstrumentor,
    )
except ImportError:
    # Fallback for testing or if instrumentors are in different structure
    from genai_otel.instrumentors import (
        AnthropicInstrumentor,
        AntigravityInstrumentor,
        AnyscaleInstrumentor,
        AutoGenAgentChatInstrumentor,
        AutoGenInstrumentor,
        AWSBedrockInstrumentor,
        AzureAIInferenceInstrumentor,
        AzureOpenAIInstrumentor,
        BedrockAgentsInstrumentor,
        CohereInstrumentor,
        CometAPIInstrumentor,
        CrewAIInstrumentor,
        DatabricksInstrumentor,
        DSPyInstrumentor,
        ElevenLabsInstrumentor,
        GoogleADKInstrumentor,
        GoogleAIInstrumentor,
        GroqInstrumentor,
        GuardrailsAIInstrumentor,
        HaystackInstrumentor,
        HuggingFaceInstrumentor,
        HyperbolicInstrumentor,
        InstructorInstrumentor,
        LangChainInstrumentor,
        LangGraphInstrumentor,
        LiteLLMLatencyInstrumentor,
        LiquidAudioInstrumentor,
        LlamaCppInstrumentor,
        LlamaIndexInstrumentor,
        MistralAIInstrumentor,
        OllamaInstrumentor,
        OpenAIInstrumentor,
        OpenAIAgentsInstrumentor,
        OpenRouterInstrumentor,
        PydanticAIInstrumentor,
        VLLMInstrumentor,
        XAIInstrumentor,
        ReplicateInstrumentor,
        SambaNovaInstrumentor,
        SarvamAIInstrumentor,
        SentenceTransformersInstrumentor,
        StrandsHarnessInstrumentor,
        TogetherAIInstrumentor,
        TypeSafeInstrumentor,
        VertexAIInstrumentor,
    )

try:
    from .mcp_semconv.client_instrumentor import MCPClientInstrumentor
except ImportError:  # pragma: no cover - mirrors the fallback used above
    from genai_otel.mcp_semconv.client_instrumentor import MCPClientInstrumentor

logger = logging.getLogger(__name__)

# OpenInference instrumentors are loaded lazily to avoid importing litellm (~10s)
# at module import time. They are resolved in setup_auto_instrumentation() instead.
OPENINFERENCE_AVAILABLE = None  # Tri-state: None=not checked, True/False=result

# Shipped in DEFAULT_INSTRUMENTORS but only resolvable once OpenInference is
# importable. Absent on a default install, which is expected, not a misconfiguration.
_OPENINFERENCE_OPTIONAL = frozenset({"smolagents", "litellm", "mcp"})

# Re-entrancy guard to prevent double-instrumentation
_INSTRUMENTATION_INITIALIZED = False
# Track active providers/collectors for uninstrument()
_active_tracer_provider = None
_active_meter_provider = None
_active_gpu_collector = None


def _check_openinference():
    """Lazily check and import OpenInference instrumentors (requires Python >= 3.10)."""
    global OPENINFERENCE_AVAILABLE
    if OPENINFERENCE_AVAILABLE is not None:
        return OPENINFERENCE_AVAILABLE
    try:
        from openinference.instrumentation.litellm import LiteLLMInstrumentor  # noqa: E402
        from openinference.instrumentation.mcp import MCPInstrumentor  # noqa: E402
        from openinference.instrumentation.smolagents import (  # noqa: E402
            SmolagentsInstrumentor,
        )

        INSTRUMENTORS.update(
            {
                "smolagents": SmolagentsInstrumentor,
                "litellm": LiteLLMInstrumentor,
                "mcp": MCPInstrumentor,
            }
        )
        OPENINFERENCE_AVAILABLE = True
    except ImportError:
        OPENINFERENCE_AVAILABLE = False
    return OPENINFERENCE_AVAILABLE


# Defines the available instrumentors. This is now at the module level for easier mocking in tests.
INSTRUMENTORS = {
    "antigravity": AntigravityInstrumentor,  # Google Antigravity agent SDK
    "openai": OpenAIInstrumentor,
    "agents": OpenAIAgentsInstrumentor,  # OpenAI Agents SDK
    "openai_agents": OpenAIAgentsInstrumentor,  # OpenAI Agents SDK (alias)
    "openrouter": OpenRouterInstrumentor,  # OpenRouter unified API aggregator
    "cometapi": CometAPIInstrumentor,  # CometAPI all-in-one API aggregator (500+ models)
    "anthropic": AnthropicInstrumentor,
    "google.generativeai": GoogleAIInstrumentor,
    "google_adk": GoogleADKInstrumentor,  # Google Agent Development Kit
    "boto3": AWSBedrockInstrumentor,
    "azure.ai.inference": AzureAIInferenceInstrumentor,
    "azure.ai.openai": AzureOpenAIInstrumentor,
    "autogen": AutoGenInstrumentor,  # AutoGen multi-agent framework (legacy)
    "autogen_agentchat": AutoGenAgentChatInstrumentor,  # AutoGen AgentChat v0.4+
    "bedrock_agents": BedrockAgentsInstrumentor,  # AWS Bedrock Agents
    "cohere": CohereInstrumentor,
    "crewai": CrewAIInstrumentor,  # CrewAI multi-agent framework
    "databricks": DatabricksInstrumentor,  # Databricks Model Serving and Foundation Model APIs
    # Streaming latency for litellm routes that bypass provider SDKs (issue #22).
    # Distinct from the OpenInference "litellm" entry, which stays as-is.
    "litellm_latency": LiteLLMLatencyInstrumentor,
    "dspy": DSPyInstrumentor,  # DSPy declarative LM programming framework
    "mistralai": MistralAIInstrumentor,
    "together": TogetherAIInstrumentor,
    "groq": GroqInstrumentor,
    "guardrails": GuardrailsAIInstrumentor,  # Guardrails AI validation framework
    "haystack": HaystackInstrumentor,  # Haystack NLP pipeline framework
    "instructor": InstructorInstrumentor,  # Instructor structured output extraction
    "llamacpp": LlamaCppInstrumentor,
    "ollama": OllamaInstrumentor,
    "vllm": VLLMInstrumentor,
    "vertexai": VertexAIInstrumentor,
    "xai_sdk": XAIInstrumentor,
    "replicate": ReplicateInstrumentor,
    "anyscale": AnyscaleInstrumentor,
    "sambanova": SambaNovaInstrumentor,
    "sarvamai": SarvamAIInstrumentor,  # Sarvam AI sovereign Indian AI platform
    "elevenlabs": ElevenLabsInstrumentor,  # ElevenLabs TTS + Scribe speech-to-text
    "hyperbolic": HyperbolicInstrumentor,
    "langchain": LangChainInstrumentor,
    "langgraph": LangGraphInstrumentor,  # LangGraph stateful workflow framework
    "llama_index": LlamaIndexInstrumentor,
    "pydantic_ai": PydanticAIInstrumentor,  # Pydantic AI type-safe agent framework
    "transformers": HuggingFaceInstrumentor,
    "sentence_transformers": SentenceTransformersInstrumentor,
    "liquid_audio": LiquidAudioInstrumentor,
    "typesafe": TypeSafeInstrumentor,
    "strands": StrandsHarnessInstrumentor,  # Strands Harness (opt-in)
    # MCP client attribution (mcp.* / commerce.* semantic conventions). Opt-in
    # rather than default: it is most useful with a tool schema map attached,
    # which only the application can supply. See genai_otel.mcp_semconv.
    "mcp_client": MCPClientInstrumentor,
}

# OpenInference instrumentors (smolagents, litellm, mcp) are added to INSTRUMENTORS
# lazily via _check_openinference() called in setup_auto_instrumentation()


def _setup_meter_provider(resource, metric_reader):
    """Create and configure the MeterProvider with histogram Views.

    Args:
        resource: OpenTelemetry Resource for the meter provider.
        metric_reader: The metric reader (OTLP or Console).
    """
    views = [
        View(
            instrument_name=SC.GEN_AI_CLIENT_OPERATION_DURATION,
            aggregation=ExplicitBucketHistogramAggregation(
                boundaries=_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS
            ),
        ),
        View(
            instrument_name="mcp.client.operation.duration",
            aggregation=ExplicitBucketHistogramAggregation(
                boundaries=_MCP_CLIENT_OPERATION_DURATION_BUCKETS
            ),
        ),
        View(
            instrument_name="mcp.request.size",
            aggregation=ExplicitBucketHistogramAggregation(boundaries=_MCP_PAYLOAD_SIZE_BUCKETS),
        ),
        View(
            instrument_name="mcp.response.size",
            aggregation=ExplicitBucketHistogramAggregation(boundaries=_MCP_PAYLOAD_SIZE_BUCKETS),
        ),
        View(
            instrument_name=SC.GEN_AI_SERVER_TTFT,
            aggregation=ExplicitBucketHistogramAggregation(boundaries=_GEN_AI_SERVER_TFTT),
        ),
        View(
            instrument_name=SC.GEN_AI_SERVER_TBT,
            aggregation=ExplicitBucketHistogramAggregation(boundaries=_GEN_AI_SERVER_TBT),
        ),
        View(
            instrument_name="gen_ai.client.token.usage.prompt",
            aggregation=ExplicitBucketHistogramAggregation(
                boundaries=_GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS
            ),
        ),
        View(
            instrument_name="gen_ai.client.token.usage.completion",
            aggregation=ExplicitBucketHistogramAggregation(
                boundaries=_GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS
            ),
        ),
        View(
            instrument_name=SC.GEN_AI_CLIENT_INFERENCE_OPERATION_INPUT_TOKENS,
            aggregation=ExplicitBucketHistogramAggregation(
                boundaries=_GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS
            ),
        ),
        View(
            instrument_name=SC.GEN_AI_CLIENT_INFERENCE_OPERATION_OUTPUT_TOKENS,
            aggregation=ExplicitBucketHistogramAggregation(
                boundaries=_GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS
            ),
        ),
    ]
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader], views=views)
    metrics.set_meter_provider(meter_provider)
    return meter_provider


def _evaluation_configs(config: OTelConfig) -> dict:
    """The detector configurations for the evaluation span processor, by keyword.

    One per enabled detector, built from ``config`` alone, so that what was
    passed to ``instrument()`` is honoured and not only what the environment says.
    """
    cache_size = config.evaluation_cache_size
    configs = {}
    if config.enable_pii_detection:
        configs["pii_config"] = PIIConfig(
            enabled=True,
            mode=PIIMode(config.pii_mode),
            threshold=config.pii_threshold,
            gdpr_mode=config.pii_gdpr_mode,
            hipaa_mode=config.pii_hipaa_mode,
            pci_dss_mode=config.pii_pci_dss_mode,
            custom_patterns=config.pii_custom_patterns,
            # Plumb hardening settings so programmatic OTelConfig overrides
            # (not just env vars) are honored.
            air_gapped=config.air_gapped,
            content_max_length=config.content_max_length,
            cache_size=cache_size,
        )
    if config.enable_toxicity_detection:
        configs["toxicity_config"] = ToxicityConfig(
            enabled=True,
            threshold=config.toxicity_threshold,
            use_perspective_api=config.toxicity_use_perspective_api,
            perspective_api_key=config.toxicity_perspective_api_key,
            block_on_detection=config.toxicity_block_on_detection,
            # Plumb third-party egress / air-gap controls from OTelConfig so
            # programmatic overrides (not just env vars) are honored.
            allow_external_egress=config.allow_external_egress,
            air_gapped=config.air_gapped,
            quantize=config.toxicity_quantize,
            torch_threads=config.toxicity_torch_threads,
            cache_size=cache_size,
        )
    if config.enable_bias_detection:
        configs["bias_config"] = BiasConfig(
            enabled=True,
            threshold=config.bias_threshold,
            block_on_detection=config.bias_block_on_detection,
            cache_size=cache_size,
        )
    if config.enable_prompt_injection_detection:
        configs["prompt_injection_config"] = PromptInjectionConfig(
            enabled=True,
            threshold=config.prompt_injection_threshold,
            block_on_detection=config.prompt_injection_block_on_detection,
            cache_size=cache_size,
        )
    if config.enable_restricted_topics:
        configs["restricted_topics_config"] = RestrictedTopicsConfig(
            enabled=True,
            threshold=config.restricted_topics_threshold,
            block_on_detection=config.restricted_topics_block_on_detection,
            cache_size=cache_size,
        )
    if config.enable_hallucination_detection:
        configs["hallucination_config"] = HallucinationConfig(
            enabled=True,
            threshold=config.hallucination_threshold,
            cache_size=cache_size,
        )
    return configs


def _any_detector_enabled(config: OTelConfig) -> bool:
    return any(
        [
            config.enable_pii_detection,
            config.enable_toxicity_detection,
            config.enable_bias_detection,
            config.enable_prompt_injection_detection,
            config.enable_restricted_topics,
            config.enable_hallucination_detection,
        ]
    )


def _evaluate_inline_under_another_provider(config: OTelConfig) -> None:
    """Run the detectors inline when the tracer provider in use is not this library's.

    OpenTelemetry keeps the first tracer provider that is set. One that the
    application installed before ``instrument()`` therefore stays in use, and
    the one created here never sees a span.

    In the background mode a span is ended by a worker thread once its
    detectors are done, some time after the call returned. It is exported only
    if whatever flushes or shuts the provider down waits for it first. This
    library sees to that for a provider it owns: its span processor is on it
    and its exit hook runs before the provider's. The application's own
    ``force_flush()`` and ``shutdown()`` know nothing of the spans still with
    their detectors, and those spans were lost. Inline, a span has ended by the
    time its call returns, as in every release before the background mode.

    Decided once, here, so that every call does the same thing.
    """
    if config.evaluation_mode == EVALUATION_MODE_INLINE or not _any_detector_enabled(config):
        return
    config.evaluation_mode = EVALUATION_MODE_INLINE
    logger.warning(
        "A tracer provider was already installed when genai_otel.instrument() was called, "
        "so the evaluation detectors will run inline, on the thread that makes each call "
        "(as with GENAI_EVALUATION_MODE=inline). In the background mode a span is exported "
        "by a worker thread when its detectors are done, and a provider this library did "
        "not create does not wait for those spans when it is flushed or shut down. To run "
        "the detectors in the background, let genai_otel.instrument() create the tracer "
        "provider: call it before anything else sets one."
    )


def setup_auto_instrumentation(config: OTelConfig):
    """
    Set up OpenTelemetry with auto-instrumentation for LLM frameworks and MCP tools.

    Args:
        config: OTelConfig instance with configuration parameters.
    """
    global _INSTRUMENTATION_INITIALIZED
    if _INSTRUMENTATION_INITIALIZED:
        logger.warning("Auto-instrumentation already initialized, skipping re-initialization")
        return
    logger.info("Starting auto-instrumentation setup...")

    # Lazily check and register OpenInference instrumentors (smolagents, litellm, mcp)
    # This defers the heavy litellm import (~10s) to instrumentation time, not package import.
    _check_openinference()

    # Configure OpenTelemetry SDK (TracerProvider, MeterProvider, etc.)
    import os

    # Default metric exemplar sampling OFF. Exemplars (trace-metric correlation
    # samples) cost a per-measurement RNG draw on the hot path - roughly a third
    # of per-call metric overhead - which is not worth it for high-throughput
    # (bank-scale) deployments. Operators who want exemplars can opt back in with
    # OTEL_METRICS_EXEMPLAR_FILTER=trace_based|always_on. setdefault respects any
    # explicit setting.
    os.environ.setdefault("OTEL_METRICS_EXEMPLAR_FILTER", "always_off")

    from .__version__ import __version__ as _pkg_version

    # Host, OS, process and instance identity, so that traffic can be attributed
    # to a machine and to one instance among several on it. See genai_otel/resource.py
    # for the attribute set and the OTEL_* variables that control it.
    resource = build_resource(config.service_name, _pkg_version, profile=config.profile)

    # Configure Tracing
    global _active_tracer_provider, _active_meter_provider

    # Apply sampling rate if configured (< 1.0 means probabilistic sampling)
    sampler = None
    if config.sampling_rate < 1.0:
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

        sampler = TraceIdRatioBased(config.sampling_rate)
        logger.info("Trace sampling rate set to %s", config.sampling_rate)

    tracer_provider = TracerProvider(resource=resource, sampler=sampler)
    _active_tracer_provider = tracer_provider
    trace.set_tracer_provider(tracer_provider)
    if trace.get_tracer_provider() is not tracer_provider:
        _evaluate_inline_under_another_provider(config)

    # Spans whose detectors run after the call are ended by a worker thread. At
    # exit they have to end before this provider shuts down, or they are
    # dropped. Exit hooks run last-registered first and the provider registered
    # its own a moment ago, so this one is registered now.
    try:
        from .instrumentors import base as _base

        _base.register_evaluation_exit_hook()
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to register the evaluation exit hook: %s", e, exc_info=True)
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    set_global_textmap(TraceContextTextMapPropagator())

    # Registered first, so every span has its usage holder before any wrapper
    # reads it. This is what lets a framework span leave the tokens and cost to
    # the provider span beneath it instead of reporting the request twice.
    try:
        from .instrumentors.base import UsageHolderSpanProcessor

        tracer_provider.add_span_processor(UsageHolderSpanProcessor())
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to add usage holder processor: %s", e, exc_info=True)

    # Add cost enrichment processor for custom instrumentors (OpenAI, Ollama, etc.)
    # These instrumentors set cost attributes directly, so processor is mainly for logging
    # Also attempts to enrich OpenInference spans (smolagents, litellm, mcp), though
    # the processor can't modify ReadableSpan - the exporter below handles that
    cost_calculator = None
    if config.enable_cost_tracking:
        try:
            # Reuse the process-wide calculator rather than building another
            # one. Both are default-priced, so this is behaviour-identical, but
            # a fresh instance rebuilds the whole exact/substring index over
            # every priced model - allocation that repeats on each
            # setup_auto_instrumentation() call and scales with the size of the
            # pricing table.
            from .instrumentors.base import _get_default_cost_calculator

            cost_calculator = _get_default_cost_calculator()
            cost_processor = CostEnrichmentSpanProcessor(cost_calculator)
            tracer_provider.add_span_processor(cost_processor)
            logger.info("Cost enrichment processor added")
        except Exception as e:
            logger.warning("Failed to add cost enrichment processor: %s", e, exc_info=True)

    # Add LiteLLM span enrichment processor for evaluation support
    # This enriches LiteLLM spans (created by OpenInference) with standardized attributes
    # enabling evaluation metrics (PII, toxicity, bias, etc.) for LiteLLM calls
    if OPENINFERENCE_AVAILABLE and "litellm" in config.enabled_instrumentors:
        try:
            litellm_enrichment_processor = LiteLLMSpanEnrichmentProcessor()
            tracer_provider.add_span_processor(litellm_enrichment_processor)
            logger.info("LiteLLM span enrichment processor added for evaluation support")
        except Exception as e:
            logger.warning("Failed to add LiteLLM span enrichment processor: %s", e, exc_info=True)

    # Add Smolagents span enrichment processor for evaluation support
    # This enriches Smolagents spans (created by OpenInference) with standardized attributes
    if OPENINFERENCE_AVAILABLE and "smolagents" in config.enabled_instrumentors:
        try:
            smolagents_enrichment_processor = SmolagentsSpanEnrichmentProcessor()
            tracer_provider.add_span_processor(smolagents_enrichment_processor)
            logger.info("Smolagents span enrichment processor added for evaluation support")
        except Exception as e:
            logger.warning(
                "Failed to add Smolagents span enrichment processor: %s", e, exc_info=True
            )

    # Add MCP span enrichment processor for evaluation support
    # This enriches MCP tool spans (created by OpenInference) with standardized attributes
    if OPENINFERENCE_AVAILABLE and "mcp" in config.enabled_instrumentors:
        try:
            mcp_enrichment_processor = MCPSpanEnrichmentProcessor()
            tracer_provider.add_span_processor(mcp_enrichment_processor)
            logger.info("MCP span enrichment processor added for evaluation support")
        except Exception as e:
            logger.warning("Failed to add MCP span enrichment processor: %s", e, exc_info=True)

    # Add evaluation and safety span processor (v0.2.0)
    # Store the evaluation processor so we can wrap it in the exporter chain
    evaluation_processor = None
    if _any_detector_enabled(config):
        try:
            # Create and add evaluation processor (lazy import to avoid loading
            # heavy ML dependencies like spacy/torch at package import time)
            from .evaluation.span_processor import EvaluationSpanProcessor

            evaluation_processor = EvaluationSpanProcessor(**_evaluation_configs(config))
            tracer_provider.add_span_processor(evaluation_processor)
            logger.info("Evaluation and safety span processor added")
        except Exception as e:
            logger.warning("Failed to add evaluation span processor: %s", e, exc_info=True)

    logger.debug("OTelConfig endpoint: %s", config.endpoint)
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
        logger.info("Excluded OTLP endpoints from instrumentation: %s", base_url)

        # Set timeout in environment variable as integer string (OTLP exporters expect int)
        os.environ["OTEL_EXPORTER_OTLP_TIMEOUT"] = str(timeout)

        # Detect OTLP protocol (grpc or http)
        # Priority: OTEL_EXPORTER_OTLP_PROTOCOL env var > port-based detection > default (http)
        otlp_protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "").lower()
        if not otlp_protocol:
            # Detect from port: 4317 = gRPC, 4318 = HTTP
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
            logger.info("Using OTLP gRPC protocol (endpoint: %s)", base_url)
        else:
            # Default to HTTP
            span_exporter = OTLPSpanExporter(
                headers=config.headers,
            )
            metric_exporter = OTLPMetricExporter(
                headers=config.headers,
            )
            logger.info("Using OTLP HTTP protocol (endpoint: %s)", base_url)

        # Build the exporter chain: span_exporter -> cost enrichment -> evaluation enrichment
        # Each wrapper enriches ReadableSpan objects before passing to the next exporter
        final_exporter = span_exporter

        # Wrap with cost enrichment if cost tracking is enabled
        if cost_calculator:
            final_exporter = CostEnrichingSpanExporter(final_exporter, cost_calculator)
            logger.info("Cost-enriching span exporter enabled for OTLP")

        # Wrap with evaluation enrichment if evaluation features are enabled
        if evaluation_processor:
            final_exporter = EvaluationEnrichingSpanExporter(final_exporter, evaluation_processor)
            logger.info("Evaluation-enriching span exporter enabled for OTLP")

        tracer_provider.add_span_processor(BatchSpanProcessor(final_exporter))
        logger.info(
            "OpenTelemetry tracing configured with OTLP endpoint: %s", span_exporter._endpoint
        )

        # Configure Metrics with Views for histogram buckets
        metric_reader = PeriodicExportingMetricReader(exporter=metric_exporter)
        _active_meter_provider = _setup_meter_provider(resource, metric_reader)
        logger.info(
            "OpenTelemetry metrics configured with OTLP endpoint: %s", metric_exporter._endpoint
        )
    else:
        # Configure Console Exporters if no OTLP endpoint is set
        span_exporter = ConsoleSpanExporter()
        # Build the exporter chain for console output
        final_exporter = span_exporter

        if cost_calculator:
            final_exporter = CostEnrichingSpanExporter(final_exporter, cost_calculator)
            logger.info("Cost-enriching span exporter enabled for console")

        if evaluation_processor:
            final_exporter = EvaluationEnrichingSpanExporter(final_exporter, evaluation_processor)
            logger.info("Evaluation-enriching span exporter enabled for console")

        tracer_provider.add_span_processor(BatchSpanProcessor(final_exporter))
        logger.info("No OTLP endpoint configured, traces will be exported to console.")

        metric_exporter = ConsoleMetricExporter()
        metric_reader = PeriodicExportingMetricReader(exporter=metric_exporter)
        _active_meter_provider = _setup_meter_provider(resource, metric_reader)
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

                logger.info("%s instrumentation enabled", name)
            except Exception as e:
                logger.error("Failed to instrument %s: %s", name, e, exc_info=True)
                if config.fail_on_error:
                    raise
        else:
            # smolagents/litellm/mcp ship in DEFAULT_INSTRUMENTORS but resolve
            # only when OpenInference is importable (Python 3.10+ plus the extra).
            # On a default install they are simply absent, and calling that a
            # warning made a clean first run look like the user had misconfigured
            # something. Only warn for a name the user actually asked for.
            if name in _OPENINFERENCE_OPTIONAL and not _check_openinference():
                logger.debug("Instrumentor '%s' needs the OpenInference extra; skipping.", name)
            else:
                logger.warning("Unknown instrumentor '%s' requested.", name)

    # Decision endpoints called over plain HTTP (/v1/systemone, /v1/decisions).
    # Not an entry in INSTRUMENTORS: it covers no library of its own, and it is
    # governed by GENAI_DECISION_HTTP_MATCHER rather than the instrumentor list.
    # The manual decision_span helper is pointed at the same configuration.
    try:
        from . import decision as _decision_helper
        from .instrumentors.decision_http import DecisionHTTPInstrumentor

        _decision_helper.configure(config)
        if config.decision_http_matcher:
            DecisionHTTPInstrumentor().instrument(config)
            logger.info("Decision HTTP path matcher enabled")
    except Exception as e:
        logger.error("Failed to set up decision instrumentation: %s", e, exc_info=True)
        if config.fail_on_error:
            raise

    # Auto-instrument MCP tools (databases, APIs, etc.)
    # NOTE: OTLP endpoints are excluded via OTEL_PYTHON_REQUESTS_EXCLUDED_URLS set above
    if config.enable_mcp_instrumentation:
        try:
            mcp_manager = MCPInstrumentorManager(config)
            mcp_manager.instrument_all(config.fail_on_error)
            logger.info("MCP tools instrumentation enabled and set up.")
        except Exception as e:
            logger.error("Failed to set up MCP tools instrumentation: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    # Start GPU metrics collection if enabled
    if config.enable_gpu_metrics:
        try:
            meter_provider = metrics.get_meter_provider()
            global _active_gpu_collector
            _active_gpu_collector = GPUMetricsCollector(
                meter_provider.get_meter("genai.gpu"),
                config,
                interval=config.gpu_collection_interval,
            )
            _active_gpu_collector.start()
            logger.info(
                "GPU metrics collection started (interval: %ss).", config.gpu_collection_interval
            )
        except Exception as e:
            logger.error("Failed to start GPU metrics collection: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    # Initialize server metrics collector (KV cache, request queue, etc.)
    try:
        meter_provider = metrics.get_meter_provider()
        initialize_server_metrics(meter_provider.get_meter("genai.server"))
        logger.info("Server metrics collector initialized (KV cache, request queue)")
    except Exception as e:
        logger.error("Failed to initialize server metrics: %s", e, exc_info=True)
        if config.fail_on_error:
            raise

    # One actionable line beats a repeating wall of connection stack traces
    # from the exporter's retry loop. Checked after setup so it does not delay
    # instrumentation, and only when the endpoint was never configured.
    try:
        _warn_if_default_collector_unreachable(config)
    except Exception as e:  # noqa: BLE001
        logger.debug("Collector reachability check skipped: %s", e)

    # Install last, so a handler is only registered once the providers it would
    # flush actually exist.
    if config.flush_on_sigterm:
        try:
            _install_sigterm_flush_handler(config)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to install SIGTERM flush handler: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    _INSTRUMENTATION_INITIALIZED = True
    logger.info("Auto-instrumentation setup complete")

    # Report anonymous usage telemetry (opt-in only, see telemetry.py)
    try:
        from .telemetry import report_usage

        report_usage(config)
    except Exception:
        pass  # Never let telemetry affect instrumentation


DEFAULT_OTLP_ENDPOINT = "http://localhost:4318"


def _warn_if_default_collector_unreachable(config: OTelConfig) -> bool:
    """Say once, clearly, that the default collector is not there.

    Only fires when the endpoint was never configured. Someone who set
    OTEL_EXPORTER_OTLP_ENDPOINT themselves knows where their collector is and
    does not need advice; someone who did not is about to receive a repeating
    wall of connection stack traces from the exporter's retry loop with no
    indication of what to do about it.

    This deliberately does NOT quiet those retries. A silent export failure is
    how telemetry disappears without anyone noticing - the same failure this
    library works to prevent elsewhere. The retries stay loud; this only adds
    the one line that says what to do.

    Returns True when the warning was emitted.
    """
    import os
    import socket
    from urllib.parse import urlparse

    if os.getenv("GENAI_SKIP_COLLECTOR_CHECK", "").lower() == "true":
        return False
    # Keyed off the resolved endpoint rather than whether the env var is set:
    # setup_auto_instrumentation writes OTEL_EXPORTER_OTLP_ENDPOINT into the
    # environment before this runs, so "did the user set it?" cannot be answered
    # here. Anything other than the default means a deliberate choice, and if
    # someone did type the default themselves the advice is still correct.
    if config.endpoint != DEFAULT_OTLP_ENDPOINT:
        return False

    parsed = urlparse(config.endpoint)
    host, port = parsed.hostname or "localhost", parsed.port or 4318
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return False  # collector is there; nothing to say
    except OSError:
        pass

    logger.warning(
        "No OTLP collector is listening on %s, which is the default endpoint. "
        "Spans and metrics will be recorded but every export will fail. "
        "Start a collector, point OTEL_EXPORTER_OTLP_ENDPOINT at one, or set it "
        "empty to print telemetry to the console instead. "
        "Set GENAI_SKIP_COLLECTOR_CHECK=true to silence this check.",
        config.endpoint,
    )
    return True


def flush_telemetry(timeout_seconds: float = 5.0) -> bool:
    """Force-flush the active providers. Returns True if everything flushed.

    Exposed publicly so an application that already owns its shutdown path can
    drain telemetry itself instead of enabling the SIGTERM handler.

    Spans still waiting for their evaluation detectors are waited for first, for
    up to the same timeout: they are not exported until the detectors are done.
    That is the spans of calls that have returned by now. Calls that other
    threads make while this waits are not waited for.
    """
    ok = True
    try:
        from .instrumentors import base as _base

        if _base.wait_for_pending_evaluations(timeout_seconds) is False:
            ok = False
    except Exception as e:  # noqa: BLE001
        logger.debug("Waiting for pending evaluations failed: %s", e)
        ok = False
    return _flush_providers(timeout_seconds) and ok


def _flush_providers(timeout_seconds: float) -> bool:
    """Force-flush the active providers as they are. True if everything flushed."""
    timeout_ms = max(1, int(timeout_seconds * 1000))
    ok = True
    for provider in (_active_tracer_provider, _active_meter_provider):
        if provider is None:
            continue
        try:
            if provider.force_flush(timeout_ms) is False:
                ok = False
        except Exception as e:  # noqa: BLE001
            logger.debug("force_flush failed during shutdown: %s", e)
            ok = False
    return ok


def _install_sigterm_flush_handler(config: OTelConfig) -> bool:
    """Flush pending telemetry when the process is terminated by SIGTERM.

    The SDK's atexit hook covers a clean exit and an uncaught exception, because
    Python runs atexit handlers on both. It does not run them when the process is
    killed by a signal, so `docker stop` and Kubernetes pod eviction drop whatever
    is still queued - silently, which is the point of handling it.

    Returns True when a handler was installed.

    Three things this deliberately does not do:

    * It does not replace an existing handler. Whatever was registered before is
      called after the flush, so an application with its own graceful shutdown
      keeps it.
    * It does not raise when called off the main thread. ``signal.signal`` only
      works there, and instrumentation must never break the host application, so
      an unusable slot is logged and skipped.
    * It does not swallow the signal. After flushing, the default disposition is
      restored and the signal re-raised, so the process still dies with the
      conventional 143 exit status instead of appearing to ignore SIGTERM.
    """
    import os
    import signal
    import threading

    if threading.current_thread() is not threading.main_thread():
        logger.warning(
            "GENAI_FLUSH_ON_SIGTERM is set but instrument() was called off the main "
            "thread, where signal handlers cannot be installed. Telemetry queued at "
            "SIGTERM will be lost; call genai_otel.flush_telemetry() from your own "
            "shutdown path instead."
        )
        return False

    try:
        previous = signal.getsignal(signal.SIGTERM)
    except (ValueError, AttributeError) as e:  # pragma: no cover - platform dependent
        logger.warning("Cannot install SIGTERM flush handler on this platform: %s", e)
        return False

    def _on_sigterm(signum, frame):
        finished = False
        try:
            # No exit hook runs for a process killed by a signal. Spans still
            # waiting for their detectors are given the same grace period and
            # then exported unevaluated, so that the flush below has them.
            #
            # Not on this thread. A handler runs on the main thread, between
            # two of its instructions; if it was handing a span to the
            # evaluation queue just then it holds the queue's lock, and
            # finishing here would be the thread waiting for itself.
            from .instrumentors import base as _base

            finished = _base.finish_pending_evaluations_in_signal_handler(
                config.sigterm_flush_timeout
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("Finishing pending evaluations at SIGTERM failed: %s", e)

        try:
            if finished:
                flushed = flush_telemetry(config.sigterm_flush_timeout)
            else:
                # The detectors needed longer than there was, or the queue
                # could not be got to. Waiting on it again would not help:
                # what has ended is exported as it is.
                _flush_providers(config.sigterm_flush_timeout)
                flushed = False
            logger.info(
                "SIGTERM received, telemetry flush %s", "complete" if flushed else "partial"
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("SIGTERM flush failed: %s", e)

        # Chain to whatever was registered before us, then fall through to the
        # default so the exit status stays conventional.
        if callable(previous):
            previous(signum, frame)
            return
        if previous == signal.SIG_IGN:
            return
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except (ValueError, OSError, RuntimeError) as e:
        logger.warning("Failed to install SIGTERM flush handler: %s", e)
        return False

    logger.info("SIGTERM flush handler installed (timeout %.1fs)", config.sigterm_flush_timeout)
    return True


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


def uninstrument():
    """Shut down OpenTelemetry providers and allow re-instrumentation.

    Stops GPU metrics collection, flushes and shuts down the TracerProvider
    and MeterProvider, and resets the initialization guard so that
    ``instrument()`` can be called again.
    """
    global _INSTRUMENTATION_INITIALIZED, _active_tracer_provider
    global _active_meter_provider, _active_gpu_collector

    if not _INSTRUMENTATION_INITIALIZED:
        logger.debug("Instrumentation not initialized, nothing to uninstrument")
        return

    # Stop GPU collector
    if _active_gpu_collector is not None:
        try:
            _active_gpu_collector.stop()
            logger.info("GPU metrics collector stopped")
        except Exception as e:
            logger.debug("Error stopping GPU collector: %s", e)
        _active_gpu_collector = None

    # Spans still waiting for their evaluation detectors end on a worker
    # thread. Let them, for the configured grace period, and export the rest
    # unevaluated: a span that ends after its provider has shut down is dropped.
    try:
        from .instrumentors import base as _base

        _base.finish_pending_evaluations()
    except Exception as e:  # noqa: BLE001
        logger.debug("Error finishing pending evaluations: %s", e)

    # Shut down TracerProvider (flushes pending spans)
    if _active_tracer_provider is not None:
        try:
            _active_tracer_provider.shutdown()
            logger.info("TracerProvider shut down")
        except Exception as e:
            logger.debug("Error shutting down TracerProvider: %s", e)
        _active_tracer_provider = None

    # Shut down MeterProvider (flushes pending metrics)
    if _active_meter_provider is not None:
        try:
            _active_meter_provider.shutdown()
            logger.info("MeterProvider shut down")
        except Exception as e:
            logger.debug("Error shutting down MeterProvider: %s", e)
        _active_meter_provider = None

    _INSTRUMENTATION_INITIALIZED = False
    logger.info("Instrumentation teardown complete")
