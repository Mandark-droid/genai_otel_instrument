# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Phase 3.4 Fallback Semantic Conventions**
  - Fixed `AttributeError` when `openlit` package is not installed
  - Added missing `GEN_AI_SERVER_TTFT` and `GEN_AI_SERVER_TBT` constants to fallback `SC` class in `base.py`
  - Fixed MCP constant names in `mcp_instrumentors/base.py` to include `_METRIC` suffix
  - Library now works correctly with or without the `openlit` package

- **Third-Party Library Warnings**
  - Suppressed pydantic deprecation warnings from external dependencies
  - Added warning filters in `__init__.py` for runtime suppression
  - Added warning filters in `pyproject.toml` for pytest suppression
  - Clean output with zero warnings in both tests and production use

- **MistralAI Instrumentor Trace Collection**
  - **BREAKING**: Complete rewrite to support Mistral SDK v1.0+ properly
  - Fixed traces not being collected (was only collecting metrics)
  - Changed from class-level patching to instance-level instrumentation (Anthropic pattern)
  - Now wraps `Mistral.__init__` to instrument each client instance
  - Properly instruments: `client.chat.complete()`, `client.chat.stream()`, `client.embeddings.create()`
  - Tests: Simplified to 5 essential tests
  - Verified working with live API calls - traces now collected correctly

- **HuggingFace Instrumentor Trace Collection**
  - Fixed traces not being collected (was only collecting metrics)
  - Fixed incorrect tracer reference (`config.tracer` → `self.tracer`)
  - Properly initialize `self.config` in `instrument()` method
  - Updated to use `tracer.start_span()` instead of deprecated `start_as_current_span()`
  - Added proper span ending with `span.end()`
  - Verified working - traces now collected correctly

### Added

- **Granular Cost Tracking Tests (Phase 3.2 Coverage)**
  - Added 3 comprehensive tests for granular cost tracking functionality
  - `test_granular_cost_tracking_with_all_cost_types` - Tests all 6 cost types (prompt, completion, reasoning, cache_read, cache_write)
  - `test_granular_cost_tracking_with_zero_costs` - Validates zero-cost handling
  - `test_granular_cost_tracking_only_prompt_cost` - Tests embedding/prompt-only scenarios
  - Improved `base.py` coverage from 83% to 91%
  - Total tests: 405 → 408, all passing
  - Overall coverage maintained at 93%

- **OpenTelemetry Semantic Convention Compliance (Phase 1 & 2)**
  - Added support for `OTEL_SEMCONV_STABILITY_OPT_IN` environment variable for dual token attribute emission
  - Added `GENAI_ENABLE_CONTENT_CAPTURE` environment variable for opt-in prompt/completion content capture as span events
  - Added comprehensive span attributes to OpenAI instrumentor:
    - Request parameters: `gen_ai.operation.name`, `gen_ai.request.temperature`, `gen_ai.request.top_p`, `gen_ai.request.max_tokens`, `gen_ai.request.frequency_penalty`, `gen_ai.request.presence_penalty`
    - Response attributes: `gen_ai.response.id`, `gen_ai.response.model`, `gen_ai.response.finish_reasons`
  - Added event-based content capture for prompts and completions (disabled by default for security)
  - Added 8 new tests for Phase 2 enhancements (381 total tests, all passing)

- **Tool/Function Call Instrumentation (Phase 3.1)**
  - Added support for tracking tool/function calls in LLM responses (OpenAI function calling)
  - New span attributes:
    - `llm.tools` - JSON-serialized tool definitions from request
    - `llm.output_messages.{choice_idx}.message.tool_calls.{tc_idx}.tool_call.id` - Tool call ID
    - `llm.output_messages.{choice_idx}.message.tool_calls.{tc_idx}.tool_call.function.name` - Function name
    - `llm.output_messages.{choice_idx}.message.tool_calls.{tc_idx}.tool_call.function.arguments` - Function arguments
  - Enhanced OpenAI instrumentor to extract and record tool call information
  - Added 2 new tests for tool call instrumentation (383 total tests)

- **Granular Cost Tracking (Phase 3.2)**
  - Added granular cost breakdown with separate tracking for:
    - Prompt tokens cost (`gen_ai.usage.cost.prompt`)
    - Completion tokens cost (`gen_ai.usage.cost.completion`)
    - Reasoning tokens cost (`gen_ai.usage.cost.reasoning`) - for OpenAI o1 models
    - Cache read cost (`gen_ai.usage.cost.cache_read`) - for Anthropic prompt caching
    - Cache write cost (`gen_ai.usage.cost.cache_write`) - for Anthropic prompt caching
  - Added 5 new cost-specific metrics counters
  - Added 6 new span attributes for cost breakdown (`gen_ai.usage.cost.*`)
  - Added `calculate_granular_cost()` method to CostCalculator
  - Enhanced OpenAI instrumentor to extract reasoning tokens from `completion_tokens_details.reasoning_tokens`
  - Enhanced Anthropic instrumentor to extract cache tokens (`cache_read_input_tokens`, `cache_creation_input_tokens`)
  - Added 4 new tests for granular cost tracking (387 total tests, all passing)
  - Cost breakdown enables detailed analysis of:
    - OpenAI o1 models with separate reasoning token costs
    - Anthropic prompt caching with read/write cost separation
    - Per-request cost attribution by token type

- **MCP Metrics for Database Operations (Phase 3.3)**
  - Added `BaseMCPInstrumentor` base class with shared MCP-specific metrics
  - New MCP metrics with optimized histogram buckets:
    - `mcp.requests` - Counter for number of MCP requests
    - `mcp.client.operation.duration` - Histogram for operation duration (1ms to 10s buckets)
    - `mcp.request.size` - Histogram for request payload size (100B to 5MB buckets)
    - `mcp.response.size` - Histogram for response payload size (100B to 5MB buckets)
  - Enhanced `DatabaseInstrumentor` to use hybrid approach:
    - Keeps built-in OpenTelemetry instrumentors for full trace/span creation
    - Adds custom wrapt wrappers for MCP metrics collection
    - Instruments PostgreSQL (psycopg2), MongoDB (pymongo), and MySQL (mysql-connector)
  - Configured Views in `auto_instrument.py` to apply MCP histogram bucket boundaries
  - Added 4 new tests for BaseMCPInstrumentor (391 total tests, all passing)
  - Metrics include attributes for `db.system` and `mcp.operation` for filtering

- **Configurable GPU Collection Interval**
  - Added `gpu_collection_interval` configuration option (default: 5 seconds, down from 10)
  - New environment variable: `GENAI_GPU_COLLECTION_INTERVAL`
  - Fixes CO2 metrics not appearing for short-running scripts
  - GPU metrics and CO2 emissions now collected more frequently

- **Streaming Metrics for TTFT and TBT (Phase 3.4)**
  - Added streaming response detection and automatic metrics collection
  - New streaming metrics with optimized histogram buckets:
    - `gen_ai.server.ttft` - Time to First Token histogram (1ms to 10s buckets)
    - `gen_ai.server.tbt` - Time Between Tokens histogram (10ms to 2.5s buckets)
  - New span attribute for streaming:
    - `gen_ai.streaming.token_count` - Total number of chunks/tokens yielded
  - Enhanced `BaseInstrumentor` to detect `stream=True` parameter automatically
  - Added `_wrap_streaming_response()` helper method for streaming iterator wrapping
  - Changed span management from context manager to manual start/end for streaming support
  - Configured Views in `auto_instrument.py` to apply streaming histogram bucket boundaries
  - Added 2 new tests for streaming metrics (405 total tests, all passing)
  - Streaming metrics enable analysis of:
    - Real-time response latency (TTFT)
    - Token generation speed consistency (TBT)
    - Overall streaming performance for user experience optimization

### Changed

- **BREAKING: Metric names now use OpenTelemetry semantic conventions**
  - `genai.requests` → `gen_ai.requests`
  - `genai.tokens` → `gen_ai.client.token.usage`
  - `genai.latency` → `gen_ai.client.operation.duration`
  - `genai.cost` → `gen_ai.usage.cost`
  - `genai.errors` → `gen_ai.client.errors`
  - All GPU metrics now use `gen_ai.gpu.*` prefix (was `genai.gpu.*`)
  - Update your dashboards and alerting rules accordingly
- **Token attribute naming now supports dual emission**
  - When `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai/dup`, both old and new token attributes are emitted:
    - New (always): `gen_ai.usage.prompt_tokens`, `gen_ai.usage.completion_tokens`
    - Old (with /dup): `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`
  - Default (`gen_ai`): Only new attributes are emitted

### Fixed

- **CRITICAL: GPU metrics now use correct metric types and callbacks**
  - Changed `gpu_utilization_counter` from Counter to ObservableGauge (utilization is 0-100%, not monotonic)
  - Fixed `gpu_memory_used_gauge` and `gpu_temperature_gauge` to use callbacks instead of manual `.add()` calls
  - Added callback methods: `_observe_gpu_utilization()`, `_observe_gpu_memory()`, `_observe_gpu_temperature()`
  - Fixed CO2 metric name from `genai.co-2.emissions` to `gen_ai.co2.emissions`
  - Removed dual-thread architecture (now uses single CO2 collection thread, ObservableGauges auto-collected)
  - All GPU metrics now correctly reported with proper data types
  - Updated 19 GPU metrics tests to match new implementation
- **Histogram buckets now properly applied via OpenTelemetry Views**
  - Created View with ExplicitBucketHistogramAggregation for `gen_ai.client.operation.duration`
  - Applies `_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS` from metrics.py
  - Buckets optimized for LLM latencies (0.01s to 81.92s)
  - No longer uses default OTel buckets (which were poorly suited for GenAI workloads)
- **CRITICAL: Made OpenInference instrumentations optional to support Python 3.8 and 3.9**
  - Moved `openinference-instrumentation-smolagents`, `openinference-instrumentation-litellm`, `openinference-instrumentation-mcp`, and `litellm` to optional dependencies
  - These packages require Python >= 3.10 and were causing installation failures on Python 3.8 and 3.9
  - Added new `openinference` optional dependency group for users on Python 3.10+
  - Install with: `pip install genai-otel-instrument[openinference]` (Python 3.10+ only)
  - Package now installs cleanly on Python 3.8, 3.9, 3.10, 3.11, and 3.12
  - Conditional imports prevent errors when OpenInference packages are not installed
  - Relaxed `opentelemetry-semantic-conventions` version constraint from `>=0.58b0` to `>=0.45b0` for Python 3.8 compatibility
  - Added missing `opentelemetry-instrumentation-mysql` to core dependencies
  - Removed `mysql==0.0.3` dependency (requires system MySQL libraries not available in CI)
  - Added `sqlalchemy>=1.4.0` to core dependencies (required by sqlalchemy instrumentor)
- **CRITICAL: Fixed CLI wrapper to execute scripts in same process**
  - Changed from `subprocess.run()` to `runpy.run_path()` to ensure instrumentation hooks are active
  - Supports both `genai-instrument python script.py` and `genai-instrument script.py` formats
  - Script now runs in the same process where instrumentation is initialized, fixing ModuleNotFoundError and ensuring proper telemetry collection
  - Added tests for both CLI usage patterns (7 tests total, all passing)

- **CRITICAL: Fixed MCP dependency conflict error**
  - Removed "mcp" from `DEFAULT_INSTRUMENTORS` list to prevent dependency conflict when mcp library (>= 1.6.0) is not installed
  - Added explanatory comments in `genai_otel/config.py` - users can still enable via `GENAI_ENABLED_INSTRUMENTORS` environment variable
  - Most users don't need the specialized Model Context Protocol library for server/client development
- **Fixed test failures in instrumentor mock tests (11 total failures resolved)**
  - Fixed `test_openai_instrumentor.py::test_instrument_client` - corrected mock to return decorator function instead of wrapped function directly
  - Fixed `test_anthropic_instrumentor.py::test_instrument_client_with_messages` - applied same decorator pattern fix
  - Fixed OpenInference instrumentor tests (litellm, mcp, smolagents) - changed assertions to expect `instrument()` without config parameter, matching actual API in `auto_instrument.py:208-211`
  - Fixed 6 MCP manager test failures in `tests/mcp_instrumentors/test_manager.py` - updated setUp() to enable HTTP instrumentation for tests that expect it
- **All tests now passing: 371 passed, 0 failed, 98% coverage**
- **CRITICAL: Fixed instrumentor null check issues**
  - Added null checks for metrics (`request_counter`, `token_counter`, `cost_counter`) in all instrumentors to prevent `AttributeError: 'NoneType' object has no attribute 'add'`
  - Fixed 9 instrumentors: Ollama, AzureOpenAI, MistralAI, Groq, Cohere, VertexAI, TogetherAI, Replicate
- **CRITICAL: Fixed wrapt decorator issues in OpenAI and Anthropic instrumentors**
  - Fixed `IndexError: tuple index out of range` by properly applying `create_span_wrapper()` decorator to original methods
  - OpenAI instrumentor (`openai_instrumentor.py:82-86`)
  - Anthropic instrumentor (`anthropic_instrumentor.py:76-80`)
- **CRITICAL: Fixed OpenInference instrumentor initialization**
  - Fixed smolagents, litellm, and mcp instrumentors not being called correctly (they don't accept config parameter)
  - Added `OPENINFERENCE_INSTRUMENTORS` set to handle different instrumentation API
  - Added smolagents, litellm, mcp to `DEFAULT_INSTRUMENTORS` list
- **CRITICAL: Fixed OTLP HTTP exporter configuration issues**
  - Fixed `AttributeError: 'function' object has no attribute 'ok'` caused by requests library instrumentation conflicting with OTLP exporters
  - Disabled `RequestsInstrumentor` in MCP manager to prevent breaking OTLP HTTP exporters that use requests internally
  - Disabled requests wrapping in `APIInstrumentor` to avoid class-level Session patching
  - Fixed endpoint configuration to use environment variables so exporters correctly append `/v1/traces` and `/v1/metrics` paths
  - Updated logging to show full endpoints for both trace and metrics exporters
- Corrected indentation and patch targets in `tests/instrumentors/test_ollama_instrumentor.py` to resolve `IndentationError` and `AttributeError`.
- Fixed test failures in `tests/test_metrics.py` by ensuring proper reset of OpenTelemetry providers and correcting assertions.
- Updated `genai_otel/instrumentors/ollama_instrumentor.py` to align with corrected test logic.
- Addressed test failures in `tests/instrumentors/test_huggingface_instrumentor.py` related to missing attributes and call assertions.
- Fix HuggingFace instrumentation to correctly set span attributes and pass tests.
- Resolve `AttributeError` related to `TraceContextTextMapPropagator` in test files by correcting import paths.
- Fixed `setup_meter` function in `genai_otel/metrics.py` to correctly configure OpenTelemetry MeterProvider with metric readers and handle invalid OTLP endpoint/headers gracefully.
- Corrected `tests/test_metrics.py` to properly reset MeterProvider state between tests and accurately access metric exporter attributes, resolving `TypeError` and `AssertionError`s.
- Fixed `cost_counter` not being called in `tests/instrumentors/test_base.py` by ensuring `BaseInstrumentor._shared_cost_counter` is patched with a distinct mock before `ConcreteInstrumentor` instantiation.
- Resolved `setup_tracing` failures in `tests/test_config.py` by correcting `genai_otel/config.py`'s `setup_tracing` function and adjusting the `reset_tracer` fixture to mock `TracerProvider` correctly.
- Refined Hugging Face instrumentation tests for better attribute handling and mock accuracy.
- Improved `tests/test_metrics.py` by ensuring proper isolation of OpenTelemetry providers using `NoOp` implementations in the `reset_otel` fixture.

### Added

- **Comprehensive CI/CD improvements**
  - Added `build-and-install-test` job to test.yml workflow for package build and installation validation
  - Added pre-release-check.yml workflow that mimics manual test_release.sh script
  - Enhanced publish.yml with full test suite, code quality checks, and installation testing before publishing
  - Added workflow documentation in .github/workflows/README.md
  - CI now tests package installation and CLI functionality in isolated environments
  - Pre-release validation runs across Ubuntu, Windows, and macOS with Python 3.9 and 3.12
- **Fine-grained HTTP instrumentation control**
  - Added `enable_http_instrumentation` configuration option (default: `false`)
  - Environment variable: `GENAI_ENABLE_HTTP_INSTRUMENTATION`
  - Allows enabling HTTP/httpx instrumentation without disabling all MCP instrumentation (databases, vector DBs, Redis, Kafka)
- Support for `SERVICE_INSTANCE_ID` and environment attributes in resource creation (Issue #XXX)
- Configurable timeout for OTLP exporters via `OTEL_EXPORTER_OTLP_TIMEOUT` environment variable (Issue #XXX)
- Added openinference instrumentation dependencies: `openinference-instrumentation==0.1.31`, `openinference-instrumentation-litellm==0.1.19`, `openinference-instrumentation-mcp==1.3.0`, `openinference-instrumentation-smolagents==0.1.11`, and `openinference-semantic-conventions==0.1.17` (Issue #XXX)
- Explicit configuration of `TraceContextTextMapPropagator` for W3C trace context propagation (Issue #XXX)
- Created examples for LiteLLM and Smolagents instrumentors

### Changed

- **HTTP instrumentation now opt-in instead of opt-out**
  - HTTP/httpx instrumentation is now disabled by default (`enable_http_instrumentation=false`)
  - MCP instrumentation remains enabled by default (databases, vector DBs, Redis, Kafka all work out of the box)
  - Set `GENAI_ENABLE_HTTP_INSTRUMENTATION=true` or `enable_http_instrumentation=True` to enable HTTP tracing
- **Updated Mistral AI example for new SDK (v1.0+)**
  - Migrated from deprecated `mistralai.client.MistralClient` to new `mistralai.Mistral` API
- Updated logging configuration to allow log level via environment variable and implement log rotation (Issue #XXX)

### Tests

- Fixed tests for base/redis and auto instrument (a701603)
- Updated `test_auto_instrument.py` assertions to match new OTLP exporter configuration (exporters now read endpoint from environment variables instead of direct parameters)

[Unreleased]: https://github.com/Mandark-droid/genai_otel_instrument/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Mandark-droid/genai_otel_instrument/releases/tag/v0.1.0