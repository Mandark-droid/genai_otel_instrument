# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

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