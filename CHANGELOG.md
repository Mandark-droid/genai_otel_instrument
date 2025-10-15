# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Fix HuggingFace instrumentation to correctly set span attributes and pass tests.
- Resolve `AttributeError` related to `TraceContextTextMapPropagator` in test files by correcting import paths.
- Fixed `setup_meter` function in `genai_otel/metrics.py` to correctly configure OpenTelemetry MeterProvider with metric readers and handle invalid OTLP endpoint/headers gracefully.
- Corrected `tests/test_metrics.py` to properly reset MeterProvider state between tests and accurately access metric exporter attributes, resolving `TypeError` and `AssertionError`s.
- Fixed `cost_counter` not being called in `tests/instrumentors/test_base.py` by ensuring `BaseInstrumentor._shared_cost_counter` is patched with a distinct mock before `ConcreteInstrumentor` instantiation.
- Resolved `setup_tracing` failures in `tests/test_config.py` by correcting `genai_otel/config.py`'s `setup_tracing` function and adjusting the `reset_tracer` fixture to mock `TracerProvider` correctly.

### Added

- Support for `SERVICE_INSTANCE_ID` and environment attributes in resource creation (Issue #XXX)
- Configurable timeout for OTLP exporters via `OTEL_EXPORTER_OTLP_TIMEOUT` environment variable (Issue #XXX)
- Added openinference instrumentation dependencies: `openinference-instrumentation==0.1.31`, `openinference-instrumentation-litellm==0.1.19`, `openinference-instrumentation-mcp==1.3.0`, `openinference-instrumentation-smolagents==0.1.11`, and `openinference-semantic-conventions==0.1.17` (Issue #XXX)
- Explicit configuration of `TraceContextTextMapPropagator` for W3C trace context propagation (Issue #XXX)

### Changed

- Updated logging configuration to allow log level via environment variable and implement log rotation (Issue #XXX)

### Tests

- Fixed tests for base/redis and auto instrument (a701603)

## [0.1.0] - 2025-10-11

### Added

- Initial release of genai-otel-instrument
- Automatic instrumentation for LLM providers:
  - OpenAI
  - Anthropic (Claude)
  - Google AI (Gemini)
  - AWS Bedrock
  - Azure OpenAI
  - Cohere
  - Mistral AI
  - Together AI
  - Groq
  - Ollama
  - Vertex AI
  - Replicate
  - Anyscale
- Framework support:
  - LangChain
  - LlamaIndex
  - HuggingFace Transformers
- MCP (Model Context Protocol) tool instrumentation:
  - Database clients (PostgreSQL, MySQL, MongoDB, SQLAlchemy)
  - Redis caching
  - Kafka message queues
  - Vector databases (Pinecone, Weaviate, Qdrant, ChromaDB, Milvus, FAISS)
  - Generic API calls
- GPU metrics collection support (NVIDIA GPUs via pynvml)
- Cost tracking and estimation for LLM API calls
- OpenTelemetry trace and metrics export
- CLI tool for running instrumented applications
- Comprehensive configuration via environment variables
- Optional dependencies for lean installations

### Features

- Automatic span creation for LLM operations
- Token usage tracking
- Request latency measurement
- Error tracking and reporting
- Configurable fail-on-error behavior
- Extensive logging configuration

[Unreleased]: https://github.com/Mandark-droid/genai_otel_instrument/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Mandark-droid/genai_otel_instrument/releases/tag/v0.1.0