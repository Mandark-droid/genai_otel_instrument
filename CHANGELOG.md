# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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