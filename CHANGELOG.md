# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.27] - 2025-12-30

### Fixed

- **PII Evaluation Attributes Export to Jaeger**
  - Fixed critical issue where PII evaluation attributes were not appearing in Jaeger traces
  - Root cause: Attributes were being set AFTER `span.end()` when span becomes immutable (ReadableSpan)
  - Solution: Added `_run_evaluation_checks()` method in `BaseInstrumentor` that runs BEFORE `span.end()`
  - PII attributes now successfully exported: `evaluation.pii.prompt.detected`, `evaluation.pii.prompt.entity_count`, `evaluation.pii.prompt.entity_types`, etc.
  - Applies to both PII and Toxicity detection attributes

- **Editable Installation Issue**
  - Fixed issue where examples were using old code due to non-editable pip install
  - Package must be installed with `pip install -e .` for development to reflect local code changes
  - Added clear documentation in scripts/README.md

### Added

- **Comprehensive Example Organization**
  - Reorganized examples into dedicated folders:
    - `examples/pii_detection/` - 10 PII detection examples (detect, redact, block modes + compliance)
    - `examples/toxicity_detection/` - 8 toxicity detection examples (Detoxify, Perspective API, categories)
    - `examples/bias_detection/` - Placeholder for future bias detection
  - All examples updated to use `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable

- **Examples Validation Script**
  - New `scripts/validate_examples.sh` for Linux/Mac
  - New `scripts/validate_examples.bat` for Windows
  - Features:
    - `--dry-run` - List all examples without running
    - `--verbose` - Show detailed output from examples
    - `--timeout N` - Configurable timeout (default: 90s)
    - `--help` - Show usage information
  - Validates all PII, Toxicity, and Bias detection examples
  - Color-coded output (PASSED/FAILED/SKIPPED)
  - Comprehensive summary with failed/skipped example lists

- **Scripts Documentation**
  - New `scripts/README.md` with comprehensive usage guide
  - Moved temporary test files to `scripts/` folder for organization
  - Moved `SOLUTION_SUMMARY.md` and `VALIDATION_REPORT.md` to `scripts/` folder

### Changed

- **Example Files Updated**
  - All PII examples now use `os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")`
  - All Toxicity examples now use `os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")`
  - Bias detection placeholder updated with env var pattern
  - Total: 19 example files updated for flexible endpoint configuration

- **File Organization**
  - Moved debug/test scripts to `scripts/` folder:
    - test_*.py files (9 temporary test scripts)
    - SOLUTION_SUMMARY.md (PII/Toxicity solution documentation)
    - VALIDATION_REPORT.md (Comprehensive validation report)

## [0.1.26] - 2025-12-09

### Added

- **Comprehensive Codecarbon Metrics Exposure**
  - Exposes ALL codecarbon EmissionsData fields as OpenTelemetry metrics
  - New metric: `gen_ai.power.consumption` - Power consumption by component (CPU/GPU/RAM) in Watts
  - New metric: `gen_ai.energy.total` - Total energy consumed (sum of CPU+GPU+RAM) in kWh
  - New metric: `gen_ai.codecarbon.task.duration` - Duration of monitoring tasks in seconds
  - Enhanced all codecarbon metrics with complete hardware and system metadata attributes:
    - Hardware: os, python_version, cpu_count, cpu_model, gpu_count, gpu_model
    - Cloud infrastructure: on_cloud, cloud_provider, cloud_region
    - Location: country, region
  - Implements full codecarbon output specification from https://mlco2.github.io/codecarbon/output.html

### Fixed

- **Codecarbon Verbose Logging Suppression**
  - Suppressed codecarbon's informational warnings by default (CPU tracking mode, multiple instances, etc.)
  - Added `GENAI_CODECARBON_LOG_LEVEL` environment variable (default: "error")
  - Users can re-enable warnings by setting `GENAI_CODECARBON_LOG_LEVEL=warning`
  - Eliminates console noise while preserving ability to enable diagnostics when needed

## [0.1.25] - 2025-12-09

### Fixed

- **Codecarbon CO2 Tracking: Task API Integration**
  - Fixed codecarbon emissions tracking by migrating from private API (_total_emissions.total) to public Task API
  - Uses stop_task/start_task cycle for continuous monitoring without accessing internals
  - Resolves compatibility issues with codecarbon 3.0.7+ where internal APIs changed
  - Properly handles task lifecycle in start() and stop() methods
  - Added enhanced logging with detailed energy breakdown

- **Codecarbon Additional Metrics**
  - New metric: `gen_ai.energy.consumed` - Energy consumption by component (CPU/GPU/RAM) in kWh
  - New metric: `gen_ai.co2.emissions_rate` - CO2 emissions rate in gCO2e/s
  - Records energy breakdown from codecarbon's EmissionsData object
  - Tracks emissions by country and region from codecarbon's location detection

### Added

- **LLM Pricing Database: Comprehensive Update - 293 New Models**
  - **DeepSeek Models** (10 new models):
    - DeepSeek V3.2 - official release with 50% cost reduction ($0.00028/$0.00042 cache hit)
    - DeepSeek V3.2-Speciale - optimized for complex reasoning, AIME 2025 96.0% accuracy
    - DeepSeek-Prover-V2-7B - formal theorem proving in Lean 4 ($0.0002/$0.0003)
    - DeepSeek-Prover-V2-671B - large-scale theorem proving ($0.0007/$0.0025)
    - DeepSeekMath V2 series - mathematical reasoning specialists
    - deepseek/ prefix variants for API compatibility
  - **xAI Grok Models** (29 new models):
    - Grok 4 - latest flagship model ($0.003/$0.015)
    - Grok 4.1 Fast - near-frontier capability ($0.0002/$0.0005)
    - Grok 3 and Grok 3 Mini - various performance tiers
    - Complete xai/ prefix variants for all Grok 2, 3, 4 models
  - **Zhipu AI GLM Models** (7 new models):
    - GLM-4.5 - 355B params, 32B active ($0.0006/$0.0022)
    - GLM-4.5-Air - 106B params, 12B active ($0.0002/$0.0011)
    - GLM-4.6 - latest version with improved performance
    - zai/ prefix variants for all GLM-4.5/4.6 models
  - **OpenAI Models** (42 new models):
    - o1-2024-12-17 - latest reasoning model ($0.015/$0.06)
    - o3-mini, o3-pro, o4-mini - next-gen reasoning models
    - GPT-5 series - gpt-5-chat, gpt-5-codex, gpt-5-mini, gpt-5-nano
    - GPT-5.1 series - latest versions with improved capabilities
    - Historical versions: gpt-3.5-turbo variants, gpt-4 32k models
  - **Anthropic Claude Models** (13 new models):
    - Claude 4 Sonnet 20250514 - latest Claude 4 Sonnet ($0.003/$0.015)
    - Claude Opus 4.5 and 4.5-20251101 - highest capability tier ($0.005/$0.025)
    - Claude 3.7 Sonnet variants - extended context versions
    - Latest aliases: claude-3-5-sonnet-latest, claude-3-opus-latest
  - **Google Gemini Models** (58 new models):
    - Gemini 2.5 Flash and Pro - latest generation models
    - Gemini 2.0 Flash variants - optimized for speed
    - Gemini 3 Pro Preview - upcoming flagship model ($0.002/$0.012)
    - Complete gemini/ prefix variants for all models
    - Experimental and preview versions with free access
  - **Mistral Models** (66 new models):
    - Mistral Large 3 - 675B total, 41B active ($0.0005/$0.0015)
    - Ministral 3 series (9 models) - compact high-performance variants
    - Codestral, Devstral, Magistral series - specialized models
    - Complete mistral/ prefix variants for all models
  - **Qwen Models** (10 new models):
    - Qwen3 Max, Qwen Plus variants via dashscope/ prefix
    - QwQ-Plus - thinking mode support
    - Qwen Turbo - cost-effective fast inference
  - **Cohere Command Models** (4 new models):
    - Command-A-03-2025 - latest flagship model
    - Command-R and Command-R-Plus 08-2024 variants
    - Command-R7B-12-2024 - compact efficient model
  - **AI21 Jamba Models** (7 new models):
    - Jamba 1.5, 1.6, 1.7 series - hybrid SSM-Transformer architecture
    - Large and Mini variants for different scale needs
  - **Additional Providers** (62 new models):
    - Moonshot Kimi models - Chinese market leaders with thinking mode
    - Together.AI and Fireworks.AI pricing tiers
    - Luminous models from Aleph Alpha
    - Morph and v0 specialized models
  - **Implementation Summary**:
    - Total additions: 293 new models (24 DeepSeek additions + 269 from comprehensive LiteLLM comparison)
    - Total model coverage: **649 chat completion models** across 25+ providers
    - Database: `genai_otel/llm_pricing.json`
    - All prices in USD per 1K tokens (prompt/completion)
    - Includes latest 2025 model releases through December

- **Embedding, Image, and Audio Models: Comprehensive Expansion - 114 New Models**
  - **Embedding Models** (47 new models):
    - Google embeddings: gemini-embedding-001, text-embedding-004/005
    - Cohere embeddings: embed-v4.0, embed-english-v3, embed-multilingual-v3
    - Cohere rerankers: rerank-v3.5, rerank-english-v3.0, rerank-multilingual-v3.0
    - Mistral embeddings: codestral-embed, codestral-embed-2505, mistral-embed
    - Amazon Titan: titan-embed-image-v1 (multimodal embedding)
    - Together.AI and Fireworks.AI embedding pricing tiers
    - Voyage rerankers: rerank-2, rerank-2-lite
    - Jina reranker: jina-reranker-v2-base-multilingual
    - Doubao (ByteDance) embedding models
    - NVIDIA reranker models
    - Total embeddings: **81 models** (was 34, +138% increase)
  - **Image Generation Models** (14 new models):
    - Google Imagen: imagen-3.0, imagen-4.0 variants (fast, standard, ultra)
    - Amazon Titan: titan-image-generator-v2
    - Recraft: recraftv2, recraftv3
    - FLUX models: FLUX-1.1-pro, FLUX.1-Kontext-pro
    - DALL-E 3 quality variants: standard and HD for various sizes
    - Total image models: **28 models** (was 14, +100% increase)
  - **Audio/Speech Models** (53 new models):
    - Deepgram models (33 models):
      * Nova 2 and 3 series - specialized for different use cases
      * Base and Enhanced tiers for various domains (finance, medical, meeting, phonecall, etc.)
      * Whisper variants (tiny, small, medium, large, base)
    - OpenAI audio models:
      * gpt-4o-audio-preview variants (2024-10-01, 2024-12-17, 2025-06-03)
      * gpt-4o-mini-audio-preview models
      * gpt-4o-transcribe and gpt-4o-mini-transcribe
      * gpt-4o-mini-tts (text-to-speech)
    - ElevenLabs: scribe_v1, scribe_v1_experimental
    - AssemblyAI: best, nano
    - Gemini TTS: gemini-2.5-flash-preview-tts, gemini-2.5-pro-preview-tts
    - Whisper-1: OpenAI's original transcription model
    - Total audio models: **64 models** (was 11, +482% increase)
  - **Pricing Structure**:
    - Embeddings: Per 1K tokens (input cost)
    - Images: Per image generation (varies by quality/size)
    - Audio: Per minute for STT, per 1K characters for TTS
  - **Implementation**:
    - Database: `genai_otel/llm_pricing.json`
    - Total non-chat additions: 114 models
    - All pricing sourced from official provider documentation and LiteLLM database

- **Ollama Instrumentor: Missing Response Attributes**
  - Added `_extract_response_attributes()` method to extract response model, finish reason, and content length
  - Added `_extract_finish_reason()` method to extract completion status from `done_reason` field
  - Fixes missing `gen_ai.response.model`, `gen_ai.response.finish_reason`, and cost tracking fields in Ollama traces
  - Handles both dict and object response formats for compatibility
  - Supports both `generate()` and `chat()` response structures
  - Enables proper cost calculation for Ollama models (previously failed due to missing response model)
  - Implementation: `genai_otel/instrumentors/ollama_instrumentor.py` (lines 199-276)
  - Tests: `tests/instrumentors/test_ollama_instrumentor.py` (6 new test functions)

- **CrewAI Instrumentor: Automatic Context Propagation**
  - **Zero-code context propagation** for complete trace continuity across threads and async execution
  - Automatic ThreadPoolExecutor patching for context propagation to worker threads
  - Enhanced instrumentation with three span types:
    - `crewai.crew.execution` - Top-level crew execution
    - `crewai.task.execution` - Individual task execution
    - `crewai.agent.execution` - Agent task execution
  - Automatic instrumentation of Task and Agent methods:
    - `Task.execute_sync()` - Synchronous task execution
    - `Task.execute_async()` - Asynchronous task execution
    - `Agent.execute_task()` - Agent task execution
  - Rich attribute extraction for better observability:
    - **Task attributes**: description, expected_output, assigned agent role, task ID
    - **Agent attributes**: role, goal, backstory, LLM model
    - **Crew attributes**: process type, agent count, task count, tools, inputs
  - Static `_propagate_context()` decorator for function-level context wrapping
  - Thread-safe context attachment/detachment using OpenTelemetry context API
  - Graceful degradation if methods don't exist (future-proof for CrewAI updates)
  - **Benefits for users**:
    - ✅ No manual context management code required
    - ✅ Complete parent-child span relationships across all execution
    - ✅ Works with FastAPI, Flask, and other async frameworks
    - ✅ Compatible with CrewAI's internal threading model
  - Implementation: `genai_otel/instrumentors/crewai_instrumentor.py` (+216 lines)
  - Example usage - before: manual `run_in_thread_with_context()` wrapper needed
  - Example usage - after: just call `crew.kickoff()` normally, context propagates automatically!

- **Codecarbon Integration for CO2 Emissions Tracking**
  - Integrated codecarbon library for accurate region-based carbon intensity calculations
  - Uses `OfflineEmissionsTracker` for offline mode (no API calls) or `EmissionsTracker` for online mode
  - Automatic region detection using country ISO codes, cloud providers, and regions
  - Fallback to manual calculation when codecarbon is not installed
  - New environment variables for codecarbon configuration:
    - `GENAI_CO2_COUNTRY_ISO_CODE` - 3-letter ISO country code (e.g., "USA", "GBR", "DEU")
    - `GENAI_CO2_REGION` - Region/state within country (e.g., "california", "texas")
    - `GENAI_CO2_CLOUD_PROVIDER` - Cloud provider name (e.g., "aws", "gcp", "azure")
    - `GENAI_CO2_CLOUD_REGION` - Cloud region (e.g., "us-east-1", "europe-west1")
    - `GENAI_CO2_OFFLINE_MODE` - Run codecarbon in offline mode (default: true)
    - `GENAI_CO2_TRACKING_MODE` - "machine" (all processes) or "process" (current only)
    - `GENAI_CO2_USE_MANUAL` - Force manual CO2 calculation using `GENAI_CARBON_INTENSITY`
  - CO2 tracking options:
    - **Automatic (codecarbon)**: Uses region-based carbon intensity data for accurate emissions
    - **Manual**: Uses `GENAI_CARBON_INTENSITY` value (gCO2e/kWh) for calculation
    - Set `GENAI_CO2_USE_MANUAL=true` to force manual calculation even when codecarbon is installed
  - Implementation in `genai_otel/gpu_metrics.py` and `genai_otel/config.py`
  - Added comprehensive tests for codecarbon integration (13 new test cases)
  - Install codecarbon with: `pip install genai-otel-instrument[co2]`

- **PII Detection and Safety Features (v0.2.0 Phase 1)**
  - Automatic PII detection with Microsoft Presidio integration
  - Three operation modes: detect (monitor only), redact (mask PII), block (prevent requests/responses)
  - GDPR compliance mode with EU-specific entity types (IBAN, UK NHS, NRP)
  - HIPAA compliance mode for healthcare data (medical licenses, PHI, dates)
  - PCI-DSS compliance mode for payment card data (credit cards, bank accounts)
  - 15+ PII entity types detected: email, phone, SSN, credit card, IP address, passport, etc.
  - Configurable confidence threshold (0.0-1.0) for detection sensitivity
  - Regex fallback patterns when Presidio library not available
  - OpenTelemetry span attributes for PII detection events:
    - `evaluation.pii.prompt.detected` - PII found in prompts
    - `evaluation.pii.response.detected` - PII found in responses
    - `evaluation.pii.*.entity_count` - Number of entities detected
    - `evaluation.pii.*.entity_types` - Array of detected entity types
    - `evaluation.pii.*.score` - Detection confidence score
    - `evaluation.pii.*.redacted` - Redacted text in redact mode
    - `evaluation.pii.*.blocked` - Whether request was blocked
  - OpenTelemetry metrics for monitoring:
    - `genai.evaluation.pii.detections` - Counter by location and mode
    - `genai.evaluation.pii.entities` - Counter by entity type
    - `genai.evaluation.pii.blocked` - Counter for blocked requests
  - Environment variable configuration:
    - `GENAI_ENABLE_PII_DETECTION` - Enable/disable PII detection
    - `GENAI_PII_MODE` - Set mode (detect/redact/block)
    - `GENAI_PII_THRESHOLD` - Confidence threshold
    - `GENAI_PII_GDPR_MODE` - Enable GDPR compliance
    - `GENAI_PII_HIPAA_MODE` - Enable HIPAA compliance
    - `GENAI_PII_PCI_DSS_MODE` - Enable PCI-DSS compliance
  - Implementation: `genai_otel/evaluation/` module
    - `config.py` - Configuration dataclasses for all evaluation features
    - `pii_detector.py` - PIIDetector with Presidio integration
    - `span_processor.py` - EvaluationSpanProcessor for span enrichment
  - Tests: `tests/evaluation/` (40+ test cases)
    - `test_pii_detector.py` - Unit tests for PII detection
    - `test_integration.py` - Integration tests with span processor
  - Example: `examples/pii_detection_example.py` (9 comprehensive scenarios)
  - Dependencies (optional): `pip install presidio-analyzer presidio-anonymizer spacy`

- **Toxicity Detection (v0.2.0 Phase 1)**
  - Automatic toxicity detection for harmful content in prompts and responses
  - Dual detection methods:
    - Google Perspective API integration (cloud-based, production-grade)
    - Detoxify local ML model (offline, privacy-friendly)
    - Automatic fallback from Perspective API to Detoxify on errors
  - Six toxicity categories detected:
    - `toxicity`: General toxic language
    - `severe_toxicity`: Extremely harmful content
    - `identity_attack`: Discrimination and hate speech
    - `insult`: Insulting or demeaning language
    - `profanity`: Swearing and obscene content
    - `threat`: Threatening or violent language
  - Configurable threshold (0.0-1.0) for detection sensitivity
  - Blocking mode to prevent toxic content processing
  - Batch processing support for efficient analysis
  - OpenTelemetry span attributes for toxicity detection:
    - `evaluation.toxicity.prompt.detected` - Toxicity in prompts
    - `evaluation.toxicity.response.detected` - Toxicity in responses
    - `evaluation.toxicity.*.max_score` - Maximum toxicity score
    - `evaluation.toxicity.*.categories` - List of toxic categories detected
    - `evaluation.toxicity.*.<category>_score` - Individual category scores
    - `evaluation.toxicity.*.blocked` - Whether content was blocked
  - OpenTelemetry metrics for monitoring:
    - `genai.evaluation.toxicity.detections` - Detection events counter
    - `genai.evaluation.toxicity.categories` - Category-specific counter
    - `genai.evaluation.toxicity.blocked` - Blocked requests counter
    - `genai.evaluation.toxicity.score` - Score distribution histogram
  - Environment variable configuration:
    - `GENAI_ENABLE_TOXICITY_DETECTION` - Enable/disable toxicity detection
    - `GENAI_TOXICITY_THRESHOLD` - Detection threshold (0.0-1.0)
    - `GENAI_TOXICITY_USE_PERSPECTIVE_API` - Use Perspective API
    - `GENAI_TOXICITY_PERSPECTIVE_API_KEY` - API key for Perspective
    - `GENAI_TOXICITY_BLOCK_ON_DETECTION` - Block toxic content
  - Implementation: `genai_otel/evaluation/` module
    - `toxicity_detector.py` - ToxicityDetector with dual detection methods
    - `span_processor.py` - Extended with toxicity detection
    - `config.py` - ToxicityConfig dataclass
  - Tests: `tests/evaluation/` (35+ test cases)
    - `test_toxicity_detector.py` - Unit tests for ToxicityDetector
    - `test_integration.py` - Integration tests with span processor
  - Example: `examples/toxicity_detection_example.py` (8 comprehensive scenarios)
  - Dependencies (optional):
    - Detoxify: `pip install detoxify`
    - Perspective API: `pip install google-api-python-client`

- **Bias Detection (v0.2.0 Phase 2)**
  - Automatic bias detection for demographic and other biases in prompts and responses
  - Pattern-based detection (always available, no external dependencies)
  - Eight bias types monitored:
    - `gender`: Gender stereotypes and discrimination
    - `race`: Racial bias and discrimination
    - `ethnicity`: Ethnic stereotypes and xenophobia
    - `religion`: Religious bias and discrimination
    - `age`: Age-based stereotypes (ageism)
    - `disability`: Disability bias and ableism
    - `sexual_orientation`: LGBTQ+ discrimination and bias
    - `political`: Political bias and partisan stereotyping
  - Comprehensive pattern matching with 50+ regex patterns and keywords
  - Score calculation based on pattern matches (0.0-1.0)
  - Configurable threshold for detection sensitivity
  - Blocking mode to prevent biased content processing
  - Batch processing support for analyzing multiple texts
  - Statistics generation for bias analysis and reporting
  - Optional ML-based detection with Fairlearn integration
  - Sensitive attributes configuration for custom monitoring
  - OpenTelemetry span attributes for bias detection:
    - `evaluation.bias.prompt.detected` - Bias in prompts
    - `evaluation.bias.response.detected` - Bias in responses
    - `evaluation.bias.*.max_score` - Maximum bias score
    - `evaluation.bias.*.detected_biases` - Array of detected bias types
    - `evaluation.bias.*.<bias_type>_score` - Individual bias type scores
    - `evaluation.bias.*.<bias_type>_patterns` - Matched patterns per type
    - `evaluation.bias.*.blocked` - Whether content was blocked
  - OpenTelemetry metrics for monitoring:
    - `genai.evaluation.bias.detections` - Detection events counter by location
    - `genai.evaluation.bias.types` - Detections by bias type
    - `genai.evaluation.bias.blocked` - Blocked requests counter
    - `genai.evaluation.bias.score` - Score distribution histogram
  - Environment variable configuration:
    - `GENAI_ENABLE_BIAS_DETECTION` - Enable/disable bias detection
    - `GENAI_BIAS_THRESHOLD` - Detection threshold (0.0-1.0, default 0.5)
    - `GENAI_BIAS_BLOCK_ON_DETECTION` - Block biased content
    - `GENAI_BIAS_TYPES` - Comma-separated list of bias types to monitor
    - `GENAI_BIAS_USE_FAIRLEARN` - Enable ML-based detection with Fairlearn
  - Implementation: `genai_otel/evaluation/` module
    - `bias_detector.py` - BiasDetector with pattern and ML-based detection
    - `span_processor.py` - Extended with bias detection support
    - `config.py` - BiasConfig dataclass
  - Tests: `tests/evaluation/` (56+ test cases)
    - `test_bias_detector.py` - Unit tests for BiasDetector (40+ test cases)
    - `test_integration.py` - Integration tests with span processor (16 test cases)
  - Example: `examples/bias_detection_example.py` (12 comprehensive scenarios)
  - Dependencies (optional):
    - Fairlearn: `pip install fairlearn scikit-learn` (for ML-based detection)

- **Prompt Injection Detection (v0.2.0 Phase 3)**
  - Automatic prompt injection detection protecting against manipulation attacks
  - 6 injection types: instruction_override, role_playing, jailbreak, context_switching, system_extraction, encoding_obfuscation
  - Pattern-based detection (always available, no dependencies)
  - Configurable threshold and blocking mode
  - Span attributes: `evaluation.prompt_injection.*` for all detection results
  - Metrics: 4 metrics (detections, types, blocked, score histogram)
  - Implementation: `prompt_injection_detector.py` (250+ lines)
  - Example: `examples/comprehensive_evaluation_example.py`

- **Restricted Topics Detection (v0.2.0 Phase 3)**
  - Topic classification for 9 sensitive categories (medical/legal/financial advice, violence, self-harm, etc.)
  - Configurable topic blacklists
  - Pattern and keyword-based detection
  - Span attributes: `evaluation.restricted_topics.*` for topic detection
  - Metrics: 4 metrics (detections, types, blocked, score histogram)
  - Implementation: `restricted_topics_detector.py` (300+ lines)
  - Example: `examples/comprehensive_evaluation_example.py`

- **Hallucination Detection (v0.2.0 Phase 3)**
  - Heuristic-based factual accuracy validation
  - Factual claim extraction, hedge word detection, citation tracking
  - Context contradiction detection
  - Span attributes: `evaluation.hallucination.*` for risk indicators
  - Metrics: 3 metrics (detections, indicators, score histogram)
  - Implementation: `hallucination_detector.py` (380+ lines)
  - Example: `examples/comprehensive_evaluation_example.py`

- **Multi-Agent & AI Framework Instrumentation (Phase 1-4)**
  - Comprehensive instrumentation for 11 AI frameworks with 13 implementations total
  - Zero-code setup with automatic tracing and cost tracking
  - Production-ready with 185+ test cases and 47 example scenarios
  - New frameworks: DSPy, Instructor, Guardrails AI

- **OpenAI Agents SDK Instrumentation**
  - Full OpenTelemetry instrumentation for OpenAI's production Agents SDK
  - Automatic tracing with `gen_ai.system="agents"` attribute
  - Agent orchestration with handoffs, sessions, and guardrails tracking
  - Implementation: `genai_otel/instrumentors/openai_agents_instrumentor.py`
  - Tests: `tests/instrumentors/test_openai_agents_instrumentor.py` (11 test cases)
  - Example: `examples/openai_agents_example.py` (4 scenarios)

- **CrewAI Multi-Agent Framework Instrumentation**
  - Full OpenTelemetry instrumentation for CrewAI framework
  - Automatic tracing with `gen_ai.system="crewai"` attribute
  - Role-based agent collaboration with crews and tasks tracking
  - Sequential and hierarchical process types supported
  - Implementation: `genai_otel/instrumentors/crewai_instrumentor.py`
  - Tests: `tests/instrumentors/test_crewai_instrumentor.py` (13 test cases)
  - Example: `examples/crewai_example.py` (3 scenarios)

- **LangGraph Stateful Workflow Instrumentation**
  - Full OpenTelemetry instrumentation for LangGraph framework
  - Automatic tracing with `gen_ai.system="langgraph"` attribute
  - Graph-based orchestration with nodes, edges, and state tracking
  - Support for sync/async execution and streaming
  - Checkpoint and state management tracking
  - Implementation: `genai_otel/instrumentors/langgraph_instrumentor.py`
  - Tests: `tests/instrumentors/test_langgraph_instrumentor.py` (12 test cases)
  - Example: `examples/langgraph_example.py` (3 scenarios)

- **AutoGen Multi-Agent Conversation Instrumentation**
  - Full OpenTelemetry instrumentation for Microsoft AutoGen framework
  - Automatic tracing with `gen_ai.system="autogen"` attribute
  - Multi-agent conversations with group chat orchestration
  - Speaker selection and manager coordination tracking
  - Support for both `autogen` and `pyautogen` package names
  - Implementation: `genai_otel/instrumentors/autogen_instrumentor.py`
  - Tests: `tests/instrumentors/test_autogen_instrumentor.py` (20 test cases)
  - Example: `examples/autogen_example.py` (4 scenarios)

- **Pydantic AI Type-Safe Agent Instrumentation**
  - Full OpenTelemetry instrumentation for Pydantic AI framework
  - Automatic tracing with `gen_ai.system="pydantic_ai"` attribute
  - Type-safe agents with Pydantic validation tracking
  - Multi-provider support (OpenAI, Anthropic, Gemini, etc.)
  - Structured outputs with Pydantic models
  - Tools/functions tracking with count and names
  - Support for sync, async, and streaming execution
  - Implementation: `genai_otel/instrumentors/pydantic_ai_instrumentor.py`
  - Tests: `tests/instrumentors/test_pydantic_ai_instrumentor.py` (24 test cases)
  - Example: `examples/pydantic_ai_example.py` (7 scenarios)

- **Haystack NLP Pipeline Instrumentation**
  - Full OpenTelemetry instrumentation for Haystack framework
  - Automatic tracing with `gen_ai.system="haystack"` attribute
  - Modular pipeline architecture with component tracking
  - RAG (Retrieval-Augmented Generation) workflow support
  - Generator, ChatGenerator, and Retriever component instrumentation
  - Pipeline graph structure tracking (nodes, edges, connections)
  - Custom metadata and configuration capture
  - Implementation: `genai_otel/instrumentors/haystack_instrumentor.py`
  - Tests: `tests/instrumentors/test_haystack_instrumentor.py` (23 test cases)
  - Example: `examples/haystack_example.py` (5 scenarios)

- **AWS Bedrock Agents Instrumentation**
  - Full OpenTelemetry instrumentation for AWS Bedrock Agents
  - Automatic tracing with `gen_ai.system="bedrock_agents"` attribute
  - Managed agent runtime with session tracking
  - Knowledge base retrieval and RAG operations
  - InvokeAgent, Retrieve, and RetrieveAndGenerate operation support
  - Session state and conversation tracking
  - Integration via boto3 BaseClient instrumentation
  - Implementation: `genai_otel/instrumentors/bedrock_agents_instrumentor.py`
  - Tests: `tests/instrumentors/test_bedrock_agents_instrumentor.py` (20 test cases)
  - Example: `examples/bedrock_agents_example.py` (4 scenarios)

- **DSPy Framework Instrumentation**
  - Full OpenTelemetry instrumentation for Stanford NLP's DSPy framework
  - Automatic tracing with `gen_ai.system="dspy"` attribute
  - Declarative language model programming with automatic optimization
  - Module execution tracking (Module.__call__, Predict, ChainOfThought, ReAct)
  - Optimizer/Teleprompter operations (COPRO, MIPROv2, BootstrapFewShot)
  - Signature and field tracking (input/output fields, rationales)
  - Tool usage and trajectory tracking for ReAct
  - Implementation: `genai_otel/instrumentors/dspy_instrumentor.py`
  - Tests: `tests/instrumentors/test_dspy_instrumentor.py` (25 test cases)
  - Example: `examples/dspy_example.py` (6 scenarios)

- **Instructor Framework Instrumentation**
  - Full OpenTelemetry instrumentation for Instructor (8K+ GitHub stars)
  - Automatic tracing with `gen_ai.system="instructor"` attribute
  - Pydantic-based structured output extraction with validation
  - Multi-provider support (OpenAI, Anthropic, Google, Ollama, etc.)
  - Automatic retry on validation failure tracking
  - Streaming partial results (Partial models)
  - Response model schema capture (fields, field count, types)
  - Implementation: `genai_otel/instrumentors/instructor_instrumentor.py`
  - Tests: `tests/instrumentors/test_instructor_instrumentor.py` (22 test cases)
  - Example: `examples/instructor_example.py` (6 scenarios)

- **Guardrails AI Framework Instrumentation**
  - Full OpenTelemetry instrumentation for Guardrails AI validation framework
  - Automatic tracing with `gen_ai.system="guardrails"` attribute
  - Input/output validation guards with risk detection
  - Validator tracking (names, on-fail actions, pass/fail status)
  - On-fail policies: reask, fix, filter, refrain, noop, exception, fix_reask
  - ValidationOutcome tracking (validation_passed, reasks count, errors)
  - Guard operations: __call__, validate, parse, use
  - Implementation: `genai_otel/instrumentors/guardrails_ai_instrumentor.py`
  - Tests: `tests/instrumentors/test_guardrails_ai_instrumentor.py` (8 test cases)

### Improved

- **Google GenAI SDK - Dual SDK Support**
  - Enhanced existing instrumentor to support BOTH legacy and new SDKs
  - Automatic SDK detection: tries new `google-genai` first, falls back to legacy `google-generativeai`
  - Deprecation warnings for legacy SDK users (support ends Nov 30, 2025)
  - Migration guidance in examples
  - Updated tests with dual SDK coverage (24 test cases)
  - Example: `examples/google_genai_example.py` with both SDK demonstrations

### Documentation

- **Framework Research Documentation**
  - Created `FRAMEWORK_RESEARCH.md` with comprehensive analysis of 9 AI frameworks
  - Tiered prioritization (Tier 1-3) based on popularity and complexity
  - Implementation estimates and recommended attributes
  - Full research report with API analysis and instrumentation strategies

- **README Updates**
  - Added "Multi-Agent Frameworks" section highlighting 6 new frameworks
  - Updated feature list with framework count
  - Comprehensive framework descriptions and capabilities

## [0.1.23] - 2025-11-13

### Added

- **SambaNova Instrumentation**
  - Full OpenTelemetry instrumentation for SambaNova AI models
  - Automatic tracing with `gen_ai.system="sambanova"` attribute
  - Token usage tracking and cost calculation
  - Support for Llama 4 Maverick and Llama 3.1 model family
  - Enabled by default in `DEFAULT_INSTRUMENTORS`
  - Example: `examples/sambanova_example.py`
  - Implementation: `genai_otel/instrumentors/sambanova_instrumentor.py`
  - Tests: `tests/instrumentors/test_sambanova_instrumentor.py`

- **Hyperbolic API Instrumentation**
  - Full OpenTelemetry instrumentation for Hyperbolic's cost-effective API
  - HTTP request-level instrumentation for raw API calls
  - Automatic tracing with `gen_ai.system="hyperbolic"` attribute
  - Token usage tracking and cost calculation
  - Support for Qwen3, DeepSeek R1/V3 models
  - **Disabled by default** - requires OTLP gRPC exporters due to requests library conflict
  - Configuration: Set `OTEL_EXPORTER_OTLP_PROTOCOL=grpc` and add "hyperbolic" to `GENAI_ENABLED_INSTRUMENTORS`
  - Example: `examples/hyperbolic_example.py` (complete working configuration)
  - Implementation: `genai_otel/instrumentors/hyperbolic_instrumentor.py`
  - Tests: `tests/instrumentors/test_hyperbolic_instrumentor.py`
  - Documentation: Added limitation section in `CLAUDE.md`

- **Nebius AI Studio Support**
  - Pricing data for Nebius models (uses OpenAI-compatible API, works automatically)
  - Model support: `openai/gpt-oss-120b` and Llama 3.1 family
  - Nebius uses OpenAI SDK with custom `base_url`, so existing OpenAI instrumentor handles it
  - Cost tracking enabled via pricing database entries

### Improved

- **Comprehensive Model Pricing Database Update**
  - Expanded pricing coverage from 240+ to 340+ models across 20+ providers
  - **DeepSeek Models** (25 new models):
    - R1 Distillations: `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` ($0.80/$2.40), `DeepSeek-R1-Distill-Qwen-1.5B` ($0.20/$0.40), `DeepSeek-R1-Distill-Llama-8B` ($0.50/$1.00), `DeepSeek-R1-Distill-Qwen-7B` ($0.40/$0.80)
    - Latest Releases: `DeepSeek-R1-0528` ($1.40/$2.80), `DeepSeek-V3.1` ($0.60/$1.70), `DeepSeek-V3-0324` ($0.60/$1.70), `DeepSeek-R1-0528-Qwen3-8B` ($0.50/$1.00)
    - Experimental: `DeepSeek-V3.2-Exp` ($0.80/$2.00), `DeepSeek-V3.1-Terminus` ($0.60/$1.70)
    - Specialized: `DeepSeek-OCR` ($1.00/$3.00 - 3.6M downloads), `deepseek-vl2` ($0.80/$2.40), `Janus-Pro-7B` ($0.80/$2.40 multimodal)
  - **Liquid AI LFM2 Series** (8 new models):
    - Edge Models: `LFM2-350M` ($0.10/$0.20), `LFM2-700M` ($0.15/$0.30), `LFM2-1.2B` ($0.20/$0.40 - 506K downloads), `LFM2-2.6B` ($0.30/$0.60)
    - MoE: `LFM2-8B-A1B` ($0.30/$0.90)
    - Vision-Language: `LFM2-VL-450M` ($0.20/$0.60), `LFM2-VL-1.6B` ($0.30/$0.90), `LFM2-VL-3B` ($0.40/$1.20)
  - **HuggingFace SmolLM Series** (8 new models):
    - SmolLM2: `SmolLM2-135M` ($0.05/$0.10 - 733K downloads), `SmolLM2-360M` ($0.10/$0.20), `SmolLM2-1.7B` ($0.20/$0.40)
    - SmolLM3: `SmolLM3-3B` ($0.30/$0.60)
    - Instruct variants for all sizes with same pricing
  - **Meta Llama Variants** (13 new models):
    - Llama 3.1/3.2: `Llama-3.1-8B-Instruct` ($0.50/$1.50 - 5M downloads), `Llama-3.1-70B-Instruct` ($2.00/$6.00), `Llama-3.2-1B-Instruct` ($0.10/$0.30), `Llama-3.2-3B-Instruct` ($0.30/$0.60)
    - Llama 3.3: `Llama-3.3-70B-Instruct` ($2.00/$6.00 - 659K downloads)
    - Vision: `Llama-3.2-11B-Vision-Instruct` ($1.00/$3.00 - 257K downloads)
    - Llama 4: `Llama-4-Scout-17B-16E-Instruct` ($1.20/$3.60 - 199K downloads)
    - Guard Models: `Llama-Guard-3-8B` ($0.50/$1.50), `Llama-Guard-3-1B` ($0.10/$0.30)
  - **Google Gemma 3 Series** (6 new models):
    - `gemma-3-1b-it` ($0.10/$0.20 - most popular), `gemma-2-2b-it` ($0.20/$0.40)
    - Vision-capable: `gemma-3-4b-it` ($0.50/$1.50), `gemma-3-12b-it` ($1.00/$3.00 - 1.5M downloads), `gemma-3-27b-it` ($1.50/$4.50)
    - Medical: `medgemma-4b-it` ($0.50/$1.50 - radiology, clinical reasoning, dermatology)
  - **ServiceNow Apriel Models** (3 new models):
    - `Apriel-5B-Instruct` ($0.50/$1.50), `Apriel-Nemotron-15b-Thinker` ($1.00/$3.00), `Apriel-1.5-15b-Thinker` ($1.00/$3.00 - 49K downloads)
  - **NVIDIA Models** (8 new models):
    - Nemotron Nano: `NVIDIA-Nemotron-Nano-9B-v2` ($0.50/$1.50), `NVIDIA-Nemotron-Nano-12B-v2` ($0.70/$2.10), `Llama-3.1-Nemotron-Nano-4B-v1.1` ($0.30/$0.90)
    - Nemotron Super: `Llama-3_3-Nemotron-Super-49B-v1_5` ($1.50/$4.50)
    - Vision: `Llama-3.1-Nemotron-Nano-VL-8B-V1` ($1.00/$3.00 - 747K downloads), `NVLM-D-72B` ($2.00/$6.00)
    - Specialized: `OpenReasoning-Nemotron-7B` ($0.40/$1.20), `Cosmos-Reason1-7B` ($0.80/$2.40 - 413K downloads)
  - **Qwen3 Series** (18 new models):
    - Base Models: `Qwen3-0.6B` ($0.05/$0.10), `Qwen3-1.7B` ($0.10/$0.20), `Qwen3-4B` ($0.30/$0.60), `Qwen3-8B` ($0.50/$1.00), `Qwen3-14B` ($0.80/$1.60), `Qwen3-32B` ($1.20/$2.40)
    - Instruct: `Qwen3-4B-Instruct-2507` ($0.30/$0.60 - 5M+ downloads), `Qwen3-4B-Thinking-2507` ($0.60/$1.80)
    - MoE: `Qwen3-30B-A3B-Instruct-2507` ($0.40/$1.20), `Qwen3-30B-A3B-Thinking-2507` ($0.80/$2.40), `Qwen3-Next-80B-A3B-Instruct` ($0.60/$1.80), `Qwen3-235B-A22B` ($0.50/$1.50)
    - Specialized: `Qwen3-Coder-30B-A3B-Instruct` ($0.40/$1.20), `Qwen3-Omni-30B-A3B-Instruct` ($0.60/$1.80 - multimodal with text-to-audio)
  - **Ollama Variants** (9 new models):
    - `gemma3:1b` ($0.10/$0.20 - 2.6M downloads), `gemma3:4b` ($0.50/$1.50), `gemma3:12b` ($1.00/$3.00)
    - `deepseek-r1:1.5b` ($0.20/$0.40 - 1.0M downloads), `deepseek-r1:671b` ($1.40/$2.80)
    - `llama3.3:70b` ($2.00/$6.00 - 659K downloads)
    - `granite3.1:1b` ($0.10/$0.30), `granite3.1:3b` ($0.30/$0.90), `granite3.1:8b` ($0.50/$1.50)
  - **Embedding Models** (4 new models):
    - Snowflake: `Snowflake/snowflake-arctic-embed-m` ($0.03/$0.03 - 496K downloads), `snowflake-arctic-embed-s` ($0.02/$0.02), `snowflake-arctic-embed-m-v2.0` ($0.03/$0.03), `snowflake-arctic-embed-xs` ($0.01/$0.01)
    - NVIDIA: `nvidia/NV-Embed-v2` ($0.05/$0.05 - 198K downloads), `llama-embed-nemotron-8b` ($0.06/$0.06), `omni-embed-nemotron-3b` ($0.04/$0.04)
    - Google: `google/embeddinggemma-300m` ($0.02/$0.02)
  - **Speech-to-Text Models** (4 new models):
    - NVIDIA Parakeet: `parakeet-tdt-0.6b-v2` ($0.15/$0.15 - 3.7M downloads), `parakeet-rnnt-0.6b` ($0.15/$0.15 - 3.1M downloads), `parakeet-tdt-0.6b-v3` ($0.15/$0.15 - 49 languages)
    - NVIDIA Canary: `canary-1b-v2` ($0.20/$0.20 - ASR + Translation, 30+ languages)
  - All pricing reflects official provider rates and HuggingFace popularity metrics as of January 2025

## [0.1.21] - 2025-11-12

### Added

- **Automatic Server Metrics for ALL Instrumentors**
  - Integrated server metrics tracking into `BaseInstrumentor` - ALL instrumentors (OpenAI, Anthropic, Ollama, etc.) now automatically track active requests
  - `gen_ai.server.requests.running` counter automatically increments/decrements during request execution
  - Works for both streaming and non-streaming requests
  - Works across success and error paths
  - Implementation in `genai_otel/instrumentors/base.py:311-391, 816-839`

- **Ollama Automatic Server Metrics Collection**
  - Created `OllamaServerMetricsPoller` that automatically polls Ollama's `/api/ps` endpoint
  - Collects per-model VRAM usage and updates `gen_ai.server.kv_cache.usage{model="llama2"}` metric
  - Extracts model details: parameter size, quantization level, format, total size
  - Updates `gen_ai.server.requests.max` based on number of loaded models
  - Runs in background daemon thread with configurable interval (default: 5 seconds)
  - Enabled by default when Ollama instrumentation is active
  - Zero configuration required - works out of the box
  - **Requires Python 3.11+** (feature is skipped on Python 3.9 and 3.10)
  - Implementation in `genai_otel/instrumentors/ollama_server_metrics_poller.py` (157 lines, 94% coverage)

- **GPU VRAM Auto-Detection**
  - Automatic GPU VRAM detection using multiple fallback methods:
    1. **nvidia-ml-py** (pynvml) - preferred method, requires `pip install genai-otel-instrument[gpu]`
    2. **nvidia-smi** - automatic fallback using command-line tool
    3. **Manual override** - `GENAI_OLLAMA_MAX_VRAM_GB` environment variable (now optional)
  - Auto-detection runs once during poller initialization
  - Logs detected VRAM: "Auto-detected GPU VRAM: 24.0GB" or "GPU VRAM not detected, using heuristic-based percentages"
  - Eliminates need for manual VRAM configuration in most cases
  - Supports multi-GPU systems (uses first GPU for Ollama)
  - Implementation in `genai_otel/instrumentors/ollama_server_metrics_poller.py:81-172`

- **Enhanced Ollama Server Metrics Configuration**
  - New environment variables for Ollama server metrics:
    - `GENAI_ENABLE_OLLAMA_SERVER_METRICS` - Enable/disable automatic metrics (default: true)
    - `OLLAMA_BASE_URL` - Ollama server URL (default: http://localhost:11434)
    - `GENAI_OLLAMA_METRICS_INTERVAL` - Polling interval in seconds (default: 5.0)
    - `GENAI_OLLAMA_MAX_VRAM_GB` - Manual VRAM override (optional, auto-detected if not set)
  - Poller integrates with OllamaInstrumentor automatically
  - Graceful error handling for offline Ollama server or missing GPU
  - Implementation in `genai_otel/instrumentors/ollama_instrumentor.py:76-104`

### Improved

- **Test Coverage Enhancements**
  - Added 31 new comprehensive tests:
    - 18 tests for `OllamaServerMetricsPoller` (metrics collection, error handling, lifecycle)
    - 8 tests for GPU VRAM auto-detection (nvidia-ml-py, nvidia-smi, fallbacks, manual override)
    - 5 tests for Ollama instrumentor integration (poller startup, configuration, error handling)
  - Total tests increased from 496 to **527** (6.25% increase)
  - Improved `ollama_server_metrics_poller.py` coverage to **94%**
  - Improved `ollama_instrumentor.py` coverage to **97%**
  - Overall coverage maintained at **84%**
  - All tests passing with zero regressions

- **Documentation Updates**
  - Added "Ollama Automatic Integration" section to `docs/SERVER_METRICS.md`
  - Documented GPU VRAM auto-detection workflow with fallback methods
  - Updated `sample.env` with detailed comments on auto-detection
  - Created comprehensive example: `examples/ollama/example_with_server_metrics.py`
  - All Ollama server metrics are now fully documented with configuration examples

### Changed

- **GENAI_OLLAMA_MAX_VRAM_GB Now Optional**
  - Environment variable is no longer required
  - Auto-detection attempts to determine GPU VRAM automatically
  - Only set this variable if you want to override auto-detection or if auto-detection fails
  - Fallback heuristic still works if both auto-detection methods fail

## [0.1.20] - 2025-11-11

### Added

- **NVIDIA NIM-Inspired Server Metrics**
  - Added KV cache usage tracking: `gen_ai.server.kv_cache.usage` (Gauge) - GPU KV-cache usage percentage per model
  - Added request queue metrics:
    - `gen_ai.server.requests.running` (Gauge) - Active requests currently executing
    - `gen_ai.server.requests.waiting` (Gauge) - Requests waiting in queue
    - `gen_ai.server.requests.max` (Gauge) - Maximum concurrent request capacity
  - New `ServerMetricsCollector` class with thread-safe manual instrumentation API
  - Exported via `genai_otel.get_server_metrics()` for programmatic access

- **Token Distribution Histograms**
  - `gen_ai.client.token.usage.prompt` (Histogram) - Distribution of prompt tokens per request
  - `gen_ai.client.token.usage.completion` (Histogram) - Distribution of completion tokens per request
  - Configurable buckets from 1 to 67M tokens for analyzing token usage patterns
  - Enables p50, p95, p99 analysis of token consumption

- **Finish Reason Tracking**
  - `gen_ai.server.request.finish` (Counter) - All finished requests by finish reason (stop, length, error, content_filter, etc.)
  - `gen_ai.server.request.success` (Counter) - Successful completions (stop/length reasons)
  - `gen_ai.server.request.failure` (Counter) - Failed requests (error/content_filter/timeout reasons)
  - `gen_ai.response.finish_reason` span attribute for detailed tracing
  - Implemented `_extract_finish_reason()` in OpenAI instrumentor (example for other providers)

### Improved

- **Test Coverage**
  - Added 16 new tests covering server metrics, token histograms, and finish reason tracking
  - Total tests increased from 480 to 496
  - Overall coverage maintained at 83%, new server_metrics.py has 100% coverage
  - All metrics are thread-safe with comprehensive concurrency tests

## [0.1.19] - 2025-01-05

### Fixed

- **LangChain Instrumentation: Standard GenAI Attributes and Cost Tracking**
  - Fixed missing standard GenAI semantic convention attributes (gen_ai.system, gen_ai.request.model, gen_ai.operation.name, gen_ai.request.message_count)
  - Fixed missing token usage metrics (gen_ai.usage.prompt_tokens, gen_ai.usage.completion_tokens, gen_ai.usage.total_tokens)
  - Fixed missing cost calculation and tracking (gen_ai.usage.cost.total and granular costs)
  - Fixed missing latency metrics recording
  - Applied fixes to all chat model methods: invoke(), ainvoke(), batch(), abatch()
  - Maintained backward compatibility with langchain.* attributes
  - Removed redundant _extract_and_record_usage() method, improved code coverage from 71% to 81%
  - LangChain instrumentation now provides the same comprehensive observability as other provider instrumentors

## [0.1.18] - 2025-11-05

### Improved

- **Test Coverage Enhancements**
  - Added comprehensive tests for GPU metrics collection (11 new tests)
  - Added comprehensive tests for cost enriching exporter (20 new tests)
  - Improved `genai_otel/gpu_metrics.py` coverage from 72% to 93%
  - Improved `genai_otel/cost_enriching_exporter.py` coverage from 20% to 100%
  - Overall test coverage improved from 81% to 83%
  - 480 total tests passing (30 new tests added)

## [0.1.17] - 2025-11-05

### Added

- **Enhanced LangChain Instrumentation**
  - Direct chat model instrumentation with support for invoke(), ainvoke(), batch(), abatch() methods
  - Captures model name, provider, message count, and token usage
  - Creates langchain.chat_model.* spans for better visibility
  - Supports both usage_metadata and response_metadata formats

- **Automated CI/CD Publishing Pipeline**
  - Full test suite execution before publishing
  - Code quality checks (black, isort validation)
  - Automated publishing to Test PyPI and production PyPI
  - Package installation verification in isolated environment
  - Release summary generation

- **Documentation Improvements**
  - Added comprehensive release documentation (.github/RELEASE_GUIDE.md, .github/RELEASE_QUICKSTART.md)
  - Enhanced environment variable documentation in sample.env
  - Added OTEL_EXPORTER_OTLP_TIMEOUT, OTEL_EXPORTER_OTLP_PROTOCOL, OTEL_SERVICE_INSTANCE_ID, OTEL_ENVIRONMENT, GENAI_GPU_COLLECTION_INTERVAL documentation
  - Cleaned up obsolete documentation files

### Fixed

- **OTLP Exporter Timeout Type Conversion Error**
  - Changed exporter_timeout from float to int in OTelConfig
  - Added _get_exporter_timeout() helper with graceful error handling
  - Invalid timeout values now default to 60 seconds with warning
  - Fixes ValueError: invalid literal for int() with base 10: '10.0'

- **Test Suite Stability**
  - Removed problematic test files that caused hanging (tests/test_cost_enriching_exporter.py, tests/test_gpu_metrics.py, tests/instrumentors/test_togetherai_instrumentor.py)
  - Test suite now completes successfully
  - Restored stable test execution for CI/CD pipeline

## [0.1.16] - 2025-11-05

### Fixed

- Reverted test coverage improvements that caused test suite hangs
  - Reverted commit 73842f5 which introduced OpenTelemetry global state pollution
  - Test suite now completes successfully (442 tests passing)
  - Eliminated hanging issues in test_vertexai_instrumentor.py and related tests
  - Restored stable test execution for CI/CD pipeline

### Note

This release focuses on stability by reverting problematic test coverage improvements. The test coverage improvements will be reintroduced in a future release with proper test isolation.

## [0.1.14] - 2025-10-29

### Changed

- **BREAKING: License changed from Apache-2.0 to AGPL-3.0-or-later**
  - Provides stronger copyleft protection for the project
  - Network provision requires sharing source code for modified versions used over network
  - Full license text in LICENSE file with Copyright (C) 2025 Kshitij Thakkar
  - Updated all license references in pyproject.toml, __init__.py, and README.md
  - Completed LICENSE template with program name, copyright, and contact information

- **Project Rebranding to TraceVerde**
  - Display name changed from "GenAI OpenTelemetry Auto-Instrumentation" to "TraceVerde"
  - Package name remains `genai-otel-instrument` for PyPI compatibility (no breaking changes)
  - Updated README.md title, branding, and license badges

### Fixed

- Removed `__version__.py` from version control (generated file, should not be tracked)
- This fixes versioning issues during builds

**⚠️ Important**: Users should review AGPL-3.0 license terms before upgrading, especially for commercial/SaaS deployments

## [0.1.12] - 2025-10-29

### Added

- **Enhanced README Documentation**
  - Added professional project logo centered at the top of README
  - Added landing page hero image showcasing the project overview
  - Added comprehensive Screenshots section with 5 embedded demonstration images:
    - OpenAI instrumentation with token usage, costs, and latency metrics
    - Ollama (local LLM) zero-code instrumentation
    - HuggingFace Transformers with automatic token counting
    - SmolAgents framework with complete agent workflow tracing
    - GPU metrics collection dashboard
  - Added links to additional screenshots (Token Cost Breakdown, OpenSearch Dashboard)
  - Added Demo Video section with placeholder for future video content
  - All images follow OSS documentation standards with professional formatting

### Changed

- **Roadmap Section Cleanup**
  - Removed Phase 4 implementation details from roadmap (Session & User Tracking, RAG/Embedding Attributes)
  - Phase 4 features are now fully implemented and documented in the Advanced Features section
  - Roadmap now focuses exclusively on future releases (v0.2.0 onwards)

### Improved

- **Comprehensive Model Pricing Database Update**
  - Expanded pricing coverage from 145+ to 240+ models across 15+ providers
  - **OpenAI GPT-5 Series** (4 new models):
    - `gpt-5` - $1.25/$10 per 1M tokens
    - `gpt-5-2025-08-07` - $1.25/$10 per 1M tokens
    - `gpt-5-mini` - $0.25/$2 per 1M tokens
    - `gpt-5-nano` - $0.10/$0.40 per 1M tokens
  - **Anthropic Claude 4/3.5 Variants** (13 new models):
    - Claude 4 Opus series: `claude-4-opus`, `claude-opus-4`, `claude-opus-4-1`, `claude-opus-4.1` - $15/$75 per 1M tokens
    - Claude 3.5 Sonnet: `claude-3-5-sonnet-20240620`, `claude-3-5-sonnet-20241022`, `claude-sonnet-4-5`, `claude-sonnet-4-5-20250929`, `claude-3-7-sonnet` - $3/$15 per 1M tokens
    - Claude 3.5 Haiku: `claude-3-5-haiku-20241022` - $0.80/$4 per 1M tokens
    - Claude Haiku 4.5: `claude-haiku-4-5` - $1/$5 per 1M tokens
  - **XAI Grok Models** (10 new models):
    - Grok 2: `grok-2-1212`, `grok-2-vision-1212` - $2/$10 per 1M tokens
    - Grok 3: `grok-3` - $3/$15 per 1M tokens, `grok-3-mini` - $0.30/$0.50 per 1M tokens
    - Grok 3 Fast: `grok-3-fast` - $5/$25 per 1M tokens, `grok-3-mini-fast` - $0.60/$4 per 1M tokens
    - Grok 4: `grok-4` - $3/$15 per 1M tokens, `grok-4-fast` - $0.20/$0.50 per 1M tokens
    - Image models: `grok-image`, `xai-grok-image` - $0.07 per image
  - **Google Gemini Variants** (2 new models):
    - `gemini-2-5-flash-image` - $0.30/$30 per 1M tokens
    - `nano-banana` - $0.30/$30 per 1M tokens
  - **Qwen Series** (6 new models):
    - `qwen3-next-80b-a3b-instruct` - $0.525/$2.10 per 1M tokens
    - `qwen3-next-80b-a3b-thinking` - $0.525/$6.30 per 1M tokens
    - `qwen3-coder-480b-a35b-instruct` - $1/$5 per 1M tokens
    - `qwen3-max`, `qwen-qwen3-max` - $1.20/$6 per 1M tokens
  - **Meta Llama 4 Scout & Maverick** (6 models with updated pricing):
    - `llama-4-scout`, `llama-4-scout-17bx16e-128k`, `meta-llama/Llama-4-Scout` - $0.15/$0.50 per 1M tokens
    - `llama-4-maverick`, `llama-4-maverick-17bx128e-128k`, `meta-llama/Llama-4-Maverick` - $0.22/$0.85 per 1M tokens
  - **IBM Granite Models** (13 new models):
    - Granite 3 series: `ibm-granite-3-1-8b-instruct`, `ibm-granite-3-8b-instruct`, `granite-3-8b-instruct` - $0.20/$0.20 per 1M tokens
    - Granite 4 series: `granite-4-0-h-small`, `granite-4-0-h-tiny`, `granite-4-0-h-micro`, `granite-4-0-micro` - $0.20/$0.20 per 1M tokens
    - Embeddings: `granite-embedding-107m-multilingual`, `granite-embedding-278m-multilingual` - $0.10/$0.10 per 1M tokens
    - Ollama variants: `granite:3b`, `granite:8b` - $0.20/$0.20 per 1M tokens
  - **Mistral AI Updates** (10 new models):
    - `mistral-large-24-11`, `mistral-large-2411` - $8/$24 per 1M tokens
    - `mistral-small-3-1`, `mistral-small-3.1` - $1/$3 per 1M tokens
    - `mistral-medium-3`, `mistral-medium-2025` - $0.40/$2 per 1M tokens
    - Magistral series: `magistral-small` - $1/$3, `magistral-medium` - $3/$9 per 1M tokens
    - Codestral: `codestral-25-01`, `codestral-2501` - $1/$3 per 1M tokens
  - **Additional Providers**:
    - **Sarvam AI**: `sarvam-m`, `sarvamai/sarvam-m`, `sarvam-chat` - Free (Open source)
    - **Liquid AI**: `lfm-7b`, `liquid/lfm-7b` - $0.30/$0.60 per 1M tokens
    - **Snowflake**: `snowflake-arctic`, `snowflake-arctic-instruct` - $0.80/$2.40 per 1M tokens, `snowflake-arctic-embed-l-v2.0` - $0.05/$0.05 per 1M tokens
    - **NVIDIA Nemotron**: `nvidia-nemotron-4-340b-instruct` - $3/$9 per 1M tokens, `nvidia-nemotron-mini` - $0.20/$0.40 per 1M tokens, `nvidia/llama-3.1-nemotron-70b-instruct` - $0.80/$0.80 per 1M tokens
    - **ServiceNow**: `servicenow-now-assist` - $1/$3 per 1M tokens
  - **Pricing Corrections**:
    - `deepseek-v3.1`: Updated to $0.56/$1.68 per 1M tokens (from $1.20/$1.20)
    - `qwen3:3b`: Renamed to `qwen3:4b` (4B parameter model)
  - All pricing reflects official provider rates as of October 2025

## [0.1.9] - 2025-01-27

### Added

- **HuggingFace AutoModelForCausalLM and AutoModelForSeq2SeqLM Instrumentation**
  - Added support for direct model usage via `AutoModelForCausalLM.generate()` and `AutoModelForSeq2SeqLM.generate()`
  - Automatic token counting from input and output tensor shapes
  - Cost calculation based on model parameter count (uses CostCalculator's local model pricing tiers)
  - Span attributes: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.operation.name`, token counts, costs
  - Metrics: request counter, token counter, latency histogram, cost counter
  - Supports generation parameters: `max_length`, `max_new_tokens`, `temperature`, `top_p`
  - Implementation in `genai_otel/instrumentors/huggingface_instrumentor.py:184-333`
  - Example usage in `examples/huggingface/example_automodel.py`
  - All 443 tests pass (added 1 new test)

### Fixed

- **CRITICAL: Cost Tracking for OpenInference Instrumentors (smolagents, litellm, mcp)**
  - Replaced `CostEnrichmentSpanProcessor` with `CostEnrichingSpanExporter` to properly add cost attributes
  - **Root Cause**: SpanProcessor's `on_end()` receives immutable `ReadableSpan` objects that cannot be modified
  - **Solution**: Custom SpanExporter that enriches span data before export, creating new ReadableSpan instances with cost attributes
  - Cost attributes now correctly appear for smolagents, litellm, and mcp spans:
    - `gen_ai.usage.cost.total`: Total cost in USD
    - `gen_ai.usage.cost.prompt`: Prompt tokens cost
    - `gen_ai.usage.cost.completion`: Completion tokens cost
  - Supports all OpenInference semantic conventions:
    - Model name: `llm.model_name`, `gen_ai.request.model`, `embedding.model_name`
    - Token counts: `llm.token_count.{prompt,completion}`, `gen_ai.usage.{prompt_tokens,completion_tokens}`
    - Span kinds: `openinference.span.kind` (LLM, EMBEDDING, CHAIN, etc.)
  - Implementation in `genai_otel/cost_enriching_exporter.py`
  - Updated `genai_otel/auto_instrument.py` to wrap OTLP and Console exporters
  - Model name normalization handles provider prefixes (e.g., `openai/gpt-3.5-turbo` → `gpt-3.5-turbo`)
  - All 442 existing tests continue to pass

- **HuggingFace AutoModelForCausalLM AttributeError Fix**
  - Fixed `AttributeError: type object 'AutoModelForCausalLM' has no attribute 'generate'`
  - Root cause: `AutoModelForCausalLM` is a factory class; `generate()` exists on `GenerationMixin`
  - Solution: Wrap `GenerationMixin.generate()` which all generative models inherit from
  - This covers all model types: `AutoModelForCausalLM`, `AutoModelForSeq2SeqLM`, `GPT2LMHeadModel`, etc.
  - Added fallback import for older transformers versions
  - Implementation in `genai_otel/instrumentors/huggingface_instrumentor.py:184-346`

## [0.1.7] - 2025-01-25

### Added

- **Phase 4: Session and User Tracking (4.1)**
  - Added `session_id_extractor` and `user_id_extractor` optional callable fields to `OTelConfig`
  - Extractor function signature: `(instance, args, kwargs) -> Optional[str]`
  - Automatically sets `session.id` and `user.id` span attributes when extractors are configured
  - Enables tracking conversations across multiple requests for the same session
  - Supports per-user analytics, cost attribution, and debugging
  - Implementation in `genai_otel/config.py:134-139` and `genai_otel/instrumentors/base.py:266-284`
  - Documented in README.md with comprehensive examples
  - Example implementation in `examples/phase4_session_rag_tracking.py`

- **Phase 4: RAG and Embedding Attributes (4.2)**
  - Added `add_embedding_attributes()` helper method to `BaseInstrumentor`
    - Sets `embedding.model_name`, `embedding.text`, `embedding.vector`, `embedding.vector.dimension`
    - Truncates text to 500 characters to avoid span size explosion
  - Added `add_retrieval_attributes()` helper method to `BaseInstrumentor`
    - Sets `retrieval.query`, `retrieval.document_count`
    - Sets per-document attributes: `retrieval.documents.{i}.document.id`, `.score`, `.content`, `.metadata.*`
    - Limits to 5 documents by default (configurable via `max_docs` parameter)
    - Truncates content and metadata to prevent excessive attribute counts
  - Enables enhanced observability for RAG (Retrieval-Augmented Generation) workflows
  - Implementation in `genai_otel/instrumentors/base.py:705-770`
  - Documented in README.md with usage examples and best practices
  - Complete RAG workflow example in `examples/phase4_session_rag_tracking.py`

- **Phase 4 Documentation and Examples**
  - Added "Advanced Features" section to README.md
  - Documented session/user tracking with extractor function patterns
  - Documented RAG/embedding attributes with helper method usage
  - Created comprehensive example file `examples/phase4_session_rag_tracking.py` demonstrating:
    - Session and user extractor functions
    - Embedding attribute capture
    - Retrieval attribute capture with document metadata
    - Complete RAG workflow with session tracking
  - Updated roadmap section to mark Phase 4 as completed
  - **Note**: Agent workflow tracking (`agent.name`, `agent.iteration`, etc.) is provided by the existing OpenInference Smolagents instrumentor, not new in Phase 4

## [0.1.5] - 2025-01-25

### Added

- **Streaming Cost Tracking and Token Usage**
  - Fixed missing cost calculation for streaming LLM requests
  - `_wrap_streaming_response()` now extracts usage from the last chunk and calculates costs
  - Streaming responses now record all cost metrics: `gen_ai.usage.cost.total`, `gen_ai.usage.cost.prompt`, `gen_ai.usage.cost.completion`, etc.
  - Token usage metrics now properly recorded for streaming: `gen_ai.usage.prompt_tokens`, `gen_ai.usage.completion_tokens`, `gen_ai.usage.total_tokens`
  - Works for all providers that include usage in final chunk (OpenAI, Anthropic, Google, etc.)
  - Streaming metrics still captured: `gen_ai.server.ttft` (histogram), `gen_ai.server.tbt` (histogram), `gen_ai.streaming.token_count` (chunk count)
  - Implementation in `genai_otel/instrumentors/base.py:551-638`
  - Resolves issue where streaming requests had TTFT/TBT but no cost/usage tracking

### Fixed

- **GPU Metrics Test Infrastructure**
  - Fixed GPU metrics test mocks to return separate Mock objects for CO2 and power cost counters
  - Updated `mock_meter` fixture in `tests/test_gpu_metrics.py` to use `side_effect` for multiple counters
  - Fixed `test_auto_instrument.py` assertions to use dynamic `config.gpu_collection_interval` instead of hardcoded values
  - All 434 tests now pass with proper GPU power cost tracking validation

## [0.1.4] - 2025-01-24

### Added

- **Custom Model Pricing via Environment Variable**
  - Added `GENAI_CUSTOM_PRICING_JSON` environment variable for custom/proprietary model pricing
  - Supports all pricing categories: chat, embeddings, audio, images
  - Custom prices merged with default `llm_pricing.json` (custom takes precedence)
  - Enables pricing for internal/proprietary models not in public pricing database
  - Format: `{"chat":{"model-name":{"promptPrice":0.001,"completionPrice":0.002}}}`
  - Added `custom_pricing_json` field to `OTelConfig` dataclass
  - Updated `CostCalculator.__init__()` to accept custom pricing parameter
  - Implemented `CostCalculator._merge_custom_pricing()` with validation and error handling
  - Added `BaseInstrumentor._setup_config()` helper to reinitialize cost calculator
  - Added 8 comprehensive tests in `TestCustomPricing` class
  - Documented in README.md with usage examples and pricing format guide
  - Documented in sample.env with multiple examples

- **GPU Power Cost Tracking**
  - Added `GENAI_POWER_COST_PER_KWH` environment variable for electricity cost tracking (default: $0.12/kWh)
  - New metric `gen_ai.power.cost` tracks cumulative electricity costs in USD based on GPU power consumption
  - Calculates cost from GPU power draw: (energy_Wh / 1000) * cost_per_kWh
  - Includes `gpu_id` and `gpu_name` attributes for multi-GPU systems
  - Works alongside existing CO2 emissions tracking (`gen_ai.co2.emissions`)
  - Added `power_cost_per_kwh` field to `OTelConfig` dataclass
  - Implemented in `GPUMetricsCollector._collect_loop()` in `gpu_metrics.py`
  - Added 2 comprehensive tests: basic tracking and custom rate validation
  - Documented in README.md, sample.env, and CHANGELOG.md
  - Common electricity rates provided as reference: US $0.12, Europe $0.20, Industrial $0.07

- **HuggingFace InferenceClient Instrumentation**
  - Added full instrumentation support for HuggingFace Inference API via `InferenceClient`
  - Enables observability for smolagents workflows using `InferenceClientModel`
  - Wraps `InferenceClient.chat_completion()` and `InferenceClient.text_generation()` methods
  - Creates child spans showing actual HuggingFace API calls under agent/tool spans
  - Extracts model name, temperature, max_tokens, top_p from API calls
  - Supports both object and dict response formats for token usage
  - Handles streaming responses with `gen_ai.server.ttft` and `gen_ai.streaming.token_count`
  - Cost tracking enabled via fallback estimation based on model parameter count
  - Implementation in `genai_otel/instrumentors/huggingface_instrumentor.py:141-222`
  - Added 10 comprehensive tests covering all InferenceClient functionality
  - Coverage increased from 85% → 98% for HuggingFace instrumentor
  - Resolves issue where only AGENT and TOOL spans were visible without LLM child spans

- **Fallback Cost Estimation for Local Models (Ollama & HuggingFace)**
  - Added 36 Ollama models to `llm_pricing.json` with parameter-count-based pricing tiers
  - Implemented intelligent fallback cost estimation for unknown local models in `CostCalculator`
  - Automatically parses parameter count from model names (e.g., "360m", "7b", "70b")
  - Supports both Ollama and HuggingFace model naming patterns:
    - Explicit sizes: `llama3:7b`, `mistral-7b-v0.1`, `smollm2:360m`
    - HuggingFace size indicators: `gpt2`, `gpt2-xl`, `bert-base`, `t5-xxl`, etc.
  - Applies tiered pricing based on parameter count:
    - Tiny (< 1B): $0.0001 / $0.0002 per 1k tokens
    - Small (1-10B): $0.0003 / $0.0006
    - Medium (10-20B): $0.0005 / $0.001
    - Large (20-80B): $0.0008 / $0.0008
    - XLarge (80B+): $0.0012 / $0.0012
  - Acknowledges that local models are free but consume GPU power and electricity
  - Provides synthetic cost estimates for carbon footprint and resource tracking
  - Added `scripts/add_ollama_pricing.py` to update pricing database with new Ollama models
  - Logs fallback pricing usage at INFO level for transparency

### Improved

- **CostEnrichmentSpanProcessor Performance Optimization**
  - Added early-exit logic to skip spans that already have cost attributes
  - Checks for `gen_ai.usage.cost.total` presence before attempting enrichment
  - Saves processing compute by avoiding redundant cost calculations
  - Eliminates warning messages for spans enriched by instrumentors
  - Benefits all instrumentors that set cost attributes directly (Mistral, OpenAI, Anthropic, etc.)
  - Implementation in `genai_otel/cost_enrichment_processor.py:69-74`
  - Added comprehensive test coverage for skip logic
  - Coverage increased from 94% → 98% for CostEnrichmentSpanProcessor

### Fixed

- **CRITICAL: Complete Rewrite of Mistral AI Instrumentor**
  - **Root problem**: Original instrumentor used instance-level wrapping which didn't work reliably
  - **Complete architectural rewrite** using class-level method wrapping with `wrapt.wrap_function_wrapper()`
  - Now properly wraps `Chat.complete`, `Chat.stream`, and `Embeddings.create` at the class level
  - All Mistral client instances now use instrumented methods automatically
  - **Streaming support** with custom `_StreamWrapper` class:
    - Iterates through streaming chunks and collects usage data
    - Records TTFT (Time To First Token) metric
    - Creates mock response objects for proper metrics recording
  - **Proper error handling** with span exception recording
  - **Cost tracking** now works correctly with BaseInstrumentor integration
  - Fixed incorrect `_record_result_metrics()` signature usage
  - Implementation in `genai_otel/instrumentors/mistralai_instrumentor.py` (180 lines, completely rewritten)
  - All 5 Mistral tests passing with proper mocking
  - Traces now collected with full details: model, tokens, costs, TTFT
  - Resolves issue where no Mistral spans were being collected

- **CRITICAL: Fixed Missing Granular Cost Counter Class Variables**
  - Fixed `AttributeError: 'OllamaInstrumentor' object has no attribute '_shared_prompt_cost_counter'`
  - **Root cause**: Granular cost counters were created in initialization but not declared as class variables
  - **Impact**: Test suite failed with 34 errors when running full suite (but passed individually)
  - Added missing class variable declarations in `BaseInstrumentor`:
    - `_shared_prompt_cost_counter`
    - `_shared_completion_cost_counter`
    - `_shared_reasoning_cost_counter`
    - `_shared_cache_read_cost_counter`
    - `_shared_cache_write_cost_counter`
  - Created instance variable references in `__init__` for all granular counters
  - Updated all references to use instance variables instead of `_shared_*` variables
  - Implementation in `genai_otel/instrumentors/base.py:85-90, 106-111`
  - All 424 tests now passing consistently
  - Affects all instrumentors using granular cost tracking

- **CRITICAL: Fixed Cost Tracking Disabled by Wrong Variable Check**
  - **Root cause**: Cost tracking checked `self._shared_cost_counter` which was always None
  - Should have checked `self.config.enable_cost_tracking` flag only
  - **Impact**: Cost attributes were never added to spans even when cost tracking was enabled
  - Removed unnecessary `cost_counter` existence check
  - Cost tracking now properly controlled by `GENAI_ENABLE_COST_TRACKING` environment variable
  - Implementation in `genai_otel/instrumentors/base.py:384`
  - Debug logging confirmed cost calculation working: "Calculating cost for model=smollm2:360m"
  - Affects all instrumentors (Ollama, Mistral, OpenAI, Anthropic, etc.)

- **CRITICAL: Fixed Token and Cost Attributes Not Being Set on Spans**
  - Fixed critical bug where `gen_ai.usage.prompt_tokens`, `gen_ai.usage.completion_tokens`, and all cost attributes were not being set on spans
  - **Root causes:**
    1. Span attributes were only set if metric counters were available, but this check was too restrictive
    2. Used wrong variable name (`self._shared_cost_counter` instead of `self.cost_counter`) in cost tracking check
  - **Impact**: Cost calculation completely failed - only `gen_ai.usage.total_tokens` was set
  - **Fixed by:**
    1. Always setting span attributes regardless of metric availability
    2. Using correct instance variables (`self.cost_counter`, `self.token_counter`)
    3. Metrics recording is now optional, but span attributes are always set
    4. Cost attributes (`gen_ai.usage.cost.total`, `gen_ai.usage.cost.prompt`, `gen_ai.usage.cost.completion`) are now always added
  - This ensures cost tracking works even if metrics initialization fails
  - Affects all instrumentors (OpenAI, Anthropic, Ollama, etc.)

- **CRITICAL: Fixed 6 Instrumentors Missing `self._instrumented = True`**
  - Ollama, Cohere, HuggingFace, Replicate, TogetherAI, and VertexAI instrumentors were completely broken
  - No traces were being collected because `self._instrumented` flag was not set after wrapping functions
  - The `create_span_wrapper()` checks this flag and skips instrumentation if False
  - Added `self._instrumented = True` after successful wrapping in all 6 instrumentors
  - All instrumentors now properly collect traces again

- **CRITICAL: CostEnrichmentSpanProcessor Now Working**
  - Fixed critical bug where `CostEnrichmentSpanProcessor` was calling `calculate_cost()` (returns float) but treating it as a dict
  - This caused all cost enrichment to silently fail with `TypeError: 'float' object is not subscriptable`
  - Now correctly calls `calculate_granular_cost()` which returns a proper dict with `total`, `prompt`, `completion` keys
  - Cost attributes (`gen_ai.usage.cost.total`, `gen_ai.usage.cost.prompt`, `gen_ai.usage.cost.completion`) will now be added to OpenInference spans (smolagents, litellm, mcp)
  - Improved error logging from `logger.debug` to `logger.warning` with full exception info for easier debugging
  - Added logging of successful cost enrichment at `INFO` level with span name, model, and token details
  - All 415 tests passing, including 20 cost enrichment processor tests

- **Fixed OpenInference Instrumentor Loading Order**
  - Corrected instrumentor initialization order to: smolagents → litellm → mcp
  - This matches the correct order found in working implementations
  - Ensures proper nested instrumentation and attribute capture

## [0.1.3] - 2025-01-23

### Added

- **Cost Enrichment for OpenInference Instrumentors**
  - **CostEnrichmentSpanProcessor**: New custom SpanProcessor that automatically adds cost tracking to spans created by OpenInference instrumentors (smolagents, litellm, mcp)
    - Extracts model name and token usage from existing span attributes
    - Calculates costs using the existing CostCalculator with 145+ model pricing data
    - Adds granular cost attributes: `gen_ai.usage.cost.total`, `gen_ai.usage.cost.prompt`, `gen_ai.usage.cost.completion`
    - **Dual Semantic Convention Support**: Works with both OpenTelemetry GenAI and OpenInference conventions
      - GenAI: `gen_ai.request.model`, `gen_ai.usage.{prompt_tokens,completion_tokens,input_tokens,output_tokens}`
      - OpenInference: `llm.model_name`, `embedding.model_name`, `llm.token_count.{prompt,completion}`
      - OpenInference span kinds: LLM, EMBEDDING, CHAIN, RETRIEVER, RERANKER, TOOL, AGENT
    - Maps operation names to call types (chat, embedding, image, audio) automatically
    - Gracefully handles missing data and errors without failing span processing
  - Enabled by default when `GENAI_ENABLE_COST_TRACKING=true`
  - Works alongside OpenInference's native instrumentation without modifying upstream code
  - 100% test coverage with 20 comprehensive test cases (includes 5 OpenInference-specific tests)

- **Comprehensive Cost Tracking Enhancements**
  - Added token usage extraction and cost calculation for **6 instrumentors**: Ollama, Cohere, Together AI, Vertex AI, HuggingFace, and Replicate
  - Implemented `create_span_wrapper()` pattern across all instrumentors for consistent metrics recording
  - Added `gen_ai.operation.name` attribute to all instrumentors for improved observability
  - Total instrumentors with cost tracking increased from 8 to **11** (37.5% increase)

- **Pricing Data Expansion**
  - Added pricing for **45+ new LLM models** from 3 major providers:
    - **Groq**: 9 models (Llama 3.1/3.3/4, Qwen, GPT-OSS, Kimi-K2)
    - **Cohere**: 5 models (Command R/R+/R7B, Command A, updated legacy pricing)
    - **Together AI**: 30+ models (DeepSeek R1/V3, Qwen 2.5/3, Mistral variants, GLM-4.5)
  - All pricing verified from official provider documentation (2025 rates)

- **Enhanced Instrumentor Implementations**
  - **Ollama**: Extracts `prompt_eval_count` and `eval_count` from response (local model usage tracking)
  - **Cohere**: Extracts from `meta.tokens` with `meta.billed_units` fallback
  - **Together AI**: OpenAI-compatible format with dual API support (client + legacy Complete API)
  - **Vertex AI**: Extracts `usage_metadata` with both snake_case and camelCase support
  - **HuggingFace**: Documented as local/free execution (no API costs)
  - **Replicate**: Documented as hardware-based pricing ($/second, not token-based)

### Improved

- **Standardization & Code Quality**
  - Standardized all instrumentors to use `BaseInstrumentor.create_span_wrapper()` pattern
  - Improved error handling with consistent `fail_on_error` support across all instrumentors
  - Enhanced documentation with comprehensive docstrings explaining pricing models
  - Added proper logging at all error points for better debugging
  - Thread-safe metrics initialization across all instrumentors

- **Test Coverage**
  - All **415 tests passing** (100% test success rate)
  - Increased overall code coverage to **89%**
  - Individual instrumentor coverage: HuggingFace (98%), OpenAI (98%), Anthropic (95%), Groq (94%)
  - Core modules at 100% coverage: config, metrics, logging, exceptions, __init__, cost_enrichment_processor
  - Updated 40+ tests to match new `create_span_wrapper()` pattern
  - Added 20 comprehensive tests for CostEnrichmentSpanProcessor (100% coverage)
    - 15 tests for GenAI semantic conventions
    - 5 tests for OpenInference semantic conventions

- **Documentation**
  - Updated all instrumentor docstrings to explain token extraction logic
  - Added comments documenting non-standard pricing models (hardware-based, local execution)
  - Improved code comments for complex fallback logic

## [0.1.2.dev0] - 2025-01-22

### Added

- **GPU Power Consumption Metric**
  - Added `gen_ai.gpu.power` observable gauge metric to track real-time GPU power consumption
  - Metric reports power usage in Watts with `gpu_id` and `gpu_name` attributes
  - Automatically collected alongside existing GPU metrics (utilization, memory, temperature)
  - Implementation in `genai_otel/gpu_metrics.py:97-102, 195-220`
  - Added test coverage in `tests/test_gpu_metrics.py:244-266`
  - Completes the GPU metrics suite with 5 total metrics: utilization, memory, temperature, power, and CO2 emissions

### Fixed

- **Test Fixes for HuggingFace and MistralAI Instrumentors**
  - Fixed HuggingFace instrumentor tests (2 failures) - corrected tracer mocking to use `instrumentor.tracer.start_span()` instead of `config.tracer.start_as_current_span()`
  - Fixed HuggingFace instrumentor tests - added `instrumentor.request_counter` mock for proper metrics assertion
  - Fixed MistralAI instrumentor test - corrected wrapt module mocking by adding to `sys.modules` instead of invalid module-level patch
  - All 395 tests now passing with zero failures
  - Tests modified: `tests/instrumentors/test_huggingface_instrumentor.py`, `tests/instrumentors/test_mistralai_instrumentor.py`

## [0.1.0] - 2025-01-20

**First Beta Release** 🎉

This is the first public release of genai-otel-instrument, a comprehensive OpenTelemetry auto-instrumentation library for LLM/GenAI applications with support for 15+ providers, frameworks, and MCP tools.

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

[Unreleased]: https://github.com/Mandark-droid/genai_otel_instrument/compare/v0.1.2.dev0...HEAD
[0.1.2.dev0]: https://github.com/Mandark-droid/genai_otel_instrument/compare/v0.1.0...v0.1.2.dev0
[0.1.0]: https://github.com/Mandark-droid/genai_otel_instrument/releases/tag/v0.1.0
