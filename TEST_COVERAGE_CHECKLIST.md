# Test Coverage Checklist for genai-otel-instrument

This checklist tracks the progress of improving test coverage for the `genai_otel` library.
**GOAL: Achieve 100% coverage for all modules before release**

## Current Status: 72% Overall Coverage (144 tests passing)

## Core Modules
- [x] genai_otel/__init__.py (Tests: Yes, Coverage: 100% ✅)
- [x] genai_otel/__version__.py (Tests: Yes, Coverage: 100% ✅)
- [x] genai_otel/auto_instrument.py (Tests: Yes, Coverage: 98% ✅ - 2 unreachable lines in dead code)
- [x] genai_otel/cli.py (Tests: Yes, Coverage: 100% ✅)
- [x] genai_otel/config.py (Tests: Yes, Coverage: 100% ✅)
- [ ] genai_otel/cost_calculator.py (Tests: Yes, Coverage: 72%, Missing: 33 lines - needs comprehensive testing)
- [x] genai_otel/exceptions.py (Tests: Yes, Coverage: 100% ✅)
- [ ] genai_otel/gpu_metrics.py (Tests: Yes, Coverage: 92%, Missing: 11 lines - lines 24-26, 118-119, 139-140, 146, 152-154)
- [x] genai_otel/logging_config.py (Tests: Yes, Coverage: 100% ✅)
- [x] genai_otel/metrics.py (Tests: Yes, Coverage: 100% ✅)

## Instrumentors
- [x] genai_otel/instrumentors/__init__.py (Tests: Yes, Coverage: 100% ✅)
- [ ] genai_otel/instrumentors/anthropic_instrumentor.py (Tests: Yes, Coverage: 54%, Missing: 26 lines)
- [x] genai_otel/instrumentors/anyscale_instrumentor.py (Tests: Yes, Coverage: 83% ✅ - 2 unreachable lines in dead code)
- [ ] genai_otel/instrumentors/aws_bedrock_instrumentor.py (Tests: Yes, Coverage: 43%, Missing: 34 lines)
- [ ] genai_otel/instrumentors/azure_openai_instrumentor.py (Tests: Yes, Coverage: 59%, Missing: 16 lines)
- [x] genai_otel/instrumentors/base.py (Tests: Yes, Coverage: 100% ✅)
- [ ] genai_otel/instrumentors/cohere_instrumentor.py (Tests: Yes, Coverage: 56%, Missing: 20 lines)
- [ ] genai_otel/instrumentors/google_ai_instrumentor.py (Tests: Yes, Coverage: 59%, Missing: 18 lines)
- [ ] genai_otel/instrumentors/groq_instrumentor.py (Tests: Yes, Coverage: 51%, Missing: 26 lines)
- [x] genai_otel/instrumentors/huggingface_instrumentor.py (Tests: Yes, Coverage: 100% ✅)
- [ ] genai_otel/instrumentors/langchain_instrumentor.py (Tests: Yes, Coverage: 48%, Missing: 24 lines)
- [x] genai_otel/instrumentors/llamaindex_instrumentor.py (Tests: Yes, Coverage: 100% ✅)
- [ ] genai_otel/instrumentors/mistralai_instrumentor.py (Tests: Yes, Coverage: 36%, Missing: 59 lines)
- [x] genai_otel/instrumentors/ollama_instrumentor.py (Tests: Yes, Coverage: 100% ✅)
- [x] genai_otel/instrumentors/openai_instrumentor.py (Tests: Yes, Coverage: 100% ✅)
- [x] genai_otel/instrumentors/replicate_instrumentor.py (Tests: Yes, Coverage: 100% ✅)
- [x] genai_otel/instrumentors/togetherai_instrumentor.py (Tests: Yes, Coverage: 100% ✅)
- [ ] genai_otel/instrumentors/vertexai_instrumentor.py (Tests: Yes, Coverage: 50%, Missing: 11 lines)

## MCP Instrumentors
- [x] genai_otel/mcp_instrumentors/__init__.py (Tests: Yes, Coverage: 100% ✅)
- [ ] genai_otel/mcp_instrumentors/api_instrumentor.py (Tests: Yes, Coverage: 37%, Missing: 38 lines)
- [ ] genai_otel/mcp_instrumentors/database_instrumentor.py (Tests: Yes, Coverage: 64%, Missing: 16 lines)
- [x] genai_otel/mcp_instrumentors/kafka_instrumentor.py (Tests: Yes, Coverage: 100% ✅)
- [ ] genai_otel/mcp_instrumentors/manager.py (Tests: Yes, Coverage: 57%, Missing: 40 lines)
- [x] genai_otel/mcp_instrumentors/redis_instrumentor.py (Tests: Yes, Coverage: 100% ✅)
- [ ] genai_otel/mcp_instrumentors/vector_db_instrumentor.py (Tests: Yes, Coverage: 67%, Missing: 55 lines)

## Priority Order for Test Coverage (Target: 100%)

### HIGH PRIORITY (Critical for Release - Large Gaps)
1. **mistralai_instrumentor.py** - 36% coverage (59 missing lines) ⚠️
2. **aws_bedrock_instrumentor.py** - 43% coverage (34 missing lines) ⚠️
3. **manager.py** - 57% coverage (40 missing lines) ⚠️
4. **vector_db_instrumentor.py** - 67% coverage (55 missing lines) ⚠️
5. **api_instrumentor.py** - 37% coverage (38 missing lines) ⚠️
6. **cost_calculator.py** - 72% coverage (33 missing lines) ⚠️

### MEDIUM PRIORITY (Important Instrumentors)
7. **groq_instrumentor.py** - 51% coverage (26 missing lines)
8. **anthropic_instrumentor.py** - 54% coverage (26 missing lines)
9. **langchain_instrumentor.py** - 48% coverage (24 missing lines)
10. **cohere_instrumentor.py** - 56% coverage (20 missing lines)
11. **google_ai_instrumentor.py** - 59% coverage (18 missing lines)
12. **azure_openai_instrumentor.py** - 59% coverage (16 missing lines)
13. **database_instrumentor.py** - 64% coverage (16 missing lines)

### LOW PRIORITY (Small Gaps - Quick Wins)
14. **gpu_metrics.py** - 92% coverage (11 missing lines)
15. **vertexai_instrumentor.py** - 50% coverage (11 missing lines)
16. ~~**openai_instrumentor.py** - 83% coverage (10 missing lines)~~ ✅ **COMPLETED - 100%**
17. ~~**replicate_instrumentor.py** - 55% coverage (10 missing lines)~~ ✅ **COMPLETED - 100%**
18. ~~**togetherai_instrumentor.py** - 55% coverage (10 missing lines)~~ ✅ **COMPLETED - 100%**
19. ~~**base.py** - 93% coverage (9 missing lines)~~ ✅ **COMPLETED - 100%**
20. ~~**llamaindex_instrumentor.py** - 60% coverage (8 missing lines)~~ ✅ **COMPLETED - 100%**
21. ~~**auto_instrument.py** - 94% coverage (7 missing lines)~~ ✅ **IMPROVED to 98%** (2 unreachable lines remain)
22. ~~**config.py** - 89% coverage (5 missing lines)~~ ✅ **COMPLETED - 100%**
23. ~~**kafka_instrumentor.py** - 73% coverage (4 missing lines)~~ ✅ **COMPLETED - 100%**
24. ~~**anyscale_instrumentor.py** - 75% coverage (3 missing lines)~~ ✅ **IMPROVED to 83%** (2 unreachable lines remain)
25. ~~**huggingface_instrumentor.py** - 96% coverage (2 missing lines)~~ ✅ **COMPLETED - 100%**
26. ~~**logging_config.py** - 96% coverage (1 missing line)~~ ✅ **COMPLETED - 100%**
27. ~~**metrics.py** - 96% coverage (1 missing line)~~ ✅ **COMPLETED - 100%**

## New Features Testing
- [ ] auto_instrument.py: Add tests for SERVICE_INSTANCE_ID and environment attributes (lines 106-112)
- [ ] auto_instrument.py: Add tests for configurable timeout for OTLP exporters (lines 128-133)
- [x] auto_instrument.py: Tests for SmolagentsInstrumentor, MCPInstrumentor, LiteLLMInstrumentor ✅
- [x] auto_instrument.py: Tests for TraceContextTextMapPropagator ✅
