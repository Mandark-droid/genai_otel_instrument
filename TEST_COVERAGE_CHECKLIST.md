# Test Coverage Checklist for genai-otel-instrument

This checklist tracks the progress of improving test coverage for the `genai_otel` library.
The goal is to achieve >80% coverage for each module and ensure tests are created for all files.

## Core Modules
- [x] genai_otel/__init__.py (Tests Created: Yes, Coverage > 80%: Yes)
- [x] genai_otel/__version__.py (Tests Created: Yes, Coverage > 80%: Yes)
- [x] genai_otel/auto_instrument.py (Tests Created: Yes, Coverage > 80%: Yes)
- [ ] genai_otel/cli.py (Tests Created: No, Coverage > 80%: No)
- [ ] genai_otel/config.py (Tests Created: No, Coverage > 80%: No)
- [ ] genai_otel/cost_calculator.py (Tests Created: No, Coverage > 80%: No)
- [x] genai_otel/exceptions.py (Tests Created: Yes, Coverage > 80%: Yes)
- [ ] genai_otel/gpu_metrics.py (Tests Created: No, Coverage > 80%: No)
- [x] genai_otel/logging_config.py (Tests Created: Yes, Coverage > 80%: Yes)
- [ ] genai_otel/otel_setup.py (Tests Created: No, Coverage > 80%: No)

## Instrumentors
- [x] genai_otel/instrumentors/__init__.py (Tests Created: Yes, Coverage > 80%: Yes)
- [ ] genai_otel/instrumentors/anthropic_instrumentor.py (Tests Created: No, Coverage > 80%: No)
- [ ] genai_otel/instrumentors/anyscale_instrumentor.py (Tests Created: No, Coverage > 80%: No)
- [ ] genai_otel/instrumentors/aws_bedrock_instrumentor.py (Tests Created: No, Coverage > 80%: No)
- [ ] genai_otel/instrumentors/azure_openai_instrumentor.py (Tests Created: No, Coverage > 80%: No)
- [x] genai_otel/instrumentors/base.py (Tests Created: Yes, Coverage > 80%: No)
- [ ] genai_otel/instrumentors/cohere_instrumentor.py (Tests Created: No, Coverage > 80%: No)
- [ ] genai_otel/instrumentors/google_ai_instrumentor.py (Tests Created: No, Coverage > 80%: No)
- [ ] genai_otel/instrumentors/groq_instrumentor.py (Tests Created: No, Coverage > 80%: No)
- [ ] genai_otel/instrumentors/huggingface_instrumentor.py (Tests Created: No, Coverage > 80%: No)
- [ ] genai_otel/instrumentors/langchain_instrumentor.py (Tests Created: No, Coverage > 80%: No)
- [ ] genai_otel/instrumentors/llamaindex_instrumentor.py (Tests Created: No, Coverage > 80%: No)
- [ ] genai_otel/instrumentors/mistralai_instrumentor.py (Tests Created: No, Coverage > 80%: No)
- [ ] genai_otel/instrumentors/ollama_instrumentor.py (Tests Created: No, Coverage > 80%: No)
- [ ] genai_otel/instrumentors/openai_instrumentor.py (Tests Created: No, Coverage > 80%: No)
- [ ] genai_otel/instrumentors/replicate_instrumentor.py (Tests Created: No, Coverage > 80%: No)
- [ ] genai_otel/instrumentors/togetherai_instrumentor.py (Tests Created: No, Coverage > 80%: No)
- [ ] genai_otel/instrumentors/vertexai_instrumentor.py (Tests Created: No, Coverage > 80%: No)

## MCP Instrumentors
- [x] genai_otel/mcp_instrumentors/__init__.py (Tests Created: Yes, Coverage > 80%: Yes)
- [ ] genai_otel/mcp_instrumentors/api_instrumentor.py (Tests Created: No, Coverage > 80%: No)
- [ ] genai_otel/mcp_instrumentors/database_instrumentor.py (Tests Created: No, Coverage > 80%: No)
- [ ] genai_otel/mcp_instrumentors/kafka_instrumentor.py (Tests Created: No, Coverage > 80%: No)
- [ ] genai_otel/mcp_instrumentors/manager.py (Tests Created: No, Coverage > 80%: No)
- [ ] genai_otel/mcp_instrumentors/redis_instrumentor.py (Tests Created: No, Coverage > 80%: No)
- [ ] genai_otel/mcp_instrumentors/vector_db_instrumentor.py (Tests Created: No, Coverage > 80%: No)
