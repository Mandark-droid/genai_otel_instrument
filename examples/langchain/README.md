# LangChain Example

This example demonstrates auto-instrumentation of LangChain chains and agents.

## Setup

1. Install dependencies:
```bash
pip install genai-otel-instrument[langchain,openai]
```

2. Copy `.env.example` to `.env` and add your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

3. Start an OTLP collector (e.g., Jaeger):
```bash
docker run -d --name jaeger \
  -e COLLECTOR_OTLP_ENABLED=true \
  -p 4318:4318 \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest
```

4. Run the example:
```bash
python example.py
```

5. View traces at http://localhost:16686

## What Gets Instrumented

- ✅ Chain execution
- ✅ Agent runs
- ✅ Nested LLM calls
- ✅ Token usage tracking
- ✅ Cost calculation
- ✅ Parent-child span relationships
