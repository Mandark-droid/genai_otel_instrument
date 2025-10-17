# AWS Bedrock Example

This example demonstrates auto-instrumentation of AWS Bedrock API calls.

## Setup

1. Install dependencies:
```bash
pip install genai-otel-instrument[aws]
```

2. Copy `.env.example` to `.env` and add your AWS credentials:
```bash
cp .env.example .env
# Edit .env and add your AWS credentials
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

- ✅ Model invocations (Claude, Titan, Jurassic, etc.)
- ✅ Streaming responses
- ✅ Token usage tracking
- ✅ Cost calculation
- ✅ Request/response metadata
- ✅ Error tracking
