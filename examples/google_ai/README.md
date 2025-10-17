# Google Gemini Example

This example demonstrates auto-instrumentation of Google Generative AI (Gemini) API calls.

## Setup

1. Install dependencies:
```bash
pip install genai-otel-instrument[google]
```

2. Copy `.env.example` to `.env` and add your Google API key:
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
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

- ✅ Content generation
- ✅ Chat conversations
- ✅ Token usage tracking
- ✅ Cost calculation
- ✅ Request/response metadata
- ✅ Error tracking
