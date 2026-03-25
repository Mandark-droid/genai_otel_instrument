# Installation

## Basic Installation

```bash
pip install genai-otel-instrument
```

This installs the core library with OpenTelemetry SDK and basic instrumentation support.

## Optional Extras

Install additional capabilities as needed:

### LLM Providers

```bash
# Individual providers
pip install genai-otel-instrument[openai]
pip install genai-otel-instrument[anthropic]
pip install genai-otel-instrument[google]
pip install genai-otel-instrument[aws]
pip install genai-otel-instrument[cohere]
pip install genai-otel-instrument[mistral]
pip install genai-otel-instrument[groq]
pip install genai-otel-instrument[ollama]
pip install genai-otel-instrument[together]

# All providers
pip install genai-otel-instrument[all-providers]
```

### Frameworks

```bash
pip install genai-otel-instrument[crewai]
pip install genai-otel-instrument[langchain]
pip install genai-otel-instrument[langgraph]
pip install genai-otel-instrument[google-adk]
pip install genai-otel-instrument[autogen-agentchat]
```

### OpenInference (Python 3.10+)

```bash
pip install genai-otel-instrument[openinference]
```

Includes LiteLLM, Smolagents, and MCP instrumentation via OpenInference.

### GPU Metrics

```bash
pip install genai-otel-instrument[gpu]       # NVIDIA only
pip install genai-otel-instrument[amd-gpu]   # AMD only
pip install genai-otel-instrument[all-gpu]   # Both
```

### Databases and MCP Tools

```bash
pip install genai-otel-instrument[databases]    # PostgreSQL, MySQL, MongoDB, Redis, etc.
pip install genai-otel-instrument[messaging]    # Kafka, RabbitMQ
pip install genai-otel-instrument[vector-dbs]   # Pinecone, Weaviate, Qdrant, ChromaDB, etc.
pip install genai-otel-instrument[all-mcp]      # All MCP tools
```

### Evaluation Features

```bash
pip install genai-otel-instrument[evaluation]   # PII, toxicity, bias detection
```

### Everything

```bash
pip install genai-otel-instrument[all]
```

## Development Installation

```bash
git clone https://github.com/Mandark-droid/genai_otel_instrument.git
cd genai_otel_instrument
pip install -e ".[dev,all]"
```

## Requirements

- Python 3.9+
- OpenTelemetry SDK 1.20.0+
- OpenInference features require Python 3.10+
