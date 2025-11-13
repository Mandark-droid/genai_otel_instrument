"""Hyperbolic API Instrumentation Example.

This example demonstrates automatic OpenTelemetry instrumentation for
Hyperbolic's API, which provides cost-effective access to models like
Qwen3, DeepSeek, and others.

IMPORTANT: Hyperbolic instrumentation requires OTLP gRPC exporters due to
conflicts with the requests library when using HTTP exporters. This example
sets up the correct configuration automatically.

Requirements:
    pip install genai-otel-instrument requests
    export HYPERBOLIC_API_KEY=your_api_key

Setup OTLP Collector:
    Ensure you have an OTLP collector running on port 4317 (gRPC).
    For local testing with Jaeger:
        docker run -d --name jaeger \
          -p 4317:4317 \
          -p 16686:16686 \
          jaegertracing/all-in-one:latest
"""

import os

import requests

import genai_otel

print("\n" + "=" * 80)
print("Hyperbolic OpenTelemetry Instrumentation Example")
print("=" * 80 + "\n")

# CRITICAL: Hyperbolic requires gRPC exporter (not HTTP)
# Set this BEFORE calling instrument()
os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "grpc"
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4317"

# Enable Hyperbolic instrumentation (disabled by default)
# Add "hyperbolic" to the enabled instrumentors list
os.environ["GENAI_ENABLED_INSTRUMENTORS"] = "hyperbolic"

print("Configuration:")
print(f"  OTLP Protocol: {os.environ['OTEL_EXPORTER_OTLP_PROTOCOL']}")
print(f"  OTLP Endpoint: {os.environ['OTEL_EXPORTER_OTLP_ENDPOINT']}")
print(f"  Enabled Instrumentors: {os.environ['GENAI_ENABLED_INSTRUMENTORS']}")
print()

# Initialize instrumentation with gRPC exporter
genai_otel.instrument(
    service_name="hyperbolic-example",
)

print("OpenTelemetry instrumentation initialized with gRPC exporter.")
print("-" * 80 + "\n")

# Check for API key
api_key = os.getenv("HYPERBOLIC_API_KEY")
if not api_key:
    print("ERROR: HYPERBOLIC_API_KEY environment variable not set")
    print("Get your API key from: https://www.hyperbolic.ai/")
    print("\nUsage: export HYPERBOLIC_API_KEY=your_key_here")
    exit(1)

# Hyperbolic API configuration
url = "https://api.hyperbolic.xyz/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
}

print("1. Testing Qwen3-Next-80B model...")
print("-" * 80)

# Example 1: Chat with Qwen3-Next-80B
data = {
    "messages": [{"role": "user", "content": "What are the benefits of serverless computing?"}],
    "model": "Qwen/Qwen3-Next-80B-A3B-Thinking",
    "max_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.8,
}

response = requests.post(url, headers=headers, json=data, timeout=30)
result = response.json()

print(f"Model: {result.get('model', 'N/A')}")
print(f"Response: {result['choices'][0]['message']['content'][:200]}...")
if "usage" in result:
    print(
        f"Tokens: {result['usage']['total_tokens']} "
        f"(prompt: {result['usage']['prompt_tokens']}, "
        f"completion: {result['usage']['completion_tokens']})"
    )
print()

print("2. Testing DeepSeek-V3 model...")
print("-" * 80)

# Example 2: DeepSeek-V3 for cost-effective inference
data = {
    "messages": [{"role": "user", "content": "Explain Docker containers briefly."}],
    "model": "deepseek-ai/DeepSeek-V3",
    "max_tokens": 100,
    "temperature": 0.5,
}

response = requests.post(url, headers=headers, json=data, timeout=30)
result = response.json()

print(f"Model: {result.get('model', 'N/A')}")
print(f"Response: {result['choices'][0]['message']['content']}")
if "usage" in result:
    print(f"Tokens: {result['usage']['total_tokens']}")
print()

print("3. Testing DeepSeek-R1 reasoning model...")
print("-" * 80)

# Example 3: DeepSeek-R1 for reasoning tasks
data = {
    "messages": [
        {
            "role": "user",
            "content": "If a train travels 60 mph for 2 hours, how far does it go?",
        }
    ],
    "model": "deepseek-ai/DeepSeek-R1",
    "max_tokens": 150,
    "temperature": 0.3,
}

response = requests.post(url, headers=headers, json=data, timeout=30)
result = response.json()

print(f"Model: {result.get('model', 'N/A')}")
print(f"Response: {result['choices'][0]['message']['content']}")
if "usage" in result:
    print(f"Tokens: {result['usage']['total_tokens']}")
print()

print("4. Multi-turn conversation with Qwen3-235B...")
print("-" * 80)

# Example 4: Multi-turn conversation
messages = [
    {"role": "user", "content": "What is Kubernetes?"},
]

data = {
    "messages": messages,
    "model": "Qwen/Qwen3-235B",
    "max_tokens": 100,
    "temperature": 0.7,
}

response = requests.post(url, headers=headers, json=data, timeout=30)
result = response.json()

assistant_msg = result["choices"][0]["message"]["content"]
print(f"User: {messages[0]['content']}")
print(f"Assistant: {assistant_msg}")

# Continue conversation
messages.append({"role": "assistant", "content": assistant_msg})
messages.append({"role": "user", "content": "What are its main benefits?"})

data["messages"] = messages
response = requests.post(url, headers=headers, json=data, timeout=30)
result = response.json()

print(f"\nUser: {messages[-1]['content']}")
print(f"Assistant: {result['choices'][0]['message']['content']}")
print()

print("=" * 80)
print("Telemetry Data Collected:")
print("=" * 80)
print(
    """
For each API call, the following data is automatically collected:

TRACES (Spans):
- Span name: hyperbolic.chat.completion
- Attributes:
  - gen_ai.system: "hyperbolic"
  - gen_ai.request.model: e.g., "Qwen/Qwen3-Next-80B-A3B-Thinking"
  - gen_ai.operation.name: "chat"
  - gen_ai.request.temperature, top_p, max_tokens
  - gen_ai.usage.prompt_tokens, completion_tokens, total_tokens
  - gen_ai.response.id, model, finish_reasons
  - gen_ai.cost.amount (estimated in USD)
  - http.status_code

METRICS:
- genai.requests: Request count by model and provider
- genai.tokens: Token usage (prompt/completion)
- genai.latency: Request duration histogram
- genai.cost: Estimated costs in USD
- genai.errors: Error counts (if any failures occur)

View these metrics in your observability platform (Jaeger, Grafana, etc.)
Access Jaeger UI at: http://localhost:16686
"""
)

print("=" * 80)
print("Hyperbolic Pricing (per 1M tokens):")
print("=" * 80)
print(
    """
- Qwen3-Next-80B:    $0.40 / $0.40  (input/output)
- Qwen3-235B:        $0.40 / $0.40  (input/output)
- DeepSeek-R1:       $2.00 / $2.00  (input/output)
- DeepSeek-V3:       $0.25 / $0.25  (input/output)

Hyperbolic claims up to 80% cost reduction vs traditional providers!
Cost tracking is automatically calculated and included in spans.
"""
)

print("=" * 80)
print("Why gRPC Exporter is Required:")
print("=" * 80)
print(
    """
Hyperbolic uses raw HTTP requests (requests.post()), and our instrumentation
wraps this globally to capture API calls. This conflicts with OTLP HTTP
exporters which also use the requests library internally.

The conflict causes this error:
  AttributeError: 'function' object has no attribute 'ok'

Solution: Use OTLP gRPC exporters (port 4317) instead of HTTP (port 4318).

This example sets the correct configuration automatically:
  - OTEL_EXPORTER_OTLP_PROTOCOL=grpc
  - OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
  - GENAI_ENABLED_INSTRUMENTORS includes "hyperbolic"

For more details, see CLAUDE.md in the repository.
"""
)

print("=" * 80)
print("Example complete! Check Jaeger UI at http://localhost:16686")
print("=" * 80 + "\n")
