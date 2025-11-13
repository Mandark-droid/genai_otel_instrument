"""SambaNova Instrumentation Example.

This example demonstrates automatic OpenTelemetry instrumentation for
SambaNova's AI models including Llama 4 Maverick and Llama 3.1 models.

SambaNova provides fast, cost-effective inference for Llama models with
competitive pricing and high throughput.

Requirements:
    pip install genai-otel-instrument[sambanova]
    export SAMBANOVA_API_KEY=your_api_key
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
"""

import os

import genai_otel

# Initialize instrumentation - SambaNova is enabled by default
genai_otel.instrument(
    service_name="sambanova-example",
    endpoint="http://localhost:4318",
)

print("\n" + "=" * 80)
print("SambaNova OpenTelemetry Instrumentation Example")
print("=" * 80 + "\n")

# Import SambaNova SDK
try:
    from sambanova import SambaNova
except ImportError:
    print("ERROR: SambaNova SDK not installed. Install with:")
    print("  pip install sambanova")
    exit(1)

# Check for API key
api_key = os.getenv("SAMBANOVA_API_KEY")
if not api_key:
    print("ERROR: SAMBANOVA_API_KEY environment variable not set")
    print("Get your API key from: https://cloud.sambanova.ai/")
    exit(1)

# Initialize SambaNova client
client = SambaNova(
    api_key=api_key,
    base_url="https://api.sambanova.ai/v1",
)

print("1. Testing Llama 4 Maverick model...")
print("-" * 80)

# Example 1: Chat completion with Llama 4 Maverick
response = client.chat.completions.create(
    model="Llama-4-Maverick-17B-128E-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain quantum computing in 2 sentences."},
    ],
    temperature=0.7,
    max_tokens=100,
)

print(f"Model: {response.model}")
print(f"Response: {response.choices[0].message.content}")
print(
    f"Tokens: {response.usage.total_tokens} (prompt: {response.usage.prompt_tokens}, "
    f"completion: {response.usage.completion_tokens})"
)
print()

print("2. Testing Llama 3.1 8B model...")
print("-" * 80)

# Example 2: Llama 3.1 8B for faster, cheaper inference
response = client.chat.completions.create(
    model="Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "What is the capital of France?"},
    ],
    temperature=0.1,
    max_tokens=50,
)

print(f"Model: {response.model}")
print(f"Response: {response.choices[0].message.content}")
print(f"Tokens: {response.usage.total_tokens}")
print()

print("3. Testing Llama 3.1 70B model...")
print("-" * 80)

# Example 3: Llama 3.1 70B for more complex tasks
response = client.chat.completions.create(
    model="Meta-Llama-3.1-70B-Instruct",
    messages=[
        {"role": "user", "content": "Write a haiku about programming."},
    ],
    temperature=0.8,
    top_p=0.9,
)

print(f"Model: {response.model}")
print(f"Response: {response.choices[0].message.content}")
print(f"Tokens: {response.usage.total_tokens}")
print()

print("4. Multi-turn conversation...")
print("-" * 80)

# Example 4: Multi-turn conversation
messages = [
    {"role": "user", "content": "What is machine learning?"},
]

response = client.chat.completions.create(
    model="Meta-Llama-3.1-8B-Instruct",
    messages=messages,
    temperature=0.7,
)

print(f"User: {messages[0]['content']}")
print(f"Assistant: {response.choices[0].message.content}")

# Continue conversation
messages.append({"role": "assistant", "content": response.choices[0].message.content})
messages.append({"role": "user", "content": "Can you give an example?"})

response = client.chat.completions.create(
    model="Meta-Llama-3.1-8B-Instruct",
    messages=messages,
    temperature=0.7,
)

print(f"\nUser: {messages[-1]['content']}")
print(f"Assistant: {response.choices[0].message.content}")
print()

print("=" * 80)
print("Telemetry Data Collected:")
print("=" * 80)
print(
    """
For each API call, the following data is automatically collected:

TRACES (Spans):
- Span name: sambanova.chat.completion
- Attributes:
  - gen_ai.system: "sambanova"
  - gen_ai.request.model: e.g., "Llama-4-Maverick-17B-128E-Instruct"
  - gen_ai.operation.name: "chat"
  - gen_ai.request.temperature, top_p, max_tokens
  - gen_ai.usage.prompt_tokens, completion_tokens, total_tokens
  - gen_ai.response.id, model, finish_reasons
  - gen_ai.cost.amount (estimated in USD)

METRICS:
- genai.requests: Request count by model and provider
- genai.tokens: Token usage (prompt/completion)
- genai.latency: Request duration histogram
- genai.cost: Estimated costs in USD

View these metrics in your observability platform (Grafana, Jaeger, etc.)
"""
)

print("=" * 80)
print("SambaNova Pricing (per 1M tokens):")
print("=" * 80)
print(
    """
- Llama 4 Maverick 17B:  $0.10 / $0.20  (input/output)
- Llama 3.1 8B:          $0.10 / $0.20  (input/output)
- Llama 3.1 70B:         $0.60 / $1.20  (input/output)
- Llama 3.1 405B:        $5.00 / $10.00 (input/output)

Cost tracking is automatically calculated and included in spans!
"""
)

print("=" * 80)
print("Example complete! Check your OTLP collector/Grafana for traces and metrics.")
print("=" * 80 + "\n")
