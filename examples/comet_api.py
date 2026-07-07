"""CometAPI Instrumentation Example.

This example demonstrates automatic OpenTelemetry instrumentation for
CometAPI (https://www.cometapi.com), an all-in-one aggregator that exposes
500+ models (GPT, Claude, Gemini, DeepSeek, Qwen, and more) behind a single
API key.

CometAPI is compatible with both the Anthropic SDK (via the /v1/messages
endpoint) and the OpenAI SDK (via the /v1/chat/completions endpoint). The
CometAPI instrumentor detects clients pointed at api.cometapi.com and traces
calls made through either SDK.

Requirements:
    pip install genai-otel-instrument[cometapi]
    export COMETAPI_KEY=your_api_key
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
"""

import os

import genai_otel

# Initialize instrumentation - CometAPI is enabled by default
genai_otel.instrument(
    service_name="cometapi-example",
    # endpoint="http://localhost:4318",
)

print("\n" + "=" * 80)
print("CometAPI OpenTelemetry Instrumentation Example")
print("=" * 80 + "\n")

# Get your CometAPI key from https://www.cometapi.com/console/token
COMETAPI_KEY = os.environ.get("COMETAPI_KEY")
if not COMETAPI_KEY:
    print("ERROR: COMETAPI_KEY environment variable not set")
    print("Get your API key from: https://www.cometapi.com/console/token")
    exit(1)

BASE_URL = "https://api.cometapi.com"

print("1. Claude model via the Anthropic SDK...")
print("-" * 80)

try:
    import anthropic

    client = anthropic.Anthropic(
        base_url=BASE_URL,
        api_key=COMETAPI_KEY,
    )
    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Hello, Claude"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Can you describe LLMs to me?"},
        ],
    )
    for block in message.content:
        if getattr(block, "type", None) == "text":
            print(block.text)
    print(f"Tokens: input={message.usage.input_tokens}, " f"output={message.usage.output_tokens}")
except ImportError:
    print("Anthropic SDK not installed. Install with: pip install anthropic")

print()
print("2. GPT model via the OpenAI SDK...")
print("-" * 80)

try:
    from openai import OpenAI

    openai_client = OpenAI(
        base_url=f"{BASE_URL}/v1",
        api_key=COMETAPI_KEY,
    )
    response = openai_client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_tokens=100,
    )
    print(f"Response: {response.choices[0].message.content}")
    print(f"Tokens: {response.usage.total_tokens}")
except ImportError:
    print("OpenAI SDK not installed. Install with: pip install openai")

print()
print("=" * 80)
print("Telemetry Data Collected:")
print("=" * 80)
print(
    """
For each API call, the following data is automatically collected:

TRACES (Spans):
- Span name: cometapi.messages.create (Anthropic SDK)
             cometapi.chat.completion (OpenAI SDK)
- Attributes:
  - gen_ai.system: "cometapi"
  - gen_ai.request.model: e.g., "claude-sonnet-5"
  - gen_ai.operation.name: "chat"
  - gen_ai.request.temperature, top_p, max_tokens
  - gen_ai.usage.prompt_tokens, completion_tokens, total_tokens
  - gen_ai.response.id, model, finish_reasons
  - gen_ai.usage.cost.total (estimated in USD, by model name)

METRICS:
- genai.requests: Request count by model and provider
- genai.tokens: Token usage (prompt/completion)
- genai.latency: Request duration histogram
- genai.cost: Estimated costs in USD

View these metrics in your observability platform (Grafana, Jaeger, etc.)
"""
)

print("=" * 80)
print("Example complete! Check your OTLP collector/Grafana for traces and metrics.")
print("=" * 80 + "\n")
