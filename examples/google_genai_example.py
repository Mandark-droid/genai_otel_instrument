"""Google GenAI SDK Instrumentation Example.

This example demonstrates automatic OpenTelemetry instrumentation for
Google's Gemini models using BOTH the legacy and new unified SDKs.

The instrumentor automatically detects which SDK is installed:
- New SDK (recommended): google-genai (GA since May 2025)
- Legacy SDK: google-generativeai (deprecated Nov 30, 2025)

Requirements:
    # New SDK (recommended):
    pip install genai-otel-instrument
    pip install google-genai

    # OR Legacy SDK (deprecated):
    pip install google-generativeai

    export GOOGLE_API_KEY=your_api_key
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
"""

import os

import genai_otel

# Initialize instrumentation - Google GenAI is enabled automatically
genai_otel.instrument(
    service_name="google-genai-example",
    endpoint="http://localhost:4318",
)

print("\n" + "=" * 80)
print("Google GenAI SDK OpenTelemetry Instrumentation Example")
print("=" * 80 + "\n")

# Check for API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: GOOGLE_API_KEY environment variable not set")
    print("Get your API key from: https://aistudio.google.com/apikey")
    exit(1)

# Try new SDK first (recommended)
try:
    from google import genai

    print("Using NEW google-genai SDK (recommended)")
    print("-" * 80)
    using_new_sdk = True
except ImportError:
    # Fall back to legacy SDK
    try:
        import google.generativeai as genai

        print("Using LEGACY google-generativeai SDK (deprecated Nov 30, 2025)")
        print("Consider migrating to google-genai for continued support")
        print("-" * 80)
        using_new_sdk = False
    except ImportError:
        print("ERROR: No Google GenAI SDK installed. Install with:")
        print("  New SDK: pip install google-genai")
        print("  OR")
        print("  Legacy SDK: pip install google-generativeai")
        exit(1)

if using_new_sdk:
    # =========================================================================
    # NEW SDK EXAMPLES (google-genai)
    # =========================================================================

    # Initialize client
    client = genai.Client(api_key=api_key)

    print("\n1. Simple Chat Completion (Gemini 2.0 Flash)...")
    print("-" * 80)

    # Simple chat completion with Gemini 2.0
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents="Explain quantum entanglement in one sentence.",
    )

    print(f"Model: gemini-2.0-flash-exp")
    print(f"Response: {response.text}")
    print()

    print("2. Chat Completion with Configuration...")
    print("-" * 80)

    # With generation config
    response = client.models.generate_content(
        model="gemini-1.5-pro",
        contents="Write a haiku about artificial intelligence.",
        config={
            "temperature": 0.9,
            "top_p": 0.95,
            "max_output_tokens": 100,
        },
    )

    print(f"Model: gemini-1.5-pro")
    print(f"Response: {response.text}")
    if hasattr(response, "usage_metadata"):
        print(f"Tokens: {response.usage_metadata.total_token_count}")
    print()

    print("3. Multi-turn Conversation...")
    print("-" * 80)

    # Multi-turn conversation
    messages = [
        {"role": "user", "parts": [{"text": "What is machine learning?"}]},
    ]

    response = client.models.generate_content(model="gemini-2.0-flash-exp", contents=messages)

    print(f"User: What is machine learning?")
    print(f"Assistant: {response.text}")

    # Continue conversation
    messages.append({"role": "model", "parts": [{"text": response.text}]})
    messages.append({"role": "user", "parts": [{"text": "Can you give a real-world example?"}]})

    response = client.models.generate_content(model="gemini-2.0-flash-exp", contents=messages)

    print(f"\nUser: Can you give a real-world example?")
    print(f"Assistant: {response.text}")
    print()

else:
    # =========================================================================
    # LEGACY SDK EXAMPLES (google.generativeai)
    # =========================================================================

    # Configure API key
    genai.configure(api_key=api_key)

    print("\n1. Simple Chat Completion (Gemini Pro)...")
    print("-" * 80)

    # Create model instance
    model = genai.GenerativeModel("gemini-pro")

    # Generate content
    response = model.generate_content("Explain quantum entanglement in one sentence.")

    print(f"Model: gemini-pro")
    print(f"Response: {response.text}")
    print()

    print("2. Chat Completion with Configuration...")
    print("-" * 80)

    # With generation config
    from google.generativeai.types import GenerationConfig

    generation_config = GenerationConfig(
        temperature=0.9,
        top_p=0.95,
        max_output_tokens=100,
    )

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(
        "Write a haiku about artificial intelligence.",
        generation_config=generation_config,
    )

    print(f"Model: gemini-pro")
    print(f"Response: {response.text}")
    if hasattr(response, "usage_metadata"):
        print(f"Tokens: {response.usage_metadata.total_token_count}")
    print()

    print("3. Multi-turn Conversation...")
    print("-" * 80)

    # Start a chat
    model = genai.GenerativeModel("gemini-pro")
    chat = model.start_chat(history=[])

    # First message
    response = chat.send_message("What is machine learning?")
    print(f"User: What is machine learning?")
    print(f"Assistant: {response.text}")

    # Second message
    response = chat.send_message("Can you give a real-world example?")
    print(f"\nUser: Can you give a real-world example?")
    print(f"Assistant: {response.text}")
    print()

    print("4. With Safety Settings...")
    print("-" * 80)

    from google.generativeai.types import HarmBlockThreshold, HarmCategory

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(
        "Tell me about renewable energy.",
        safety_settings=safety_settings,
    )

    print(f"Model: gemini-pro")
    print(f"Response: {response.text}")
    print()

print("=" * 80)
print("Telemetry Data Collected:")
print("=" * 80)
print(
    """
For each API call, the following data is automatically collected:

TRACES (Spans):
- Span name:
  * New SDK: google.genai.generate_content or google.genai.models.generate_content
  * Legacy SDK: google.generativeai.generate_content
- Attributes:
  - gen_ai.system: "google"
  - gen_ai.operation.name: "chat"
  - gen_ai.request.model: Model name (e.g., "gemini-2.0-flash-exp")
  - gen_ai.request.temperature: Temperature parameter
  - gen_ai.request.top_p: Top-p parameter
  - gen_ai.request.max_tokens: Max output tokens
  - gen_ai.request.safety_settings_count: Number of safety settings (legacy SDK)
  - gen_ai.usage.prompt_tokens: Prompt token count
  - gen_ai.usage.completion_tokens: Completion token count
  - gen_ai.usage.total_tokens: Total token count
  - gen_ai.response.model: Actual model used in response
  - gen_ai.response.finish_reasons: Finish reasons array
  - gen_ai.safety.*: Safety ratings by category
  - gen_ai.cost.amount: Estimated cost in USD

METRICS:
- genai.requests: Request count by model
- genai.tokens: Token usage (prompt/completion)
- genai.latency: Request duration histogram
- genai.cost: Estimated costs in USD

View these metrics in your observability platform (Grafana, Jaeger, etc.)
"""
)

print("=" * 80)
print("Google GenAI SDK Migration:")
print("=" * 80)
print(
    """
SDK Comparison:

LEGACY SDK (google-generativeai):
- Package: google-generativeai
- Status: Deprecated (support ends Nov 30, 2025)
- Import: import google.generativeai as genai
- Models: GenerativeModel class
- Configuration: genai.configure(api_key=...)

NEW SDK (google-genai):
- Package: google-genai
- Status: GA (released May 2025, latest Nov 12, 2025)
- Import: from google import genai
- Models: Client-based approach
- Configuration: client = genai.Client(api_key=...)
- Unified: Works with both Gemini Developer API and Vertex AI

Migration Path:
1. Install new SDK: pip install google-genai
2. Update imports: from google import genai
3. Use Client: client = genai.Client(api_key=api_key)
4. Call methods: client.models.generate_content(...)

Benefits of New SDK:
- Unified interface for both APIs
- Latest Gemini 2.0 models
- Stable GA release
- Better performance and features
- Continued long-term support

The instrumentor supports BOTH SDKs automatically with zero code changes!
"""
)

print("=" * 80)
print("Gemini Models Available:")
print("=" * 80)
print(
    """
New SDK (Gemini 2.0 - Latest):
- gemini-2.0-flash-exp: Fast, efficient, experimental
- gemini-1.5-pro: Powerful, versatile, production-ready
- gemini-1.5-flash: Fast, cost-effective

Legacy SDK (Gemini 1.x):
- gemini-pro: General-purpose model
- gemini-pro-vision: Multi-modal (text + images)

Cost tracking is automatically calculated for all models!
"""
)

print("=" * 80)
print("Example complete! Check your OTLP collector/Grafana for traces and metrics.")
print("=" * 80 + "\n")
