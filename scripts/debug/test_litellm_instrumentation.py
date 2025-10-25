"""Diagnostic script to test LiteLLM instrumentation.

This script helps debug why litellm child spans may be missing in smolagents traces.
"""

import logging
import os

# Set logging to see what's happening
logging.basicConfig(level=logging.INFO)

print("=" * 80)
print("LiteLLM Instrumentation Diagnostic")
print("=" * 80)

# Step 1: Check if OpenInference packages are installed
print("\n1. Checking OpenInference package versions...")
try:
    import openinference.instrumentation.litellm

    print(
        f"   [OK] openinference-instrumentation-litellm: {openinference.instrumentation.litellm.__version__}"
    )
except ImportError as e:
    print(f"   [ERROR] openinference-instrumentation-litellm not installed: {e}")
    exit(1)

try:
    import openinference.instrumentation.smolagents

    print(
        f"   [OK] openinference-instrumentation-smolagents: {openinference.instrumentation.smolagents.__version__}"
    )
except ImportError as e:
    print(f"   [ERROR] openinference-instrumentation-smolagents not installed: {e}")

# Step 2: Check if litellm is installed
print("\n2. Checking if litellm is installed...")
try:
    import litellm

    print(f"   [OK] litellm version: {litellm.__version__}")
except ImportError:
    print("   [ERROR] litellm not installed. Install with: pip install litellm")
    exit(1)

# Step 3: Check instrumentor initialization
print("\n3. Testing instrumentor initialization...")
import genai_otel
from genai_otel.config import OTelConfig

config = OTelConfig()
print(f"   Enabled instrumentors: {config.enabled_instrumentors}")

if "litellm" in config.enabled_instrumentors:
    print("   [OK] litellm instrumentor is in enabled list")
else:
    print("   [WARNING] litellm instrumentor is NOT in enabled list")
    print("   Add it with: GENAI_ENABLED_INSTRUMENTORS=smolagents,litellm")

if "smolagents" in config.enabled_instrumentors:
    print("   [OK] smolagents instrumentor is in enabled list")
else:
    print("   [WARNING] smolagents instrumentor is NOT in enabled list")

# Step 4: Initialize instrumentation
print("\n4. Initializing instrumentation...")
genai_otel.instrument()

# Step 5: Test litellm directly
print("\n5. Testing litellm directly (this will make an actual API call)...")
print("   NOTE: This requires a valid API key for the model provider")
print("   Set environment variable to test:")
print("   - OPENAI_API_KEY for OpenAI models")
print("   - Or other provider keys")

# Check if we should run the test
if os.getenv("OPENAI_API_KEY"):
    print("\n   Running test with OpenAI...")
    try:
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'test successful' in 2 words"}],
            max_tokens=10,
        )
        print(f"   [OK] LiteLLM call successful: {response.choices[0].message.content}")
        print("\n   Check your trace backend for a span with:")
        print("   - Span name: 'completion' or 'litellm.completion'")
        print("   - Attributes: llm.model_name, llm.token_count.prompt, etc.")
    except Exception as e:
        print(f"   [ERROR] LiteLLM call failed: {e}")
else:
    print("   [SKIPPED] No API key found. To test, set OPENAI_API_KEY or another provider key")

print("\n" + "=" * 80)
print("Diagnostic complete!")
print("=" * 80)
print("\nFor smolagents to create litellm child spans, ensure:")
print("1. smolagents agent is configured to use litellm as the model backend")
print("2. Use LiteLLMModel from smolagents, NOT InferenceClientModel")
print("3. Instrumentor order is: smolagents -> litellm -> mcp")
print("\nExample smolagents configuration with litellm:")
print("   from smolagents import CodeAgent, LiteLLMModel")
print("   model = LiteLLMModel(model_id='gpt-3.5-turbo')")
print("   agent = CodeAgent(model=model)")
