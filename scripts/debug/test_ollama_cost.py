"""Debug script to test Ollama cost tracking.

This script helps identify why cost attributes are not appearing on Ollama spans.
"""

import json
import logging
import os

# Enable detailed logging
logging.basicConfig(level=logging.INFO)

print("=" * 80)
print("Ollama Cost Tracking Diagnostic")
print("=" * 80)

# Step 1: Initialize instrumentation
print("\n1. Initializing genai-otel instrumentation...")
import genai_otel

genai_otel.instrument()

# Step 2: Check if Ollama is installed
print("\n2. Checking if Ollama is installed...")
try:
    import ollama

    print(
        f"   [OK] Ollama library version: {ollama.__version__ if hasattr(ollama, '__version__') else 'unknown'}"
    )
except ImportError:
    print("   [ERROR] Ollama library not installed. Install with: pip install ollama")
    exit(1)

# Step 3: Test Ollama call
print("\n3. Testing Ollama generate call...")
print("   NOTE: This requires Ollama to be running locally with model 'smollm2:360m'")
print("   Start Ollama with: ollama run smollm2:360m")

try:
    # Make a test call to Ollama
    response = ollama.generate(model="smollm2:360m", prompt="Say hello in 3 words")

    print(f"\n   [OK] Ollama call successful!")
    print(
        f"\n   Raw response keys: {list(response.keys() if isinstance(response, dict) else dir(response))}"
    )

    # Check token counts in response
    if isinstance(response, dict):
        prompt_eval = response.get("prompt_eval_count", "MISSING")
        eval_count = response.get("eval_count", "MISSING")
        print(f"   prompt_eval_count: {prompt_eval}")
        print(f"   eval_count: {eval_count}")
    else:
        prompt_eval = getattr(response, "prompt_eval_count", "MISSING")
        eval_count = getattr(response, "eval_count", "MISSING")
        print(f"   prompt_eval_count: {prompt_eval}")
        print(f"   eval_count: {eval_count}")

    # Pretty print the response
    print(f"\n   Full response:")
    if isinstance(response, dict):
        print(f"   {json.dumps(response, indent=2)}")
    else:
        for key in dir(response):
            if not key.startswith("_"):
                print(f"   {key}: {getattr(response, key, None)}")

except Exception as e:
    print(f"   [ERROR] Ollama call failed: {e}")
    print(f"\n   Make sure Ollama is running and model 'smollm2:360m' is available")
    print(f"   Run: ollama pull smollm2:360m")
    exit(1)

print("\n" + "=" * 80)
print("Diagnostic complete!")
print("=" * 80)

print("\nExpected span attributes for cost tracking:")
print("  - gen_ai.request.model: <model_name>")
print("  - gen_ai.usage.prompt_tokens: <count>")
print("  - gen_ai.usage.completion_tokens: <count>")
print("  - gen_ai.usage.total_tokens: <count>")
print("\nIf prompt_tokens or completion_tokens are missing from your trace:")
print("  1. Check if Ollama response includes 'prompt_eval_count' and 'eval_count'")
print("  2. Check if these values are > 0")
print("  3. Check if token_counter metric is initialized in the instrumentor")
