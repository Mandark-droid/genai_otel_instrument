"""Ollama Example with Automatic Server Metrics Collection

This example demonstrates automatic collection of Ollama server metrics including:
- VRAM usage per model (gen_ai.server.kv_cache.usage)
- Running requests counter (gen_ai.server.requests.running)
- Number of models loaded in memory

The metrics are collected automatically by polling Ollama's /api/ps endpoint.
"""

import time

import genai_otel

# Initialize instrumentation with Ollama server metrics enabled (default)
# You can configure via environment variables:
# - GENAI_ENABLE_OLLAMA_SERVER_METRICS=true (default)
# - OLLAMA_BASE_URL=http://localhost:11434 (default)
# - GENAI_OLLAMA_METRICS_INTERVAL=5.0 (default, in seconds)
# - GENAI_OLLAMA_MAX_VRAM_GB=24 (optional, for accurate % calculation)

genai_otel.instrument()

import ollama

print("=" * 70)
print("Ollama Server Metrics Demo")
print("=" * 70)
print()
print("This demo will:")
print("1. Make a request to Ollama (loads model into VRAM)")
print("2. Show automatic server metrics collection in the background")
print("3. Display VRAM usage and running model metrics")
print()
print("Starting in 2 seconds...")
time.sleep(2)

# Make a request - this will load the model into memory
print("\n[1] Making request to Ollama...")
try:
    response = ollama.generate(model="smollm2:360m", prompt="What is OpenTelemetry?")
    print(f"Response: {response['response'][:100]}...")
    print("[SUCCESS] Request completed")
except Exception as e:
    print(f"[ERROR] Failed to connect to Ollama: {e}")
    print("\nMake sure Ollama is running:")
    print("  1. Install Ollama: https://ollama.com/download")
    print("  2. Start Ollama: ollama serve")
    print("  3. Pull a model: ollama pull smollm2:360m")
    exit(1)

# Wait for metrics to be collected
print("\n[2] Waiting for server metrics to be collected...")
print("    (The poller runs every 5 seconds by default)")
time.sleep(6)

# Get server metrics to show what was collected
from genai_otel import get_server_metrics

server_metrics = get_server_metrics()

if server_metrics:
    print("\n[3] Server Metrics Collected:")
    print("-" * 70)
    print(f"  Running requests: {server_metrics._num_requests_running}")
    print(f"  Waiting requests: {server_metrics._num_requests_waiting}")
    print(f"  Max capacity: {server_metrics._num_requests_max}")
    print()
    print(f"  KV Cache Usage (per model):")
    if server_metrics._kv_cache_usage:
        for model, usage in server_metrics._kv_cache_usage.items():
            print(f"    - {model}: {usage:.2f}%")
    else:
        print("    (No models loaded or metrics not yet collected)")
    print("-" * 70)
else:
    print("[ERROR] Server metrics collector not initialized")

print()
print("=" * 70)
print("Metrics Export")
print("=" * 70)
print()
print("The following metrics are now available in your telemetry backend:")
print()
print("  gen_ai.server.kv_cache.usage{model='smollm2:360m'}")
print("    -> VRAM usage percentage for the loaded model")
print()
print("  gen_ai.server.requests.running")
print("    -> Number of currently executing requests")
print()
print("  gen_ai.server.requests.waiting")
print("    -> Number of requests waiting in queue")
print()
print("  gen_ai.server.requests.max")
print("    -> Maximum concurrent request capacity")
print()
print("=" * 70)
print("Configuration")
print("=" * 70)
print()
print("You can customize the poller via environment variables:")
print()
print("  GENAI_ENABLE_OLLAMA_SERVER_METRICS=true")
print("    -> Enable/disable automatic metrics (default: true)")
print()
print("  OLLAMA_BASE_URL=http://localhost:11434")
print("    -> Ollama server URL (default: http://localhost:11434)")
print()
print("  GENAI_OLLAMA_METRICS_INTERVAL=5.0")
print("    -> Polling interval in seconds (default: 5.0)")
print()
print("  GENAI_OLLAMA_MAX_VRAM_GB=24")
print("    -> Your GPU's VRAM in GB for accurate % calculation")
print("    -> Example: 24 for RTX 4090, 16 for RTX 4080, etc.")
print()
print("=" * 70)
