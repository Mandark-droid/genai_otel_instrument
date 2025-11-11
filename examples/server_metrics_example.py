"""Server Metrics Example - KV Cache and Request Queue Tracking.

This example demonstrates how to use the ServerMetricsCollector for manual
instrumentation of server-side LLM serving metrics like KV cache usage and
request queue depth.

These metrics are designed for production LLM serving scenarios where you have
access to server-side statistics from frameworks like vLLM, TGI, or custom
serving infrastructure.

Requirements:
    pip install genai-otel-instrument
"""

import random
import time
from threading import Thread

import genai_otel

# Initialize instrumentation
genai_otel.instrument()

# Get the server metrics collector
server_metrics = genai_otel.get_server_metrics()

print("\n" + "=" * 80)
print("Server Metrics Collector - Production Pattern Example")
print("=" * 80 + "\n")

# Simulate server initialization
print("1. Initializing server configuration...")
server_metrics.set_requests_max(50)  # Max 50 concurrent requests
print("   - Max concurrent requests: 50\n")

# Simulate different models being loaded
models = ["gpt-4", "llama-2-70b", "mistral-7b"]
print("2. Loading models...")
for model in models:
    # Simulate initial KV cache state
    initial_cache = random.uniform(5.0, 15.0)  # nosec B311
    server_metrics.set_kv_cache_usage(model, initial_cache)
    print(f"   - {model}: KV cache at {initial_cache:.1f}%")
print()


def simulate_request_processing(request_id: int, model: str):
    """Simulate processing a single request."""
    # Increment running requests
    server_metrics.increment_requests_running()

    # Simulate KV cache usage increase during processing
    current_cache = random.uniform(20.0, 80.0)  # nosec B311
    server_metrics.set_kv_cache_usage(model, current_cache)

    # Simulate processing time
    time.sleep(random.uniform(0.1, 0.5))  # nosec B311

    # Decrement running requests when done
    server_metrics.decrement_requests_running()

    print(f"   Request {request_id} completed for {model} (cache: {current_cache:.1f}%)")


def simulate_request_queue():
    """Simulate requests coming in and being queued."""
    print("3. Simulating request traffic...\n")

    threads = []
    for i in range(10):
        # Random model selection
        model = random.choice(models)  # nosec B311

        # Simulate queue waiting
        if i > 3:  # After 3 concurrent, start queuing
            waiting = i - 3
            server_metrics.set_requests_waiting(waiting)

        # Start request processing
        thread = Thread(target=simulate_request_processing, args=(i + 1, model))
        thread.start()
        threads.append(thread)

        time.sleep(0.1)  # Stagger request arrivals

    # Wait for all requests to complete
    for thread in threads:
        thread.join()

    # Reset queue to 0
    server_metrics.set_requests_waiting(0)
    print("\n   All requests completed!\n")


# Run simulation
simulate_request_queue()

# Show final state
print("=" * 80)
print("Final Server State:")
print("=" * 80)
print("\nKV Cache Usage by Model:")
for model in models:
    # Note: In real implementation, you'd track this separately
    print(f"  - {model}: ~XX.X% (last recorded value)")

print("\nRequest Queue State:")
print("  - Running: 0")
print("  - Waiting: 0")
print("  - Max capacity: 50")

print("\n" + "=" * 80)
print("Metrics Available in Prometheus/OTLP:")
print("=" * 80)
print(
    """
# KV Cache Usage (Gauge per model)
gen_ai.server.kv_cache.usage{model="gpt-4"} XX.X
gen_ai.server.kv_cache.usage{model="llama-2-70b"} XX.X
gen_ai.server.kv_cache.usage{model="mistral-7b"} XX.X

# Request Queue Metrics (Gauges)
gen_ai.server.requests.running 0
gen_ai.server.requests.waiting 0
gen_ai.server.requests.max 50
"""
)

print("=" * 80)
print("Production Integration Pattern:")
print("=" * 80)
print(
    """
# For vLLM:
from vllm import AsyncLLMEngine
import genai_otel

genai_otel.instrument()
server_metrics = genai_otel.get_server_metrics()

async def update_metrics(engine: AsyncLLMEngine):
    while True:
        # Get KV cache stats from vLLM
        stats = await engine.get_cache_stats()
        for model, cache_usage in stats.items():
            server_metrics.set_kv_cache_usage(model, cache_usage)

        # Get request queue stats
        queue_stats = await engine.get_queue_stats()
        server_metrics.set_requests_running(queue_stats['running'])
        server_metrics.set_requests_waiting(queue_stats['waiting'])

        await asyncio.sleep(1)  # Update every second

# For TGI (Text Generation Inference):
# Similar pattern - poll TGI metrics endpoint and update server_metrics

# For custom serving:
# Integrate directly in your request handler
class RequestHandler:
    def handle_request(self, request):
        server_metrics.increment_requests_running()
        try:
            result = self.process(request)
            return result
        finally:
            server_metrics.decrement_requests_running()
"""
)

print("\nExample complete! Check your Prometheus/Grafana for metrics.")
print("=" * 80 + "\n")
