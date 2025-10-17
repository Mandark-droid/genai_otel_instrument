"""Debug script to test why example_usage.py doesn't send traces"""

import logging
import os

# Set up logging FIRST to see all messages
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

print("=== Testing example_usage.py configuration ===\n")

# Mirror example_usage.py: NO environment variables set explicitly
# This relies on defaults from config.py:
# - OTEL_SERVICE_NAME defaults to "genai-app"
# - OTEL_EXPORTER_OTLP_ENDPOINT defaults to "http://localhost:4318"
# - GENAI_ENABLE_GPU_METRICS defaults to "true"
# - All instrumentors enabled by default
# - MCP instrumentation enabled by default

print("[INFO] Using default configuration (no env vars set)")
print("[INFO] Expected defaults:")
print("  - service_name: 'genai-app'")
print("  - endpoint: 'http://localhost:4318'")
print("  - enable_gpu_metrics: true")
print("  - enabled_instrumentors: all (openai, anthropic, google, etc.)")
print("  - enable_mcp_instrumentation: true\n")

import genai_otel

genai_otel.instrument()
print("[OK] Instrumentation complete - no AttributeError!\n")

# Create a test span
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("test-operation") as span:
    span.set_attribute("test.key", "test.value")
    span.set_attribute("test.number", 42)
    print("[OK] Created test span with attributes\n")

# Wait for export
import time

print("[INFO] Waiting 5 seconds for span export...")
time.sleep(5)

print("\n=== Test Summary ===")
print("1. Configuration used defaults (like example_usage.py)")
print("2. MCP instrumentation is enabled")
print("3. All instrumentors are enabled by default")
print("4. GPU metrics collection is enabled by default")
print("5. Span should be exported to: http://localhost:4318/v1/traces")
print("\nCheck your collector logs to see if traces arrived.")
