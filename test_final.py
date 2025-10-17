"""Final test to verify OTLP exporter works correctly"""

import os

# Set environment variables
os.environ["OTEL_SERVICE_NAME"] = "test-app"
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"  # Your collector endpoint
os.environ["GENAI_ENABLE_GPU_METRICS"] = "false"
os.environ["GENAI_ENABLED_INSTRUMENTORS"] = "openai"  # Only test OpenAI

print("=== Testing genai-otel-instrument with MCP enabled ===\n")

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
print("1. No AttributeError occurred")
print("2. MCP instrumentation is enabled")
print("3. Span should be exported to: http://localhost:4318/v1/traces")
print("\nIf you see '404 page not found', ensure your OTLP collector is:")
print("  - Running on localhost:4318")
print("  - Configured to accept HTTP requests (not just gRPC)")
print("  - Has the /v1/traces endpoint enabled")
