"""Quick test to verify exporter configuration works"""

import logging
import os

logging.basicConfig(level=logging.INFO)

# Set environment variables
os.environ["OTEL_SERVICE_NAME"] = "test-app"
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"
os.environ["GENAI_ENABLE_GPU_METRICS"] = "false"
os.environ["GENAI_ENABLE_MCP_INSTRUMENTATION"] = "false"  # Disable MCP to isolate the issue

# Import and instrument
import genai_otel

try:
    genai_otel.instrument()
    print("✓ Instrumentation setup successful")
except Exception as e:
    print(f"✗ Instrumentation failed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# Try a simple traced operation
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("test-span") as span:
    span.set_attribute("test.attribute", "test-value")
    print("✓ Created test span successfully")

# Wait a bit for export
import time

time.sleep(2)

print("✓ Test completed - check for any export errors above")
