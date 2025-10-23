import logging
import os
import sys
import time
from unittest.mock import patch

# Set OTEL_EXPORTER_OTLP_ENDPOINT to an empty string before any genai_otel imports
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = ""
# Remove noop providers to allow console exporters to function
if "OTEL_TRACER_PROVIDER" in os.environ:
    del os.environ["OTEL_TRACER_PROVIDER"]
if "OTEL_METRIC_READER" in os.environ:
    del os.environ["OTEL_METRIC_READER"]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script to verify genai_otel instrumentation without external dependencies
"""


# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# --- Test Configuration ---
os.environ.setdefault("OTEL_SERVICE_NAME", "genai-test-app")
os.environ.setdefault("GENAI_ENABLE_COST_TRACKING", "false")
os.environ.setdefault("GENAI_ENABLE_GPU_METRICS", "false")
os.environ.setdefault("GENAI_FAIL_ON_ERROR", "false")


def print_header(title):
    print("\n" + "=" * 70)
    print(f"{title}")
    print("=" * 70)


def print_status(message, success=True):
    symbol = "✓" if success else "✗"
    print(f"{symbol} {message}")


def main():
    print_header("GenAI OpenTelemetry Instrumentation - Simple Test")
    import genai_otel
    from genai_otel.config import OTelConfig

    # 1. Test Initialization
    print("\n1. Initializing instrumentation...")
    try:
        genai_otel.instrument()
        print_status("Instrumentation initialized successfully")
    except Exception as e:
        print_status(f"Failed to initialize instrumentation: {e}", success=False)
        sys.exit(1)

    # 2. Test OpenTelemetry Integration
    print("\n2. Testing OpenTelemetry integration...")
    from opentelemetry import metrics, trace

    try:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test.span") as span:
            span.set_attribute("test.attribute", "test_value")
        print_status("Tracing is working")
    except Exception as e:
        print_status(f"Tracing error: {e}", success=False)

    try:
        meter = metrics.get_meter(__name__)
        counter = meter.create_counter("test.counter")
        counter.add(1)
        print_status("Metrics are working")
    except Exception as e:
        print_status(f"Metrics error: {e}", success=False)

    # 3. Test Cost Calculation
    print("\n3. Testing cost calculation...")
    try:
        from genai_otel.cost_calculator import CostCalculator

        calculator = CostCalculator()
        usage = {"prompt_tokens": 1000, "completion_tokens": 500}
        cost = calculator.calculate_cost("gpt-4o", usage, "chat")
        assert cost > 0
        print_status(f"Cost calculation working. Test cost: ${cost:.6f}")
    except Exception as e:
        print_status(f"Cost calculation error: {e}", success=False)

    # 4. Test GPU Metrics
    print("\n4. Testing GPU metrics...")
    try:
        from genai_otel.gpu_metrics import NVML_AVAILABLE, GPUMetricsCollector

        if NVML_AVAILABLE:
            # This is a simplified check; we don't need a real meter for this test
            with patch("opentelemetry.metrics.get_meter") as mock_meter:
                # In the GPU test (replace the collector init):
                collector = GPUMetricsCollector(
                    meter=mock_meter, config=OTelConfig()  # Use a fresh instance with defaults
                )
                collector.start()
                assert collector.running
                print_status("GPU metrics collector started successfully")
                collector.stop()
        else:
            print_status("GPU metrics disabled (nvidia-ml-py not installed)", success=True)
    except Exception as e:
        print_status(f"GPU metrics error: {e}", success=False)

    # 5. Test MCP Instrumentation Status
    print("\n5. Testing MCP instrumentation status...")
    # Check if the instrumentors have been applied by checking for wrapper attributes
    checks = {
        "HTTP/API (requests)": "requests.sessions.Session.request",
        "HTTP/API (httpx)": "httpx.Client.request",
        "Database (SQLAlchemy)": "sqlalchemy.engine.Engine.connect",
        "Database (psycopg2)": "psycopg2.connect",
        "Redis": "redis.Redis.execute_command",
        "Kafka": "kafka.KafkaProducer.send",
        "Vector DB (Pinecone)": "pinecone.data.index.Index.query",
    }
    for component, attribute in checks.items():
        try:
            parts = attribute.split(".")
            module = __import__(parts[0])
            obj = module
            for part in parts[1:]:
                obj = getattr(obj, part)

            # Check if the object has been wrapped by OpenTelemetry
            if hasattr(obj, "__wrapped__"):
                print_status(f"{component}: Instrumented")
            else:
                print_status(f"{component}: Not instrumented", success=False)
        except (ImportError, AttributeError):
            print_status(f"{component}: Not installed or found")

    print_header("Test Summary")
    print_status("All tests completed!", success=True)
    print("\nNext steps:")
    print("  1. Set API keys (e.g., export OPENAI_API_KEY=your_key)")
    print("  2. Set OTLP endpoint to a running collector (e.g., Jaeger)")
    print("     export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318")
    print("  3. Run example_usage.py to test with real API calls.")


if __name__ == "__main__":
    main()
    print("\nDone!")
    print("\nNote: Any OTLP export errors at the end are expected if no collector is running.")
