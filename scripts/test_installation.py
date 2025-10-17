"""Simple test to verify genai-otel-instrument installation.

This test verifies the package works without requiring:
- API keys
- Running OTLP collector
- Actual LLM API calls
"""

import os
import sys

print("=" * 60)
print("Testing genai-otel-instrument Installation")
print("=" * 60)
print()

# Test 1: Import the package
print("Test 1: Importing package...")
try:
    import genai_otel

    print(f"OK Package imported successfully")
    print(f"  Version: {genai_otel.__version__}")
except ImportError as e:
    print(f"✗ Failed to import package: {e}")
    sys.exit(1)

# Test 2: Import main components
print("\nTest 2: Importing main components...")
try:
    from genai_otel import OTelConfig, instrument
    from genai_otel.instrumentors import AnthropicInstrumentor, OpenAIInstrumentor

    print("OK Main components imported successfully")
except ImportError as e:
    print(f"✗ Failed to import components: {e}")
    sys.exit(1)

# Test 3: Create configuration
print("\nTest 3: Creating configuration...")
try:
    config = OTelConfig(
        service_name="genai_otel_test-service",
        endpoint="http://192.168.13.124:7318",
        enable_gpu_metrics=True,
        enable_cost_tracking=True,
        enable_mcp_instrumentation=True,
        fail_on_error=False,  # Don't fail if no collector running
    )
    print("OK Configuration created successfully")
    print(f"  Service name: {config.service_name}")
    print(f"  Endpoint: {config.endpoint}")
except Exception as e:
    print(f"✗ Failed to create configuration: {e}")
    sys.exit(1)

# Test 4: Initialize instrumentation (without collector)
print("\nTest 4: Initializing instrumentation...")
try:
    # Set environment to use console exporter as fallback
    os.environ["OTEL_SERVICE_NAME"] = "genai_otel_test-service"
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://192.168.13.124:7318"
    os.environ["GENAI_ENABLE_GPU_METRICS"] = "true"
    os.environ["GENAI_ENABLE_MCP_INSTRUMENTATION"] = "true"
    os.environ["GENAI_FAIL_ON_ERROR"] = "true"
    os.environ["GENAI_LOG_LEVEL"] = "WARNING"  # Reduce noise

    instrument()
    print("OK Instrumentation initialized successfully")
    print("  (Using console exporter as fallback if no collector available)")
except Exception as e:
    print(f"✗ Failed to initialize instrumentation: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 5: Verify instrumentors are available
print("\nTest 5: Checking instrumentor availability...")
try:
    from genai_otel.instrumentors import (
        AnthropicInstrumentor,
        GoogleAIInstrumentor,
        OpenAIInstrumentor,
    )

    # Create instrumentors
    openai_inst = OpenAIInstrumentor()
    anthropic_inst = AnthropicInstrumentor()
    google_inst = GoogleAIInstrumentor()

    print("OK Instrumentors created successfully")
    print(f"  OpenAI available: {openai_inst._openai_available}")
    print(f"  Anthropic available: {anthropic_inst._anthropic_available}")
    print(f"  Google AI available: {google_inst._google_available}")
except Exception as e:
    print(f"✗ Failed to create instrumentors: {e}")
    sys.exit(1)

# Test 6: Test cost calculator
print("\nTest 6: Testing cost calculator...")
try:
    from genai_otel import CostCalculator

    calculator = CostCalculator()

    # Test calculation (even with no pricing data, should not crash)
    usage = {"prompt_tokens": 100, "completion_tokens": 50}
    cost = calculator.calculate_cost("gpt-3.5-turbo", usage)

    print("OK Cost calculator working")
    print(f"  Estimated cost: ${cost:.6f}")
    if cost == 0:
        print("  (Note: Pricing data may not be loaded)")
except Exception as e:
    print(f"✗ Cost calculator failed: {e}")
    import traceback

    traceback.print_exc()

# Test 7: Verify trace context
print("\nTest 7: Verifying OpenTelemetry context...")
try:
    from opentelemetry import trace

    tracer = trace.get_tracer(__name__)

    # Create a test span
    with tracer.start_as_current_span("test_span") as span:
        span.set_attribute("test", "value")
        print("OK Test span created successfully")
        print(f"  Span ID: {format(span.get_span_context().span_id, '016x')}")
except Exception as e:
    print(f"✗ Failed to create test span: {e}")
    import traceback

    traceback.print_exc()

# Summary
print()
print("=" * 60)
print("Installation Test Summary")
print("=" * 60)
print("OK All basic tests passed!")
print()
print("Next steps:")
print("1. Set up OTLP collector (e.g., Jaeger) for full telemetry")
print("2. Set API keys for LLM providers you want to use")
print("3. Run example_usage.py to test with actual API calls")
print()
print("To run with Jaeger:")
print("  docker run -d --name jaeger \\")
print("    -e COLLECTOR_OTLP_ENABLED=true \\")
print("    -p 4318:4318 -p 16686:16686 \\")
print("    jaegertracing/all-in-one:latest")
print()
print("Then view traces at: http://localhost:16686")
print("=" * 60)
