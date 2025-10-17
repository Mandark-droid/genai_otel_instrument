# Example usage for GenAI OpenTelemetry Auto-Instrumentation

import logging
import os
import sys

import genai_otel

# Configure logging for the example
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# --- Setup Auto-Instrumentation ---
# Option 1: Using environment variables (recommended for zero-code setup)
# export OTEL_SERVICE_NAME=my-llm-app
# export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
# export GENAI_ENABLE_COST_TRACKING=true
# export GENAI_ENABLE_GPU_METRICS=true
# export GENAI_FAIL_ON_ERROR=false

# Option 2: Programmatic setup (if env vars are not set or need overriding)
try:
    # This will load configuration from environment variables or use defaults.
    # If you want to override, you can pass arguments like:
    genai_otel.instrument(
        service_name="heal-geni-otel-instrument", endpoint="http://192.168.13.124:7318"
    )
    # genai_otel.instrument()
    print("GenAI OpenTelemetry instrumentation setup complete.")
except Exception as e:
    print(f"Failed to setup GenAI OpenTelemetry instrumentation: {e}")

# Import AI libraries after instrumentation setup
try:
    import openai
except ImportError:
    openai = None
    print("Warning: openai not installed. Skipping OpenAI examples.")

try:
    import anthropic
except ImportError:
    anthropic = None
    print("Warning: anthropic not installed. Skipping Anthropic examples.")

try:
    import google.generativeai as genai
except ImportError:
    genai = None
    print("Warning: google-generativeai not installed. Skipping Google AI examples.")

try:
    import psycopg2
except ImportError:
    psycopg2 = None
    print("Warning: psycopg2 not installed. Skipping database examples.")

try:
    import redis
except ImportError:
    redis = None
    print("Warning: redis not installed. Skipping Redis examples.")

try:
    import pinecone
except ImportError:
    pinecone = None
    print("Warning: pinecone not installed. Skipping Pinecone examples.")

try:
    from kafka import KafkaProducer
except ImportError:
    KafkaProducer = None
    print("Warning: kafka-python not installed. Skipping Kafka examples.")


# --- Example 1: OpenAI ---
if openai:
    try:
        print("\n--- Testing OpenAI ---")
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello! What is OpenTelemetry?"}],
            max_tokens=100,
        )
        print(f"OpenAI Response (first 50 chars): {response.choices[0].message.content[:50]}...")
        print(f"OpenAI Usage: {response.usage}")
    except Exception as e:
        print(f"Error testing OpenAI: {e}")


# --- Example 2: Anthropic ---
if anthropic:
    try:
        print("\n--- Testing Anthropic ---")
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Explain the concept of distributed tracing."}],
        )
        # Handle both TextBlock and string content
        content_text = (
            message.content[0].text
            if hasattr(message.content[0], "text")
            else str(message.content[0])
        )
        print(f"Anthropic Response (first 50 chars): {content_text[:50]}...")
        print(f"Anthropic Usage: {message.usage}")
    except Exception as e:
        print(f"Error testing Anthropic: {e}")


# --- Example 3: Google AI (Gemini) ---
if genai:
    try:
        print("\n--- Testing Google AI (Gemini) ---")
        # Ensure you have GOOGLE_API_KEY set in your environment or provide it here
        # genai.configure(api_key="YOUR_API_KEY")
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content("What are the benefits of observability?")
        print(f"Gemini Response (first 50 chars): {response.text[:50]}...")
        print(f"Gemini Usage: {response.usage_metadata}")
    except Exception as e:
        print(f"Error testing Google AI: {e}")


# --- Example 4: Database, Cache, Vector DB, and Message Queue ---
try:
    print("\n--- Testing Infrastructure Instrumentation ---")

    # Database calls are auto-instrumented if libraries are installed (e.g., psycopg2)
    if psycopg2:
        try:
            print("Testing database instrumentation (psycopg2)...")
            # Replace with actual connection details if you have a DB setup
            # conn = psycopg2.connect("dbname=mydb user=user password=password host=localhost")
            # cursor = conn.cursor()
            # cursor.execute("SELECT 1")
            # print("psycopg2 connection and query successful.")
            print("psycopg2 instrumentation ready (requires actual DB connection to verify).")
        except Exception as e:
            print(f"Could not simulate psycopg2 connection: {e}")

    # Redis calls are auto-instrumented if library is installed
    if redis:
        try:
            print("Testing redis instrumentation...")
            r = redis.Redis(host="localhost", port=6379, decode_responses=True)
            r.ping()
            r.set("example_key", "example_value")
            print(f"Redis set: {r.get('example_key')}")
        except redis.exceptions.ConnectionError as e:
            print(f"Redis connection error: {e}. Ensure Redis is running on localhost:6379.")
        except Exception as e:
            print(f"Error testing redis: {e}")

    # Vector DB calls are auto-instrumented if libraries are installed (e.g., pinecone)
    if pinecone:
        try:
            print("Testing pinecone instrumentation...")
            # Replace with actual Pinecone credentials and index name
            # pinecone.init(api_key="YOUR_PINECONE_API_KEY", environment="YOUR_PINECONE_ENV")
            # index = pinecone.Index("my-index")
            # index.query(vector=[0.1]*8, top_k=3)
            print("Pinecone instrumentation ready (requires Pinecone setup to verify).")
        except Exception as e:
            print(f"Error testing Pinecone: {e}")

    # Kafka calls are auto-instrumented if library is installed
    if KafkaProducer:
        try:
            print("Testing kafka instrumentation...")
            # producer = KafkaProducer(bootstrap_servers='localhost:9092')
            # producer.send('my-topic', b'message')
            print("Kafka instrumentation ready (requires Kafka setup to verify).")
        except Exception as e:
            print(f"Error testing Kafka: {e}")

except Exception as e:
    print(f"Error during infrastructure instrumentation example: {e}")


# --- Custom Configuration Example ---
try:
    print("\n--- Testing Custom Configuration ---")
    # This demonstrates how to override defaults or set specific configurations programmatically.
    # Ensure GENAI_FAIL_ON_ERROR is set to false in env or here if you want the script to continue on errors.
    # genai_otel.instrument(
    #     service_name="my-custom-example-app",
    #     endpoint="http://your-otel-collector:4318",
    #     enable_gpu_metrics=False,
    #     enable_cost_tracking=False,
    #     enable_mcp_instrumentation=False,
    #     fail_on_error=False
    # )
    print("Custom configuration example shown in comments. Modify as needed.")
except Exception as e:
    print(f"Error during custom configuration example: {e}")

print("\n--- Example Usage Finished ---")
print("\nNote: To test with actual API calls, ensure you have:")
print(
    "  - API keys set in environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY)"
)
print("  - Required services running (Redis, PostgreSQL, Kafka, etc.)")
print("  - An OpenTelemetry collector endpoint configured (OTEL_EXPORTER_OTLP_ENDPOINT)")

# Wait for telemetry export before exit
# The BatchSpanProcessor batches spans and exports them periodically.
# We need to wait a few seconds to ensure pending spans are flushed.
import time

print("\n[INFO] Waiting 5 seconds for telemetry export...")
time.sleep(5)
print("[OK] Telemetry export complete. Check your collector for traces and metrics.")
