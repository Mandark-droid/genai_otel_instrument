# Example usage for GenAI OpenTelemetry Auto-Instrumentation

import os
import sys
import logging
import openai
import anthropic
import google.generativeai as genai
import psycopg2
import redis
import pinecone
from kafka import KafkaProducer
import genai_otel

# LangChain imports (some might be unused in this specific example)
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.tools import DuckDuckGoSearchRun

# Configure logging for the example
# Set a higher level for the example to see more output if needed
# You can also set GENAI_LOG_LEVEL environment variable
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Setup Auto-Instrumentation ---
# Option 1: Using environment variables (recommended for zero-code setup)
# export OTEL_SERVICE_NAME=my-llm-app
# export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
# export GENAI_ENABLE_COST_TRACKING=true
# export GENAI_ENABLE_GPU_METRICS=true
# export GENAI_FAIL_ON_ERROR=false

# Option 2: Programmatic setup (if env vars are not set or need overriding)
# Ensure you have set the necessary environment variables or provide them here.
# For this example, we assume some defaults or env vars are set.
try:
    # This will load configuration from environment variables or use defaults.
    # If you want to override, you can pass arguments like:
    # genai_otel.instrument(service_name="my-example-app", endpoint="http://localhost:4318")
    genai_otel.instrument()
    print("GenAI OpenTelemetry instrumentation setup complete.")
except Exception as e:
    print(f"Failed to setup GenAI OpenTelemetry instrumentation: {e}")
    # Depending on GENAI_FAIL_ON_ERROR, this might stop the script.
    # For this example, we'll just print and continue if possible.


# --- Example 1: OpenAI ---
try:
    print("\n--- Testing OpenAI ---")
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello! What is OpenTelemetry?"}],
        max_tokens=100
    )
    print(f"OpenAI Response (first 50 chars): {response.choices[0].message.content[:50]}...")
    print(f"OpenAI Usage: {response.usage}")
except Exception as e:
    print(f"Error testing OpenAI: {e}")


# --- Example 2: Anthropic ---
try:
    print("\n--- Testing Anthropic ---")
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": "Explain the concept of distributed tracing."
            }
        ]
    )
    print(f"Anthropic Response (first 50 chars): {message.content[:50]}...")
    print(f"Anthropic Usage: {message.usage}")
except Exception as e:
    print(f"Error testing Anthropic: {e}")


# --- Example 3: Google AI (Gemini) ---
try:
    print("\n--- Testing Google AI (Gemini) ---")
    # Ensure you have GOOGLE_API_KEY set in your environment or provide it here
    # genai.configure(api_key="YOUR_API_KEY")
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("What are the benefits of observability?")
    print(f"Gemini Response (first 50 chars): {response.text[:50]}...")
    print(f"Gemini Usage: {response.usage_metadata}")
except Exception as e:
    print(f"Error testing Google AI: {e}")


# --- Example 4: LangChain with MCP Tools ---
try:
    print("\n--- Testing LangChain with MCP Tools ---")
    # Note: For LangChain instrumentation to work fully, you might need to ensure
    # the underlying libraries (like OpenAI, psycopg2, redis, pinecone, kafka) are installed and configured.

    # Mocking a LangChain agent and tool for demonstration if actual setup is complex
    # In a real scenario, you would initialize these properly.

    # Database calls are auto-instrumented if libraries are installed (e.g., psycopg2)
    try:
        print("Testing database instrumentation (psycopg2)...")
        # Replace with actual connection details if you have a DB setup
        # conn = psycopg2.connect("dbname=mydb user=user password=password host=localhost")
        # cursor = conn.cursor()
        # cursor.execute("SELECT 1")
        # print("psycopg2 connection and query simulated.")
        print("psycopg2 instrumentation simulated (requires actual DB connection to verify).")
    except Exception as e:
        print(f"Could not simulate psycopg2 connection: {e}. Ensure psycopg2 is installed and DB is available.")

    # Redis calls are auto-instrumented if library is installed
    try:
        print("Testing redis instrumentation...")
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping() # Check connection
        r.set('example_key', 'example_value')
        print(f"Redis set: {r.get('example_key')}")
    except redis.exceptions.ConnectionError as e:
        print(f"Redis connection error: {e}. Ensure Redis is running on localhost:6379.")
    except Exception as e:
        print(f"Error testing redis: {e}")

    # Vector DB calls are auto-instrumented if libraries are installed (e.g., pinecone)
    try:
        print("Testing pinecone instrumentation...")
        # Replace "YOUR_KEY" and "my-index" with actual Pinecone credentials and index name
        # pinecone.init(api_key="YOUR_PINECONE_API_KEY", environment="YOUR_PINECONE_ENV")
        # index = pinecone.Index("my-index")
        # index.query(vector=[0.1]*8, top_k=3)
        print("Pinecone instrumentation simulated (requires Pinecone setup to verify).")
    except Exception as e:
        print(f"Error testing Pinecone: {e}")

    # Kafka calls are auto-instrumented if library is installed
try:
    print("Testing kafka instrumentation...")
    # producer = KafkaProducer(bootstrap_servers='localhost:9092')
    # producer.send('my-topic', b'message')
    print("Kafka instrumentation simulated (requires Kafka setup to verify).")
    pass
except Exception as e:
    print(f"Error during LangChain/MCP Tools example: {e}")


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
    #     fail_on_error=False # Explicitly set fail_on_error
    # )
    print("Custom configuration example shown in comments. Modify as needed.")
except Exception as e:
    print(f"Error during custom configuration example: {e}")

print("\n--- Example Usage Finished ---")