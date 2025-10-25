"""Test GPU metrics collection"""

import logging
import time

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

import genai_otel

genai_otel.instrument(service_name="gpu-test")

print("Waiting 20 seconds for GPU metrics to be collected...")
time.sleep(20)
print("Done!")
