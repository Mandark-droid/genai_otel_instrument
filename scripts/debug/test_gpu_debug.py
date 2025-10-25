"""Test GPU metrics collection with debug output"""

import logging
import os
import time

# Set up logging to DEBUG level
logging.basicConfig(level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s")

import genai_otel
from genai_otel.config import OTelConfig

# Check config
config = OTelConfig(service_name="gpu-test")
print(f"enable_gpu_metrics = {config.enable_gpu_metrics}")
print(f"GENAI_ENABLE_GPU_METRICS env var = {os.getenv('GENAI_ENABLE_GPU_METRICS')}")

# Now instrument
genai_otel.instrument(service_name="gpu-test")

print("Waiting 5 seconds...")
time.sleep(5)
print("Done!")
