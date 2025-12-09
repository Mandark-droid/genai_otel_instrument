"""
Test script to verify codecarbon CPU/RAM metrics are being collected and exported.
"""

import os
import time

# Configure OTEL
os.environ["OTEL_SERVICE_NAME"] = "codecarbon-test"
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://192.168.206.128:55681"
os.environ["GENAI_ENABLE_GPU_METRICS"] = "true"
os.environ["GENAI_ENABLE_COST_TRACKING"] = "false"
os.environ["GENAI_ENABLE_CO2_TRACKING"] = "true"
os.environ["GENAI_CO2_COUNTRY_ISO_CODE"] = "IND"  # Set to your country: IND, USA, GBR, etc.
os.environ["GENAI_CODECARBON_LOG_LEVEL"] = "info"  # Enable to see what's happening

# Import and initialize instrumentation
from genai_otel import instrument

print("Initializing instrumentation...")
instrument()

print("\nWaiting 30 seconds for codecarbon to collect metrics...")
print("Codecarbon collects metrics every GPU_COLLECTION_INTERVAL seconds (default: 10)")
print("After 30 seconds, check your Prometheus/OTEL collector for:")
print("  - gen_ai_power_consumption (CPU/GPU/RAM power in Watts)")
print("  - gen_ai_energy_consumed (CPU/GPU/RAM energy in kWh)")
print("  - gen_ai_energy_total (total energy)")
print("  - gen_ai_co2_emissions (CO2 emissions)")
print("  - gen_ai_codecarbon_task_duration (task duration)")
print()

for i in range(30):
    print(f"  {i+1}/30 seconds...", end="\r")
    time.sleep(1)

print("\n\nDone! Check your metrics endpoint now.")
print(f"OTEL Endpoint: {os.environ['OTEL_EXPORTER_OTLP_ENDPOINT']}")
print("\nNote: Metrics are exported via OTLP and should appear in your configured backend.")
