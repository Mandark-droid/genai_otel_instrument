"""Check if codecarbon metrics are in Prometheus."""

import json

import requests

PROMETHEUS_URL = "http://192.168.206.128:9090"

print("Checking Prometheus for gen_ai metrics...\n")

# Query for all metric names
try:
    response = requests.get(f"{PROMETHEUS_URL}/api/v1/label/__name__/values", timeout=10)
    if response.status_code == 200:
        metrics = response.json().get("data", [])

        # Filter for gen_ai metrics
        gen_ai_metrics = [m for m in metrics if "gen_ai" in m.lower() or "codecarbon" in m.lower()]

        if gen_ai_metrics:
            print(f"Found {len(gen_ai_metrics)} gen_ai metrics:")
            for metric in sorted(gen_ai_metrics):
                print(f"  - {metric}")

                # Get current value
                query_response = requests.get(
                    f"{PROMETHEUS_URL}/api/v1/query", params={"query": metric}, timeout=10
                )
                if query_response.status_code == 200:
                    result = query_response.json().get("data", {}).get("result", [])
                    if result:
                        print(f"    Current values: {len(result)} time series")
                        for ts in result[:3]:  # Show first 3
                            labels = ts.get("metric", {})
                            value = ts.get("value", [None, None])[1]
                            print(f"      {labels} = {value}")
        else:
            print("No gen_ai metrics found in Prometheus!")
            print("\nPossible issues:")
            print("1. OTEL collector not receiving metrics from application")
            print("2. OTEL collector not exporting to Prometheus")
            print("3. Prometheus not scraping OTEL collector")
            print("4. Metrics haven't been exported yet (wait 10-60 seconds)")

    else:
        print(f"Error querying Prometheus: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"Error connecting to Prometheus: {e}")
    print(f"\nMake sure Prometheus is accessible at {PROMETHEUS_URL}")

# Also check if OTEL collector metrics endpoint is accessible
print("\n" + "=" * 60)
print("Checking OTEL Collector metrics...")
try:
    # Try common OTEL collector ports
    for port in [8888, 8889, 9091]:
        try:
            response = requests.get(f"http://192.168.206.128:{port}/metrics", timeout=2)
            if response.status_code == 200:
                lines = response.text.split("\n")
                gen_ai_lines = [l for l in lines if "gen_ai" in l.lower()]
                if gen_ai_lines:
                    print(f"\nFound gen_ai metrics on port {port}:")
                    for line in gen_ai_lines[:10]:
                        print(f"  {line}")
                else:
                    print(f"\nPort {port} accessible but no gen_ai metrics found")
        except:
            pass
except Exception as e:
    print(f"Error: {e}")
