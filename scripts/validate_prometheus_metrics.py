#!/usr/bin/env python3
"""
Prometheus Metrics Validation Script

This script validates that all evaluation metrics are being properly exported
to Prometheus for each evaluation type (PII, Toxicity, Bias, Prompt Injection,
Restricted Topics, and Hallucination detection).

Usage:
    python scripts/validate_prometheus_metrics.py [prometheus_url]

Default Prometheus URL: http://192.168.206.128:9091
"""

import json
import sys
from typing import Dict, List, Set
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Prometheus configuration
DEFAULT_PROMETHEUS_URL = "http://192.168.206.128:9091"

# Expected metrics for each evaluation type
EXPECTED_METRICS = {
    "pii_detection": [
        "genai_evaluation_pii_detections_total",
        "genai_evaluation_pii_entities_total",
        "genai_evaluation_pii_blocked_total",
    ],
    "toxicity_detection": [
        "genai_evaluation_toxicity_detections_total",
        "genai_evaluation_toxicity_categories_total",
        "genai_evaluation_toxicity_blocked_total",
    ],
    "bias_detection": [
        "genai_evaluation_bias_detections_total",
        "genai_evaluation_bias_types_total",
        "genai_evaluation_bias_blocked_total",
    ],
    "prompt_injection": [
        "genai_evaluation_prompt_injection_detections_total",
        "genai_evaluation_prompt_injection_types_total",
        "genai_evaluation_prompt_injection_blocked_total",
    ],
    "restricted_topics": [
        "genai_evaluation_restricted_topics_detections_total",
        "genai_evaluation_restricted_topics_types_total",
        "genai_evaluation_restricted_topics_blocked_total",
    ],
    "hallucination": [
        "genai_evaluation_hallucination_detections_total",
        "genai_evaluation_hallucination_indicators_total",
    ],
}


def fetch_prometheus_metrics(prometheus_url: str) -> List[str]:
    """Fetch all metric names from Prometheus."""
    url = f"{prometheus_url}/api/v1/label/__name__/values"

    try:
        request = Request(url)
        with urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode())
            if data["status"] == "success":
                return data["data"]
            else:
                print(f"Error fetching metrics: {data}")
                return []
    except HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}")
        return []
    except URLError as e:
        print(f"URL Error: {e.reason}")
        return []
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return []


def query_metric_value(prometheus_url: str, metric_name: str) -> Dict:
    """Query a specific metric from Prometheus."""
    url = f"{prometheus_url}/api/v1/query?query={metric_name}"

    try:
        request = Request(url)
        with urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode())
            if data["status"] == "success":
                return data["data"]
            else:
                return {}
    except Exception as e:
        print(f"Error querying metric {metric_name}: {e}")
        return {}


def validate_metrics(prometheus_url: str) -> Dict:
    """Validate all evaluation metrics."""
    print("=" * 80)
    print("Prometheus Metrics Validation")
    print("=" * 80)
    print(f"Prometheus URL: {prometheus_url}")
    print()

    # Fetch all available metrics
    all_metrics = fetch_prometheus_metrics(prometheus_url)
    if not all_metrics:
        print("[X] Failed to fetch metrics from Prometheus")
        return {"status": "failed", "error": "Failed to fetch metrics"}

    print(f"Total metrics available: {len(all_metrics)}")

    # Filter to genai evaluation metrics
    genai_eval_metrics = [m for m in all_metrics if m.startswith("genai_evaluation_")]
    print(f"GenAI evaluation metrics found: {len(genai_eval_metrics)}")
    print()

    # Validate each category
    results = {}
    total_expected = 0
    total_found = 0
    total_missing = 0

    for category, expected_metrics in EXPECTED_METRICS.items():
        print(f"{category.upper().replace('_', ' ')}:")
        print("-" * 80)

        found = []
        missing = []

        for metric in expected_metrics:
            total_expected += 1
            if metric in genai_eval_metrics:
                found.append(metric)
                total_found += 1

                # Query metric value
                result = query_metric_value(prometheus_url, metric)
                result_count = len(result.get("result", []))
                if result_count > 0:
                    print(f"  [+] {metric} ({result_count} series)")
                else:
                    print(f"  [+] {metric} (0 series - no data yet)")
            else:
                missing.append(metric)
                total_missing += 1
                print(f"  [-] {metric} (MISSING)")

        results[category] = {
            "expected": len(expected_metrics),
            "found": len(found),
            "missing": len(missing),
            "missing_metrics": missing,
            "status": "complete" if len(missing) == 0 else "incomplete",
        }

        print()

    # Print summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total Expected Metrics: {total_expected}")
    print(f"[+] Found: {total_found}")
    print(f"[-] Missing: {total_missing}")
    print()

    # Print missing metrics by category
    if total_missing > 0:
        print("Missing Metrics by Category:")
        print("-" * 80)
        for category, result in results.items():
            if result["missing"]:
                print(f"\n{category}:")
                for metric in result["missing_metrics"]:
                    print(f"  - {metric}")
        print()

    # Overall status
    if total_missing == 0:
        print("[OK] All evaluation metrics are present in Prometheus!")
        return {"status": "success", "results": results}
    else:
        print(f"[X] {total_missing} metrics are missing from Prometheus")
        return {"status": "incomplete", "results": results}


def main():
    """Main entry point."""
    prometheus_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROMETHEUS_URL

    result = validate_metrics(prometheus_url)

    if result["status"] == "success":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
