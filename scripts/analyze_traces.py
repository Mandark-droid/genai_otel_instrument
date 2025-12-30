#!/usr/bin/env python3
"""
Trace Analysis Script for GenAI OTEL Instrumentation

This script fetches traces from Jaeger and analyzes them to verify that
evaluation attributes are being correctly captured and exported.

Usage:
    python scripts/analyze_traces.py
"""

import json
import sys
from typing import Dict, List, Set
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Jaeger configuration
JAEGER_HOST = "192.168.206.128"
JAEGER_PORT = "16686"
JAEGER_API_URL = f"http://{JAEGER_HOST}:{JAEGER_PORT}/api/traces"

# Trace IDs to analyze
TRACES = {
    # PII Detection
    "HIPAA Example": {
        "trace_id": "d6f9d525e2d03599fc154b95d356a755",
        "category": "pii_detection",
        "expected_attributes": [
            "evaluation.pii.prompt.detected",
            "evaluation.pii.prompt.entity_count",
            "evaluation.pii.prompt.entity_types",
        ],
    },
    "Redaction Example": {
        "trace_id": "8bbcc4097d59faff546bffc58d2af03e",
        "category": "pii_detection",
        "expected_attributes": [
            "evaluation.pii.prompt.detected",
            "evaluation.pii.prompt.entity_count",
            "evaluation.pii.prompt.redacted",
        ],
    },
    "Combined PII Example": {
        "trace_id": "1959a412151010d6d1646c0b0a0bd18b",
        "category": "pii_detection",
        "expected_attributes": [
            "evaluation.pii.prompt.detected",
            "evaluation.pii.prompt.entity_count",
            "evaluation.pii.response.detected",
            "evaluation.pii.response.entity_count",
        ],
    },
    # Bias Detection
    "Bias Blocking Example": {
        "trace_id": "86bbcc408a967a5d1ddf16b818c088e4",
        "category": "bias_detection",
        "expected_attributes": [
            "evaluation.bias.prompt.detected",
            "evaluation.bias.prompt.max_score",
            "evaluation.bias.prompt.detected_biases",
        ],
    },
    # Prompt Injection
    "Prompt Injection": {
        "trace_id": "17e5444566606a2755cd548a7e8fc78b",
        "category": "prompt_injection",
        "expected_attributes": [
            "evaluation.prompt_injection.detected",
            "evaluation.prompt_injection.score",
            "evaluation.prompt_injection.types",
        ],
    },
    "Prompt Jailbreak": {
        "trace_id": "3483ad1362c3855111e7923a42a4a94a",
        "category": "prompt_injection",
        "expected_attributes": [
            "evaluation.prompt_injection.detected",
            "evaluation.prompt_injection.score",
            "evaluation.prompt_injection.types",
            "evaluation.prompt_injection.jailbreak_patterns",
        ],
    },
    "Prompt Basic": {
        "trace_id": "8412f74c0a26c077382bbbfd6b60f921",
        "category": "prompt_injection",
        "expected_attributes": [
            "evaluation.prompt_injection.detected",
            "evaluation.prompt_injection.score",
        ],
    },
    "Prompt System Override": {
        "trace_id": "f247b2f6edafd422e8d92d943a299cbc",
        "category": "prompt_injection",
        "expected_attributes": [
            "evaluation.prompt_injection.detected",
            "evaluation.prompt_injection.score",
            "evaluation.prompt_injection.types",
            "evaluation.prompt_injection.system_override_patterns",
        ],
    },
    # Hallucination
    "Hallucination RAG Faithfulness": {
        "trace_id": "24ff914474c5d9fa001ed415c2afdb01",
        "category": "hallucination",
        "expected_attributes": [
            "evaluation.hallucination.response.detected",
            "evaluation.hallucination.response.score",
            "evaluation.hallucination.response.citations",
            "evaluation.hallucination.response.claims",
        ],
    },
    "Hallucination Basic": {
        "trace_id": "09bd0ded0332653e03c1a99012860642",
        "category": "hallucination",
        "expected_attributes": [
            "evaluation.hallucination.response.detected",
            "evaluation.hallucination.response.score",
            "evaluation.hallucination.response.citations",
        ],
    },
    "Hallucination Citation": {
        "trace_id": "35fc0e6df61664fa29a4f9245471ad6c",
        "category": "hallucination",
        "expected_attributes": [
            "evaluation.hallucination.response.detected",
            "evaluation.hallucination.response.score",
            "evaluation.hallucination.response.citations",
            "evaluation.hallucination.response.hedge_words",
        ],
    },
}


def fetch_trace(trace_id: str) -> Dict:
    """Fetch trace data from Jaeger API."""
    url = f"{JAEGER_API_URL}/{trace_id}"

    try:
        request = Request(url)
        with urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode())
            return data
    except HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}")
        return None
    except URLError as e:
        print(f"URL Error: {e.reason}")
        return None
    except Exception as e:
        print(f"Error fetching trace: {e}")
        return None


def extract_span_attributes(trace_data: Dict) -> Dict[str, Set[str]]:
    """Extract all attributes from all spans in the trace."""
    all_attributes = {}

    if not trace_data or "data" not in trace_data:
        return all_attributes

    for trace in trace_data.get("data", []):
        for span in trace.get("spans", []):
            span_name = span.get("operationName", "unknown")
            tags = span.get("tags", [])

            attributes = set()
            for tag in tags:
                key = tag.get("key", "")
                if key.startswith("evaluation."):
                    attributes.add(key)

            if attributes:
                all_attributes[span_name] = attributes

    return all_attributes


def analyze_trace(trace_name: str, trace_info: Dict) -> Dict:
    """Analyze a single trace and return findings."""
    trace_id = trace_info["trace_id"]
    category = trace_info["category"]
    expected_attrs = set(trace_info["expected_attributes"])

    print(f"\n{'='*80}")
    print(f"Analyzing: {trace_name}")
    print(f"{'='*80}")
    print(f"Trace ID: {trace_id}")
    print(f"Category: {category}")
    print(f"Expected Attributes: {len(expected_attrs)}")

    # Fetch trace
    trace_data = fetch_trace(trace_id)
    if not trace_data:
        print(f"[X] FAILED to fetch trace")
        return {
            "trace_name": trace_name,
            "trace_id": trace_id,
            "category": category,
            "status": "fetch_failed",
            "found_attributes": set(),
            "missing_attributes": expected_attrs,
        }

    # Extract attributes
    span_attributes = extract_span_attributes(trace_data)

    # Combine all found attributes across all spans
    all_found_attrs = set()
    for span_name, attrs in span_attributes.items():
        all_found_attrs.update(attrs)

    # Calculate missing attributes
    missing_attrs = expected_attrs - all_found_attrs
    extra_attrs = all_found_attrs - expected_attrs

    # Report findings
    print(f"\nSpans in trace: {len(span_attributes)}")
    for span_name, attrs in span_attributes.items():
        print(f"  - {span_name}: {len(attrs)} evaluation attributes")

    print(f"\nFound Attributes ({len(all_found_attrs)}):")
    for attr in sorted(all_found_attrs):
        print(f"  [+] {attr}")

    if missing_attrs:
        print(f"\n[X] Missing Attributes ({len(missing_attrs)}):")
        for attr in sorted(missing_attrs):
            print(f"  [-] {attr}")
    else:
        print(f"\n[OK] All expected attributes found!")

    if extra_attrs:
        print(f"\nExtra Attributes ({len(extra_attrs)}):")
        for attr in sorted(extra_attrs):
            print(f"  [*] {attr}")

    status = "complete" if not missing_attrs else "incomplete"

    return {
        "trace_name": trace_name,
        "trace_id": trace_id,
        "category": category,
        "status": status,
        "found_attributes": all_found_attrs,
        "missing_attributes": missing_attrs,
        "extra_attributes": extra_attrs,
    }


def print_summary(results: List[Dict]):
    """Print summary of all trace analyses."""
    print(f"\n{'='*80}")
    print("TRACE ANALYSIS SUMMARY")
    print(f"{'='*80}\n")

    # Group by category
    by_category = {}
    for result in results:
        category = result["category"]
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(result)

    # Print by category
    for category, traces in sorted(by_category.items()):
        print(f"\n{category.upper().replace('_', ' ')}:")
        print("-" * 80)

        for trace in traces:
            status_icon = "[OK]" if trace["status"] == "complete" else "[X]"
            missing_count = len(trace.get("missing_attributes", []))
            print(f"{status_icon} {trace['trace_name']}")
            print(f"   Trace ID: {trace['trace_id']}")
            print(f"   Status: {trace['status']}")
            if missing_count > 0:
                print(f"   Missing Attributes: {missing_count}")
                for attr in sorted(trace["missing_attributes"]):
                    print(f"      - {attr}")

    # Overall statistics
    total = len(results)
    complete = sum(1 for r in results if r["status"] == "complete")
    incomplete = sum(1 for r in results if r["status"] == "incomplete")
    fetch_failed = sum(1 for r in results if r["status"] == "fetch_failed")

    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}")
    print(f"Total Traces Analyzed: {total}")
    print(f"[OK] Complete (all attributes found): {complete}")
    print(f"[X] Incomplete (missing attributes): {incomplete}")
    print(f"[X] Failed to Fetch: {fetch_failed}")

    # Identify common missing attributes
    all_missing = set()
    for result in results:
        all_missing.update(result.get("missing_attributes", []))

    if all_missing:
        print(f"\nCommonly Missing Attributes:")
        for attr in sorted(all_missing):
            count = sum(1 for r in results if attr in r.get("missing_attributes", []))
            print(f"  - {attr} (missing in {count}/{total} traces)")


def main():
    """Main entry point."""
    print("=" * 80)
    print("GenAI OTEL Trace Analysis")
    print("=" * 80)
    print(f"Jaeger Endpoint: {JAEGER_API_URL}")
    print(f"Total Traces to Analyze: {len(TRACES)}")

    results = []

    for trace_name, trace_info in TRACES.items():
        result = analyze_trace(trace_name, trace_info)
        results.append(result)

    print_summary(results)

    # Exit code
    incomplete = sum(1 for r in results if r["status"] != "complete")
    if incomplete > 0:
        print(f"\n[!] {incomplete} traces have issues")
        sys.exit(1)
    else:
        print(f"\n[OK] All traces validated successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
