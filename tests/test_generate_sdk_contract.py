"""Tests for ``scripts/generate_sdk_contract.py``.

The generator turns this library's real behaviour into machine-readable files
that the Node, .NET, Java and Go ports test themselves against. Those ports
cannot read Python, so what is asserted here is the part of the contract they
rely on blindly: the files exist and parse, the output is deterministic, the
manifest hashes are honest, and a handful of facts that must never silently
change (a default flipping, an unpriced model reading as priced).

The generator runs in a subprocess. It installs a recording MeterProvider and
simulates NVML, both of which are process-global, so running it in-process
would leak into the rest of the suite.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "scripts" / "generate_sdk_contract.py"

JSON_FILES = [
    "semconv.json",
    "metrics.json",
    "config.json",
    "cost_vectors.json",
    "eval_vectors.json",
    "eval_rules.json",
]
DETECTORS = [
    "pii",
    "toxicity",
    "bias",
    "prompt_injection",
    "restricted_topics",
    "hallucination",
]


def _generate(out_dir: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(GENERATOR), "--out", str(out_dir)],
        cwd=str(REPO_ROOT),
        # The generator reads nothing. Inheriting stdin fails on Windows when the
        # test runner itself has none ("WinError 6: The handle is invalid").
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, (
        f"generator failed ({completed.returncode})\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


@pytest.fixture(scope="module")
def contract_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("sdk-contract-a")
    _generate(out)
    return out


@pytest.fixture(scope="module")
def second_contract_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("sdk-contract-b")
    _generate(out)
    return out


def _load(directory: Path, name: str):
    return json.loads((directory / name).read_text(encoding="utf-8"))


def test_all_files_exist_and_parse(contract_dir):
    for name in JSON_FILES + ["MANIFEST.json"]:
        path = contract_dir / name
        assert path.is_file(), f"{name} was not written"
        json.loads(path.read_text(encoding="utf-8"))
    schema = contract_dir / "SCHEMA.md"
    assert schema.is_file(), "SCHEMA.md was not written"
    text = schema.read_text(encoding="utf-8")
    for name in JSON_FILES + ["MANIFEST.json"]:
        assert name in text, f"SCHEMA.md does not describe {name}"


def test_every_file_says_how_it_was_extracted(contract_dir):
    for name in JSON_FILES:
        assert _load(contract_dir, name).get("extraction"), f"{name} has no extraction note"


def test_json_is_strict_and_sorted(contract_dir):
    """Ports parse these with strict parsers: no NaN/Infinity, keys sorted."""
    for name in JSON_FILES + ["MANIFEST.json"]:
        raw = (contract_dir / name).read_text(encoding="utf-8")

        def _reject(token):
            raise AssertionError(f"{name} contains the non-JSON constant {token}")

        parsed = json.loads(raw, parse_constant=_reject)
        assert raw == json.dumps(parsed, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def test_known_metrics_present(contract_dir):
    metrics = {m["name"]: m for m in _load(contract_dir, "metrics.json")["metrics"]}
    assert "gen_ai.gpu.utilization" in metrics
    assert "gen_ai.co2.emissions" in metrics
    assert metrics["gen_ai.gpu.utilization"]["kind"] == "observable_gauge"
    assert metrics["gen_ai.co2.emissions"]["kind"] == "counter"
    assert metrics["gen_ai.co2.emissions"]["unit"] == "gCO2e"
    # A View-configured histogram carries the boundaries the library really uses.
    duration = metrics["gen_ai.client.operation.duration"]
    assert duration["kind"] == "histogram"
    assert duration["bucket_boundaries"][0] == 0.01
    assert duration["bucket_boundaries"][-1] == 81.92


def test_config_defaults(contract_dir):
    variables = {v["env"]: v for v in _load(contract_dir, "config.json")["variables"]}
    assert variables["GENAI_ENABLE_COST_TRACKING"]["default"] is True
    assert variables["GENAI_ENABLE_COST_TRACKING"]["type"] == "bool"
    assert variables["GENAI_ENABLE_CO2_TRACKING"]["default"] is False
    # Variables read outside OTelConfig are part of the contract too.
    assert "GENAI_SKIP_COLLECTOR_CHECK" in variables
    assert "GENAI_SERVICE_INSTANCE_ID_MODE" in variables


def test_config_records_what_garbage_really_does(contract_dir):
    variables = {v["env"]: v for v in _load(contract_dir, "config.json")["variables"]}

    def case(env, raw):
        matches = [c for c in variables[env]["cases"] if c["raw"] == raw]
        assert matches, f"{env} has no case for {raw!r}"
        return matches[0]

    # Only the literal "true" (any case) is true. "yes" is not.
    assert case("GENAI_ENABLE_GPU_METRICS", "yes")["parsed"] is False
    assert case("GENAI_ENABLE_GPU_METRICS", "TRUE")["parsed"] is True
    # A non-numeric float is not defaulted: constructing the config raises.
    garbage = case("GENAI_SAMPLING_RATE", "abc")
    assert garbage["error"]["type"] == "ValueError"
    assert "parsed" not in garbage
    # ...whereas the exporter timeout does fall back.
    assert case("OTEL_EXPORTER_OTLP_TIMEOUT", "abc")["parsed"] == 60


def test_cost_vectors_cover_the_pricing_table(contract_dir):
    vectors = _load(contract_dir, "cost_vectors.json")["vectors"]
    assert len(vectors) >= 300
    pricing = json.loads((REPO_ROOT / "genai_otel" / "llm_pricing.json").read_text("utf-8"))
    # Keys that differ only by case collapse to one entry in the calculator's
    # index, so coverage is checked case-insensitively.
    for category in ("chat", "embeddings", "images", "audio", "speech_to_text", "decision"):
        resolved = {
            v["resolved_key"].lower()
            for v in vectors
            if v.get("resolved_category") == category and v["resolved_key"]
        }
        missing = {key.lower() for key in pricing[category]} - resolved
        assert not missing, f"{category}: no vector resolves to {sorted(missing)[:5]}"


def test_unpriced_cost_vector_exists(contract_dir):
    vectors = _load(contract_dir, "cost_vectors.json")["vectors"]
    unpriced = [v for v in vectors if v["priced"] is False and v["call_type"] == "chat"]
    assert unpriced, "no unpriced chat vector"
    for vector in unpriced:
        # Recorded exactly as the library returns it: 0.0, not null.
        assert vector["cost"] == 0.0
        assert vector["pricing_source"] == "unpriced"
    estimated = [v for v in vectors if v["pricing_source"] == "estimated"]
    assert estimated and all(v["priced"] for v in estimated)


def test_cost_vectors_pin_the_boundary_cases(contract_dir):
    by_model = {}
    for vector in _load(contract_dir, "cost_vectors.json")["vectors"]:
        by_model.setdefault(vector["model"], []).append(vector)
    # The mis-bill tests/test_pricing_substring_boundaries.py exists to prevent.
    assert all(v["resolved_key"] != "o1" for v in by_model["Sao10K/L3-8B-Stheno-v3.2"])
    # A dated snapshot routes to its family, not to a shorter sibling.
    assert by_model["gpt-5.5-mini-2026-04-01"][0]["resolved_key"] == "gpt-5.5-mini"


def test_speech_to_text_and_decision_are_priced(contract_dir):
    document = _load(contract_dir, "cost_vectors.json")
    assert {"speech_to_text", "decision"} <= set(document["supported_call_types"])
    vectors = document["vectors"]
    stt = [v for v in vectors if v["kind"] == "table" and v["call_type"] == "speech_to_text"]
    # Priced since 1.32 (some entries are free, so not every cost is above 0).
    assert stt and all(v["priced"] for v in stt) and any(v["cost"] > 0 for v in stt)
    # Usage carrying "seconds" is billed from the audio table.
    by_seconds = [
        v for v in vectors if v["call_type"] == "speech_to_text" and "seconds" in v["usage"]
    ]
    assert by_seconds and all(v["resolved_category"] == "audio" for v in by_seconds)
    # A decision model is never priced from the chat table.
    gpt4o = [v for v in vectors if v["call_type"] == "decision" and v["model"] == "gpt-4o"]
    assert gpt4o and all(v["priced"] is False and v["cost"] == 0.0 for v in gpt4o)


def test_session_and_agent_graph_surface(contract_dir):
    variables = {v["env"]: v for v in _load(contract_dir, "config.json")["variables"]}
    for env in ("GENAI_ENABLE_AGENT_GRAPH_ATTRIBUTES", "GENAI_ENABLE_SESSION_PROPAGATION"):
        assert variables[env]["default"] is True
    keys = variables["GENAI_SESSION_BAGGAGE_KEYS"]
    assert keys["default"] == []
    cases = {c["raw"]: c["parsed"] for c in keys["cases"]}
    # Only session.id and user.id are carried, in that order, whatever was asked.
    assert cases["user.id,session.id"] == ["session.id", "user.id"]
    assert cases["session.id,gen_ai.conversation.id"] == ["session.id"]

    semconv = _load(contract_dir, "semconv.json")
    attributes = {a["name"] for a in semconv["attributes"]}
    for name in (
        "agent.name",
        "graph.node.id",
        "graph.node.parent_id",
        "graph.node.name",
        "graph.node.display_name",
        "openinference.span.kind",
        "langgraph.node",
        "langgraph.step",
        "langgraph.triggers",
        "session.id",
        "user.id",
        "gen_ai.conversation.id",
    ):
        assert name in attributes, f"{name} missing from semconv.json"
    span_names = {s["name"] for s in semconv["span_names"]}
    # Named through a helper, not at the span call: the scan has to follow it.
    assert "langgraph.node.{name}" in span_names
    assert "aws.bedrock.converse_stream" in span_names


def test_eval_vectors_cover_every_detector(contract_dir):
    document = _load(contract_dir, "eval_vectors.json")
    for detector in DETECTORS:
        cases = [v for v in document["vectors"] if v["detector"] == detector]
        assert len(cases) >= 20, f"{detector}: only {len(cases)} cases"
        assert any(not v["detected"] for v in cases), f"{detector}: no clean case"
    # Every detector that has a rule-based path must actually fire on something.
    for detector in set(DETECTORS) - {"toxicity"}:
        assert any(
            v["detected"] for v in document["vectors"] if v["detector"] == detector
        ), f"{detector}: nothing detected"


def test_eval_rules_export_the_rule_data(contract_dir):
    rules = _load(contract_dir, "eval_rules.json")["detectors"]
    assert set(rules) == set(DETECTORS)
    assert rules["prompt_injection"]["patterns"]["jailbreak"]
    assert rules["pii"]["fallback_patterns"]
    # Toxicity has no rule-based implementation; the contract must say so
    # rather than leave a port to invent one.
    assert rules["toxicity"]["rule_based"] is False


def test_manifest_hashes_match(contract_dir):
    manifest = _load(contract_dir, "MANIFEST.json")
    assert set(manifest["sha256"]) == set(JSON_FILES + ["SCHEMA.md"])
    for name, expected in manifest["sha256"].items():
        actual = hashlib.sha256((contract_dir / name).read_bytes()).hexdigest()
        assert actual == expected, f"{name} hash mismatch"
    pricing = hashlib.sha256((REPO_ROOT / "genai_otel" / "llm_pricing.json").read_bytes())
    assert manifest["pricing_sha256"] == pricing.hexdigest()
    assert manifest["generated_at"]
    assert "python_tag" in manifest and "python_package_version" in manifest


def test_output_is_deterministic(contract_dir, second_contract_dir):
    for name in JSON_FILES + ["SCHEMA.md"]:
        first = (contract_dir / name).read_bytes()
        second = (second_contract_dir / name).read_bytes()
        assert first == second, f"{name} differs between two runs"
    first = _load(contract_dir, "MANIFEST.json")
    second = _load(second_contract_dir, "MANIFEST.json")
    first.pop("generated_at")
    second.pop("generated_at")
    assert first == second
