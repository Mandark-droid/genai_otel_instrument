"""Turn a directory of result files into markdown tables.

    python -m benchlib.report <results-dir>

Stdlib only; runs under any Python 3.9+.
"""

import glob
import json
import os
import statistics
import sys

TITLES = {
    "sync": "Sync, non-streaming",
    "async_seq": "Async, non-streaming, awaited one at a time",
    "async_gather10": "Async, asyncio.gather of 10",
    "sync_stream": "Sync streaming, 50 chunks",
    "async_stream": "Async streaming, 50 chunks",
}
ORDER = ["sync", "async_seq", "async_gather10", "sync_stream", "async_stream"]


def load(results_dir):
    runs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "run_*.json"))):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        rnd = int(os.path.basename(path).split("_")[1][1:])
        key = (d["payload"], d["library"], d["config"], d["mode"])
        runs.setdefault(key, {})[rnd] = d
    return runs


def fmt(x, nd=1):
    return "n/a" if x is None else ("%." + str(nd) + "f") % x


def render(results_dir, out=sys.stdout):
    runs = load(results_dir)
    if not runs:
        out.write("No result files in %s\n" % results_dir)
        return
    any_run = next(iter(next(iter(runs.values())).values()))
    w = out.write
    w("# Benchmark results\n\n")
    w("Generated from the result files in this directory. Overheads are microseconds per call,\n")
    w("instrumented minus bare SDK, measured in the same process.\n\n")
    w("## Environment\n\n")
    for k, v in any_run["system"].items():
        w("- %s: %s\n" % (k, v))
    w("- openai: %s\n" % any_run["openai"])
    w("- repetitions per scenario: %s, scale: %s\n\n" % (any_run["reps"], any_run["scale"]))

    w("## What each configuration records\n\n| library | config | records |\n|---|---|---|\n")
    seen = set()
    for (payload, lib, cfg, mode), by_round in sorted(runs.items()):
        if (lib, cfg) in seen or lib == "none":
            continue
        seen.add((lib, cfg))
        w("| %s | %s | %s |\n" % (lib, cfg, next(iter(by_round.values())).get("records", "")))
    w("\n")

    payloads = sorted({k[0] for k in runs})
    for payload in payloads:
        w("## Payload: %s\n\n" % payload)
        keys = sorted(k for k in runs if k[0] == payload)
        scenarios = [
            s
            for s in ORDER
            if any(
                s in d["overhead"] or s in d.get("unsupported", {})
                for k in keys
                for d in runs[k].values()
            )
        ]
        for sc in scenarios:
            stream = "stream" in sc
            w("### %s\n\n" % TITLES[sc])
            head = [
                "library",
                "config",
                "mode",
                "baseline min",
                "instrumented min",
                "overhead min per round",
                "overhead min",
                "overhead median",
                "x baseline",
            ]
            if stream:
                head.append("per chunk")
            head += ["flush per call", "rounds agree"]
            w("| " + " | ".join(head) + " |\n")
            w("|" + "|".join(["---"] * 3 + ["---:"] * (len(head) - 4) + ["---"]) + "|\n")
            for key in keys:
                _, lib, cfg, mode = key
                per_round = [
                    d["overhead"][sc] for _, d in sorted(runs[key].items()) if sc in d["overhead"]
                ]
                if not per_round:
                    why = next(iter(runs[key].values())).get("unsupported", {}).get(sc)
                    if why:
                        w(
                            "| %s | %s | %s | not measured |%s %s |\n"
                            % (lib, cfg, mode, " |" * (len(head) - 5), why)
                        )
                    continue
                omin = [o["overhead_min"] for o in per_round]
                mean_min = statistics.mean(omin)
                drain = [o["drain_median"] for o in per_round if "drain_median" in o]
                if lib == "none":
                    agree = "control"
                elif len(omin) < 2:
                    agree = "single round"
                else:
                    spread = (max(omin) - min(omin)) / abs(mean_min) if mean_min else 0.0
                    agree = ("yes (%.0f%%)" if spread <= 0.10 else "NO (%.0f%%)") % (spread * 100)
                cells = [
                    lib,
                    cfg,
                    mode,
                    fmt(statistics.mean(o["base_min"] for o in per_round)),
                    fmt(statistics.mean(o["instr_min"] for o in per_round)),
                    " / ".join(fmt(x) for x in omin),
                    fmt(mean_min),
                    fmt(statistics.mean(o["overhead_median"] for o in per_round)),
                    "%.3f" % statistics.mean(o["multiple_min"] for o in per_round),
                ]
                if stream:
                    cells.append(fmt(mean_min / 50.0, 2))
                cells += [fmt(statistics.mean(drain), 1) if drain else "-", agree]
                w("| " + " | ".join(cells) + " |\n")
            w("\n")

    w("## Run facts\n\n")
    w(
        "| payload | library | config | mode | spans exported | own-backend requests | hook traces | blocked egress | RSS after setup MB | setup s |\n"
    )
    w("|---|---|---|---|---:|---:|---:|---|---:|---:|\n")
    for key in sorted(runs):
        d = runs[key][min(runs[key])]
        act = d["activity"]
        w(
            "| %s | %s | %s | %s | %d | %d | %d | %s | %s | %.2f |\n"
            % (
                key[0],
                key[1],
                key[2],
                key[3],
                act["spans_exported"],
                act["backend_requests"],
                act["hook_traces"],
                d["egress_attempts"] or "none",
                fmt(d.get("rss_after_setup_mb"), 0),
                d["setup_seconds"],
            )
        )
    w("\n## Pinned versions seen at run time\n\n")
    keep = (
        "openai",
        "httpx",
        "pydantic",
        "pydantic-core",
        "wrapt",
        "opentelemetry-api",
        "opentelemetry-sdk",
        "genai-otel-instrument",
        "openlit",
        "traceloop-sdk",
        "opentelemetry-instrumentation-openai",
        "openinference-instrumentation-openai",
        "langfuse",
        "opik",
        "galileo",
    )
    libs = {}
    for key, by_round in runs.items():
        libs.setdefault(key[1], next(iter(by_round.values())).get("versions", {}))
    for lib, versions in sorted(libs.items()):
        w(
            "- %s: %s\n"
            % (lib, ", ".join("%s %s" % (k, versions[k]) for k in keep if k in versions))
        )


def main():
    results_dir = sys.argv[1]
    with open(os.path.join(results_dir, "tables.md"), "w", encoding="utf-8") as f:
        render(results_dir, f)
    print("wrote", os.path.join(results_dir, "tables.md"))


if __name__ == "__main__":
    main()
