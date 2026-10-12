"""Markdown tables for the evaluation-detector benchmark.

    python -m benchlib.eval_report <results-dir>

Stdlib only.
"""

import glob
import json
import os
import sys


def ms(us, nd=2):
    return "n/a" if us is None else ("%." + str(nd) + "f") % (us / 1000.0)


def base_for(cfg, runs):
    if cfg.startswith("og_"):
        return "og_base"
    if "_cap" in cfg and cfg != "ev_base_cap" and "ev_base_cap" in runs:
        return "ev_base_cap"
    return "ev_base"


def render(results_dir, w):
    runs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "eval_*.json"))):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        runs[d["config"]] = d
    if not runs:
        w("No result files in %s\n" % results_dir)
        return
    any_run = next(iter(runs.values()))
    w("# Evaluation-detector overhead\n\n")
    w("Generated from the result files in this directory. `added` columns are the configuration\n")
    w(
        "minus the baseline configuration (same workload, detectors off), each in its own process.\n\n"
    )
    w("## Environment\n\n")
    for k, v in any_run["system"].items():
        w("- %s: %s\n" % (k, v))
    w("- openai: %s\n\n" % any_run["openai"])
    order = sorted(runs, key=lambda c: (c.startswith("og_"), c not in ("ev_base", "og_base"), c))

    for mode in ("sync", "async"):
        w("## Latency, %s\n\n" % mode)
        head = "| configuration | workload | n | blocked | median ms | p95 ms | p99 ms | added median ms | added p99 ms |"
        sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|"
        if mode == "async":
            head += " loop hold max ms | baseline loop hold max ms |"
            sep += "---:|---:|"
        w(head + "\n" + sep + "\n")
        for cfg in order:
            b = runs.get(base_for(cfg, runs))
            for name, cell in runs[cfg]["cells"].items():
                c = cell[mode]
                bc = b["cells"].get(name, {}).get(mode) if b else None
                row = [
                    cfg,
                    name,
                    str(c["n"]),
                    str(c["blocked"]),
                    ms(c["median"]),
                    ms(c["p95"]),
                    ms(c["p99"]),
                ]
                if bc and cfg != base_for(cfg, runs):
                    row += [ms(c["median"] - bc["median"]), ms(c["p99"] - bc["p99"])]
                else:
                    row += ["-", "-"]
                if mode == "async":
                    row += [
                        ms(c["loop_hold"]["max_us"]),
                        ms(bc["loop_hold"]["max_us"]) if bc else "-",
                    ]
                w("| " + " | ".join(row) + " |\n")
        w("\n")

    w(
        "## CPU (sync cells)\n\n| configuration | workload | wall ms per call | CPU ms per call | CPU-seconds per 1,000 calls | cores used (CPU / wall) | OS threads |\n|---|---|---:|---:|---:|---:|---:|\n"
    )
    for cfg in order:
        for name, cell in runs[cfg]["cells"].items():
            c = cell["sync"]
            w(
                "| %s | %s | %s | %s | %.2f | %.2f | %s |\n"
                % (
                    cfg,
                    name,
                    ms(c["wall_us_per_call"]),
                    ms(c["cpu_us_per_call"]),
                    c["cpu_us_per_call"] / 1000.0,
                    c["cores_used"] or 0,
                    c["os_threads_after"],
                )
            )
    w(
        "\n## Memory, threads and cold start\n\n| configuration | RSS after import openai MB | RSS after setup MB | peak RSS MB | OS threads after setup | OS threads at end | setup s | first call ms |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for cfg in order:
        r = runs[cfg]
        st = r["stages"]

        def f(x):
            return "n/a" if x is None else "%.0f" % x

        w(
            "| %s | %s | %s | %s | %s | %s | %.2f | %.1f |\n"
            % (
                cfg,
                f(st["after_import_openai"]["rss_mb"]),
                f(st["after_setup"]["rss_mb"]),
                f(st["end"]["peak_rss_mb"]),
                st["after_setup"]["os_threads"],
                st["end"]["os_threads"],
                r["setup_s"],
                r["first_call_s"] * 1000,
            )
        )
    w(
        "\n## Throughput under concurrency\n\n| configuration | workload | 1 thread calls/s | 8 threads calls/s | 8 threads cores used | gather(50) calls/s |\n|---|---|---:|---:|---:|---:|\n"
    )
    for cfg in order:
        for name, c in runs[cfg]["concurrency"].items():
            w(
                "| %s | %s | %.1f | %.1f | %.2f | %.1f |\n"
                % (
                    cfg,
                    name,
                    c["single_thread_calls_per_s"],
                    c["threads8"]["calls_per_s"],
                    c["threads8"]["cores_used"],
                    c["gather50"]["calls_per_s"],
                )
            )
    w(
        "\n## What lands on the spans of one call\n\n| configuration | workload | sync: added attributes | sync: their characters | sync: added events | async: added attributes | raised |\n|---|---|---:|---:|---:|---:|---|\n"
    )
    for cfg in order:
        for name, s in runs[cfg]["span"].items():
            w(
                "| %s | %s | %d | %d | %d | %d | %s |\n"
                % (
                    cfg,
                    name,
                    s["sync"]["added_attrs"],
                    s["sync"]["added_attr_chars"],
                    s["sync"]["added_events"],
                    s["async"]["added_attrs"],
                    (s.get("sync_raised") or "")[:70],
                )
            )
    w("\n## Blocked network attempts\n\n")
    for cfg in order:
        w("- %s: %s\n" % (cfg, runs[cfg]["egress_attempts"] or "none"))


def main():
    results_dir = sys.argv[1]
    with open(os.path.join(results_dir, "eval_tables.md"), "w", encoding="utf-8") as f:
        render(results_dir, f.write)
    print("wrote", os.path.join(results_dir, "eval_tables.md"))


if __name__ == "__main__":
    main()
