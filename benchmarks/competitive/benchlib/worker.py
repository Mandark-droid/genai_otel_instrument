"""One benchmark run: bare-SDK baseline, then one library/config/mode, same process.

Invoked by run_benchmark.py inside the library's own environment:

    python -m benchlib.worker <library> <config> <mode> <out.json>

Settings arrive as environment variables (BENCH_PAYLOAD, BENCH_SCENARIOS,
BENCH_REPS, BENCH_SCALE, BENCH_SYNC_SCALE, BENCH_PIN_CPUS, BENCH_WORKDIR).

The baseline is measured before the library is imported, because several of the
libraries patch the OpenAI SDK at class level and cannot be measured next to an
untouched client afterwards. The `none` control runs both phases on the bare SDK
and shows how much the phase order alone moves the numbers.
"""

import faulthandler
import gc
import json
import os
import sys
import threading
import time

from . import common

common.install_network_guard()
_PIN = common.pin_process(
    [int(c) for c in os.environ.get("BENCH_PIN_CPUS", "").split(",") if c.strip()]
)
# If a run stops making progress, dump every thread's stack and exit instead of hanging.
faulthandler.dump_traceback_later(int(os.environ.get("BENCH_WATCHDOG_S", "900")), exit=True)

import openai  # noqa: E402

from . import adapters  # noqa: E402


def inspect(library, config, out_path):
    """Record what one configuration writes for one sync and one streamed call."""
    a = adapters.setup(library, config, "INSPECT")
    out = {"library": library, "config": config, "calls": {}, "errors": {}}

    def take():
        if a.drain:
            a.drain()
        spans = list(a.span_store.get_finished_spans()) if a.span_store is not None else []
        if a.span_store is not None:
            a.span_store.clear()
        payloads = [(name, len(body)) for name, body in a.payloads]
        del a.payloads[:]
        rows = []
        for s in spans:
            attrs = dict(s.attributes or {})
            rows.append(
                {
                    "name": s.name,
                    "attributes": {
                        k: (repr(v) if len(repr(v)) < 60 else repr(v)[:57] + "...")
                        for k, v in sorted(attrs.items())
                    },
                    "attribute_chars": sum(len(str(v)) for v in attrs.values()),
                    "events": [
                        {"name": e.name, "attribute_keys": sorted(e.attributes or {})}
                        for e in s.events
                    ],
                }
            )
        return {"spans": rows, "own_sender_payloads": payloads}

    for name, kind, req in (
        ("sync", "sync_json", common.REQ),
        ("sync_stream", "sync_sse", common.REQ_STREAM),
    ):
        try:
            result = a.make(kind).chat.completions.create(**req)
            if name == "sync_stream":
                for _ in result:
                    pass
            out["calls"][name] = take()
        except Exception as e:  # noqa: BLE001
            out["errors"][name] = "%s: %s" % (type(e).__name__, e)
    if a.metric_reader is not None:
        names = []
        data = a.metric_reader.get_metrics_data()
        for rm in data.resource_metrics if data else []:
            for sm in rm.scope_metrics:
                names.extend(m.name for m in sm.metrics)
        out["metrics"] = sorted(names)
    out["egress_attempts"] = common.egress_summary()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


def main():
    library, config, mode, out_path = sys.argv[1:5]
    if mode == "INSPECT":
        inspect(library, config, out_path)
        os._exit(0)
    reps = int(os.environ.get("BENCH_REPS", "5"))
    scale = float(os.environ.get("BENCH_SCALE", "1.0"))
    sync_scale = float(os.environ.get("BENCH_SYNC_SCALE", "1.0"))
    wanted = [
        s for s in os.environ.get("BENCH_SCENARIOS", "").split(",") if s
    ] or common.ALL_SCENARIOS

    def scale_for(sc):
        return scale * sync_scale if sc in ("sync", "sync_stream") else scale

    started = time.time()
    result = {
        "library": library,
        "config": config,
        "mode": mode,
        "system": common.system_info(),
        "pin": _PIN,
        "payload": common.PAYLOAD,
        "payload_description": common.PAYLOADS[common.PAYLOAD],
        "reps": reps,
        "scale": scale,
        "sync_scale": sync_scale,
        "scenarios": wanted,
        "openai": openai.__version__,
    }
    common.spin(1.0)
    base_clients = common.plain_clients()
    baseline = {}
    for sc in wanted:
        baseline[sc] = common.measure(
            sc, base_clients[common.SCENARIOS[sc][0]], reps=reps, scale=scale_for(sc)
        )
    result["baseline"] = baseline
    result["rss_before_setup_mb"] = common.memory_mb()

    t0 = time.perf_counter()
    a = adapters.setup(library, config, mode)
    result["setup_seconds"] = time.perf_counter() - t0
    result["notes"] = a.notes
    result["records"] = adapters.describe(library, config) if library != "none" else "nothing"
    result["unsupported"] = a.unsupported
    result["versions"] = common.installed_versions()
    gc.collect()
    result["rss_after_setup_mb"] = common.memory_mb()
    result["threads_after_setup"] = threading.active_count()

    instrumented = {}
    before = dict(common.ACTIVITY)
    for sc in wanted:
        if sc in a.unsupported:
            continue
        client = a.make(common.SCENARIOS[sc][0])
        instrumented[sc] = common.measure(sc, client, drain=a.drain, reps=reps, scale=scale_for(sc))
    if a.drain:
        a.drain()
    result["instrumented"] = instrumented
    result["activity"] = {k: common.ACTIVITY[k] - before[k] for k in common.ACTIVITY}
    result["rss_end_mb"] = common.memory_mb()
    result["egress_attempts"] = common.egress_summary()

    overhead = {}
    for sc, r in instrumented.items():
        b = baseline[sc]
        o = {
            "base_min": b["us_min"],
            "base_median": b["us_median"],
            "instr_min": r["us_min"],
            "instr_median": r["us_median"],
            "overhead_min": r["us_min"] - b["us_min"],
            "overhead_median": r["us_median"] - b["us_median"],
            "multiple_min": r["us_min"] / b["us_min"],
        }
        if "stream" in sc:
            o["overhead_per_chunk_min"] = o["overhead_min"] / common.N_STREAM_CHUNKS
        if "drain_us_median" in r:
            o["drain_median"] = r["drain_us_median"]
        overhead[sc] = o
    result["overhead"] = overhead
    result["wall_seconds"] = time.time() - started
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)

    print(
        "%s/%s/%s payload=%s wall=%.0fs activity=%s egress=%s"
        % (
            library,
            config,
            mode,
            common.PAYLOAD,
            result["wall_seconds"],
            result["activity"],
            result["egress_attempts"] or "none",
        )
    )
    for sc, o in overhead.items():
        print(
            "  %-15s base %8.1f  instrumented %8.1f  overhead min %7.1f  median %7.1f us"
            % (sc, o["base_min"], o["instr_min"], o["overhead_min"], o["overhead_median"])
        )
    for sc, why in a.unsupported.items():
        if sc in wanted:
            print("  %-15s not measured: %s" % (sc, why))
    sys.stdout.flush()
    # Background senders of some libraries keep the interpreter alive at exit.
    os._exit(0)


if __name__ == "__main__":
    main()
