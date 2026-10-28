"""One evaluation-detector configuration, one process.

    python -m benchlib.eval_worker <config> <out.json>

Configurations starting with `ev_` are TraceVerde detector settings (see
eval_common.EVAL_CONFIGS). Configurations starting with `og_` are OpenLIT's local
guards, the only inline in-SDK counterpart among the compared libraries.

Order inside the process: cold start (setup and first call), then per workload a
sync cell and an async cell (latency samples, CPU time, event-loop hold), then
concurrency (8 threads, gather of 50) on the 500-token workloads, then what the
configuration writes onto a span.
"""

import asyncio
import faulthandler
import json
import os
import sys
import threading
import time

from . import common
from . import eval_common as ec

common.install_network_guard()
faulthandler.dump_traceback_later(int(os.environ.get("BENCH_WATCHDOG_S", "1800")), exit=True)

from . import adapters  # noqa: E402
from . import eval_cells as cells  # noqa: E402

OPENLIT_GUARDS = {
    "og_base": lambda o: [],
    "og_pii": lambda o: [o.PII()],
    "og_inj": lambda o: [o.PromptInjection()],
    "og_mod": lambda o: [o.Moderation()],
    "og_topic": lambda o: [o.SensitiveTopic()],
    "og_all4": lambda o: [o.PII(), o.PromptInjection(), o.Moderation(), o.SensitiveTopic()],
    "og_all4_warn": lambda o: [
        o.PII(action="warn"),
        o.PromptInjection(action="warn"),
        o.Moderation(action="warn"),
        o.SensitiveTopic(action="warn"),
    ],
}


def mem():
    return {
        "rss_mb": common.memory_mb(),
        "peak_rss_mb": ec.peak_rss_mb(),
        "os_threads": ec.os_thread_count(),
        "py_threads": threading.active_count(),
    }


def main():
    config, out_path = sys.argv[1:3]
    budget = float(os.environ.get("BENCH_EVAL_BUDGET_S", "2.0"))
    sizes = [s for s in os.environ.get("BENCH_EVAL_SIZES", "S,M,L").split(",") if s]
    workloads = [(s, v) for s in sizes for v in ("clean", "dirty")]
    out = {
        "config": config,
        "system": common.system_info(),
        "sizes": sizes,
        "budget_s": budget,
        "stages": {},
    }
    out["stages"]["start"] = mem()
    import openai

    out["openai"] = openai.__version__
    out["stages"]["after_import_openai"] = mem()

    cpu0, t0 = time.process_time(), time.perf_counter()
    if config.startswith("og_"):
        import openlit

        guards = OPENLIT_GUARDS[config](openlit)
        a = adapters.Adapter("openlit", "default", "P1")
        adapters.setup_openlit(a, extra_kwargs={"guards": guards} if guards else None)
        from openlit.guard import GuardDeniedError as Blocked

        out["guards"] = [type(g).__name__ + ":" + g.action.value for g in guards]
        prefix = "guard"
    else:
        a = adapters.Adapter("traceverde", "default", "P1")
        adapters.setup_traceverde(a, extra_env=ec.EVAL_CONFIGS[config])
        from genai_otel.exceptions import PolicyViolationError as Blocked
        from genai_otel.instrumentors.base import BaseInstrumentor as B

        out["env"] = ec.EVAL_CONFIGS[config]
        out["blocking_policies"] = sorted(B._blocking_policies)
        out["detectors_registered"] = sorted(
            name
            for name in (
                "pii",
                "toxicity",
                "bias",
                "prompt_injection",
                "restricted_topics",
                "hallucination",
            )
            if getattr(B, "_%s_detector" % name) is not None
        )
        prefix = "evaluation."
    out["setup_s"] = time.perf_counter() - t0
    out["setup_cpu_s"] = time.process_time() - cpu0
    out["stages"]["after_setup"] = mem()
    out["versions"] = common.installed_versions()
    if "torch" in sys.modules:
        import torch

        out["torch"] = {
            "version": torch.__version__,
            "num_threads": torch.get_num_threads(),
            "cuda_available": bool(torch.cuda.is_available()),
        }

    blocked = (Blocked,)
    wl = {k: ec.workload(*k) for k in workloads}
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def sync_client(body):
        return openai.OpenAI(**ec.client_kwargs(body, sync=True))

    def async_client(body):
        return openai.AsyncOpenAI(**ec.client_kwargs(body, sync=False))

    def async_client_with(handler):
        import httpx

        return openai.AsyncOpenAI(
            api_key=common.DUMMY_KEY,
            base_url=common.BASE_URL,
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

    first = wl[workloads[0]]
    t0 = time.perf_counter()
    try:
        sync_client(first["body"]).chat.completions.create(**first["req"])
    except blocked:
        pass
    out["first_call_s"] = time.perf_counter() - t0
    out["stages"]["after_first_call"] = mem()

    out["cells"] = {}
    for key in workloads:
        w = wl[key]
        name = "%s_%s" % key
        cell = {"prompt_chars": len(w["prompt"]), "response_chars": len(w["response"])}
        cell["sync"] = cells.sync_cell(sync_client(w["body"]), w["req"], blocked, budget_s=budget)
        cell["async"] = cells.async_cell(
            loop,
            async_client(w["body"]),
            async_client_with,
            w["body"],
            w["req"],
            blocked,
            budget_s=budget,
        )
        out["cells"][name] = cell
        print(
            "%-14s %-8s sync median %10.1f us  p99 %10.1f  cpu %10.1f us  blocked %d | async median %10.1f  loop hold max %10.1f us"
            % (
                config,
                name,
                cell["sync"]["median"],
                cell["sync"]["p99"],
                cell["sync"]["cpu_us_per_call"],
                cell["sync"]["blocked"],
                cell["async"]["median"],
                cell["async"]["loop_hold"]["max_us"],
            )
        )
        sys.stdout.flush()
    out["stages"]["after_cells"] = mem()

    out["concurrency"] = {}
    for key in workloads:
        if key[0] != ("M" if "M" in sizes else sizes[0]):
            continue
        w = wl[key]
        name = "%s_%s" % key
        est = out["cells"][name]["sync"]["median"] / 1e6
        out["concurrency"][name] = {
            "threads8": cells.threads_cell(
                sync_client(w["body"]), w["req"], blocked, est, budget_s=budget * 2
            ),
            "gather50": cells.gather_cell(
                loop,
                async_client(w["body"]),
                w["req"],
                blocked,
                out["cells"][name]["async"]["median"] / 1e6,
                budget_s=budget * 2,
            ),
            "single_thread_calls_per_s": 1e6 / out["cells"][name]["sync"]["median"],
        }
    out["stages"]["after_concurrency"] = mem()

    from opentelemetry import trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    store = InMemorySpanExporter()
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(store))
    tracer = trace.get_tracer("bench.app")
    out["span"] = {}
    for key in workloads:
        w = wl[key]
        res = {}
        for mode in ("sync", "async"):
            store.clear()
            # An application span around the call: OpenLIT attaches guard events to it.
            with tracer.start_as_current_span("app.request"):
                try:
                    if mode == "sync":
                        sync_client(w["body"]).chat.completions.create(**w["req"])
                    else:
                        loop.run_until_complete(
                            async_client(w["body"]).chat.completions.create(**w["req"])
                        )
                except blocked as e:
                    res[mode + "_raised"] = "%s: %s" % (type(e).__name__, str(e)[:120])
            res[mode] = cells.span_summary(list(store.get_finished_spans()), prefix)
        out["span"]["%s_%s" % key] = res
    out["stages"]["end"] = mem()
    out["egress_attempts"] = common.egress_summary()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
