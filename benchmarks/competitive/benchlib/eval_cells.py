"""Measurement cells for the evaluation-detector benchmark."""

import asyncio
import gc
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import common
from . import eval_common as ec

pc = time.perf_counter_ns


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _pick_n(est_s, budget_s, nmin, nmax):
    """Calls per cell. Slow calls get fewer samples so a cell stays within seconds;
    the count is recorded with every result, and percentiles from fewer than 100
    samples should be read as 'worst of N'."""
    if est_s > 0.3:
        nmin = 20
    elif est_s > 0.05:
        nmin = 30
    return _clamp(int(budget_s / est_s), nmin, nmax)


def sync_cell(client, req, blocked_exc, budget_s=3.0, nmin=60, nmax=3000):
    """Per-call latency samples, CPU time and thread count for one workload, sync."""
    create = client.chat.completions.create
    blocked = [0]

    def one():
        t0 = pc()
        try:
            create(**req)
        except blocked_exc:
            blocked[0] += 1
        return pc() - t0

    warm = [one() for _ in range(5)]
    est = statistics.median(warm) / 1e9
    n = _pick_n(est, budget_s, nmin, nmax)
    blocked[0] = 0
    gc.collect()
    threads_before = ec.os_thread_count()
    cpu0, w0 = time.process_time(), time.perf_counter()
    lat = [one() for _ in range(n)]
    cpu, wall = time.process_time() - cpu0, time.perf_counter() - w0
    out = ec.stats_us(lat)
    out.update(
        first_call_us=warm[0] / 1000.0,
        cpu_us_per_call=cpu / n * 1e6,
        wall_us_per_call=wall / n * 1e6,
        cores_used=(cpu / wall) if wall > 0 else None,
        blocked=blocked[0],
        os_threads_before=threads_before,
        os_threads_after=ec.os_thread_count(),
    )
    return out


async def _async_latency(client, req, blocked_exc, budget_s, nmin, nmax):
    create = client.chat.completions.create
    blocked = 0

    async def one():
        nonlocal blocked
        t0 = pc()
        try:
            await create(**req)
        except blocked_exc:
            blocked += 1
        return pc() - t0

    warm = [await one() for _ in range(5)]
    est = statistics.median(warm) / 1e9
    n = _pick_n(est, budget_s, nmin, nmax)
    blocked = 0
    gc.collect()
    cpu0, w0 = time.process_time(), time.perf_counter()
    lat = [await one() for _ in range(n)]
    cpu, wall = time.process_time() - cpu0, time.perf_counter() - w0
    out = ec.stats_us(lat)
    out.update(
        first_call_us=warm[0] / 1000.0,
        cpu_us_per_call=cpu / n * 1e6,
        wall_us_per_call=wall / n * 1e6,
        cores_used=(cpu / wall) if wall > 0 else None,
        blocked=blocked,
    )
    return out


async def _loop_lag(make_async_client, body, req, blocked_exc, n_calls):
    """How long does one call hold the event loop?

    The fake backend yields once (as real network I/O would), and a ticker task
    that does nothing but `await asyncio.sleep(0)` records the gap between its own
    turns. A gap is one uninterrupted stretch of somebody else's code on the loop.
    """
    import httpx

    async def handler(request):
        await asyncio.sleep(0)
        return httpx.Response(200, headers=common.JSON_HEADERS, content=body)

    client = make_async_client(handler)
    create = client.chat.completions.create
    gaps = []
    stop = False

    async def ticker():
        last = pc()
        while not stop:
            await asyncio.sleep(0)
            now = pc()
            gaps.append(now - last)
            last = now

    for _ in range(3):
        try:
            await create(**req)
        except blocked_exc:
            pass
    task = asyncio.create_task(ticker())
    await asyncio.sleep(0)
    for _ in range(n_calls):
        try:
            await create(**req)
        except blocked_exc:
            pass
        await asyncio.sleep(0)
    stop = True
    await task
    v = sorted(g / 1000.0 for g in gaps)
    return {
        "ticks": len(v),
        "median_us": statistics.median(v),
        "p99_us": ec.pct(v, 99),
        "max_us": v[-1],
    }


def async_cell(
    loop, client, make_async_client, body, req, blocked_exc, budget_s=3.0, nmin=60, nmax=3000
):
    out = loop.run_until_complete(_async_latency(client, req, blocked_exc, budget_s, nmin, nmax))
    n_lag = _clamp(out["n"] // 4, 10, 200)
    out["loop_hold"] = loop.run_until_complete(
        _loop_lag(make_async_client, body, req, blocked_exc, n_lag)
    )
    return out


def threads_cell(client, req, blocked_exc, est_s, n_threads=8, budget_s=5.0):
    create = client.chat.completions.create
    total = _clamp(int(budget_s / max(est_s, 1e-6)), 40, 4000)
    per = max(1, total // n_threads)
    total = per * n_threads
    barrier = threading.Barrier(n_threads + 1)

    def worker():
        barrier.wait()
        for _ in range(per):
            try:
                create(**req)
            except blocked_exc:
                pass

    with ThreadPoolExecutor(n_threads) as ex:
        futs = [ex.submit(worker) for _ in range(n_threads)]
        time.sleep(0.05)
        cpu0, w0 = time.process_time(), time.perf_counter()
        barrier.wait()
        for f in futs:
            f.result()
        cpu, wall = time.process_time() - cpu0, time.perf_counter() - w0
    return {
        "threads": n_threads,
        "calls": total,
        "wall_s": wall,
        "calls_per_s": total / wall,
        "cpu_s": cpu,
        "cores_used": cpu / wall,
        "os_threads_after": ec.os_thread_count(),
    }


async def _gather(client, req, blocked_exc, rounds, width):
    create = client.chat.completions.create

    async def one():
        try:
            await create(**req)
        except blocked_exc:
            pass

    await asyncio.gather(*[one() for _ in range(width)])
    cpu0, w0 = time.process_time(), time.perf_counter()
    for _ in range(rounds):
        await asyncio.gather(*[one() for _ in range(width)])
    cpu, wall = time.process_time() - cpu0, time.perf_counter() - w0
    return {
        "width": width,
        "rounds": rounds,
        "calls": rounds * width,
        "wall_s": wall,
        "calls_per_s": rounds * width / wall,
        "cpu_s": cpu,
        "cores_used": cpu / wall,
    }


def gather_cell(loop, client, req, blocked_exc, est_s, width=50, budget_s=4.0):
    rounds = _clamp(int(budget_s / max(est_s * width, 1e-6)), 2, 60)
    return loop.run_until_complete(_gather(client, req, blocked_exc, rounds, width))


def span_summary(spans, prefix):
    """Counts and sizes of what a detector layer added to the spans of one call."""
    out = {
        "n_spans": len(spans),
        "attrs_total": 0,
        "attr_chars_total": 0,
        "added_attrs": 0,
        "added_attr_chars": 0,
        "added_names": [],
        "events_total": 0,
        "event_chars_total": 0,
        "added_events": 0,
        "otlp_bytes": None,
    }
    for s in spans:
        attrs = dict(s.attributes or {})
        out["attrs_total"] += len(attrs)
        for k, v in attrs.items():
            size = len(str(k)) + len(str(v))
            out["attr_chars_total"] += size
            if str(k).startswith(prefix):
                out["added_attrs"] += 1
                out["added_attr_chars"] += size
                out["added_names"].append(k)
        for ev in s.events:
            out["events_total"] += 1
            size = len(ev.name) + sum(
                len(str(k)) + len(str(v)) for k, v in (ev.attributes or {}).items()
            )
            out["event_chars_total"] += size
            if ev.name.startswith(prefix) or "guard" in ev.name:
                out["added_events"] += 1
    try:
        from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans

        out["otlp_bytes"] = len(encode_spans(spans).SerializeToString()) if spans else 0
    except Exception:  # noqa: BLE001
        pass
    out["added_names"] = sorted(out["added_names"])
    return out
