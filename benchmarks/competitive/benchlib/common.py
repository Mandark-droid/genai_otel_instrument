"""Library-neutral pieces: fake OpenAI backend, no-op sinks, network guard, timing.

Everything in this module runs inside the virtual environment of the library
under test. It imports nothing from any tracing library.
"""

import asyncio
import gc
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# System facts, echoed into every result file
# ---------------------------------------------------------------------------


def cpu_model():
    try:
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if sys.platform == "darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or "unknown"


def installed_versions():
    from importlib.metadata import distributions

    out = {}
    for dist in distributions():
        name = dist.metadata["Name"] if dist.metadata else None
        if name:
            out[name.lower()] = dist.version
    return dict(sorted(out.items()))


def system_info():
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count(),
    }


def memory_mb():
    """Resident set size of this process in MB, or None where it cannot be read cheaply."""
    try:
        if sys.platform == "win32":
            import ctypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32)] + [
                    (n, ctypes.c_size_t)
                    for n in (
                        "PeakWorkingSetSize",
                        "WorkingSetSize",
                        "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage",
                        "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage",
                        "PagefileUsage",
                        "PeakPagefileUsage",
                        "PrivateUsage",
                    )
                ]

            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.K32GetProcessMemoryInfo.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(PMC),
                ctypes.c_uint32,
            ]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
            return pmc.WorkingSetSize / 1048576.0
        if os.path.exists("/proc/self/statm"):
            with open("/proc/self/statm", encoding="ascii") as f:
                pages = int(f.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE") / 1048576.0
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 1048576.0 if sys.platform == "darwin" else peak / 1024.0
    except Exception:  # noqa: BLE001
        return None


def pin_process(cpus):
    """Optionally restrict this process to a set of logical CPUs (list of ints)."""
    if not cpus:
        return "not pinned"
    try:
        if sys.platform == "win32":
            import ctypes

            mask = 0
            for c in cpus:
                mask |= 1 << c
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            ok = k32.SetProcessAffinityMask(k32.GetCurrentProcess(), mask)
            return "pinned to %s (%s)" % (cpus, "ok" if ok else "failed")
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(cpus))
            return "pinned to %s" % (cpus,)
    except Exception as e:  # noqa: BLE001
        return "pinning failed: %r" % (e,)
    return "pinning not supported on this platform"


# ---------------------------------------------------------------------------
# Network guard. Nothing may leave the machine during a measurement. Any attempt
# to resolve or connect to a non-loopback address is refused and recorded.
# Loopback stays open because the asyncio event loop needs it on some platforms.
# ---------------------------------------------------------------------------

EGRESS_ATTEMPTS = []


def _is_loopback(host):
    host = host.decode() if isinstance(host, bytes) else str(host)
    return host in ("localhost", "::1", "", "0.0.0.0") or host.startswith("127.")


def install_network_guard():
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_gai = socket.getaddrinfo

    def connect(self, address):
        if isinstance(address, tuple) and address and not _is_loopback(address[0]):
            EGRESS_ATTEMPTS.append(["connect", str(address[0]), address[1]])
            raise ConnectionRefusedError("blocked by the benchmark network guard")
        return real_connect(self, address)

    def connect_ex(self, address):
        if isinstance(address, tuple) and address and not _is_loopback(address[0]):
            EGRESS_ATTEMPTS.append(["connect_ex", str(address[0]), address[1]])
            return 111
        return real_connect_ex(self, address)

    def getaddrinfo(host, port, *args, **kwargs):
        if host is not None and not _is_loopback(host):
            EGRESS_ATTEMPTS.append(["dns", str(host), port])
            raise socket.gaierror(-2, "blocked by the benchmark network guard")
        return real_gai(host, port, *args, **kwargs)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.getaddrinfo = getaddrinfo


def egress_summary():
    out = {}
    for kind, host, port in EGRESS_ATTEMPTS:
        key = "%s %s:%s" % (kind, host, port)
        out[key] = out.get(key, 0) + 1
    return out


# ---------------------------------------------------------------------------
# The fake OpenAI backend
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a support assistant for a retail bank. Answer briefly and precisely, "
    "never reveal internal identifiers, and escalate to a human agent whenever the "
    "customer mentions fraud, a lost card or a disputed transaction."
)
USER_PROMPT = (
    "Hi, I moved to a new flat last month and I still receive my statements at the old "
    "address. I already changed the address in the mobile app twice, but the paper "
    "statement keeps going to the previous one. Could you explain why that happens, "
    "how long the change takes to apply to printed statements, and whether I can switch "
    "to electronic statements only so that this stops being a problem altogether?"
)
COMPLETION_TEXT = (
    "Printed statements are generated from a snapshot of your postal details taken a few "
    "days before the end of each billing cycle, so an address change made after that "
    "snapshot only shows up on the following statement. A change normally applies within "
    "one full cycle. You can switch to electronic statements under Settings, Documents, "
    "Delivery preferences; once that is enabled no paper statement is produced, and you "
    "will get a notification when each new statement is ready to download."
)
MODEL = "gpt-4o"
RESPONSE_MODEL = "gpt-4o-2024-08-06"
N_STREAM_CHUNKS = 50
DUMMY_KEY = "sk-bench-not-a-real-key"
BASE_URL = "https://api.openai.com/v1"
JSON_HEADERS = {"content-type": "application/json", "x-request-id": "req_benchfake"}
SSE_HEADERS = {"content-type": "text/event-stream", "x-request-id": "req_benchfake"}

# name -> description; selected with the BENCH_PAYLOAD environment variable
PAYLOADS = {
    "default": "one system and one user message (about 130 prompt tokens), about 100 completion tokens",
    "S": "about 50 tokens on each side",
    "M": "about 500 tokens on each side",
    "L": "about 4,000 tokens on each side",
    "conv20": "a 20-turn conversation (about 2,000 tokens of history) with a 500-token answer",
}


def _prose(n_chars, offset=0):
    fill = USER_PROMPT + " " + COMPLETION_TEXT + " "
    text = (fill * (n_chars // len(fill) + 2))[offset : offset + n_chars]
    return text.rsplit(" ", 1)[0] if " " in text else text


def _usage(prompt_tokens, completion_tokens):
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
        "completion_tokens_details": {
            "reasoning_tokens": 0,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    }


def build_payload(name):
    """Return (request kwargs, completion text, usage dict) for a payload name."""
    if name == "default":
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]
        return (
            dict(model=MODEL, messages=messages, temperature=0.2, max_tokens=256),
            COMPLETION_TEXT,
            _usage(131, 96),
        )
    size = {"S": 200, "M": 2000, "L": 16000, "conv20": 2000}[name]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if name == "conv20":
        for i in range(10):
            messages.append({"role": "user", "content": _prose(400, 7 * i)})
            messages.append({"role": "assistant", "content": _prose(400, 11 * i + 3)})
        messages.append({"role": "user", "content": _prose(400, 5)})
    else:
        messages.append({"role": "user", "content": _prose(size)})
    prompt_tokens = sum(len(m["content"]) for m in messages) // 4
    req = dict(model=MODEL, messages=messages, temperature=0.2, max_tokens=4096)
    return req, _prose(size, 17), _usage(prompt_tokens, size // 4)


PAYLOAD = os.environ.get("BENCH_PAYLOAD", "default")
REQ, _COMPLETION, USAGE = build_payload(PAYLOAD)
REQ_STREAM = dict(REQ, stream=True, stream_options={"include_usage": True})

CHAT_BODY = json.dumps(
    {
        "id": "chatcmpl-BenchFake0000000000000000001",
        "object": "chat.completion",
        "created": 1760000000,
        "model": RESPONSE_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": _COMPLETION,
                    "refusal": None,
                    "annotations": [],
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": USAGE,
        "service_tier": "default",
        "system_fingerprint": "fp_benchfake01",
    }
).encode()


def _build_sse():
    """50 server-sent events: role, 47 content deltas, finish, usage; then [DONE]."""
    base = {
        "id": "chatcmpl-BenchFake0000000000000000002",
        "object": "chat.completion.chunk",
        "created": 1760000000,
        "model": RESPONSE_MODEL,
        "service_tier": "default",
        "system_fingerprint": "fp_benchfake01",
    }
    words = _COMPLETION.split(" ")
    n_content = N_STREAM_CHUNKS - 3
    per = max(1, len(words) // n_content)
    events = []

    def event(choices, usage=None):
        d = dict(base)
        d["choices"] = choices
        d["usage"] = usage
        events.append(b"data: " + json.dumps(d).encode() + b"\n\n")

    event(
        [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "", "refusal": None},
                "logprobs": None,
                "finish_reason": None,
            }
        ]
    )
    for i in range(n_content):
        piece = words[i * per : (i + 1) * per] if i < n_content - 1 else words[i * per :]
        event(
            [
                {
                    "index": 0,
                    "delta": {"content": " ".join(piece) + " "},
                    "logprobs": None,
                    "finish_reason": None,
                }
            ]
        )
    event([{"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}])
    event([], USAGE)
    assert len(events) == N_STREAM_CHUNKS
    events.append(b"data: [DONE]\n\n")
    return b"".join(events)


SSE_BODY = _build_sse()


def client_kwargs(kind):
    """kwargs for openai.OpenAI / AsyncOpenAI wired to the in-process fake.

    kind is one of sync_json, sync_sse, async_json, async_sse.
    """
    import httpx

    body, headers = (CHAT_BODY, JSON_HEADERS) if kind.endswith("json") else (SSE_BODY, SSE_HEADERS)

    def handler(request):
        return httpx.Response(200, headers=headers, content=body)

    transport = httpx.MockTransport(handler)
    if kind.startswith("sync"):
        http_client = httpx.Client(transport=transport)
    else:
        http_client = httpx.AsyncClient(transport=transport)
    return dict(api_key=DUMMY_KEY, base_url=BASE_URL, http_client=http_client, max_retries=0)


def plain_clients():
    import openai

    return {
        kind: (openai.OpenAI if kind.startswith("sync") else openai.AsyncOpenAI)(
            **client_kwargs(kind)
        )
        for kind in ("sync_json", "sync_sse", "async_json", "async_sse")
    }


# ---------------------------------------------------------------------------
# No-op OpenTelemetry sinks. Every OpenTelemetry-based library gets exactly these.
# ---------------------------------------------------------------------------

# Proof that the instrumentation under test really ran during the timed phase.
ACTIVITY = {"spans_exported": 0, "backend_requests": 0, "hook_traces": 0}


def make_noop_span_exporter():
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    class NoOpSpanExporter(SpanExporter):
        def export(self, spans):
            ACTIVITY["spans_exported"] += len(spans)
            return SpanExportResult.SUCCESS

        def shutdown(self):
            return None

        def force_flush(self, timeout_millis=30000):
            return True

    return NoOpSpanExporter


def make_noop_metric_exporter():
    from opentelemetry.sdk.metrics.export import MetricExporter, MetricExportResult

    class NoOpMetricExporter(MetricExporter):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def export(self, metrics_data, timeout_millis=10000, **kwargs):
            return MetricExportResult.SUCCESS

        def force_flush(self, timeout_millis=10000):
            return True

        def shutdown(self, timeout_millis=30000, **kwargs):
            return None

    return NoOpMetricExporter


def make_no_export_processor():
    from opentelemetry.sdk.trace import SpanProcessor

    class NoExportProcessor(SpanProcessor):
        """Does nothing: the 'no exporting processor' mode."""

        def __init__(self, *args, **kwargs):
            pass

    return NoExportProcessor


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

# name -> (client kind, calls per repetition, warm-up calls)
SCENARIOS = {
    "sync": ("sync_json", 2000, 400),
    "async_seq": ("async_json", 2000, 400),
    "async_gather10": ("async_json", 2000, 400),
    "sync_stream": ("sync_sse", 400, 80),
    "async_stream": ("async_sse", 400, 80),
}
ALL_SCENARIOS = ["sync", "async_seq", "async_gather10", "sync_stream", "async_stream"]


def _run_sync(client, n):
    create, req = client.chat.completions.create, REQ
    t0 = time.perf_counter_ns()
    for _ in range(n):
        create(**req)
    return time.perf_counter_ns() - t0


def _run_sync_stream(client, n):
    create, req = client.chat.completions.create, REQ_STREAM
    t0 = time.perf_counter_ns()
    for _ in range(n):
        for _chunk in create(**req):
            pass
    return time.perf_counter_ns() - t0


async def _run_async_seq(client, n):
    create, req = client.chat.completions.create, REQ
    t0 = time.perf_counter_ns()
    for _ in range(n):
        await create(**req)
    return time.perf_counter_ns() - t0


async def _run_async_gather(client, n):
    create, req = client.chat.completions.create, REQ
    t0 = time.perf_counter_ns()
    for _ in range(n // 10):
        await asyncio.gather(*[create(**req) for _ in range(10)])
    return time.perf_counter_ns() - t0


async def _run_async_stream(client, n):
    create, req = client.chat.completions.create, REQ_STREAM
    t0 = time.perf_counter_ns()
    for _ in range(n):
        stream = await create(**req)
        async for _chunk in stream:
            pass
    return time.perf_counter_ns() - t0


_LOOP = None


def loop():
    global _LOOP
    if _LOOP is None:
        _LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_LOOP)
    return _LOOP


def run_once(scenario, client, n):
    """Run n calls of a scenario; return elapsed nanoseconds."""
    if scenario == "sync":
        return _run_sync(client, n)
    if scenario == "sync_stream":
        return _run_sync_stream(client, n)
    if scenario == "async_seq":
        return loop().run_until_complete(_run_async_seq(client, n))
    if scenario == "async_gather10":
        return loop().run_until_complete(_run_async_gather(client, n))
    if scenario == "async_stream":
        return loop().run_until_complete(_run_async_stream(client, n))
    raise ValueError(scenario)


def measure(scenario, client, drain=None, reps=5, scale=1.0):
    """Warm up, then time `reps` repetitions; return per-call microseconds.

    `drain` is the library's flush. It is timed separately from the calls, so the
    inline cost a caller waits for and the deferred cost of pushing the telemetry
    out are both visible instead of one hiding in the other.
    """
    _kind, n, warm = SCENARIOS[scenario]
    n = max(10, int(n * scale))
    warm = max(10, int(warm * scale))
    run_once(scenario, client, warm)
    if drain:
        drain()
    inline, deferred = [], []
    for _ in range(reps):
        gc.collect()
        ns = run_once(scenario, client, n)
        inline.append(ns / n / 1000.0)
        if drain:
            t0 = time.perf_counter_ns()
            drain()
            deferred.append((time.perf_counter_ns() - t0) / n / 1000.0)
    out = {
        "n": n,
        "reps": reps,
        "us_min": min(inline),
        "us_median": statistics.median(inline),
        "us_max": max(inline),
        "us_all": [round(x, 3) for x in inline],
    }
    if deferred:
        out["drain_us_min"] = min(deferred)
        out["drain_us_median"] = statistics.median(deferred)
    return out


def spin(seconds=1.0):
    """Burn CPU briefly so the first timed block does not sit on a clock ramp."""
    end = time.perf_counter() + seconds
    x = 0
    while time.perf_counter() < end:
        x += 1
    return x
