"""Per-library setup.

setup(library, config, mode) performs the library's instrumentation exactly once
in this process and returns an Adapter that hands out instrumented OpenAI clients.

Modes
-----
P0   no exporting span processor: the wrapper's own work, plus any non-exporting
     processors the library installs for itself.
P1   BatchSpanProcessor (SDK defaults) feeding an exporter that discards.
INSPECT  SimpleSpanProcessor + in-memory exporter, used by `--inspect` to list
     what a configuration records.

Libraries that are not plain OpenTelemetry instrumentations are mapped onto the
closest equivalent; each adapter says how in `Adapter.notes`, and that text is
copied into the result file.

To add a library: write a setup_<name>(adapter) function, add its configurations
and requirement file, and register it in SETUPS at the bottom of this file.
"""

import os
import sys

from . import common

HERE = os.path.dirname(os.path.abspath(__file__))
WORKDIR = os.environ.get("BENCH_WORKDIR") or os.getcwd()


class Adapter:
    def __init__(self, library, config, mode):
        self.library = library
        self.config = config
        self.mode = mode
        self.notes = []
        self.drain = None
        self.unsupported = {}  # scenario -> reason
        self.span_store = None
        self.metric_reader = None
        self.payloads = []
        self._wrap_client = None

    def make(self, kind):
        import openai

        cls = openai.OpenAI if kind.startswith("sync") else openai.AsyncOpenAI
        client = cls(**common.client_kwargs(kind))
        return self._wrap_client(client) if self._wrap_client is not None else client


def _assert_in_venv(module):
    path = os.path.abspath(module.__file__)
    if os.path.abspath(sys.prefix) not in path:
        raise SystemExit(
            "%s was not imported from the benchmark environment: %s" % (module.__name__, path)
        )


def _tracer_provider(a, set_global=True):
    """The tracer provider every OpenTelemetry-based library is handed, per mode."""
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    tp = TracerProvider(resource=Resource.create({"service.name": "bench"}))
    if a.mode == "P1":
        tp.add_span_processor(BatchSpanProcessor(common.make_noop_span_exporter()()))
    elif a.mode == "INSPECT":
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        a.span_store = InMemorySpanExporter()
        tp.add_span_processor(SimpleSpanProcessor(a.span_store))
    if set_global:
        trace.set_tracer_provider(tp)
    a.drain = tp.force_flush
    return tp


def _meter_provider(a):
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader, PeriodicExportingMetricReader

    if a.mode == "INSPECT":
        a.metric_reader = reader = InMemoryMetricReader()
    else:
        reader = PeriodicExportingMetricReader(common.make_noop_metric_exporter()())
    metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))


# ---------------------------------------------------------------------------
# TraceVerde (genai-otel-instrument)
# ---------------------------------------------------------------------------

TRACEVERDE_FORCED_OFF = [
    "GENAI_ENABLE_GPU_METRICS",
    "GENAI_ENABLE_CO2_TRACKING",
    "GENAI_ENABLE_PII_DETECTION",
    "GENAI_ENABLE_TOXICITY_DETECTION",
    "GENAI_ENABLE_BIAS_DETECTION",
    "GENAI_ENABLE_PROMPT_INJECTION_DETECTION",
    "GENAI_ENABLE_RESTRICTED_TOPICS",
    "GENAI_ENABLE_HALLUCINATION_DETECTION",
    "GENAI_ENABLE_MCP_INSTRUMENTATION",
]
TRACEVERDE_CONFIGS = {
    "default": ({}, "tokens, cost, metrics; no prompt or response text"),
    "content_on": (
        {"GENAI_ENABLE_CONTENT_CAPTURE": "true"},
        "as default plus prompt/response text, each value cut at 200 characters",
    ),
    "content_on_full": (
        {"GENAI_ENABLE_CONTENT_CAPTURE": "true", "GENAI_CONTENT_MAX_LENGTH": "0"},
        "as default plus prompt/response text, uncut",
    ),
    "cost_off": (
        {"GENAI_ENABLE_COST_TRACKING": "false"},
        "as default without cost attributes and the cost metric",
    ),
    "lean": (
        {"GENAI_ENABLE_COST_TRACKING": "false", "GENAI_ENABLE_CONCURRENCY_METRICS": "false"},
        "as cost_off without the running-requests gauge",
    ),
}


def setup_traceverde(a, extra_env=None):
    for name in TRACEVERDE_FORCED_OFF:
        os.environ[name] = "false"
    os.environ["GENAI_ENABLED_INSTRUMENTORS"] = "openai"
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = ""
    env = dict(TRACEVERDE_CONFIGS[a.config][0]) if a.config in TRACEVERDE_CONFIGS else {}
    env.update(extra_env or {})
    os.environ.update(env)

    import genai_otel

    _assert_in_venv(genai_otel)
    import genai_otel.auto_instrument as ai

    # instrument() builds its own providers. With no OTLP endpoint it takes the
    # console branch; the console exporters are swapped for discarding ones.
    ai.ConsoleMetricExporter = common.make_noop_metric_exporter()
    if a.mode == "INSPECT":
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        a.span_store = InMemorySpanExporter()
        ai.ConsoleSpanExporter = lambda: a.span_store
        ai.BatchSpanProcessor = SimpleSpanProcessor
        a.metric_reader = InMemoryMetricReader()
        ai.PeriodicExportingMetricReader = lambda exporter=None, **kw: a.metric_reader
    else:
        ai.ConsoleSpanExporter = common.make_noop_span_exporter()
        if a.mode == "P0":
            ai.BatchSpanProcessor = common.make_no_export_processor()
    genai_otel.instrument()

    from opentelemetry import trace

    tp = trace.get_tracer_provider()
    a.drain = tp.force_flush
    a.notes.append(
        "genai_otel.instrument() with OTEL_EXPORTER_OTLP_ENDPOINT='' and env %s. GPU, CO2, MCP and "
        "all evaluation features forced off unless the configuration turns them on. Its own "
        "non-exporting span processors stay in place in both modes." % (env,)
    )


# ---------------------------------------------------------------------------
# OpenLIT
# ---------------------------------------------------------------------------

OPENLIT_CONFIGS = {
    "default": ({}, "content, cost, metrics and log events (its defaults)"),
    "content_off": ({"capture_message_content": False}, "as default without message content"),
    "lean": (
        {"capture_message_content": False, "disable_metrics": True, "disable_events": True},
        "no content, no metrics, no log events",
    ),
}


def _openlit_pricing_path():
    # OpenLIT's default downloads its pricing table at init. A one-model table in the
    # same format is shipped here so no network is needed; the lookup is a dict get
    # either way. The path must be relative: OpenLIT treats anything urlparse() gives
    # a scheme as a URL, and a Windows drive letter counts as one.
    src = os.path.join(HERE, "data", "openlit_pricing_minimal.json")
    try:
        rel = os.path.relpath(src)
    except ValueError:
        rel = None
    if not rel or ":" in rel:
        import shutil

        rel = "openlit_pricing_minimal.json"
        shutil.copyfile(src, rel)
    return rel


def setup_openlit(a, extra_kwargs=None):
    _tracer_provider(a)
    _meter_provider(a)
    from opentelemetry import _logs
    from opentelemetry.sdk._logs import LoggerProvider

    lp = LoggerProvider()
    if a.mode == "P1":
        from opentelemetry.sdk._logs.export import (
            BatchLogRecordProcessor,
            LogRecordExporter,
            LogRecordExportResult,
        )

        class NoOpLogExporter(LogRecordExporter):
            def export(self, batch):
                return LogRecordExportResult.SUCCESS

            def shutdown(self):
                return None

            def force_flush(self, timeout_millis=30000):
                return True

        lp.add_log_record_processor(BatchLogRecordProcessor(NoOpLogExporter()))
    _logs.set_logger_provider(lp)

    import openlit

    _assert_in_venv(openlit)
    kwargs = dict(OPENLIT_CONFIGS[a.config][0]) if a.config in OPENLIT_CONFIGS else {}
    kwargs.update(extra_kwargs or {})
    openlit.init(service_name="bench", pricing_json=_openlit_pricing_path(), **kwargs)
    a.notes.append(
        "openlit.init(pricing_json=<bundled one-model table>, %s). Tracer, meter and logger "
        "providers are created by the harness and set globally first; OpenLIT reuses them."
        % ({k: v for k, v in kwargs.items() if k != "guards"},)
    )


# ---------------------------------------------------------------------------
# Traceloop / OpenLLMetry
# ---------------------------------------------------------------------------

TRACELOOP_CONFIGS = {
    "default": ({}, "content and metrics (its defaults)"),
    "content_off": ({"TRACELOOP_TRACE_CONTENT": "false"}, "as default without message content"),
}


def setup_traceloop(a):
    os.environ.update(TRACELOOP_CONFIGS[a.config][0])
    os.environ["TRACELOOP_TELEMETRY"] = "false"
    import traceloop.sdk as tsdk
    from traceloop.sdk import Traceloop

    _assert_in_venv(tsdk)
    kw = dict(
        app_name="bench",
        telemetry_enabled=False,
        metrics_exporter=common.make_noop_metric_exporter()(),
    )
    if a.mode == "INSPECT":
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        a.span_store = InMemorySpanExporter()
        kw["processor"] = SimpleSpanProcessor(a.span_store)
    elif a.mode == "P0":
        kw["processor"] = common.make_no_export_processor()()
    else:
        kw["exporter"] = common.make_noop_span_exporter()()
    Traceloop.init(**kw)
    from opentelemetry import trace

    a.drain = trace.get_tracer_provider().force_flush
    a.notes.append(
        "Traceloop.init(app_name, %s, metrics_exporter=<discarding>), env %s. Its sync wrapper "
        "calls asyncio.run() once per call; on Windows the runner reduces the sync volume for "
        "this library, see the README."
        % (
            "processor=<do-nothing>" if a.mode == "P0" else "exporter=<discarding>",
            TRACELOOP_CONFIGS[a.config][0],
        )
    )


# ---------------------------------------------------------------------------
# OpenInference (the instrumentor Arize Phoenix installs)
# ---------------------------------------------------------------------------

OPENINFERENCE_CONFIGS = {
    "default": ({}, "inputs, outputs and message content (its defaults); no metrics"),
    "content_off": (
        {"hide_inputs": True, "hide_outputs": True},
        "content replaced by a redaction marker",
    ),
}


def setup_openinference(a):
    tp = _tracer_provider(a)
    import openinference.instrumentation.openai as oio
    from openinference.instrumentation import TraceConfig
    from openinference.instrumentation.openai import OpenAIInstrumentor

    _assert_in_venv(oio)
    cfg = OPENINFERENCE_CONFIGS[a.config][0]
    if cfg:
        OpenAIInstrumentor().instrument(tracer_provider=tp, config=TraceConfig(**cfg))
    else:
        OpenAIInstrumentor().instrument(tracer_provider=tp)
    a.notes.append(
        "OpenAIInstrumentor().instrument(tracer_provider=<harness provider>, TraceConfig(%s)). "
        "phoenix.otel.register() is not used: its default exports synchronously to a collector."
        % (cfg,)
    )


# ---------------------------------------------------------------------------
# Langfuse
# ---------------------------------------------------------------------------

LANGFUSE_CONFIGS = {
    "default": (
        {},
        "input, output, model parameters, usage (its defaults); no content switch exists "
        "for the OpenAI integration",
    ),
}


def setup_langfuse(a):
    import langfuse
    from langfuse import Langfuse
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

    _assert_in_venv(langfuse)
    tp = TracerProvider(resource=Resource.create({"service.name": "bench"}))
    if a.mode == "INSPECT":
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        a.span_store = exporter = InMemorySpanExporter()
    else:
        exporter = common.make_noop_span_exporter()()
    client = Langfuse(
        public_key="pk-lf-bench-dummy",
        secret_key="sk-lf-bench-dummy",
        base_url="http://127.0.0.1:9",
        tracer_provider=tp,
        span_exporter=exporter,
    )
    if a.mode == "P0":
        for processor in tp._active_span_processor._span_processors:
            processor.on_end = lambda span: None
    import langfuse.openai  # noqa: F401  patches the OpenAI resource classes

    a.drain = client.flush
    a.notes.append(
        "Langfuse(dummy keys, base_url=http://127.0.0.1:9, tracer_provider=<fresh>, "
        "span_exporter=<discarding>) then `import langfuse.openai`. P0 replaces its span "
        "processor's on_end with a no-op; P1 runs it unmodified."
    )


# ---------------------------------------------------------------------------
# Opik
# ---------------------------------------------------------------------------

OPIK_CONFIGS = {
    "default": (
        {},
        "input, output, usage, metadata (its defaults); track_openai() has no content switch",
    ),
}


def setup_opik(a):
    import httpx

    empty = os.path.join(WORKDIR, "empty_opik.config")
    if not os.path.exists(empty):
        open(empty, "w", encoding="utf-8").close()
    os.environ.update(
        OPIK_URL_OVERRIDE="http://127.0.0.1:9/api/",
        OPIK_SENTRY_ENABLE="false",
        OPIK_ANALYTICS_ENABLE="false",
        OPIK_CONFIG_PATH=empty,
        OPIK_PROJECT_NAME="bench",
    )
    import opik
    import opik.hooks

    _assert_in_venv(opik)

    def backend(request):
        common.ACTIVITY["backend_requests"] += 1
        if a.mode == "INSPECT":
            a.payloads.append((request.method + " " + request.url.path, request.content))
        return httpx.Response(200, json={}) if request.method == "GET" else httpx.Response(204)

    opik.hooks.add_httpx_client_hook(
        opik.hooks.HttpxClientHook(
            client_modifier=None, client_init_arguments={"transport": httpx.MockTransport(backend)}
        )
    )
    from opik.integrations.openai import track_openai

    a._wrap_client = track_openai
    client = opik.get_global_client()
    if a.mode == "P0":
        client._streamer.put = lambda message, force=False: None
    a.drain = opik.flush_tracker
    a.notes.append(
        "track_openai(client). Sentry and product analytics disabled by environment. The SDK's "
        "REST client gets an in-process httpx.MockTransport through its documented client hook, "
        "so batching, JSON encoding and gzip run and the send is answered locally. P0 drops the "
        "messages before the background queue; P1 runs the whole pipeline."
    )


# ---------------------------------------------------------------------------
# Galileo (experimental: needs a private-cache injection to run without an account)
# ---------------------------------------------------------------------------

GALILEO_CONFIGS = {
    "default": ({}, "input and output messages, usage; sync Completions.create only"),
}


def setup_galileo(a):
    os.environ.update(GALILEO_PROJECT="bench", GALILEO_LOG_STREAM="bench")
    import galileo
    from galileo import galileo_context
    from galileo.utils.singleton import GalileoLoggerSingleton

    _assert_in_venv(galileo)

    def hook(request):
        common.ACTIVITY["hook_traces"] += len(request.traces)
        if a.mode == "INSPECT":
            a.payloads.append(("ingestion_hook", request.model_dump_json().encode()))

    logger = galileo_context.get_logger_instance(
        project="bench", log_stream="bench", ingestion_hook=hook
    )
    cache = GalileoLoggerSingleton()._galileo_loggers
    for key in list(cache):
        cache[key[:-1]] = cache[key]
    import galileo.openai  # noqa: F401

    a.drain = logger.flush
    reason = "not instrumented: galileo.openai wraps only the sync Completions.create"
    a.unsupported = {
        "async_seq": reason,
        "async_gather10": reason,
        "async_stream": reason,
        "sync_stream": "skipped: the streaming wrapper raised into the caller when the "
        "stream ended with a usage chunk (galileo 2.6.0)",
    }
    a.notes.append(
        "`import galileo.openai`. No account: a GalileoLogger built with the documented "
        "ingestion_hook is placed in the SDK's private logger cache under the default key, "
        "because the OpenAI wrapper cannot be given a hook through public API. Experimental."
    )


SETUPS = {
    "traceverde": (setup_traceverde, TRACEVERDE_CONFIGS),
    "openlit": (setup_openlit, OPENLIT_CONFIGS),
    "traceloop": (setup_traceloop, TRACELOOP_CONFIGS),
    "openinference": (setup_openinference, OPENINFERENCE_CONFIGS),
    "langfuse": (setup_langfuse, LANGFUSE_CONFIGS),
    "opik": (setup_opik, OPIK_CONFIGS),
    "galileo": (setup_galileo, GALILEO_CONFIGS),
}
# Run only when named explicitly.
EXPERIMENTAL = {"galileo"}
# Modes each library is run in by default.
DEFAULT_MODES = {"galileo": ["P1"]}


def describe(library, config):
    return SETUPS[library][1][config][1]


def setup(library, config, mode):
    a = Adapter(library, config, mode)
    if library == "none":
        a.notes.append("control: no instrumentation; both phases run the bare SDK")
        return a
    fn, configs = SETUPS[library]
    if config not in configs:
        raise SystemExit(
            "unknown configuration %r for %s; known: %s" % (config, library, sorted(configs))
        )
    fn(a)
    return a
