"""genai-otel must work in an application that does not have `requests` installed.

`requests` is not a declared dependency (only an optional extra), but the Ollama
server-metrics poller imported it at module import time. The Ollama instrumentor imports
the poller, `auto_instrument` imports the instrumentors, so `import genai_otel` raised
`ModuleNotFoundError: requests` and an application without `requests` got no
instrumentation at all. Found on the TraceVerse session-aggregator, which ran without
self-monitoring for two weeks because of it.

Each check runs in a subprocess so blocking `requests` cannot leak into other tests.
"""

import subprocess
import sys
import textwrap


def _run(code: str) -> subprocess.CompletedProcess:
    # The OTLP HTTP exporter is loaded first: older exporter releases import `requests`
    # themselves (newer ones do not), and that is the exporter's dependency, not ours.
    # Everything genai-otel imports after this point must work without `requests`.
    prelude = (
        "import sys\n"
        "import opentelemetry.exporter.otlp.proto.http.trace_exporter\n"
        "import opentelemetry.exporter.otlp.proto.http.metric_exporter\n"
        "sys.modules['requests'] = None  # behave as if requests is not installed\n"
    )
    return subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_import_genai_otel_without_requests():
    proc = _run("import genai_otel; print('ok')")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "ok" in proc.stdout


def test_instrument_without_requests():
    proc = _run(
        """
        import os
        os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'] = 'http://127.0.0.1:9'
        os.environ['GENAI_ENABLE_GPU_METRICS'] = 'false'
        os.environ['GENAI_FAIL_ON_ERROR'] = 'true'
        import genai_otel
        genai_otel.instrument(service_name='no-requests-app')
        print('instrumented')
        """
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "instrumented" in proc.stdout


def test_ollama_poller_without_requests_does_not_start_and_says_why():
    proc = _run(
        """
        import logging
        logging.basicConfig(level=logging.WARNING)
        from genai_otel.instrumentors import ollama_server_metrics_poller as p
        poller = p.start_ollama_metrics_poller(interval=60)
        print('running', poller._running)
        p.stop_ollama_metrics_poller()
        """
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "running False" in proc.stdout
    assert "requests" in proc.stderr  # one warning naming what is missing
