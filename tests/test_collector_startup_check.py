"""Tests for the default-collector reachability check.

A first-time user has no collector, so the exporter's retry loop buries them in
connection stack traces with no indication of what to do. This adds one
actionable line - and deliberately does not quiet the retries, since a silent
export failure is how telemetry disappears unnoticed.
"""

import socket
from unittest.mock import MagicMock, patch

import pytest

from genai_otel.auto_instrument import DEFAULT_OTLP_ENDPOINT, _warn_if_default_collector_unreachable
from genai_otel.config import OTelConfig


def _cfg(endpoint):
    c = OTelConfig(service_name="t")
    c.endpoint = endpoint
    return c


@pytest.fixture
def no_collector():
    with patch("socket.create_connection", side_effect=OSError("refused")):
        yield


@pytest.fixture
def collector_up():
    with patch("socket.create_connection", return_value=MagicMock()):
        yield


def test_warns_on_default_endpoint_with_no_collector(no_collector, monkeypatch):
    monkeypatch.delenv("GENAI_SKIP_COLLECTOR_CHECK", raising=False)
    assert _warn_if_default_collector_unreachable(_cfg(DEFAULT_OTLP_ENDPOINT)) is True


def test_silent_when_collector_is_reachable(collector_up, monkeypatch):
    monkeypatch.delenv("GENAI_SKIP_COLLECTOR_CHECK", raising=False)
    assert _warn_if_default_collector_unreachable(_cfg(DEFAULT_OTLP_ENDPOINT)) is False


def test_silent_for_a_configured_endpoint(no_collector, monkeypatch):
    """A non-default endpoint is a deliberate choice; unreachability there is
    the operator's business, and the exporter already reports it."""
    monkeypatch.delenv("GENAI_SKIP_COLLECTOR_CHECK", raising=False)
    assert _warn_if_default_collector_unreachable(_cfg("http://collector:4318")) is False


def test_silent_when_exporting_to_console(no_collector, monkeypatch):
    monkeypatch.delenv("GENAI_SKIP_COLLECTOR_CHECK", raising=False)
    assert _warn_if_default_collector_unreachable(_cfg("")) is False


def test_opt_out_is_honoured(no_collector, monkeypatch):
    monkeypatch.setenv("GENAI_SKIP_COLLECTOR_CHECK", "true")
    assert _warn_if_default_collector_unreachable(_cfg(DEFAULT_OTLP_ENDPOINT)) is False


def test_probe_is_bounded(no_collector, monkeypatch):
    """The probe must not stall startup when the host blackholes packets."""
    monkeypatch.delenv("GENAI_SKIP_COLLECTOR_CHECK", raising=False)
    with patch("socket.create_connection", side_effect=OSError) as sock:
        _warn_if_default_collector_unreachable(_cfg(DEFAULT_OTLP_ENDPOINT))
    assert sock.call_args.kwargs["timeout"] <= 1.0


def test_message_says_what_to_do(no_collector, monkeypatch, caplog):
    monkeypatch.delenv("GENAI_SKIP_COLLECTOR_CHECK", raising=False)
    with caplog.at_level("WARNING"):
        _warn_if_default_collector_unreachable(_cfg(DEFAULT_OTLP_ENDPOINT))
    msg = caplog.text
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" in msg, "must name the variable to set"
    assert "GENAI_SKIP_COLLECTOR_CHECK" in msg, "must say how to silence it"


def test_check_failure_never_breaks_instrumentation(monkeypatch):
    """Instrumentation must survive anything this probe does."""
    monkeypatch.delenv("GENAI_SKIP_COLLECTOR_CHECK", raising=False)
    with patch("socket.create_connection", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            _warn_if_default_collector_unreachable(_cfg(DEFAULT_OTLP_ENDPOINT))
    # setup_auto_instrumentation wraps the call, so the raise stays contained there.
