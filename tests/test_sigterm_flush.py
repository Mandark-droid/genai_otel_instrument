"""Tests for the opt-in SIGTERM flush handler (issue #20).

The SDK's atexit hook covers a clean exit and an uncaught exception. It does not
run when the process is terminated by a signal, so `docker stop` and Kubernetes
pod eviction drop whatever is queued in the batch processor. These cover the
handler that closes that gap, and the three ways it must not misbehave.
"""

import signal
import threading
from unittest.mock import MagicMock, patch

import pytest

from genai_otel import auto_instrument
from genai_otel.auto_instrument import _install_sigterm_flush_handler, flush_telemetry
from genai_otel.config import OTelConfig


@pytest.fixture
def providers():
    """Install fake active providers and restore the originals afterwards."""
    tracer, meter = MagicMock(), MagicMock()
    tracer.force_flush.return_value = True
    meter.force_flush.return_value = True
    old_t, old_m = auto_instrument._active_tracer_provider, auto_instrument._active_meter_provider
    auto_instrument._active_tracer_provider = tracer
    auto_instrument._active_meter_provider = meter
    yield tracer, meter
    auto_instrument._active_tracer_provider = old_t
    auto_instrument._active_meter_provider = old_m


@pytest.fixture
def restore_sigterm():
    """Signal handlers are process-global; put the original back."""
    original = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, original)


class TestFlushTelemetry:
    def test_flushes_both_providers(self, providers):
        tracer, meter = providers
        assert flush_telemetry(2.0) is True
        tracer.force_flush.assert_called_once_with(2000)
        meter.force_flush.assert_called_once_with(2000)

    def test_reports_partial_flush(self, providers):
        tracer, _ = providers
        tracer.force_flush.return_value = False
        assert flush_telemetry() is False

    def test_provider_raising_does_not_propagate(self, providers):
        tracer, _ = providers
        tracer.force_flush.side_effect = RuntimeError("collector down")
        assert flush_telemetry() is False

    def test_no_providers_is_not_an_error(self):
        old_t, old_m = (
            auto_instrument._active_tracer_provider,
            auto_instrument._active_meter_provider,
        )
        auto_instrument._active_tracer_provider = None
        auto_instrument._active_meter_provider = None
        try:
            assert flush_telemetry() is True
        finally:
            auto_instrument._active_tracer_provider = old_t
            auto_instrument._active_meter_provider = old_m

    def test_timeout_is_never_zero(self, providers):
        """A sub-millisecond timeout must not become force_flush(0)."""
        tracer, _ = providers
        flush_telemetry(0.0)
        assert tracer.force_flush.call_args[0][0] >= 1


class TestHandlerInstallation:
    def test_installs_when_enabled(self, providers, restore_sigterm):
        cfg = OTelConfig(service_name="t")
        cfg.flush_on_sigterm = True
        assert _install_sigterm_flush_handler(cfg) is True
        assert callable(signal.getsignal(signal.SIGTERM))

    def test_not_installed_off_main_thread(self, providers):
        """signal.signal only works on the main thread; instrumentation must warn
        and carry on rather than raise into the host application."""
        cfg = OTelConfig(service_name="t")
        result = {}

        def worker():
            result["installed"] = _install_sigterm_flush_handler(cfg)

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert result["installed"] is False

    def test_install_failure_returns_false(self, providers, restore_sigterm):
        cfg = OTelConfig(service_name="t")
        with patch("signal.signal", side_effect=ValueError("no slot")):
            assert _install_sigterm_flush_handler(cfg) is False

    def test_disabled_by_default(self):
        assert OTelConfig(service_name="t").flush_on_sigterm is False

    def test_enabled_via_env(self, monkeypatch):
        monkeypatch.setenv("GENAI_FLUSH_ON_SIGTERM", "true")
        monkeypatch.setenv("GENAI_SIGTERM_FLUSH_TIMEOUT", "12.5")
        cfg = OTelConfig(service_name="t")
        assert cfg.flush_on_sigterm is True
        assert cfg.sigterm_flush_timeout == 12.5


class TestHandlerBehaviour:
    def test_flushes_then_chains_previous_handler(self, providers, restore_sigterm):
        """An application that already handles SIGTERM keeps its handler."""
        tracer, _ = providers
        calls = []
        signal.signal(signal.SIGTERM, lambda s, f: calls.append(("previous", s)))

        cfg = OTelConfig(service_name="t")
        _install_sigterm_flush_handler(cfg)
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

        assert tracer.force_flush.called, "must flush before handing on"
        assert calls == [("previous", signal.SIGTERM)]

    def test_restores_default_and_reraises(self, providers, restore_sigterm):
        """With no prior handler the signal must not be swallowed - the process
        should still die with the conventional status, not appear to ignore it."""
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        cfg = OTelConfig(service_name="t")
        _install_sigterm_flush_handler(cfg)

        with patch("os.kill") as mock_kill:
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

        mock_kill.assert_called_once()
        assert mock_kill.call_args[0][1] == signal.SIGTERM

    def test_honours_sig_ign(self, providers, restore_sigterm):
        """An application that deliberately ignores SIGTERM keeps ignoring it."""
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        cfg = OTelConfig(service_name="t")
        _install_sigterm_flush_handler(cfg)

        with patch("os.kill") as mock_kill:
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

        mock_kill.assert_not_called()

    def test_flush_failure_still_chains(self, providers, restore_sigterm):
        """A collector that is down must not strand the process in our handler."""
        tracer, _ = providers
        tracer.force_flush.side_effect = RuntimeError("collector down")
        calls = []
        signal.signal(signal.SIGTERM, lambda s, f: calls.append("previous"))

        cfg = OTelConfig(service_name="t")
        _install_sigterm_flush_handler(cfg)
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

        assert calls == ["previous"]

    def test_uses_configured_timeout(self, providers, restore_sigterm):
        tracer, _ = providers
        signal.signal(signal.SIGTERM, lambda s, f: None)
        cfg = OTelConfig(service_name="t")
        cfg.sigterm_flush_timeout = 3.0
        _install_sigterm_flush_handler(cfg)
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        assert tracer.force_flush.call_args[0][0] == 3000
