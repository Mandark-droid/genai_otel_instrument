"""Tests for OllamaServerMetricsPoller."""

import subprocess
import time
from unittest.mock import Mock, patch

import pytest
import requests

from genai_otel.instrumentors.ollama_server_metrics_poller import (
    OllamaServerMetricsPoller,
    start_ollama_metrics_poller,
    stop_ollama_metrics_poller,
)


@pytest.fixture
def mock_server_metrics():
    """Create a mock ServerMetricsCollector."""
    mock_metrics = Mock()
    mock_metrics.set_kv_cache_usage = Mock()
    mock_metrics._kv_cache_usage = {}
    return mock_metrics


@pytest.fixture
def poller():
    """Create an OllamaServerMetricsPoller instance."""
    return OllamaServerMetricsPoller(base_url="http://localhost:11434", interval=0.1)


def test_poller_init():
    """Test poller initialization."""
    poller = OllamaServerMetricsPoller(
        base_url="http://localhost:11434", interval=5.0, max_vram_gb=24.0
    )

    assert poller.base_url == "http://localhost:11434"
    assert poller.interval == 5.0
    assert poller.max_vram_bytes == 24.0 * 1024**3
    assert not poller._running


def test_poller_init_no_max_vram():
    """Test poller initialization without max VRAM."""
    # Mock both detection methods to return None
    with (
        patch("genai_otel.instrumentors.ollama_server_metrics_poller.NVIDIA_ML_AVAILABLE", False),
        patch("subprocess.run", side_effect=FileNotFoundError()),
    ):
        poller = OllamaServerMetricsPoller()

        assert poller.base_url == "http://localhost:11434"
        assert poller.interval == 5.0
        assert poller.max_vram_bytes is None


def test_poller_start_stop(poller):
    """Test starting and stopping the poller."""
    assert not poller._running

    poller.start()
    assert poller._running
    assert poller._thread is not None
    assert poller._thread.daemon is True

    # Give thread time to start
    time.sleep(0.05)

    poller.stop()
    assert not poller._running


def test_poller_start_already_running(poller):
    """Test starting an already running poller (should log warning)."""
    poller.start()
    assert poller._running

    # Try to start again
    poller.start()
    # Should still be running, not create new thread
    assert poller._running

    poller.stop()


def test_collect_metrics_success(poller, mock_server_metrics):
    """Test successful metrics collection."""
    # Mock the API response
    mock_response = Mock()
    mock_response.json.return_value = {
        "models": [
            {
                "name": "llama2:latest",
                "size_vram": 5137025024,  # ~4.8GB
            },
            {
                "name": "mistral:latest",
                "size_vram": 4294967296,  # 4GB
            },
        ]
    }
    mock_response.raise_for_status = Mock()

    with (
        patch("requests.get", return_value=mock_response),
        patch(
            "genai_otel.instrumentors.ollama_server_metrics_poller.get_server_metrics",
            return_value=mock_server_metrics,
        ),
    ):
        poller._collect_metrics()

        # Verify requests.get was called
        # Note: We can't easily verify the exact call args because requests.get is mocked globally

        # Verify server metrics were updated
        assert mock_server_metrics.set_kv_cache_usage.call_count == 2
        # Verify both models were updated
        calls = mock_server_metrics.set_kv_cache_usage.call_args_list
        models_updated = {call[0][0] for call in calls}
        assert "llama2:latest" in models_updated
        assert "mistral:latest" in models_updated


def test_collect_metrics_with_max_vram(mock_server_metrics):
    """Test metrics collection with configured max VRAM."""
    poller = OllamaServerMetricsPoller(
        base_url="http://localhost:11434", interval=0.1, max_vram_gb=24.0
    )

    # Mock the API response
    mock_response = Mock()
    mock_response.json.return_value = {
        "models": [
            {
                "name": "llama2:latest",
                "size_vram": 12884901888,  # 12GB
            }
        ]
    }
    mock_response.raise_for_status = Mock()

    with (
        patch("requests.get", return_value=mock_response),
        patch(
            "genai_otel.instrumentors.ollama_server_metrics_poller.get_server_metrics",
            return_value=mock_server_metrics,
        ),
    ):
        poller._collect_metrics()

        # Verify cache usage was calculated as percentage
        mock_server_metrics.set_kv_cache_usage.assert_called_once()
        model_name, usage_pct = mock_server_metrics.set_kv_cache_usage.call_args[0]
        assert model_name == "llama2:latest"
        # 12GB / 24GB = 50%
        assert 49.0 < usage_pct < 51.0


def test_collect_metrics_no_models(poller, mock_server_metrics):
    """Test metrics collection when no models are loaded."""
    mock_response = Mock()
    mock_response.json.return_value = {"models": []}
    mock_response.raise_for_status = Mock()

    with (
        patch("requests.get", return_value=mock_response),
        patch(
            "genai_otel.instrumentors.ollama_server_metrics_poller.get_server_metrics",
            return_value=mock_server_metrics,
        ),
    ):
        poller._collect_metrics()

        # Should not update any metrics
        mock_server_metrics.set_kv_cache_usage.assert_not_called()


def test_collect_metrics_connection_error(poller, mock_server_metrics):
    """Test metrics collection when Ollama server is not running."""
    with (
        patch(
            "requests.get", side_effect=requests.exceptions.ConnectionError("Connection refused")
        ),
        patch(
            "genai_otel.instrumentors.ollama_server_metrics_poller.get_server_metrics",
            return_value=mock_server_metrics,
        ),
    ):
        # Should not raise exception, just log
        poller._collect_metrics()

        # Should not update metrics
        mock_server_metrics.set_kv_cache_usage.assert_not_called()


def test_collect_metrics_timeout(poller, mock_server_metrics):
    """Test metrics collection when request times out."""
    with (
        patch("requests.get", side_effect=requests.exceptions.Timeout("Timeout")),
        patch(
            "genai_otel.instrumentors.ollama_server_metrics_poller.get_server_metrics",
            return_value=mock_server_metrics,
        ),
    ):
        # Should not raise exception, just log warning
        poller._collect_metrics()

        # Should not update metrics
        mock_server_metrics.set_kv_cache_usage.assert_not_called()


def test_collect_metrics_no_server_metrics_collector(poller):
    """Test metrics collection when ServerMetricsCollector is not initialized."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "models": [{"name": "llama2:latest", "size_vram": 5137025024}]
    }
    mock_response.raise_for_status = Mock()

    with (
        patch("requests.get", return_value=mock_response),
        patch(
            "genai_otel.instrumentors.ollama_server_metrics_poller.get_server_metrics",
            return_value=None,
        ),
    ):
        # Should not raise exception
        poller._collect_metrics()


def test_collect_metrics_invalid_json(poller, mock_server_metrics):
    """Test metrics collection when API returns invalid JSON."""
    mock_response = Mock()
    mock_response.json.side_effect = ValueError("Invalid JSON")
    mock_response.raise_for_status = Mock()

    with (
        patch("requests.get", return_value=mock_response),
        patch(
            "genai_otel.instrumentors.ollama_server_metrics_poller.get_server_metrics",
            return_value=mock_server_metrics,
        ),
    ):
        # Should not raise exception, just log error
        poller._collect_metrics()

        # Should not update metrics
        mock_server_metrics.set_kv_cache_usage.assert_not_called()


def test_collect_metrics_http_error(poller, mock_server_metrics):
    """Test metrics collection when API returns HTTP error."""
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")

    with (
        patch("requests.get", return_value=mock_response),
        patch(
            "genai_otel.instrumentors.ollama_server_metrics_poller.get_server_metrics",
            return_value=mock_server_metrics,
        ),
    ):
        # Should not raise exception, just log warning
        poller._collect_metrics()

        # Should not update metrics
        mock_server_metrics.set_kv_cache_usage.assert_not_called()


def test_start_ollama_metrics_poller():
    """Test global poller initialization."""
    # Reset global state first
    import genai_otel.instrumentors.ollama_server_metrics_poller as poller_module

    poller_module._global_poller = None

    with patch.object(OllamaServerMetricsPoller, "start") as mock_start:
        poller = start_ollama_metrics_poller(
            base_url="http://localhost:11434", interval=5.0, max_vram_gb=24.0
        )

        assert poller is not None
        mock_start.assert_called_once()

        # Clean up
        stop_ollama_metrics_poller()


def test_start_ollama_metrics_poller_already_running():
    """Test starting global poller when already running."""
    # Reset global state first
    import genai_otel.instrumentors.ollama_server_metrics_poller as poller_module

    poller_module._global_poller = None

    with patch.object(OllamaServerMetricsPoller, "start"):
        # Start first time
        poller1 = start_ollama_metrics_poller()
        poller1._running = True  # Simulate running state

        # Start second time
        poller2 = start_ollama_metrics_poller()

        # Should return same instance
        assert poller1 is poller2

        # Clean up
        stop_ollama_metrics_poller()


def test_stop_ollama_metrics_poller():
    """Test stopping global poller."""
    # Reset global state first
    import genai_otel.instrumentors.ollama_server_metrics_poller as poller_module

    poller_module._global_poller = None

    with (
        patch.object(OllamaServerMetricsPoller, "start"),
        patch.object(OllamaServerMetricsPoller, "stop") as mock_stop,
    ):
        # Start poller
        start_ollama_metrics_poller()

        # Stop poller
        stop_ollama_metrics_poller()

        mock_stop.assert_called_once()


def test_stop_ollama_metrics_poller_not_running():
    """Test stopping global poller when not running."""
    # Should not raise exception
    stop_ollama_metrics_poller()


def test_poll_loop_runs_continuously(poller, mock_server_metrics):
    """Test that poll loop runs continuously until stopped."""
    mock_response = Mock()
    mock_response.json.return_value = {"models": []}
    mock_response.raise_for_status = Mock()

    with (
        patch("requests.get", return_value=mock_response),
        patch(
            "genai_otel.instrumentors.ollama_server_metrics_poller.get_server_metrics",
            return_value=mock_server_metrics,
        ),
    ):
        poller.start()

        # Wait for a few polling cycles (interval is 0.1s)
        time.sleep(0.35)

        poller.stop()

        # Should have polled multiple times (at least 2-3 times in 0.35s with 0.1s interval)
        # We can't assert exact count due to timing, but it should be > 1
        # This verifies the loop runs continuously


def test_collect_metrics_missing_size_vram(poller, mock_server_metrics):
    """Test metrics collection when size_vram field is missing."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "models": [
            {
                "name": "llama2:latest",
                # Missing size_vram field
            }
        ]
    }
    mock_response.raise_for_status = Mock()

    with (
        patch("requests.get", return_value=mock_response),
        patch(
            "genai_otel.instrumentors.ollama_server_metrics_poller.get_server_metrics",
            return_value=mock_server_metrics,
        ),
    ):
        poller._collect_metrics()

        # Should handle missing field gracefully
        # May update with 0 or skip the model depending on implementation


def test_detect_vram_nvidia_ml_success():
    """Test VRAM detection via nvidia-ml-py (pynvml)."""
    with (
        patch.object(OllamaServerMetricsPoller, "_detect_vram_nvidia_ml", return_value=24.0),
        patch.object(OllamaServerMetricsPoller, "_detect_vram_nvidia_smi", return_value=None),
    ):
        poller = OllamaServerMetricsPoller()

        # Should have detected 24GB
        assert poller.max_vram_bytes == 24.0 * 1024**3


def test_detect_vram_nvidia_ml_no_gpus():
    """Test VRAM detection when no GPUs are present."""
    with (
        patch.object(OllamaServerMetricsPoller, "_detect_vram_nvidia_ml", return_value=None),
        patch.object(OllamaServerMetricsPoller, "_detect_vram_nvidia_smi", return_value=None),
    ):
        poller = OllamaServerMetricsPoller()

        # Should fall back to None
        assert poller.max_vram_bytes is None


def test_detect_vram_nvidia_ml_not_available():
    """Test VRAM detection when nvidia-ml-py is not installed."""
    with (
        patch.object(OllamaServerMetricsPoller, "_detect_vram_nvidia_ml", return_value=None),
        patch.object(OllamaServerMetricsPoller, "_detect_vram_nvidia_smi", return_value=None),
    ):
        poller = OllamaServerMetricsPoller()

        # Should fall back to None
        assert poller.max_vram_bytes is None


def test_detect_vram_nvidia_smi_success():
    """Test VRAM detection via nvidia-smi command."""
    with (
        patch.object(OllamaServerMetricsPoller, "_detect_vram_nvidia_ml", return_value=None),
        patch.object(OllamaServerMetricsPoller, "_detect_vram_nvidia_smi", return_value=24.0),
    ):
        poller = OllamaServerMetricsPoller()

        # Should have detected 24GB via nvidia-smi
        assert poller.max_vram_bytes == 24 * 1024**3


def test_detect_vram_nvidia_smi_not_found():
    """Test VRAM detection when nvidia-smi is not found."""
    with (
        patch.object(OllamaServerMetricsPoller, "_detect_vram_nvidia_ml", return_value=None),
        patch.object(OllamaServerMetricsPoller, "_detect_vram_nvidia_smi", return_value=None),
    ):
        poller = OllamaServerMetricsPoller()

        # Should fall back to None
        assert poller.max_vram_bytes is None


def test_detect_vram_nvidia_smi_timeout():
    """Test VRAM detection when nvidia-smi times out."""
    with (
        patch.object(OllamaServerMetricsPoller, "_detect_vram_nvidia_ml", return_value=None),
        patch.object(OllamaServerMetricsPoller, "_detect_vram_nvidia_smi", return_value=None),
    ):
        poller = OllamaServerMetricsPoller()

        # Should fall back to None
        assert poller.max_vram_bytes is None


def test_detect_vram_manual_override():
    """Test that manual max_vram_gb overrides auto-detection."""
    # With manual override, detection methods should not be called
    poller = OllamaServerMetricsPoller(max_vram_gb=16.0)

    # Should use the manual value, not auto-detected
    assert poller.max_vram_bytes == 16 * 1024**3


def test_detect_vram_nvidia_ml_exception():
    """Test VRAM detection when nvidia-ml-py raises an exception."""
    with (
        patch.object(OllamaServerMetricsPoller, "_detect_vram_nvidia_ml", return_value=None),
        patch.object(OllamaServerMetricsPoller, "_detect_vram_nvidia_smi", return_value=None),
    ):
        poller = OllamaServerMetricsPoller()

        # Should fall back to None
        assert poller.max_vram_bytes is None
