import sys
import time
from unittest.mock import MagicMock, Mock, call, patch

import pytest

# Skip all GPU tests if nvidia-ml-py is not installed
pytest.importorskip("pynvml", reason="nvidia-ml-py (pynvml) not installed")


# Configure pytest
def pytest_configure(config):
    # No global module-level mocks here. Use patch in fixtures/tests instead.
    pass


# Common fixtures


@pytest.fixture
def mock_otel_config():
    """Fixture for a mock OTelConfig."""
    config = Mock()
    config.enable_co2_tracking = True
    config.carbon_intensity = 0.4  # gCO2e/kWh
    config.power_cost_per_kwh = 0.12  # USD per kWh
    return config


@pytest.fixture
def mock_meter():
    """Fixture for a mock OpenTelemetry Meter."""
    meter = Mock()
    # create_counter is called for both CO2 and power cost counters
    # Return different Mock objects for each counter to avoid confusion
    meter.create_counter.side_effect = [Mock(), Mock()]
    # Return a new Mock() each time create_observable_gauge is called
    meter.create_observable_gauge.return_value = Mock()
    return meter


@pytest.fixture
def mock_pynvml_gpu_available():
    """Fixture to mock pynvml when GPUs are available."""
    with patch("genai_otel.gpu_metrics.pynvml") as mock_pynvml:
        import genai_otel.gpu_metrics

        genai_otel.gpu_metrics.NVML_AVAILABLE = True
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_handle = Mock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetName.return_value = b"NVIDIA GeForce RTX 3080"
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = Mock(gpu=50, memory=60)
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = Mock(
            total=10000000000, used=5000000000, free=5000000000
        )  # Bytes
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 65
        mock_pynvml.NVML_TEMPERATURE_GPU = 0  # Mock constant
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 150000  # mW
        mock_pynvml.nvmlShutdown.return_value = None
        yield mock_pynvml


@pytest.fixture
def mock_pynvml_no_gpu():
    """Fixture to mock pynvml when no GPUs are available."""
    with patch("genai_otel.gpu_metrics.pynvml") as mock_pynvml:
        import genai_otel.gpu_metrics

        genai_otel.gpu_metrics.NVML_AVAILABLE = True
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 0
        mock_pynvml.nvmlShutdown.return_value = None
        yield mock_pynvml


class TestGPUMetricsCollector:
    @patch("genai_otel.gpu_metrics.logger")
    @patch.dict(sys.modules, {"pynvml": None})
    @patch("genai_otel.gpu_metrics.NVML_AVAILABLE", False)
    def test_init_nvml_not_available(self, mock_logger, mock_meter, mock_otel_config):
        import genai_otel.gpu_metrics

        # NVML_AVAILABLE is already False due to patch
        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)
        assert not collector.gpu_available
        mock_logger.warning.assert_called_with(
            "GPU metrics collection not available - nvidia-ml-py not installed. "
            "Install with: pip install genai-otel-instrument[gpu]"
        )
        # Both CO2 and power cost counters are created even when NVML not available
        assert mock_meter.create_counter.call_count == 2
        mock_meter.create_observable_gauge.assert_not_called()  # other gauges not created

    @patch("genai_otel.gpu_metrics.logger")
    def test_init_nvml_init_fails(
        self, mock_logger, mock_meter, mock_otel_config, mock_pynvml_gpu_available
    ):
        import genai_otel.gpu_metrics

        mock_pynvml_gpu_available.nvmlInit.side_effect = Exception("NVML init failed")
        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)
        assert not collector.gpu_available
        # Both CO2 and power cost counters are created (before GPU availability check)
        assert mock_meter.create_counter.call_count == 2
        # GPU gauges are created even if no GPUs available
        assert mock_meter.create_observable_gauge.call_count == 5

    @patch("genai_otel.gpu_metrics.CODECARBON_AVAILABLE", False)
    @patch("genai_otel.gpu_metrics.logger")
    def test_init_no_gpus(self, mock_logger, mock_meter, mock_otel_config, mock_pynvml_no_gpu):
        import genai_otel.gpu_metrics

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)
        assert not collector.gpu_available
        assert collector.device_count == 0
        mock_pynvml_no_gpu.nvmlInit.assert_called_once()
        mock_pynvml_no_gpu.nvmlDeviceGetCount.assert_called_once()
        # No GPU-related warning if NVML is available but no GPUs
        # (codecarbon warnings may occur separately)

    @patch("genai_otel.gpu_metrics.logger")
    def test_init_with_gpus(
        self, mock_logger, mock_meter, mock_otel_config, mock_pynvml_gpu_available
    ):
        import genai_otel.gpu_metrics

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)
        assert collector.gpu_available
        assert collector.device_count == 1
        mock_pynvml_gpu_available.nvmlInit.assert_called_once()
        # Both CO2 and power cost counters are created
        assert mock_meter.create_counter.call_count == 2
        # All five metrics are now ObservableGauges
        assert (
            mock_meter.create_observable_gauge.call_count == 5
        )  # utilization, memory used, memory total, temperature, power

    @patch("genai_otel.gpu_metrics.logger")
    def test_init_metric_instrument_creation_fails(
        self, mock_logger, mock_meter, mock_otel_config, mock_pynvml_gpu_available
    ):
        import genai_otel.gpu_metrics

        # Make create_observable_gauge fail
        mock_meter.create_observable_gauge.side_effect = StopIteration()
        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)
        assert collector.gpu_available
        mock_logger.error.assert_called_with(
            "Failed to create GPU metrics instruments: %s",
            mock_meter.create_observable_gauge.side_effect,
            exc_info=True,
        )

    @patch("genai_otel.gpu_metrics.logger")
    @patch.dict(sys.modules, {"pynvml": None})
    @patch("genai_otel.gpu_metrics.NVML_AVAILABLE", False)
    def test_observe_gpu_utilization_nvml_not_available(
        self, mock_logger, mock_meter, mock_otel_config
    ):
        import genai_otel.gpu_metrics

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)
        # ObservableGauge callbacks return generators, should return empty if NVML not available
        observations = list(collector._observe_gpu_utilization(None))
        assert observations == []

    @patch("genai_otel.gpu_metrics.logger")
    def test_observe_gpu_utilization_nvml_init_fails(
        self, mock_logger, mock_meter, mock_otel_config, mock_pynvml_gpu_available
    ):
        import genai_otel.gpu_metrics

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)
        # Reset and make nvmlInit fail in callback
        mock_pynvml_gpu_available.nvmlInit.reset_mock()
        mock_pynvml_gpu_available.nvmlInit.side_effect = Exception(
            "NVML init failed during observe"
        )

        observations = list(collector._observe_gpu_utilization(None))
        assert observations == []
        mock_logger.error.assert_called_with(
            "Error observing GPU utilization: %s", mock_pynvml_gpu_available.nvmlInit.side_effect
        )

    @patch("genai_otel.gpu_metrics.logger")
    def test_observe_gpu_utilization_successful(
        self, mock_logger, mock_meter, mock_otel_config, mock_pynvml_gpu_available
    ):
        from opentelemetry.metrics import Observation

        import genai_otel.gpu_metrics

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)
        # Reset mocks from init
        mock_pynvml_gpu_available.nvmlInit.reset_mock()
        mock_pynvml_gpu_available.nvmlDeviceGetCount.reset_mock()

        observations = list(collector._observe_gpu_utilization(None))

        assert len(observations) == 1
        assert isinstance(observations[0], Observation)
        assert observations[0].value == 50
        assert observations[0].attributes == {"gpu_id": "0", "gpu_name": "NVIDIA GeForce RTX 3080"}

        mock_pynvml_gpu_available.nvmlInit.assert_called_once()
        mock_pynvml_gpu_available.nvmlShutdown.assert_called_once()

    @patch("genai_otel.gpu_metrics.logger")
    def test_observe_gpu_metrics_partial_failures(
        self, mock_logger, mock_meter, mock_otel_config, mock_pynvml_gpu_available
    ):
        from opentelemetry.metrics import Observation

        import genai_otel.gpu_metrics

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)

        # Reset mocks from init
        mock_pynvml_gpu_available.nvmlInit.reset_mock()
        mock_pynvml_gpu_available.nvmlDeviceGetCount.reset_mock()

        # Mock memory info to fail
        mock_pynvml_gpu_available.nvmlDeviceGetMemoryInfo.side_effect = Exception("Memory error")

        observations = list(collector._observe_gpu_memory(None))

        # Should get no observations but no crash
        assert observations == []
        mock_logger.debug.assert_called_with(
            "Failed to get GPU memory for GPU %d: %s",
            0,
            mock_pynvml_gpu_available.nvmlDeviceGetMemoryInfo.side_effect,
        )

    @patch("genai_otel.gpu_metrics.logger")
    def test_observe_gpu_power_successful(
        self, mock_logger, mock_meter, mock_otel_config, mock_pynvml_gpu_available
    ):
        from opentelemetry.metrics import Observation

        import genai_otel.gpu_metrics

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)
        # Reset mocks from init
        mock_pynvml_gpu_available.nvmlInit.reset_mock()
        mock_pynvml_gpu_available.nvmlDeviceGetCount.reset_mock()

        observations = list(collector._observe_gpu_power(None))

        assert len(observations) == 1
        assert isinstance(observations[0], Observation)
        # Power usage is 150000 mW = 150 W
        assert observations[0].value == 150.0
        assert observations[0].attributes == {"gpu_id": "0", "gpu_name": "NVIDIA GeForce RTX 3080"}

        mock_pynvml_gpu_available.nvmlInit.assert_called_once()
        mock_pynvml_gpu_available.nvmlShutdown.assert_called_once()

    @patch("genai_otel.gpu_metrics.threading.Thread")
    @patch("genai_otel.gpu_metrics.logger")
    @patch.dict(sys.modules, {"pynvml": None})
    @patch("genai_otel.gpu_metrics.NVML_AVAILABLE", False)
    def test_start_nvml_not_available(self, mock_logger, mock_thread, mock_meter, mock_otel_config):
        import genai_otel.gpu_metrics

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)
        collector.start()
        mock_logger.warning.assert_any_call(
            "Cannot start GPU metrics collection - nvidia-ml-py not available"
        )
        mock_thread.assert_not_called()

    @patch("genai_otel.gpu_metrics.CODECARBON_AVAILABLE", False)
    @patch("genai_otel.gpu_metrics.threading.Thread")
    @patch("genai_otel.gpu_metrics.logger")
    def test_start_gpu_not_available(
        self, mock_logger, mock_thread, mock_meter, mock_otel_config, mock_pynvml_no_gpu
    ):
        import genai_otel.gpu_metrics

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)
        assert not collector.gpu_available  # From mock_pynvml_no_gpu
        collector.start()
        mock_thread.assert_not_called()

    @patch("genai_otel.gpu_metrics.threading.Thread")
    @patch("genai_otel.gpu_metrics.logger")
    def test_start_metrics_not_initialized(
        self, mock_logger, mock_thread, mock_meter, mock_otel_config, mock_pynvml_gpu_available
    ):
        import genai_otel.gpu_metrics

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)
        # With new implementation, start() always works - ObservableGauges handle collection
        collector.start()
        # Only CO2 collection thread is started
        mock_thread.assert_called_once_with(target=collector._collect_loop, daemon=True)
        mock_thread.return_value.start.assert_called_once()

    @patch("genai_otel.gpu_metrics.threading.Thread")
    @patch("genai_otel.gpu_metrics.logger")
    def test_start_successful(
        self, mock_logger, mock_thread, mock_meter, mock_otel_config, mock_pynvml_gpu_available
    ):
        import genai_otel.gpu_metrics

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)
        collector.start()
        # Only CO2 collection thread started (no more _run thread)
        mock_thread.assert_called_once_with(target=collector._collect_loop, daemon=True)
        mock_thread.return_value.start.assert_called_once()
        mock_logger.info.assert_called_with("Starting GPU metrics collection (CO2 tracking)")

    @patch("genai_otel.gpu_metrics.CODECARBON_AVAILABLE", False)
    @patch("genai_otel.gpu_metrics.time.time", return_value=1000.0)
    @patch("genai_otel.gpu_metrics.logger")
    def test_collect_loop_co2_enabled(
        self, mock_logger, mock_time, mock_meter, mock_otel_config, mock_pynvml_gpu_available
    ):
        import genai_otel.gpu_metrics

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)
        collector.device_count = 1
        collector.last_timestamp = [990.0]  # 10 seconds ago
        collector.interval = 1  # Short interval for testing

        # Mock _stop_event.wait to return True after one iteration
        collector._stop_event.wait = Mock(side_effect=[False, True])

        collector._collect_loop()

        mock_pynvml_gpu_available.nvmlDeviceGetPowerUsage.assert_called_once_with(
            mock_pynvml_gpu_available.nvmlDeviceGetHandleByIndex.return_value
        )
        # Power usage is 150W (150000 mW)
        # delta_time_hours = (1000 - 990) / 3600 = 10 / 3600 hours
        # delta_energy_wh = (150 / 1000) * (10 / 3600 * 3600) = 0.15 * 10 = 1.5 Wh
        # delta_co2_g = (1.5 / 1000) * 0.4 = 0.0006 gCO2e
        collector.co2_counter.add.assert_called_once_with(pytest.approx(0.0006), {"gpu_id": "0"})
        assert collector.cumulative_energy_wh[0] == pytest.approx(1.5)
        assert collector.last_timestamp[0] == pytest.approx(1000.0)
        assert collector._stop_event.wait.call_count == 2
        collector._stop_event.wait.assert_has_calls([call(1), call(1)])

    @patch("genai_otel.gpu_metrics.time.time", return_value=1000.0)
    @patch("genai_otel.gpu_metrics.logger")
    def test_collect_loop_co2_disabled(
        self, mock_logger, mock_time, mock_meter, mock_otel_config, mock_pynvml_gpu_available
    ):
        import genai_otel.gpu_metrics

        mock_otel_config.enable_co2_tracking = False
        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)
        collector.device_count = 1
        collector.last_timestamp = [990.0]
        collector.interval = 1

        collector._stop_event.wait = Mock(side_effect=[False, True])

        collector._collect_loop()

        collector.co2_counter.add.assert_not_called()
        assert collector.cumulative_energy_wh[0] == 1.5
        assert collector.last_timestamp[0] == 1000.0

    @patch("genai_otel.gpu_metrics.time.time", return_value=1000.0)
    @patch("genai_otel.gpu_metrics.logger")
    def test_collect_loop_power_cost_tracking(
        self, mock_logger, mock_time, mock_meter, mock_otel_config, mock_pynvml_gpu_available
    ):
        """Test that power cost is calculated and recorded correctly."""
        import genai_otel.gpu_metrics

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)
        collector.device_count = 1
        collector.last_timestamp = [990.0]  # 10 seconds ago
        collector.interval = 1

        # Mock _stop_event.wait to return True after one iteration
        collector._stop_event.wait = Mock(side_effect=[False, True])

        collector._collect_loop()

        # Power usage is 150W (150000 mW)
        # delta_time_hours = (1000 - 990) / 3600 = 10 / 3600 hours
        # delta_energy_wh = (150 / 1000) * (10 / 3600 * 3600) = 0.15 * 10 = 1.5 Wh
        # delta_cost_usd = (1.5 / 1000) * 0.12 = 0.00018 USD
        expected_cost = (1.5 / 1000.0) * 0.12
        collector.power_cost_counter.add.assert_called_once()
        call_args = collector.power_cost_counter.add.call_args
        assert call_args[0][0] == pytest.approx(expected_cost)
        assert call_args[0][1] == {"gpu_id": "0", "gpu_name": "NVIDIA GeForce RTX 3080"}

    @patch("genai_otel.gpu_metrics.time.time", return_value=1000.0)
    @patch("genai_otel.gpu_metrics.logger")
    def test_collect_loop_power_cost_with_different_rate(
        self, mock_logger, mock_time, mock_meter, mock_otel_config, mock_pynvml_gpu_available
    ):
        """Test power cost calculation with different electricity rate."""
        import genai_otel.gpu_metrics

        mock_otel_config.power_cost_per_kwh = 0.25  # Higher electricity cost
        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)
        collector.device_count = 1
        collector.last_timestamp = [990.0]
        collector.interval = 1

        collector._stop_event.wait = Mock(side_effect=[False, True])

        collector._collect_loop()

        # Same power and time, but different rate
        # delta_energy_wh = 1.5 Wh
        # delta_cost_usd = (1.5 / 1000) * 0.25 = 0.000375 USD
        expected_cost = (1.5 / 1000.0) * 0.25
        call_args = collector.power_cost_counter.add.call_args
        assert call_args[0][0] == pytest.approx(expected_cost)

    @patch("genai_otel.gpu_metrics.time.time", return_value=1000.0)
    @patch("genai_otel.gpu_metrics.logger")
    def test_collect_loop_error_handling(
        self, mock_logger, mock_time, mock_meter, mock_otel_config, mock_pynvml_gpu_available
    ):
        import genai_otel.gpu_metrics

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)
        collector.device_count = 1
        collector.last_timestamp = [990.0]
        collector.interval = 1

        power_error = Exception("Power usage error")
        mock_pynvml_gpu_available.nvmlDeviceGetPowerUsage.side_effect = power_error
        collector._stop_event.wait = Mock(side_effect=[False, True])

        collector._collect_loop()

        mock_logger.error.assert_called_once_with(
            "Error collecting GPU %d metrics: %s", 0, power_error
        )
        collector.co2_counter.add.assert_not_called()
        assert collector.cumulative_energy_wh[0] == 0.0  # No energy added due to error
        assert collector.last_timestamp[0] == 990.0  # Timestamp not updated due to error

    @patch("genai_otel.gpu_metrics.CODECARBON_AVAILABLE", False)
    @patch("genai_otel.gpu_metrics.logger")
    @patch("genai_otel.gpu_metrics.threading.Event")
    def test_stop_no_threads_running(
        self, mock_event_class, mock_logger, mock_meter, mock_otel_config, mock_pynvml_gpu_available
    ):
        import genai_otel.gpu_metrics

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)
        collector._thread = None
        collector._stop_event = mock_event_class.return_value  # Assign the mock instance
        collector.stop()
        collector._stop_event.set.assert_called_once()
        mock_pynvml_gpu_available.nvmlShutdown.assert_called_once()

    @patch("genai_otel.gpu_metrics.CODECARBON_AVAILABLE", False)
    @patch("genai_otel.gpu_metrics.logger")
    @patch("genai_otel.gpu_metrics.threading.Event")
    def test_stop_threads_running(
        self, mock_event_class, mock_logger, mock_meter, mock_otel_config, mock_pynvml_gpu_available
    ):
        import genai_otel.gpu_metrics

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)
        # With new implementation, there's no self.running or self.thread attributes
        collector._thread = Mock()
        collector._thread.is_alive.return_value = True
        collector._stop_event = mock_event_class.return_value  # Assign the mock instance

        collector.stop()

        # Only CO2 collection thread
        collector._thread.join.assert_called_once_with(timeout=5)
        mock_logger.info.assert_any_call("GPU CO2 metrics collection thread stopped.")
        collector._stop_event.set.assert_called_once()
        mock_pynvml_gpu_available.nvmlShutdown.assert_called_once()

    @patch("genai_otel.gpu_metrics.logger")
    def test_stop_gpu_not_available(
        self, mock_logger, mock_meter, mock_otel_config, mock_pynvml_no_gpu
    ):
        import genai_otel.gpu_metrics

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)
        assert not collector.gpu_available  # From mock_pynvml_no_gpu
        collector.stop()
        mock_pynvml_no_gpu.nvmlShutdown.assert_not_called()  # Should not be called if no GPU

    @patch("genai_otel.gpu_metrics.pynvml")
    def test_get_device_name_exception(self, mock_pynvml, mock_meter, mock_otel_config):
        """Test _get_device_name handles exceptions gracefully."""
        import genai_otel.gpu_metrics

        genai_otel.gpu_metrics.NVML_AVAILABLE = True
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "mock_handle"
        mock_pynvml.nvmlDeviceGetName.side_effect = Exception("Device name error")

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)

        # Should return fallback name on exception
        device_name = collector._get_device_name("mock_handle", 0)
        assert device_name == "GPU_0"

    @patch("genai_otel.gpu_metrics.pynvml")
    def test_observe_gpu_utilization_exception(self, mock_pynvml, mock_meter, mock_otel_config):
        """Test _observe_gpu_utilization handles per-device exceptions."""
        import genai_otel.gpu_metrics

        genai_otel.gpu_metrics.NVML_AVAILABLE = True
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "mock_handle"
        mock_pynvml.nvmlDeviceGetName.return_value = "Tesla V100"
        mock_pynvml.nvmlDeviceGetUtilizationRates.side_effect = Exception("Utilization error")

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)

        # Should handle exception gracefully and not yield observations
        observations = list(collector._observe_gpu_utilization(None))
        assert len(observations) == 0

    @patch("genai_otel.gpu_metrics.pynvml")
    def test_observe_gpu_memory_exception(self, mock_pynvml, mock_meter, mock_otel_config):
        """Test _observe_gpu_memory handles per-device exceptions."""
        import genai_otel.gpu_metrics

        genai_otel.gpu_metrics.NVML_AVAILABLE = True
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "mock_handle"
        mock_pynvml.nvmlDeviceGetName.return_value = "Tesla V100"
        mock_pynvml.nvmlDeviceGetMemoryInfo.side_effect = Exception("Memory error")

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)

        # Should handle exception gracefully
        observations = list(collector._observe_gpu_memory(None))
        assert len(observations) == 0

    @patch("genai_otel.gpu_metrics.pynvml")
    def test_observe_gpu_memory_total(self, mock_pynvml, mock_meter, mock_otel_config):
        """Test _observe_gpu_memory_total returns memory capacity."""
        import genai_otel.gpu_metrics

        genai_otel.gpu_metrics.NVML_AVAILABLE = True
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "mock_handle"
        mock_pynvml.nvmlDeviceGetName.return_value = "Tesla V100"

        mock_memory_info = Mock()
        mock_memory_info.total = 16 * 1024 * 1024 * 1024  # 16 GiB in bytes
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory_info

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)

        observations = list(collector._observe_gpu_memory_total(None))
        assert len(observations) == 1
        assert observations[0].value == 16384.0  # 16 GiB in MiB

    @patch("genai_otel.gpu_metrics.pynvml")
    def test_observe_gpu_memory_total_exception(self, mock_pynvml, mock_meter, mock_otel_config):
        """Test _observe_gpu_memory_total handles exceptions."""
        import genai_otel.gpu_metrics

        genai_otel.gpu_metrics.NVML_AVAILABLE = True
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "mock_handle"
        mock_pynvml.nvmlDeviceGetName.return_value = "Tesla V100"
        mock_pynvml.nvmlDeviceGetMemoryInfo.side_effect = Exception("Memory total error")

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)

        observations = list(collector._observe_gpu_memory_total(None))
        assert len(observations) == 0

    @patch("genai_otel.gpu_metrics.pynvml")
    def test_observe_gpu_temperature(self, mock_pynvml, mock_meter, mock_otel_config):
        """Test _observe_gpu_temperature returns temperature."""
        import genai_otel.gpu_metrics

        genai_otel.gpu_metrics.NVML_AVAILABLE = True
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "mock_handle"
        mock_pynvml.nvmlDeviceGetName.return_value = "Tesla V100"
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 65
        mock_pynvml.NVML_TEMPERATURE_GPU = 0

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)

        observations = list(collector._observe_gpu_temperature(None))
        assert len(observations) == 1
        assert observations[0].value == 65

    @patch("genai_otel.gpu_metrics.pynvml")
    def test_observe_gpu_temperature_exception(self, mock_pynvml, mock_meter, mock_otel_config):
        """Test _observe_gpu_temperature handles exceptions."""
        import genai_otel.gpu_metrics

        genai_otel.gpu_metrics.NVML_AVAILABLE = True
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "mock_handle"
        mock_pynvml.nvmlDeviceGetName.return_value = "Tesla V100"
        mock_pynvml.nvmlDeviceGetTemperature.side_effect = Exception("Temperature error")

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)

        observations = list(collector._observe_gpu_temperature(None))
        assert len(observations) == 0

    @patch("genai_otel.gpu_metrics.pynvml")
    def test_observe_gpu_power_exception(self, mock_pynvml, mock_meter, mock_otel_config):
        """Test _observe_gpu_power handles exceptions."""
        import genai_otel.gpu_metrics

        genai_otel.gpu_metrics.NVML_AVAILABLE = True
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "mock_handle"
        mock_pynvml.nvmlDeviceGetName.return_value = "Tesla V100"
        mock_pynvml.nvmlDeviceGetPowerUsage.side_effect = Exception("Power error")

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)

        observations = list(collector._observe_gpu_power(None))
        assert len(observations) == 0

    @patch("genai_otel.gpu_metrics.pynvml")
    def test_stop_with_nvml_shutdown(self, mock_pynvml, mock_meter, mock_otel_config):
        """Test stop method calls nvmlShutdown when GPU available."""
        import genai_otel.gpu_metrics

        genai_otel.gpu_metrics.NVML_AVAILABLE = True
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 1

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)
        collector.start()
        time.sleep(0.1)  # Let thread start

        collector.stop()

        # Verify shutdown was called
        assert not collector.running
        mock_pynvml.nvmlShutdown.assert_called()

    @patch("genai_otel.gpu_metrics.pynvml")
    def test_stop_nvml_shutdown_exception(self, mock_pynvml, mock_meter, mock_otel_config):
        """Test stop handles nvmlShutdown exceptions gracefully."""
        import genai_otel.gpu_metrics

        genai_otel.gpu_metrics.NVML_AVAILABLE = True
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlShutdown.side_effect = Exception("Shutdown error")

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)
        collector.start()
        time.sleep(0.1)

        # Should not raise exception
        collector.stop()
        assert not collector.running


class TestCodecarbonIntegration:
    """Tests for codecarbon integration in GPUMetricsCollector."""

    @pytest.fixture
    def mock_otel_config_with_codecarbon(self):
        """Fixture for OTelConfig with codecarbon settings."""
        config = Mock()
        config.enable_co2_tracking = True
        config.carbon_intensity = 475.0
        config.power_cost_per_kwh = 0.12
        config.service_name = "test-service"
        config.gpu_collection_interval = 5
        config.co2_country_iso_code = "USA"
        config.co2_region = "california"
        config.co2_cloud_provider = None
        config.co2_cloud_region = None
        config.co2_offline_mode = True
        config.co2_tracking_mode = "machine"
        config.co2_use_manual = False  # Default: use codecarbon if available
        return config

    @patch("genai_otel.gpu_metrics.CODECARBON_AVAILABLE", False)
    @patch("genai_otel.gpu_metrics.logger")
    def test_init_codecarbon_not_available(
        self, mock_logger, mock_meter, mock_otel_config_with_codecarbon, mock_pynvml_gpu_available
    ):
        """Test that manual CO2 calculation is used when codecarbon is not available."""
        import genai_otel.gpu_metrics

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(
            mock_meter, mock_otel_config_with_codecarbon
        )

        assert not collector._use_codecarbon
        assert collector._emissions_tracker is None
        mock_logger.info.assert_any_call(
            "codecarbon not installed, using manual CO2 calculation with "
            "carbon_intensity=%s gCO2e/kWh. Install codecarbon for automatic "
            "region-based carbon intensity: pip install genai-otel-instrument[co2]",
            475.0,
        )

    @patch("genai_otel.gpu_metrics.CODECARBON_AVAILABLE", True)
    @patch("genai_otel.gpu_metrics.OfflineEmissionsTracker")
    @patch("genai_otel.gpu_metrics.logger")
    def test_init_codecarbon_success(
        self,
        mock_logger,
        mock_offline_tracker_class,
        mock_meter,
        mock_otel_config_with_codecarbon,
        mock_pynvml_gpu_available,
    ):
        """Test successful codecarbon initialization in offline mode."""
        import genai_otel.gpu_metrics

        mock_tracker = Mock()
        mock_offline_tracker_class.return_value = mock_tracker

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(
            mock_meter, mock_otel_config_with_codecarbon
        )

        assert collector._use_codecarbon
        assert collector._emissions_tracker == mock_tracker
        mock_offline_tracker_class.assert_called_once()
        # Verify the kwargs passed to OfflineEmissionsTracker
        call_kwargs = mock_offline_tracker_class.call_args[1]
        assert call_kwargs["project_name"] == "test-service"
        assert call_kwargs["country_iso_code"] == "USA"
        assert call_kwargs["region"] == "california"
        assert call_kwargs["tracking_mode"] == "machine"

    @patch("genai_otel.gpu_metrics.CODECARBON_AVAILABLE", True)
    @patch("genai_otel.gpu_metrics.OfflineEmissionsTracker")
    @patch("genai_otel.gpu_metrics.logger")
    def test_init_codecarbon_failure(
        self,
        mock_logger,
        mock_offline_tracker_class,
        mock_meter,
        mock_otel_config_with_codecarbon,
        mock_pynvml_gpu_available,
    ):
        """Test fallback to manual calculation when codecarbon init fails."""
        import genai_otel.gpu_metrics

        mock_offline_tracker_class.side_effect = Exception("Codecarbon init failed")

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(
            mock_meter, mock_otel_config_with_codecarbon
        )

        assert not collector._use_codecarbon
        mock_logger.warning.assert_called_with(
            "Failed to initialize codecarbon, falling back to manual CO2 calculation: %s",
            mock_offline_tracker_class.side_effect,
        )

    @patch("genai_otel.gpu_metrics.logger")
    def test_init_codecarbon_disabled(
        self, mock_logger, mock_meter, mock_otel_config, mock_pynvml_gpu_available
    ):
        """Test that codecarbon is not initialized when CO2 tracking is disabled."""
        import genai_otel.gpu_metrics

        mock_otel_config.enable_co2_tracking = False

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(mock_meter, mock_otel_config)

        assert not collector._use_codecarbon
        assert collector._emissions_tracker is None

    @patch("genai_otel.gpu_metrics.CODECARBON_AVAILABLE", True)
    @patch("genai_otel.gpu_metrics.OfflineEmissionsTracker")
    @patch("genai_otel.gpu_metrics.logger")
    def test_start_with_codecarbon(
        self,
        mock_logger,
        mock_offline_tracker_class,
        mock_meter,
        mock_otel_config_with_codecarbon,
        mock_pynvml_gpu_available,
    ):
        """Test that codecarbon tracker is started with start()."""
        import genai_otel.gpu_metrics

        mock_tracker = Mock()
        mock_offline_tracker_class.return_value = mock_tracker

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(
            mock_meter, mock_otel_config_with_codecarbon
        )
        collector.start()

        mock_tracker.start.assert_called_once()
        assert collector._last_emissions_kg == 0.0

    @patch("genai_otel.gpu_metrics.CODECARBON_AVAILABLE", True)
    @patch("genai_otel.gpu_metrics.OfflineEmissionsTracker")
    @patch("genai_otel.gpu_metrics.logger")
    def test_start_codecarbon_failure(
        self,
        mock_logger,
        mock_offline_tracker_class,
        mock_meter,
        mock_otel_config_with_codecarbon,
        mock_pynvml_gpu_available,
    ):
        """Test fallback when codecarbon start() fails."""
        import genai_otel.gpu_metrics

        mock_tracker = Mock()
        mock_tracker.start.side_effect = Exception("Start failed")
        mock_offline_tracker_class.return_value = mock_tracker

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(
            mock_meter, mock_otel_config_with_codecarbon
        )
        collector.start()

        assert not collector._use_codecarbon
        mock_logger.warning.assert_any_call(
            "Failed to start codecarbon tracker: %s", mock_tracker.start.side_effect
        )

    @patch("genai_otel.gpu_metrics.CODECARBON_AVAILABLE", True)
    @patch("genai_otel.gpu_metrics.OfflineEmissionsTracker")
    def test_collect_codecarbon_emissions(
        self,
        mock_offline_tracker_class,
        mock_meter,
        mock_otel_config_with_codecarbon,
        mock_pynvml_gpu_available,
    ):
        """Test collecting emissions from codecarbon."""
        import genai_otel.gpu_metrics

        mock_tracker = Mock()
        mock_total_emissions = Mock()
        mock_total_emissions.total = 0.001  # 0.001 kg = 1 gram
        mock_tracker._total_emissions = mock_total_emissions
        mock_offline_tracker_class.return_value = mock_tracker

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(
            mock_meter, mock_otel_config_with_codecarbon
        )
        collector._last_emissions_kg = 0.0

        collector._collect_codecarbon_emissions()

        mock_tracker.flush.assert_called_once()
        # Should record 1 gram of emissions
        collector.co2_counter.add.assert_called_once()
        call_args = collector.co2_counter.add.call_args
        assert call_args[0][0] == pytest.approx(1.0)  # 0.001 kg * 1000 = 1 gram
        assert call_args[0][1]["source"] == "codecarbon"
        assert collector._last_emissions_kg == 0.001

    @patch("genai_otel.gpu_metrics.CODECARBON_AVAILABLE", True)
    @patch("genai_otel.gpu_metrics.OfflineEmissionsTracker")
    def test_collect_codecarbon_no_new_emissions(
        self,
        mock_offline_tracker_class,
        mock_meter,
        mock_otel_config_with_codecarbon,
        mock_pynvml_gpu_available,
    ):
        """Test no recording when there are no new emissions."""
        import genai_otel.gpu_metrics

        mock_tracker = Mock()
        mock_total_emissions = Mock()
        mock_total_emissions.total = 0.001
        mock_tracker._total_emissions = mock_total_emissions
        mock_offline_tracker_class.return_value = mock_tracker

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(
            mock_meter, mock_otel_config_with_codecarbon
        )
        collector._last_emissions_kg = 0.001  # Same as current

        collector._collect_codecarbon_emissions()

        # Should not record since delta is 0
        collector.co2_counter.add.assert_not_called()

    @patch("genai_otel.gpu_metrics.CODECARBON_AVAILABLE", True)
    @patch("genai_otel.gpu_metrics.OfflineEmissionsTracker")
    def test_stop_with_codecarbon(
        self,
        mock_offline_tracker_class,
        mock_meter,
        mock_otel_config_with_codecarbon,
        mock_pynvml_gpu_available,
    ):
        """Test that codecarbon tracker is stopped and final emissions recorded."""
        import genai_otel.gpu_metrics

        mock_tracker = Mock()
        mock_tracker.stop.return_value = 0.005  # 5 grams total
        mock_offline_tracker_class.return_value = mock_tracker

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(
            mock_meter, mock_otel_config_with_codecarbon
        )
        collector._last_emissions_kg = 0.003  # 3 grams already recorded

        collector.stop()

        mock_tracker.stop.assert_called_once()
        # Should record remaining 2 grams
        collector.co2_counter.add.assert_called()
        call_args = collector.co2_counter.add.call_args
        assert call_args[0][0] == pytest.approx(2.0)  # (0.005 - 0.003) * 1000 = 2 grams

    @patch("genai_otel.gpu_metrics.time.time", return_value=1000.0)
    @patch("genai_otel.gpu_metrics.CODECARBON_AVAILABLE", True)
    @patch("genai_otel.gpu_metrics.OfflineEmissionsTracker")
    def test_collect_loop_uses_codecarbon_not_manual(
        self,
        mock_offline_tracker_class,
        mock_time,
        mock_meter,
        mock_otel_config_with_codecarbon,
        mock_pynvml_gpu_available,
    ):
        """Test that collect loop uses codecarbon and not manual CO2 calculation."""
        import genai_otel.gpu_metrics

        mock_tracker = Mock()
        mock_total_emissions = Mock()
        mock_total_emissions.total = 0.0
        mock_tracker._total_emissions = mock_total_emissions
        mock_offline_tracker_class.return_value = mock_tracker

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(
            mock_meter, mock_otel_config_with_codecarbon
        )
        collector.device_count = 1
        collector.last_timestamp = [990.0]
        collector.interval = 1

        collector._stop_event.wait = Mock(side_effect=[False, True])

        collector._collect_loop()

        # Codecarbon flush should be called
        mock_tracker.flush.assert_called()
        # Manual CO2 calculation should NOT be called (co2_counter.add only from codecarbon)
        # The power_cost_counter should still be called
        collector.power_cost_counter.add.assert_called()

    @patch("genai_otel.gpu_metrics.CODECARBON_AVAILABLE", True)
    @patch("genai_otel.gpu_metrics.OfflineEmissionsTracker")
    @patch("genai_otel.gpu_metrics.logger")
    def test_init_codecarbon_default_country(
        self,
        mock_logger,
        mock_offline_tracker_class,
        mock_meter,
        mock_otel_config_with_codecarbon,
        mock_pynvml_gpu_available,
    ):
        """Test that USA is used as default country when not specified in offline mode."""
        import genai_otel.gpu_metrics

        mock_otel_config_with_codecarbon.co2_country_iso_code = None  # Not specified
        mock_tracker = Mock()
        mock_offline_tracker_class.return_value = mock_tracker

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(
            mock_meter, mock_otel_config_with_codecarbon
        )

        # Verify USA is used as default
        call_kwargs = mock_offline_tracker_class.call_args[1]
        assert call_kwargs["country_iso_code"] == "USA"

    @patch("genai_otel.gpu_metrics.CODECARBON_AVAILABLE", True)
    @patch("genai_otel.gpu_metrics.OfflineEmissionsTracker")
    @patch("genai_otel.gpu_metrics.logger")
    def test_init_codecarbon_use_manual_override(
        self,
        mock_logger,
        mock_offline_tracker_class,
        mock_meter,
        mock_otel_config_with_codecarbon,
        mock_pynvml_gpu_available,
    ):
        """Test that codecarbon is skipped when co2_use_manual is True."""
        import genai_otel.gpu_metrics

        mock_otel_config_with_codecarbon.co2_use_manual = True
        mock_otel_config_with_codecarbon.carbon_intensity = 500.0

        collector = genai_otel.gpu_metrics.GPUMetricsCollector(
            mock_meter, mock_otel_config_with_codecarbon
        )

        # Codecarbon should NOT be initialized
        assert not collector._use_codecarbon
        assert collector._emissions_tracker is None
        mock_offline_tracker_class.assert_not_called()
        mock_logger.info.assert_any_call(
            "Using manual CO2 calculation (GENAI_CO2_USE_MANUAL=true) with "
            "carbon_intensity=%s gCO2e/kWh",
            500.0,
        )
