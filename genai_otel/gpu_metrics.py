"""Module for collecting GPU metrics using nvidia-ml-py and reporting them via OpenTelemetry.

This module provides the `GPUMetricsCollector` class, which periodically collects
GPU utilization, memory usage, and temperature, and exports these as OpenTelemetry
metrics. It relies on the `nvidia-ml-py` library for interacting with NVIDIA GPUs.
"""

import logging
import threading
import time
from typing import Optional

from opentelemetry.metrics import Meter, ObservableCounter, ObservableGauge

from genai_otel.config import OTelConfig

logger = logging.getLogger(__name__)

# Try to import nvidia-ml-py (official replacement for pynvml)
try:
    import pynvml

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logger.debug("nvidia-ml-py not available, GPU metrics will be disabled")


class GPUMetricsCollector:
    """Collects and reports GPU metrics using nvidia-ml-py."""

    def __init__(self, meter: Meter, config: OTelConfig, interval: int = 10):
        """Initializes the GPUMetricsCollector.

        Args:
            meter (Meter): The OpenTelemetry meter to use for recording metrics.
        """
        self.meter = meter
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.gpu_utilization_counter: Optional[ObservableCounter] = None
        self.gpu_memory_used_gauge: Optional[ObservableGauge] = None
        self.gpu_temperature_gauge: Optional[ObservableGauge] = None
        self.config = config
        self.interval = interval  # seconds
        self.gpu_available = False

        self.device_count = 0
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
                if self.device_count > 0:
                    self.gpu_available = True
            except Exception as e:
                logger.error("Failed to initialize NVML to get device count: %s", e)

        self.cumulative_energy_wh = [0.0] * self.device_count  # Per GPU, in Wh
        self.last_timestamp = [time.time()] * self.device_count
        self.co2_counter = meter.create_counter(
            "genai.co-2.emissions",
            description="Cumulative CO2 equivalent emissions in grams",
            unit="gCO2e",
        )
        if not NVML_AVAILABLE:
            logger.warning(
                "GPU metrics collection not available - nvidia-ml-py not installed. "
                "Install with: pip install genai-otel-instrument[gpu]"
            )
            return

        try:
            self.gpu_utilization_counter = self.meter.create_counter(
                "genai.gpu.utilization", description="GPU utilization percentage", unit="%"
            )
            self.gpu_memory_used_gauge = self.meter.create_observable_gauge(
                "genai.gpu.memory.used", description="GPU memory used in MiB", unit="MiB"
            )
            self.gpu_temperature_gauge = self.meter.create_observable_gauge(
                "genai.gpu.temperature", description="GPU temperature in Celsius", unit="Cel"
            )
        except Exception as e:
            logger.error("Failed to create GPU metrics instruments: %s", e, exc_info=True)

    def _collect_metrics(self):
        """Collects GPU metrics and records them."""
        if not NVML_AVAILABLE:
            return

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # Get device name
                try:
                    device_name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(device_name, bytes):
                        device_name = device_name.decode("utf-8")
                except Exception as e:
                    device_name = f"GPU_{i}"
                    logger.debug("Failed to get GPU name: %s", e)

                # Get utilization
                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = utilization.gpu
                    if self.gpu_utilization_counter:
                        self.gpu_utilization_counter.add(
                            gpu_util, {"gpu_id": str(i), "gpu_name": device_name}
                        )
                except Exception as e:
                    logger.debug("Failed to get GPU utilization: %s", e)

                # Get memory info
                try:
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_used = memory_info.used / (1024**2)
                    if self.gpu_memory_used_gauge:
                        self.gpu_memory_used_gauge.add(
                            gpu_memory_used, {"gpu_id": str(i), "gpu_name": device_name}
                        )
                except Exception as e:
                    logger.debug("Failed to get GPU memory info: %s", e)

                # Get temperature
                try:
                    gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    if self.gpu_temperature_gauge:
                        self.gpu_temperature_gauge.add(
                            gpu_temp, {"gpu_id": str(i), "gpu_name": device_name}
                        )
                except Exception as e:
                    logger.debug("Failed to get GPU temperature: %s", e)

            pynvml.nvmlShutdown()

        except Exception as e:
            if "NVML" in str(e.__class__.__name__):
                logger.error("NVML error during GPU metrics collection: %s", e)
            else:
                logger.error("Error during GPU metrics collection: %s", e, exc_info=True)

    def _run(self):
        """The main loop for collecting metrics periodically."""
        while self.running:
            self._collect_metrics()
            time.sleep(15)

    def start(self):
        """Starts the GPU metrics collection in a separate thread."""
        if not NVML_AVAILABLE:
            logger.warning("Cannot start GPU metrics collection - nvidia-ml-py not available")
            return

        if not self.gpu_available:
            return
        logger.info("Starting GPU metrics collection including CO2")
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()

        if not self.running and self.gpu_utilization_counter:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            logger.info("GPU metrics collection thread started.")
        elif not self.gpu_utilization_counter:
            logger.warning("GPU metrics instruments not initialized. Cannot start collection.")

    def _collect_loop(self):
        while not self._stop_event.wait(self.interval):
            current_time = time.time()
            for i in range(self.device_count):
                try:
                    handle = self.nvml.nvmlDeviceGetHandleByIndex(i)
                    power_w = self.nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Watts
                    delta_time_hours = (current_time - self.last_timestamp[i]) / 3600.0
                    delta_energy_wh = (power_w / 1000.0) * (
                        delta_time_hours * 3600.0
                    )  # Wh (power in kW * hours = kWh, but track in Wh for precision)
                    self.cumulative_energy_wh[i] += delta_energy_wh
                    if self.config.enable_co2_tracking:
                        delta_co2_g = (
                            delta_energy_wh / 1000.0
                        ) * self.config.carbon_intensity  # gCO2e
                        self.co2_counter.add(delta_co2_g, {"gpu_id": str(i)})
                    self.last_timestamp[i] = current_time
                except Exception as e:
                    logger.error(f"Error collecting GPU {i} metrics: {e}")

    def stop(self):
        """Stops the GPU metrics collection thread."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
            logger.info("GPU metrics collection thread stopped.")
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            logger.info("CO2 metrics collection thread stopped.")
        if self.gpu_available:
            self.nvml.nvmlShutdown()
