"""Module for collecting GPU metrics using pynvml and reporting them via OpenTelemetry.

This module provides the `GPUMetricsCollector` class, which periodically collects
GPU utilization, memory usage, and temperature, and exports these as OpenTelemetry
metrics. It relies on the `pynvml` library for interacting with NVIDIA GPUs.
"""

import logging
import threading
import time
from typing import Optional

import pynvml  # Moved to top
from opentelemetry.metrics import Meter, ObservableGauge, ObservableCounter

logger = logging.getLogger(__name__)


class GPUMetricsCollector:
    """Collects and reports GPU metrics using pynvml."""

    def __init__(self, meter: Meter):
        """Initializes the GPUMetricsCollector.

        Args:
            meter (Meter): The OpenTelemetry meter to use for recording metrics.
        """
        self.meter = meter
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.gpu_utilization_counter: Optional[ObservableCounter] = None
        self.gpu_memory_used_gauge: Optional[ObservableGauge] = None
        self.gpu_temperature_gauge: Optional[ObservableGauge] = None

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
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                device_name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")

                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
                if self.gpu_utilization_counter:
                    self.gpu_utilization_counter.add(
                        gpu_util, {"gpu_id": str(i), "gpu_name": device_name}
                    )

                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_used = memory_info.used / (1024**2)
                if self.gpu_memory_used_gauge:
                    self.gpu_memory_used_gauge.add(
                        gpu_memory_used, {"gpu_id": str(i), "gpu_name": device_name}
                    )

                gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                if self.gpu_temperature_gauge:
                    self.gpu_temperature_gauge.add(
                        gpu_temp, {"gpu_id": str(i), "gpu_name": device_name}
                    )

            pynvml.nvmlShutdown()

        except pynvml.NVMLError as e:
            logger.error("NVMLError occurred: %s", e, exc_info=True)
        except Exception as e:
            logger.error("An error occurred during GPU metrics collection: %s", e, exc_info=True)

    def _run(self):
        """The main loop for collecting metrics periodically."""
        while self.running:
            self._collect_metrics()
            time.sleep(15)

    def start(self):
        """Starts the GPU metrics collection in a separate thread."""
        if not self.running and self.gpu_utilization_counter:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            logger.info("GPU metrics collection thread started.")
        elif not self.gpu_utilization_counter:
            logger.warning("GPU metrics instruments not initialized. Cannot start collection.")

    def stop(self):
        """Stops the GPU metrics collection thread."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
            logger.info("GPU metrics collection thread stopped.")
