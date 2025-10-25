"""Module for collecting GPU metrics using nvidia-ml-py and reporting them via OpenTelemetry.

This module provides the `GPUMetricsCollector` class, which periodically collects
GPU utilization, memory usage, and temperature, and exports these as OpenTelemetry
metrics. It relies on the `nvidia-ml-py` library for interacting with NVIDIA GPUs.
"""

import logging
import threading
import time
from typing import Optional

from opentelemetry.metrics import Meter, ObservableCounter, ObservableGauge, Observation

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
        self._thread: Optional[threading.Thread] = None  # Initialize _thread
        self._stop_event = threading.Event()
        self.gpu_utilization_counter: Optional[ObservableCounter] = None
        self.gpu_memory_used_gauge: Optional[ObservableGauge] = None
        self.gpu_memory_total_gauge: Optional[ObservableGauge] = None
        self.gpu_temperature_gauge: Optional[ObservableGauge] = None
        self.gpu_power_gauge: Optional[ObservableGauge] = None
        self.config = config
        self.interval = interval  # seconds
        self.gpu_available = False

        self.device_count = 0
        self.nvml = None
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
                if self.device_count > 0:
                    self.gpu_available = True
                self.nvml = pynvml
            except Exception as e:
                logger.error("Failed to initialize NVML to get device count: %s", e)

        self.cumulative_energy_wh = [0.0] * self.device_count  # Per GPU, in Wh
        self.last_timestamp = [time.time()] * self.device_count
        self.co2_counter = meter.create_counter(
            "gen_ai.co2.emissions",  # Fixed metric name
            description="Cumulative CO2 equivalent emissions in grams",
            unit="gCO2e",
        )
        self.power_cost_counter = meter.create_counter(
            "gen_ai.power.cost",  # New metric name
            description="Cumulative electricity cost in USD based on GPU power consumption",
            unit="USD",
        )
        if not NVML_AVAILABLE:
            logger.warning(
                "GPU metrics collection not available - nvidia-ml-py not installed. "
                "Install with: pip install genai-otel-instrument[gpu]"
            )
            return

        try:
            # Use ObservableGauge for all GPU metrics (not Counter!)
            self.gpu_utilization_gauge = self.meter.create_observable_gauge(
                "gen_ai.gpu.utilization",  # Fixed metric name
                callbacks=[self._observe_gpu_utilization],
                description="GPU utilization percentage",
                unit="%",
            )
            self.gpu_memory_used_gauge = self.meter.create_observable_gauge(
                "gen_ai.gpu.memory.used",  # Fixed metric name
                callbacks=[self._observe_gpu_memory],
                description="GPU memory used in MiB",
                unit="MiB",
            )
            self.gpu_memory_total_gauge = self.meter.create_observable_gauge(
                "gen_ai.gpu.memory.total",  # Fixed metric name
                callbacks=[self._observe_gpu_memory_total],
                description="Total GPU memory capacity in MiB",
                unit="MiB",
            )
            self.gpu_temperature_gauge = self.meter.create_observable_gauge(
                "gen_ai.gpu.temperature",  # Fixed metric name
                callbacks=[self._observe_gpu_temperature],
                description="GPU temperature in Celsius",
                unit="Cel",
            )
            self.gpu_power_gauge = self.meter.create_observable_gauge(
                "gen_ai.gpu.power",  # Fixed metric name
                callbacks=[self._observe_gpu_power],
                description="GPU power consumption in Watts",
                unit="W",
            )
        except Exception as e:
            logger.error("Failed to create GPU metrics instruments: %s", e, exc_info=True)

    def _get_device_name(self, handle, index):
        """Get GPU device name safely."""
        try:
            device_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(device_name, bytes):
                device_name = device_name.decode("utf-8")
            return device_name
        except Exception as e:
            logger.debug("Failed to get GPU name: %s", e)
            return f"GPU_{index}"

    def _observe_gpu_utilization(self, options):
        """Observable callback for GPU utilization."""
        if not NVML_AVAILABLE or not self.gpu_available:
            return

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                device_name = self._get_device_name(handle, i)

                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    yield Observation(
                        value=utilization.gpu,
                        attributes={"gpu_id": str(i), "gpu_name": device_name},
                    )
                except Exception as e:
                    logger.debug("Failed to get GPU utilization for GPU %d: %s", i, e)

            pynvml.nvmlShutdown()
        except Exception as e:
            logger.error("Error observing GPU utilization: %s", e)

    def _observe_gpu_memory(self, options):
        """Observable callback for GPU memory usage."""
        if not NVML_AVAILABLE or not self.gpu_available:
            return

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                device_name = self._get_device_name(handle, i)

                try:
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_used = memory_info.used / (1024**2)  # Convert to MiB
                    yield Observation(
                        value=gpu_memory_used,
                        attributes={"gpu_id": str(i), "gpu_name": device_name},
                    )
                except Exception as e:
                    logger.debug("Failed to get GPU memory for GPU %d: %s", i, e)

            pynvml.nvmlShutdown()
        except Exception as e:
            logger.error("Error observing GPU memory: %s", e)

    def _observe_gpu_memory_total(self, options):
        """Observable callback for total GPU memory capacity."""
        if not NVML_AVAILABLE or not self.gpu_available:
            return

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                device_name = self._get_device_name(handle, i)

                try:
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_total = memory_info.total / (1024**2)  # Convert to MiB
                    yield Observation(
                        value=gpu_memory_total,
                        attributes={"gpu_id": str(i), "gpu_name": device_name},
                    )
                except Exception as e:
                    logger.debug("Failed to get total GPU memory for GPU %d: %s", i, e)

            pynvml.nvmlShutdown()
        except Exception as e:
            logger.error("Error observing total GPU memory: %s", e)

    def _observe_gpu_temperature(self, options):
        """Observable callback for GPU temperature."""
        if not NVML_AVAILABLE or not self.gpu_available:
            return

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                device_name = self._get_device_name(handle, i)

                try:
                    gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    yield Observation(
                        value=gpu_temp, attributes={"gpu_id": str(i), "gpu_name": device_name}
                    )
                except Exception as e:
                    logger.debug("Failed to get GPU temperature for GPU %d: %s", i, e)

            pynvml.nvmlShutdown()
        except Exception as e:
            logger.error("Error observing GPU temperature: %s", e)

    def _observe_gpu_power(self, options):
        """Observable callback for GPU power consumption."""
        if not NVML_AVAILABLE or not self.gpu_available:
            return

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                device_name = self._get_device_name(handle, i)

                try:
                    # Power usage is returned in milliwatts, convert to watts
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    power_w = power_mw / 1000.0
                    yield Observation(
                        value=power_w, attributes={"gpu_id": str(i), "gpu_name": device_name}
                    )
                except Exception as e:
                    logger.debug("Failed to get GPU power for GPU %d: %s", i, e)

            pynvml.nvmlShutdown()
        except Exception as e:
            logger.error("Error observing GPU power: %s", e)

    def start(self):
        """Starts the GPU metrics collection.

        ObservableGauges are automatically collected by the MeterProvider,
        so we only need to start the CO2 collection thread.
        """
        if not NVML_AVAILABLE:
            logger.warning("Cannot start GPU metrics collection - nvidia-ml-py not available")
            return

        if not self.gpu_available:
            return

        logger.info("Starting GPU metrics collection (CO2 tracking)")
        # Only start CO2 collection thread - ObservableGauges are auto-collected
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()

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

                    # Calculate and record CO2 emissions if enabled
                    if self.config.enable_co2_tracking:
                        delta_co2_g = (
                            delta_energy_wh / 1000.0
                        ) * self.config.carbon_intensity  # gCO2e
                        self.co2_counter.add(delta_co2_g, {"gpu_id": str(i)})

                    # Calculate and record power cost
                    # delta_energy_wh is in Wh, convert to kWh and multiply by cost per kWh
                    delta_cost_usd = (delta_energy_wh / 1000.0) * self.config.power_cost_per_kwh
                    device_name = self._get_device_name(handle, i)
                    self.power_cost_counter.add(
                        delta_cost_usd, {"gpu_id": str(i), "gpu_name": device_name}
                    )

                    self.last_timestamp[i] = current_time
                except Exception as e:
                    logger.error(f"Error collecting GPU {i} metrics: {e}")

    def stop(self):
        """Stops the GPU metrics collection thread."""
        # Stop CO2 collection thread
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
            logger.info("GPU CO2 metrics collection thread stopped.")

        # ObservableGauges will automatically stop when MeterProvider is shutdown
        if self.gpu_available and NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception as e:
                logger.debug("Error shutting down NVML: %s", e)
