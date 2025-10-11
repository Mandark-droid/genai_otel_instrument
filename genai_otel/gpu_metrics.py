import threading
import time
import logging
from typing import Optional
from opentelemetry.metrics import Meter
import pynvml

logger = logging.getLogger(__name__)


class GPUMetricsCollector:
    """Collect GPU metrics using pynvml with error handling"""

    def __init__(self, meter: Meter, interval: int = 10):
        self.meter = meter
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.gpu_available = False
        self.device_count = 0
        self.nvml = None

        try:
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.gpu_available = True
            self.device_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"GPU metrics available: {self.device_count} device(s) detected")

            # Create metrics
            self._create_metrics()

        except ImportError:
            logger.info("pynvml not installed, GPU metrics not available")
        except pynvml.NVMLError as e:
            logger.warning(f"GPU metrics not available: {e}")

    def _create_metrics(self):
        """Create GPU metric instruments"""
        try:
            self.gpu_utilization = self.meter.create_observable_gauge(
                "genai.gpu.utilization",
                callbacks=[self._get_gpu_utilization],
                description="GPU utilization percentage"
            )
            self.gpu_memory_used = self.meter.create_observable_gauge(
                "genai.gpu.memory.used",
                callbacks=[self._get_gpu_memory_used],
                description="GPU memory used in bytes",
                unit="By"
            )
            self.gpu_memory_total = self.meter.create_observable_gauge(
                "genai.gpu.memory.total",
                callbacks=[self._get_gpu_memory_total],
                description="GPU total memory in bytes",
                unit="By"
            )
            self.gpu_temperature = self.meter.create_observable_gauge(
                "genai.gpu.temperature",
                callbacks=[self._get_gpu_temperature],
                description="GPU temperature in Celsius",
                unit="Cel"
            )
            self.gpu_power_usage = self.meter.create_observable_gauge(
                "genai.gpu.power.usage",
                callbacks=[self._get_gpu_power],
                description="GPU power usage in watts",
                unit="W"
            )
        except Exception as e:
            logger.error(f"Failed to create GPU metrics: {e}", exc_info=True)

    def start(self):
        """Start GPU metrics collection"""
        if not self.gpu_available:
            logger.debug("GPU metrics not started (not available)")
            return
        logger.info("GPU metrics collection started")

    def _safe_metric_collection(self, metric_func, metric_name):
        """Safely collect a GPU metric"""
        if not self.gpu_available:
            return []

        observations = []
        for i in range(self.device_count):
            try:
                handle = self.nvml.nvmlDeviceGetHandleByIndex(i)
                value = metric_func(handle, i)
                if value is not None:
                    observations.append((value, {"gpu_id": str(i)}))
            except pynvml.NVMLError as e:
                logger.debug(f"Failed to collect {metric_name} for GPU {i}: {e}")

        return observations

    def _get_gpu_utilization(self, options):
        """Get GPU utilization with error handling"""

        def get_util(handle, gpu_id):
            util = self.nvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu

        return self._safe_metric_collection(get_util, "utilization")

    def _get_gpu_memory_used(self, options):
        """Get GPU memory used with error handling"""

        def get_mem(handle, gpu_id):
            info = self.nvml.nvmlDeviceGetMemoryInfo(handle)
            return info.used

        return self._safe_metric_collection(get_mem, "memory_used")

    def _get_gpu_memory_total(self, options):
        """Get GPU total memory with error handling"""

        def get_mem(handle, gpu_id):
            info = self.nvml.nvmlDeviceGetMemoryInfo(handle)
            return info.total

        return self._safe_metric_collection(get_mem, "memory_total")

    def _get_gpu_temperature(self, options):
        """Get GPU temperature with error handling"""

        def get_temp(handle, gpu_id):
            return self.nvml.nvmlDeviceGetTemperature(
                handle,
                self.nvml.NVML_TEMPERATURE_GPU
            )

        return self._safe_metric_collection(get_temp, "temperature")

    def _get_gpu_power(self, options):
        """Get GPU power usage with error handling"""

        def get_power(handle, gpu_id):
            power = self.nvml.nvmlDeviceGetPowerUsage(handle)
            return power / 1000.0  # Convert to watts

        return self._safe_metric_collection(get_power, "power")

    def stop(self):
        """Stop GPU metrics collection and cleanup"""
        if self.gpu_available and self.nvml:
            try:
                self.nvml.nvmlShutdown()
                logger.info("GPU metrics collection stopped")
            except pynvml.NVMLError as e:
                logger.error(f"Error stopping GPU metrics: {e}")