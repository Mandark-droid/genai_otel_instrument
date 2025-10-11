# genai_otel/gpu_metrics.py

import logging
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import GPU monitoring libraries
try:
    # import nvidia-ml-py as nvml
    import pynvml as nvml

    HAS_NVML = True
    logger.debug("nvidia-ml-py available for GPU monitoring")
except ImportError:
    try:
        # Fallback to pynvml for backward compatibility
        import pynvml as nvml

        HAS_NVML = True
        logger.warning(
            "pynvml is deprecated. Please install nvidia-ml-py instead for GPU monitoring."
        )
    except ImportError:
        HAS_NVML = False
        logger.debug("GPU monitoring not available (nvidia-ml-py or pynvml not installed)")


class GPUMetricsCollector:
    """Collect GPU metrics if available."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and HAS_NVML
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._metrics: Dict[str, Any] = {}

        if self.enabled:
            try:
                nvml.nvmlInit()
                self.device_count = nvml.nvmlDeviceGetCount()
                logger.info(f"GPU monitoring enabled for {self.device_count} devices")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU monitoring: {e}")
                self.enabled = False
        else:
            logger.debug("GPU monitoring disabled")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current GPU metrics."""
        if not self.enabled:
            return {"gpu_available": False}

        try:
            metrics = {"gpu_available": True, "devices": []}

            for i in range(self.device_count):
                try:
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)

                    # Get utilization
                    utilization = nvml.nvmlDeviceGetUtilizationRates(handle)

                    # Get memory info
                    memory = nvml.nvmlDeviceGetMemoryInfo(handle)

                    # Get temperature
                    temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)

                    device_metrics = {
                        "device_id": i,
                        "utilization_gpu": utilization.gpu,
                        "utilization_memory": utilization.memory,
                        "memory_used": memory.used,
                        "memory_total": memory.total,
                        "memory_free": memory.free,
                        "temperature": temp,
                    }
                    metrics["devices"].append(device_metrics)

                except Exception as e:
                    logger.warning(f"Failed to get metrics for GPU {i}: {e}")
                    continue

            return metrics

        except Exception as e:
            logger.warning(f"Failed to collect GPU metrics: {e}")
            return {"gpu_available": False, "error": str(e)}

    def start_collecting(self, interval: int = 30):
        """Start collecting metrics in background thread."""
        if not self.enabled:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._collect_loop, args=(interval,), daemon=True)
        self._thread.start()
        logger.debug(f"Started GPU metrics collection with {interval}s interval")

    def _collect_loop(self, interval: int):
        """Background collection loop."""
        while not self._stop_event.is_set():
            try:
                self._metrics = self.get_metrics()
            except Exception as e:
                logger.warning(f"Error in GPU metrics collection: {e}")

            self._stop_event.wait(interval)

    def stop_collecting(self):
        """Stop background collection."""
        if self._thread:
            self._stop_event.set()
            self._thread.join(timeout=5)
            self._thread = None
            logger.debug("Stopped GPU metrics collection")

    def __del__(self):
        """Cleanup on destruction."""
        self.stop_collecting()
        if self.enabled:
            try:
                nvml.nvmlShutdown()
            except:
                pass


# Convenience function that handles missing dependencies
def create_gpu_collector(enabled: bool = True) -> Optional[GPUMetricsCollector]:
    """Create a GPU collector if dependencies are available."""
    if not HAS_NVML:
        if enabled:
            logger.debug("GPU monitoring requested but nvidia-ml-py not available")
        return None
    return GPUMetricsCollector(enabled=enabled)
