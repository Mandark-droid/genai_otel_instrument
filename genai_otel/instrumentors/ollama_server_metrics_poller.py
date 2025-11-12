"""Ollama server metrics poller for automatic KV cache and memory tracking.

This module provides automatic collection of Ollama server metrics by polling
the /api/ps endpoint to get running models and their VRAM usage.

GPU VRAM is auto-detected using:
1. pynvml (nvidia-ml-py) - preferred method
2. nvidia-smi subprocess - fallback
3. Environment variable - manual override
"""

import logging
import subprocess  # nosec B404 - Only used for nvidia-smi with hardcoded args
import threading
import time
from typing import Optional

import requests

from ..server_metrics import get_server_metrics

logger = logging.getLogger(__name__)

# Try to import nvidia-ml-py for GPU VRAM detection
# Package: nvidia-ml-py, Import: pynvml
try:
    import pynvml

    NVIDIA_ML_AVAILABLE = True
except ImportError:
    NVIDIA_ML_AVAILABLE = False


class OllamaServerMetricsPoller:
    """Polls Ollama server for metrics and updates ServerMetricsCollector.

    This poller queries the /api/ps endpoint to get:
    - Running models and their VRAM usage
    - Total VRAM usage across all models
    - Number of models currently loaded

    It automatically updates the global ServerMetricsCollector with:
    - KV cache usage per model (approximated from VRAM usage)
    - Running models count
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        interval: float = 5.0,
        max_vram_gb: Optional[float] = None,
    ):
        """Initialize the Ollama server metrics poller.

        Args:
            base_url: Base URL of the Ollama server (default: http://localhost:11434)
            interval: Polling interval in seconds (default: 5.0)
            max_vram_gb: Maximum VRAM in GB for percentage calculation.
                        If None, will attempt auto-detection via nvidia-ml-py or nvidia-smi.
        """
        self.base_url = base_url.rstrip("/")
        self.interval = interval

        # Auto-detect GPU VRAM if not provided
        if max_vram_gb is not None:
            self.max_vram_bytes = max_vram_gb * 1024**3
            logger.info(f"Using configured GPU VRAM: {max_vram_gb}GB")
        else:
            detected_vram_gb = self._detect_gpu_vram()
            if detected_vram_gb:
                self.max_vram_bytes = detected_vram_gb * 1024**3
                logger.info(f"Auto-detected GPU VRAM: {detected_vram_gb}GB")
            else:
                self.max_vram_bytes = None
                logger.info("GPU VRAM not detected, using heuristic-based percentages")

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def _detect_gpu_vram(self) -> Optional[float]:
        """Auto-detect GPU VRAM in GB using nvidia-ml-py or nvidia-smi.

        Returns:
            GPU VRAM in GB, or None if detection failed
        """
        # Try nvidia-ml-py first (preferred method)
        vram_gb = self._detect_vram_nvidia_ml()
        if vram_gb:
            return vram_gb

        # Fallback to nvidia-smi
        vram_gb = self._detect_vram_nvidia_smi()
        if vram_gb:
            return vram_gb

        return None

    def _detect_vram_nvidia_ml(self) -> Optional[float]:
        """Detect GPU VRAM using nvidia-ml-py library.

        Returns:
            GPU VRAM in GB for the first GPU, or None if failed
        """
        if not NVIDIA_ML_AVAILABLE:
            logger.debug("nvidia-ml-py not available for VRAM detection")
            return None

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            if device_count == 0:
                logger.debug("No NVIDIA GPUs detected via nvidia-ml-py")
                pynvml.nvmlShutdown()
                return None

            # Get VRAM from first GPU (Ollama typically uses GPU 0)
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_bytes = mem_info.total
            vram_gb = vram_bytes / (1024**3)

            # Get GPU name for logging
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            logger.debug(f"Detected via nvidia-ml-py: {gpu_name} with {vram_gb:.2f}GB VRAM")

            pynvml.nvmlShutdown()
            return vram_gb

        except Exception as e:
            logger.debug(f"Failed to detect VRAM via nvidia-ml-py: {e}")
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            return None

    def _detect_vram_nvidia_smi(self) -> Optional[float]:
        """Detect GPU VRAM using nvidia-smi command.

        Returns:
            GPU VRAM in GB for the first GPU, or None if failed
        """
        try:
            # Query nvidia-smi for total memory of GPU 0
            result = subprocess.run(  # nosec B603 B607
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                    "--id=0",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )

            # Parse output (in MB)
            vram_mb = float(result.stdout.strip())
            vram_gb = vram_mb / 1024

            logger.debug(f"Detected via nvidia-smi: {vram_gb:.2f}GB VRAM")
            return vram_gb

        except FileNotFoundError:
            logger.debug("nvidia-smi command not found")
            return None
        except subprocess.TimeoutExpired:
            logger.debug("nvidia-smi command timed out")
            return None
        except Exception as e:
            logger.debug(f"Failed to detect VRAM via nvidia-smi: {e}")
            return None

    def start(self):
        """Start the background polling thread."""
        if self._running:
            logger.warning("Ollama server metrics poller already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        self._running = True
        logger.info(
            f"Ollama server metrics poller started (interval={self.interval}s, url={self.base_url})"
        )

    def stop(self):
        """Stop the background polling thread."""
        if not self._running:
            return

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 1.0)
        self._running = False
        logger.info("Ollama server metrics poller stopped")

    def _poll_loop(self):
        """Background polling loop that runs in a separate thread."""
        while not self._stop_event.is_set():
            try:
                self._collect_metrics()
            except Exception as e:
                logger.warning(f"Failed to collect Ollama server metrics: {e}")

            # Wait for interval or until stop is requested
            self._stop_event.wait(self.interval)

    def _collect_metrics(self):
        """Query Ollama /api/ps and update server metrics."""
        try:
            # Query /api/ps endpoint
            response = requests.get(f"{self.base_url}/api/ps", timeout=2.0)
            response.raise_for_status()
            data = response.json()

            # Get server metrics collector
            server_metrics = get_server_metrics()
            if not server_metrics:
                logger.debug("Server metrics collector not initialized, skipping update")
                return

            # Extract running models
            models = data.get("models", [])
            num_models = len(models)

            # Update the "max" capacity to reflect number of model slots
            # This gives visibility into how many models can be loaded simultaneously
            # Note: This is approximate - actual capacity depends on VRAM
            if num_models > 0:
                server_metrics.set_requests_max(num_models)

            logger.debug(f"Ollama has {num_models} models loaded in memory")

            # Process each model's VRAM usage and details
            total_vram_bytes = 0
            total_size_bytes = 0

            for model_info in models:
                model_name = model_info.get("name", "unknown")
                size_vram = model_info.get("size_vram", 0)
                size_total = model_info.get("size", 0)
                total_vram_bytes += size_vram
                total_size_bytes += size_total

                # Calculate VRAM usage percentage for this model
                if self.max_vram_bytes and self.max_vram_bytes > 0:
                    # If we know max VRAM, calculate actual percentage
                    vram_usage_pct = (size_vram / self.max_vram_bytes) * 100
                    vram_usage_pct = min(100.0, vram_usage_pct)  # Cap at 100%
                else:
                    # If we don't know max VRAM, use a simple heuristic:
                    # Models in memory are "using" the cache, so report a non-zero value
                    # This is an approximation - actual KV cache varies with context
                    vram_usage_pct = min(100.0, (size_vram / (8 * 1024**3)) * 100)

                # Update KV cache usage for this model
                server_metrics.set_kv_cache_usage(model_name, vram_usage_pct)

                # Extract model details for logging
                details = model_info.get("details", {})
                param_size = details.get("parameter_size", "unknown")
                quant_level = details.get("quantization_level", "unknown")
                model_format = details.get("format", "unknown")

                logger.debug(
                    f"Model {model_name}: "
                    f"VRAM={size_vram / 1024**3:.2f}GB ({vram_usage_pct:.1f}%), "
                    f"Size={size_total / 1024**3:.2f}GB, "
                    f"Params={param_size}, Quant={quant_level}, Format={model_format}"
                )

            # Log aggregate metrics
            if total_vram_bytes > 0:
                logger.debug(
                    f"Total Ollama usage: VRAM={total_vram_bytes / 1024**3:.2f}GB, "
                    f"Total={total_size_bytes / 1024**3:.2f}GB, "
                    f"Models={num_models}"
                )

            # Note: We don't update requests.running here because that's tracked
            # automatically by BaseInstrumentor when requests are made

        except requests.exceptions.ConnectionError:
            logger.debug(f"Cannot connect to Ollama server at {self.base_url}")
        except requests.exceptions.Timeout:
            logger.warning("Ollama server request timed out")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error querying Ollama server: {e}")
        except Exception as e:
            logger.error(f"Unexpected error collecting Ollama metrics: {e}", exc_info=True)


# Global instance
_global_poller: Optional[OllamaServerMetricsPoller] = None


def start_ollama_metrics_poller(
    base_url: str = "http://localhost:11434",
    interval: float = 5.0,
    max_vram_gb: Optional[float] = None,
) -> OllamaServerMetricsPoller:
    """Start the global Ollama server metrics poller.

    Args:
        base_url: Base URL of the Ollama server (default: http://localhost:11434)
        interval: Polling interval in seconds (default: 5.0)
        max_vram_gb: Maximum VRAM in GB for percentage calculation (optional)

    Returns:
        The global poller instance
    """
    global _global_poller

    if _global_poller is not None and _global_poller._running:
        logger.warning("Ollama metrics poller already running")
        return _global_poller

    _global_poller = OllamaServerMetricsPoller(
        base_url=base_url, interval=interval, max_vram_gb=max_vram_gb
    )
    _global_poller.start()
    return _global_poller


def stop_ollama_metrics_poller():
    """Stop the global Ollama server metrics poller."""
    global _global_poller

    if _global_poller is not None:
        _global_poller.stop()
        _global_poller = None
