"""Module for collecting AMD GPU metrics using amdsmi and reporting them via OpenTelemetry.

This module provides the `AMDGPUCollector` class, which collects AMD GPU metrics
including utilization, memory usage, temperature, and power consumption, and exports
these as OpenTelemetry metrics. It relies on the `amdsmi` library for interacting
with AMD GPUs.
"""

import logging
from typing import List, Optional

from opentelemetry.metrics import Meter, Observation

logger = logging.getLogger(__name__)

# Try to import amdsmi
try:
    import amdsmi

    AMDSMI_AVAILABLE = True
except ImportError:
    AMDSMI_AVAILABLE = False
    logger.debug("amdsmi not available, AMD GPU metrics will be disabled")


class AMDGPUCollector:
    """Collects and reports AMD GPU metrics using amdsmi."""

    def __init__(self, meter: Meter, config):
        """Initializes the AMDGPUCollector.

        Args:
            meter (Meter): The OpenTelemetry meter to use for recording metrics.
            config: Configuration for the collector.
        """
        self.meter = meter
        self.config = config
        self.available = False
        self.devices: List = []
        self.device_count = 0

        if not AMDSMI_AVAILABLE:
            logger.debug("amdsmi not available, AMD GPU metrics will be disabled")
            return

        try:
            # Initialize AMD SMI
            amdsmi.amdsmi_init()

            # Get device handles
            self.devices = amdsmi.amdsmi_get_processor_handles()
            self.device_count = len(self.devices)

            if self.device_count > 0:
                self.available = True
                logger.info(f"Initialized AMD GPU monitoring with {self.device_count} device(s)")
            else:
                logger.debug("No AMD GPUs detected")
        except Exception as e:
            logger.error("Failed to initialize AMD SMI: %s", e)
            self.available = False

    def _get_device_name(self, device, index: int) -> str:
        """Get AMD GPU device name safely.

        Args:
            device: AMD GPU device handle
            index: Device index

        Returns:
            str: Device name or fallback name
        """
        try:
            # Get ASIC info which contains the device name
            asic_info = amdsmi.amdsmi_get_gpu_asic_info(device)
            if isinstance(asic_info, dict) and "market_name" in asic_info:
                return asic_info["market_name"]
            elif isinstance(asic_info, dict) and "name" in asic_info:
                return asic_info["name"]
        except Exception as e:
            logger.debug("Failed to get AMD GPU name for device %d: %s", index, e)

        # Fallback name
        return f"AMD_GPU_{index}"

    def _observe_gpu_utilization(self, options):
        """Observable callback for AMD GPU utilization.

        Yields:
            Observation: GPU utilization observations
        """
        if not AMDSMI_AVAILABLE or not self.available:
            return

        try:
            amdsmi.amdsmi_init()

            for i, device in enumerate(self.devices):
                device_name = self._get_device_name(device, i)

                try:
                    # Get GPU activity
                    activity = amdsmi.amdsmi_get_gpu_activity(device)

                    # gfx_activity is the GPU core utilization percentage
                    gpu_utilization = activity.get("gfx_activity", 0)

                    yield Observation(
                        value=gpu_utilization,
                        attributes={
                            "gpu_id": str(i),
                            "gpu_vendor": "amd",
                            "gpu_name": device_name,
                        },
                    )
                except Exception as e:
                    logger.debug("Failed to get AMD GPU utilization for GPU %d: %s", i, e)

            amdsmi.amdsmi_shut_down()
        except Exception as e:
            logger.error("Error observing AMD GPU utilization: %s", e)

    def _observe_gpu_memory(self, options):
        """Observable callback for AMD GPU memory usage.

        Yields:
            Observation: GPU memory usage observations in MiB
        """
        if not AMDSMI_AVAILABLE or not self.available:
            return

        try:
            amdsmi.amdsmi_init()

            for i, device in enumerate(self.devices):
                device_name = self._get_device_name(device, i)

                try:
                    # Get VRAM usage
                    vram = amdsmi.amdsmi_get_gpu_vram_usage(device)

                    # Convert bytes to MiB
                    vram_used_bytes = vram.get("vram_used", 0)
                    vram_used_mib = vram_used_bytes / (1024**2)

                    yield Observation(
                        value=vram_used_mib,
                        attributes={
                            "gpu_id": str(i),
                            "gpu_vendor": "amd",
                            "gpu_name": device_name,
                        },
                    )
                except Exception as e:
                    logger.debug("Failed to get AMD GPU memory for GPU %d: %s", i, e)

            amdsmi.amdsmi_shut_down()
        except Exception as e:
            logger.error("Error observing AMD GPU memory: %s", e)

    def _observe_gpu_memory_total(self, options):
        """Observable callback for total AMD GPU memory capacity.

        Yields:
            Observation: Total GPU memory observations in MiB
        """
        if not AMDSMI_AVAILABLE or not self.available:
            return

        try:
            amdsmi.amdsmi_init()

            for i, device in enumerate(self.devices):
                device_name = self._get_device_name(device, i)

                try:
                    # Get VRAM usage (includes total)
                    vram = amdsmi.amdsmi_get_gpu_vram_usage(device)

                    # Convert bytes to MiB
                    vram_total_bytes = vram.get("vram_total", 0)
                    vram_total_mib = vram_total_bytes / (1024**2)

                    yield Observation(
                        value=vram_total_mib,
                        attributes={
                            "gpu_id": str(i),
                            "gpu_vendor": "amd",
                            "gpu_name": device_name,
                        },
                    )
                except Exception as e:
                    logger.debug("Failed to get total AMD GPU memory for GPU %d: %s", i, e)

            amdsmi.amdsmi_shut_down()
        except Exception as e:
            logger.error("Error observing total AMD GPU memory: %s", e)

    def _observe_gpu_temperature(self, options):
        """Observable callback for AMD GPU temperature.

        Yields:
            Observation: GPU temperature observations in Celsius
        """
        if not AMDSMI_AVAILABLE or not self.available:
            return

        try:
            amdsmi.amdsmi_init()

            for i, device in enumerate(self.devices):
                device_name = self._get_device_name(device, i)

                try:
                    # Get edge temperature in millidegrees Celsius
                    temp_milli = amdsmi.amdsmi_get_temp_metric(
                        device,
                        amdsmi.AmdSmiTemperatureType.EDGE,
                        amdsmi.AmdSmiTemperatureMetric.CURRENT,
                    )

                    # Convert millidegrees to degrees Celsius
                    temp_celsius = temp_milli / 1000.0

                    yield Observation(
                        value=temp_celsius,
                        attributes={
                            "gpu_id": str(i),
                            "gpu_vendor": "amd",
                            "gpu_name": device_name,
                        },
                    )
                except Exception as e:
                    logger.debug("Failed to get AMD GPU temperature for GPU %d: %s", i, e)

            amdsmi.amdsmi_shut_down()
        except Exception as e:
            logger.error("Error observing AMD GPU temperature: %s", e)

    def _observe_gpu_power(self, options):
        """Observable callback for AMD GPU power consumption.

        Yields:
            Observation: GPU power consumption observations in Watts
        """
        if not AMDSMI_AVAILABLE or not self.available:
            return

        try:
            amdsmi.amdsmi_init()

            for i, device in enumerate(self.devices):
                device_name = self._get_device_name(device, i)

                try:
                    # Get power info
                    power_info = amdsmi.amdsmi_get_power_info(device)

                    # Extract current socket power in watts
                    # power_info is a dictionary with 'current_socket_power'
                    power_watts = power_info.get("current_socket_power", 0) / 1000.0  # mW to W

                    yield Observation(
                        value=power_watts,
                        attributes={
                            "gpu_id": str(i),
                            "gpu_vendor": "amd",
                            "gpu_name": device_name,
                        },
                    )
                except Exception as e:
                    logger.debug("Failed to get AMD GPU power for GPU %d: %s", i, e)

            amdsmi.amdsmi_shut_down()
        except Exception as e:
            logger.error("Error observing AMD GPU power: %s", e)

    def get_power_usage(self, device_index: int) -> Optional[float]:
        """Get power usage for a specific AMD GPU device.

        Args:
            device_index: Index of the GPU device

        Returns:
            float: Power usage in watts, or None if unavailable
        """
        if not AMDSMI_AVAILABLE or not self.available:
            return None

        if device_index >= self.device_count:
            return None

        try:
            device = self.devices[device_index]
            power_info = amdsmi.amdsmi_get_power_info(device)
            # Convert milliwatts to watts
            return power_info.get("current_socket_power", 0) / 1000.0
        except Exception as e:
            logger.debug("Failed to get power usage for AMD GPU %d: %s", device_index, e)
            return None

    def shutdown(self):
        """Shutdown AMD SMI."""
        if self.available and AMDSMI_AVAILABLE:
            try:
                amdsmi.amdsmi_shut_down()
                logger.debug("AMD SMI shut down successfully")
            except Exception as e:
                logger.debug("Error shutting down AMD SMI: %s", e)
