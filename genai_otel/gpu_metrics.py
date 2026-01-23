"""Module for collecting GPU metrics using nvidia-ml-py and amdsmi, reporting them via OpenTelemetry.

This module provides the `GPUMetricsCollector` class, which periodically collects
GPU utilization, memory usage, and temperature, and exports these as OpenTelemetry
metrics. It supports both NVIDIA GPUs (via nvidia-ml-py) and AMD GPUs (via amdsmi).

CO2 emissions tracking is provided via codecarbon integration, which offers:
- Automatic region-based carbon intensity lookup
- Cloud provider carbon intensity data
- More accurate emission factors based on location
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
    logger.debug("nvidia-ml-py not available, NVIDIA GPU metrics will be disabled")

# Try to import AMD GPU collector
try:
    from genai_otel.gpu_metrics_amd import AMDSMI_AVAILABLE, AMDGPUCollector
except ImportError:
    AMDSMI_AVAILABLE = False
    AMDGPUCollector = None
    logger.debug("AMD GPU collector not available")

# Try to import codecarbon for CO2 emissions tracking
try:
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker

    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    EmissionsTracker = None  # type: ignore
    OfflineEmissionsTracker = None  # type: ignore
    logger.debug("codecarbon not available, will use manual CO2 calculation")


class GPUMetricsCollector:
    """Collects and reports GPU metrics using nvidia-ml-py/amdsmi and codecarbon for CO2 tracking."""

    def __init__(self, meter: Meter, config: OTelConfig, interval: int = 10):
        """Initializes the GPUMetricsCollector.

        Args:
            meter (Meter): The OpenTelemetry meter to use for recording metrics.
            config (OTelConfig): Configuration for the collector.
            interval (int): Collection interval in seconds.
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
        # Enhanced GPU metrics
        self.gpu_memory_utilization_gauge: Optional[ObservableGauge] = None
        self.gpu_power_limit_gauge: Optional[ObservableGauge] = None
        self.gpu_sm_clock_gauge: Optional[ObservableGauge] = None
        self.gpu_memory_clock_gauge: Optional[ObservableGauge] = None
        self.gpu_fan_speed_gauge: Optional[ObservableGauge] = None
        self.gpu_performance_state_gauge: Optional[ObservableGauge] = None
        self.gpu_pcie_tx_gauge: Optional[ObservableGauge] = None
        self.gpu_pcie_rx_gauge: Optional[ObservableGauge] = None
        self.gpu_throttle_thermal_gauge: Optional[ObservableGauge] = None
        self.gpu_throttle_power_gauge: Optional[ObservableGauge] = None
        self.gpu_throttle_hw_slowdown_gauge: Optional[ObservableGauge] = None
        self.gpu_ecc_errors_corrected_gauge: Optional[ObservableGauge] = None
        self.gpu_ecc_errors_uncorrected_gauge: Optional[ObservableGauge] = None
        # Aggregate GPU metrics (across all GPUs)
        self.gpu_aggregate_mean_utilization_gauge: Optional[ObservableGauge] = None
        self.gpu_aggregate_total_memory_used_gauge: Optional[ObservableGauge] = None
        self.gpu_aggregate_total_power_gauge: Optional[ObservableGauge] = None
        self.gpu_aggregate_max_temperature_gauge: Optional[ObservableGauge] = None
        self.config = config
        self.interval = interval  # seconds
        self.gpu_available = False

        # Multi-vendor GPU support
        self.amd_collector: Optional["AMDGPUCollector"] = None
        self.nvidia_device_count = 0
        self.amd_device_count = 0

        # Codecarbon emissions tracker
        self._emissions_tracker: Optional["EmissionsTracker"] = None
        self._last_emissions_kg: float = 0.0
        self._use_codecarbon: bool = False

        # Initialize NVIDIA GPUs
        self.device_count = 0
        self.nvml = None
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvidia_device_count = pynvml.nvmlDeviceGetCount()
                self.device_count = self.nvidia_device_count  # For backward compatibility
                if self.nvidia_device_count > 0:
                    self.gpu_available = True
                    logger.info(
                        f"Initialized NVIDIA GPU monitoring with {self.nvidia_device_count} device(s)"
                    )
                self.nvml = pynvml
            except Exception as e:
                logger.error("Failed to initialize NVML to get device count: %s", e)

        # Initialize AMD GPUs
        if AMDSMI_AVAILABLE and AMDGPUCollector is not None:
            try:
                self.amd_collector = AMDGPUCollector(meter, config)
                if self.amd_collector.available:
                    self.amd_device_count = self.amd_collector.device_count
                    self.gpu_available = True
                else:
                    self.amd_collector = None
            except Exception as e:
                logger.error("Failed to initialize AMD GPU collector: %s", e)
                self.amd_collector = None

        self.cumulative_energy_wh = [0.0] * self.device_count  # Per GPU, in Wh
        self.last_timestamp = [time.time()] * self.device_count
        self.co2_counter = meter.create_counter(
            "gen_ai.co2.emissions",
            description="Cumulative CO2 equivalent emissions in grams",
            unit="gCO2e",
        )
        self.power_cost_counter = meter.create_counter(
            "gen_ai.power.cost",
            description="Cumulative electricity cost in USD based on power consumption",
            unit="USD",
        )
        self.energy_counter = meter.create_counter(
            "gen_ai.energy.consumed",
            description="Cumulative energy consumed by component (CPU/GPU/RAM)",
            unit="kWh",
        )
        self.total_energy_counter = meter.create_counter(
            "gen_ai.energy.total",
            description="Total cumulative energy consumed (sum of CPU+GPU+RAM)",
            unit="kWh",
        )
        self.emissions_rate_gauge = meter.create_histogram(
            "gen_ai.co2.emissions_rate",
            description="CO2 emissions rate (rate of emissions per second)",
            unit="gCO2e/s",
        )
        self.power_gauge = meter.create_histogram(
            "gen_ai.power.consumption",
            description="Power consumption by component (CPU/GPU/RAM)",
            unit="W",
        )
        self.task_duration_histogram = meter.create_histogram(
            "gen_ai.codecarbon.task.duration",
            description="Duration of codecarbon monitoring tasks",
            unit="s",
        )

        # Initialize codecarbon if available and CO2 tracking is enabled
        self._init_codecarbon()

        if not NVML_AVAILABLE and not (AMDSMI_AVAILABLE and self.amd_collector):
            logger.warning(
                "GPU metrics collection not available - neither nvidia-ml-py nor amdsmi installed. "
                "Install with: pip install genai-otel-instrument[all-gpu]"
            )
            return

        try:
            # Collect callbacks from available GPU vendors
            utilization_callbacks = []
            memory_callbacks = []
            memory_total_callbacks = []
            temperature_callbacks = []
            power_callbacks = []
            # Enhanced metrics callbacks (NVIDIA-only for now)
            memory_utilization_callbacks = []
            power_limit_callbacks = []
            sm_clock_callbacks = []
            memory_clock_callbacks = []
            fan_speed_callbacks = []
            performance_state_callbacks = []
            pcie_tx_callbacks = []
            pcie_rx_callbacks = []
            throttle_thermal_callbacks = []
            throttle_power_callbacks = []
            throttle_hw_slowdown_callbacks = []
            ecc_corrected_callbacks = []
            ecc_uncorrected_callbacks = []
            # Aggregate metrics callbacks
            aggregate_mean_utilization_callbacks = []
            aggregate_total_memory_callbacks = []
            aggregate_total_power_callbacks = []
            aggregate_max_temp_callbacks = []

            # Add NVIDIA callbacks if available
            if NVML_AVAILABLE and self.nvidia_device_count > 0:
                utilization_callbacks.append(self._observe_gpu_utilization)
                memory_callbacks.append(self._observe_gpu_memory)
                memory_total_callbacks.append(self._observe_gpu_memory_total)
                temperature_callbacks.append(self._observe_gpu_temperature)
                power_callbacks.append(self._observe_gpu_power)
                # Enhanced NVIDIA-specific metrics
                memory_utilization_callbacks.append(self._observe_memory_utilization)
                power_limit_callbacks.append(self._observe_power_limit)
                sm_clock_callbacks.append(self._observe_sm_clock)
                memory_clock_callbacks.append(self._observe_memory_clock)
                fan_speed_callbacks.append(self._observe_fan_speed)
                performance_state_callbacks.append(self._observe_performance_state)
                pcie_tx_callbacks.append(self._observe_pcie_tx)
                pcie_rx_callbacks.append(self._observe_pcie_rx)
                throttle_thermal_callbacks.append(self._observe_throttle_thermal)
                throttle_power_callbacks.append(self._observe_throttle_power)
                throttle_hw_slowdown_callbacks.append(self._observe_throttle_hw_slowdown)
                ecc_corrected_callbacks.append(self._observe_ecc_errors_corrected)
                ecc_uncorrected_callbacks.append(self._observe_ecc_errors_uncorrected)
                # Aggregate metrics (NVIDIA)
                aggregate_mean_utilization_callbacks.append(
                    self._observe_aggregate_mean_utilization
                )
                aggregate_total_memory_callbacks.append(self._observe_aggregate_total_memory)
                aggregate_total_power_callbacks.append(self._observe_aggregate_total_power)
                aggregate_max_temp_callbacks.append(self._observe_aggregate_max_temperature)

            # Add AMD callbacks if available
            if self.amd_collector and self.amd_collector.available:
                utilization_callbacks.append(self.amd_collector._observe_gpu_utilization)
                memory_callbacks.append(self.amd_collector._observe_gpu_memory)
                memory_total_callbacks.append(self.amd_collector._observe_gpu_memory_total)
                temperature_callbacks.append(self.amd_collector._observe_gpu_temperature)
                power_callbacks.append(self.amd_collector._observe_gpu_power)

            # Use ObservableGauge for all GPU metrics (not Counter!)
            if utilization_callbacks:
                self.gpu_utilization_gauge = self.meter.create_observable_gauge(
                    "gen_ai.gpu.utilization",  # Fixed metric name
                    callbacks=utilization_callbacks,
                    description="GPU utilization percentage",
                    unit="%",
                )
            if memory_callbacks:
                self.gpu_memory_used_gauge = self.meter.create_observable_gauge(
                    "gen_ai.gpu.memory.used",  # Fixed metric name
                    callbacks=memory_callbacks,
                    description="GPU memory used in MiB",
                    unit="MiB",
                )
            if memory_total_callbacks:
                self.gpu_memory_total_gauge = self.meter.create_observable_gauge(
                    "gen_ai.gpu.memory.total",  # Fixed metric name
                    callbacks=memory_total_callbacks,
                    description="Total GPU memory capacity in MiB",
                    unit="MiB",
                )
            if temperature_callbacks:
                self.gpu_temperature_gauge = self.meter.create_observable_gauge(
                    "gen_ai.gpu.temperature",  # Fixed metric name
                    callbacks=temperature_callbacks,
                    description="GPU temperature in Celsius",
                    unit="Cel",
                )
            if power_callbacks:
                self.gpu_power_gauge = self.meter.create_observable_gauge(
                    "gen_ai.gpu.power",  # Fixed metric name
                    callbacks=power_callbacks,
                    description="GPU power consumption in Watts",
                    unit="W",
                )

            # Enhanced GPU metrics
            if memory_utilization_callbacks:
                self.gpu_memory_utilization_gauge = self.meter.create_observable_gauge(
                    "gen_ai.gpu.memory.utilization",
                    callbacks=memory_utilization_callbacks,
                    description="GPU memory controller utilization percentage",
                    unit="%",
                )
            if power_limit_callbacks:
                self.gpu_power_limit_gauge = self.meter.create_observable_gauge(
                    "gen_ai.gpu.power.limit",
                    callbacks=power_limit_callbacks,
                    description="GPU power limit in Watts",
                    unit="W",
                )
            if sm_clock_callbacks:
                self.gpu_sm_clock_gauge = self.meter.create_observable_gauge(
                    "gen_ai.gpu.clock.sm",
                    callbacks=sm_clock_callbacks,
                    description="GPU SM (streaming multiprocessor) clock speed in MHz",
                    unit="MHz",
                )
            if memory_clock_callbacks:
                self.gpu_memory_clock_gauge = self.meter.create_observable_gauge(
                    "gen_ai.gpu.clock.memory",
                    callbacks=memory_clock_callbacks,
                    description="GPU memory clock speed in MHz",
                    unit="MHz",
                )
            if fan_speed_callbacks:
                self.gpu_fan_speed_gauge = self.meter.create_observable_gauge(
                    "gen_ai.gpu.fan.speed",
                    callbacks=fan_speed_callbacks,
                    description="GPU fan speed percentage",
                    unit="%",
                )
            if performance_state_callbacks:
                self.gpu_performance_state_gauge = self.meter.create_observable_gauge(
                    "gen_ai.gpu.performance.state",
                    callbacks=performance_state_callbacks,
                    description="GPU performance state (P-state: 0=P0 highest, 15=P15 lowest)",
                    unit="1",
                )
            if pcie_tx_callbacks:
                self.gpu_pcie_tx_gauge = self.meter.create_observable_gauge(
                    "gen_ai.gpu.pcie.tx",
                    callbacks=pcie_tx_callbacks,
                    description="GPU PCIe transmit throughput in KB/s",
                    unit="KB/s",
                )
            if pcie_rx_callbacks:
                self.gpu_pcie_rx_gauge = self.meter.create_observable_gauge(
                    "gen_ai.gpu.pcie.rx",
                    callbacks=pcie_rx_callbacks,
                    description="GPU PCIe receive throughput in KB/s",
                    unit="KB/s",
                )
            if throttle_thermal_callbacks:
                self.gpu_throttle_thermal_gauge = self.meter.create_observable_gauge(
                    "gen_ai.gpu.throttle.thermal",
                    callbacks=throttle_thermal_callbacks,
                    description="GPU thermal throttling active (1=throttling, 0=not throttling)",
                    unit="1",
                )
            if throttle_power_callbacks:
                self.gpu_throttle_power_gauge = self.meter.create_observable_gauge(
                    "gen_ai.gpu.throttle.power",
                    callbacks=throttle_power_callbacks,
                    description="GPU power throttling active (1=throttling, 0=not throttling)",
                    unit="1",
                )
            if throttle_hw_slowdown_callbacks:
                self.gpu_throttle_hw_slowdown_gauge = self.meter.create_observable_gauge(
                    "gen_ai.gpu.throttle.hw_slowdown",
                    callbacks=throttle_hw_slowdown_callbacks,
                    description="GPU hardware slowdown active (1=slowdown, 0=normal)",
                    unit="1",
                )
            if ecc_corrected_callbacks:
                self.gpu_ecc_errors_corrected_gauge = self.meter.create_observable_gauge(
                    "gen_ai.gpu.ecc.errors.corrected",
                    callbacks=ecc_corrected_callbacks,
                    description="GPU ECC corrected memory errors count",
                    unit="1",
                )
            if ecc_uncorrected_callbacks:
                self.gpu_ecc_errors_uncorrected_gauge = self.meter.create_observable_gauge(
                    "gen_ai.gpu.ecc.errors.uncorrected",
                    callbacks=ecc_uncorrected_callbacks,
                    description="GPU ECC uncorrected memory errors count",
                    unit="1",
                )

            # Aggregate GPU metrics (across all GPUs)
            if aggregate_mean_utilization_callbacks:
                self.gpu_aggregate_mean_utilization_gauge = self.meter.create_observable_gauge(
                    "gen_ai.gpu.aggregate.mean_utilization",
                    callbacks=aggregate_mean_utilization_callbacks,
                    description="Mean GPU utilization across all GPUs",
                    unit="%",
                )
            if aggregate_total_memory_callbacks:
                self.gpu_aggregate_total_memory_used_gauge = self.meter.create_observable_gauge(
                    "gen_ai.gpu.aggregate.total_memory_used",
                    callbacks=aggregate_total_memory_callbacks,
                    description="Total GPU memory used across all GPUs in GiB",
                    unit="GiB",
                )
            if aggregate_total_power_callbacks:
                self.gpu_aggregate_total_power_gauge = self.meter.create_observable_gauge(
                    "gen_ai.gpu.aggregate.total_power",
                    callbacks=aggregate_total_power_callbacks,
                    description="Total power consumption across all GPUs in Watts",
                    unit="W",
                )
            if aggregate_max_temp_callbacks:
                self.gpu_aggregate_max_temperature_gauge = self.meter.create_observable_gauge(
                    "gen_ai.gpu.aggregate.max_temperature",
                    callbacks=aggregate_max_temp_callbacks,
                    description="Maximum temperature across all GPUs in Celsius",
                    unit="Cel",
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

    def _init_codecarbon(self):
        """Initialize codecarbon EmissionsTracker if available and CO2 tracking is enabled."""
        if not self.config.enable_co2_tracking:
            logger.debug("CO2 tracking disabled, skipping codecarbon initialization")
            return

        # Check if user wants to force manual calculation
        if self.config.co2_use_manual:
            logger.info(
                "Using manual CO2 calculation (GENAI_CO2_USE_MANUAL=true) with "
                "carbon_intensity=%s gCO2e/kWh",
                self.config.carbon_intensity,
            )
            return

        if not CODECARBON_AVAILABLE:
            logger.info(
                "codecarbon not installed, using manual CO2 calculation with "
                "carbon_intensity=%s gCO2e/kWh. Install codecarbon for automatic "
                "region-based carbon intensity: pip install genai-otel-instrument[co2]",
                self.config.carbon_intensity,
            )
            return

        try:
            # Suppress codecarbon's verbose logging before initialization
            # This prevents warnings about CPU tracking mode, multiple instances, etc.
            import logging as stdlib_logging

            # Map log level string to logging constants
            log_level_map = {
                "debug": stdlib_logging.DEBUG,
                "info": stdlib_logging.INFO,
                "warning": stdlib_logging.WARNING,
                "error": stdlib_logging.ERROR,
                "critical": stdlib_logging.CRITICAL,
            }
            codecarbon_log_level = log_level_map.get(
                self.config.codecarbon_log_level.lower(), stdlib_logging.ERROR
            )

            codecarbon_logger = stdlib_logging.getLogger("codecarbon")
            codecarbon_logger.setLevel(codecarbon_log_level)

            # Build codecarbon configuration from OTelConfig
            tracker_kwargs = {
                "project_name": self.config.service_name,
                "measure_power_secs": self.config.gpu_collection_interval,
                "save_to_file": False,  # We report via OpenTelemetry, not CSV
                "save_to_api": False,  # Don't send to codecarbon API
                "logging_logger": logger,  # Use our logger
                "log_level": self.config.codecarbon_log_level.lower(),  # Use configured log level
            }

            # Tracking mode: "machine" (all processes) or "process" (current only)
            tracker_kwargs["tracking_mode"] = self.config.co2_tracking_mode

            # Determine country code for offline mode
            country_code = self.config.co2_country_iso_code
            if self.config.co2_offline_mode and not country_code:
                # Default to USA if not specified in offline mode
                country_code = "USA"
                logger.debug(
                    "No country ISO code specified for offline mode, defaulting to USA. "
                    "Set GENAI_CO2_COUNTRY_ISO_CODE for accurate carbon intensity."
                )

            # Use OfflineEmissionsTracker for offline mode, EmissionsTracker otherwise
            if self.config.co2_offline_mode:
                # OfflineEmissionsTracker requires country_iso_code
                tracker_kwargs["country_iso_code"] = country_code

                # Optional region within country (e.g., "california")
                if self.config.co2_region:
                    tracker_kwargs["region"] = self.config.co2_region

                # Cloud provider configuration for more accurate carbon intensity
                if self.config.co2_cloud_provider:
                    tracker_kwargs["cloud_provider"] = self.config.co2_cloud_provider
                if self.config.co2_cloud_region:
                    tracker_kwargs["cloud_region"] = self.config.co2_cloud_region

                self._emissions_tracker = OfflineEmissionsTracker(**tracker_kwargs)
            else:
                # Online mode - EmissionsTracker can auto-detect location
                if self.config.co2_cloud_provider:
                    tracker_kwargs["cloud_provider"] = self.config.co2_cloud_provider
                if self.config.co2_cloud_region:
                    tracker_kwargs["cloud_region"] = self.config.co2_cloud_region

                self._emissions_tracker = EmissionsTracker(**tracker_kwargs)

            self._use_codecarbon = True
            logger.info(
                "Codecarbon initialized for CO2 tracking (offline=%s, country=%s, region=%s)",
                self.config.co2_offline_mode,
                country_code or "auto-detect",
                self.config.co2_region or "auto-detect",
            )
        except Exception as e:
            logger.warning(
                "Failed to initialize codecarbon, falling back to manual CO2 calculation: %s", e
            )
            self._use_codecarbon = False

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

    # ==================== Enhanced GPU Metrics Callbacks ====================

    def _observe_memory_utilization(self, options):
        """Observable callback for GPU memory controller utilization."""
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
                        value=utilization.memory,
                        attributes={"gpu_id": str(i), "gpu_name": device_name},
                    )
                except Exception as e:
                    logger.debug("Failed to get memory utilization for GPU %d: %s", i, e)

            pynvml.nvmlShutdown()
        except Exception as e:
            logger.error("Error observing memory utilization: %s", e)

    def _observe_power_limit(self, options):
        """Observable callback for GPU power limit."""
        if not NVML_AVAILABLE or not self.gpu_available:
            return

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                device_name = self._get_device_name(handle, i)

                try:
                    # Power limit is returned in milliwatts, convert to watts
                    power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                    power_limit_w = power_limit_mw / 1000.0
                    yield Observation(
                        value=power_limit_w,
                        attributes={"gpu_id": str(i), "gpu_name": device_name},
                    )
                except Exception as e:
                    logger.debug("Failed to get power limit for GPU %d: %s", i, e)

            pynvml.nvmlShutdown()
        except Exception as e:
            logger.error("Error observing power limit: %s", e)

    def _observe_sm_clock(self, options):
        """Observable callback for GPU SM clock speed."""
        if not NVML_AVAILABLE or not self.gpu_available:
            return

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                device_name = self._get_device_name(handle, i)

                try:
                    sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                    yield Observation(
                        value=sm_clock,
                        attributes={"gpu_id": str(i), "gpu_name": device_name},
                    )
                except Exception as e:
                    logger.debug("Failed to get SM clock for GPU %d: %s", i, e)

            pynvml.nvmlShutdown()
        except Exception as e:
            logger.error("Error observing SM clock: %s", e)

    def _observe_memory_clock(self, options):
        """Observable callback for GPU memory clock speed."""
        if not NVML_AVAILABLE or not self.gpu_available:
            return

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                device_name = self._get_device_name(handle, i)

                try:
                    mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                    yield Observation(
                        value=mem_clock,
                        attributes={"gpu_id": str(i), "gpu_name": device_name},
                    )
                except Exception as e:
                    logger.debug("Failed to get memory clock for GPU %d: %s", i, e)

            pynvml.nvmlShutdown()
        except Exception as e:
            logger.error("Error observing memory clock: %s", e)

    def _observe_fan_speed(self, options):
        """Observable callback for GPU fan speed."""
        if not NVML_AVAILABLE or not self.gpu_available:
            return

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                device_name = self._get_device_name(handle, i)

                try:
                    fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                    yield Observation(
                        value=fan_speed,
                        attributes={"gpu_id": str(i), "gpu_name": device_name},
                    )
                except Exception as e:
                    # Fan speed may not be available on all GPUs (e.g., passively cooled)
                    logger.debug("Failed to get fan speed for GPU %d: %s", i, e)

            pynvml.nvmlShutdown()
        except Exception as e:
            logger.error("Error observing fan speed: %s", e)

    def _observe_performance_state(self, options):
        """Observable callback for GPU performance state (P-state)."""
        if not NVML_AVAILABLE or not self.gpu_available:
            return

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                device_name = self._get_device_name(handle, i)

                try:
                    # P-state is returned as an enum (P0-P15), we convert to int
                    pstate = pynvml.nvmlDeviceGetPerformanceState(handle)
                    yield Observation(
                        value=int(pstate),
                        attributes={"gpu_id": str(i), "gpu_name": device_name},
                    )
                except Exception as e:
                    logger.debug("Failed to get performance state for GPU %d: %s", i, e)

            pynvml.nvmlShutdown()
        except Exception as e:
            logger.error("Error observing performance state: %s", e)

    def _observe_pcie_tx(self, options):
        """Observable callback for GPU PCIe TX throughput."""
        if not NVML_AVAILABLE or not self.gpu_available:
            return

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                device_name = self._get_device_name(handle, i)

                try:
                    # Returns throughput in KB/s
                    tx_throughput = pynvml.nvmlDeviceGetPcieThroughput(
                        handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
                    )
                    yield Observation(
                        value=tx_throughput,
                        attributes={"gpu_id": str(i), "gpu_name": device_name},
                    )
                except Exception as e:
                    logger.debug("Failed to get PCIe TX throughput for GPU %d: %s", i, e)

            pynvml.nvmlShutdown()
        except Exception as e:
            logger.error("Error observing PCIe TX throughput: %s", e)

    def _observe_pcie_rx(self, options):
        """Observable callback for GPU PCIe RX throughput."""
        if not NVML_AVAILABLE or not self.gpu_available:
            return

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                device_name = self._get_device_name(handle, i)

                try:
                    # Returns throughput in KB/s
                    rx_throughput = pynvml.nvmlDeviceGetPcieThroughput(
                        handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
                    )
                    yield Observation(
                        value=rx_throughput,
                        attributes={"gpu_id": str(i), "gpu_name": device_name},
                    )
                except Exception as e:
                    logger.debug("Failed to get PCIe RX throughput for GPU %d: %s", i, e)

            pynvml.nvmlShutdown()
        except Exception as e:
            logger.error("Error observing PCIe RX throughput: %s", e)

    def _observe_throttle_thermal(self, options):
        """Observable callback for GPU thermal throttling status."""
        if not NVML_AVAILABLE or not self.gpu_available:
            return

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                device_name = self._get_device_name(handle, i)

                try:
                    throttle_reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                    # Check if thermal throttling bit is set
                    is_thermal_throttling = (
                        1
                        if throttle_reasons & pynvml.nvmlClocksThrottleReasonSwThermalSlowdown
                        else 0
                    )
                    yield Observation(
                        value=is_thermal_throttling,
                        attributes={"gpu_id": str(i), "gpu_name": device_name},
                    )
                except Exception as e:
                    logger.debug("Failed to get thermal throttle status for GPU %d: %s", i, e)

            pynvml.nvmlShutdown()
        except Exception as e:
            logger.error("Error observing thermal throttle status: %s", e)

    def _observe_throttle_power(self, options):
        """Observable callback for GPU power throttling status."""
        if not NVML_AVAILABLE or not self.gpu_available:
            return

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                device_name = self._get_device_name(handle, i)

                try:
                    throttle_reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                    # Check if power throttling bit is set
                    is_power_throttling = (
                        1 if throttle_reasons & pynvml.nvmlClocksThrottleReasonSwPowerCap else 0
                    )
                    yield Observation(
                        value=is_power_throttling,
                        attributes={"gpu_id": str(i), "gpu_name": device_name},
                    )
                except Exception as e:
                    logger.debug("Failed to get power throttle status for GPU %d: %s", i, e)

            pynvml.nvmlShutdown()
        except Exception as e:
            logger.error("Error observing power throttle status: %s", e)

    def _observe_throttle_hw_slowdown(self, options):
        """Observable callback for GPU hardware slowdown status."""
        if not NVML_AVAILABLE or not self.gpu_available:
            return

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                device_name = self._get_device_name(handle, i)

                try:
                    throttle_reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                    # Check if hardware slowdown bit is set
                    is_hw_slowdown = (
                        1 if throttle_reasons & pynvml.nvmlClocksThrottleReasonHwSlowdown else 0
                    )
                    yield Observation(
                        value=is_hw_slowdown,
                        attributes={"gpu_id": str(i), "gpu_name": device_name},
                    )
                except Exception as e:
                    logger.debug("Failed to get HW slowdown status for GPU %d: %s", i, e)

            pynvml.nvmlShutdown()
        except Exception as e:
            logger.error("Error observing HW slowdown status: %s", e)

    def _observe_ecc_errors_corrected(self, options):
        """Observable callback for GPU ECC corrected memory errors."""
        if not NVML_AVAILABLE or not self.gpu_available:
            return

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                device_name = self._get_device_name(handle, i)

                try:
                    # Get volatile (since last reboot) corrected errors
                    corrected_errors = pynvml.nvmlDeviceGetTotalEccErrors(
                        handle,
                        pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                        pynvml.NVML_VOLATILE_ECC,
                    )
                    yield Observation(
                        value=corrected_errors,
                        attributes={"gpu_id": str(i), "gpu_name": device_name},
                    )
                except pynvml.NVMLError as e:
                    # ECC not supported on all GPUs - this is expected
                    logger.debug("ECC not supported or error for GPU %d: %s", i, e)
                except Exception as e:
                    logger.debug("Failed to get ECC corrected errors for GPU %d: %s", i, e)

            pynvml.nvmlShutdown()
        except Exception as e:
            logger.error("Error observing ECC corrected errors: %s", e)

    def _observe_ecc_errors_uncorrected(self, options):
        """Observable callback for GPU ECC uncorrected memory errors."""
        if not NVML_AVAILABLE or not self.gpu_available:
            return

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                device_name = self._get_device_name(handle, i)

                try:
                    # Get volatile (since last reboot) uncorrected errors
                    uncorrected_errors = pynvml.nvmlDeviceGetTotalEccErrors(
                        handle,
                        pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                        pynvml.NVML_VOLATILE_ECC,
                    )
                    yield Observation(
                        value=uncorrected_errors,
                        attributes={"gpu_id": str(i), "gpu_name": device_name},
                    )
                except pynvml.NVMLError as e:
                    # ECC not supported on all GPUs - this is expected
                    logger.debug("ECC not supported or error for GPU %d: %s", i, e)
                except Exception as e:
                    logger.debug("Failed to get ECC uncorrected errors for GPU %d: %s", i, e)

            pynvml.nvmlShutdown()
        except Exception as e:
            logger.error("Error observing ECC uncorrected errors: %s", e)

    # ==================== Aggregate GPU Metrics Callbacks ====================

    def _observe_aggregate_mean_utilization(self, options):
        """Observable callback for mean GPU utilization across all GPUs."""
        if not NVML_AVAILABLE or not self.gpu_available:
            return

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            if device_count == 0:
                pynvml.nvmlShutdown()
                return

            total_utilization = 0.0
            valid_count = 0

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    total_utilization += utilization.gpu
                    valid_count += 1
                except Exception as e:
                    logger.debug("Failed to get utilization for GPU %d in aggregate: %s", i, e)

            if valid_count > 0:
                mean_utilization = total_utilization / valid_count
                yield Observation(
                    value=mean_utilization,
                    attributes={"gpu_count": str(valid_count)},
                )

            pynvml.nvmlShutdown()
        except Exception as e:
            logger.error("Error observing aggregate mean utilization: %s", e)

    def _observe_aggregate_total_memory(self, options):
        """Observable callback for total GPU memory used across all GPUs."""
        if not NVML_AVAILABLE or not self.gpu_available:
            return

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            if device_count == 0:
                pynvml.nvmlShutdown()
                return

            total_memory_used_bytes = 0
            valid_count = 0

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                try:
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_memory_used_bytes += memory_info.used
                    valid_count += 1
                except Exception as e:
                    logger.debug("Failed to get memory for GPU %d in aggregate: %s", i, e)

            if valid_count > 0:
                # Convert bytes to GiB
                total_memory_gib = total_memory_used_bytes / (1024**3)
                yield Observation(
                    value=total_memory_gib,
                    attributes={"gpu_count": str(valid_count)},
                )

            pynvml.nvmlShutdown()
        except Exception as e:
            logger.error("Error observing aggregate total memory: %s", e)

    def _observe_aggregate_total_power(self, options):
        """Observable callback for total power consumption across all GPUs."""
        if not NVML_AVAILABLE or not self.gpu_available:
            return

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            if device_count == 0:
                pynvml.nvmlShutdown()
                return

            total_power_w = 0.0
            valid_count = 0

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                try:
                    # Power usage is returned in milliwatts
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    total_power_w += power_mw / 1000.0
                    valid_count += 1
                except Exception as e:
                    logger.debug("Failed to get power for GPU %d in aggregate: %s", i, e)

            if valid_count > 0:
                yield Observation(
                    value=total_power_w,
                    attributes={"gpu_count": str(valid_count)},
                )

            pynvml.nvmlShutdown()
        except Exception as e:
            logger.error("Error observing aggregate total power: %s", e)

    def _observe_aggregate_max_temperature(self, options):
        """Observable callback for maximum temperature across all GPUs."""
        if not NVML_AVAILABLE or not self.gpu_available:
            return

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            if device_count == 0:
                pynvml.nvmlShutdown()
                return

            max_temp = None
            valid_count = 0

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    if max_temp is None or temp > max_temp:
                        max_temp = temp
                    valid_count += 1
                except Exception as e:
                    logger.debug("Failed to get temperature for GPU %d in aggregate: %s", i, e)

            if valid_count > 0 and max_temp is not None:
                yield Observation(
                    value=max_temp,
                    attributes={"gpu_count": str(valid_count)},
                )

            pynvml.nvmlShutdown()
        except Exception as e:
            logger.error("Error observing aggregate max temperature: %s", e)

    def start(self):
        """Starts the GPU metrics collection.

        ObservableGauges are automatically collected by the MeterProvider,
        so we only need to start the CO2 collection thread.
        """
        if not NVML_AVAILABLE and not (AMDSMI_AVAILABLE and self.amd_collector):
            logger.warning("Cannot start GPU metrics collection - no GPU libraries available")
            return

        if not self.gpu_available:
            return

        # Start codecarbon emissions tracker if available and configured
        if self._use_codecarbon and self._emissions_tracker:
            try:
                self._emissions_tracker.start()
                # Start a continuous task for periodic emissions collection
                self._emissions_tracker.start_task("gpu_monitoring")
                self._last_emissions_kg = 0.0
                logger.info("Codecarbon emissions tracker started with continuous task monitoring")
            except Exception as e:
                logger.warning("Failed to start codecarbon tracker: %s", e)
                self._use_codecarbon = False

        logger.info("Starting GPU metrics collection (CO2 tracking)")
        # Only start CO2 collection thread - ObservableGauges are auto-collected
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()

    def _collect_loop(self):
        while not self._stop_event.wait(self.interval):
            current_time = time.time()

            # Collect CO2 emissions from codecarbon if available
            if self.config.enable_co2_tracking:
                self._collect_codecarbon_emissions()

            for i in range(self.device_count):
                try:
                    handle = self.nvml.nvmlDeviceGetHandleByIndex(i)
                    power_w = self.nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Watts
                    delta_time_hours = (current_time - self.last_timestamp[i]) / 3600.0
                    delta_energy_wh = (power_w / 1000.0) * (
                        delta_time_hours * 3600.0
                    )  # Wh (power in kW * hours = kWh, but track in Wh for precision)
                    self.cumulative_energy_wh[i] += delta_energy_wh

                    # Calculate and record CO2 emissions using manual calculation
                    # (only if codecarbon is not available/enabled)
                    if self.config.enable_co2_tracking and not self._use_codecarbon:
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
                    logger.error("Error collecting GPU %d metrics: %s", i, e)

    def _collect_codecarbon_emissions(self):
        """Collect CO2 emissions from codecarbon and report to OpenTelemetry."""
        if not self._use_codecarbon or not self._emissions_tracker:
            return

        try:
            # Stop the current task and get emissions data
            # This returns an EmissionsData object with detailed metrics
            emissions_data = self._emissions_tracker.stop_task("gpu_monitoring")

            # Immediately restart the task for continuous monitoring
            self._emissions_tracker.start_task("gpu_monitoring")

            if emissions_data:
                # Extract emissions in kg CO2e from the task data
                task_emissions_kg = emissions_data.emissions  # kg CO2e

                # Convert kg to grams and record
                task_emissions_g = task_emissions_kg * 1000.0

                # Record total emissions
                self.co2_counter.add(
                    task_emissions_g,
                    {
                        "source": "codecarbon",
                        "country": emissions_data.country_iso_code or "unknown",
                        "region": emissions_data.region or "unknown",
                    },
                )

                # Record emissions rate (gCO2e/s)
                if hasattr(emissions_data, "emissions_rate") and emissions_data.emissions_rate:
                    # emissions_rate is in kg/s, convert to g/s
                    rate_g_per_s = emissions_data.emissions_rate * 1000.0
                    self.emissions_rate_gauge.record(
                        rate_g_per_s,
                        {
                            "source": "codecarbon",
                            "country": emissions_data.country_iso_code or "unknown",
                        },
                    )

                # Common attributes for all metrics including hardware/system metadata
                base_attrs = {
                    "source": "codecarbon",
                    "country": emissions_data.country_iso_code or "unknown",
                    "region": emissions_data.region or "unknown",
                }

                # Add hardware and system metadata as attributes
                if hasattr(emissions_data, "os") and emissions_data.os:
                    base_attrs["os"] = emissions_data.os
                if hasattr(emissions_data, "python_version") and emissions_data.python_version:
                    base_attrs["python_version"] = emissions_data.python_version
                if hasattr(emissions_data, "cpu_count") and emissions_data.cpu_count:
                    base_attrs["cpu_count"] = str(emissions_data.cpu_count)
                if hasattr(emissions_data, "cpu_model") and emissions_data.cpu_model:
                    # Truncate CPU model to avoid attribute size limits
                    base_attrs["cpu_model"] = str(emissions_data.cpu_model)[:100]
                if hasattr(emissions_data, "gpu_count") and emissions_data.gpu_count:
                    base_attrs["gpu_count"] = str(emissions_data.gpu_count)
                if hasattr(emissions_data, "gpu_model") and emissions_data.gpu_model:
                    # Truncate GPU model to avoid attribute size limits
                    base_attrs["gpu_model"] = str(emissions_data.gpu_model)[:100]
                if hasattr(emissions_data, "on_cloud") and emissions_data.on_cloud:
                    base_attrs["on_cloud"] = emissions_data.on_cloud
                if hasattr(emissions_data, "cloud_provider") and emissions_data.cloud_provider:
                    base_attrs["cloud_provider"] = emissions_data.cloud_provider
                if hasattr(emissions_data, "cloud_region") and emissions_data.cloud_region:
                    base_attrs["cloud_region"] = emissions_data.cloud_region

                # Record duration
                if hasattr(emissions_data, "duration") and emissions_data.duration:
                    self.task_duration_histogram.record(emissions_data.duration, base_attrs)

                # Record power consumption (W)
                if hasattr(emissions_data, "cpu_power") and emissions_data.cpu_power:
                    self.power_gauge.record(
                        emissions_data.cpu_power,
                        {**base_attrs, "component": "cpu"},
                    )
                if hasattr(emissions_data, "gpu_power") and emissions_data.gpu_power:
                    self.power_gauge.record(
                        emissions_data.gpu_power,
                        {**base_attrs, "component": "gpu"},
                    )
                if hasattr(emissions_data, "ram_power") and emissions_data.ram_power:
                    self.power_gauge.record(
                        emissions_data.ram_power,
                        {**base_attrs, "component": "ram"},
                    )

                # Record energy consumption breakdown (kWh)
                total_energy = 0.0
                if hasattr(emissions_data, "cpu_energy") and emissions_data.cpu_energy:
                    self.energy_counter.add(
                        emissions_data.cpu_energy,  # Already in kWh
                        {**base_attrs, "component": "cpu"},
                    )
                    total_energy += emissions_data.cpu_energy
                if hasattr(emissions_data, "gpu_energy") and emissions_data.gpu_energy:
                    self.energy_counter.add(
                        emissions_data.gpu_energy,  # Already in kWh
                        {**base_attrs, "component": "gpu"},
                    )
                    total_energy += emissions_data.gpu_energy
                if hasattr(emissions_data, "ram_energy") and emissions_data.ram_energy:
                    self.energy_counter.add(
                        emissions_data.ram_energy,  # Already in kWh
                        {**base_attrs, "component": "ram"},
                    )
                    total_energy += emissions_data.ram_energy

                # Record total energy consumed (can also use energy_consumed from emissions_data)
                if hasattr(emissions_data, "energy_consumed") and emissions_data.energy_consumed:
                    self.total_energy_counter.add(emissions_data.energy_consumed, base_attrs)
                elif total_energy > 0:
                    self.total_energy_counter.add(total_energy, base_attrs)

                # Update cumulative total
                self._last_emissions_kg += task_emissions_kg

                logger.debug(
                    "Recorded %.4f gCO2e emissions from codecarbon task "
                    "(duration: %.2fs, rate: %.4f gCO2e/s, "
                    "power: CPU=%.2fW GPU=%.2fW RAM=%.2fW, "
                    "energy: CPU=%.6f GPU=%.6f RAM=%.6f total=%.6f kWh, "
                    "cumulative: %.4f kg)",
                    task_emissions_g,
                    emissions_data.duration if hasattr(emissions_data, "duration") else 0,
                    rate_g_per_s if hasattr(emissions_data, "emissions_rate") else 0,
                    emissions_data.cpu_power if hasattr(emissions_data, "cpu_power") else 0,
                    emissions_data.gpu_power if hasattr(emissions_data, "gpu_power") else 0,
                    emissions_data.ram_power if hasattr(emissions_data, "ram_power") else 0,
                    emissions_data.cpu_energy if hasattr(emissions_data, "cpu_energy") else 0,
                    emissions_data.gpu_energy if hasattr(emissions_data, "gpu_energy") else 0,
                    emissions_data.ram_energy if hasattr(emissions_data, "ram_energy") else 0,
                    total_energy,
                    self._last_emissions_kg,
                )

        except Exception as e:
            logger.debug("Error collecting codecarbon emissions: %s", e)

    def stop(self):
        """Stops the GPU metrics collection thread."""
        # Stop CO2 collection thread
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
            logger.info("GPU CO2 metrics collection thread stopped.")

        # Stop codecarbon emissions tracker and get final emissions
        if self._use_codecarbon and self._emissions_tracker:
            try:
                # Stop the ongoing task first to get any remaining emissions
                try:
                    final_task_emissions = self._emissions_tracker.stop_task("gpu_monitoring")
                    if final_task_emissions and final_task_emissions.emissions > 0:
                        task_emissions_g = final_task_emissions.emissions * 1000.0
                        self.co2_counter.add(
                            task_emissions_g,
                            {
                                "source": "codecarbon",
                                "country": final_task_emissions.country_iso_code or "unknown",
                                "region": final_task_emissions.region or "unknown",
                            },
                        )
                        self._last_emissions_kg += final_task_emissions.emissions
                except Exception as task_error:
                    logger.debug("No active task to stop: %s", task_error)

                # Then stop the tracker
                final_emissions_kg = self._emissions_tracker.stop()
                if final_emissions_kg is not None:
                    logger.info(
                        "Codecarbon emissions tracker stopped. Total emissions: %.4f kg CO2e",
                        final_emissions_kg,
                    )
            except Exception as e:
                logger.debug("Error stopping codecarbon tracker: %s", e)

        # ObservableGauges will automatically stop when MeterProvider is shutdown
        if self.gpu_available and NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception as e:
                logger.debug("Error shutting down NVML: %s", e)
