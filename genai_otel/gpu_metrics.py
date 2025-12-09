"""Module for collecting GPU metrics using nvidia-ml-py and reporting them via OpenTelemetry.

This module provides the `GPUMetricsCollector` class, which periodically collects
GPU utilization, memory usage, and temperature, and exports these as OpenTelemetry
metrics. It relies on the `nvidia-ml-py` library for interacting with NVIDIA GPUs.

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
    logger.debug("nvidia-ml-py not available, GPU metrics will be disabled")

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
    """Collects and reports GPU metrics using nvidia-ml-py and codecarbon for CO2 tracking."""

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
        self.config = config
        self.interval = interval  # seconds
        self.gpu_available = False

        # Codecarbon emissions tracker
        self._emissions_tracker: Optional["EmissionsTracker"] = None
        self._last_emissions_kg: float = 0.0
        self._use_codecarbon: bool = False

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
        self.emissions_rate_gauge = meter.create_histogram(
            "gen_ai.co2.emissions_rate",
            description="CO2 emissions rate (rate of emissions per second)",
            unit="gCO2e/s",
        )

        # Initialize codecarbon if available and CO2 tracking is enabled
        self._init_codecarbon()

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
            # Build codecarbon configuration from OTelConfig
            tracker_kwargs = {
                "project_name": self.config.service_name,
                "measure_power_secs": self.config.gpu_collection_interval,
                "save_to_file": False,  # We report via OpenTelemetry, not CSV
                "save_to_api": False,  # Don't send to codecarbon API
                "logging_logger": logger,  # Use our logger
                "log_level": "warning",  # Reduce codecarbon's logging noise
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

                # Record energy consumption breakdown (kWh)
                if hasattr(emissions_data, "cpu_energy") and emissions_data.cpu_energy:
                    self.energy_counter.add(
                        emissions_data.cpu_energy,  # Already in kWh
                        {"component": "cpu", "source": "codecarbon"},
                    )
                if hasattr(emissions_data, "gpu_energy") and emissions_data.gpu_energy:
                    self.energy_counter.add(
                        emissions_data.gpu_energy,  # Already in kWh
                        {"component": "gpu", "source": "codecarbon"},
                    )
                if hasattr(emissions_data, "ram_energy") and emissions_data.ram_energy:
                    self.energy_counter.add(
                        emissions_data.ram_energy,  # Already in kWh
                        {"component": "ram", "source": "codecarbon"},
                    )

                # Update cumulative total
                self._last_emissions_kg += task_emissions_kg

                logger.debug(
                    "Recorded %.4f gCO2e emissions from codecarbon task "
                    "(duration: %.2fs, rate: %.4f gCO2e/s, "
                    "energy: CPU=%.6f GPU=%.6f RAM=%.6f kWh, total: %.4f kg)",
                    task_emissions_g,
                    emissions_data.duration if hasattr(emissions_data, "duration") else 0,
                    rate_g_per_s if hasattr(emissions_data, "emissions_rate") else 0,
                    emissions_data.cpu_energy if hasattr(emissions_data, "cpu_energy") else 0,
                    emissions_data.gpu_energy if hasattr(emissions_data, "gpu_energy") else 0,
                    emissions_data.ram_energy if hasattr(emissions_data, "ram_energy") else 0,
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
