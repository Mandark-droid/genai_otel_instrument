import os
import logging
from typing import Optional
from .auto_instrument import setup_auto_instrumentation
from .config import OTelConfig
from .exceptions import InstrumentationError
from .logging_config import setup_logging

# Setup logging
logger = setup_logging(
    level=os.getenv("GENAI_LOG_LEVEL", "INFO"),
    log_file=os.getenv("GENAI_LOG_FILE")
)


def instrument(
        service_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        enable_gpu_metrics: bool = True,
        enable_cost_tracking: bool = True,
        enable_auto_instrument: bool = True,
        enable_mcp_instrumentation: bool = True,
        fail_on_error: bool = False
):
    """
    Single function to instrument your GenAI application.

    Args:
        service_name: Service name for telemetry
        endpoint: OTLP endpoint URL
        enable_gpu_metrics: Enable GPU metrics collection
        enable_cost_tracking: Enable cost calculation
        enable_auto_instrument: Enable automatic instrumentation
        enable_mcp_instrumentation: Enable MCP tool instrumentation
        fail_on_error: If True, raise exceptions on instrumentation errors.
                      If False, log errors and continue (production default)

    Returns:
        OTelConfig object

    Raises:
        InstrumentationError: If fail_on_error=True and setup fails
    """
    try:
        config = OTelConfig(
            service_name=service_name,
            endpoint=endpoint,
            enable_gpu_metrics=enable_gpu_metrics,
            enable_cost_tracking=enable_cost_tracking,
            enable_mcp_instrumentation=enable_mcp_instrumentation
        )

        logger.info(
            f"Initializing GenAI instrumentation for service: {config.service_name}"
        )
        logger.debug(f"Configuration: endpoint={config.endpoint}, "
                     f"gpu_metrics={config.enable_gpu_metrics}, "
                     f"cost_tracking={config.enable_cost_tracking}, "
                     f"mcp={config.enable_mcp_instrumentation}")

        if enable_auto_instrument:
            setup_auto_instrumentation(config)

        logger.info(f"✓ GenAI instrumentation initialized successfully for {config.service_name}")
        return config

    except Exception as e:
        logger.error(f"Failed to initialize GenAI instrumentation: {e}", exc_info=True)
        if fail_on_error:
            raise InstrumentationError(f"Instrumentation setup failed: {e}") from e
        logger.warning("Continuing without instrumentation due to initialization error")
        return None
