import os
from dataclasses import dataclass, field
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class OTelConfig:
    """Configuration for OpenTelemetry instrumentation.

    Loads settings from environment variables with sensible defaults.
    """
    service_name: str = field(default_factory=lambda: os.getenv("OTEL_SERVICE_NAME", "genai-app"))
    endpoint: str = field(default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"))
    enable_gpu_metrics: bool = field(
        default_factory=lambda: os.getenv("GENAI_ENABLE_GPU_METRICS", "true").lower() == "true")
    enable_cost_tracking: bool = field(
        default_factory=lambda: os.getenv("GENAI_ENABLE_COST_TRACKING", "true").lower() == "true")
    enable_mcp_instrumentation: bool = field(
        default_factory=lambda: os.getenv("GENAI_ENABLE_MCP_INSTRUMENTATION", "true").lower() == "true")
    # Add fail_on_error configuration
    fail_on_error: bool = field(
        default_factory=lambda: os.getenv("GENAI_FAIL_ON_ERROR", "false").lower() == "true")
    headers: Optional[Dict[str, str]] = None

    def __post_init__(self):
        """Post-initialization hook to parse headers from environment variable."""
        if self.headers is None:
            headers_str = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
            if headers_str:
                try:
                    self.headers = dict(h.split("=") for h in headers_str.split(","))
                except ValueError:
                    logger.error(f"Failed to parse OTEL_EXPORTER_OTLP_HEADERS: '{headers_str}'. Expected format 'key1=value1,key2=value2'.")
