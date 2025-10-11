import os
from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class OTelConfig:
    """Configuration for OpenTelemetry instrumentation"""
    service_name: str = field(default_factory=lambda: os.getenv("OTEL_SERVICE_NAME", "genai-app"))
    endpoint: str = field(default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"))
    enable_gpu_metrics: bool = field(
        default_factory=lambda: os.getenv("GENAI_ENABLE_GPU_METRICS", "true").lower() == "true")
    enable_cost_tracking: bool = field(
        default_factory=lambda: os.getenv("GENAI_ENABLE_COST_TRACKING", "true").lower() == "true")
    enable_mcp_instrumentation: bool = field(
        default_factory=lambda: os.getenv("GENAI_ENABLE_MCP_INSTRUMENTATION", "true").lower() == "true")
    headers: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if self.headers is None:
            headers_str = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
            if headers_str:
                self.headers = dict(h.split("=") for h in headers_str.split(","))
