"""Base class for MCP instrumentors with shared metrics.

This module provides the `BaseMCPInstrumentor` class which creates and manages
MCP-specific metrics (requests, duration, payload sizes) that are shared across
all MCP instrumentors (databases, APIs, vector DBs, etc.).
"""

import logging
from typing import Optional

from opentelemetry import metrics
from opentelemetry.metrics import Counter, Histogram

logger = logging.getLogger(__name__)

# Import semantic conventions
try:
    from openlit.semcov import SemanticConvention as SC
except ImportError:
    # Fallback if openlit not available
    class SC:
        MCP_REQUESTS = "mcp.requests"
        MCP_CLIENT_OPERATION_DURATION_METRIC = "mcp.client.operation.duration"
        MCP_REQUEST_SIZE = "mcp.request.size"
        MCP_RESPONSE_SIZE_METRIC = "mcp.response.size"


class BaseMCPInstrumentor:
    """Base class for MCP instrumentors with shared metrics.

    This class provides MCP-specific metrics that can be used by all MCP instrumentors
    to track requests, operation duration, and payload sizes.

    Metrics:
        - mcp.requests: Counter for number of MCP requests
        - mcp.client.operation.duration: Histogram for operation duration in seconds
        - mcp.request.size: Histogram for request payload size in bytes
        - mcp.response.size: Histogram for response payload size in bytes
    """

    # Class-level shared metrics (created once, shared by all instances)
    _shared_request_counter: Optional[Counter] = None
    _shared_duration_histogram: Optional[Histogram] = None
    _shared_request_size_histogram: Optional[Histogram] = None
    _shared_response_size_histogram: Optional[Histogram] = None
    _metrics_initialized = False

    def __init__(self):
        """Initialize BaseMCPInstrumentor and create shared metrics if needed."""
        if not BaseMCPInstrumentor._metrics_initialized:
            self._create_shared_metrics()

        # Instance references to shared metrics
        self.mcp_request_counter = BaseMCPInstrumentor._shared_request_counter
        self.mcp_duration_histogram = BaseMCPInstrumentor._shared_duration_histogram
        self.mcp_request_size_histogram = BaseMCPInstrumentor._shared_request_size_histogram
        self.mcp_response_size_histogram = BaseMCPInstrumentor._shared_response_size_histogram

    @classmethod
    def _create_shared_metrics(cls):
        """Create shared MCP metrics once at class level."""
        if cls._metrics_initialized:
            return

        try:
            meter = metrics.get_meter(__name__)

            # MCP request counter
            cls._shared_request_counter = meter.create_counter(
                SC.MCP_REQUESTS,
                description="Number of MCP requests",
                unit="1",
            )

            # MCP operation duration histogram
            cls._shared_duration_histogram = meter.create_histogram(
                SC.MCP_CLIENT_OPERATION_DURATION_METRIC,
                description="MCP operation duration",
                unit="s",
            )

            # MCP request size histogram
            cls._shared_request_size_histogram = meter.create_histogram(
                SC.MCP_REQUEST_SIZE,
                description="MCP request payload size",
                unit="By",
            )

            # MCP response size histogram
            cls._shared_response_size_histogram = meter.create_histogram(
                SC.MCP_RESPONSE_SIZE_METRIC,
                description="MCP response payload size",
                unit="By",
            )

            cls._metrics_initialized = True
            logger.debug("MCP shared metrics created successfully")

        except Exception as e:
            logger.warning(f"Failed to create MCP shared metrics: {e}")
            # Set to None if creation fails
            cls._shared_request_counter = None
            cls._shared_duration_histogram = None
            cls._shared_request_size_histogram = None
            cls._shared_response_size_histogram = None
