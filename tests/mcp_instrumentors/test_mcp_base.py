"""Tests for BaseMCPInstrumentor."""

import unittest
from unittest.mock import MagicMock, patch

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View

from genai_otel.mcp_instrumentors.base import BaseMCPInstrumentor
from genai_otel.metrics import _MCP_CLIENT_OPERATION_DURATION_BUCKETS, _MCP_PAYLOAD_SIZE_BUCKETS


class TestBaseMCPInstrumentor(unittest.TestCase):
    """Tests for BaseMCPInstrumentor class."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset class-level state
        BaseMCPInstrumentor._shared_request_counter = None
        BaseMCPInstrumentor._shared_duration_histogram = None
        BaseMCPInstrumentor._shared_request_size_histogram = None
        BaseMCPInstrumentor._shared_response_size_histogram = None
        BaseMCPInstrumentor._metrics_initialized = False

        # Set up MeterProvider with InMemoryMetricReader
        self.reader = InMemoryMetricReader()

        views = [
            View(
                instrument_name="mcp.client.operation.duration",
                aggregation=ExplicitBucketHistogramAggregation(
                    boundaries=_MCP_CLIENT_OPERATION_DURATION_BUCKETS
                ),
            ),
            View(
                instrument_name="mcp.request.size",
                aggregation=ExplicitBucketHistogramAggregation(
                    boundaries=_MCP_PAYLOAD_SIZE_BUCKETS
                ),
            ),
            View(
                instrument_name="mcp.response.size",
                aggregation=ExplicitBucketHistogramAggregation(
                    boundaries=_MCP_PAYLOAD_SIZE_BUCKETS
                ),
            ),
        ]

        self.meter_provider = MeterProvider(metric_readers=[self.reader], views=views)
        metrics.set_meter_provider(self.meter_provider)

    def tearDown(self):
        """Clean up after tests."""
        metrics.set_meter_provider(None)
        BaseMCPInstrumentor._shared_request_counter = None
        BaseMCPInstrumentor._shared_duration_histogram = None
        BaseMCPInstrumentor._shared_request_size_histogram = None
        BaseMCPInstrumentor._shared_response_size_histogram = None
        BaseMCPInstrumentor._metrics_initialized = False

    def test_init_creates_metrics(self):
        """Test that __init__ creates shared metrics."""
        instrumentor = BaseMCPInstrumentor()

        # Verify metrics were created
        self.assertIsNotNone(instrumentor.mcp_request_counter)
        self.assertIsNotNone(instrumentor.mcp_duration_histogram)
        self.assertIsNotNone(instrumentor.mcp_request_size_histogram)
        self.assertIsNotNone(instrumentor.mcp_response_size_histogram)

        # Verify class-level metrics are set
        self.assertIsNotNone(BaseMCPInstrumentor._shared_request_counter)
        self.assertIsNotNone(BaseMCPInstrumentor._shared_duration_histogram)
        self.assertIsNotNone(BaseMCPInstrumentor._shared_request_size_histogram)
        self.assertIsNotNone(BaseMCPInstrumentor._shared_response_size_histogram)
        self.assertTrue(BaseMCPInstrumentor._metrics_initialized)

    def test_multiple_instances_share_metrics(self):
        """Test that multiple instances share the same metrics."""
        instrumentor1 = BaseMCPInstrumentor()
        instrumentor2 = BaseMCPInstrumentor()

        # Verify they share the same metric instances
        self.assertIs(instrumentor1.mcp_request_counter, instrumentor2.mcp_request_counter)
        self.assertIs(instrumentor1.mcp_duration_histogram, instrumentor2.mcp_duration_histogram)
        self.assertIs(
            instrumentor1.mcp_request_size_histogram,
            instrumentor2.mcp_request_size_histogram,
        )
        self.assertIs(
            instrumentor1.mcp_response_size_histogram,
            instrumentor2.mcp_response_size_histogram,
        )

    def test_metrics_can_record_values(self):
        """Test that metrics can record values without raising exceptions."""
        instrumentor = BaseMCPInstrumentor()

        # These should not raise exceptions
        try:
            instrumentor.mcp_request_counter.add(1, {"db.system": "test"})
            instrumentor.mcp_duration_histogram.record(0.5, {"db.system": "test"})
            instrumentor.mcp_request_size_histogram.record(1024, {"db.system": "test"})
            instrumentor.mcp_response_size_histogram.record(2048, {"db.system": "test"})
        except Exception as e:
            self.fail(f"Recording metrics raised an exception: {e}")

    @patch("genai_otel.mcp_instrumentors.base.logger")
    @patch("genai_otel.mcp_instrumentors.base.metrics.get_meter")
    def test_metrics_creation_failure_handled(self, mock_get_meter, mock_logger):
        """Test that metric creation failures are handled gracefully."""
        # Reset state
        BaseMCPInstrumentor._shared_request_counter = None
        BaseMCPInstrumentor._shared_duration_histogram = None
        BaseMCPInstrumentor._shared_request_size_histogram = None
        BaseMCPInstrumentor._shared_response_size_histogram = None
        BaseMCPInstrumentor._metrics_initialized = False

        # Make get_meter raise an exception
        mock_get_meter.side_effect = Exception("Meter creation failed")

        # Create instrumentor
        instrumentor = BaseMCPInstrumentor()

        # Verify metrics are None
        self.assertIsNone(instrumentor.mcp_request_counter)
        self.assertIsNone(instrumentor.mcp_duration_histogram)
        self.assertIsNone(instrumentor.mcp_request_size_histogram)
        self.assertIsNone(instrumentor.mcp_response_size_histogram)

        # Verify warning was logged
        mock_logger.warning.assert_called_once()
        self.assertIn("Failed to create MCP shared metrics", str(mock_logger.warning.call_args))


if __name__ == "__main__":
    unittest.main()
