import unittest
from unittest.mock import MagicMock, patch

from genai_otel.mcp_instrumentors.kafka_instrumentor import KafkaInstrumentor


class TestKafkaInstrumentor(unittest.TestCase):
    """Tests for KafkaInstrumentor"""

    def test_init(self):
        """Test that __init__ stores config correctly."""
        config = MagicMock()
        instrumentor = KafkaInstrumentor(config)

        self.assertEqual(instrumentor.config, config)

    @patch("genai_otel.mcp_instrumentors.kafka_instrumentor.OTelKafkaInstrumentor")
    @patch("genai_otel.mcp_instrumentors.kafka_instrumentor.logger")
    def test_instrument_success(self, mock_logger, mock_kafka_instrumentor):
        """Test successful Kafka instrumentation."""
        config = MagicMock()
        instrumentor = KafkaInstrumentor(config)

        # Mock the instrumentor to succeed
        mock_instance = MagicMock()
        mock_kafka_instrumentor.return_value = mock_instance

        # Act
        instrumentor.instrument()

        # Assert
        mock_kafka_instrumentor.assert_called_once()
        mock_instance.instrument.assert_called_once()
        mock_logger.info.assert_called_once_with("Kafka instrumentation enabled")

    @patch("genai_otel.mcp_instrumentors.kafka_instrumentor.OTelKafkaInstrumentor")
    @patch("genai_otel.mcp_instrumentors.kafka_instrumentor.logger")
    def test_instrument_import_error(self, mock_logger, mock_kafka_instrumentor):
        """Test Kafka instrumentation when kafka-python is not installed."""
        config = MagicMock()
        instrumentor = KafkaInstrumentor(config)

        # Mock the instrumentor to raise ImportError
        mock_kafka_instrumentor.side_effect = ImportError("kafka-python not found")

        # Act
        instrumentor.instrument()

        # Assert
        mock_logger.debug.assert_called_once_with(
            "Kafka-python not installed, skipping instrumentation."
        )

    @patch("genai_otel.mcp_instrumentors.kafka_instrumentor.OTelKafkaInstrumentor")
    @patch("genai_otel.mcp_instrumentors.kafka_instrumentor.logger")
    def test_instrument_general_exception(self, mock_logger, mock_kafka_instrumentor):
        """Test Kafka instrumentation when a general exception occurs."""
        config = MagicMock()
        instrumentor = KafkaInstrumentor(config)

        # Mock the instrumentor to raise a general exception
        mock_kafka_instrumentor.side_effect = RuntimeError("Something went wrong")

        # Act
        instrumentor.instrument()

        # Assert
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0][0]
        self.assertIn("Kafka instrumentation failed", call_args)
        self.assertIn("Something went wrong", call_args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
