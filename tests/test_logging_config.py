import logging
import sys
from unittest.mock import MagicMock, call, patch

from genai_otel.logging_config import setup_logging


def test_setup_logging_default_level():
    """Test setup_logging with default INFO level and no file handler."""
    with patch("logging.basicConfig") as mock_basic_config, patch(
        "logging.getLogger"
    ) as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        logger = setup_logging()

        # Verify basicConfig was called with correct parameters
        mock_basic_config.assert_called_once()
        call_args = mock_basic_config.call_args

        # Check level
        assert call_args.kwargs["level"] == logging.INFO

        # Check format
        assert call_args.kwargs["format"] == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # Check handlers - should have one StreamHandler with stdout
        handlers = call_args.kwargs["handlers"]
        assert len(handlers) == 1
        assert isinstance(handlers[0], logging.StreamHandler)
        assert handlers[0].stream == sys.stdout

        # Verify logger setup
        mock_get_logger.assert_called_once_with("genai_otel")
        mock_logger.setLevel.assert_called_once_with(logging.INFO)
        assert logger == mock_logger


def test_setup_logging_custom_level():
    """Test setup_logging with a custom logging level (e.g., DEBUG)."""
    with patch("logging.basicConfig") as mock_basic_config, patch(
        "logging.getLogger"
    ) as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        setup_logging(level="DEBUG")

        mock_basic_config.assert_called_once()
        call_args = mock_basic_config.call_args
        assert call_args.kwargs["level"] == logging.DEBUG

        mock_get_logger.assert_called_once_with("genai_otel")
        mock_logger.setLevel.assert_called_once_with(logging.DEBUG)


def test_setup_logging_with_file_handler():
    """Test setup_logging with a log file specified."""
    with patch("logging.basicConfig") as mock_basic_config, patch(
        "logging.FileHandler"
    ) as mock_file_handler, patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_file_instance = MagicMock()
        mock_file_handler.return_value = mock_file_instance
        mock_get_logger.return_value = mock_logger

        test_log_file = "test.log"
        setup_logging(log_file=test_log_file)

        mock_basic_config.assert_called_once()
        call_args = mock_basic_config.call_args

        # Check that FileHandler was created with correct filename
        mock_file_handler.assert_called_once_with(test_log_file)

        # Check that handlers include both StreamHandler and FileHandler
        handlers = call_args.kwargs["handlers"]
        assert len(handlers) == 2
        # First handler should be StreamHandler with stdout
        assert handlers[0].stream == sys.stdout
        # Second handler should be our mocked FileHandler
        assert handlers[1] == mock_file_instance

        # Verify logger setup
        mock_get_logger.assert_called_once_with("genai_otel")
        mock_logger.setLevel.assert_called_once_with(logging.INFO)


def test_setup_logging_returns_logger():
    """Test that setup_logging returns the configured logger instance."""
    with patch("logging.basicConfig"), patch("logging.getLogger") as mock_get_logger:
        expected_logger = MagicMock()
        mock_get_logger.return_value = expected_logger

        returned_logger = setup_logging()
        assert returned_logger == expected_logger


def test_setup_logging_format():
    """Test that logging format is properly set."""
    with patch("logging.basicConfig") as mock_basic_config, patch("logging.getLogger"):
        setup_logging()

        mock_basic_config.assert_called_once()
        call_args = mock_basic_config.call_args

        # Check that format is specified (no datefmt in your implementation)
        assert "format" in call_args.kwargs
        assert "datefmt" not in call_args.kwargs  # Your function doesn't use datefmt

        # Verify format contains expected elements
        format_str = call_args.kwargs["format"]
        assert "%(asctime)s" in format_str
        assert "%(levelname)s" in format_str
        assert "%(name)s" in format_str
        assert "%(message)s" in format_str


def test_setup_logging_invalid_level_falls_back_to_info():
    """Test that invalid level falls back to INFO."""
    with patch("logging.basicConfig") as mock_basic_config, patch(
        "logging.getLogger"
    ) as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        setup_logging(level="INVALID_LEVEL")

        mock_basic_config.assert_called_once()
        call_args = mock_basic_config.call_args
        assert call_args.kwargs["level"] == logging.INFO

        mock_get_logger.assert_called_once_with("genai_otel")
        mock_logger.setLevel.assert_called_once_with(logging.INFO)


def test_setup_logging_stream_handler_uses_stdout():
    """Test that StreamHandler uses sys.stdout."""
    with patch("logging.basicConfig") as mock_basic_config, patch("logging.getLogger"):
        setup_logging()

        mock_basic_config.assert_called_once()
        call_args = mock_basic_config.call_args

        handlers = call_args.kwargs["handlers"]
        stream_handler = handlers[0]
        assert stream_handler.stream == sys.stdout


def test_setup_logging_multiple_calls():
    """Test that setup_logging can be called multiple times safely."""
    with patch("logging.basicConfig") as mock_basic_config, patch(
        "logging.getLogger"
    ) as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Call setup_logging multiple times
        setup_logging()
        setup_logging(level="DEBUG")
        setup_logging(log_file="test.log")

        # Should call basicConfig multiple times
        assert mock_basic_config.call_count == 3
        assert mock_get_logger.call_count == 3


def test_setup_logging_with_file_and_custom_level():
    """Test setup_logging with both file handler and custom level."""
    with patch("logging.basicConfig") as mock_basic_config, patch(
        "logging.FileHandler"
    ) as mock_file_handler, patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_file_instance = MagicMock()
        mock_file_handler.return_value = mock_file_instance
        mock_get_logger.return_value = mock_logger

        test_log_file = "test.log"
        setup_logging(level="WARNING", log_file=test_log_file)

        mock_basic_config.assert_called_once()
        call_args = mock_basic_config.call_args

        # Check level
        assert call_args.kwargs["level"] == logging.WARNING

        # Check handlers
        handlers = call_args.kwargs["handlers"]
        assert len(handlers) == 2
        assert handlers[0].stream == sys.stdout
        assert handlers[1] == mock_file_instance

        # Verify logger setup
        mock_get_logger.assert_called_once_with("genai_otel")
        mock_logger.setLevel.assert_called_once_with(logging.WARNING)


def test_setup_logging_case_insensitive_level():
    """Test that level is case-insensitive."""
    with patch("logging.basicConfig") as mock_basic_config, patch(
        "logging.getLogger"
    ) as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Test lowercase
        setup_logging(level="debug")
        mock_basic_config.assert_called_once()
        call_args = mock_basic_config.call_args
        assert call_args.kwargs["level"] == logging.DEBUG

        # Reset mock for second test
        mock_basic_config.reset_mock()
        mock_get_logger.reset_mock()
        mock_get_logger.return_value = mock_logger

        # Test mixed case
        setup_logging(level="Error")
        call_args = mock_basic_config.call_args
        assert call_args.kwargs["level"] == logging.ERROR
