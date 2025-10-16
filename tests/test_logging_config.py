import logging
import os
import shutil
import sys
import time
from logging.handlers import RotatingFileHandler
from unittest.mock import MagicMock, call, patch

import pytest

from genai_otel.logging_config import setup_logging


@pytest.fixture(scope="function")
def setup_log_env():
    """Fixture to set up and tear down a temporary logs directory and environment variable."""
    log_dir = "test_logs"

    # Ensure a clean state before each test
    logger = logging.getLogger("genai_otel")
    for handler in logger.handlers[:]:  # Iterate over a copy
        handler.close()
        logger.removeHandler(handler)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:  # Iterate over a copy
        handler.close()
        root_logger.removeHandler(handler)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)

    # Set environment variable for testing
    os.environ["GENAI_OTEL_LOG_LEVEL"] = "DEBUG"

    yield log_dir

    # Teardown: Close all handlers associated with the 'genai_otel' logger
    logger = logging.getLogger("genai_otel")
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    # Teardown: Close all handlers associated with the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)

    # Explicitly remove the log file first, then the directory
    log_file_path = os.path.join(log_dir, "genai_otel.log")
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    if os.path.exists(log_dir):
        time.sleep(0.1)  # Small delay to allow file handle to be released
        shutil.rmtree(log_dir)
    if "GENAI_OTEL_LOG_LEVEL" in os.environ:
        del os.environ["GENAI_OTEL_LOG_LEVEL"]


def test_setup_logging_default_level():
    """Test setup_logging with default INFO level and no file handler."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        logger = setup_logging()

        # Verify logger setup
        mock_get_logger.assert_called_once_with("genai_otel")
        mock_logger.setLevel.assert_called_once_with(logging.INFO)

        # Check handlers - should have two handlers (StreamHandler and RotatingFileHandler)
        handlers = mock_logger.addHandler.call_args_list
        assert len(handlers) == 2
        assert isinstance(handlers[0].args[0], logging.StreamHandler)
        assert handlers[0].args[0].stream == sys.stdout
        assert isinstance(handlers[1].args[0], RotatingFileHandler)

        assert logger == mock_logger


def test_setup_logging_custom_level():
    """Test setup_logging with a custom logging level (e.g., DEBUG)."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        setup_logging(level="DEBUG")

        mock_get_logger.assert_called_once_with("genai_otel")
        mock_logger.setLevel.assert_called_once_with(logging.DEBUG)


def test_setup_logging_with_file_handler():
    """Test setup_logging with a log file specified."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        test_log_file = "test.log"
        setup_logging(log_file_name=test_log_file)

        # Verify logger setup
        mock_get_logger.assert_called_once_with("genai_otel")
        mock_logger.setLevel.assert_called_once_with(logging.INFO)

        # Check that handlers include both StreamHandler and RotatingFileHandler
        handlers = mock_logger.addHandler.call_args_list
        assert len(handlers) == 2
        assert isinstance(handlers[0].args[0], logging.StreamHandler)
        assert handlers[0].args[0].stream == sys.stdout
        assert isinstance(handlers[1].args[0], RotatingFileHandler)
        assert handlers[1].args[0].baseFilename.endswith(test_log_file)


def test_setup_logging_returns_logger():
    """Test that setup_logging returns the configured logger instance."""
    with patch("logging.getLogger") as mock_get_logger:
        expected_logger = MagicMock()
        mock_get_logger.return_value = expected_logger

        returned_logger = setup_logging()
        assert returned_logger == expected_logger


def test_setup_logging_format():
    """Test that logging format is properly set."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        setup_logging()

        # Check that format is specified
        for handler_call in mock_logger.addHandler.call_args_list:
            handler = handler_call.args[0]
            if handler.formatter:
                format_str = handler.formatter._fmt
                assert "%(asctime)s" in format_str
                assert "%(levelname)s" in format_str
                assert "%(name)s" in format_str
                assert "%(message)s" in format_str


def test_setup_logging_invalid_level_falls_back_to_info():
    """Test that invalid level falls back to INFO."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        setup_logging(level="INVALID_LEVEL")

        mock_get_logger.assert_called_once_with("genai_otel")
        mock_logger.setLevel.assert_called_once_with(logging.INFO)


def test_setup_logging_stream_handler_uses_stdout():
    """Test that StreamHandler uses sys.stdout."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        setup_logging()

        handlers = mock_logger.addHandler.call_args_list
        stream_handler_found = False
        for handler_call in handlers:
            handler = handler_call.args[0]
            if isinstance(handler, logging.StreamHandler):
                stream_handler_found = True
                assert handler.stream == sys.stdout
                break
        assert stream_handler_found


def test_setup_logging_multiple_calls():
    """Test that setup_logging can be called multiple times safely."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Call setup_logging multiple times
        setup_logging()
        setup_logging(level="DEBUG")
        setup_logging(log_file_name="test.log")

        # Should call getLogger multiple times
        assert mock_get_logger.call_count == 3
        # Handlers should be cleared and re-added each time
        assert mock_logger.addHandler.call_count == 6  # 2 handlers * 3 calls


def test_setup_logging_with_file_and_custom_level():
    """Test setup_logging with both file handler and custom level."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        test_log_file = "test.log"
        setup_logging(level="WARNING", log_file_name=test_log_file)

        # Check level
        mock_get_logger.assert_called_once_with("genai_otel")
        mock_logger.setLevel.assert_called_once_with(logging.WARNING)

        # Check handlers
        handlers = mock_logger.addHandler.call_args_list
        assert len(handlers) == 2
        assert isinstance(handlers[0].args[0], logging.StreamHandler)
        assert handlers[0].args[0].stream == sys.stdout
        assert isinstance(handlers[1].args[0], RotatingFileHandler)
        assert handlers[1].args[0].baseFilename.endswith(test_log_file)


def test_setup_logging_case_insensitive_level():
    """Test that level is case-insensitive."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Test lowercase
        setup_logging(level="debug")
        mock_get_logger.assert_called_once_with("genai_otel")
        mock_logger.setLevel.assert_called_once_with(logging.DEBUG)

        # Reset mock for second test
        mock_get_logger.reset_mock()
        mock_logger.reset_mock()
        mock_get_logger.return_value = mock_logger

        # Test mixed case
        setup_logging(level="Error")
        mock_get_logger.assert_called_once_with("genai_otel")
        mock_logger.setLevel.assert_called_once_with(logging.ERROR)


def test_setup_logging_with_env_var_and_rotation(setup_log_env):
    """
    Test that logging is configured correctly with environment variable,
    creates a logs directory, and uses RotatingFileHandler.
    """
    log_dir = setup_log_env
    log_file_name = "genai_otel.log"
    log_file_path = os.path.join(log_dir, log_file_name)

    # Ensure no handlers are present from previous tests
    logging.getLogger().handlers = []
    logging.getLogger("genai_otel").handlers = []

    logger = setup_logging(log_dir=log_dir, log_file_name=log_file_name)

    assert logger.level == logging.DEBUG
    assert os.path.exists(log_dir)
    assert os.path.exists(log_file_path)

    # Check if RotatingFileHandler is used
    file_handler_found = False
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            file_handler_found = True
            assert handler.baseFilename.endswith(log_file_path)
            assert handler.maxBytes == 10 * 1024 * 1024
            assert handler.backupCount == 10
            break
    assert file_handler_found, "RotatingFileHandler not found in logger handlers"

    # Test logging to file
    test_message = "This is a test log message."
    logger.debug(test_message)

    with open(log_file_path, "r") as f:
        content = f.read()
        assert test_message in content


def test_setup_logging_default_level_no_env_var(setup_log_env):
    """
    Test that logging defaults to INFO if no environment variable is set
    and no level is passed to the function.
    """
    log_dir = setup_log_env
    log_file_name = "genai_otel.log"

    # Unset environment variable for this specific test
    if "GENAI_OTEL_LOG_LEVEL" in os.environ:
        del os.environ["GENAI_OTEL_LOG_LEVEL"]

    # Ensure no handlers are present from previous tests
    logging.getLogger().handlers = []
    logging.getLogger("genai_otel").handlers = []

    logger = setup_logging(log_dir=log_dir, log_file_name=log_file_name)
    assert logger.level == logging.INFO


def test_setup_logging_explicit_level_overrides_env_var(setup_log_env):
    """
    Test that an explicit level passed to setup_logging overrides the environment variable.
    """
    log_dir = setup_log_env
    log_file_name = "genai_otel.log"

    # GENAI_OTEL_LOG_LEVEL is set to DEBUG by fixture

    # Ensure no handlers are present from previous tests
    logging.getLogger().handlers = []
    logging.getLogger("genai_otel").handlers = []

    logger = setup_logging(level="WARNING", log_dir=log_dir, log_file_name=log_file_name)
    assert logger.level == logging.WARNING


def test_setup_logging_closes_existing_handlers():
    """Test that existing handlers are properly closed when setup_logging is called again."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Create mock handlers with close method
        mock_handler1 = MagicMock()
        mock_handler2 = MagicMock()
        mock_logger.handlers = [mock_handler1, mock_handler2]

        # Call setup_logging - should close existing handlers
        setup_logging()

        # Verify both handlers were closed
        mock_handler1.close.assert_called_once()
        mock_handler2.close.assert_called_once()
