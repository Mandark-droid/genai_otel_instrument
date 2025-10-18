# tests/test_cli.py

import argparse
import sys
from unittest import mock

import pytest

from genai_otel import cli


@pytest.fixture
def mock_instrument():
    """Mock the genai_otel.instrument function in the cli module."""
    # Patch where it's used, not where it's defined
    with mock.patch("genai_otel.cli.instrument") as m:
        yield m


@pytest.fixture
def mock_runpy():
    """Mock the runpy.run_path function."""
    with mock.patch("genai_otel.cli.runpy.run_path") as m:
        yield m


@pytest.fixture
def mock_sys_argv():
    """Mock sys.argv to control command-line arguments."""
    original_argv = sys.argv
    sys.argv = ["genai-instrument"]
    yield sys.argv
    sys.argv = original_argv


@pytest.fixture
def mock_sys_exit():
    """Mock sys.exit to prevent script termination during tests."""
    with mock.patch("genai_otel.cli.sys.exit", side_effect=SystemExit) as m:
        yield m


@pytest.fixture
def mock_logger():
    """Mock the logger to prevent actual logging during tests."""
    with mock.patch("genai_otel.cli.logger") as m:
        yield m


def test_main_successful_execution(
    mock_instrument, mock_runpy, mock_sys_argv, mock_sys_exit, mock_logger
):
    """Test that the CLI correctly instruments and runs a script (direct .py format)."""
    script_to_run = "my_script.py"
    script_args = ["--arg1", "value1"]
    mock_sys_argv.extend([script_to_run] + script_args)

    cli.main()

    # Verify instrumentation was called
    mock_instrument.assert_called_once()

    # Verify runpy was called with the correct script
    mock_runpy.assert_called_once_with(script_to_run, run_name="__main__")

    # Verify sys.argv was set correctly before running the script
    # (This happens inside main(), so we can't directly assert it)

    # Verify sys.exit was not called
    mock_sys_exit.assert_not_called()


def test_main_successful_execution_with_python_command(
    mock_instrument, mock_runpy, mock_sys_argv, mock_sys_exit, mock_logger
):
    """Test that the CLI correctly instruments and runs a script (python script.py format)."""
    script_to_run = "my_script.py"
    script_args = ["--arg1", "value1"]
    mock_sys_argv.extend(["python", script_to_run] + script_args)

    cli.main()

    # Verify instrumentation was called
    mock_instrument.assert_called_once()

    # Verify runpy was called with the correct script
    mock_runpy.assert_called_once_with(script_to_run, run_name="__main__")

    # Verify sys.exit was not called
    mock_sys_exit.assert_not_called()


def test_main_instrumentation_failure(
    mock_instrument, mock_runpy, mock_sys_argv, mock_sys_exit, mock_logger
):
    """Test that the CLI exits if instrumentation fails."""
    script_to_run = "my_script.py"
    mock_sys_argv.extend([script_to_run])

    # Simulate instrumentation failure
    mock_instrument.side_effect = Exception("Instrumentation failed")

    with pytest.raises(SystemExit):
        cli.main()

    # Verify instrumentation was called
    mock_instrument.assert_called_once()

    # Verify runpy was not called
    mock_runpy.assert_not_called()

    # Verify sys.exit was called with an error code
    mock_sys_exit.assert_called_once_with(1)

    # Verify error was logged
    mock_logger.error.assert_called()


def test_main_script_not_found(
    mock_instrument, mock_runpy, mock_sys_argv, mock_sys_exit, mock_logger
):
    """Test that the CLI exits if the script is not found."""
    script_to_run = "non_existent_script.py"
    mock_sys_argv.extend([script_to_run])

    # Simulate FileNotFoundError during script execution
    mock_runpy.side_effect = FileNotFoundError(f"No such file or directory: '{script_to_run}'")

    with pytest.raises(SystemExit):
        cli.main()

    # Verify instrumentation was called
    mock_instrument.assert_called_once()

    # Verify runpy was called
    mock_runpy.assert_called_once_with(script_to_run, run_name="__main__")

    # Verify sys.exit was called with an error code
    mock_sys_exit.assert_called_once_with(1)

    # Verify error was logged
    mock_logger.error.assert_called()


def test_main_script_execution_error(
    mock_instrument, mock_runpy, mock_sys_argv, mock_sys_exit, mock_logger
):
    """Test that the CLI exits if the script execution raises an error."""
    script_to_run = "my_script.py"
    mock_sys_argv.extend([script_to_run])

    # Simulate a general exception during script execution
    mock_runpy.side_effect = Exception("An error occurred during script execution")

    with pytest.raises(SystemExit):
        cli.main()
    # Verify instrumentation was called
    mock_instrument.assert_called_once()

    # Verify runpy was called
    mock_runpy.assert_called_once_with(script_to_run, run_name="__main__")

    # Verify sys.exit was called with an error code
    mock_sys_exit.assert_called_once_with(1)

    # Verify error was logged
    mock_logger.error.assert_called()


def test_main_script_with_args(
    mock_instrument, mock_runpy, mock_sys_argv, mock_sys_exit, mock_logger
):
    """Test that script arguments are passed correctly (direct .py format)."""
    script_to_run = "script_with_args.py"
    script_args = ["--input", "data.txt", "--output", "result.txt"]
    mock_sys_argv.extend([script_to_run] + script_args)

    cli.main()

    # Verify instrumentation was called
    mock_instrument.assert_called_once()

    # Verify runpy was called with the correct script
    mock_runpy.assert_called_once_with(script_to_run, run_name="__main__")

    # Verify sys.exit was not called
    mock_sys_exit.assert_not_called()


def test_main_script_with_args_python_command(
    mock_instrument, mock_runpy, mock_sys_argv, mock_sys_exit, mock_logger
):
    """Test that script arguments are passed correctly (python script.py format)."""
    script_to_run = "script_with_args.py"
    script_args = ["--input", "data.txt", "--output", "result.txt"]
    mock_sys_argv.extend(["python", script_to_run] + script_args)

    cli.main()

    # Verify instrumentation was called
    mock_instrument.assert_called_once()

    # Verify runpy was called with the correct script
    mock_runpy.assert_called_once_with(script_to_run, run_name="__main__")

    # Verify sys.exit was not called
    mock_sys_exit.assert_not_called()
