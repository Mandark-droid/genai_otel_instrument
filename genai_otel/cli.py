"""CLI tool for running instrumented applications"""

import argparse
import logging
import os
import runpy
import sys

from genai_otel import instrument

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the genai-instrument CLI tool.

    Parses command-line arguments, initializes OpenTelemetry instrumentation,
    and then executes the specified command/script with its arguments.

    Supports two usage patterns:
    1. genai-instrument python script.py [args...]
    2. genai-instrument script.py [args...]

    In both cases, the Python script is executed in the same process to ensure
    instrumentation hooks are active.
    """
    parser = argparse.ArgumentParser(
        description=("Run a Python script with GenAI OpenTelemetry instrumentation.")
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="The command to run (python script.py or script.py)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Load configuration from environment variables
    # The `instrument` function will handle loading config.
    try:
        # Initialize instrumentation. This reads env vars like OTEL_SERVICE_NAME, etc.
        # If GENAI_FAIL_ON_ERROR is true and setup fails, it will raise an exception.
        instrument()
    except Exception as e:
        logger.error(f"Failed to initialize instrumentation: {e}", exc_info=True)
        sys.exit(1)  # Exit if instrumentation setup fails and fail_on_error is true

    # Parse the command to extract the Python script and its arguments
    script_path = None
    script_args = []

    # Check if command starts with 'python' or 'python3' or 'python.exe'
    if args.command[0].lower() in [
        "python",
        "python3",
        "python.exe",
        "python3.exe",
    ] or os.path.basename(args.command[0]).lower().startswith("python"):
        # Format: genai-instrument python script.py [args...]
        if len(args.command) < 2:
            logger.error("No Python script specified after 'python' command")
            sys.exit(1)
        script_path = args.command[1]
        script_args = args.command[2:]
    elif args.command[0].endswith(".py"):
        # Format: genai-instrument script.py [args...]
        script_path = args.command[0]
        script_args = args.command[1:]
    else:
        logger.error(
            f"Invalid command format. Expected 'python script.py' or 'script.py', got: {' '.join(args.command)}"
        )
        sys.exit(1)

    # Set sys.argv to simulate running the script directly
    # This ensures the target script receives the correct arguments
    sys.argv = [script_path] + script_args

    # Run the target script in the same process using runpy
    # This ensures instrumentation hooks are active in the script
    try:
        runpy.run_path(script_path, run_name="__main__")
    except FileNotFoundError:
        logger.error(f"Script not found: {script_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running script {script_path}: {e}", exc_info=True)
        sys.exit(1)
