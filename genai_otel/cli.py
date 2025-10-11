"""CLI tool for running instrumented applications"""

import sys
import argparse
import runpy
import logging

from genai_otel import instrument

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the genai-instrument CLI tool.

    Parses command-line arguments, initializes OpenTelemetry instrumentation,
    and then executes the specified Python script with its arguments.
    """
    parser = argparse.ArgumentParser(
        description=("Run Python scripts with GenAI OpenTelemetry instrumentation.")
    )
    parser.add_argument("script", help="The Python script to run.")
    parser.add_argument(
        "script_args", nargs=argparse.REMAINDER, help="Arguments to pass to the script."
    )

    args = parser.parse_args()

    # Load configuration from environment variables
    # The `instrument` function will handle loading config.
    # If you need to override specific settings via CLI args, you'd parse them here
    # and pass them to `instrument`.
    try:
        # Initialize instrumentation. This reads env vars like OTEL_SERVICE_NAME, etc.
        # If GENAI_FAIL_ON_ERROR is true and setup fails, it will raise an exception.
        instrument()
    except Exception as e:
        logger.error(f"Failed to initialize instrumentation: {e}", exc_info=True)
        sys.exit(1)  # Exit if instrumentation setup fails and fail_on_error is true

    # Run the target script
    try:
        # sys.argv needs to be manipulated to pass script name and its args correctly
        # sys.argv[0] is the script name itself (e.g., genai-instrument)
        # sys.argv[1] is the target script to run
        # sys.argv[2:] are the arguments for the target script
        sys.argv = [args.script] + args.script_args
        runpy.run_path(args.script, run_name="__main__")
    except FileNotFoundError:
        logger.error(f"Script not found: {args.script}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running script {args.script}: {e}", exc_info=True)
        sys.exit(1)
