"""CLI tool for running instrumented applications"""
import sys
import os
import argparse
import runpy


def main():
    """Entry point for CLI"""
    parser = argparse.ArgumentParser(
        description="Run a Python application with GenAI OpenTelemetry instrumentation"
    )
    parser.add_argument("script", help="Python script to run")
    parser.add_argument("args", nargs="*", help="Arguments to pass to the script")

    args = parser.parse_args()

    # Set up instrumentation
    import genai_otel
    genai_otel.instrument()

    # Modify sys.argv for the target script
    sys.argv = [args.script] + args.args

    # Run the target script
    runpy.run_path(args.script, run_name="__main__")


if __name__ == "__main__":
    main()
