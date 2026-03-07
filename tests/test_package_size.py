"""Tests for package size and import time constraints (G-D3, G-P1)."""

import subprocess
import sys
import time

# Maximum allowed installed package size in KB
MAX_PACKAGE_SIZE_KB = 5 * 1024  # 5 MB


def test_package_installed_size():
    """Verify the installed package does not exceed 5MB."""
    import importlib.util
    import os

    spec = importlib.util.find_spec("genai_otel")
    assert spec is not None, "genai_otel is not installed"
    package_dir = os.path.dirname(spec.origin)

    total_size = 0
    for dirpath, _dirnames, filenames in os.walk(package_dir):
        for f in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, f))

    size_kb = total_size / 1024
    assert (
        size_kb < MAX_PACKAGE_SIZE_KB
    ), f"Package size {size_kb:.0f} KB exceeds limit of {MAX_PACKAGE_SIZE_KB} KB"


def test_import_time_under_threshold():
    """Verify that 'import genai_otel' completes in under 500ms.

    This runs in a subprocess to get a clean import measurement.
    """
    code = (
        "import time; s=time.perf_counter(); import genai_otel; "
        "print(int((time.perf_counter()-s)*1000))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"Import failed: {result.stderr}"
    elapsed_ms = int(result.stdout.strip())
    # Allow up to 500ms for the bare import (no attribute access)
    assert elapsed_ms < 500, f"import genai_otel took {elapsed_ms}ms, exceeds 500ms threshold"
