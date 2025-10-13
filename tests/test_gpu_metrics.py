import sys
from unittest.mock import Mock

import pytest


# Configure pytest
def pytest_configure(config):
    # Add module-level mocks
    sys.modules["pynvml"] = Mock()
    sys.modules["opentelemetry.metrics"] = Mock()
    sys.modules["genai_otel.config"] = Mock()


# Common fixtures
@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks before each test"""
    import genai_otel.gpu_metrics

    if hasattr(genai_otel.gpu_metrics, "NVML_AVAILABLE"):
        genai_otel.gpu_metrics.NVML_AVAILABLE = True
    yield
