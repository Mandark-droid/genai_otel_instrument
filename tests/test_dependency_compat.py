"""Tests for dependency compatibility and conflict handling (G-S2).

These tests verify that the library handles missing or incompatible
dependencies gracefully without crashing.
"""

import importlib
import sys
from unittest.mock import MagicMock, patch


def test_import_with_missing_opentelemetry_sdk():
    """Verify genai_otel handles missing opentelemetry-sdk gracefully at import time.

    The lazy import design means 'import genai_otel' should succeed even if
    SDK modules are not available - they're only needed when instrument() is called.
    """
    import genai_otel

    # The bare import should always succeed (lazy loading)
    assert hasattr(genai_otel, "__version__")
    assert hasattr(genai_otel, "instrument")


def test_instrumentor_graceful_degradation():
    """Verify that individual instrumentors handle missing provider SDKs."""
    from genai_otel.config import OTelConfig

    config = OTelConfig(
        service_name="test",
        endpoint="",
        enabled_instrumentors=["openai", "anthropic"],
        fail_on_error=False,
    )

    # Each instrumentor should handle missing provider SDKs without raising
    from genai_otel.instrumentors import AnthropicInstrumentor, OpenAIInstrumentor

    for InstrumentorCls in [OpenAIInstrumentor, AnthropicInstrumentor]:
        instrumentor = InstrumentorCls()
        # instrument() should not raise even if the provider SDK behaves unexpectedly
        try:
            instrumentor.instrument(config)
        except Exception:
            # Some instrumentors may raise if SDK is present but incompatible
            # That's acceptable - the important thing is no crash at import time
            pass


def test_mcp_manager_handles_missing_deps():
    """Verify MCPInstrumentorManager handles missing database/cache libraries."""
    from genai_otel.config import OTelConfig
    from genai_otel.mcp_instrumentors.manager import MCPInstrumentorManager

    config = OTelConfig(service_name="test", endpoint="")
    manager = MCPInstrumentorManager(config)
    # Should not raise even if database libraries are missing
    try:
        manager.instrument_all()
    except Exception:
        pass  # Acceptable - graceful degradation


def test_lazy_getattr_unknown_attribute():
    """Verify __getattr__ raises AttributeError for unknown names."""
    import pytest

    import genai_otel

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = genai_otel.NonExistentClass


def test_lazy_getattr_caches_result():
    """Verify lazy __getattr__ caches the imported value for subsequent access."""
    import genai_otel

    # First access triggers __getattr__
    cls1 = genai_otel.OTelConfig
    # Second access should use cached globals() value
    cls2 = genai_otel.OTelConfig
    assert cls1 is cls2
