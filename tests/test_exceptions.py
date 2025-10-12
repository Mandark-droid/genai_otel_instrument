import pytest

from genai_otel.exceptions import (
    ConfigurationError,
    InstrumentationError,
    ProviderInstrumentationError,
    TelemetryExportError,
)


def test_instrumentation_error():
    """Test InstrumentationError can be raised and caught."""
    with pytest.raises(InstrumentationError, match="Test instrumentation error"):
        raise InstrumentationError("Test instrumentation error")


def test_provider_instrumentation_error():
    """Test ProviderInstrumentationError can be raised and caught."""
    with pytest.raises(ProviderInstrumentationError, match="Test provider error"):
        raise ProviderInstrumentationError("Test provider error")


def test_telemetry_export_error():
    """Test TelemetryExportError can be raised and caught."""
    with pytest.raises(TelemetryExportError, match="Test telemetry export error"):
        raise TelemetryExportError("Test telemetry export error")


def test_configuration_error():
    """Test ConfigurationError can be raised and caught."""
    with pytest.raises(ConfigurationError, match="Test configuration error"):
        raise ConfigurationError("Test configuration error")


def test_provider_instrumentation_error_inherits_instrumentation_error():
    """Test ProviderInstrumentationError inherits from InstrumentationError."""
    with pytest.raises(InstrumentationError):
        raise ProviderInstrumentationError("Inheritance test")


def test_telemetry_export_error_inherits_instrumentation_error():
    """Test TelemetryExportError inherits from InstrumentationError."""
    with pytest.raises(InstrumentationError):
        raise TelemetryExportError("Inheritance test")


def test_configuration_error_inherits_instrumentation_error():
    """Test ConfigurationError inherits from InstrumentationError."""
    with pytest.raises(InstrumentationError):
        raise ConfigurationError("Inheritance test")
