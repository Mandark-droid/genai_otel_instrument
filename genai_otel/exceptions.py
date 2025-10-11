"""Custom exceptions for better error handling"""


class InstrumentationError(Exception):
    """Base exception for instrumentation errors"""


class ProviderInstrumentationError(InstrumentationError):
    """Error instrumenting a specific provider"""


class TelemetryExportError(InstrumentationError):
    """Error exporting telemetry data"""


class ConfigurationError(InstrumentationError):
    """Error in configuration"""
