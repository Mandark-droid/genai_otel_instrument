"""Custom exceptions for better error handling"""

class InstrumentationError(Exception):
    """Base exception for instrumentation errors"""
    pass

class ProviderInstrumentationError(InstrumentationError):
    """Error instrumenting a specific provider"""
    pass

class TelemetryExportError(InstrumentationError):
    """Error exporting telemetry data"""
    pass

class ConfigurationError(InstrumentationError):
    """Error in configuration"""
    pass
