"""Custom exceptions for better error handling"""


class InstrumentationError(Exception):
    """Base exception for instrumentation errors"""


class ProviderInstrumentationError(InstrumentationError):
    """Error instrumenting a specific provider"""


class TelemetryExportError(InstrumentationError):
    """Error exporting telemetry data"""


class ConfigurationError(InstrumentationError):
    """Error in configuration"""


class PolicyViolationError(Exception):
    """Raised to BLOCK an LLM request before it is sent.

    Emitted by the pre-call evaluation hook when a detector configured with
    ``block_on_detection`` (or PII ``block`` mode) flags the prompt. Unlike the
    ``InstrumentationError`` family, this is NOT an instrumentation failure - it
    is a deliberate, caller-visible policy decision, so the wrapped call is never
    executed and the exception propagates to the application.
    """

    def __init__(self, message, *, policy=None, details=None):
        super().__init__(message)
        self.policy = policy
        self.details = details or {}
