"""Small activation bridges for agent SDKs that emit OpenTelemetry natively."""

from opentelemetry import trace


class MicrosoftAgentFrameworkInstrumentor:
    """Enable Microsoft Agent Framework's native telemetry without wrapping it."""

    def instrument(self, config=None):
        """Enable framework telemetry while keeping sensitive message capture off."""
        from agent_framework.observability import enable_instrumentation

        enable_instrumentation(enable_sensitive_data=False)


class StrandsAgentsInstrumentor:
    """Activate Strands' native OpenTelemetry spans through the global provider."""

    def instrument(self, config=None):
        """Confirm that Strands is installed and uses the configured global provider."""
        import strands  # noqa: F401

        # Strands emits native GenAI OTel spans. TraceVerde's cost exporter reads
        # their model and token attributes directly, avoiding a second span tree.
        trace.get_tracer_provider()
