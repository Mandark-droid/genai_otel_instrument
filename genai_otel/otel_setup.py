# genai_otel/otel_setup.py
import logging
from typing import Optional

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from .config import OTelConfig

logger = logging.getLogger(__name__)


def configure_opentelemetry(config: OTelConfig):
    """
    Configures and initializes the OpenTelemetry SDK for tracing and metrics.
    """
    resource = Resource.create({"service.name": config.service_name})

    # Configure Tracing
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)

    if config.endpoint:
        span_exporter = OTLPSpanExporter(endpoint=config.endpoint)
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        logger.info(f"OpenTelemetry tracing configured with endpoint: {config.endpoint}")
    else:
        logger.warning("No OTLP endpoint configured, traces will not be exported.")

    # Configure Metrics
    if config.endpoint:
        metric_exporter = OTLPMetricExporter(endpoint=config.endpoint)
        metric_reader = PeriodicExportingMetricReader(exporter=metric_exporter)
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        logger.info("OpenTelemetry metrics configured")
    else:
        # Still set a default meter provider even if not exporting
        meter_provider = MeterProvider(resource=resource)
        metrics.set_meter_provider(meter_provider)
        logger.warning("No OTLP endpoint configured, metrics will not be exported.")
