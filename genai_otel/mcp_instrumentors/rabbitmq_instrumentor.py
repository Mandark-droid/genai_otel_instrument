"""OpenTelemetry instrumentor for RabbitMQ clients via pika.

This module provides the `RabbitMQInstrumentor` class, which automatically
instruments RabbitMQ operations such as publishing, consuming, and queue
management, enabling tracing of message broker interactions within GenAI
applications.
"""

import logging

import wrapt
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from ..config import OTelConfig

logger = logging.getLogger(__name__)


class RabbitMQInstrumentor:  # pylint: disable=R0903
    """Instrument RabbitMQ clients via pika"""

    def __init__(self, config: OTelConfig):
        self.config = config
        self.tracer = trace.get_tracer(__name__)

    def instrument(self):
        """Instrument RabbitMQ/pika operations"""
        try:
            import pika  # noqa: F401

            tracer = self.tracer

            # Methods to instrument on BlockingChannel
            publish_methods = [
                ("pika.adapters.blocking_connection", "BlockingChannel.basic_publish"),
            ]
            consume_methods = [
                ("pika.adapters.blocking_connection", "BlockingChannel.basic_consume"),
                ("pika.adapters.blocking_connection", "BlockingChannel.basic_get"),
            ]
            admin_methods = [
                ("pika.adapters.blocking_connection", "BlockingChannel.queue_declare"),
                ("pika.adapters.blocking_connection", "BlockingChannel.queue_delete"),
                ("pika.adapters.blocking_connection", "BlockingChannel.exchange_declare"),
                ("pika.adapters.blocking_connection", "BlockingChannel.exchange_delete"),
            ]

            def _create_wrapper(operation, span_kind=SpanKind.CLIENT):
                def wrapper(wrapped, instance, args, kwargs):
                    span_name = f"rabbitmq.{operation}"
                    with tracer.start_as_current_span(span_name, kind=span_kind) as span:
                        span.set_attribute("messaging.system", "rabbitmq")
                        span.set_attribute("messaging.operation", operation)
                        # Try to extract exchange and routing_key
                        if operation == "basic_publish":
                            exchange = kwargs.get("exchange", args[0] if args else "")
                            routing_key = kwargs.get(
                                "routing_key", args[1] if len(args) > 1 else ""
                            )
                            if exchange:
                                span.set_attribute("messaging.destination", str(exchange))
                            if routing_key:
                                span.set_attribute(
                                    "messaging.rabbitmq.routing_key",
                                    str(routing_key),
                                )
                        elif operation in ("queue_declare", "queue_delete"):
                            queue = kwargs.get("queue", args[0] if args else "")
                            if queue:
                                span.set_attribute("messaging.destination", str(queue))
                        elif operation in ("exchange_declare", "exchange_delete"):
                            exchange = kwargs.get("exchange", args[0] if args else "")
                            if exchange:
                                span.set_attribute("messaging.destination", str(exchange))
                        elif operation == "basic_consume":
                            queue = kwargs.get("queue", args[0] if args else "")
                            if queue:
                                span.set_attribute("messaging.destination", str(queue))
                        try:
                            result = wrapped(*args, **kwargs)
                            span.set_status(Status(StatusCode.OK))
                            return result
                        except Exception as e:
                            span.set_status(Status(StatusCode.ERROR, str(e)))
                            span.record_exception(e)
                            raise

                return wrapper

            for module, method in publish_methods:
                op = method.split(".")[-1]
                wrapt.wrap_function_wrapper(
                    module,
                    method.split(".", 1)[1],
                    _create_wrapper(op, SpanKind.PRODUCER),
                )

            for module, method in consume_methods:
                op = method.split(".")[-1]
                wrapt.wrap_function_wrapper(
                    module,
                    method.split(".", 1)[1],
                    _create_wrapper(op, SpanKind.CONSUMER),
                )

            for module, method in admin_methods:
                op = method.split(".")[-1]
                wrapt.wrap_function_wrapper(
                    module,
                    method.split(".", 1)[1],
                    _create_wrapper(op),
                )

            logger.info("RabbitMQ instrumentation enabled")
            return True

        except ImportError:
            logger.debug("pika not installed, skipping RabbitMQ instrumentation")
            return False
        except Exception as e:
            logger.error("Failed to instrument RabbitMQ: %s", e, exc_info=True)
            return False
