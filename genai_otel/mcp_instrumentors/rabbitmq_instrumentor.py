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

# Identity-tracked registry of already-instrumented ``pika`` module objects, to
# make instrument() idempotent (repeated calls must not stack wrappers). Keyed on
# object identity so mocked modules in unit tests -- a fresh object per test --
# never collide, while the process-stable real ``pika`` module is wrapped once.
_INSTRUMENTED_MODULES = []
# (owner_class, method_name) pairs wrapped, for uninstrument().
_WRAPPED_TARGETS = []


def _is_module_instrumented(module) -> bool:
    """Return True if ``module`` (by identity) was already instrumented."""
    return any(m is module for m in _INSTRUMENTED_MODULES)


class RabbitMQInstrumentor:  # pylint: disable=R0903
    """Instrument RabbitMQ clients via pika"""

    def __init__(self, config: OTelConfig):
        self.config = config
        self.tracer = trace.get_tracer(__name__)

    def instrument(self):
        """Instrument RabbitMQ/pika operations"""
        try:
            import pika  # noqa: F401

            # Double-wrap guard: skip if this module object was already wrapped.
            if _is_module_instrumented(pika):
                logger.debug("RabbitMQ already instrumented; skipping to prevent double-wrap")
                return True

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

            # Record wrapped targets and mark instrumented only after success.
            for module, method in publish_methods + consume_methods + admin_methods:
                _WRAPPED_TARGETS.append((module, method.split(".", 1)[1]))
            _INSTRUMENTED_MODULES.append(pika)

            logger.info("RabbitMQ instrumentation enabled")
            return True

        except ImportError:
            logger.debug("pika not installed, skipping RabbitMQ instrumentation")
            return False
        except Exception as e:
            logger.error("Failed to instrument RabbitMQ: %s", e, exc_info=True)
            return False

    def uninstrument(self):
        """Best-effort removal of RabbitMQ instrumentation.

        Restores wrapped ``BlockingChannel`` methods (via wrapt's
        ``__wrapped__``) and clears the double-wrap registry. Returns True if
        RabbitMQ was known to be instrumented.
        """
        try:
            import importlib

            import pika  # noqa: F401
        except ImportError:
            return False

        was_instrumented = _is_module_instrumented(pika)
        for module_path, attr_path in list(_WRAPPED_TARGETS):
            try:
                mod = importlib.import_module(module_path)
                class_name, method_name = attr_path.split(".", 1)
                owner = getattr(mod, class_name, None)
                fn = getattr(owner, method_name, None)
                original = getattr(fn, "__wrapped__", None)
                if owner is not None and original is not None:
                    setattr(owner, method_name, original)
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("Failed to unwrap RabbitMQ %s.%s: %s", module_path, attr_path, e)
        _WRAPPED_TARGETS.clear()
        _INSTRUMENTED_MODULES[:] = [m for m in _INSTRUMENTED_MODULES if m is not pika]
        if was_instrumented:
            logger.info("RabbitMQ instrumentation removed")
        return was_instrumented
