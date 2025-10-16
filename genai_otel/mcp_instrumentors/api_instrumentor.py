"""OpenTelemetry instrumentor for generic API calls.

This module provides the `APIInstrumentor` class, which automatically traces
HTTP requests made using popular libraries like `requests` and `httpx`.
It enriches spans with relevant attributes, including detected GenAI system
information based on the URL.
"""

import logging
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx
import requests
import wrapt

from genai_otel.instrumentors.base import BaseInstrumentor

from ..config import OTelConfig

logger = logging.getLogger(__name__)


class APIInstrumentor(BaseInstrumentor):
    """Instrument custom API calls, adding GenAI-specific attributes.

    This instrumentor targets common HTTP client libraries like `requests` and `httpx`.
    It aims to add relevant attributes to spans, including GenAI system information
    if detectable from the URL or headers.
    """

    def __init__(self, config: OTelConfig):
        """Initializes the APIInstrumentor.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        super().__init__()  # Initialize BaseInstrumentor
        self.config = config

    def instrument(self, config: OTelConfig):
        """Instrument requests and httpx libraries for API calls.

        Applies wrappers to `requests.request`, `requests.Session.request`,
        and `httpx.Client.request` to capture API call details.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        self.config = config  # Store the config

        # CRITICAL: Do NOT wrap requests library when using OTLP HTTP exporters
        # Wrapping requests.Session breaks OTLP exporters that use it internally
        # try:
        #     # Wrap requests.Session.request and requests.request
        #     wrapt.wrap_function_wrapper(requests, "request", self._wrap_api_call)
        #     wrapt.wrap_function_wrapper(requests.sessions.Session, "request", self._wrap_api_call)
        #     logger.info("requests library instrumented for API calls.")
        # except ImportError:
        #     logger.debug("requests library not found, skipping instrumentation.")
        # except Exception as e:
        #     logger.error("Failed to instrument requests library: %s", e, exc_info=True)
        #     if self.config.fail_on_error:
        #         raise

        logger.warning(
            "requests library instrumentation disabled to prevent OTLP exporter conflicts"
        )

        try:
            # Wrap httpx.Client.request
            wrapt.wrap_function_wrapper(httpx.Client, "request", self._wrap_api_call)
            logger.info("httpx library instrumented for API calls.")
        except ImportError:
            logger.debug("httpx library not found, skipping instrumentation.")
        except Exception as e:
            logger.error("Failed to instrument httpx library: %s", e, exc_info=True)
            if self.config.fail_on_error:
                raise

    def _wrap_api_call(self, wrapped, instance, args, kwargs):
        """Wrapper function for API calls using create_span_wrapper.

        This method prepares the arguments for `create_span_wrapper` and applies it.
        """
        method = kwargs.get("method", args[0] if args else "unknown").upper()
        url = kwargs.get("url", args[1] if len(args) > 1 else None)
        span_name = f"api.call.{method.lower()}"
        if url:
            try:
                parsed_url = urlparse(url)
                span_name = f"api.call.{method.lower()}.{parsed_url.hostname}"
            except Exception:
                pass  # Keep default span name if URL parsing fails

        instrumented_call = self.create_span_wrapper(
            span_name=span_name, extract_attributes=self._extract_api_attributes
        )
        return instrumented_call(wrapped, instance, args, kwargs)

    def _extract_api_attributes(
        self, instance: Any, args: Any, kwargs: Any
    ) -> Dict[str, Any]:  # pylint: disable=W0613
        """Extract attributes from API call arguments for OpenTelemetry spans.

        Args:
            instance: The instance of the class the method is called on (e.g., requests.Session).
            args: Positional arguments passed to the method.
            kwargs: Keyword arguments passed to the method.

        Returns:
            Dict[str, Any]: A dictionary of attributes to be set on the span.
        """
        attrs = {}
        method = kwargs.get("method", args[0] if args else "unknown").upper()
        url = kwargs.get("url", args[1] if len(args) > 1 else None)

        if url:
            try:
                parsed_url = urlparse(url)
                if parsed_url.hostname:
                    attrs["net.peer.name"] = parsed_url.hostname
                attrs["url.full"] = url
                attrs["http.method"] = method
            except Exception as e:
                logger.warning("Failed to parse URL '%s' for attributes: %s", url, e)

        if url:
            if "openai.com" in url:
                attrs["gen_ai.system"] = "openai"
            elif "anthropic.com" in url:
                attrs["gen_ai.system"] = "anthropic"
            elif "google.com" in url:
                attrs["gen_ai.system"] = "google"

        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:  # pylint: disable=W0613
        """API calls typically don't have direct token usage like LLMs.

        This method is part of the BaseInstrumentor interface but is not implemented
        for generic API calls as token usage is not a standard concept here.
        """
        return None
