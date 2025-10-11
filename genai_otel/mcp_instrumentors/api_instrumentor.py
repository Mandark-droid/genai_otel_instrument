import logging
import wrapt
import json
from typing import Dict, Any, Optional, Callable

from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode
from urllib.parse import urlparse # Import urlparse here

from ..config import OTelConfig
from .base import BaseInstrumentor # Inherit from BaseInstrumentor

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
        super().__init__() # Initialize BaseInstrumentor
        self.config = config

    def instrument(self):
        """Instrument requests and httpx libraries for API calls.

        Applies wrappers to `requests.request`, `requests.Session.request`, 
        and `httpx.Client.request` to capture API call details.
        """
        try:
            import requests
            # Wrap requests.Session.request and requests.request
            wrapt.wrap_function_wrapper(requests, "request", self._wrap_api_call)
            wrapt.wrap_function_wrapper(requests.Session, "request", self._wrap_api_call)
            logger.info("requests library instrumented for API calls.")
        except ImportError:
            logger.debug("requests library not found, skipping instrumentation.")
        except Exception as e:
            logger.error(f"Failed to instrument requests library: {e}", exc_info=True)
            if self.config.fail_on_error:
                raise

        try:
            import httpx
            # Wrap httpx.Client.request
            wrapt.wrap_function_wrapper(httpx.Client, "request", self._wrap_api_call)
            logger.info("httpx library instrumented for API calls.")
        except ImportError:
            logger.debug("httpx library not found, skipping instrumentation.")
        except Exception as e:
            logger.error(f"Failed to instrument httpx library: {e}", exc_info=True)
            if self.config.fail_on_error:
                raise

    def _wrap_api_call(self, wrapped, instance, args, kwargs):
        """Wrapper function for API calls using create_span_wrapper.

        This method prepares the arguments for `create_span_wrapper` and applies it.
        """
        # Determine span name based on HTTP method and URL.
        method = kwargs.get('method', args[0] if args else 'unknown').upper()
        url = kwargs.get('url', args[1] if len(args) > 1 else None)
        span_name = f"api.call.{method.lower()}"
        if url:
            try:
                parsed_url = urlparse(url)
                span_name = f"api.call.{method.lower()}.{parsed_url.hostname}"
            except Exception:
                pass # Keep default span name if URL parsing fails

        # Use create_span_wrapper to handle span creation, timing, and metric recording.
        # The actual attribute extraction is delegated to _extract_api_attributes.
        instrumented_call = self.create_span_wrapper(
            span_name=span_name,
            extract_attributes=self._extract_api_attributes
        )
        return instrumented_call(wrapped, instance, args, kwargs)

    def _extract_api_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Extract attributes from API call arguments for OpenTelemetry spans.

        Args:
            instance: The instance of the class the method is called on (e.g., requests.Session).
            args: Positional arguments passed to the method.
            kwargs: Keyword arguments passed to the method.

        Returns:
            Dict[str, Any]: A dictionary of attributes to be set on the span.
        """
        attrs = {}
        method = kwargs.get('method', args[0] if args else 'unknown').upper()
        url = kwargs.get('url', args[1] if len(args) > 1 else None)

        if url:
            try:
                parsed_url = urlparse(url)
                if parsed_url.hostname:
                    attrs["net.peer.name"] = parsed_url.hostname
                attrs["url.full"] = url
                attrs["http.method"] = method
            except Exception as e:
                logger.warning(f"Failed to parse URL '{url}' for attributes: {e}")

        # Add GenAI specific attributes if detectable (e.g., based on URL patterns or headers)
        # This is a placeholder and would need more sophisticated logic for robust detection.
        if url:
            if "openai.com" in url:
                attrs["gen_ai.system"] = "openai"
            elif "anthropic.com" in url:
                attrs["gen_ai.system"] = "anthropic"
            elif "google.com" in url:
                attrs["gen_ai.system"] = "google"
            # Add more providers as needed

        # Optionally add request body/headers if relevant and not too large/sensitive.
        # For production, consider logging these only at DEBUG level or based on config.
        # Example: Truncate body to avoid excessive log size.
        # body = kwargs.get('data') or kwargs.get('json')
        # if body:
        #     try:
        #         attrs['http.request.body'] = json.dumps(body)[:200] # Truncate
        #     except TypeError:
        #         attrs['http.request.body'] = str(body)[:200] # Fallback for non-JSON serializable bodies

        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """API calls typically don't have direct token usage like LLMs.

        This method is part of the BaseInstrumentor interface but is not implemented
        for generic API calls as token usage is not a standard concept here.
        """
        # Return None as API calls don't have direct token usage in the same way LLMs do.
        # If specific APIs return usage, this method would need to be extended.
        return None