"""OpenTelemetry instrumentor for Instructor framework.

This instrumentor automatically traces structured output extraction using
Instructor's Pydantic-based response models with automatic validation and retries.

Instructor is a popular library (8K+ GitHub stars) for extracting structured data
from LLMs using Pydantic models, supporting OpenAI, Anthropic, Google, and more.

Requirements:
    pip install instructor
"""

import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class InstructorInstrumentor(BaseInstrumentor):
    """Instrumentor for Instructor framework"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._instructor_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if Instructor is available."""
        try:
            import instructor

            self._instructor_available = True
            logger.debug("Instructor framework detected and available for instrumentation")
        except ImportError:
            logger.debug("Instructor not installed, instrumentation will be skipped")
            self._instructor_available = False

    def instrument(self, config: OTelConfig):
        """Instrument Instructor if available.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._instructor_available:
            logger.debug("Skipping Instructor instrumentation - library not available")
            return

        self.config = config

        try:
            import wrapt

            # Wrap from_provider method
            wrapt.wrap_function_wrapper(
                "instructor",
                "from_provider",
                self._wrap_from_provider,
            )

            # Wrap patch method (legacy API)
            wrapt.wrap_function_wrapper(
                "instructor",
                "patch",
                self._wrap_patch,
            )

            # Wrap the actual completion create method
            # This happens after patching, so we wrap the process_response method
            try:
                wrapt.wrap_function_wrapper(
                    "instructor.client",
                    "Instructor.create_with_completion",
                    self._wrap_create_with_completion,
                )
            except (ImportError, AttributeError):
                logger.debug("create_with_completion method not available")

            # Wrap retry logic
            try:
                wrapt.wrap_function_wrapper(
                    "instructor.retry",
                    "retry_sync",
                    self._wrap_retry_sync,
                )
            except (ImportError, AttributeError):
                logger.debug("retry_sync not available for instrumentation")

            self._instrumented = True
            logger.info("Instructor instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument Instructor: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _wrap_from_provider(self, wrapped, instance, args, kwargs):
        """Wrap instructor.from_provider to trace client creation.

        Args:
            wrapped: The original method
            instance: The instance (None for module function)
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            The result of the wrapped method
        """
        return self.create_span_wrapper(
            span_name="instructor.from_provider",
            extract_attributes=lambda inst, args, kwargs: self._extract_from_provider_attributes(
                args, kwargs
            ),
        )(wrapped)(*args, **kwargs)

    def _wrap_patch(self, wrapped, instance, args, kwargs):
        """Wrap instructor.patch to trace client patching.

        Args:
            wrapped: The original method
            instance: The instance (None for module function)
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            The result of the wrapped method
        """
        return self.create_span_wrapper(
            span_name="instructor.patch",
            extract_attributes=lambda inst, args, kwargs: self._extract_patch_attributes(
                args, kwargs
            ),
        )(wrapped)(*args, **kwargs)

    def _wrap_create_with_completion(self, wrapped, instance, args, kwargs):
        """Wrap Instructor.create_with_completion to trace structured extraction.

        Args:
            wrapped: The original method
            instance: The Instructor instance
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            The result of the wrapped method
        """
        return self.create_span_wrapper(
            span_name="instructor.create_with_completion",
            extract_attributes=lambda inst, args, kwargs: self._extract_create_attributes(
                instance, kwargs
            ),
            extract_response_attributes=self._extract_create_response_attributes,
        )(wrapped)(*args, **kwargs)

    def _wrap_retry_sync(self, wrapped, instance, args, kwargs):
        """Wrap retry_sync to trace retry attempts.

        Args:
            wrapped: The original method
            instance: The instance
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            The result of the wrapped method
        """
        return self.create_span_wrapper(
            span_name="instructor.retry",
            extract_attributes=lambda inst, args, kwargs: self._extract_retry_attributes(kwargs),
        )(wrapped)(*args, **kwargs)

    def _extract_from_provider_attributes(
        self, args: Any, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract attributes from from_provider call.

        Args:
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "instructor"
        attrs["gen_ai.operation.name"] = "from_provider"

        try:
            # Extract provider string
            if args and len(args) > 0:
                provider_str = args[0]
                attrs["instructor.provider"] = str(provider_str)

                # Parse provider/model format
                if "/" in provider_str:
                    provider, model = provider_str.split("/", 1)
                    attrs["instructor.provider.name"] = provider
                    attrs["gen_ai.request.model"] = model

            # Extract mode if provided
            if "mode" in kwargs:
                attrs["instructor.mode"] = str(kwargs["mode"])

        except Exception as e:
            logger.debug("Failed to extract from_provider attributes: %s", e)

        return attrs

    def _extract_patch_attributes(self, args: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract attributes from patch call.

        Args:
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "instructor"
        attrs["gen_ai.operation.name"] = "patch"

        try:
            # Extract client type
            if args and len(args) > 0:
                client = args[0]
                client_type = type(client).__name__
                attrs["instructor.client.type"] = client_type

            # Extract mode
            if "mode" in kwargs:
                attrs["instructor.mode"] = str(kwargs["mode"])

        except Exception as e:
            logger.debug("Failed to extract patch attributes: %s", e)

        return attrs

    def _extract_create_attributes(self, instance: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract attributes from create_with_completion call.

        Args:
            instance: The Instructor instance
            kwargs: Keyword arguments

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "instructor"
        attrs["gen_ai.operation.name"] = "create_with_completion"

        try:
            # Extract response_model information
            if "response_model" in kwargs:
                response_model = kwargs["response_model"]

                # Get model name
                if hasattr(response_model, "__name__"):
                    attrs["instructor.response_model.name"] = response_model.__name__
                elif hasattr(response_model, "__class__"):
                    attrs["instructor.response_model.name"] = response_model.__class__.__name__

                # Extract field information from Pydantic model
                if hasattr(response_model, "model_fields"):
                    fields = response_model.model_fields
                    field_names = list(fields.keys())[:10]
                    attrs["instructor.response_model.fields"] = field_names
                    attrs["instructor.response_model.fields_count"] = len(fields)

                # Check if it's a streaming model (Partial)
                if hasattr(response_model, "__origin__"):
                    attrs["instructor.response_model.is_partial"] = True

            # Extract max_retries
            if "max_retries" in kwargs:
                attrs["instructor.max_retries"] = kwargs["max_retries"]

            # Extract model from messages/kwargs
            if "model" in kwargs:
                attrs["gen_ai.request.model"] = str(kwargs["model"])

            # Extract streaming flag
            if "stream" in kwargs:
                attrs["instructor.stream"] = bool(kwargs["stream"])

            # Extract validation mode
            if "validation_context" in kwargs:
                attrs["instructor.has_validation_context"] = True

        except Exception as e:
            logger.debug("Failed to extract create_with_completion attributes: %s", e)

        return attrs

    def _extract_retry_attributes(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract attributes from retry call.

        Args:
            kwargs: Keyword arguments

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "instructor"
        attrs["gen_ai.operation.name"] = "retry"

        try:
            # Extract max attempts
            if "max_retries" in kwargs:
                attrs["instructor.retry.max_attempts"] = kwargs["max_retries"]

            # Extract retry context
            if "context" in kwargs:
                attrs["instructor.retry.has_context"] = True

        except Exception as e:
            logger.debug("Failed to extract retry attributes: %s", e)

        return attrs

    def _extract_create_response_attributes(self, result: Any) -> Dict[str, Any]:
        """Extract response attributes from create_with_completion result.

        Args:
            result: The structured output result (Pydantic model instance)

        Returns:
            Dict[str, Any]: Dictionary of response attributes.
        """
        attrs = {}

        try:
            # Check if result is a Pydantic model
            if hasattr(result, "model_dump"):
                # Get result type
                attrs["instructor.response.type"] = result.__class__.__name__

                # Try to get field count
                if hasattr(result, "model_fields"):
                    attrs["instructor.response.fields_count"] = len(result.model_fields)

                # Extract some field values (limit to avoid huge spans)
                try:
                    dumped = result.model_dump()
                    if dumped:
                        # Get first few keys
                        keys = list(dumped.keys())[:5]
                        attrs["instructor.response.fields"] = keys

                        # Extract first few values (truncated)
                        for key in keys[:3]:
                            value = dumped[key]
                            if isinstance(value, (str, int, float, bool)):
                                value_str = str(value)
                                attrs[f"instructor.response.{key}"] = value_str[:200]
                except Exception:
                    pass

                # Validation successful if we got a Pydantic model
                attrs["instructor.validation.success"] = True
            else:
                # No Pydantic model means validation failed
                attrs["instructor.validation.success"] = False

        except Exception as e:
            logger.debug("Failed to extract create_with_completion response attributes: %s", e)
            attrs["instructor.validation.success"] = False

        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from Instructor result.

        Note: Instructor wraps LLM provider calls.
        Token usage is captured by underlying provider instrumentors.

        Args:
            result: The Instructor operation result.

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        # Token usage is tracked by underlying LLM provider instrumentors
        # Instructor responses don't typically expose token usage directly
        return None

    def _extract_finish_reason(self, result) -> Optional[str]:
        """Extract finish reason from Instructor result.

        Args:
            result: The Instructor operation result.

        Returns:
            Optional[str]: The finish reason string or None if not available.
        """
        try:
            # For successful Pydantic model extraction
            if hasattr(result, "model_dump"):
                return "completed"

            # Check for validation metadata
            if hasattr(result, "_raw_response"):
                raw_response = result._raw_response
                if hasattr(raw_response, "choices") and raw_response.choices:
                    choice = raw_response.choices[0]
                    if hasattr(choice, "finish_reason"):
                        return choice.finish_reason

        except Exception as e:
            logger.debug("Failed to extract finish reason: %s", e)

        return None
