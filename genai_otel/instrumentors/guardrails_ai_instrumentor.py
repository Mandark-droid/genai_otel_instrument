"""OpenTelemetry instrumentor for Guardrails AI framework.

This instrumentor automatically traces validation guards that detect, quantify,
and mitigate risks in LLM outputs using Guardrails AI's validation framework.

Guardrails AI is a popular validation library for LLMs with input/output guards,
validators, and on-fail policies (reask, fix, filter, refrain).

Requirements:
    pip install guardrails-ai
"""

import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class GuardrailsAIInstrumentor(BaseInstrumentor):
    """Instrumentor for Guardrails AI framework"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._guardrails_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if Guardrails AI is available."""
        try:
            import guardrails

            self._guardrails_available = True
            logger.debug("Guardrails AI framework detected and available for instrumentation")
        except ImportError:
            logger.debug("Guardrails AI not installed, instrumentation will be skipped")
            self._guardrails_available = False

    def instrument(self, config: OTelConfig):
        """Instrument Guardrails AI if available.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._guardrails_available:
            logger.debug("Skipping Guardrails AI instrumentation - library not available")
            return

        self.config = config

        try:
            import wrapt

            # Wrap Guard.__call__ for full LLM execution with guards
            wrapt.wrap_function_wrapper(
                "guardrails.guard",
                "Guard.__call__",
                self._wrap_guard_call,
            )

            # Wrap Guard.validate for validation-only operations
            wrapt.wrap_function_wrapper(
                "guardrails.guard",
                "Guard.validate",
                self._wrap_guard_validate,
            )

            # Wrap Guard.parse for parsing LLM outputs
            wrapt.wrap_function_wrapper(
                "guardrails.guard",
                "Guard.parse",
                self._wrap_guard_parse,
            )

            # Wrap Guard.use for adding validators
            try:
                wrapt.wrap_function_wrapper(
                    "guardrails.guard",
                    "Guard.use",
                    self._wrap_guard_use,
                )
            except (ImportError, AttributeError):
                logger.debug("Guard.use not available for instrumentation")

            self._instrumented = True
            logger.info("Guardrails AI instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument Guardrails AI: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _wrap_guard_call(self, wrapped, instance, args, kwargs):
        """Wrap Guard.__call__ to trace full LLM execution with guards.

        Args:
            wrapped: The original method
            instance: The Guard instance
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            The result of the wrapped method
        """
        return self.create_span_wrapper(
            span_name="guardrails.guard.call",
            extract_attributes=lambda inst, args, kwargs: self._extract_guard_call_attributes(
                instance, kwargs
            ),
            extract_response_attributes=self._extract_guard_call_response_attributes,
        )(wrapped)(*args, **kwargs)

    def _wrap_guard_validate(self, wrapped, instance, args, kwargs):
        """Wrap Guard.validate to trace validation operations.

        Args:
            wrapped: The original method
            instance: The Guard instance
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            The result of the wrapped method
        """
        return self.create_span_wrapper(
            span_name="guardrails.guard.validate",
            extract_attributes=lambda inst, args, kwargs: self._extract_guard_validate_attributes(
                instance, args, kwargs
            ),
            extract_response_attributes=self._extract_guard_validate_response_attributes,
        )(wrapped)(*args, **kwargs)

    def _wrap_guard_parse(self, wrapped, instance, args, kwargs):
        """Wrap Guard.parse to trace parsing operations.

        Args:
            wrapped: The original method
            instance: The Guard instance
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            The result of the wrapped method
        """
        return self.create_span_wrapper(
            span_name="guardrails.guard.parse",
            extract_attributes=lambda inst, args, kwargs: self._extract_guard_parse_attributes(
                instance, args, kwargs
            ),
            extract_response_attributes=self._extract_guard_parse_response_attributes,
        )(wrapped)(*args, **kwargs)

    def _wrap_guard_use(self, wrapped, instance, args, kwargs):
        """Wrap Guard.use to trace validator additions.

        Args:
            wrapped: The original method
            instance: The Guard instance
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            The result of the wrapped method
        """
        return self.create_span_wrapper(
            span_name="guardrails.guard.use",
            extract_attributes=lambda inst, args, kwargs: self._extract_guard_use_attributes(
                args, kwargs
            ),
        )(wrapped)(*args, **kwargs)

    def _extract_guard_call_attributes(
        self, instance: Any, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract attributes from Guard.__call__.

        Args:
            instance: The Guard instance
            kwargs: Keyword arguments

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "guardrails"
        attrs["gen_ai.operation.name"] = "guard.call"

        try:
            # Extract validators from guard
            if hasattr(instance, "_validators") and instance._validators:
                validator_names = []
                on_fail_actions = []

                for validator in instance._validators[:10]:  # Limit to first 10
                    if hasattr(validator, "__class__"):
                        validator_names.append(validator.__class__.__name__)

                    if hasattr(validator, "on_fail_descriptor"):
                        on_fail = str(validator.on_fail_descriptor)
                        if on_fail not in on_fail_actions:
                            on_fail_actions.append(on_fail)

                if validator_names:
                    attrs["guardrails.validators"] = validator_names
                    attrs["guardrails.validators_count"] = len(instance._validators)

                if on_fail_actions:
                    attrs["guardrails.on_fail_actions"] = on_fail_actions

            # Extract num_reasks
            if "num_reasks" in kwargs:
                attrs["guardrails.num_reasks"] = kwargs["num_reasks"]

            # Extract LLM API if provided
            if "llm_api" in kwargs and kwargs["llm_api"]:
                llm_api = kwargs["llm_api"]
                if hasattr(llm_api, "__name__"):
                    attrs["guardrails.llm_api"] = llm_api.__name__

            # Extract metadata
            if "metadata" in kwargs and kwargs["metadata"]:
                attrs["guardrails.has_metadata"] = True

            # Extract prompt params
            if "prompt_params" in kwargs and kwargs["prompt_params"]:
                attrs["guardrails.has_prompt_params"] = True

            # Extract full_schema_reask flag
            if "full_schema_reask" in kwargs:
                attrs["guardrails.full_schema_reask"] = bool(kwargs["full_schema_reask"])

        except Exception as e:
            logger.debug("Failed to extract guard.__call__ attributes: %s", e)

        return attrs

    def _extract_guard_validate_attributes(
        self, instance: Any, args: Any, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract attributes from Guard.validate.

        Args:
            instance: The Guard instance
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "guardrails"
        attrs["gen_ai.operation.name"] = "guard.validate"

        try:
            # Extract validators
            if hasattr(instance, "_validators") and instance._validators:
                validator_names = [v.__class__.__name__ for v in instance._validators[:10]]
                attrs["guardrails.validators"] = validator_names
                attrs["guardrails.validators_count"] = len(instance._validators)

            # Extract LLM output to validate (first positional arg)
            if args and len(args) > 0:
                llm_output = args[0]
                if isinstance(llm_output, str):
                    attrs["guardrails.llm_output_length"] = len(llm_output)
                    # Truncate output for tracing
                    attrs["guardrails.llm_output_preview"] = llm_output[:200]

        except Exception as e:
            logger.debug("Failed to extract guard.validate attributes: %s", e)

        return attrs

    def _extract_guard_parse_attributes(
        self, instance: Any, args: Any, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract attributes from Guard.parse.

        Args:
            instance: The Guard instance
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "guardrails"
        attrs["gen_ai.operation.name"] = "guard.parse"

        try:
            # Extract validators
            if hasattr(instance, "_validators") and instance._validators:
                validator_names = [v.__class__.__name__ for v in instance._validators[:10]]
                attrs["guardrails.validators"] = validator_names
                attrs["guardrails.validators_count"] = len(instance._validators)

            # Extract LLM output to parse
            if args and len(args) > 0:
                llm_output = args[0]
                if isinstance(llm_output, str):
                    attrs["guardrails.llm_output_length"] = len(llm_output)

            # Extract num_reasks
            if "num_reasks" in kwargs:
                attrs["guardrails.num_reasks"] = kwargs["num_reasks"]

            # Extract metadata
            if "metadata" in kwargs and kwargs["metadata"]:
                attrs["guardrails.has_metadata"] = True

        except Exception as e:
            logger.debug("Failed to extract guard.parse attributes: %s", e)

        return attrs

    def _extract_guard_use_attributes(self, args: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract attributes from Guard.use.

        Args:
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "guardrails"
        attrs["gen_ai.operation.name"] = "guard.use"

        try:
            # Extract validator being added
            if args and len(args) > 0:
                validator = args[0]
                if hasattr(validator, "__name__"):
                    attrs["guardrails.validator.name"] = validator.__name__
                elif hasattr(validator, "__class__"):
                    attrs["guardrails.validator.name"] = validator.__class__.__name__

            # Extract on_fail parameter
            if "on_fail" in kwargs:
                attrs["guardrails.validator.on_fail"] = str(kwargs["on_fail"])

        except Exception as e:
            logger.debug("Failed to extract guard.use attributes: %s", e)

        return attrs

    def _extract_guard_call_response_attributes(self, result: Any) -> Dict[str, Any]:
        """Extract response attributes from Guard.__call__.

        Args:
            result: The ValidationOutcome result

        Returns:
            Dict[str, Any]: Dictionary of response attributes.
        """
        attrs = {}

        try:
            # Check if result is ValidationOutcome
            if hasattr(result, "validation_passed"):
                attrs["guardrails.validation.passed"] = bool(result.validation_passed)

            # Extract validated output
            if hasattr(result, "validated_output"):
                validated_output = result.validated_output
                if validated_output is not None:
                    if isinstance(validated_output, str):
                        attrs["guardrails.validated_output_length"] = len(validated_output)
                        attrs["guardrails.validated_output_preview"] = validated_output[:200]
                    else:
                        attrs["guardrails.validated_output_type"] = type(validated_output).__name__

            # Extract reask count
            if hasattr(result, "reasks"):
                attrs["guardrails.reasks_count"] = len(result.reasks) if result.reasks else 0

            # Extract error information
            if hasattr(result, "error") and result.error:
                attrs["guardrails.has_error"] = True
                attrs["guardrails.error_message"] = str(result.error)[:200]

        except Exception as e:
            logger.debug("Failed to extract guard.__call__ response attributes: %s", e)

        return attrs

    def _extract_guard_validate_response_attributes(self, result: Any) -> Dict[str, Any]:
        """Extract response attributes from Guard.validate.

        Args:
            result: The ValidationOutcome result

        Returns:
            Dict[str, Any]: Dictionary of response attributes.
        """
        attrs = {}

        try:
            # Check validation result
            if hasattr(result, "validation_passed"):
                attrs["guardrails.validation.passed"] = bool(result.validation_passed)

            # Extract validator results
            if hasattr(result, "validator_logs") and result.validator_logs:
                passed_validators = []
                failed_validators = []

                for log in result.validator_logs[:10]:  # Limit to first 10
                    if hasattr(log, "validator_name"):
                        validator_name = log.validator_name
                        if hasattr(log, "validation_result") and log.validation_result:
                            passed_validators.append(validator_name)
                        else:
                            failed_validators.append(validator_name)

                if passed_validators:
                    attrs["guardrails.validators.passed"] = passed_validators
                if failed_validators:
                    attrs["guardrails.validators.failed"] = failed_validators

        except Exception as e:
            logger.debug("Failed to extract guard.validate response attributes: %s", e)

        return attrs

    def _extract_guard_parse_response_attributes(self, result: Any) -> Dict[str, Any]:
        """Extract response attributes from Guard.parse.

        Args:
            result: The ValidationOutcome result

        Returns:
            Dict[str, Any]: Dictionary of response attributes.
        """
        attrs = {}

        try:
            # Check validation result
            if hasattr(result, "validation_passed"):
                attrs["guardrails.validation.passed"] = bool(result.validation_passed)

            # Extract validated output
            if hasattr(result, "validated_output"):
                validated_output = result.validated_output
                if validated_output is not None:
                    if isinstance(validated_output, str):
                        attrs["guardrails.validated_output_length"] = len(validated_output)

            # Extract reask count
            if hasattr(result, "reasks"):
                attrs["guardrails.reasks_count"] = len(result.reasks) if result.reasks else 0

        except Exception as e:
            logger.debug("Failed to extract guard.parse response attributes: %s", e)

        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from Guardrails AI result.

        Note: Guardrails AI wraps LLM provider calls.
        Token usage is captured by underlying provider instrumentors.

        Args:
            result: The Guardrails AI operation result.

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        # Token usage is tracked by underlying LLM provider instrumentors
        return None

    def _extract_finish_reason(self, result) -> Optional[str]:
        """Extract finish reason from Guardrails AI result.

        Args:
            result: The Guardrails AI operation result.

        Returns:
            Optional[str]: The finish reason string or None if not available.
        """
        try:
            # Check validation outcome
            if hasattr(result, "validation_passed"):
                if result.validation_passed:
                    return "validated"
                else:
                    return "validation_failed"

            # Check for errors
            if hasattr(result, "error") and result.error:
                return "error"

        except Exception as e:
            logger.debug("Failed to extract finish reason: %s", e)

        return None
