"""OpenTelemetry instrumentor for DSPy framework.

This instrumentor automatically traces DSPy programs, modules, predictions,
chain-of-thought reasoning, and optimizer operations.

DSPy is a Stanford NLP framework for programming language models declaratively
using modular components that can be optimized automatically.

Requirements:
    pip install dspy-ai
"""

import logging
from typing import Any, Dict, Optional

from ..config import OTelConfig
from .base import BaseInstrumentor

logger = logging.getLogger(__name__)


class DSPyInstrumentor(BaseInstrumentor):
    """Instrumentor for DSPy framework"""

    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._dspy_available = False
        self._check_availability()

    def _check_availability(self):
        """Check if DSPy is available."""
        try:
            import dspy

            self._dspy_available = True
            logger.debug("DSPy framework detected and available for instrumentation")
        except ImportError:
            logger.debug("DSPy not installed, instrumentation will be skipped")
            self._dspy_available = False

    def instrument(self, config: OTelConfig):
        """Instrument DSPy if available.

        Args:
            config (OTelConfig): The OpenTelemetry configuration object.
        """
        if not self._dspy_available:
            logger.debug("Skipping DSPy instrumentation - library not available")
            return

        self.config = config

        try:
            import dspy
            import wrapt
            from dspy.primitives.module import BaseModule

            # Wrap Module.__call__ to trace all module executions
            wrapt.wrap_function_wrapper(
                "dspy.primitives.module",
                "BaseModule.__call__",
                self._wrap_module_call,
            )

            # Wrap Predict.forward for prediction operations
            wrapt.wrap_function_wrapper(
                "dspy.predict.predict",
                "Predict.forward",
                self._wrap_predict_forward,
            )

            # Wrap ChainOfThought.forward if available
            try:
                wrapt.wrap_function_wrapper(
                    "dspy.predict.chain_of_thought",
                    "ChainOfThought.forward",
                    self._wrap_chain_of_thought_forward,
                )
            except (ImportError, AttributeError):
                logger.debug("ChainOfThought not available for instrumentation")

            # Wrap ReAct.forward if available
            try:
                wrapt.wrap_function_wrapper(
                    "dspy.predict.react",
                    "ReAct.forward",
                    self._wrap_react_forward,
                )
            except (ImportError, AttributeError):
                logger.debug("ReAct not available for instrumentation")

            # Wrap optimizer compile methods
            self._wrap_optimizers()

            self._instrumented = True
            logger.info("DSPy instrumentation enabled")

        except Exception as e:
            logger.error("Failed to instrument DSPy: %s", e, exc_info=True)
            if config.fail_on_error:
                raise

    def _wrap_module_call(self, wrapped, instance, args, kwargs):
        """Wrap Module.__call__ to trace module execution.

        Args:
            wrapped: The original method
            instance: The Module instance
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            The result of the wrapped method
        """
        # Get module class name
        module_name = instance.__class__.__name__

        # Create span name based on module type
        if module_name == "BaseModule":
            span_name = "dspy.module.call"
        else:
            span_name = f"dspy.module.{module_name.lower()}"

        return self.create_span_wrapper(
            span_name=span_name,
            extract_attributes=lambda inst, args, kwargs: self._extract_module_attributes(
                instance, args, kwargs
            ),
            extract_response_attributes=self._extract_module_response_attributes,
        )(wrapped)(*args, **kwargs)

    def _wrap_predict_forward(self, wrapped, instance, args, kwargs):
        """Wrap Predict.forward to trace predictions.

        Args:
            wrapped: The original method
            instance: The Predict instance
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            The result of the wrapped method
        """
        return self.create_span_wrapper(
            span_name="dspy.predict",
            extract_attributes=lambda inst, args, kwargs: self._extract_predict_attributes(
                instance, kwargs
            ),
            extract_response_attributes=self._extract_predict_response_attributes,
        )(wrapped)(*args, **kwargs)

    def _wrap_chain_of_thought_forward(self, wrapped, instance, args, kwargs):
        """Wrap ChainOfThought.forward to trace chain-of-thought reasoning.

        Args:
            wrapped: The original method
            instance: The ChainOfThought instance
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            The result of the wrapped method
        """
        return self.create_span_wrapper(
            span_name="dspy.chain_of_thought",
            extract_attributes=lambda inst, args, kwargs: self._extract_cot_attributes(
                instance, kwargs
            ),
            extract_response_attributes=self._extract_cot_response_attributes,
        )(wrapped)(*args, **kwargs)

    def _wrap_react_forward(self, wrapped, instance, args, kwargs):
        """Wrap ReAct.forward to trace ReAct (reasoning + acting).

        Args:
            wrapped: The original method
            instance: The ReAct instance
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            The result of the wrapped method
        """
        return self.create_span_wrapper(
            span_name="dspy.react",
            extract_attributes=lambda inst, args, kwargs: self._extract_react_attributes(
                instance, kwargs
            ),
            extract_response_attributes=self._extract_react_response_attributes,
        )(wrapped)(*args, **kwargs)

    def _wrap_optimizers(self):
        """Wrap optimizer compile methods."""
        try:
            import wrapt

            # Wrap COPRO optimizer
            try:
                wrapt.wrap_function_wrapper(
                    "dspy.teleprompt.copro_optimizer",
                    "COPRO.compile",
                    self._wrap_optimizer_compile,
                )
            except (ImportError, AttributeError):
                logger.debug("COPRO optimizer not available for instrumentation")

            # Wrap MIPROv2 optimizer if available
            try:
                wrapt.wrap_function_wrapper(
                    "dspy.teleprompt.mipro_optimizer_v2",
                    "MIPROv2.compile",
                    self._wrap_optimizer_compile,
                )
            except (ImportError, AttributeError):
                logger.debug("MIPROv2 optimizer not available for instrumentation")

            # Wrap BootstrapFewShot teleprompter
            try:
                wrapt.wrap_function_wrapper(
                    "dspy.teleprompt.bootstrap",
                    "BootstrapFewShot.compile",
                    self._wrap_optimizer_compile,
                )
            except (ImportError, AttributeError):
                logger.debug("BootstrapFewShot not available for instrumentation")

        except Exception as e:
            logger.debug("Failed to wrap optimizers: %s", e)

    def _wrap_optimizer_compile(self, wrapped, instance, args, kwargs):
        """Wrap optimizer compile method.

        Args:
            wrapped: The original method
            instance: The optimizer instance
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            The result of the wrapped method
        """
        optimizer_name = instance.__class__.__name__

        return self.create_span_wrapper(
            span_name=f"dspy.optimizer.{optimizer_name.lower()}",
            extract_attributes=lambda inst, args, kwargs: self._extract_optimizer_attributes(
                instance, args, kwargs
            ),
            extract_response_attributes=self._extract_optimizer_response_attributes,
        )(wrapped)(*args, **kwargs)

    def _extract_module_attributes(self, instance: Any, args: Any, kwargs: Any) -> Dict[str, Any]:
        """Extract attributes from Module execution.

        Args:
            instance: The Module instance
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "dspy"
        attrs["gen_ai.operation.name"] = "module.call"

        try:
            # Module information
            module_name = instance.__class__.__name__
            attrs["dspy.module.name"] = module_name

            # Check if module has a name attribute
            if hasattr(instance, "name"):
                attrs["dspy.module.instance_name"] = str(instance.name)

            # Extract input kwargs
            if kwargs:
                # Limit to first few keys
                input_keys = list(kwargs.keys())[:10]
                attrs["dspy.module.input_keys"] = input_keys
                attrs["dspy.module.input_count"] = len(kwargs)

            # Extract signature if available
            if hasattr(instance, "signature"):
                sig = instance.signature
                if hasattr(sig, "__name__"):
                    attrs["dspy.module.signature"] = sig.__name__
                elif hasattr(sig, "__class__"):
                    attrs["dspy.module.signature"] = sig.__class__.__name__

        except Exception as e:
            logger.debug("Failed to extract module attributes: %s", e)

        return attrs

    def _extract_predict_attributes(self, instance: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract attributes from Predict execution.

        Args:
            instance: The Predict instance
            kwargs: Keyword arguments

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "dspy"
        attrs["gen_ai.operation.name"] = "predict"

        try:
            # Extract signature
            if hasattr(instance, "signature"):
                sig = instance.signature
                if hasattr(sig, "__name__"):
                    attrs["dspy.predict.signature"] = sig.__name__
                if hasattr(sig, "instructions") and sig.instructions:
                    attrs["dspy.predict.instructions"] = str(sig.instructions)[:500]

                # Extract input and output fields
                if hasattr(sig, "input_fields"):
                    input_fields = [f.input_variable for f in sig.input_fields]
                    attrs["dspy.predict.input_fields"] = input_fields[:10]

                if hasattr(sig, "output_fields"):
                    output_fields = [f.output_variable for f in sig.output_fields]
                    attrs["dspy.predict.output_fields"] = output_fields[:10]

            # Extract input values
            if kwargs:
                # Get first input value for tracing
                for key, value in list(kwargs.items())[:3]:
                    if isinstance(value, str):
                        attrs[f"dspy.predict.input.{key}"] = value[:500]
                    else:
                        attrs[f"dspy.predict.input.{key}"] = str(value)[:200]

        except Exception as e:
            logger.debug("Failed to extract predict attributes: %s", e)

        return attrs

    def _extract_cot_attributes(self, instance: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract attributes from ChainOfThought execution.

        Args:
            instance: The ChainOfThought instance
            kwargs: Keyword arguments

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "dspy"
        attrs["gen_ai.operation.name"] = "chain_of_thought"

        try:
            # Extract signature
            if hasattr(instance, "signature"):
                sig = instance.signature
                if hasattr(sig, "__name__"):
                    attrs["dspy.cot.signature"] = sig.__name__

            # Extract reasoning fields
            if hasattr(instance, "extended_signature"):
                ext_sig = instance.extended_signature
                if hasattr(ext_sig, "output_fields"):
                    output_fields = [f.output_variable for f in ext_sig.output_fields]
                    attrs["dspy.cot.output_fields"] = output_fields[:10]

            # Input values
            if kwargs:
                for key, value in list(kwargs.items())[:3]:
                    if isinstance(value, str):
                        attrs[f"dspy.cot.input.{key}"] = value[:500]

        except Exception as e:
            logger.debug("Failed to extract chain_of_thought attributes: %s", e)

        return attrs

    def _extract_react_attributes(self, instance: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract attributes from ReAct execution.

        Args:
            instance: The ReAct instance
            kwargs: Keyword arguments

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "dspy"
        attrs["gen_ai.operation.name"] = "react"

        try:
            # Extract signature
            if hasattr(instance, "signature"):
                sig = instance.signature
                if hasattr(sig, "__name__"):
                    attrs["dspy.react.signature"] = sig.__name__

            # Extract tools if available
            if hasattr(instance, "tools"):
                tools = instance.tools
                if tools:
                    tool_names = [t.__name__ if hasattr(t, "__name__") else str(t) for t in tools]
                    attrs["dspy.react.tools"] = tool_names[:10]
                    attrs["dspy.react.tools_count"] = len(tools)

            # Input values
            if kwargs:
                for key, value in list(kwargs.items())[:3]:
                    if isinstance(value, str):
                        attrs[f"dspy.react.input.{key}"] = value[:500]

        except Exception as e:
            logger.debug("Failed to extract react attributes: %s", e)

        return attrs

    def _extract_optimizer_attributes(
        self, instance: Any, args: Any, kwargs: Any
    ) -> Dict[str, Any]:
        """Extract attributes from optimizer compile.

        Args:
            instance: The optimizer instance
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Dict[str, Any]: Dictionary of attributes to set on the span.
        """
        attrs = {}

        # Core attributes
        attrs["gen_ai.system"] = "dspy"
        attrs["gen_ai.operation.name"] = "optimizer.compile"

        try:
            # Optimizer information
            optimizer_name = instance.__class__.__name__
            attrs["dspy.optimizer.name"] = optimizer_name

            # Extract optimizer parameters
            if hasattr(instance, "metric"):
                attrs["dspy.optimizer.has_metric"] = True

            # Training set size
            if "trainset" in kwargs:
                trainset = kwargs["trainset"]
                if hasattr(trainset, "__len__"):
                    attrs["dspy.optimizer.trainset_size"] = len(trainset)

            # Validation set size
            if "valset" in kwargs:
                valset = kwargs["valset"]
                if hasattr(valset, "__len__"):
                    attrs["dspy.optimizer.valset_size"] = len(valset)

            # COPRO specific
            if optimizer_name == "COPRO":
                if hasattr(instance, "breadth"):
                    attrs["dspy.optimizer.copro.breadth"] = instance.breadth
                if hasattr(instance, "depth"):
                    attrs["dspy.optimizer.copro.depth"] = instance.depth

        except Exception as e:
            logger.debug("Failed to extract optimizer attributes: %s", e)

        return attrs

    def _extract_module_response_attributes(self, result: Any) -> Dict[str, Any]:
        """Extract response attributes from Module execution.

        Args:
            result: The Module execution result

        Returns:
            Dict[str, Any]: Dictionary of response attributes.
        """
        attrs = {}

        try:
            # Check if result is a Prediction object
            if hasattr(result, "__class__") and result.__class__.__name__ == "Prediction":
                # Extract output keys
                if hasattr(result, "_store"):
                    output_keys = list(result._store.keys())
                    attrs["dspy.module.output_keys"] = output_keys[:10]
                    attrs["dspy.module.output_count"] = len(output_keys)

        except Exception as e:
            logger.debug("Failed to extract module response attributes: %s", e)

        return attrs

    def _extract_predict_response_attributes(self, result: Any) -> Dict[str, Any]:
        """Extract response attributes from Predict execution.

        Args:
            result: The Predict execution result (Prediction object)

        Returns:
            Dict[str, Any]: Dictionary of response attributes.
        """
        attrs = {}

        try:
            # Extract prediction outputs
            if hasattr(result, "_store"):
                store = result._store
                for key, value in list(store.items())[:5]:
                    if isinstance(value, str):
                        attrs[f"dspy.predict.output.{key}"] = value[:500]
                    else:
                        attrs[f"dspy.predict.output.{key}"] = str(value)[:200]

        except Exception as e:
            logger.debug("Failed to extract predict response attributes: %s", e)

        return attrs

    def _extract_cot_response_attributes(self, result: Any) -> Dict[str, Any]:
        """Extract response attributes from ChainOfThought execution.

        Args:
            result: The ChainOfThought execution result

        Returns:
            Dict[str, Any]: Dictionary of response attributes.
        """
        attrs = {}

        try:
            # Extract reasoning and answer
            if hasattr(result, "_store"):
                store = result._store

                # Look for common reasoning field names
                reasoning_fields = ["rationale", "reasoning", "thought", "chain_of_thought"]
                for field in reasoning_fields:
                    if field in store:
                        attrs["dspy.cot.reasoning"] = str(store[field])[:1000]
                        break

                # Extract final answer/output
                for key, value in list(store.items())[:5]:
                    if key not in reasoning_fields:
                        if isinstance(value, str):
                            attrs[f"dspy.cot.output.{key}"] = value[:500]

        except Exception as e:
            logger.debug("Failed to extract chain_of_thought response attributes: %s", e)

        return attrs

    def _extract_react_response_attributes(self, result: Any) -> Dict[str, Any]:
        """Extract response attributes from ReAct execution.

        Args:
            result: The ReAct execution result

        Returns:
            Dict[str, Any]: Dictionary of response attributes.
        """
        attrs = {}

        try:
            # Extract actions and observations
            if hasattr(result, "_store"):
                store = result._store

                # Look for action/observation traces
                if "trajectory" in store:
                    attrs["dspy.react.has_trajectory"] = True

                # Extract final answer
                for key, value in list(store.items())[:5]:
                    if isinstance(value, str):
                        attrs[f"dspy.react.output.{key}"] = value[:500]

        except Exception as e:
            logger.debug("Failed to extract react response attributes: %s", e)

        return attrs

    def _extract_optimizer_response_attributes(self, result: Any) -> Dict[str, Any]:
        """Extract response attributes from optimizer compile.

        Args:
            result: The compiled program

        Returns:
            Dict[str, Any]: Dictionary of response attributes.
        """
        attrs = {}

        try:
            # The result is the compiled/optimized program
            if hasattr(result, "__class__"):
                attrs["dspy.optimizer.result_type"] = result.__class__.__name__

            # Check if program has demos/examples after optimization
            if hasattr(result, "demos"):
                demos = result.demos
                if demos:
                    attrs["dspy.optimizer.demos_count"] = len(demos)

        except Exception as e:
            logger.debug("Failed to extract optimizer response attributes: %s", e)

        return attrs

    def _extract_usage(self, result) -> Optional[Dict[str, int]]:
        """Extract token usage from DSPy result.

        Note: DSPy tracks usage via internal usage_tracker.
        Token usage is captured by underlying LM provider instrumentors.

        Args:
            result: The DSPy operation result.

        Returns:
            Optional[Dict[str, int]]: Dictionary with token counts or None.
        """
        # DSPy's usage tracking is internal and aggregated
        # Token usage is tracked by underlying LM instrumentors (OpenAI, Anthropic, etc.)
        return None

    def _extract_finish_reason(self, result) -> Optional[str]:
        """Extract finish reason from DSPy result.

        Args:
            result: The DSPy operation result.

        Returns:
            Optional[str]: The finish reason string or None if not available.
        """
        try:
            # Check if result has finish reason
            if hasattr(result, "_store") and "finish_reason" in result._store:
                return result._store["finish_reason"]

            # For successful predictions, assume completion
            if hasattr(result, "_store") and result._store:
                return "completed"

        except Exception as e:
            logger.debug("Failed to extract finish reason: %s", e)

        return None
