"""
Example demonstrating OpenTelemetry instrumentation for DSPy framework.

This example shows:
1. Basic prediction with automatic instrumentation
2. Chain-of-Thought reasoning
3. ReAct (Reasoning + Acting) with tools
4. RAG with retrieval and generation
5. Optimizer/Teleprompter usage

DSPy is a Stanford NLP framework for programming language models declaratively
using modular components that can be optimized automatically.

Requirements:
    pip install genai-otel dspy-ai

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
"""

import os

# Step 1: Set up OpenTelemetry instrumentation
# This should be done BEFORE importing DSPy
from genai_otel import instrument

instrument(
    service_name="dspy-example",
    endpoint="http://localhost:4318",  # OTLP endpoint
)

# Step 2: Import DSPy after instrumentation is set up
import dspy
from dspy import Example
from dspy.teleprompt import BootstrapFewShot


def example_basic_predict():
    """
    Example 1: Basic Prediction

    This demonstrates a simple DSPy prediction with automatic instrumentation.

    Telemetry Captured:
    - Span: dspy.module.predict
    - Span: dspy.predict
    - Attributes:
        - gen_ai.system: "dspy"
        - gen_ai.operation.name: "predict"
        - dspy.predict.signature: Signature name
        - dspy.predict.input_fields: Input field names
        - dspy.predict.output_fields: Output field names
        - dspy.predict.input.*: Input values
        - dspy.predict.output.*: Output values
    """
    print("=" * 80)
    print("Example 1: Basic Prediction")
    print("=" * 80)

    # Configure DSPy with OpenAI
    lm = dspy.LM("openai/gpt-3.5-turbo", max_tokens=200)
    dspy.configure(lm=lm)

    # Define a simple signature
    class QuestionAnswer(dspy.Signature):
        """Answer questions with short factual answers."""

        question: str = dspy.InputField()
        answer: str = dspy.OutputField(desc="often between 1 and 5 words")

    # Create predictor - This will be automatically instrumented
    predictor = dspy.Predict(QuestionAnswer)

    # Make prediction
    result = predictor(question="What is the capital of France?")

    print(f"\nQuestion: What is the capital of France?")
    print(f"Answer: {result.answer}")

    print("\n✓ Prediction completed. Check your telemetry backend for traces!")
    print("  - Trace shows prediction execution")
    print("  - Signature fields captured")
    print("  - Input and output values tracked")
    print("  - Token usage from OpenAI calls aggregated\n")


def example_chain_of_thought():
    """
    Example 2: Chain-of-Thought Reasoning

    This demonstrates DSPy's ChainOfThought module for step-by-step reasoning.

    Telemetry Captured:
    - Span: dspy.module.chainofthought
    - Span: dspy.chain_of_thought
    - Attributes:
        - gen_ai.operation.name: "chain_of_thought"
        - dspy.cot.signature: Signature name
        - dspy.cot.output_fields: Output fields including rationale
        - dspy.cot.reasoning: The reasoning steps
        - dspy.cot.output.answer: Final answer
    """
    print("=" * 80)
    print("Example 2: Chain-of-Thought Reasoning")
    print("=" * 80)

    # Configure DSPy
    lm = dspy.LM("openai/gpt-3.5-turbo", max_tokens=300)
    dspy.configure(lm=lm)

    # Define signature for reasoning
    class ReasoningQA(dspy.Signature):
        """Answer questions with detailed reasoning."""

        question: str = dspy.InputField()
        answer: str = dspy.OutputField(desc="a concise answer")

    # Create ChainOfThought predictor - This will be automatically instrumented
    cot = dspy.ChainOfThought(ReasoningQA)

    # Make prediction with reasoning
    result = cot(
        question="If a train travels 120 miles in 2 hours, what is its average speed in mph?"
    )

    print(f"\nQuestion: If a train travels 120 miles in 2 hours, what is its average speed?")
    print(f"Reasoning: {result.rationale}")
    print(f"Answer: {result.answer}")

    print("\n✓ Chain-of-thought completed. Check your telemetry backend!")
    print("  - Trace shows reasoning process")
    print("  - Rationale/thinking captured")
    print("  - Step-by-step logic visible\n")


def example_react_with_tools():
    """
    Example 3: ReAct (Reasoning + Acting)

    This demonstrates DSPy's ReAct module for reasoning with tool usage.

    Telemetry Captured:
    - Span: dspy.module.react
    - Span: dspy.react
    - Attributes:
        - gen_ai.operation.name: "react"
        - dspy.react.signature: Signature name
        - dspy.react.tools: List of tool names
        - dspy.react.tools_count: Number of tools
        - dspy.react.has_trajectory: Action/observation trace
    """
    print("=" * 80)
    print("Example 3: ReAct (Reasoning + Acting)")
    print("=" * 80)

    # Configure DSPy
    lm = dspy.LM("openai/gpt-3.5-turbo", max_tokens=400)
    dspy.configure(lm=lm)

    # Define simple tool functions
    def search(query: str) -> str:
        """Search for information."""
        # Mock search function
        if "capital" in query.lower():
            return "Paris is the capital of France. It has a population of 2.2 million."
        return "No results found."

    def calculate(expression: str) -> str:
        """Calculate mathematical expressions."""
        try:
            result = eval(expression)  # nosec B307 - Example code only
            return f"Result: {result}"
        except:
            return "Invalid expression"

    # Define signature
    class AgentTask(dspy.Signature):
        """Solve tasks using available tools."""

        task: str = dspy.InputField()
        answer: str = dspy.OutputField(desc="final answer to the task")

    # Create ReAct agent - This will be automatically instrumented
    react_agent = dspy.ReAct(AgentTask, tools=[search, calculate])

    # Execute task
    result = react_agent(task="Find the population of France's capital and multiply by 2")

    print(f"\nTask: Find the population of France's capital and multiply by 2")
    print(f"Answer: {result.answer}")

    print("\n✓ ReAct agent completed. Check your telemetry backend!")
    print("  - Trace shows reasoning and action steps")
    print("  - Tool usage tracked")
    print("  - Action-observation pairs captured\n")


def example_rag_pipeline():
    """
    Example 4: RAG (Retrieval-Augmented Generation)

    This demonstrates a DSPy RAG pipeline with retrieval and generation.

    Telemetry Captured:
    - Multiple dspy.module spans for pipeline components
    - Nested spans showing retrieval → generation flow
    - Input queries and generated outputs
    """
    print("=" * 80)
    print("Example 4: RAG Pipeline")
    print("=" * 80)

    # Configure DSPy
    lm = dspy.LM("openai/gpt-3.5-turbo", max_tokens=300)
    dspy.configure(lm=lm)

    # Define RAG signature
    class GenerateAnswer(dspy.Signature):
        """Answer questions based on retrieved context."""

        context: str = dspy.InputField(desc="relevant facts and information")
        question: str = dspy.InputField()
        answer: str = dspy.OutputField(desc="concise answer based on context")

    # Create RAG module
    class RAG(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

        def forward(self, question):
            # Mock retrieval (in practice, use dspy.Retrieve with a retriever)
            context = """
            DSPy is a framework for programming language models.
            It was developed at Stanford NLP.
            DSPy uses signatures and modules for declarative programming.
            DSPy supports automatic optimization with teleprompters.
            """

            # Generate answer with context
            result = self.generate_answer(context=context, question=question)
            return result

    # Create and use RAG - This will be automatically instrumented
    rag = RAG()
    result = rag(question="Who developed DSPy?")

    print(f"\nQuestion: Who developed DSPy?")
    print(f"Answer: {result.answer}")

    print("\n✓ RAG pipeline completed. Check your telemetry backend!")
    print("  - Trace shows complete RAG workflow")
    print("  - Retrieval and generation tracked")
    print("  - Context and query visible\n")


def example_optimizer():
    """
    Example 5: Optimizer/Teleprompter

    This demonstrates DSPy's BootstrapFewShot optimizer for automatic
    prompt optimization.

    Telemetry Captured:
    - Span: dspy.optimizer.bootstrapfewshot
    - Attributes:
        - gen_ai.operation.name: "optimizer.compile"
        - dspy.optimizer.name: Optimizer name
        - dspy.optimizer.trainset_size: Training set size
        - dspy.optimizer.has_metric: Whether metric is provided
        - dspy.optimizer.demos_count: Number of demonstrations
    """
    print("=" * 80)
    print("Example 5: Optimizer/Teleprompter")
    print("=" * 80)

    # Configure DSPy
    lm = dspy.LM("openai/gpt-3.5-turbo", max_tokens=200)
    dspy.configure(lm=lm)

    # Define signature
    class Emotion(dspy.Signature):
        """Classify the emotion of a sentence."""

        sentence: str = dspy.InputField()
        emotion: str = dspy.OutputField(desc="the predominant emotion")

    # Create training examples
    trainset = [
        Example(sentence="I love this so much!", emotion="joy").with_inputs("sentence"),
        Example(sentence="This is terrible and disappointing.", emotion="sadness").with_inputs(
            "sentence"
        ),
        Example(sentence="I'm feeling angry about this.", emotion="anger").with_inputs("sentence"),
        Example(sentence="This is absolutely amazing!", emotion="joy").with_inputs("sentence"),
    ]

    # Define metric
    def emotion_metric(example, prediction, trace=None):
        return example.emotion.lower() == prediction.emotion.lower()

    # Create predictor
    predictor = dspy.Predict(Emotion)

    # Create optimizer - This will be automatically instrumented
    optimizer = BootstrapFewShot(metric=emotion_metric, max_bootstrapped_demos=2)

    print(f"\nOptimizing predictor with {len(trainset)} examples...")

    # Compile/optimize - This creates optimized prompts
    optimized_predictor = optimizer.compile(predictor, trainset=trainset)

    # Test optimized predictor
    result = optimized_predictor(sentence="I'm thrilled about this opportunity!")

    print(f"\nSentence: I'm thrilled about this opportunity!")
    print(f"Emotion: {result.emotion}")

    print("\n✓ Optimizer completed. Check your telemetry backend!")
    print("  - Trace shows optimization process")
    print("  - Training set size captured")
    print("  - Metric usage tracked")
    print("  - Demonstrations count visible\n")


def example_custom_module():
    """
    Example 6: Custom DSPy Module

    This demonstrates creating and using a custom DSPy module.

    Telemetry Captured:
    - Span: dspy.module.multihopqa (custom module name)
    - Nested spans for internal predictions
    - Module composition visible in trace hierarchy
    """
    print("=" * 80)
    print("Example 6: Custom DSPy Module")
    print("=" * 80)

    # Configure DSPy
    lm = dspy.LM("openai/gpt-3.5-turbo", max_tokens=250)
    dspy.configure(lm=lm)

    # Define custom multi-hop QA module
    class MultiHopQA(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate_query = dspy.ChainOfThought("question -> search_query")
            self.generate_answer = dspy.ChainOfThought("context, question -> answer")

        def forward(self, question):
            # Generate search query
            query_result = self.generate_query(question=question)

            # Mock retrieval
            context = f"Retrieved context based on: {query_result.search_query}"

            # Generate final answer
            answer_result = self.generate_answer(context=context, question=question)

            return answer_result

    # Create and use custom module - This will be automatically instrumented
    qa = MultiHopQA()
    result = qa(question="What programming paradigm does DSPy use?")

    print(f"\nQuestion: What programming paradigm does DSPy use?")
    print(f"Answer: {result.answer}")

    print("\n✓ Custom module completed. Check your telemetry backend!")
    print("  - Trace shows custom module execution")
    print("  - Nested module calls visible")
    print("  - Module composition tracked\n")


def main():
    """
    Run all DSPy instrumentation examples.
    """
    print("\n" + "=" * 80)
    print("DSPy OpenTelemetry Instrumentation Examples")
    print("=" * 80)
    print("\nThese examples demonstrate automatic tracing of DSPy programs.")
    print("Make sure you have:")
    print("  1. OTEL_EXPORTER_OTLP_ENDPOINT configured (default: http://localhost:4318)")
    print("  2. OPENAI_API_KEY set in environment")
    print("  3. An OTLP-compatible backend running (Jaeger, Grafana, etc.)\n")

    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠ Warning: OPENAI_API_KEY not set. Examples may fail.")
        print("  Set it with: export OPENAI_API_KEY=your_key_here\n")

    try:
        # Run examples
        example_basic_predict()
        example_chain_of_thought()
        example_react_with_tools()
        example_rag_pipeline()
        example_optimizer()
        example_custom_module()

        print("=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        print("\nWhat to look for in your telemetry:")
        print("  📊 Spans:")
        print("    - dspy.module.*: Module executions (custom modules named)")
        print("    - dspy.predict: Basic predictions")
        print("    - dspy.chain_of_thought: Reasoning with rationale")
        print("    - dspy.react: Reasoning + Acting with tools")
        print("    - dspy.optimizer.*: Optimization/compilation")
        print("    - openai.chat.completions: Underlying LLM calls")
        print("\n  🏷️  Attributes:")
        print("    - gen_ai.system: 'dspy'")
        print("    - dspy.module.name: Module class name")
        print("    - dspy.module.signature: Signature used")
        print("    - dspy.predict.input_fields: Input field names")
        print("    - dspy.predict.output_fields: Output field names")
        print("    - dspy.predict.input.*: Input values")
        print("    - dspy.predict.output.*: Output values")
        print("    - dspy.cot.reasoning: Chain-of-thought rationale")
        print("    - dspy.react.tools: Available tools")
        print("    - dspy.optimizer.name: Optimizer type")
        print("    - dspy.optimizer.trainset_size: Training examples")
        print("\n  💰 Cost Tracking:")
        print("    - Token usage from OpenAI calls aggregated")
        print("    - Cost calculated automatically per prediction")
        print("    - Per-module costs visible")
        print("\n  🔍 Trace Visualization:")
        print("    - Module composition visible (nested modules)")
        print("    - RAG workflow fully traced (retrieve → generate)")
        print("    - Chain-of-thought reasoning steps clear")
        print("    - ReAct action-observation pairs tracked")
        print("    - Optimizer training process visible")
        print("    - Custom module hierarchies shown\n")

        print("📚 Key Features of DSPy:")
        print("  - Declarative programming of language models")
        print("  - Modular architecture with composable components")
        print("  - Automatic prompt optimization with teleprompters")
        print("  - Built-in reasoning modules (ChainOfThought, ReAct)")
        print("  - Signature-based type safety")
        print("  - Support for multiple LM providers\n")

    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Make sure you have installed: pip install dspy-ai")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Check your configuration and try again.")


if __name__ == "__main__":
    main()
