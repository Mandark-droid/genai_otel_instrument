"""
Example demonstrating OpenTelemetry instrumentation for Instructor framework.

This example shows:
1. Basic structured output extraction with Pydantic models
2. Nested complex data structures
3. Validation and automatic retries
4. Streaming partial results
5. Multiple provider support

Instructor is a popular library (8K+ GitHub stars) for extracting structured data
from LLMs using Pydantic models with automatic validation and retries.

Requirements:
    pip install genai-otel instructor openai

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
"""

import os
from typing import List, Optional

# Step 1: Set up OpenTelemetry instrumentation
# This should be done BEFORE importing Instructor
from genai_otel import instrument

instrument(
    service_name="instructor-example",
    endpoint="http://localhost:4318",  # OTLP endpoint
)

# Step 2: Import Instructor and Pydantic after instrumentation is set up
import instructor
from pydantic import BaseModel, Field


def example_basic_extraction():
    """
    Example 1: Basic Structured Output Extraction

    This demonstrates extracting a simple Pydantic model from an LLM.

    Telemetry Captured:
    - Span: instructor.from_provider
    - Span: instructor.create_with_completion
    - Attributes:
        - gen_ai.system: "instructor"
        - gen_ai.operation.name: "create_with_completion"
        - instructor.provider: Provider and model
        - instructor.response_model.name: Model class name
        - instructor.response_model.fields: List of fields
        - instructor.validation.success: Whether validation passed
    """
    print("=" * 80)
    print("Example 1: Basic Structured Output Extraction")
    print("=" * 80)

    # Define Pydantic model for structured output
    class UserProfile(BaseModel):
        """User profile information"""

        name: str = Field(description="The user's full name")
        age: int = Field(description="The user's age in years")
        email: str = Field(description="The user's email address")

    # Create Instructor client - This will be automatically instrumented
    client = instructor.from_provider("openai/gpt-3.5-turbo")

    # Extract structured data
    user = client.chat.completions.create(
        response_model=UserProfile,
        messages=[
            {
                "role": "user",
                "content": "Extract: Jason Liu is 25 years old and his email is jason@example.com",
            }
        ],
    )

    print(f"\nExtracted User Profile:")
    print(f"  Name: {user.name}")
    print(f"  Age: {user.age}")
    print(f"  Email: {user.email}")

    print("\n✓ Extraction completed. Check your telemetry backend for traces!")
    print("  - Trace shows structured extraction")
    print("  - Pydantic model schema captured")
    print("  - Validation success tracked")
    print("  - Token usage from OpenAI calls aggregated\n")


def example_nested_structures():
    """
    Example 2: Nested Complex Data Structures

    This demonstrates extracting deeply nested Pydantic models.

    Telemetry Captured:
    - Complex model structure with nested fields
    - Field count tracking
    - Nested validation success
    """
    print("=" * 80)
    print("Example 2: Nested Complex Data Structures")
    print("=" * 80)

    # Define nested models
    class Address(BaseModel):
        """Address information"""

        street: str
        city: str
        country: str
        zip_code: str

    class Company(BaseModel):
        """Company information"""

        name: str
        industry: str
        address: Address

    class Person(BaseModel):
        """Person with company information"""

        name: str
        role: str
        company: Company

    # Create client
    client = instructor.from_provider("openai/gpt-3.5-turbo")

    # Extract nested structure
    person = client.chat.completions.create(
        response_model=Person,
        messages=[
            {
                "role": "user",
                "content": """
                Extract: Sarah Johnson works as a Software Engineer at TechCorp,
                a technology company located at 123 Main St, San Francisco, CA 94105, USA.
                """,
            }
        ],
    )

    print(f"\nExtracted Person:")
    print(f"  Name: {person.name}")
    print(f"  Role: {person.role}")
    print(f"  Company: {person.company.name} ({person.company.industry})")
    print(f"  Location: {person.company.address.city}, {person.company.address.country}")

    print("\n✓ Nested extraction completed. Check your telemetry backend!")
    print("  - Trace shows nested model structure")
    print("  - All fields validated successfully")
    print("  - Complex relationships captured\n")


def example_validation_and_retries():
    """
    Example 3: Validation and Automatic Retries

    This demonstrates Instructor's automatic retry on validation failure.

    Telemetry Captured:
    - instructor.max_retries: Maximum retry attempts
    - instructor.retry spans: Individual retry attempts
    - Validation errors and recovery
    """
    print("=" * 80)
    print("Example 3: Validation and Automatic Retries")
    print("=" * 80)

    # Define model with validation constraints
    class Recipe(BaseModel):
        """Recipe information with constraints"""

        name: str = Field(description="Recipe name")
        prep_time_minutes: int = Field(description="Preparation time in minutes", gt=0, lt=300)
        servings: int = Field(description="Number of servings", gt=0, lt=20)
        ingredients: List[str] = Field(description="List of ingredients", min_length=2)

    # Create client
    client = instructor.from_provider("openai/gpt-3.5-turbo")

    # Extract with retry configuration
    recipe = client.chat.completions.create(
        response_model=Recipe,
        max_retries=3,  # Allow up to 3 retries on validation failure
        messages=[
            {
                "role": "user",
                "content": "Give me a recipe for chocolate chip cookies with prep time and ingredients",
            }
        ],
    )

    print(f"\nExtracted Recipe:")
    print(f"  Name: {recipe.name}")
    print(f"  Prep Time: {recipe.prep_time_minutes} minutes")
    print(f"  Servings: {recipe.servings}")
    print(f"  Ingredients: {', '.join(recipe.ingredients[:3])}...")

    print("\n✓ Extraction with retries completed. Check your telemetry backend!")
    print("  - Trace shows retry configuration")
    print("  - Validation constraints enforced")
    print("  - Automatic reask on failure\n")


def example_list_extraction():
    """
    Example 4: Extracting Lists of Objects

    This demonstrates extracting multiple structured objects.

    Telemetry Captured:
    - List extraction patterns
    - Multiple object validation
    - Batch processing tracking
    """
    print("=" * 80)
    print("Example 4: Extracting Lists of Objects")
    print("=" * 80)

    # Define model for individual items
    class Task(BaseModel):
        """Task information"""

        title: str = Field(description="Task title")
        priority: str = Field(description="Priority: high, medium, or low")
        estimated_hours: Optional[float] = Field(
            description="Estimated hours to complete", default=None
        )

    class TaskList(BaseModel):
        """List of tasks"""

        tasks: List[Task] = Field(description="List of tasks")

    # Create client
    client = instructor.from_provider("openai/gpt-3.5-turbo")

    # Extract list of tasks
    task_list = client.chat.completions.create(
        response_model=TaskList,
        messages=[
            {
                "role": "user",
                "content": """
                Extract tasks from this text:
                - Implement authentication (high priority, 8 hours)
                - Write documentation (medium priority, 4 hours)
                - Deploy to production (high priority)
                """,
            }
        ],
    )

    print(f"\nExtracted {len(task_list.tasks)} Tasks:")
    for i, task in enumerate(task_list.tasks, 1):
        hours = f" ({task.estimated_hours}h)" if task.estimated_hours else ""
        print(f"  {i}. {task.title} [{task.priority}]{hours}")

    print("\n✓ List extraction completed. Check your telemetry backend!")
    print("  - Trace shows list processing")
    print("  - Multiple objects validated")
    print("  - Batch extraction tracked\n")


def example_streaming_partial_results():
    """
    Example 5: Streaming Partial Results

    This demonstrates streaming partial Pydantic objects as they're generated.

    Telemetry Captured:
    - instructor.stream: true
    - instructor.response_model.is_partial: Partial model tracking
    - Progressive field population
    """
    print("=" * 80)
    print("Example 5: Streaming Partial Results")
    print("=" * 80)

    # Define model for streaming
    class Article(BaseModel):
        """Article information"""

        title: str = Field(description="Article title")
        summary: str = Field(description="Article summary")
        key_points: List[str] = Field(description="Key points", default_factory=list)

    # Create client
    client = instructor.from_provider("openai/gpt-3.5-turbo")

    # Stream partial results
    print("\nStreaming partial results...")

    article_stream = client.chat.completions.create(
        response_model=Article,
        stream=True,  # Enable streaming
        messages=[
            {
                "role": "user",
                "content": "Write an article about the benefits of OpenTelemetry instrumentation",
            }
        ],
    )

    # Process stream
    for partial_article in article_stream:
        # Each iteration provides a progressively more complete object
        pass

    # Final complete article
    print(f"\nFinal Article:")
    print(f"  Title: {partial_article.title if hasattr(partial_article, 'title') else 'N/A'}")
    print(
        f"  Summary: {partial_article.summary[:100] if hasattr(partial_article, 'summary') else 'N/A'}..."
    )

    print("\n✓ Streaming completed. Check your telemetry backend!")
    print("  - Trace shows streaming operation")
    print("  - Partial model generation tracked")
    print("  - Progressive updates visible\n")


def example_multiple_providers():
    """
    Example 6: Multiple Provider Support

    This demonstrates using Instructor with different LLM providers.

    Telemetry Captured:
    - instructor.provider.name: Provider identification
    - gen_ai.request.model: Model used
    - Cross-provider comparison
    """
    print("=" * 80)
    print("Example 6: Multiple Provider Support")
    print("=" * 80)

    # Define simple model
    class Sentiment(BaseModel):
        """Sentiment analysis result"""

        text: str = Field(description="The analyzed text")
        sentiment: str = Field(description="Sentiment: positive, negative, or neutral")
        confidence: float = Field(description="Confidence score 0-1", ge=0, le=1)

    # Example with OpenAI
    print("\n1. Using OpenAI...")
    client_openai = instructor.from_provider("openai/gpt-3.5-turbo")

    result_openai = client_openai.chat.completions.create(
        response_model=Sentiment,
        messages=[
            {
                "role": "user",
                "content": "Analyze sentiment: I absolutely love this new feature!",
            }
        ],
    )

    print(f"   OpenAI Result: {result_openai.sentiment} ({result_openai.confidence:.2f})")

    # Example with Anthropic (if API key is available)
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("\n2. Using Anthropic Claude...")
        client_anthropic = instructor.from_provider("anthropic/claude-3-haiku-20240307")

        result_anthropic = client_anthropic.chat.completions.create(
            response_model=Sentiment,
            messages=[
                {
                    "role": "user",
                    "content": "Analyze sentiment: I absolutely love this new feature!",
                }
            ],
        )

        print(
            f"   Anthropic Result: {result_anthropic.sentiment} ({result_anthropic.confidence:.2f})"
        )

    print("\n✓ Multi-provider extraction completed. Check your telemetry backend!")
    print("  - Trace shows provider differentiation")
    print("  - Each provider tracked separately")
    print("  - Cross-provider comparison available\n")


def main():
    """
    Run all Instructor instrumentation examples.
    """
    print("\n" + "=" * 80)
    print("Instructor OpenTelemetry Instrumentation Examples")
    print("=" * 80)
    print("\nThese examples demonstrate automatic tracing of Instructor operations.")
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
        example_basic_extraction()
        example_nested_structures()
        example_validation_and_retries()
        example_list_extraction()
        example_streaming_partial_results()
        example_multiple_providers()

        print("=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        print("\nWhat to look for in your telemetry:")
        print("  📊 Spans:")
        print("    - instructor.from_provider: Client initialization")
        print("    - instructor.patch: Legacy client patching")
        print("    - instructor.create_with_completion: Structured extraction")
        print("    - instructor.retry: Retry attempts on validation failure")
        print("    - openai.chat.completions: Underlying LLM calls")
        print("\n  🏷️  Attributes:")
        print("    - gen_ai.system: 'instructor'")
        print("    - instructor.provider: Provider and model string")
        print("    - instructor.provider.name: Provider name")
        print("    - instructor.response_model.name: Pydantic model name")
        print("    - instructor.response_model.fields: List of model fields")
        print("    - instructor.response_model.fields_count: Number of fields")
        print("    - instructor.max_retries: Maximum retry attempts")
        print("    - instructor.stream: Streaming enabled")
        print("    - instructor.validation.success: Validation result")
        print("    - instructor.response.type: Response model type")
        print("    - instructor.response.fields: Extracted field names")
        print("\n  💰 Cost Tracking:")
        print("    - Token usage from underlying LLM calls")
        print("    - Cost calculated per extraction")
        print("    - Retry costs aggregated")
        print("\n  🔍 Trace Visualization:")
        print("    - Nested model extraction visible")
        print("    - Validation and retry flows clear")
        print("    - Streaming partial updates tracked")
        print("    - Multi-provider usage distinguished")
        print("    - Field-level extraction traced\n")

        print("📚 Key Features of Instructor:")
        print("  - Pydantic-based structured output extraction")
        print("  - Automatic validation and retries")
        print("  - Support for 15+ LLM providers")
        print("  - Streaming partial results")
        print("  - Deeply nested data structures")
        print("  - Type-safe with Python type hints\n")

    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Make sure you have installed: pip install instructor openai")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Check your configuration and try again.")


if __name__ == "__main__":
    main()
