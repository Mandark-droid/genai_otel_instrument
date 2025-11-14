"""Example demonstrating Bias Detection in GenAI applications.

This example shows how to use the bias detection feature to identify
demographic and other biases in LLM prompts and responses.

Bias Detection supports:
- 8 bias types: gender, race, ethnicity, religion, age, disability, sexual_orientation, political
- Pattern-based detection (always available)
- Optional ML-based detection with Fairlearn
- Configurable threshold and blocking mode
- Batch processing for multiple texts
- Statistics generation

Requirements:
    pip install genai-otel-instrument[evaluation]

    Optional (for ML-based detection):
    pip install fairlearn scikit-learn

Usage:
    python bias_detection_example.py
"""

import logging
import os

from genai_otel import instrument
from genai_otel.evaluation import BiasConfig, BiasDetector

# Configure logging to see bias detection details
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_bias_detection():
    """Example 1: Basic Bias Detection with default settings."""
    print("\n" + "=" * 80)
    print("Example 1: Basic Bias Detection")
    print("=" * 80)

    # Configure bias detection
    config = BiasConfig(
        enabled=True,
        threshold=0.5,  # Default threshold
        bias_types=[  # All bias types enabled by default
            "gender",
            "race",
            "ethnicity",
            "religion",
            "age",
            "disability",
            "sexual_orientation",
            "political",
        ],
    )

    detector = BiasDetector(config)

    # Test various texts
    test_cases = [
        ("Neutral text", "The team includes people of diverse backgrounds."),
        ("Gender bias", "Women are always too emotional to be good leaders."),
        ("Race bias", "All Asian people are naturally good at mathematics."),
        ("Age bias", "Old people can't learn new technology, it's too hard for them."),
        ("Religious bias", "All Muslims are extremists and dangerous."),
    ]

    for name, text in test_cases:
        result = detector.detect(text)
        print(f"\n{name}:")
        print(f"  Text: {text}")
        print(f"  Has Bias: {result.has_bias}")
        if result.has_bias:
            print(f"  Max Score: {result.max_score:.2f}")
            print(f"  Detected Biases: {', '.join(result.detected_biases)}")
            print(f"  Patterns Matched:")
            for bias_type, patterns in result.patterns_matched.items():
                print(f"    {bias_type}: {patterns[:2]}")  # Show first 2 patterns


def example_2_different_bias_types():
    """Example 2: Detecting Different Types of Bias."""
    print("\n" + "=" * 80)
    print("Example 2: Different Bias Types")
    print("=" * 80)

    config = BiasConfig(enabled=True, threshold=0.3)
    detector = BiasDetector(config)

    bias_examples = {
        "gender": "Boys shouldn't cry or show emotions like girls do.",
        "race": "Black people are always late and unreliable.",
        "ethnicity": "All foreigners should go back to their countries.",
        "religion": "Christians always try to convert everyone they meet.",
        "age": "Millennials are lazy and don't want to work hard.",
        "disability": "Disabled people can't contribute as much as normal people.",
        "sexual_orientation": "The gay agenda is corrupting our children.",
        "political": "All liberals are snowflakes who can't handle reality.",
    }

    for bias_type, text in bias_examples.items():
        result = detector.detect(text)
        print(f"\n{bias_type.upper()} Bias:")
        print(f"  Text: {text}")
        print(f"  Detected: {result.has_bias}")
        if result.has_bias:
            print(f"  Score: {result.bias_scores.get(bias_type, 0.0):.2f}")
            print(f"  Detected Types: {', '.join(result.detected_biases)}")


def example_3_threshold_configuration():
    """Example 3: Configuring Detection Threshold."""
    print("\n" + "=" * 80)
    print("Example 3: Threshold Configuration")
    print("=" * 80)

    text = "Women are always emotional and irrational."

    # Test with different thresholds
    thresholds = [0.3, 0.5, 0.7, 0.9]

    for threshold in thresholds:
        config = BiasConfig(enabled=True, threshold=threshold)
        detector = BiasDetector(config)
        result = detector.detect(text)

        print(f"\nThreshold: {threshold}")
        print(f"  Text: {text}")
        print(f"  Has Bias: {result.has_bias}")
        print(f"  Max Score: {result.max_score:.2f}")
        print(f"  Above Threshold: {result.max_score >= threshold}")


def example_4_blocking_mode():
    """Example 4: Blocking Mode for Bias Detection."""
    print("\n" + "=" * 80)
    print("Example 4: Blocking Mode")
    print("=" * 80)

    # Configure with blocking enabled
    config = BiasConfig(
        enabled=True,
        threshold=0.5,
        block_on_detection=True,  # Block biased content
    )

    detector = BiasDetector(config)

    test_cases = [
        "This is neutral content about leadership.",
        "Women are too emotional to be CEOs.",
        "The team has diverse representation.",
    ]

    for text in test_cases:
        result = detector.detect(text)
        blocked = config.block_on_detection and result.has_bias

        print(f"\nText: {text}")
        print(f"  Has Bias: {result.has_bias}")
        print(f"  Would Block: {blocked}")
        if result.has_bias:
            print(f"  Max Score: {result.max_score:.2f}")
            print(f"  Detected Biases: {', '.join(result.detected_biases)}")


def example_5_specific_bias_types():
    """Example 5: Monitoring Specific Bias Types Only."""
    print("\n" + "=" * 80)
    print("Example 5: Specific Bias Types")
    print("=" * 80)

    # Only monitor gender and age bias
    config = BiasConfig(
        enabled=True,
        threshold=0.3,
        bias_types=["gender", "age"],  # Only these two
    )

    detector = BiasDetector(config)

    test_cases = [
        ("Gender bias", "Women are always emotional and irrational."),
        ("Age bias", "Old people can't learn new things."),
        ("Race bias (not monitored)", "All Asian people are good at math."),
        ("Political bias (not monitored)", "All liberals are wrong."),
    ]

    for name, text in test_cases:
        result = detector.detect(text)
        print(f"\n{name}:")
        print(f"  Text: {text}")
        print(f"  Has Bias: {result.has_bias}")
        print(f"  Bias Scores: {result.bias_scores}")
        if result.has_bias:
            print(f"  Detected Biases: {', '.join(result.detected_biases)}")


def example_6_batch_processing():
    """Example 6: Batch Processing Multiple Texts."""
    print("\n" + "=" * 80)
    print("Example 6: Batch Processing")
    print("=" * 80)

    config = BiasConfig(enabled=True, threshold=0.3)
    detector = BiasDetector(config)

    # Process multiple texts at once
    texts = [
        "Women are always emotional.",
        "This is neutral content.",
        "Old people can't use technology.",
        "The team has diverse backgrounds.",
        "All Muslims are terrorists.",
        "People of all ages contribute valuable perspectives.",
    ]

    print(f"\nProcessing {len(texts)} texts...")
    results = detector.analyze_batch(texts)

    print("\nResults:")
    for i, (text, result) in enumerate(zip(texts, results), 1):
        print(f"\n{i}. {text}")
        print(f"   Has Bias: {result.has_bias}")
        if result.has_bias:
            print(f"   Max Score: {result.max_score:.2f}")
            print(f"   Detected Biases: {', '.join(result.detected_biases)}")


def example_7_statistics_generation():
    """Example 7: Generating Statistics from Multiple Results."""
    print("\n" + "=" * 80)
    print("Example 7: Statistics Generation")
    print("=" * 80)

    config = BiasConfig(enabled=True, threshold=0.3)
    detector = BiasDetector(config)

    # Sample dataset of LLM responses
    texts = [
        "Women are always emotional and can't lead.",
        "Men never show emotions or vulnerability.",
        "This is neutral content about leadership.",
        "Old people can't learn new technology.",
        "Young people are lazy and entitled.",
        "Effective teams value diversity.",
        "All Asian people are good at math.",
        "Diverse perspectives improve decision-making.",
    ]

    print(f"\nAnalyzing {len(texts)} texts...")
    results = detector.analyze_batch(texts)
    stats = detector.get_statistics(results)

    print("\n" + "-" * 80)
    print("Statistics:")
    print("-" * 80)
    print(f"Total Texts Analyzed: {stats['total_texts_analyzed']}")
    print(f"Biased Texts Count: {stats['biased_texts_count']}")
    print(f"Bias Rate: {stats['bias_rate']:.1%}")
    print(f"\nBias Type Distribution:")
    for bias_type, count in sorted(stats["bias_type_counts"].items()):
        print(f"  {bias_type}: {count}")
    print(f"\nMost Common Bias: {stats['most_common_bias']}")
    print(f"\nAverage Scores by Type:")
    for bias_type, score in sorted(stats["average_scores"].items()):
        if score > 0:
            print(f"  {bias_type}: {score:.2f}")


def example_8_integration_with_instrumentation():
    """Example 8: Integration with OpenTelemetry Instrumentation."""
    print("\n" + "=" * 80)
    print("Example 8: Integration with Instrumentation")
    print("=" * 80)

    # Enable bias detection via instrumentation
    instrument(
        enable_bias_detection=True,
        bias_threshold=0.5,
        bias_block_on_detection=False,
        bias_types=["gender", "race", "age", "religion"],
    )

    print("\nBias detection enabled via instrumentation!")
    print("\nConfiguration:")
    print("  - Threshold: 0.5")
    print("  - Block on Detection: False")
    print("  - Monitored Types: gender, race, age, religion")
    print("\nBias detection will now run automatically on all LLM calls.")
    print("\nExample span attributes that will be added:")
    print("  - evaluation.bias.prompt.detected: bool")
    print("  - evaluation.bias.prompt.max_score: float")
    print("  - evaluation.bias.prompt.detected_biases: list[str]")
    print("  - evaluation.bias.prompt.{bias_type}_score: float")
    print("  - evaluation.bias.prompt.{bias_type}_patterns: list[str]")


def example_9_environment_variables():
    """Example 9: Configuration via Environment Variables."""
    print("\n" + "=" * 80)
    print("Example 9: Environment Variable Configuration")
    print("=" * 80)

    print("\nBias detection can be configured via environment variables:")
    print()
    print("export GENAI_ENABLE_BIAS_DETECTION=true")
    print("export GENAI_BIAS_THRESHOLD=0.5")
    print("export GENAI_BIAS_BLOCK_ON_DETECTION=false")
    print("export GENAI_BIAS_TYPES=gender,race,age")
    print("export GENAI_BIAS_USE_FAIRLEARN=false")
    print()

    # Simulate environment variable configuration
    os.environ["GENAI_ENABLE_BIAS_DETECTION"] = "true"
    os.environ["GENAI_BIAS_THRESHOLD"] = "0.6"
    os.environ["GENAI_BIAS_BLOCK_ON_DETECTION"] = "false"

    print("Environment variables set. Now instrument your application:")
    print()
    print("from genai_otel import instrument")
    print("instrument()  # Will read from environment variables")
    print()
    print("Bias detection is now active with environment-based configuration!")

    # Clean up
    del os.environ["GENAI_ENABLE_BIAS_DETECTION"]
    del os.environ["GENAI_BIAS_THRESHOLD"]
    del os.environ["GENAI_BIAS_BLOCK_ON_DETECTION"]


def example_10_combined_detection():
    """Example 10: Combined PII, Toxicity, and Bias Detection."""
    print("\n" + "=" * 80)
    print("Example 10: Combined Detection")
    print("=" * 80)

    # Enable all three detection types
    instrument(
        # PII Detection
        enable_pii_detection=True,
        pii_mode="detect",
        # Toxicity Detection
        enable_toxicity_detection=True,
        toxicity_threshold=0.7,
        # Bias Detection
        enable_bias_detection=True,
        bias_threshold=0.5,
    )

    print("\nAll evaluation features enabled:")
    print("  ✓ PII Detection (detect mode)")
    print("  ✓ Toxicity Detection (threshold: 0.7)")
    print("  ✓ Bias Detection (threshold: 0.5)")
    print()
    print("Example prompts that would trigger each:")
    print()
    print("PII:")
    print("  'My email is test@example.com'")
    print("  → Would detect EMAIL_ADDRESS")
    print()
    print("Toxicity:")
    print("  'You're an idiot and should be fired'")
    print("  → Would detect toxicity and insult")
    print()
    print("Bias:")
    print("  'Women are too emotional to be leaders'")
    print("  → Would detect gender bias")
    print()
    print("All three combined:")
    print("  'Contact me at test@example.com. Women like you are stupid and emotional.'")
    print("  → Would detect PII (email), toxicity (insult), and bias (gender)")


def example_11_sensitive_attributes():
    """Example 11: Working with Sensitive Attributes."""
    print("\n" + "=" * 80)
    print("Example 11: Sensitive Attributes")
    print("=" * 80)

    config = BiasConfig(
        enabled=True,
        sensitive_attributes=["gender", "race", "religion", "disability"],
    )
    detector = BiasDetector(config)

    print("\nConfigured Sensitive Attributes:")
    attributes = detector.get_sensitive_attributes()
    for attr in sorted(attributes):
        print(f"  - {attr}")

    print("\nThese attributes represent protected characteristics that")
    print("should be monitored for bias in your AI application.")


def example_12_real_world_use_cases():
    """Example 12: Real-World Use Cases."""
    print("\n" + "=" * 80)
    print("Example 12: Real-World Use Cases")
    print("=" * 80)

    config = BiasConfig(enabled=True, threshold=0.5)
    detector = BiasDetector(config)

    use_cases = [
        (
            "HR Recruitment Assistant",
            "Based on the resume, this female candidate might be too emotional "
            "for the engineering role and could leave for maternity.",
        ),
        (
            "Customer Service Bot",
            "I understand you're frustrated. Let me help you resolve this issue promptly.",
        ),
        (
            "Content Moderation",
            "Young people today are lazy and don't understand hard work like older generations.",
        ),
        (
            "Educational Assistant",
            "Students from diverse backgrounds bring unique perspectives to problem-solving.",
        ),
    ]

    for use_case, text in use_cases:
        result = detector.detect(text)
        print(f"\n{use_case}:")
        print(f"  Response: {text}")
        print(f"  Has Bias: {result.has_bias}")
        if result.has_bias:
            print(f"  ⚠️  WARNING: Bias detected!")
            print(f"  Max Score: {result.max_score:.2f}")
            print(f"  Types: {', '.join(result.detected_biases)}")
            print(f"  Recommendation: Review and revise this response")
        else:
            print(f"  ✓ No bias detected - response is acceptable")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("BIAS DETECTION EXAMPLES")
    print("=" * 80)
    print("\nDemonstrating bias detection capabilities for GenAI applications")
    print()

    try:
        example_1_basic_bias_detection()
        example_2_different_bias_types()
        example_3_threshold_configuration()
        example_4_blocking_mode()
        example_5_specific_bias_types()
        example_6_batch_processing()
        example_7_statistics_generation()
        example_8_integration_with_instrumentation()
        example_9_environment_variables()
        example_10_combined_detection()
        example_11_sensitive_attributes()
        example_12_real_world_use_cases()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Install optional dependencies: pip install fairlearn scikit-learn")
        print("  2. Configure bias detection for your use case")
        print("  3. Monitor bias metrics in your observability platform")
        print("  4. Set up alerts for bias detection in production")
        print("  5. Review and improve prompts/responses with high bias scores")
        print()

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
