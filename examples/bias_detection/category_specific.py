"""
Example: Category-Specific Bias Detection

This example demonstrates how to configure bias detection for specific categories
and adjust sensitivity thresholds for different bias types.

Categories:
- gender: Gender stereotypes and sexism
- race: Racial bias and stereotypes
- ethnicity: Ethnic bias and xenophobia
- religion: Religious discrimination
- age: Age bias (ageism)
- disability: Ableism and disability discrimination
- sexual_orientation: LGBTQ+ discrimination
- political: Political bias

Requirements:
    pip install genai-otel-instrument openai

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
"""

import os

from genai_otel import instrument

# Configure specific bias categories to monitor
instrument(
    service_name="bias-category-specific",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_bias_detection=True,
    bias_threshold=0.4,  # Lower threshold for better detection in examples
    # You can customize which categories to check via config
)

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("Category-Specific Bias Detection")
print("=" * 80)
print("\nMonitoring specific bias categories...")
print()

# Test different bias categories
examples = [
    {
        "category": "Gender Bias",
        "prompt": "Men are naturally better at math and science than women.",
    },
    {
        "category": "Racial Bias",
        "prompt": "All Asian people are good at math and technology.",
    },
    {
        "category": "Age Bias",
        "prompt": "Old people can't learn new technology. They're too set in their ways.",
    },
    {
        "category": "Disability Bias",
        "prompt": "Disabled people can't be as productive as normal employees.",
    },
    {
        "category": "Religious Bias",
        "prompt": "Muslims are always extremists who don't respect other religions.",
    },
]

for i, example in enumerate(examples, 1):
    print(f"Example {i}: {example['category']}")
    print("-" * 80)
    print(f"Prompt: '{example['prompt']}'")

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": example["prompt"]}],
        )
        print(f"Response: {response.choices[0].message.content[:150]}...")
    except Exception as e:
        print(f"Note: {e}")

    print()

print("=" * 80)
print("Telemetry Attributes by Category:")
print("=" * 80)
print("Gender bias:")
print("  - evaluation.bias.prompt.gender_score = <0.0-1.0>")
print("\nRacial bias:")
print("  - evaluation.bias.prompt.race_score = <0.0-1.0>")
print("\nAge bias:")
print("  - evaluation.bias.prompt.age_score = <0.0-1.0>")
print("\nDisability bias:")
print("  - evaluation.bias.prompt.disability_score = <0.0-1.0>")
print("\nReligious bias:")
print("  - evaluation.bias.prompt.religion_score = <0.0-1.0>")
print("\nCheck your telemetry backend for category-specific scores!")
print("=" * 80)
