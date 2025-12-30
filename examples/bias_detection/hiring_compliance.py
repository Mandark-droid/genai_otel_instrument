"""
Example: Hiring Compliance - EEO Bias Detection

This example demonstrates bias detection for HR and hiring applications
to ensure compliance with Equal Employment Opportunity (EEO) regulations.
Detects bias related to protected characteristics in job descriptions,
interview questions, and candidate evaluations.

Protected Classes (US EEO):
- Race/Color
- National Origin/Ethnicity
- Religion
- Sex/Gender
- Age (40+)
- Disability
- Genetic Information

Requirements:
    pip install genai-otel-instrument openai

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
"""

import os

from genai_otel import instrument

# Configure bias detection for hiring compliance
instrument(
    service_name="hiring-compliance-bias-detection",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_bias_detection=True,
    bias_threshold=0.4,  # Moderate threshold for compliance
)

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("Hiring Compliance - EEO Bias Detection")
print("=" * 80)
print("\nMonitoring hiring-related content for bias...")
print()

# Test cases for hiring scenarios
hiring_examples = [
    {
        "scenario": "Job Description - Age Bias",
        "content": "Seeking recent college graduate with youthful energy and fresh perspective.",
        "issue": "Discriminates against older workers",
    },
    {
        "scenario": "Job Description - Gender Bias",
        "content": "Looking for a strong, assertive salesman to join our team.",
        "issue": "Gendered language excludes women",
    },
    {
        "scenario": "Interview Question - Religion",
        "content": "What religious holidays do you observe? Will you need time off for them?",
        "issue": "Illegal to ask about religious practices",
    },
    {
        "scenario": "Candidate Evaluation - Disability",
        "content": "Candidate seems competent despite their disability.",
        "issue": "Implies disability is a limitation",
    },
    {
        "scenario": "Job Requirement - Age Proxy",
        "content": "Must be a digital native comfortable with modern technology.",
        "issue": "Digital native is code for young",
    },
    {
        "scenario": "COMPLIANT - Inclusive Job Description",
        "content": "Seeking qualified candidate with 5+ years software development experience.",
        "issue": "None - focus on qualifications",
    },
]

for i, example in enumerate(hiring_examples, 1):
    print(f"Example {i}: {example['scenario']}")
    print("-" * 80)
    print(f"Content: '{example['content']}'")
    print(f"Compliance Issue: {example['issue']}")

    try:
        # Use LLM to analyze or rewrite the content
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"Analyze this hiring content for bias: {example['content']}",
                }
            ],
        )
        print(f"Analysis: {response.choices[0].message.content[:150]}...")
    except Exception as e:
        print(f"Note: {e}")

    print()

print("=" * 80)
print("EEO Compliance Guidelines:")
print("=" * 80)
print(
    """
AVOID in Job Descriptions:
- Age: "recent grad", "digital native", "young", "energetic"
- Gender: "he/she", "salesman", "waitress", "strong/assertive"
- Race/Ethnicity: "native speaker", "articulate", "professional appearance"
- Disability: "able-bodied", "physically fit" (unless bona fide requirement)
- Religion: Any mention of religious practices or holidays

AVOID in Interviews:
- Age: "When did you graduate?", "How old are you?"
- Marital Status: "Are you married?", "Do you have children?"
- Religion: "What holidays do you observe?"
- Disability: "Do you have any disabilities?"
- National Origin: "Where are you from?", "What's your native language?"

USE Instead:
- Focus on job-related qualifications and skills
- "X years of experience in..."
- "Ability to perform [specific job function]"
- "Available to work [required hours/schedule]"
- "Authorized to work in [country]"

Telemetry Attributes:
  - evaluation.bias.prompt.detected = true/false
  - evaluation.bias.prompt.gender_score = <score>
  - evaluation.bias.prompt.age_score = <score>
  - evaluation.bias.prompt.race_score = <score>
  - evaluation.bias.prompt.religion_score = <score>
  - evaluation.bias.prompt.disability_score = <score>

Check your telemetry backend for compliance violations!
"""
)
print("=" * 80)
