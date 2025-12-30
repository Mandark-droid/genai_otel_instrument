"""
Example: Industry-Specific Compliance Detection

This example demonstrates how to configure restricted topics for
industry-specific compliance requirements (healthcare, finance, education).

Industry Restrictions:
- Healthcare (HIPAA): Medical advice, diagnoses, treatment recommendations
- Finance (SEC/FINRA): Investment advice, stock tips, financial planning
- Education (FERPA): Student records, grades, personal information
- Legal: Legal advice, case strategy, attorney-client privileged info

Requirements:
    pip install genai-otel-instrument openai

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
"""

import os

from genai_otel import instrument

instrument(
    service_name="restricted-topics-industry-compliance",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_restricted_topics_detection=True,
)

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("Industry-Specific Compliance Detection")
print("=" * 80)
print("\nMonitoring industry-restricted topics...")
print()

compliance_examples = [
    {
        "industry": "Healthcare (HIPAA)",
        "prompt": "I have these symptoms: fever, cough. What disease do I have?",
        "risk": "Providing medical diagnosis without license",
    },
    {
        "industry": "Finance (SEC/FINRA)",
        "prompt": "Should I invest my retirement savings in this penny stock?",
        "risk": "Unlicensed financial advice, fiduciary violation",
    },
    {
        "industry": "Legal",
        "prompt": "How should I respond to this lawsuit filed against me?",
        "risk": "Practicing law without license",
    },
    {
        "industry": "Pharmaceutical",
        "prompt": "What dosage of this prescription medication should I take?",
        "risk": "Unauthorized prescription guidance",
    },
    {
        "industry": "COMPLIANT - General Info",
        "prompt": "What are common symptoms of the flu?",
        "risk": "None - general health education is acceptable",
    },
]

for i, example in enumerate(compliance_examples, 1):
    print(f"Example {i}: {example['industry']}")
    print("-" * 80)
    print(f"Prompt: '{example['prompt']}'")
    print(f"Compliance Risk: {example['risk']}")

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
print("Industry Compliance Guidelines:")
print("=" * 80)
print("""
Healthcare (HIPAA):
  AVOID: Diagnoses, treatment plans, medication dosages
  ALLOW: General health information, symptom descriptions
  DISCLAIMER: "This is not medical advice. Consult a healthcare provider."

Finance (SEC/FINRA):
  AVOID: "Buy/sell" recommendations, specific investment advice
  ALLOW: General financial education, definition of terms
  DISCLAIMER: "Not financial advice. Consult a licensed advisor."

Legal:
  AVOID: Case strategy, legal opinions, contract review
  ALLOW: General legal information, explanation of laws
  DISCLAIMER: "This is not legal advice. Consult an attorney."

Education (FERPA):
  AVOID: Discussing student records, grades, personal info
  ALLOW: General educational content, study techniques
  SAFEGUARD: Never store or process student PII

Telemetry Configuration:
  - evaluation.restricted_topics.prompt.detected = true/false
  - evaluation.restricted_topics.prompt.topics = ["medical", "financial", "legal"]
  - evaluation.restricted_topics.prompt.medical_score = <score>
  - evaluation.restricted_topics.prompt.financial_score = <score>
  - evaluation.restricted_topics.prompt.legal_score = <score>

Response Templates:
  Medical: "I can provide general health information, but for specific
            medical advice, please consult a qualified healthcare provider."

  Financial: "I can explain financial concepts, but I'm not licensed to
              provide investment advice. Please consult a financial advisor."

  Legal: "I can explain legal concepts generally, but cannot provide legal
          advice for your situation. Please consult an attorney."
""")
print("=" * 80)
