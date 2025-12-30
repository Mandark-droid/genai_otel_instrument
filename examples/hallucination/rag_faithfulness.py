"""
Example: RAG Faithfulness Detection

This example demonstrates hallucination detection in Retrieval-Augmented Generation
(RAG) scenarios, where responses should be grounded in retrieved context.

RAG Hallucination Types:
- Adding information not in the context
- Contradicting the retrieved documents
- Making up citations or sources
- Combining facts incorrectly
- Exaggerating or speculating beyond context

Requirements:
    pip install genai-otel-instrument openai

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
"""

import os

from genai_otel import instrument

instrument(
    service_name="hallucination-rag-faithfulness",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_hallucination_detection=True,
)

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("RAG Faithfulness Detection")
print("=" * 80)
print("\nDetecting hallucinations in RAG responses...")
print()

# Simulated retrieved context
context = """
Company Policy Document - Vacation Time:
Employees receive 15 days of paid vacation per year.
Vacation days accrue monthly at a rate of 1.25 days per month.
Unused vacation days can be carried over up to 5 days maximum.
Employees must request vacation at least 2 weeks in advance.
"""

# Example 1: Faithful response (grounded in context)
print("Example 1: Faithful RAG Response")
print("-" * 80)
question1 = "How many vacation days do employees get?"
print(f"Question: {question1}")
print(f"Context: {context[:100]}...")

prompt1 = f"Based on this context:\n{context}\n\nQuestion: {question1}\nAnswer based only on the context provided:"

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt1}],
    )
    print(f"\nResponse: {response.choices[0].message.content}")
    print("Analysis: Should cite '15 days' from context")
except Exception as e:
    print(f"Note: {e}")

print()

# Example 2: Question designed to elicit hallucination
print("Example 2: Question Not Answered in Context")
print("-" * 80)
question2 = "What is the sick leave policy?"
print(f"Question: {question2}")
print(f"Context: {context[:100]}...")

prompt2 = f"Based on this context:\n{context}\n\nQuestion: {question2}\nAnswer based only on the context provided:"

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt2}],
    )
    print(f"\nResponse: {response.choices[0].message.content}")
    print("Analysis: Should state 'not mentioned in context' vs. hallucinating answer")
except Exception as e:
    print(f"Note: {e}")

print()

# Example 3: Aggregation question
print("Example 3: Aggregation from Context")
print("-" * 80)
question3 = "Can I carry over unused vacation days?"
print(f"Question: {question3}")

prompt3 = f"Based on this context:\n{context}\n\nQuestion: {question3}\nAnswer based only on the context provided:"

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt3}],
    )
    print(f"\nResponse: {response.choices[0].message.content}")
    print("Analysis: Should cite 'up to 5 days maximum'")
except Exception as e:
    print(f"Note: {e}")

print()
print("=" * 80)
print("RAG Hallucination Detection Strategy:")
print("=" * 80)
print("""
Implementation Pattern:

```python
def check_rag_faithfulness(context, response):
    # 1. Extract claims from response
    claims = extract_factual_claims(response)

    # 2. Verify each claim against context
    unsupported_claims = []
    for claim in claims:
        if not is_supported_by_context(claim, context):
            unsupported_claims.append(claim)

    # 3. Calculate faithfulness score
    if not claims:
        return 1.0  # No claims to verify
    faithfulness_score = 1.0 - (len(unsupported_claims) / len(claims))

    return {
        "faithfulness_score": faithfulness_score,
        "unsupported_claims": unsupported_claims,
        "has_hallucination": len(unsupported_claims) > 0
    }
```

Best Practices:

1. Prompt Engineering:
   - Explicitly instruct to answer only from context
   - Ask model to cite sources
   - Request "I don't know" for unanswerable questions

2. Validation:
   - Check if response contains context information
   - Detect phrases like "based on the context"
   - Flag responses that go beyond provided information

3. Mitigation:
   - Use smaller context windows for better grounding
   - Implement citation requirements
   - Add post-processing to filter unsupported claims
   - Use RAG-specific models trained for faithfulness

4. Monitoring:
   - Track hallucination rate by document type
   - Monitor query complexity vs. hallucination
   - Alert on high hallucination scores in production

Telemetry Attributes:
  - evaluation.hallucination.response.detected = true/false
  - evaluation.hallucination.response.score = <score>
  - evaluation.hallucination.response.unsupported_claims = [claims]
  - evaluation.hallucination.response.citations = <count>
  - gen_ai.rag.context = <context provided>
  - gen_ai.rag.faithfulness_score = <score>

Metrics to Track:
  - Hallucination rate by document source
  - Average faithfulness score
  - Unsupported claims per response
  - User feedback on accuracy
""")
print("=" * 80)
