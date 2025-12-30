"""
Example: Citation and Source Verification

This example demonstrates detection of hallucinations through citation analysis.
Responses making factual claims should include sources and citations.

Citation Indicators:
- Number of citations provided
- Factual claims without citations
- Vague references ("studies show", "experts say")
- Hedge words indicating uncertainty
- Claims that should have sources

Requirements:
    pip install genai-otel-instrument openai

Environment Setup:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OPENAI_API_KEY=your_api_key_here
"""

import os

from genai_otel import instrument

instrument(
    service_name="hallucination-citation-verification",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
    enable_hallucination_detection=True,
)

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"))

print("=" * 80)
print("Citation and Source Verification")
print("=" * 80)
print("\nAnalyzing citation quality in responses...")
print()

# Example 1: Request with citation requirement
print("Example 1: Explicit Citation Request")
print("-" * 80)
prompt_with_citation = "What are the health benefits of exercise? Please cite sources."
print(f"Prompt: '{prompt_with_citation}'")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt_with_citation}],
    )
    response_text = response.choices[0].message.content
    print(f"Response: {response_text}")
    print(f"\nCitation Analysis:")
    print(f"  - Citations detected: {response_text.count('[') + response_text.count('(source:')}")
    print(
        f"  - Hedge words: {sum(1 for word in ['might', 'could', 'possibly', 'may'] if word in response_text.lower())}"
    )
except Exception as e:
    print(f"Note: {e}")

print()

# Example 2: Statistical claim (should have source)
print("Example 2: Statistical Claim")
print("-" * 80)
stat_prompt = "What percentage of businesses use cloud computing in 2024?"
print(f"Prompt: '{stat_prompt}'")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": stat_prompt}],
    )
    print(f"Response: {response.choices[0].message.content}")
    print("\nAnalysis: Specific statistics should cite sources")
except Exception as e:
    print(f"Note: {e}")

print()

# Example 3: Scientific claim
print("Example 3: Scientific Claim")
print("-" * 80)
science_prompt = "How does caffeine affect the brain?"
print(f"Prompt: '{science_prompt}'")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": science_prompt}],
    )
    print(f"Response: {response.choices[0].message.content}")
    print("\nAnalysis: Scientific claims should reference studies/research")
except Exception as e:
    print(f"Note: {e}")

print()

# Example 4: Opinion/general knowledge (lower citation need)
print("Example 4: General Knowledge")
print("-" * 80)
general_prompt = "Why is teamwork important in the workplace?"
print(f"Prompt: '{general_prompt}'")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": general_prompt}],
    )
    print(f"Response: {response.choices[0].message.content}")
    print("\nAnalysis: General knowledge/opinions need fewer citations")
except Exception as e:
    print(f"Note: {e}")

print()
print("=" * 80)
print("Citation Quality Assessment:")
print("=" * 80)
print(
    """
Citation Types (from best to worst):

1. Specific Citations (BEST):
   - "According to Smith et al. (2023) in Nature..."
   - "[1] Johnson, M. (2024). Title. Journal, 15(2), 123-145."
   - "Source: CDC Guidelines 2024"

2. General Attribution:
   - "Studies have shown..."
   - "Research indicates..."
   - "Experts suggest..."

3. Vague References (WEAK):
   - "It is believed that..."
   - "Some say that..."
   - "There is evidence..."

4. No Citation (RED FLAG for factual claims):
   - "75% of companies..." (where did this number come from?)
   - "The study found..." (which study?)

Prompt Engineering for Better Citations:

```python
citation_prompt = \"\"\"
Please answer the following question and cite your sources.
Format citations as [Source: Description] or use academic notation.
If you don't have a specific source, clearly indicate this with phrases
like "based on general knowledge" or "commonly understood that".

Question: {question}
\"\"\"
```

Automated Citation Verification:

```python
def verify_citations(response_text):
    # Extract factual claims
    claims = extract_factual_claims(response_text)

    # Check for citation markers
    has_citations = bool(re.search(r'\\[.*?\\]|\\(.*?\\)|Source:', response_text))

    # Count hedge words (uncertainty indicators)
    hedge_words = ['might', 'could', 'possibly', 'perhaps', 'may']
    hedge_count = sum(1 for word in hedge_words if word in response_text.lower())

    # Assess citation quality
    if len(claims) > 0 and not has_citations:
        return {
            'quality': 'poor',
            'issue': 'factual_claims_without_citations',
            'claim_count': len(claims),
            'citation_count': 0
        }
    elif hedge_count > len(claims):
        return {
            'quality': 'uncertain',
            'issue': 'excessive_hedging',
            'hedge_ratio': hedge_count / max(len(claims), 1)
        }
    else:
        return {
            'quality': 'good',
            'citation_count': response_text.count('[') + response_text.count('Source:')
        }
```

Telemetry Attributes:
  - evaluation.hallucination.response.citations = <count>
  - evaluation.hallucination.response.claims = <count>
  - evaluation.hallucination.response.hedge_words = <count>
  - evaluation.hallucination.response.citation_quality = "good"/"poor"/"uncertain"
  - evaluation.hallucination.response.indicators = ["no_citations", "vague_references"]

Use Cases:

1. Research Chatbots:
   - Require citations for all factual claims
   - Block responses without proper sources
   - Provide links to cited materials

2. Customer Support:
   - Cite company policies and documentation
   - Reference specific KB articles
   - Track which sources are most cited

3. Educational Applications:
   - Teach citation practices
   - Verify academic integrity
   - Link to authoritative sources
"""
)
print("=" * 80)
