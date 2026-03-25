# Evaluation and Safety

TraceVerde includes 6 built-in evaluation detectors for monitoring LLM input/output quality and safety.

All features are opt-in and can be enabled individually.

## PII Detection

Detect personally identifiable information using Microsoft Presidio.

```python
genai_otel.instrument(
    enable_pii_detection=True,
    pii_mode="redact",       # "detect", "redact", or "block"
    pii_threshold=0.5,
    pii_gdpr_mode=True,
    pii_hipaa_mode=True,
    pii_pci_dss_mode=True,
)
```

Detects 15+ entity types: email, phone, SSN, credit cards, IP addresses, and more.

## Toxicity Detection

Detect harmful content using Perspective API (cloud) or Detoxify (local).

```python
genai_otel.instrument(
    enable_toxicity_detection=True,
    toxicity_threshold=0.7,
)
```

Categories: toxicity, severe_toxicity, identity_attack, insult, profanity, threat.

## Bias Detection

Identify demographic biases in prompts and responses.

```python
genai_otel.instrument(
    enable_bias_detection=True,
    bias_threshold=0.4,
)
```

8 bias types: gender, race, ethnicity, religion, age, disability, sexual_orientation, political.

## Prompt Injection Detection

Protect against prompt manipulation attacks.

```python
genai_otel.instrument(
    enable_prompt_injection_detection=True,
    prompt_injection_threshold=0.5,
)
```

6 injection types: instruction_override, role_playing, jailbreak, context_switching, system_extraction, encoding_obfuscation.

## Restricted Topics Detection

Monitor and block sensitive topics.

```python
genai_otel.instrument(
    enable_restricted_topics=True,
    restricted_topics=["medical_advice", "legal_advice", "financial_advice"],
    restricted_topics_threshold=0.5,
)
```

## Hallucination Detection

Track factual accuracy and groundedness.

```python
genai_otel.instrument(
    enable_hallucination_detection=True,
    hallucination_threshold=0.7,
)
```

Includes factual claim extraction, hedge word detection, citation tracking, and context contradiction detection.

## Enable All

```python
genai_otel.instrument(
    enable_pii_detection=True,
    enable_toxicity_detection=True,
    enable_bias_detection=True,
    enable_prompt_injection_detection=True,
    enable_restricted_topics=True,
    enable_hallucination_detection=True,
)
```

## Metrics

Each detector records metrics for monitoring:

- `genai.evaluation.<detector>.detections` - Detection events
- `genai.evaluation.<detector>.blocked` - Blocked requests
- `genai.evaluation.<detector>.score` - Score distribution (histogram)

See the README for the complete list of span attributes added by each detector.
