# Evaluation and Safety

TraceVerde includes 6 built-in evaluation detectors for monitoring LLM input/output quality and safety. All features are opt-in and can be enabled individually.

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

**Examples:**

- [Basic detect mode](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/pii_detection/basic_detect_mode.py)
- [Redaction mode](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/pii_detection/redaction_mode.py)
- [Blocking mode](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/pii_detection/blocking_mode.py)
- [GDPR compliance](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/pii_detection/gdpr_compliance.py)
- [HIPAA compliance](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/pii_detection/hipaa_compliance.py)
- [PCI-DSS compliance](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/pii_detection/pci_dss_compliance.py)
- [Combined compliance](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/pii_detection/combined_compliance.py)
- [Custom threshold](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/pii_detection/custom_threshold.py)
- [Response detection](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/pii_detection/response_detection.py)
- [Env var config](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/pii_detection/env_var_config.py)
- [PII with Anthropic](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/anthropic/pii_detection_example.py)
- [PII with Ollama](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/ollama/pii_detection_example.py)
- [PII with HuggingFace](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/huggingface/pii_example.py)

## Toxicity Detection

Detect harmful content using Perspective API (cloud) or Detoxify (local).

```python
genai_otel.instrument(
    enable_toxicity_detection=True,
    toxicity_threshold=0.7,
)
```

Categories: toxicity, severe_toxicity, identity_attack, insult, profanity, threat.

**Examples:**

- [Basic Detoxify (local)](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/toxicity_detection/basic_detoxify.py)
- [Perspective API (cloud)](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/toxicity_detection/perspective_api.py)
- [Blocking mode](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/toxicity_detection/blocking_mode.py)
- [Category detection](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/toxicity_detection/category_detection.py)
- [Custom threshold](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/toxicity_detection/custom_threshold.py)
- [Combined with PII](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/toxicity_detection/combined_with_pii.py)
- [Response detection](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/toxicity_detection/response_detection.py)
- [Toxicity with Anthropic](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/anthropic/toxicity_detection_example.py)
- [Toxicity with Ollama](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/ollama/toxicity_detection_example.py)
- [Toxicity with HuggingFace](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/huggingface/toxicity_example.py)

## Bias Detection

Identify demographic biases in prompts and responses.

```python
genai_otel.instrument(
    enable_bias_detection=True,
    bias_threshold=0.4,
)
```

8 bias types: gender, race, ethnicity, religion, age, disability, sexual_orientation, political.

**Examples:**

- [Basic detection](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/bias_detection/basic_detection.py)
- [Blocking mode](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/bias_detection/blocking_mode.py)
- [Category-specific](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/bias_detection/category_specific.py)
- [Custom threshold](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/bias_detection/custom_threshold.py)
- [Hiring compliance](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/bias_detection/hiring_compliance.py)
- [Response detection](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/bias_detection/response_detection.py)
- [Multiple evaluations](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/bias_detection/multiple_evaluations.py)
- [Bias with Ollama](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/ollama/bias_detection_example.py)
- [Bias with HuggingFace](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/huggingface/bias_example.py)

## Prompt Injection Detection

Protect against prompt manipulation attacks.

```python
genai_otel.instrument(
    enable_prompt_injection_detection=True,
    prompt_injection_threshold=0.5,
)
```

6 injection types: instruction_override, role_playing, jailbreak, context_switching, system_extraction, encoding_obfuscation.

**Examples:**

- [Basic detection](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/prompt_injection/basic_detection.py)
- [Blocking mode](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/prompt_injection/blocking_mode.py)
- [System override](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/prompt_injection/system_override.py)
- [Jailbreak techniques](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/prompt_injection/jailbreak_techniques.py)
- [Payload injection](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/prompt_injection/payload_injection.py)
- [Prompt injection with HuggingFace](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/huggingface/prompt_injection_example.py)

## Restricted Topics Detection

Monitor and block sensitive topics.

```python
genai_otel.instrument(
    enable_restricted_topics=True,
    restricted_topics=["medical_advice", "legal_advice", "financial_advice"],
    restricted_topics_threshold=0.5,
)
```

**Examples:**

- [Basic detection](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/restricted_topics/basic_detection.py)
- [Blocking mode](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/restricted_topics/blocking_mode.py)
- [Content policy](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/restricted_topics/content_policy.py)
- [Industry compliance](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/restricted_topics/industry_compliance.py)

## Hallucination Detection

Track factual accuracy and groundedness.

```python
genai_otel.instrument(
    enable_hallucination_detection=True,
    hallucination_threshold=0.7,
)
```

Includes factual claim extraction, hedge word detection, citation tracking, and context contradiction detection.

**Examples:**

- [Basic detection](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/hallucination/basic_detection.py)
- [Citation verification](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/hallucination/citation_verification.py)
- [RAG faithfulness](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/hallucination/rag_faithfulness.py)
- [Hallucination with Ollama](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/ollama/hallucination_detection_example.py)
- [Hallucination with Mistral](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/mistralai/hallucination_detection_example.py)
- [Hallucination with HuggingFace](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/huggingface/hallucination_example.py)

## Enable All Evaluations

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

See the [comprehensive evaluation example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/comprehensive_evaluation_example.py) and [Ollama multiple evaluations](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/ollama/multiple_evaluations_detection_example.py).

## Metrics

Each detector records metrics for monitoring:

- `genai.evaluation.<detector>.detections` - Detection events
- `genai.evaluation.<detector>.blocked` - Blocked requests
- `genai.evaluation.<detector>.score` - Score distribution (histogram)

See the [README](https://github.com/Mandark-droid/genai_otel_instrument#evaluation-features-v10) for the complete list of span attributes.

## All Examples

Browse all evaluation examples in the [examples/ directory](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples).
