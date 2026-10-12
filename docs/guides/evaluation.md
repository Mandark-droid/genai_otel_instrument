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

## Where Detectors Apply

Detectors and block policies apply to sync and async clients alike, to the OpenAI Responses
API as well as Chat Completions, to all four AWS Bedrock runtime calls (`converse`,
`converse_stream`, `invoke_model`, `invoke_model_with_response_stream`, including the system
prompt), and to streamed requests, which are checked before they are sent. A blocked request raises `PolicyViolationError` before anything reaches the provider,
whether or not its trace is sampled.

Framework-level spans (LangChain, LlamaIndex and agent frameworks) are not checked
themselves. Detection happens on the provider call beneath them, so the provider's
instrumentor has to be enabled for a block policy to take effect. `input` passed to an
embeddings call is not treated as a prompt.

Not covered yet: a prompt passed positionally (LiteLLM, llama.cpp), and providers whose prompt
is not `messages`, `prompt` or the Responses `input`: Google Gemini and Vertex AI (`contents`),
Cohere chat (`message`) and Replicate (`input`). A block policy does not stop those requests.

## Where Detectors Run

A block policy is checked before the request is sent, and the call waits for the verdict. On
async clients that check runs on a worker thread, so it does not hold your event loop.

The detectors that annotate a span run after your call has returned, on a worker thread. The
span is exported when they have finished, with the end time of your call, so its duration is
the call's and not the detectors'. Two things follow from that:

- `evaluation.*` attributes are not on the span at the moment the call returns. Code that reads
  them in the same process waits first:

    ```python
    genai_otel.flush_telemetry()  # waits for pending evaluations, then flushes
    ```

- An LLM span can be exported after the span that encloses it.

To have the detectors finish before the call returns, as they did up to 1.35.0:

```python
genai_otel.instrument(enable_pii_detection=True, evaluation_mode="inline")
```

or set `GENAI_EVALUATION_MODE=inline`. Mistral, Hyperbolic, HuggingFace pipelines and Pydantic AI
always evaluate inline. A streamed response is evaluated when its stream ends, on the thread that
ends it, and only when content capture is on.

If your application installs its own tracer provider before calling `instrument()`, the
detectors run inline and a warning says so: a provider this library did not create does not
wait for spans that are still being evaluated when it is flushed or shut down.

### When Detectors Fall Behind

At most `GENAI_EVALUATION_QUEUE_SIZE` spans wait for detectors. Past that a span is exported at
once without evaluation, marked `evaluation.skipped="overloaded"`, and a warning, at most one a
minute, says how many were skipped. Tracing is never slowed or dropped because detectors are
slow. Filter on `evaluation.skipped` to see how much went unevaluated.

Detectors share the Python interpreter with your application. A process that spends its time in
Python code, as opposed to waiting on the network, leaves them less time.

### At Shutdown

At exit and on `uninstrument()`, pending evaluations get `GENAI_EVALUATION_SHUTDOWN_TIMEOUT`
seconds. Spans the detectors have not reached by then are exported without evaluation, marked
`evaluation.skipped="shutdown"`, so no span is lost. In a serverless function, call
`genai_otel.flush_telemetry()` before returning, or use inline mode.

## How Much Text Is Analysed

Detectors read the last `GENAI_EVALUATION_MAX_CHARS` characters of a prompt, which is the newest
content of a conversation, and the first of a response. The default is 1500. A span whose text
was longer says so:

| Attribute | Meaning |
|-----------|---------|
| `evaluation.truncated` | `true` when detectors read only part of the text |
| `evaluation.prompt.length` | Characters in the prompt |
| `evaluation.prompt.analyzed_length` | Characters of the prompt the detectors read |
| `evaluation.response.length` | Characters in the response |
| `evaluation.response.analyzed_length` | Characters of the response the detectors read |
| `evaluation.pii.prompt.pattern_analyzed_length` | Characters the PII patterns covered |
| `evaluation.pii.prompt.model_analyzed_length` | Characters the PII entity model read |
| `evaluation.<detector>.prompt.analyzed_length` | Characters a pattern-based detector with a block policy read: the whole prompt |

PII detection keeps pattern coverage of the whole prompt. Card numbers, e-mail addresses, phone
numbers, US social security numbers, IP addresses, the Indian identifiers and your custom
patterns are found wherever they are. Entity types that need the language model, such as names
and locations, are looked for only in the analysed part.

The local toxicity model has its own, smaller limit: it reads the last
`GENAI_TOXICITY_MAX_CHARS` characters of the prompt (default 500), recorded in
`evaluation.toxicity.prompt.analyzed_length`. With both limits at `0` it scores the whole
text in pieces and reports the highest score per category.

PII in `redact` mode is not limited: a redacted copy is only built from text that was
analysed. Under `GENAI_PROFILE=strict` both limits default to `0`.

A block policy on prompt injection, restricted topics or bias always reads the whole prompt,
whatever the limit, so padding a prompt cannot hide text from it. The span records that in
`evaluation.<detector>.prompt.analyzed_length`.

!!! warning "The limit applies to PII and toxicity block policies"
    With a block policy, text outside the analysed part is not checked for toxicity, and not for
    names or locations. Set `GENAI_EVALUATION_MAX_CHARS=0` and `GENAI_TOXICITY_MAX_CHARS=0`
    when every character must be checked. Detection time then grows with the length of the
    prompt.

With a block policy a prompt is analysed once per call: the check before the request and the
annotation after it share the result. Each detector remembers its most recent results, keyed by
a digest of the analysed text and never by the text itself (`GENAI_EVALUATION_CACHE_SIZE`).

## Toxicity Model Options

`GENAI_TOXICITY_QUANTIZE=true` runs the local Detoxify model with 8-bit weights. It is faster,
and its scores differ from the default model's by enough to change some verdicts near the
threshold, so compare the two on your own data before relying on it. It is off by default.

`GENAI_TOXICITY_TORCH_THREADS` sets how many threads torch may use. The setting is process-wide
and applies to every torch model in the process, so it is unset by default.

Enabling only the pattern-based detectors (bias, prompt injection, restricted topics,
hallucination) does not load torch or the toxicity model.

## Metrics

Each detector records metrics for monitoring:

- `genai.evaluation.<detector>.detections` - Detection events
- `genai.evaluation.<detector>.blocked` - Blocked requests
- `genai.evaluation.<detector>.score` - Score distribution (histogram)

See the [README](https://github.com/Mandark-droid/genai_otel_instrument#evaluation-features-v10) for the complete list of span attributes.

## All Examples

Browse all evaluation examples in the [examples/ directory](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples).
