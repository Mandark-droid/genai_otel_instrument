# LLM Providers

TraceVerde auto-instruments 28 LLM providers. No code changes are needed - just install the provider SDK and TraceVerde handles the rest.

The table below is the complete list, and `tests/test_docs_provider_coverage.py` asserts
it stays in step with the `INSTRUMENTORS` registry, so a provider cannot be added in code
without appearing here.

## Providers with Full Cost Tracking

| Provider | Models | Install Extra | Example |
|----------|--------|---------------|---------|
| OpenAI | GPT-4o, GPT-4 Turbo, GPT-5.2, o1/o3, embeddings, Responses API (50+) | `[openai]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/openai/example.py) |
| OpenRouter | All models via OpenAI-compatible API | `[openrouter]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/openrouter/example.py) |
| CometAPI | 500+ models via OpenAI- or Anthropic-compatible API | `[cometapi]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/comet_api.py) |
| Databricks | Model Serving and Foundation Model APIs, native SDK queries, and MLflow deployment clients (including streaming) | `[databricks]` | [examples](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/databricks) |
| Anthropic | Claude Sonnet 4.6, Claude 3.5/3 series (15+) | `[anthropic]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/anthropic/example.py) |
| Google AI | Gemini 2.5/2.0 Pro/Flash, PaLM 2 (30+) | `[google]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/google_ai/example.py) |
| AWS Bedrock | Amazon Titan, Claude, Llama, Mistral (25+), Converse API | `[aws]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/aws_bedrock/example.py) |
| Azure OpenAI | Same as OpenAI with Azure pricing | `[openai]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/azure_openai/example.py) |
| Cohere | Command R/R+, Embed v4/v3, rerankers (15+) | `[cohere]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/cohere/example.py) |
| Mistral AI | Large/Medium/Small, Mixtral, embeddings (20+) | `[mistral]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/mistralai/example.py) |
| Together AI | DeepSeek-R1, Llama 3.x, Qwen (25+) | `[together]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/togetherai/example.py) |
| Groq | Llama 3.x, Mixtral, Gemma, Whisper (20+) | `[groq]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/groq/example.py) |
| Ollama | All local models with token tracking | `[ollama]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/ollama/example.py) |
| vLLM | In-process batch inference with queue/prefill/decode latency | `[vllm]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/inference_engines/vllm_example.py) |
| llama.cpp | Local GGUF models via llama-cpp-python | `[llamacpp]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/inference_engines/llamacpp_example.py) |
| Vertex AI | Gemini models via Google Cloud | `[vertexai]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/vertexai/example.py) |
| xAI SDK | Grok chat, image, and video generation; xAI voice REST calls | `[xai]` | [examples](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/xai_sdk) |
| SambaNova | sarvam-m, Saarika, Bulbul (12+) | `[sambanova]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/sambanova_example.py) |
| Sarvam AI | Indian language models | `[sarvamai]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/sarvam/) |
| ElevenLabs | Text-to-speech + Scribe speech-to-text | `[elevenlabs]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/elevenlabs_example.py) |
| Replicate | Hardware-based pricing ($/second), embeddings (BGE/E5/GTE/MPNet/MiniLM families) | `[replicate]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/replicate/example.py) |
| Azure AI Inference | Serverless and managed endpoints on Azure AI Foundry | `[azure-ai-inference]` | - |
| Anyscale | Llama, Mistral and Zephyr via an OpenAI-compatible endpoint | `[openai]` | - |
| Liquid Audio | Liquid Foundation audio models | `[liquid-audio]` | - |
| HuggingFace Transformers | Local models, cost estimated from parameter count | `[huggingface]` | - |
| Sentence Transformers | Local embedding models | `[huggingface]` | - |
| Hyperbolic | Open-weight models over raw HTTP. Disabled by default - see below | - | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/hyperbolic_example.py) |
| TypeSafe AI / Jev | SystemOne decisions (Jev, LiquidAI D1, self-hosted Hub models) as DECISION spans; see [Decision models](#decision-models) | `[typesafe]` | [example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/typesafe/example.py) |

## xAI SDK

Install the optional SDK dependency and set your xAI API key as `XAI_API_KEY`:

```bash
pip install "genai-otel-instrument[xai]"
```

Enable `xai_sdk` before creating the xAI client. Chat `sample()` and `stream()`,
image generation, and video generation are traced for both `Client` and `AsyncClient`,
including model, usage, latency, and xAI's exact reported request cost. The xAI
Text to Speech REST endpoint is also traced; cost uses the published per-character
rate because its audio response does not include a cost field. STT REST calls are
priced when the response provides audio duration. Request text and audio are not
captured. See the [chat](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/xai_sdk/chat.py),
[image/video](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/xai_sdk/image_video.py),
and [voice](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/xai_sdk/voice_tts.py)
examples.

The current official Python package exposes chat, image, and video helpers. Voice
helpers are not part of its released `Client` API yet, so TTS and STT are currently
instrumented through the documented xAI REST endpoints. WebSocket voice sessions
are not covered by this integration.

```python
import genai_otel
from xai_sdk import Client
from xai_sdk.chat import user

genai_otel.instrument(enabled_instrumentors=["xai_sdk"])
client = Client()
chat = client.chat.create(model="grok-4.6")
chat.append(user("Hello"))
response = chat.sample()
print(response.content)
```

xAI also offers an opt-in telemetry exporter in its SDK. Configure one exporter
path at a time to avoid duplicate spans for the same SDK calls.

## Databricks Model Serving

Install the integration and OpenAI-compatible SDK:

```bash
pip install "genai-otel-instrument[databricks]"
```

Enable the Databricks instrumentor before creating the client. Point it at a
Databricks workspace serving endpoint and pass the endpoint or model-service name in
`model`. Credentials stay in environment variables:

```python
import os

import genai_otel
from openai import OpenAI

genai_otel.instrument(enabled_instrumentors=["databricks"])

client = OpenAI(
    api_key=os.environ["DATABRICKS_TOKEN"],
    base_url=f"{os.environ['DATABRICKS_HOST'].rstrip('/')}/serving-endpoints",
)
response = client.chat.completions.create(
    model=os.environ["DATABRICKS_MODEL"],
    messages=[{"role": "user", "content": "Hello from TraceVerde"}],
)
```

Set `DATABRICKS_HOST` to your workspace URL, `DATABRICKS_TOKEN` to a token with
query access, and `DATABRICKS_MODEL` to the serving endpoint name. The instrumentor
also recognizes Databricks AI Gateway routes under `/ai-gateway/` and works with
`databricks-openai` clients that use these endpoints. It records chat, completions,
embeddings, and Responses API calls as `databricks.*` spans with
`gen_ai.provider.name=databricks`. See the complete [Databricks example](../../examples/databricks/example.py).

The native Databricks SDK is supported too:

```python
import genai_otel
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

genai_otel.instrument(enabled_instrumentors=["databricks"])
workspace = WorkspaceClient()  # Uses Databricks SDK authentication configuration
result = workspace.serving_endpoints.query(
    name="my-chat-endpoint",
    messages=[ChatMessage(role=ChatMessageRole.USER, content="Hello")],
)
```
See the complete [native SDK example](../../examples/databricks/sdk_example.py).

MLflow deployment clients are supported as well. The `[databricks]` extra includes
`mlflow-skinny` for this path. Calls to `predict` and `predict_stream` on
`get_deploy_client("databricks")` are recorded as Databricks spans; streaming spans
stay open through iteration so final-chunk token usage is captured. Other MLflow
deployment targets are left to their own provider instrumentation.

```python
import genai_otel
from mlflow.deployments import get_deploy_client

genai_otel.instrument(enabled_instrumentors=["databricks"])
client = get_deploy_client("databricks")
response = client.predict(
    endpoint="my-chat-endpoint",
    inputs={"messages": [{"role": "user", "content": "Hello"}]},
)
```

See the [MLflow example](../../examples/databricks/mlflow_example.py). `ChatDatabricks`
from `databricks-langchain` is also covered when its underlying client uses a
Databricks serving-endpoint or AI Gateway URL; the Databricks client span is emitted
under any framework span produced by LangChain.

## Decision models

A decision model takes a state plus a set of typed questions and returns, for each question,
a probability distribution over answers you supplied. It writes no text. TraceVerde records
every decision-model call as an OpenInference `DECISION` span (`openinference.span.kind =
DECISION`), whichever way it was made, with one attribute shape and its own cost.

### Protocols and call paths

Almost every decision model speaks one of two wire protocols, so the library instruments
**call paths, not models**. A Hub decision model served through any of these paths is covered
without an instrumentor of its own.

| Call path | Protocol | Span | Covers |
|---|---|---|---|
| `client.decisions.create` (OpenAI SDK, sync and async, `with_raw_response` too) | OpenAI Decisions (`POST /v1/decisions`) | `openai.decisions` | OpenAI `gpt-6-luna`; Azure OpenAI through `AzureOpenAI`; OpenRouter (`openrouter.decisions`); any server that answers `/v1/decisions` |
| `TypeSafeClient.system_one` / `AsyncTypeSafeClient.system_one` (`typesafe-sdk`, any `base_url`) | SystemOne (`POST /v1/systemone`) | `typesafe.system_one` | TypeSafe Jev, LiquidAI D1 / D1 Ultra, Cloudflare Workers AI, vLLM / SGLang / Ollajev / laya-serve serving Hub decision models (JEV, Clef, GEV, decider, ...) |
| a POST with `requests`, `httpx` or `httpx2` to `.../v1/systemone`, `.../decisions/v1/systemone` or `.../v1/decisions`, outside an SDK call | either | `decision.http.systemone` / `decision.http.openai` | apps that call the endpoints without an SDK, such as LiquidAI's image examples |
| OpenAI chat completions with `extra_body={"chat_template_kwargs": {"adapter_name": "decision"}}` | Granite Switch decision adapter | `openai.chat.decision` | IBM Granite Switch 4.1 on vLLM. Its other adapters (`guardian-core`) stay chat spans |
| `genai_otel.decision_span` (context manager or decorator) | in-process | `decision` (or your name) | Laya `Router`, Clef's `joint_schema_model.systemone`, `transformers` checkpoints, rules engines |

`GENAI_ENABLED_INSTRUMENTORS=openai` covers OpenAI Decisions and Granite Switch;
`typesafe` covers SystemOne. The HTTP path matcher is governed by
`GENAI_DECISION_HTTP_MATCHER` (on by default) and never wraps a call already inside an SDK
decision span, so one decision is one span. An `openai` release without the Decisions
resource (before 3.26) is skipped silently.

LLM structured outputs, logprob classification on a plain chat model and Instructor still
produce generated text that is parsed, so they stay LLM spans. To record one of them as a
decision, use `decision_span`.

### Self-hosted servers

Point the client at your server:

```python
from typesafe_sdk import TypeSafeClient

client = TypeSafeClient(api_key="unused", base_url="http://localhost:8000")  # vLLM / SGLang
client.system_one(state=ticket, questions=questions, model="Cloudflare/clef-flash")
```

`decision.provider` is taken from the host: `api.openai.com` -> `openai`, `openrouter.ai` ->
`openrouter`, `api.liquid.ai` -> `liquid`, `api.typesafe.ai` -> `typesafe`,
`api.cloudflare.com` -> `cloudflare`, `*.openai.azure.com` -> `azure`. Any other host is
self-hosted and gets no provider, unless `GENAI_DECISION_PROVIDER` names one.

### In-process models

```python
from genai_otel import decision_span

with decision_span(model="laya-router-421m", questions=questions, state=message) as d:
    result = router(message)
    d.record({"next_tool": {"type": "choice", "choice": result.label,
                            "confidence": result.confidence,
                            "probabilities": result.probabilities}},
             usage=None)   # leave usage out when the model reports none
```

`questions` and answers may be a SystemOne mapping or an OpenAI Decisions list. As a
decorator, the function's return value is recorded when it is a response (has `answers`) or a
set of answers. `d.fail(message)` ends the span as an error without an exception.

### Attributes

| Attribute | When | Notes |
|---|---|---|
| `openinference.span.kind` = `DECISION` | always | `LLM` on SystemOne calls under `GENAI_DECISION_LEGACY_LLM_KIND=true` |
| `decision.system` | always | the API shape: `openai`, `typesafe`, or `custom` for the helper |
| `decision.provider` | recognised host, or `GENAI_DECISION_PROVIDER` | never guessed |
| `decision.model_name` | always | response model, else request model |
| `decision.request.model_name`, `decision.response.model_name` | when they differ | |
| `decision.token_count.input`, `.output` | only when the provider reports them | never 0 for a missing value |
| `gen_ai.system`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` | always / when reported | kept so `gen_ai.*` readers see decisions |
| `gen_ai.operation.name` | always | `openai.decisions`, `typesafe.system_one`, `openai.chat.decision` or `decision` |
| `gen_ai.request.type` = `decision` | always | selects the `decision` pricing section |
| `gen_ai.usage.cost.total`, `gen_ai.usage.cost.decision` | when the model is priced | the same value on both |
| `genai.decision.protocol` | always | `openai_decisions`, `systemone`, `granite_adapter` or `manual` |
| `genai.decision.question_count`, `.refusal_count`, `.question.types` | always | counts and types only (`predicate`, `choice`, `score`, `refusal`, `box`; SystemOne's `noul` is recorded as `predicate`) |
| `genai.decision.option_count` | one-question calls | how many choices or levels were offered |
| `genai.decision.question.names` | **content capture on** | caller labels can be sensitive |
| `genai.decision.min_confidence` | **content capture on** | lowest confidence across choice and score answers |
| `genai.decision.choice`, `.confidence`, `.probability`, `.score`, `.margin` | **content capture on**, one-question calls | `margin` is the top probability minus the second |
| `input.value`, `output.value`, `*.mime_type` = `application/json` | **content capture on** | the request and response JSON, images replaced with `[image omitted]`, capped at `GENAI_CONTENT_MAX_LENGTH` |

SystemOne spans also keep the attributes they carried as LLM spans before 1.38
(`llm.request.model_name`, `llm.response.model_name`, `llm.token_count.*`,
`llm.invocation_parameters`, `gen_ai.response`) until 2.0.

### What content capture decides

With content capture off (the default) a decision span says which model decided, through
which provider, how many questions of which types, how many were refused, tokens, cost,
latency and status, and **nothing about the answer**: no chosen value, no confidence or score,
no question names, no state. `GENAI_ENABLE_CONTENT_CAPTURE=true` adds them. The default
200-character content cap truncates the request and response JSON, which is why the summary
attributes exist.

### Cost

Decision models are priced from their own `decision` section of the pricing table, and **only**
from it. A model missing from that section gets no cost attribute rather than a chat price:
`gpt-6-luna` used as a decision model bills input only, while `gpt-6-luna` the chat model bills
output at $0.50/1M.

| Model key | Input per 1M | Output |
|---|---|---|
| `gpt-6-luna` (OpenAI Decisions; cache reads and writes free) | $0.10 | $0 |
| `openai/gpt-6-luna-decisions` (OpenRouter) | $0.10 | $0 |
| `d1` / `d1-ultra` (LiquidAI, any route) | $0.02 / $0.042 | $0 |
| `jev` (TypeSafe) | $0.042 | $0 |
| `clef`, `clef-flash`, `clef-omni` | not priced until Cloudflare publishes a price | |

Dated and routed variants resolve to their key (`jev-1.13.0-20260917`, `liquid/d1`).
Self-hosted weights (Hub models, Granite Switch, Laya) have no list price and get no cost
attribute; price one with a `decision` section in `GENAI_CUSTOM_PRICING_JSON`. Regional and
long-context multipliers are not modelled.

### Evaluation

On decision spans the input is still screened for PII, prompt injection and restricted topics.
Hallucination and response checks are skipped, because there is no generated text, and the
span says so with `evaluation.skipped_reason = decision_span`.

### Examples

`examples/decisions/` has one runnable example per call path; the README there says which ones
need a key or a server.

## TypeSafe AI / Jev

TypeSafe's System One API evaluates typed questions against unstructured state and
returns decisions instead of generated prose. TraceVerde wraps both
`TypeSafeClient.system_one` and `AsyncTypeSafeClient.system_one` natively, as
`DECISION` spans (see [Decision models](#decision-models)): the model, the provider
behind `base_url`, question counts and types, token usage and Jev's published input
price, and, with content capture on, the typed answers and their confidence.

Before 1.38 these were `LLM` spans. The `llm.*` attributes are still emitted until
2.0, and `GENAI_DECISION_LEGACY_LLM_KIND=true` keeps the old span kind.

```bash
pip install 'genai-otel-instrument[typesafe]'
export TYPESAFE_API_KEY="your-key"
export OTEL_EXPORTER_OTLP_ENDPOINT="https://otel.example.internal:4318"
export OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf"
export OTEL_EXPORTER_OTLP_CERTIFICATE="/absolute/path/to/platform-ca.crt"
export GENAI_ENABLED_INSTRUMENTORS="typesafe"
export GENAI_ENABLE_CONTENT_CAPTURE=true
export GENAI_CONTENT_MAX_LENGTH=0
python examples/typesafe/example.py
```

The example sends one state with a `Choice`, `Noul`, and `Score` question so you
can inspect the complete structured response in the terminal and the resulting
`typesafe.system_one` span in the platform backend. The OTLP endpoint must use
HTTPS, and `OTEL_EXPORTER_OTLP_CERTIFICATE` must point to the CA certificate
that signed it.

Replicate hosts arbitrary community models behind one generic `run()` call,
with no fixed input/output schema and no dedicated embeddings endpoint to
hook. Embedding calls are recognized by matching the model reference against
known embedding-only model families (BGE, E5, GTE, MPNet, MiniLM, and any
model whose name contains "embed"); a model outside those families is traced
as a plain `replicate.run` span, cost tracking excluded, same as before.

## Audio Providers: Billing by Media, Not Tokens

ElevenLabs is billed per unit of media rather than per token, so its spans carry
different usage attributes. Text-to-speech is priced per character of input text
and Scribe speech-to-text per second of audio.

```python
import genai_otel
genai_otel.instrument()

from elevenlabs import ElevenLabs
client = ElevenLabs(api_key="...")

# Text-to-speech: convert() returns an iterator of audio bytes. Draining it is
# what completes the span and records time-to-first-byte.
audio = b"".join(client.text_to_speech.convert(
    voice_id="21m00Tcm4TlvDq8ikWAM",
    text="Hello from TraceVerde.",
    model_id="eleven_multilingual_v2",
))

# Scribe speech-to-text
transcript = client.speech_to_text.convert(model_id="scribe_v1", file=open("call.mp3", "rb"))
```

| Attribute | Operation | Meaning |
|-----------|-----------|---------|
| `gen_ai.usage.characters` | text_to_speech | Input characters, the billed unit |
| `gen_ai.server.time_to_first_token` | text_to_speech | Time to first audio byte (also emitted as `gen_ai.server.ttft`) |
| `gen_ai.request.voice_id` | text_to_speech | Voice used for synthesis |
| `gen_ai.usage.audio_duration_seconds` | speech_to_text | Audio seconds, the billed unit |
| `gen_ai.response.transcript_length` | speech_to_text | Characters of transcript returned |

Time-to-first-byte matters more than total duration for voice, since streamed
synthesis begins playing before generation finishes - it is what the caller
actually waits for on a voice turn.

**Audio payloads are never attached to spans.** Only sizes and durations are
recorded. For voice workloads the audio is frequently personal data, so
reference-only is the default rather than something to opt into.

## Quick Example: OpenAI

```python
import genai_otel
genai_otel.instrument()

from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is OpenTelemetry?"},
    ],
    max_tokens=150,
)

print(f"Response: {response.choices[0].message.content}")
print(f"Tokens used: {response.usage.total_tokens}")
# Traces, metrics, and costs are automatically captured
```

## Quick Example: Responses API

`client.responses.create` is traced as its own span (`openai.responses`). This is
the default path for native GPT-5.6+ models, because Chat Completions rejects
function tools combined with reasoning -- so agent runtimes that use tools and
reasoning together end up here rather than on `chat.completions.create`.

```python
import genai_otel
genai_otel.instrument()

from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5.6",
    instructions="You are a helpful assistant.",
    input="What is OpenTelemetry?",
    max_output_tokens=150,
)

print(response.output_text)
```

The Responses shape differs from Chat Completions in every field the
instrumentation reads, and each is mapped onto the same semantic conventions so
Responses and Chat Completions spans stay comparable:

| Responses | Chat Completions | Recorded as |
|---|---|---|
| `input` (string or list), `instructions` | `messages` | `gen_ai.request.message_count`, `gen_ai.request.first_message`, `gen_ai.request.instructions` |
| `max_output_tokens` | `max_tokens` | `gen_ai.request.max_tokens` |
| `output[]` items | `choices[]` | completion events, `gen_ai.response` |
| `usage.input_tokens` / `output_tokens` | `usage.prompt_tokens` / `completion_tokens` | `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens` |
| `output_tokens_details.reasoning_tokens` | `completion_tokens_details.reasoning_tokens` | `gen_ai.usage.reasoning_tokens` |
| `input_tokens_details.cached_tokens` | `prompt_tokens_details.cached_tokens` | `gen_ai.usage.cache_read.input_tokens` |
| `status` / `incomplete_details.reason` | `choices[].finish_reason` | `gen_ai.response.finish_reasons` |

Reasoning tokens are attributed as output, because that is how they are billed.
Tool calls come from the `function_call` items of `output[]`, and `response.id`
is recorded so `store=true` responses stay joinable server-side. Streaming works
the same as elsewhere -- TTFT and inter-token latency are measured as the
response events arrive.

## Quick Example: AWS Bedrock Converse

All four Bedrock runtime calls are traced: `invoke_model`,
`invoke_model_with_response_stream`, `converse` and `converse_stream`.

Converse is the unified API AWS points callers at, and the practical path for
every non-Anthropic model -- it removes the per-vendor request body that makes
`invoke_model` awkward. Because it is model-agnostic, the span shape does not
depend on `modelId`.

```python
import genai_otel
genai_otel.instrument()

import boto3

client = boto3.client("bedrock-runtime", region_name="us-east-1")
response = client.converse(
    modelId="meta.llama3-70b-instruct-v1:0",
    system=[{"text": "You are a helpful assistant."}],
    messages=[{"role": "user", "content": [{"text": "What is OpenTelemetry?"}]}],
    inferenceConfig={"maxTokens": 256, "temperature": 0.2},
)

print(response["output"]["message"]["content"][0]["text"])
```

Converse differs from `invoke_model` in every field the instrumentation reads,
and each is mapped onto the same semantic conventions the other providers emit:

| Converse | Recorded as |
|---|---|
| `modelId` | `gen_ai.request.model` |
| `messages[].content[]` typed blocks (`text`, `image`, `toolUse`, `toolResult`) | `gen_ai.request.message_count`, `gen_ai.request.first_message` |
| `system` (a top-level parameter, **not** a message role) | `gen_ai.request.instructions` |
| `inferenceConfig.{maxTokens, temperature, topP, stopSequences}` | `gen_ai.request.{max_tokens, temperature, top_p, stop_sequences}` |
| `output.message.content[]` | completion events, `gen_ai.response` |
| `usage.{inputTokens, outputTokens, totalTokens}` (camelCase) | `gen_ai.usage.{input_tokens, output_tokens}` |
| `stopReason` | `gen_ai.response.finish_reasons` |
| `toolUse` blocks | tool-call attributes |

### Streaming

`converse_stream` returns `{"stream": ...}` immediately -- the model generates
while you iterate -- so the span stays open until the event stream is exhausted
rather than closing on return, which would report near-zero latency and no
tokens. Token counts arrive only in the trailing `metadata` event and are picked
up from there.

```python
response = client.converse_stream(
    modelId="meta.llama3-70b-instruct-v1:0",
    messages=[{"role": "user", "content": [{"text": "Explain tracing."}]}],
)

for event in response["stream"]:
    if "contentBlockDelta" in event:
        print(event["contentBlockDelta"]["delta"]["text"], end="")
```

## Quick Example: Embeddings and RAG

Embedding calls are traced as their own spans, so a retrieval-augmented call
shows both of its legs: the lookup that chose the context and the generation
that used it. Without the embedding span, the retrieval step is invisible and
its tokens and cost go unrecorded.

```python
import genai_otel
genai_otel.instrument()

from openai import OpenAI
from opentelemetry import trace

client = OpenAI()
tracer = trace.get_tracer("rag.demo")

with tracer.start_as_current_span("rag.pipeline"):
    # 1. Index - one span, input_count = number of chunks
    indexed = client.embeddings.create(
        model="text-embedding-3-small",
        input=["chunk one", "chunk two", "chunk three"],
    )

    # 2. Retrieve - one span for the query embedding
    query = client.embeddings.create(
        model="text-embedding-3-small",
        input="what did chunk two say?",
    )

    # 3. Generate - the chat span, as usual
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Answer from the retrieved context."}],
    )
```

Wrapping the three calls in a parent span groups them into a single trace, so
the whole pipeline is one unit in the UI rather than three unrelated calls.

Embedding spans carry `gen_ai.request.input_count`,
`gen_ai.response.embedding_count` and `gen_ai.response.vector_size` alongside
the usual token and cost attributes, priced against the embeddings table. The
embedded text itself is recorded as `embedding.text` only when
`GENAI_ENABLE_CONTENT_CAPTURE=true`, since retrieval inputs frequently contain
user data. Vectors stay off entirely unless explicitly requested - they would
otherwise dominate span size. See the
[semantic conventions reference](../reference/semantic-conventions.md#embeddings)
for the full attribute list.

## Quick Example: Anthropic

```python
import genai_otel
genai_otel.instrument()

import anthropic

client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain quantum computing in one sentence."}
    ],
)

print(message.content[0].text)
# Cost tracking and token usage automatically captured
```

## Quick Example: CometAPI

CometAPI is an all-in-one aggregator that exposes 500+ models (GPT, Claude, Gemini, DeepSeek, Qwen, and more) behind a single API key. It is compatible with both the Anthropic SDK and the OpenAI SDK - point either client's `base_url` at `https://api.cometapi.com` and TraceVerde detects and traces the calls with `gen_ai.system = "cometapi"`.

```python
import genai_otel
genai_otel.instrument()

import anthropic

client = anthropic.Anthropic(
    base_url="https://api.cometapi.com",
    api_key="your-cometapi-key",  # from https://www.cometapi.com/console/token
)
message = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Can you describe LLMs to me?"}
    ],
)

print(message.content[0].text)
# Spans named cometapi.messages.create with token usage and cost tracking
```

The OpenAI SDK works the same way with `base_url="https://api.cometapi.com/v1"` (spans are named `cometapi.chat.completion`).

!!! note "One span per call"
    Aggregator clients (CometAPI, OpenRouter) are traced **only** by their
    dedicated instrumentor - the generic OpenAI/Anthropic instrumentors skip
    clients whose `base_url` points at an aggregator, so each call produces
    exactly one span and one set of token/cost metrics (since v1.5.1). If you
    disable the aggregator instrumentor, the generic SDK instrumentor traces
    those clients instead.

## Quick Example: Ollama (Local)

```python
import genai_otel
genai_otel.instrument()

import ollama

response = ollama.chat(
    model="llama2",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
)

print(response["message"]["content"])
# Local model traces captured with token counting
```

## Special Providers

### HuggingFace Transformers

Local model execution with estimated costs based on parameter count.

```bash
pip install genai-otel-instrument[huggingface]
```

Instruments:

- `pipeline()`
- `AutoModelForCausalLM.generate()`
- `AutoModelForSeq2SeqLM.generate()`
- `InferenceClient` API calls

See examples:

- [Basic HuggingFace](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/huggingface/example.py)
- [AutoModel](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/huggingface/example_automodel.py)
- [With PII detection](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/huggingface/pii_example.py)
- [With toxicity detection](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/huggingface/toxicity_example.py)
- [With bias detection](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/huggingface/bias_example.py)
- [Multiple evaluations](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/huggingface/multiple_evaluations_example.py)

### Hyperbolic

Requires OTLP gRPC exporter due to `requests` library conflicts.

```bash
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export GENAI_ENABLED_INSTRUMENTORS="openai,anthropic,hyperbolic"
```

See [Hyperbolic example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/hyperbolic_example.py).

### Google GenAI (new SDK)

```bash
pip install genai-otel-instrument[google]
```

See [Google GenAI example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/google_genai_example.py).

### LiteLLM (Multi-Provider Proxy)

```bash
pip install genai-otel-instrument[openinference]
```

LiteLLM enables cost tracking across 100+ providers via a single proxy. Streaming latency
is reported automatically for routes litellm sends through the OpenAI SDK; for
routes it serves with its own HTTP client, enable the opt-in `litellm_latency`
instrumentor (see [Configuration](../getting-started/configuration.md)). See [LiteLLM example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/litellm/example.py).

### Smolagents (HuggingFace Agents)

```bash
pip install genai-otel-instrument[openinference]
```

See [Smolagents example](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples/smolagents/example.py).

## Captured Attributes

For every LLM call:

| Attribute | Description |
|-----------|-------------|
| `gen_ai.system` | Provider name (e.g., "openai") |
| `gen_ai.request.model` | Requested model |
| `gen_ai.response.model` | Actual model used |
| `gen_ai.request.type` | Call type (chat, embedding) |
| `gen_ai.usage.input_tokens` | Input token count |
| `gen_ai.usage.output_tokens` | Output token count |
| `gen_ai.usage.total_tokens` | Total tokens |
| `gen_ai.usage.cost.total` | Estimated cost in USD |

## All Examples

Browse all provider examples in the [examples/ directory](https://github.com/Mandark-droid/genai_otel_instrument/tree/main/examples).
