# Upstream issues — ready to file

Source: `traceverse-chaos-lab/docs/VOICE_RAG_USECASE_SPEC.md` §8 (U1–U6), re-verified against this
checkout at **`v1.19.0-2-gbfa8bc1`** (HEAD `bfa8bc1`) and the live upstream issue on **2026-08-20**.
U1 is already tracked upstream as [#23](https://github.com/Mandark-droid/genai_otel_instrument/issues/23)
and must not be filed as a duplicate. The five sections below are U2–U6; two earlier proposals were
dropped as already-solved.

## Implementation status for the v1.20.0 release candidate

The checkout now implements and tests the concrete gaps described here:

- U1 / upstream #23: provider embedding spans for Cohere (sync and async clients), Google GenAI
  legacy/new SDKs including async models, Ollama module and async-client paths, Together sync/async
  embeddings, Bedrock model payloads, LiteLLM direct embedding calls, Azure AI Inference, Hugging Face
  feature extraction, and SentenceTransformers. The provider contract includes input count, embedding
  count, vector size, shared pricing dispatch, and content/vector-capture gates.
- U2: ASR attributes for the Hugging Face ASR pipeline and direct audio model generation, plus the
  optional Liquid Audio generator instrumentor. Sarvam and hosted TTS/ASR paths retain vendor fields
  while emitting shared media fields.
- U3/U4: the documented additive `rag.*` helper, score summaries across supported vector responses,
  and awaited `AsyncQdrantClient` query/search coverage.
- U5/U6: shared TTS fields and Sarvam streaming measurement, plus the public `gen_ai.degraded` event
  helper and semantic-convention documentation.

The remaining issue-23 follow-up is provider-specific HTTP instrumentation for applications that call
an embedding service directly without an instrumented SDK; this release covers the SDK entry points
owned by this repository. The implementation and validation are in
[PR #25](https://github.com/Mandark-droid/genai_otel_instrument/pull/25), and the
issue alignment comment has been posted on
[#23](https://github.com/Mandark-droid/genai_otel_instrument/issues/23#issuecomment-5352281588).
After the tag is published, update that comment with the release link and close
the issue or split the remaining raw-HTTP follow-up.

Each `##` section below is one GitHub issue. The `### Title` line goes in the title field; everything
under it goes in the body. U1/#23 is an existing issue and its suggested
alignment comment is already filed; the remaining sections are draft issue
content unless marked otherwise.

---

## U1 — already tracked upstream; do not file a duplicate

### Title

Reference only: close the provider-coverage gap in upstream [#23](https://github.com/Mandark-droid/genai_otel_instrument/issues/23)

### Body (comment for #23, not a new issue)

The embedding gap below is already tracked by upstream #23. Do not submit this section as a new
GitHub issue. The issue is still **open** and had **no comments** when checked on 2026-08-20. Add the
comment below to align the issue with the current checkout and offer a concrete implementation path.

**Suggested comment for #23**

Thanks for opening this. I rechecked the current `main` at `bfa8bc1` against the issue scope. The
OpenAI SDK has both sync and async embedding coverage, and Mistral wraps `Embeddings.create`; the
remaining gap is real for the other provider paths.

The first implementation slice should be the provider-specific APIs that are already represented by
an instrumentor in this repository:

| Provider/path | Current state at `bfa8bc1` | Proposed first slice |
|---|---|---|
| Cohere `Client.embed` | `Client.generate` only | Add an embedding span; keep rerank separate unless the PR also has response-shape coverage |
| Google GenAI `embed_content` / `models.embed_content` | `generate_content` only | Add legacy and new-SDK embedding coverage |
| Ollama `embed` / `embeddings` | `generate` and `chat` only | Add module-level embedding coverage for both API spellings |
| Together embeddings | chat/completion only | Add coverage where the installed SDK exposes an embeddings method |
| HuggingFace feature extraction | generic pipeline span only | Preserve the generic span and add embedding-specific response attributes where the result is identifiable |

The harder paths should remain explicit follow-ups rather than being silently marked done: Bedrock
`invoke_model` needs model-specific request/response decoding, LiteLLM needs direct `embedding` /
`aembedding` coverage without relying on a transitive OpenAI route, and Azure AI Inference is a
different SDK from `openai.AzureOpenAI`. SentenceTransformers `encode` is also still unwrapped and
is the most important local in-process path alongside Ollama.

For each provider, the acceptance contract from this issue is:

- one embedding span per SDK call, with a provider-specific name ending in `.embeddings` and
  `gen_ai.operation.name = "embeddings"`;
- `gen_ai.request.type = "embedding"` (singular, because cost dispatch uses that value),
  `gen_ai.request.model`, and `gen_ai.request.input_count`;
- `gen_ai.response.embedding_count` and `gen_ai.response.vector_size` when present;
- usage and cost resolved through the `embeddings` pricing category;
- text captured only when content capture is enabled and vectors disabled unless explicitly enabled;
- sync and async variants covered where the SDK exposes both; and
- a focused test proving the span count and attributes with a stubbed client, without a live vendor
  API call.

Extending an existing provider instrumentor does not require a new registration entry. Preserve its
current idempotency guard and optional dependency. Add a new optional extra only for a genuinely new
SDK dependency, such as `sentence-transformers` if that path is accepted.

The first slice can follow the existing OpenAI/Mistral test shape; leave the issue open until the
remaining provider rows have either landed or been split into linked follow-ups.

The TraceVerse voice-RAG workload is the motivating case: Ollama `embed`/`embeddings` is the common
local-RAG path and is the blocking dependency for that lab.

### Supporting evidence (not part of the suggested comment)

**Problem**

`OllamaInstrumentor.instrument()` wraps exactly two module-level functions —
`ollama.generate` and `ollama.chat` (`genai_otel/instrumentors/ollama_instrumentor.py:76-90`).
It does not wrap `ollama.embed` or `ollama.embeddings`, so an application that embeds against a
local Ollama server produces no embedding span at all.

On the HuggingFace side, `SentenceTransformer.encode` is not wrapped anywhere —
`sentence-transformers` is not imported by any instrumentor. A
`transformers.pipeline("feature-extraction")` call *does* get a span, but only the generic
`huggingface.pipeline` span from `huggingface_instrumentor.py:144-204`, carrying
`gen_ai.system`, `gen_ai.request.model`, `gen_ai.operation.name = "feature-extraction"` and
`huggingface.task` — no vector dimensionality, no input count.

OpenAI embeddings were instrumented in `408b4e0` (span `openai.embeddings`,
`gen_ai.operation.name = "embeddings"`, `gen_ai.request.type = "embedding"`,
`gen_ai.request.input_count`, `gen_ai.response.embedding_count`, `gen_ai.response.vector_size`,
`embedding.model_name`, `embedding.text`). That commit is **after `v1.19.0`**, so it is absent from
PyPI 1.18.0. Local backends have no equivalent at any version.

**Why it matters**

Any on-premise, edge, or air-gapped RAG deployment embeds locally by definition — usually the whole
reason it is on-premise. For those deployments the library instruments the vector-DB search (via
`genai_otel/mcp_instrumentors/vector_db_instrumentor.py`) and the LLM generation, but leaves a hole
exactly between them. The embedding step is where model identity, vector dimensionality, and
embedder-versus-index consistency live. A drifted or downgraded embedder returns a normal hit count
at normal latency with a plausible answer; with no embedding span there is nothing in the trace that
could have caught it.

**Local-RAG implementation acceptance criteria (supporting U1)**

- Ollama embedding calls emit a span, whether issued via the `ollama` Python client (`ollama.embed`,
  `ollama.embeddings`) or via `httpx`/`requests` against `${OLLAMA_BASE_URL}/api/embed` and
  `/api/embeddings`. Preferably by extending `OllamaInstrumentor` rather than adding a separate
  entry, since the existing instrumentor already owns the module-level wrap and the idempotency
  guard `_genai_otel_ollama_instrumented`.
- A `SentenceTransformersInstrumentor` emits a span for `SentenceTransformer.encode`.
- Spans reuse the vocabulary `openai_instrumentor.py` already established, rather than inventing a
  parallel one: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.operation.name = "embeddings"`,
  `gen_ai.request.input_count`, `gen_ai.response.embedding_count`, `gen_ai.response.vector_size`.
- Both use the existing helper `BaseInstrumentor.add_embedding_attributes(span, model, input_text,
  vector=None)` (`genai_otel/instrumentors/base.py:2249`) for `embedding.model_name` /
  `embedding.text` instead of setting those inline. Note that helper is currently called by nothing —
  `openai_instrumentor._add_embedding_content` duplicates its logic — so adopting it in new code is
  the cheapest way to stop the divergence.
- Extending `OllamaInstrumentor` preserves its existing registration, optional `ollama` extra, and
  `_genai_otel_ollama_instrumented` idempotency guard. If SentenceTransformers is implemented as a
  new instrumentor, register it in the existing import, instrumentor map, defaults, and optional
  dependency surfaces rather than adding an unrelated second path.
- A test asserts that a local embed call against a stubbed Ollama endpoint produces exactly one span
  with the attributes above.

**Provenance**

Came out of TraceVerse chaos-lab voice-RAG work: building a local-first voice RAG pipeline against
this library, the embedding leg was the only stage with no span, which made embedder-drift chaos
scenarios undetectable from traces.

---

## Issue U2

### Title

Add ASR-specific attributes to local speech-to-text spans (HF `automatic-speech-recognition`, vendor audio runtimes)

### Body

**Problem**

Local ASR is not entirely uninstrumented — a `transformers.pipeline("automatic-speech-recognition")`
call does produce a span, because `huggingface_instrumentor.py:144-204` wraps `transformers.pipeline`
generically. But that span carries only `gen_ai.system`, `gen_ai.request.model`,
`gen_ai.operation.name = <task>`, and `huggingface.task`. None of the things that make a
transcription span useful are on it.

This is doubly frustrating because the library **already computes** the missing audio duration:
`genai_otel/cost_estimation.py:166` defines `audio_seconds(args, kwargs, sampling_rate=None)`, which
handles dicts with `array` + `sampling_rate` and numpy/torch tensors, and the ASR/audio task list at
`cost_estimation.py:270` routes through it. The value is used for cost estimation and then discarded —
it is never set as a span attribute.

Separately, non-pipeline paths are uncovered: direct model-class ASR and vendor runtimes such as
LiquidAI `liquid-audio` (`LFM2AudioModel` / `LFM2AudioProcessor`) get nothing.
`_instrument_model_classes` wraps `GenerationMixin.generate` only.

**Why it matters**

Voice agents are increasingly local-first for latency and privacy reasons, and ASR is the first stage
of the pipeline — an error or quality regression there propagates into retrieval and generation, but
currently appears in the trace only as an unexplained change in downstream behaviour. Without
transcription confidence, detected language, audio duration, or real-time factor on the span, ASR
quality drift is undiagnosable from traces alone.

**Acceptance criteria**

- Local transcription spans carry the vocabulary the **hosted** ASR instrumentors already established,
  rather than a new one. Specifically, from
  `genai_otel/instrumentors/elevenlabs_instrumentor.py:285,300-315`:
  - `gen_ai.operation.name = "speech_to_text"` (**not** `"transcribe"` — that value appears nowhere
    in the codebase and would fragment the namespace)
  - `gen_ai.request.type = "speech_to_text"`
  - `gen_ai.usage.audio_duration_seconds`
  - `gen_ai.response.transcript_length`
  - `gen_ai.request.language_code` / `gen_ai.response.language_code`
- New, genuinely missing attributes to add across both hosted and local:
  `gen_ai.request.audio.sample_rate`, transcription confidence where the model exposes it, and
  real-time factor (`processing_time / audio_duration`).
- The HF pipeline wrapper populates `gen_ai.usage.audio_duration_seconds` from the value
  `cost_estimation.audio_seconds()` already computes, instead of dropping it.
- Coverage extends beyond `transformers.pipeline` to at least one vendor runtime (`liquid-audio`
  proposed).
- If a separate instrumentor is warranted, register it in all five places per
  `.claude/skills/add-provider/SKILL.md` §3 (`INSTRUMENTORS` at `auto_instrument.py:185`,
  `DEFAULT_INSTRUMENTORS` at `config.py:32`, `instrumentors/__init__.py`, both `auto_instrument.py`
  import blocks, and a `pyproject.toml` extra).
- A test asserts one span per transcription call with the attributes above.

**Provenance**

Came out of TraceVerse chaos-lab voice-RAG work.

---

## Issue U3

### Title

Add a retrieval-quality attribute set and an `add_retrieval_quality_attributes` helper

### Body

**Problem**

`BaseInstrumentor.add_retrieval_attributes` (`genai_otel/instrumentors/base.py:2268`) emits
`retrieval.query`, `retrieval.documents.{i}.document.{id,score,content}` plus
`.metadata.{key}`, and `retrieval.document_count`. These describe *what was retrieved*.

Two corrections to how this is often described:

- It does **not** emit `gen_ai.rag.context`. That attribute is only *read*, by
  `genai_otel/evaluation/span_processor.py:1214` and
  `genai_otel/evaluation_enriching_exporter.py:532`, as a context source for hallucination
  detection. An application has to set it itself, and nothing documents that.
- `top_k` **does** already exist, as `db.vector.top_k` — set by the vector-DB instrumentors
  (`genai_otel/mcp_instrumentors/vector_db_instrumentor.py:20-22`, and per-backend at the Qdrant and
  Chroma wrappers), alongside the legacy spellings `vector.limit` / `vector.n_results` / `vector.k`.
  A new `rag.search.top_k` would duplicate it.

What is genuinely absent is retrieval *quality*: no similarity threshold, no distance metric, no
score distribution summary, no corpus or index version, and — most importantly — no way to express
that the query-time embedding model differs from the model the index was built with. Note also that
the vector-DB wrappers set request-side attributes only; they `return wrapped(*args, **kwargs)`
without inspecting the result, so the scores that *are* in the client response are never recorded.

**Why it matters**

The highest-severity RAG failure is silent: a drifted, swapped, or downgraded embedding model returns
the configured number of documents, at normal latency, with a fluent and wrong answer. Every
conventional health signal stays green. The only cheap, reliable detector is a single boolean
comparing the query embedder against the index embedder, and there is currently no standard attribute
to put it in. Because the library already owns the `retrieval.*` and `db.vector.*` prefixes, an
official quality vocabulary here would prevent every downstream user from inventing a private,
mutually incompatible one — which is the situation today.

**Acceptance criteria**

- A documented set of retrieval-quality attributes covering at minimum:
  `rag.embedding.model`, `rag.embedding.index_model`, `rag.embedding.model_match` (bool),
  `rag.embedding.dim`, `rag.search.score_floor`, `rag.search.distance`,
  `rag.result.score_max`, `rag.result.score_min`, `rag.result.score_mean`, `rag.result.score_margin`,
  `rag.corpus.version`, `rag.context.tokens_est`, `rag.context.truncated`, `rag.answer.refused`.
- **`top_k` and result count are deliberately excluded** — reuse the existing `db.vector.top_k` and
  `retrieval.document_count` rather than adding `rag.search.top_k` / `rag.result.count`. If a
  duplicate is wanted anyway, the issue should say why in the docs.
- A public helper `BaseInstrumentor.add_retrieval_quality_attributes(span, **kwargs)` mirroring the
  ergonomics of `add_retrieval_attributes`. (`base.py` currently exposes exactly four public methods:
  `instrument`, `create_span_wrapper`, `add_embedding_attributes`, `add_retrieval_attributes`.)
- Documentation stating explicitly that the new attributes are additive to, not a replacement for,
  `retrieval.*`, `db.vector.*`, and `gen_ai.rag.context` — and documenting that `gen_ai.rag.context`
  is an application-set input to the evaluation processors, not something the library emits.
- Vector-DB instrumentors populate `rag.result.score_*` automatically where the client response
  already contains the scores, which requires them to start inspecting the return value.

**Provenance**

Came out of TraceVerse chaos-lab voice-RAG work: the embedder-drift chaos scenario needed exactly the
`rag.embedding.model_match` boolean, and there was nowhere standard to put it.

---

## Issue U4

### Title

Instrument `AsyncQdrantClient` (and async clients generally) in the vector-DB instrumentor

### Body

**Blocking:** No for the lab — it works around this by using the sync client — but the workaround is
exactly the kind of thing users will not think to do.

**Problem**

`VectorDBInstrumentor._instrument_qdrant`
(`genai_otel/mcp_instrumentors/vector_db_instrumentor.py:216-272`) wraps
`qdrant_client.QdrantClient.query_points` and `qdrant_client.QdrantClient.search`. It does not wrap
`qdrant_client.AsyncQdrantClient`; the file contains no async references, so an application using the
async client produces no vector-DB span.

**Why it matters**

Async is the default choice for agent frameworks, web services, and voice pipelines where blocking the
event loop is unacceptable. The better-concurrency configuration therefore gets less telemetry, with
no error, log line, or missing-instrumentation signal.

**Acceptance criteria**

- `AsyncQdrantClient.query_points` (and `.search` where `query_points` is unavailable) emits the same
  span and attributes as the sync path: `qdrant.query_points`, `db.system`, `db.operation`,
  `db.collection.name`, `vector.collection`, `db.vector.top_k`, and `vector.limit`.
- The wrapper is `async def` and awaits the coroutine, so the span covers the actual round-trip.
- Audit other backends for async clients, with coverage or an explicit documented unsupported note.
- Document covered client classes in the module's attribute-naming section.
- Add a test proving an async Qdrant query emits exactly one span whose duration covers the awaited call.

**Provenance**

Found while writing the TraceVerse chaos-lab voice-RAG spec: the original optimized mode used
`AsyncQdrantClient`, which would have removed vector-DB spans from the good configuration.

---

## Issue U5

### Title

Normalise the TTS span shape across ElevenLabs and Sarvam, and document it

### Body

**Problem**

A TTS span shape already exists in two places, and they disagree.

`ElevenLabsInstrumentor` (`genai_otel/instrumentors/elevenlabs_instrumentor.py:188-202`) sets
`gen_ai.operation.name = "text_to_speech"`, `gen_ai.usage.characters`, `gen_ai.request.voice_id`,
span name `elevenlabs.text_to_speech.<method>`. For streaming synthesis it calls
`BaseInstrumentor._record_time_to_first_token` (`base.py:2008`) at line 265, which emits
`gen_ai.server.time_to_first_token` and `gen_ai.server.ttft` plus the TTFT histograms — so
time-to-first-byte **is** already measured for one provider.

`SarvamAIInstrumentor` (`sarvam_instrumentor.py:528-565`) also sets
`gen_ai.operation.name = "text_to_speech"`, but everything else is vendor-prefixed:
`sarvam.tts.{pace,temperature,pitch,loudness,speech_sample_rate,enable_preprocessing,output_audio_codec}`.
It records no TTFT at all.

Neither shape is documented anywhere in `docs/`. There is no `docs/guides/` page for speech, and the
attributes appear only in source.

**Why it matters**

In a voice agent, perceived responsiveness is dominated by TTS time-to-first-byte, not by total
generation time — a 3-second utterance that starts speaking in 130 ms feels fast, and one that starts
in 1.3 s feels broken. The library already measures this for ElevenLabs and not for Sarvam, and
documents it for neither, so a user comparing two providers gets an apples-to-oranges trace and no way
to know why. Undocumented, provider-specific attribute names also mean every voice application either
rediscovers them by reading source or invents its own.

**Acceptance criteria**

- A documented TTS span shape, in `docs/guides/` alongside the existing `sarvam-ai.md` and
  `multimodal.md` pages, built on what already ships:
  `gen_ai.operation.name = "text_to_speech"` (**not** `"synthesize"` — that value appears nowhere and
  would break both existing instrumentors), `gen_ai.request.model`, `gen_ai.request.voice_id`,
  `gen_ai.usage.characters`, and `gen_ai.server.time_to_first_token` for TTFB.
- Genuinely missing attributes added to both providers:
  `gen_ai.usage.audio_duration_seconds` (already used on the ASR side, so reuse it),
  `gen_ai.response.output_format`, and `gen_ai.request.streamed` (bool).
- Sarvam's `sarvam.tts.*` attributes stay, but the portable subset (voice, sample rate, output codec)
  is *also* emitted under the shared names, so cross-provider queries work.
- `SarvamAIInstrumentor` calls `_record_time_to_first_token` on its streaming synthesis path, the way
  ElevenLabs does.
- TTFT/TTFB is documented as absent (not zero) for non-streaming synthesis — consistent with the
  existing reasoning in `_record_time_per_output_token`, which already refuses to fabricate a zero.

**Provenance**

Came out of TraceVerse chaos-lab voice-RAG work.

---

## Issue U6

### Title

Standardise a `gen_ai.degraded` span event for silent capability downgrades

### Body

**Problem**

Libraries and SDKs routinely degrade silently: a hosted model falls back to a local one after an auth
failure, a large model falls back to a small one under memory pressure, a streaming path falls back to
batch, a hybrid retriever falls back to dense-only. These fallbacks are usually *correct* engineering —
the request succeeds — and they are almost always invisible. There is no convention in this library
for recording that the system is now doing something weaker than it was asked to do.

Verified absent at `bfa8bc1`: no `degraded` attribute or event name exists anywhere in `genai_otel/`.
`genai_otel/instrumentors/base.py` contains no `add_event(` call at all, so there is no existing span-event
helper to extend. The only place the concept is even named is
`genai_otel/mcp_semconv/attributes.py:125,131`, where two commerce attributes are *commented* as
degradation signals — but that is a domain-specific mismatch check, not a general mechanism.

**Why it matters**

A published, well-regarded voice-RAG tutorial contains this exact line in its own logs:
`cloud turn detector failed (401 Unauthorized); falling back to local mini model`. The session
continued, the traces stayed green, and the author did not notice. That is the general shape of the
problem — a capability downgrade that produces no error, no latency change, and no cost change, and
therefore no signal in any conventional dashboard. A standard span event makes the downgrade a
first-class, queryable, alertable fact instead of a log line nobody reads. It is also cheap: one event
at the fallback site.

**Acceptance criteria**

- A documented span event named `gen_ai.degraded` with attributes:
  `gen_ai.degraded.component` (e.g. `embedding`, `llm`, `asr`, `tts`, `retriever`),
  `gen_ai.degraded.from`, `gen_ai.degraded.to`, `gen_ai.degraded.reason`,
  `gen_ai.degraded.recoverable` (bool).
- A public helper `BaseInstrumentor.record_degradation(span, ...)` for application code to emit the
  same event from its own fallback paths — sitting alongside the existing public helpers
  `add_embedding_attributes` and `add_retrieval_attributes` in `base.py`.
- Bundled instrumentors that implement a fallback path emit the event at the fallback site. Existing
  candidates found while surveying: `azure_openai_instrumentor.py:167` (falls back to the `text`
  attribute for completions), `evaluation/span_processor.py:93` (PII detector unavailable → regex
  fallback patterns) and `:564` (falls back to first message content),
  `litellm_span_enrichment_processor.py:235` (same first-message fallback).
- Documented guidance that the span status stays `OK`: a degradation is not an error, which is
  precisely why it needs its own event.

**Provenance**

Came out of TraceVerse chaos-lab voice-RAG work; this was the single highest-leverage gap found for
real-world incident detection.

---

## Dropped — verified already solved

### Dropped — "Flush spans on interpreter teardown (`atexit` / `force_flush` / `shutdown`)"

**Dropped. The capability exists, under a different name than the issue proposed.**

- `genai_otel.flush_telemetry(timeout_seconds: float = 5.0) -> bool` is defined at
  `genai_otel/auto_instrument.py:775` and publicly exported (`genai_otel/__init__.py:37` and `:143`).
  The issue asked for `genai_otel.flush(timeout_millis: int = 5000)`; only the name and unit differ.
- The atexit hook is not missing. `TracerProvider` is constructed at `auto_instrument.py:342` with the
  SDK default `shutdown_on_exit=True`, so the SDK's own atexit handler flushes the
  `BatchSpanProcessor`s added at lines 584 and 609 on clean exit *and* on an uncaught exception.
  `config.py:261` documents this explicitly.
- The real residual gap — termination by signal, which atexit does not cover — is already handled:
  `_install_sigterm_flush_handler` (`auto_instrument.py:794`), gated by `GENAI_FLUSH_ON_SIGTERM` with
  a bound from `GENAI_SIGTERM_FLUSH_TIMEOUT` (`config.py:270-278`). It chains rather than replaces an
  existing handler and re-raises so the exit status stays 143.
- `uninstrument()` (`auto_instrument.py:881`) shuts down both the tracer and meter providers.

Nothing left worth an issue. If anything, a docs pointer to `flush_telemetry` would close the gap.

### Dropped — "Bound the size and cardinality of `add_retrieval_attributes` document content"

**Dropped. The bounds already exist; the premise is factually wrong.**

`add_retrieval_attributes` (`genai_otel/instrumentors/base.py:2268`) already caps everything the issue
said was unbounded:

- `max_docs: int = 5` parameter, applied as `documents[:max_docs]`.
- Document content truncated to 500 chars; query truncated to 500 chars; metadata keys to 50 chars and
  metadata values to 200 chars.
- `retrieval.document_count` is set from `len(documents)` — the **true** count, not the truncated one.
  That was one of the issue's own acceptance criteria, already met.

The worked example in the issue ("`top_k = 20` produces 60 attributes ... spans into the megabytes")
does not happen: `top_k = 20` produces at most 5 documents × 3 fields, each ≤ 500 chars.

Residual nits, too small to file on their own: the caps are not wired to `OTelConfig` or environment
variables; there is no `content_truncated` marker; there is no `retrieval.documents_recorded`
companion to `retrieval.document_count`.

---

## Incidental findings (not filed)

- `BaseInstrumentor.add_embedding_attributes` (`base.py:2263`) gates vector capture on
  `hasattr(self.config, "capture_embedding_vectors")`, which is true whenever the attribute exists —
  including when it is set to `False`. `openai_instrumentor.py:563` does it correctly with
  `getattr(config, "capture_embedding_vectors", False)`. Also, `capture_embedding_vectors` is not
  declared on `OTelConfig` at all.
- `BaseInstrumentor.add_embedding_attributes` and `add_retrieval_attributes` have zero callers inside
  the package. `openai_instrumentor._add_embedding_content` reimplements the former inline.

## Verification basis

All claims above were checked against the working tree at **`v1.19.0-2-gbfa8bc1`** (HEAD `bfa8bc1`),
not against PyPI `genai-otel-instrument` 1.18.0. The difference is material for #23: OpenAI
embedding instrumentation landed in `408b4e0`, which is post-`v1.19.0` and therefore absent from
1.18.0.
