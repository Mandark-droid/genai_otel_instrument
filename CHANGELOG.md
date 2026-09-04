# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Coverage established by live testing, not assumption.** Each engine was run
  against real models across a matrix of entry point x streaming x
  batching x finish reason, which found four defects unit tests could not reach:
  a llama.cpp chat call emitted **two** spans (it delegates to
  `create_completion`) and so **double-counted tokens and cost**; vLLM streaming
  was not instrumented at all, because `LLM.generate` returns completed outputs
  and streaming goes through `AsyncLLM`; a vLLM batch that ended different ways
  reported only the first finish reason, hiding the truncated requests; and
  streamed spans on both engines carried no outcome at all. An SGLang
  instrumentor exists in the tree but is held back until it can be put through
  the same matrix.

  Streamed calls now carry what the engine actually reports -- full usage on
  vLLM (each yielded `RequestOutput` is cumulative), finish reason only on
  llama.cpp (it emits no `usage` block in stream mode). Token counts are never
  derived from chunk counts. Telemetry failures never interrupt a caller's
  stream.

- **Instrumentors for two self-hosted inference engines: vLLM and llama.cpp.**
  Both are instrumented at their **in-process Python APIs**
  (`vllm.LLM.generate` / `.chat` / `.encode` and the streaming `AsyncLLM`,
  `llama_cpp.Llama.create_chat_completion` / `create_completion`), not at an
  OpenAI-compatible HTTP endpoint. A served engine can already be traced by
  pointing OpenAI-SDK instrumentation at it, but offline batch generation never
  makes an HTTP request and is invisible to every HTTP-based approach.

  Where the engine reports it, spans carry the latency breakdown -- queue,
  prefill, decode, time-to-first-token, end-to-end -- under the
  `gen_ai.latency.*` keys that vLLM and SGLang independently converged on and
  that
  [semantic-conventions-genai#408](https://github.com/open-telemetry/semantic-conventions-genai/issues/408)
  proposes standardising, so an operator running vLLM's own OTLP tracing
  alongside this library gets one vocabulary rather than two to join.

  **Availability caveat, established by live testing rather than assumed:**
  vLLM's **V1 engine sets `RequestOutput.metrics` to `None`** and exposes no
  per-request timing on the Python API at all (verified against vLLM 0.24 and
  0.27). On any current vLLM the `gen_ai.latency.*` attributes are therefore
  **absent**; everything else on the span -- tokens, cost, finish reason,
  request id, batch size -- is unaffected. They are emitted wherever `metrics`
  is populated. No wall-clock substitute is invented in its place: the span
  duration already records end-to-end time honestly, and a fabricated "prefill"
  figure would be indistinguishable from a real engine measurement to any
  consumer that trusted it. vLLM's `num_cached_tokens` **is** populated on V1
  and is recorded as `gen_ai.usage.cache_read.input_tokens`.

  vLLM token counts are summed across a batch, since `generate()` returns one
  `RequestOutput` per prompt; for a batch the **slowest** request's timings are
  reported, because a batch is only as fast as its tail. vLLM reports
  `model_forward_time` in milliseconds while its other fields are seconds, and
  llama.cpp reports milliseconds throughout -- both normalised. SGLang's
  `cached_tokens` maps onto the conventions' `cache_read` concept. A phase an
  engine did not report is omitted rather than recorded as zero.

## [1.25.0] - 2026-09-04

### Added

- **Per-invocation agent budget governance** (`gen_ai.agent.token_budget`,
  `.token_budget.consumed`, `gen_ai.agent.iteration_budget`,
  `.iteration_budget.consumed`, plus a
  `gen_ai.invoke_agent.token_budget.utilization` histogram), implementing
  [semantic-conventions-genai#425](https://github.com/open-telemetry/semantic-conventions-genai/issues/425).
  Budgets are read from CrewAI (`max_iter` / `max_tokens`), LangGraph
  (`recursion_limit`), AutoGen and the OpenAI Agents SDK (`max_turns`), and
  Google ADK (`max_llm_calls`).

  Consumption is accumulated in a context-local frame rather than summed from
  child spans afterwards, because head sampling drops exactly those children --
  a sampled trace would otherwise report a runaway agent as having consumed
  nothing. Nested agents account separately, so a supervisor reports its own
  direct usage rather than its delegates', which is what its own budget governs.
  Budgets are never synthesised from an iteration limit times a per-call
  `max_tokens`, which #425 explicitly forbids. Known limitation: a call on a
  thread that does not carry the context is not counted.

- **`server.address` / `server.port` on spans.** Derived centrally from the SDK
  client's base URL, so every instrumentor using the shared span wrapper reports
  the endpoint a call actually went to. Both are conditionally required on
  inference spans upstream and the library previously emitted neither, which made
  self-hosted, proxied and gateway traffic indistinguishable from a vendor's
  public API. The attributes are omitted entirely when the SDK exposes no base
  URL: an absent attribute reads as "endpoint unknown", whereas defaulting to the
  provider's public host would misattribute exactly the traffic worth
  distinguishing. An instrumentor that sets these itself is never overridden.

- **Per-modality token breakdown.** `gen_ai.usage.{text,image,audio}.{input,output}_tokens`
  and `gen_ai.usage.{text,image,audio}.cache_read.input_tokens`, from
  [semantic-conventions-genai#440](https://github.com/open-telemetry/semantic-conventions-genai/pull/440).
  Instrumentors normalise provider shapes into flat usage keys and the span
  attributes are emitted from one place, as cache and reasoning tokens already
  were; the OpenAI instrumentor populates them from `prompt_tokens_details` and
  `completion_tokens_details`. Each value is a **subset** of the corresponding
  total, so consumers must not sum them alongside `gen_ai.usage.input_tokens`.
  A modality the provider did not report is omitted rather than emitted as zero.

- **Request parameters recorded centrally:** `gen_ai.request.seed`,
  `gen_ai.request.stream`, `gen_ai.request.top_k`, `gen_ai.request.choice.count`
  (from `n`, or Google's `candidate_count`) and `gen_ai.output.type` (derived
  from `response_format`). Only parameters the caller actually passed are
  recorded -- materialising provider defaults would report choices the
  application never made.

- **`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` is now honoured.** This
  is the content-capture switch shared with other OpenTelemetry GenAI
  instrumentations; previously only `GENAI_ENABLE_CONTENT_CAPTURE` worked, so an
  application migrating from one of them got no content and no indication why. It
  accepts `NO_CONTENT`, `SPAN_ONLY`, `EVENT_ONLY` and `SPAN_AND_EVENT`
  (case-insensitive) and takes precedence over `GENAI_ENABLE_CONTENT_CAPTURE`.
  `EVENT_ONLY` captures nothing here, because this library has a single boolean
  capture switch rather than separate span and event sinks. An unrecognised value
  warns and captures nothing rather than reading through to the other variable --
  a typo on a privacy switch should fail closed.

### Changed

- **`capture_embedding_vectors` is now a real configuration field with an
  environment variable (`GENAI_CAPTURE_EMBEDDING_VECTORS`).** It was read only
  via `getattr(config, "capture_embedding_vectors", False)` with no field
  declared anywhere, so despite being documented as a supported option it could
  be set only by assigning the attribute in Python, was invisible to
  `dataclasses.fields()`, and had no env var. Behaviour is unchanged: still off
  by default, and deliberately independent of `enable_content_capture`, because
  a 3072-dimension vector serialises to tens of kilobytes and one embeddings
  span would dwarf every other span in a trace.

- **Three attribute names now match the conventions, with the old spellings kept
  under `gen_ai/dup`.** Each was a name no conforming consumer looks for:
  `gen_ai.response.finish_reasons` (an array) supersedes the singular
  `gen_ai.response.finish_reason` -- the library previously disagreed with
  itself here, since several instrumentors already emitted the plural;
  `gen_ai.embeddings.dimension.count` supersedes `gen_ai.request.dimensions`;
  and `gen_ai.request.encoding_formats` (an array) supersedes the singular
  `gen_ai.request.encoding_format`.

- **`gen_ai.usage.cache_creation.input_tokens` is superseded by
  `gen_ai.usage.cache_write.input_tokens`.**
  [semantic-conventions-genai#440](https://github.com/open-telemetry/semantic-conventions-genai/pull/440)
  renamed it upstream; the library emitted only the old name, so a backend
  following the current conventions read zero cache-write tokens. Both names are
  now emitted under the default `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai/dup`, and
  only `cache_write` under `gen_ai` -- the same policy already used for
  prompt/completion tokens. `gen_ai.usage.cache_read.input_tokens` was not
  renamed and is unaffected.

## [1.24.0] - 2026-09-03

### Added

- **Spans now identify the host, the process and the service instance.** The
  resource previously carried `service.name` and the SDK's own `telemetry.sdk.*`
  keys and nothing else, so two processes of the same service - on one host or
  on fifty - produced byte-identical resources. Traffic could not be attributed
  to a machine or to one instance among several.

  The OpenTelemetry SDK already ships `host`, `os` and `process` detectors but
  runs none of them unless asked, so `OTEL_EXPERIMENTAL_RESOURCE_DETECTORS`
  now defaults to `host,os,process`. That adds `host.name`, `host.arch`,
  `os.type`, `os.version`, `process.pid`, `process.command`,
  `process.command_line`, `process.command_args`, `process.executable.*`,
  `process.owner` and `process.runtime.*`. `host.ip` has no upstream detector
  and is resolved by the library, excluding loopback interfaces as the
  convention requires.

  Every name is from the OpenTelemetry registry, and the control surface is the
  standard `OTEL_*` variables: set `OTEL_EXPERIMENTAL_RESOURCE_DETECTORS` to
  `host,os` to drop the process group, or `otel` to restore the previous
  minimal resource. `OTEL_RESOURCE_ATTRIBUTES` always wins over anything
  detected.

- **`service.instance.id` is always set.** Taken from `OTEL_SERVICE_INSTANCE_ID`
  or `OTEL_RESOURCE_ATTRIBUTES` when supplied, otherwise generated as a v4 UUID
  that is stable for the life of the process. `GENAI_SERVICE_INSTANCE_ID_MODE=derived`
  switches to a v5 UUID over the host name, service name and normalised startup
  arguments - under the namespace the specification designates for derived
  instance IDs - which survives a restart, at the cost of collapsing identical
  workers on one host onto a single ID. Prefer setting
  `OTEL_SERVICE_INSTANCE_ID` where the deployment can supply a real one.

### Security

- **Credential values in `process.command_args` are redacted.** A value
  following a flag whose name looks like a credential (`--password`,
  `--api-key`, `--token`, `--client-secret`, and similar) becomes
  `***REDACTED***`, in both the `--flag value` and `--flag=value` forms, as are
  inline `name=value` pairs and `scheme://user:password@host` URLs.
  `process.command_line` is rebuilt from the redacted vector so the two cannot
  disagree.

  This has to happen in the SDK. `process.command_args` is an array, so a flag
  and its value arrive as separate elements; once flattened, the value element
  carries no indication of which flag it belonged to and no downstream
  `name=value` rule can reconnect them. Redaction is pattern-based and cannot
  catch a credential passed under an unrecognised flag name, so
  `GENAI_PROFILE=strict|bfsi|bank` defaults the detector list to `host,os`,
  keeping command lines off the wire entirely while leaving host and instance
  identity intact.

### Fixed

- The OpenRouter example requested `nvidia/nemotron-3-nano-30b-a3b:free`, a
  listing that has since been withdrawn, so the example failed with a 404.
  It now requests `nvidia/nemotron-3.5-lightning:free` - still a free listing,
  verified to bill nothing - and carries a note on how to find a replacement
  if that one is withdrawn too. Its banner also no longer claims to be calling
  Claude 3.5 Sonnet while requesting a Nemotron model.

### Changed

- **Behaves identically across the supported `opentelemetry-sdk` range**, which
  changed twice underneath this feature. 1.44 added an `include_command_args`
  flag to `ProcessResourceDetector` and defaulted it off, so the detector loaded
  from the entry point stopped emitting `process.command_line` and
  `process.command_args` at all; they are now requested explicitly whenever the
  process detector is enabled. 1.44 also began auto-generating
  `service.instance.id`, which meant "the attribute is already set" could no
  longer distinguish an operator's choice from the SDK's, and
  `GENAI_SERVICE_INSTANCE_ID_MODE` was accepted and then silently ignored; the
  two standard sources are now read directly. Both were caught by end-to-end
  validation against a fresh install, not by the unit suite, which resolved an
  older SDK.

- Resource attributes now use their current registry names:
  `deployment.environment.name` (was a bare `environment` key, which was never a
  registry name) and `telemetry.distro.name` / `telemetry.distro.version` (were
  `telemetry.auto.*`, renamed upstream). The superseded spellings are still
  emitted alongside the current ones, as the GenAI token attributes are under
  `gen_ai/dup`, and will be removed at 2.0.

## [1.23.1] - 2026-09-03

> Merged to `main` but never published as its own release; these changes ship in 1.24.0.

### Added

- **August 2026 pricing sweep gap-fill: DeepSeek V4 Flash Vision Exp.**
  Re-ran the August 2026 first-party/hyperscaler model scan against
  models.dev (the sweep shipped in 1.22.1/1.23.0 covered nine of the ten
  in-scope releases; this one was missed):

  | Model | Provider | Input / 1M | Output / 1M | Released |
  |---|---|---|---|---|
  | DeepSeek V4 Flash Vision Exp | DeepSeek | $0.14 | $0.28 | 2026-08-21 |

  Also priced: reasoning $0.28/1M, cache read $0.0028/1M. Added the
  `deepseek/v4-flash-vision-exp` provider-prefixed alias mirroring the
  `deepseek-v4-flash` sibling.

### Fixed

- **`deepseek-v4-flash-vision-exp` had no pricing key**, so it fell through
  to the shorter `deepseek-v4-flash` sibling by longest-substring match.
  The prompt/completion rate happened to be identical, so this was not a
  mis-bill, but the model had no reasoning or cache-read price at all
  (both silently $0). It now has its own explicit key with the full rate
  card.

### Deferred

- No other August 2026 first-party/hyperscaler releases were found without
  a reliable published price, or missing from the table, in this re-scan.

### Pricing data sources

Re-checked for this gap-fill sweep: models.dev, cross-referenced against
`api-docs.deepseek.com/quick_start/pricing` (first-party, wins on conflict).

## [1.23.0] - 2026-09-03

### Fixed

- **Model names no longer match a pricing key in the middle of a word.** Both
  resolution paths used plain substring containment, so a short token matched
  inside an unrelated name and billed that model's price:

  | Requested model | Matched | Was billed | Should be |
  |---|---|---|---|
  | `Sao10K/L3-8B-Stheno-v3.2` | `o1`, inside `sa`**`o1`**`0k` | $15/1M | ~$0.10/1M |
  | `google/veo3.1` (a video model) | `o3` | $2/1M | unpriced |
  | `deepgram/nova-2-automotive` | `auto` | $0.85/1M | unpriced |
  | `openai-gpt-52` | `gpt-5` | GPT-5 rate | unpriced |
  | `spacexai/grok-4.20-*` | `xai/grok-4`, inside `spacex`**`ai/`** | Grok 4 rate | Grok 4.20 |

  The same flaw sat in the local-model fallback: the generic size word `mini` is
  inside `ge`**`mini`**, so **96 Gemini ids with no pricing entry** were read as
  0.02B local models and given a fabricated price. A fabricated price is worse
  than a missing one, because downstream it is indistinguishable from a real
  figure and the error never surfaces.

  A match must now be flanked by a separator or the string edge. Verified across
  a 7,395-name corpus (pricing keys + models.dev + LiteLLM): 98 resolutions
  change, all corrections. Names in families that spell a decimal point as `p`
  (`glm-5p3`, `deepseek-v3p1`, `minimax-m2p1`) move from a wrong price to no
  price pending an alias backfill.

- **Bedrock and Vertex regional endpoints were under-billed by 10%.** Anthropic
  prices regional and multi-region endpoints at a 10% premium over the global
  endpoint, for Claude Sonnet 4.5, Haiku 4.5, Opus 4.5 and every later model.
  The table applied that only to `eu.`, so US, AU and JP callers were charged
  the global rate:

  | Model | Regions affected | Was | Now |
  |---|---|---|---|
  | Claude Opus 5 | us, au, jp | $5.00 / $25.00 per 1M | $5.50 / $27.50 |
  | Claude Sonnet 5 | us, au, jp | $2.00 / $10.00 per 1M | $2.20 / $11.00 |
  | Claude Sonnet 4.6 | us, jp | $3.00 / $15.00 per 1M | $3.30 / $16.50 |

  Cache read and write prices are scaled too, since the multiplier applies to
  every token category. Eight `us.`/`eu.`/`au.`/`jp.` dot-form aliases for Sonnet
  5 and Sonnet 4.6 were missing entirely and resolved to the base price by
  substring fallback, which under-billed by the same 10%; they are now explicit.
  A parametrized invariant test asserts every regional alias is exactly 1.1x its
  global counterpart, so a new family cannot repeat this.

### Added

- **Claude Fable 5.1** (released 2026-09-01) - input $10/1M, output $50/1M.
  Cache hits are **$0.25/1M**, a 0.025x multiplier rather than the 0.1x every
  other model uses, so a shared `cacheReadPrice` would have over-billed cached
  reads four-fold. 5-minute cache writes are $12.50/1M. Fifteen aliases: bare,
  dotted, `@default`, the Bedrock `anthropic.` forms, and the `us`/`eu`/`au`/`jp`
  regional profiles at the documented **10% premium** ($11/$55) over global.
  Source: platform.claude.com/docs/en/about-claude/pricing

- **Gemini 3.8 Flash** (released 2026-09-02) - input $0.75/1M, output $3.75/1M,
  with dashed and `gemini/`-prefixed aliases. Previously unpriced, which meant a
  1M-in/200k-out call was billed at $0.14 instead of $1.50. Source: models.dev.

  Both models shipped after the August sweep's window and would not have been
  picked up until the October refresh.

### Changed

- The documented provider count is now **23** and is enforced by a test.
  `docs/index.md` claimed 19+, `docs/guides/llm-providers.md` and `README.md`
  claimed 21+, and the table itself listed 17. The six providers that were
  instrumented but undocumented - Azure AI Inference, Anyscale, Liquid Audio,
  HuggingFace Transformers, Sentence Transformers and Hyperbolic - now have
  rows, and `tests/test_docs_provider_coverage.py` fails if the registry and the
  docs drift apart again.
## [1.22.1] - 2026-09-03

> Merged to `main` but never published as its own release; these changes ship in 1.23.0.

### Added

- **Monthly pricing sweep for August 2026 releases.** Cross-checked
  models.dev against first-party vendor docs for every first-party /
  hyperscaler model released in the target month:

  | Model | Provider | Input / 1M | Output / 1M | Released |
  |---|---|---|---|---|
  | GLM-5.3 | Zhipu / Z.AI | $1.40 | $4.40 | 2026-08-14 |
  | GLM-5.3-Flash | Zhipu / Z.AI | $0.075 | $0.25 | 2026-08-26 |
  | Grok 4.6 | xAI | $2.00 | $6.00 | 2026-08-12 |
  | Qwen3.8 Flash | Alibaba | $0.15 | $0.47 | 2026-08-26 |
  | Qwen3.8 27B | Open-weight (deepinfra/huggingface) | $0.40 | $3.00 | 2026-08-14 |
  | Gemini 3.7 Flash | Google | $0.75 | $3.75 | 2026-08-13 |
  | Sakana Namazu | Sakana AI | $0.95 | $4.00 | 2026-08-03 |
  | Solar Pro 4 | Upstage | $0.30 | $1.20 | 2026-08-06 |
  | Grok Imagine Image 2.0 | xAI (image, flat per-image) | $0.04/image | - | 2026-08-07 |

  Each model also received provider-prefixed, dashed-version, and (where
  applicable) HuggingFace/Bedrock alias keys, mirroring sibling entries in the
  same family.

### Fixed

- **GLM-5.3, GLM-5.3-Flash, and Grok 4.6 were silently mis-billed.** None of
  the three had an explicit pricing key, so name lookups fell through to the
  longest matching *shorter* sibling: any `glm-5.3*` id resolved to the Feb
  2026 `glm-5` entry (billing GLM-5.3 at $1.00/1M input instead of $1.40, and
  GLM-5.3-Flash at $1.00/1M instead of its real $0.075/1M), and `grok-4.6`
  resolved to `grok-4` (billing at $3.00/1M input instead of the real
  $2.00/1M). Both families now have explicit keys so their own price wins the
  longest-substring match instead of the older sibling.
- **`gemini-flash-latest` carried a stale price.** Google repointed the
  `-latest` alias from the 2026-05-19 flash release to Gemini 3.7 Flash on
  2026-08-13; the entry still carried the old release's rate ($1.50/1M
  input), double the new target's real $0.75/1M. Updated to track the
  current target.

### Deferred

- **Grok Imagine Image 2.0**: models.dev lists no `cost` for this model;
  priced instead from xAI's own pricing page
  (`docs.x.ai/developers/pricing`, flat $0.04/image).
- No other August 2026 first-party/hyperscaler releases were found without a
  reliable published price this sweep.

### Pricing data sources

Refreshed for this sweep. The `genai_otel/llm_pricing.json` database draws
from, in order of precedence (first-party always wins on conflict):

- **First-party provider docs**: Anthropic (`platform.claude.com`), OpenAI,
  Google (`ai.google.dev`), Moonshot AI (`platform.moonshot.ai`), Xiaomi
  (`platform.xiaomimimo.com`), Zhipu / Z.AI (`docs.z.ai`), Alibaba DashScope,
  DeepSeek (`api-docs.deepseek.com`), MiniMax (`platform.minimax.io`), xAI
  (`docs.x.ai`), Cohere, Nvidia, Sakana AI, Upstage, Meituan LongCat
  (`longcatai.org`).
- **Hyperscalers**: AWS Bedrock, Azure AI Foundry, Google Vertex AI.
- **Open-weight host consensus**: when a model has no first-party API SKU
  (e.g. Qwen3.8 27B), a rate agreed on by two or more allowed hosting
  providers (deepinfra, huggingface, fireworks-ai, togetherai, baseten,
  cloudflare-workers-ai, nebius, groq) is used instead of a single aggregator.
- **Aggregators / cross-checks**: OpenRouter, LiteLLM, Artificial Analysis.
- **models.dev** (`https://models.dev/api.json`).

## [1.22.0] - 2026-08-26

### Added

- **The AWS Bedrock Converse API is now traced** (#27). `aws_bedrock_instrumentor`
  wrapped `invoke_model` and nothing else, so `converse` and `converse_stream` --
  the unified API AWS points callers at, and the practical path for every
  non-Anthropic model -- produced **no span at all**. `invoke_model_with_response_stream`
  was unwrapped too, so even the covered call went dark as soon as a caller streamed.
  Same silent-empty failure mode as #26: the instrumentor loaded, reported success,
  and captured nothing.

  All four runtime calls are now wrapped. Converse differs from `invoke_model` in
  every field the instrumentation reads, and each is mapped onto the existing
  semantic conventions so Bedrock spans stay comparable with every other provider:

  | Converse | Note |
  |---|---|
  | `messages[].content[]` | typed blocks (`text`, `image`, `toolUse`, `toolResult`), not strings |
  | `system` | a top-level parameter, **not** a message role -- it no longer inflates the message count |
  | `inferenceConfig.{maxTokens, temperature, topP, stopSequences}` | mapped to `gen_ai.request.*` |
  | `output.message.content[]`, `stopReason` | completion, finish reason and tool calls |
  | `usage.{inputTokens, outputTokens, totalTokens}` | camelCase, and top-level rather than inside a JSON body |

  Being model-agnostic, Converse needs none of the per-model body parsing
  `invoke_model` requires.

### Fixed

- **A Converse response was priced at zero.** `_extract_usage` gated on
  `contentType` and a JSON `body`, neither of which a Converse response has, so it
  returned `None` and the span was treated as having no usage rather than being
  flagged. Converse token counts are now read from the top-level camelCase `usage`.

- **`converse_stream` spans closed before the model had generated anything.**
  Bedrock returns `{"stream": ...}` immediately and generates while the caller
  iterates, and there is no `stream=True` keyword for the generic wrapper to key
  on. The event stream is now wrapped so the span stays open until it is
  exhausted, reporting real latency and picking up token counts from the trailing
  `metadata` event. Sampled-out spans are handled explicitly -- a `NonRecordingSpan`
  has no `.name`, which the measurement path reads.

### Removed

- **`.idea/` untracked and stripped from all git history.** The directory had been
  listed in `.gitignore` since long before the files were committed, but gitignore
  does not apply to paths already in the index, so seven IntelliJ config files stayed
  tracked. They are now removed from the index (local IDE settings are untouched) and
  purged from history with `git-filter-repo`.

  `.idea/` entered history in October 2025, so this **rewrites the commit SHAs of
  every tag in the repository -- `v0.1.0` through `v1.21.0`**, a wider range than the
  `docs/proposals` purge in 1.21.0, which stopped at `v1.1.0`. Anyone pinned to a git
  SHA, or consuming a GitHub release tarball from any version, must re-pin.
  **PyPI wheels and sdists are unaffected** and need no action -- published artifacts
  already excluded `.idea` via `MANIFEST.in`.

## [1.21.0] - 2026-08-26

### Added

- **The OpenAI Responses API is now traced** (#26). `openai_instrumentor` wrapped
  `chat.completions.create` and `embeddings.create` only, so any caller on
  `client.responses.create` produced **no LLM span at all** -- no model, no tokens,
  no cost, no tool calls. The instrumentor loaded, reported success, and captured
  nothing. This is not an edge case: Chat Completions rejects function tools combined
  with reasoning on GPT-5.6+, so agent runtimes on native OpenAI models route through
  `/v1/responses` by default.

  `responses.create` is now wrapped on both sync and async clients as an
  `openai.responses` span. The Responses shape differs from Chat Completions in every
  field the instrumentation reads, and each is mapped onto the existing semantic
  conventions so the two stay comparable:

  | Responses | Chat Completions |
  |---|---|
  | `input` (string or list), `instructions` | `messages` |
  | `max_output_tokens` | `max_tokens` |
  | `output[]` items | `choices[]` |
  | `usage.input_tokens` / `output_tokens` | `usage.prompt_tokens` / `completion_tokens` |
  | `status` / `incomplete_details.reason` | `choices[].finish_reason` |

  Reasoning tokens are attributed as output because that is how they are billed;
  cached prompt tokens are recorded under the same canonical key as Anthropic's cache
  reads; `response.id` is captured so `store=true` responses stay joinable. Streaming
  reports TTFT and inter-token latency on the same terms as every other provider.

### Fixed

- **A Responses call was recorded as an embedding.** Content capture routed to the
  embeddings path whenever a request carried `input` and no `messages` -- but the
  Responses API keys on `input` too. The completion was dropped and the span was
  labelled `embedding.model_name`. Routing now decides on the *response* shape, which
  is unambiguous, rather than on a request key both APIs share.

### Removed

- **`docs/proposals/` removed and stripped from all git history.** The folder held
  pre-submission drafts of an upstream OpenTelemetry semantic-conventions proposal.
  That proposal has since been filed and is tracked upstream as
  [semantic-conventions#3672](https://github.com/open-telemetry/semantic-conventions/issues/3672),
  so the local drafts were a stale, misleading second copy. They were also being
  published to the docs site and bundled into every sdist since 1.1.1; `MANIFEST.in`
  now prunes the path so it cannot ship again.

  The drafts were removed from history with `git-filter-repo`, which **rewrites the
  commit SHAs of every tag from `v1.1.0` onward**. Anyone pinned to a git SHA or
  consuming a GitHub release tarball from that range must re-pin; PyPI wheels and
  sdists are unaffected. Documentation now links to the upstream issue instead.

### Fixed

- **A misconfigured carbon country code silently produced a global average instead of a
  regional measurement.** codecarbon's offline dataset is keyed by ALPHA-3 ISO codes. Given
  the intuitive 2-letter form it logs `Does not support country with ISO code IN` and then
  emits **475.0 gCO2e/kWh** — byte-identical to this library's own manual fallback constant,
  and still labelled `source: codecarbon`. Measured on codecarbon 3.2.8:

  | code | applied factor |
  |---|---:|
  | `IN` (unsupported) | 475.0 gCO2e/kWh |
  | `IND` | **713.4** gCO2e/kWh |
  | `USA` | 369.5 gCO2e/kWh |
  | `FRA` | 56.0 gCO2e/kWh |

  An operator setting `IN` therefore got numbers indistinguishable from having codecarbon
  switched off, plus an error log that reads as noise. `normalize_country_iso_code()` now
  repairs the common alpha-2 case (`IN` → `IND`) and **refuses** anything codecarbon cannot
  resolve, validated against codecarbon's own `global_energy_mix.json` (213 countries)
  rather than a duplicate list that would drift. A misconfigured code now disables the
  integration with an explicit error instead of degrading it invisibly.

- The unset-country default now warns rather than logging at debug: falling back to `USA`
  (369.5 gCO2e/kWh) is almost always wrong and previously happened silently.

## [1.20.2] - 2026-08-20

### Added

- OpenRouter and CometAPI clients now trace embedding calls too. Both are
  OpenAI-compatible aggregators with a real `/v1/embeddings` endpoint, but
  their dedicated instrumentors only wrapped `chat.completions.create`; since
  the generic OpenAI instrumentor deliberately skips clients claimed by an
  aggregator (to avoid double-instrumenting), embedding calls through either
  produced no span at all. Follows the same `gen_ai.request.type=embedding`
  contract as every other provider: input counting, response
  embedding_count/vector_size, and content/vector capture gated the same way.

### Fixed

- The sdist no longer bundles internal planning documents or IDE
  configuration (`.idea/`). `setuptools_scm`'s file-finder adds every
  git-tracked file to the sdist by default, which silently overrides
  `MANIFEST.in`'s intended minimal file list; `MANIFEST.in` now carries
  explicit `exclude`/`prune` rules for the files that should never have
  shipped. v1.20.0 and v1.20.1 were pulled from PyPI for this reason - please
  upgrade directly to this version.

## [1.20.1] - 2026-08-20

### Added

- Closed the last provider gap from upstream issue #23: Replicate embedding
  models (BGE, E5, GTE, MPNet, MiniLM families, or any model reference
  containing "embed") are now classified and traced as `embeddings` calls,
  with input counting and response vector-size/embedding-count attributes.
  Replicate's generic `run()` has no fixed schema, so non-embedding models
  are traced exactly as before - a plain `replicate.run` span.
- The shared embedding content-capture fallback now unwraps Replicate's
  nested `input={"text": ...}` payload shape, so embedding text capture
  (under `GENAI_ENABLE_CONTENT_CAPTURE`) works for Replicate too.

## [1.20.0] - 2026-08-20

### Added

- Closed the provider-coverage slice for embedding telemetry tracked by
  upstream issue #23: Cohere, Google GenAI, Ollama, Together, Bedrock,
  LiteLLM, Azure AI Inference, Hugging Face feature extraction,
  SentenceTransformers, and shared embedding response dimensions now use the
  common `embeddings` contract.
- Added additive retrieval-quality attributes and a public helper, including
  score distributions from vector responses and async Qdrant query coverage.
- Added ASR attributes for Hugging Face pipelines and direct audio model
  generation, an optional Liquid Audio instrumentor, and shared TTS fields for
  Sarvam and ElevenLabs streaming paths.
- Added the public `gen_ai.degraded` span-event helper and documentation for
  retrieval, speech, audio, and degradation telemetry.

### Fixed

- Embedding content and vector capture now honor the actual configuration
  flags instead of treating an unset vector-capture attribute as enabled.
- Bedrock embedding usage extraction handles mapping-shaped usage payloads.
- Bedrock Titan Text generation calls were misclassified as embeddings
  because the heuristic keyed on the shared `inputText` body field; it now
  decides via the model-ID family so mainstream chat calls no longer get
  priced against the embeddings table.
- Azure AI Inference embedding calls resolved to no pricing entry at all
  (silently billed as $0) because the pricing table only had the
  `azure_ai/`-prefixed catalog names, not the bare deployment model name the
  SDK actually reports.
- `capture_embedding_vectors` behaved inconsistently across providers after
  this release's refactor tightened one call site to a strict boolean check;
  it's back to a plain truthy check everywhere.
- SentenceTransformers `encode()` on a single string reported
  `embedding_count` as the vector dimension instead of `1`.
- Weaviate query spans never got `rag.result.score_*` attributes because the
  score extractor didn't match Weaviate's actual GraphQL response shape.

## [1.19.0] - 2026-08-17

### Added

- **OpenAI embeddings are instrumented.** `client.embeddings.create` produced no
  span at all: only `chat.completions.create` was wrapped, so a
  retrieval-augmented call was traced as its generation half alone. The lookup
  that selected the context was invisible, and its tokens and cost went
  unrecorded even though the embeddings pricing table was already present.

  Both the sync and async clients are covered. Embedding spans are named
  `openai.embeddings` and carry `gen_ai.operation.name=embeddings`,
  `gen_ai.request.type=embedding`, `gen_ai.request.input_count`,
  `gen_ai.response.embedding_count` and `gen_ai.response.vector_size`, plus the
  usual token and cost attributes priced against the `embeddings` table rather
  than the chat one. Batched requests count each text; pre-tokenised input
  counts as one.

  The embedded text is recorded as `embedding.text` only when
  `GENAI_ENABLE_CONTENT_CAPTURE=true`, and is truncated by
  `GENAI_CONTENT_MAX_LENGTH` like any other captured content - retrieval inputs
  routinely carry user data. Vectors remain off unless
  `capture_embedding_vectors` is set, since they would dominate span size.

  Clients pointed at an aggregator are skipped exactly as they are for chat, so
  no duplicate spans or double-counted cost.

### Fixed

- **Hyperbolic reported cost under an attribute nothing else uses.** It set
  `gen_ai.cost.amount` while every other instrumentor sets
  `gen_ai.usage.cost.total`, so Hyperbolic spend was missing from any
  cross-provider cost query. It now emits the standard attribute, keeping the
  legacy one alongside for dashboards already built on it. The span attribute is
  also no longer skipped when no cost metric counter is configured, which
  previously dropped cost from the span as well as the metric.

### Documentation

- **Embeddings and RAG guidance** in the provider guide and the semantic
  conventions reference, including the full embedding attribute list.

- **Corrected the documented cost attribute.** The reference and provider guide
  both presented `gen_ai.cost.amount` as the cost attribute; it is
  `gen_ai.usage.cost.total`, and `gen_ai.cost.amount` was only ever emitted by
  Hyperbolic.

- **Arize AX guide** (`docs/guides/arize-ax.md`). The export path was only
  documented in `examples/arize_ax/README.md`, which is not on the docs site.
  Covers the four-variable setup, the regional endpoints, the two
  misconfigurations that silently drop spans (a path on the endpoint, a missing
  project-name resource attribute), how AX normalises `gen_ai.*` onto its
  OpenInference model, and how to verify a trace landed.

- **Corrected the documented default for `OTEL_SEMCONV_STABILITY_OPT_IN`.**
  Configuration reference stated `gen_ai`; the actual default has been
  `gen_ai/dup` since dual emission was made the default, and the surrounding
  guidance still told readers to opt into a setting they already had.

## [1.18.0] - 2026-08-15

### Fixed

- **Streamed calls made through a raw-response wrapper reported no latency,
  tokens or cost.** Resolves #22.

  A caller that wants the provider's response headers reaches the OpenAI SDK
  through `with_raw_response.create`, which returns a `LegacyAPIResponse` /
  `AsyncAPIResponse` instead of the stream. The stream only exists once
  `.parse()` is called, so at await time there was nothing iterable to detect
  and a streamed call was indistinguishable from a buffered one. Streaming
  detection now defers to `.parse()`, handling both its sync and async forms.

  litellm takes exactly this path, so `litellm.acompletion(stream=True)` against
  OpenAI, Azure and OpenAI-compatible endpoints now reports
  `gen_ai.server.time_to_first_token`, `gen_ai.server.time_per_output_token`,
  token usage and cost. All four were missing before, not just the latency.

### Added

- **`litellm_latency`, an opt-in instrumentor for litellm routes that bypass
  provider SDKs.**

  litellm sends OpenAI, Azure and OpenAI-compatible traffic through the OpenAI
  SDK, which is already instrumented. Every other provider (Anthropic, Bedrock,
  Gemini, Cohere, HuggingFace, ...) is implemented with litellm's own HTTP
  client, which no provider instrumentor ever sees, so streaming latency was
  absent for that whole set regardless of the fix above. Wrapping litellm's own
  entry points catches every route, because litellm returns a
  `CustomStreamWrapper` whatever the transport underneath.

  The span it creates is the parent of any inner provider span. **When an inner
  span already measured the request, this one records no tokens, cost or latency
  of its own** -- `litellm.acompletion` re-enters `litellm.completion`
  internally, so without that rule a single request would be billed more than
  once.

  **Opt-in, not on by default.** It participates in token and cost accounting
  for every litellm call, so it earns default-on status only after a release of
  real-world use:

  ```bash
  export GENAI_ENABLED_INSTRUMENTORS="openai,anthropic,litellm,litellm_latency"
  ```

  Verified end to end against live endpoints on all four combinations of
  provider-SDK vs own-HTTP-client and streamed vs non-streamed, asserting that
  exactly one span carries cost per request and that non-streamed calls carry no
  TTFT at all.

## [1.17.1] - 2026-08-15

### Fixed

- **The demo OpenSearch pipeline could not chart the new streaming-latency
  fields.** `examples/demo/opensearch-setup.sh` goes to v2.7.

  1.17.0 emits `gen_ai.server.time_to_first_token` and
  `gen_ai.server.time_per_output_token`, but those arrive under `tag.*`, which
  the index template maps to `keyword` via the blanket dynamic template
  inherited from Jaeger. A keyword TTFT can be displayed but never averaged or
  charted — the same trap that hid audio usage until v2.6.

  Both are now promoted to typed top-level fields
  (`gen_ai_server_time_to_first_token`, `gen_ai_server_time_per_output_token`,
  both `double`), along with `gen_ai_streaming_tpot_unavailable_reason`
  (`keyword`) so "not measured" stays distinguishable from "nobody looked".
  `gen_ai_server_ttft` — mapped since v2.5 but never populated — is now also
  back-filled from the semconv name.

  Verified with `_ingest/pipeline/_simulate` against both a streamed span
  carrying usage and one without: the streamed span yields numeric TTFT and
  TPOT, and the usage-less span yields TTFT plus the reason and no TPOT.

  Note that this script provisions a demo from scratch: it `PUT`s
  `genai-ingest-pipeline` wholesale, so it is not a safe upgrade path for a
  pipeline that has been modified since it was created. Add the new field
  promotions to your own pipeline instead. Chartability is worth confirming
  with an `avg` aggregation, which errors on a `keyword` field and is the only
  unambiguous test.

  This is a demo-infrastructure change. No module under `genai_otel/` differs
  from 1.17.0, so the installed library is unchanged; the distribution differs
  only by the bundled `CHANGELOG.md`.

## [1.17.0] - 2026-08-15

### Added

- **Streaming spans now carry time-to-first-token and time-per-output-token
  under the OTel GenAI semantic-convention names.** Resolves #21.

  Streamed calls now set:

  | Attribute | Meaning |
  |-----------|---------|
  | `gen_ai.server.time_to_first_token` | Seconds to the first streamed chunk |
  | `gen_ai.server.time_per_output_token` | `(duration - ttft) / max(output_tokens - 1, 1)` |

  Both are also recorded as histograms of the same names. The library's older
  `gen_ai.server.ttft` spelling is still emitted, so nothing reading it breaks.

  **Absent, never zero.** Non-streamed calls carry neither attribute. TPOT
  additionally requires a real output-token count, so when a provider sends no
  usage in the stream the span gets TTFT plus
  `gen_ai.streaming.tpot_unavailable_reason=output_token_count_unavailable` and
  no TPOT — the chunk count is deliberately *not* used as a stand-in, since
  chunks are not tokens. A zero TTFT is indistinguishable from an instant first
  token and quietly drags down any average it enters; an absent attribute can be
  reported as "not measured".

### Fixed

- **Async streaming calls were mistimed and lost their token usage entirely.**
  Awaiting an async streaming call returns the stream object, not the answer —
  the model does its work while the caller iterates. The span was closed at the
  `await`, so it recorded the handshake (~0.7s in testing) as the whole call and
  never saw the final usage chunk, leaving async streamed requests with no
  tokens, no cost and no TTFT. The span is now handed to the streaming wrapper
  and closed when iteration finishes.

- **Providers returning a bare generator for `stream=True` were timed but never
  measured.** Streaming detection now runs before the generic generator
  handling in `create_span_wrapper`, so these calls get TTFT, end-of-stream
  usage and cost like any other stream.

- **Groq, Azure OpenAI and Sarvam chat completions ignored streaming
  altogether.** Each built its own span with `start_as_current_span`, which
  closed it the moment the SDK handed back an iterator. All three now route
  through the shared streaming wrapper.

- **A stream abandoned part-way through (a `break`, or a timeout) leaked its
  span.** The span is now closed with whatever was measured instead of being
  left open with the running-requests counter still incremented.

## [1.16.1] - 2026-08-14

### Fixed

- **The demo OpenSearch pipeline could not chart audio usage, and silently
  dropped telemetry from clients on current semantic-convention names.**
  `examples/demo/opensearch-setup.sh` is updated to v2.6.

  Audio duration and character counts only ever reached OpenSearch under
  `tag.gen_ai@usage@*`, which the index template maps to `keyword` via a blanket
  `tag.*` dynamic template inherited from Jaeger. Keyword fields cannot be summed
  or averaged, so audio seconds could be displayed but never charted. They are
  now promoted to `gen_ai_usage_audio_duration_seconds` (float) and
  `gen_ai_usage_characters` (long), the same way cost already was.

  More seriously, the extractor read only `gen_ai.system`. That attribute was
  superseded upstream by `gen_ai.provider.name`, which 1.10.0 began emitting -
  so a client running `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai` for current-only
  names recorded **no provider at all**. Dual emission, the default since 1.10.0,
  is the only reason this was not already visible. `gen_ai.provider.name` is now
  read, and falls back to populating `gen_ai_system` when the superseded tag is
  absent.

  Verified with `_ingest/pipeline/_simulate` against a span carrying only the
  current names: `gen_ai_system` is populated from `gen_ai.provider.name`, and
  both audio fields come through.

  This is a demo-infrastructure change; the published package is unaffected.

## [1.16.0] - 2026-08-14

### Added

- **`prices_checked` records when a price was last confirmed against the
  vendor's own page**, and `CostCalculator.price_checked(model, call_type)`
  reads it back.

  Most of the table is inherited from an upstream aggregate. That is a reasonable
  starting point and a poor source of truth: this month's audit found rates stale
  by a full model generation (`deepgram/nova-3` carrying Nova-2 pricing),
  transposed between tiers (AssemblyAI `best` cheaper than `nano`), and off by
  2.5x (Fireworks Whisper). None of them looked wrong in the file, because
  nothing recorded whether anyone had ever checked.

  **51 entries are stamped - 2.9% of 1756 priced entries.** That number is the
  point. Only rows actually opened against a vendor page this month carry a date;
  stamping the rest with today's would have made the audit worse than having no
  dates at all. `None` means "never verified here", not "suspect".

- **`CostCalculator.stale_prices(older_than_days=180)`** turns the audit into a
  query, returning `(pricing_key, date_or_None)` for entries due a re-check. The
  unverified set is included by default and is the large one - reporting only
  aged entries would imply the remainder are fine, when most have simply never
  been looked at.

  Both `prices_checked` and `deprecated` are metadata registries keyed by pricing
  key, kept out of the model index so a `call_type` matching either name cannot
  resolve model names against it.

  The idea is borrowed from [pydantic/genai-prices](https://github.com/pydantic/genai-prices),
  whose provider YAML carries a `prices_checked:` date per model.

## [1.15.0] - 2026-08-14

### Added

- **Audio prices declare their unit.** A bare number cannot say what it is *per*.
  Text-to-speech bills per character, transcription per second, and some audio
  models bill per token - but all three were stored as an undifferentiated float,
  with the unit inferred from whatever the caller happened to pass.

  That inference is what allowed 42 entries to sit at a per-minute rate against a
  per-second contract for months. Nothing in the data contradicted it, because
  nothing in the data said what the number meant. Entries now carry the unit in
  the key:

  ```jsonc
  "eleven_multilingual_v2":  { "per_1k_chars":  0.10     },
  "elevenlabs/scribe_v1":    { "per_second":    6.11e-05 },
  "gpt-4o-transcribe":       { "per_1k_tokens": 0.0025   }
  ```

  Billing a per-second model by character is now **refused with a warning**
  rather than silently returning a plausible number. Guessing a conversion is
  precisely how a 1000x error looks reasonable.

  104 of the 111 audio entries are migrated, each unit taken from the upstream
  price list's own field or from a vendor page checked during the 1.11.x audit.
  The seven left as bare numbers are vendors whose billing unit was never
  established - Cartesia, PlayHT, Hume, and one Gemini live-audio model. They are
  named in a test, so the set cannot grow silently, and guessing was not on the
  table.

- **`per_1k_tokens` is now a supported audio unit.** 23 entries are token-billed
  (`gpt-4o-transcribe`, `gpt-4o-audio-preview`, the Gemini TTS family), and the
  calculator previously understood only characters and seconds - so those could
  not be priced correctly by any caller.

  Bare numbers are still accepted, for backwards compatibility and for
  user-supplied custom pricing, and keep the previous inferred-unit behaviour.

  The idea is borrowed from [pydantic/genai-prices](https://github.com/pydantic/genai-prices),
  whose schema encodes units in key suffixes (`_mtok`, `_kcount`, `_mchars`) and
  whose contributor guide warns that a per-Mtok figure under a `_kcount` key "is
  valid YAML and wrong by 1000x". That is the same failure, described by someone
  who had evidently met it too.

## [1.14.1] - 2026-08-14

### Fixed

- **The package-size test measured build artifacts rather than the package.** It
  walked the directory containing `genai_otel/__init__.py`, which under an
  editable install is the working tree - where `__pycache__` accumulates a full
  set of bytecode for every interpreter version used. That reported 5781 KB
  against a 5120 KB limit and had been failing for some time. Compiled bytecode
  is now excluded, which is what wheels do: the published 1.14.0 wheel contains
  **zero** `.pyc` bytes, and its `genai_otel/` is 1592 KB. The test now measures
  1626 KB, so the limit has roughly 3x headroom and still catches real growth.

  No packaging change - the distribution was never oversized. The measurement was
  wrong.

- **A test inherited evaluation settings from the developer's shell.**
  `test_setup_enables_all_components` asserts the span exporter is exactly a
  `CostEnrichingSpanExporter`, but enabling any evaluation feature wraps another
  exporter around it. Anyone with `GENAI_ENABLE_PII_DETECTION` (or bias,
  toxicity, hallucination, prompt-injection, restricted-topics) exported saw this
  fail while CI passed. Those flags are now pinned in the test's config.

  With both fixed the suite is green in a clean environment and in one with every
  evaluation feature exported - 1837 passed, 0 failed. Neither of the two
  long-standing failures was a product defect.

## [1.14.0] - 2026-08-14

### Fixed

- **A clean first run no longer looks like a broken one.** `pip install` followed
  by the two lines from the README printed a page of warnings and errors, then a
  repeating wall of connection stack traces. Everything worked - exit code 0 -
  but it read as a broken library, which is a poor thing to hand someone in the
  first thirty seconds. A default install now prints one actionable line.

- **Six of those were a real logic slip, not cosmetic.** The MCP instrumentors
  set their class to `None` when the optional package is absent, so calling it
  raises `TypeError` rather than `ImportError`. That fell past the
  `except ImportError` guard into the generic handler and was logged as
  `SQLAlchemy instrumentation failed: 'NoneType' object is not callable` - and
  the same for PostgreSQL, MongoDB, MySQL, Redis and Kafka. None of those are
  expected on a default install. Now guarded explicitly and logged at debug.

- **`Unknown instrumentor 'smolagents' requested`** was us warning the user about
  our own defaults. `smolagents`, `litellm` and `mcp` ship in
  `DEFAULT_INSTRUMENTORS` but only resolve with the OpenInference extra; absent
  is expected, not a misconfiguration.

- **`mistralai` logged a warning where every other provider logs debug** for the
  identical condition. An absent provider SDK is the normal case.

- **GPU unavailability reported twice, at warning level.** GPU metrics default to
  on, which means "collect if this machine has a GPU", not "the user asked for
  GPU metrics". Most machines do not have one. Both messages are now debug.

  Warnings for features the user *did* enable are deliberately untouched: set
  `GENAI_ENABLE_PII_DETECTION=true` without Presidio installed and it still says
  so, loudly.

### Added

- **A startup check for the default collector.** When the endpoint is the default
  `http://localhost:4318` and nothing is listening, setup says so once and names
  the variable to set, instead of leaving the exporter's retry loop to emit
  connection stack traces indefinitely with no indication of what to do.

  It deliberately does **not** quiet those retries. A silent export failure is how
  telemetry disappears without anyone noticing - the failure this library exists
  to prevent. Bounded to a 250 ms probe, skippable with
  `GENAI_SKIP_COLLECTOR_CHECK=true`, and silent whenever the endpoint was
  configured to anything else.

## [1.13.0] - 2026-08-13

### Added

- **Deprecation metadata in the pricing schema**, and two span attributes that
  surface it: `gen_ai.request.model.deprecated` and
  `gen_ai.request.model.deprecation_note`.

  A deprecated model is invisible to cost telemetry, because nothing about it is
  wrong: it bills normally, at a real rate, right until the provider withdraws
  it. The only prior warning is a line in a vendor changelog. This turns "what
  are we running that has an end date?" into a query rather than an audit.

  46 entries are marked, all confirmed against vendor documentation during the
  1.11.x pricing audit: the 10 `moonshot-v1` keys (platform sunset 2026-08-31),
  `assemblyai/slam-1` (retired, migrate to `universal-3-pro`), 34 Deepgram legacy
  tiers that no longer appear on Deepgram's pricing page, and the retired
  `gpt-3.5-turbo-0301` snapshot. The Deepgram set was previously only recorded as
  prose in the 1.11.2 notes; it is now machine-readable.

  Deprecation is kept deliberately separate from `pricing_source`, which stays
  `table` for these models. Conflating them would make a retiring model look
  unpriced while it is still costing money - the inverse of the failure the
  `pricing_source` attribute exists to prevent.

  The data lives in a top-level `deprecated` map keyed by pricing key rather than
  as an inline field, because several categories store bare numbers
  (`"tts-1": 0.015`) that cannot carry a flag without changing their shape and
  the arithmetic that reads them. The map is excluded from category indexing, so
  a `call_type` of `"deprecated"` cannot resolve model names against it.

- **`CostCalculator.deprecation(model, call_type)`** returning the reason string
  or `None`, resolved through the same alias lookup as pricing so a
  provider-prefixed id gets the same answer as the canonical one.

## [1.12.1] - 2026-08-13

### Added

- **Muse Glimmer 30B** - input $0.30/1M, output $1.20/1M. Meta's dense open-weight
  multimodal model distilled from Muse Spark, released 2026-08-09. Keyed as
  `muse-glimmer-30b`, `muse-glimmer` and `meta/muse-glimmer-30b`.

- **Nemotron 3.5 Lightning** - input $0.08/1M, output $0.20/1M. NVIDIA open MoE,
  3B active of 30B, released 2026-08-11. Keyed as `nemotron-3.5-lightning`,
  `nemotron-3-5-lightning` and `nvidia/nemotron-3.5-lightning`. Being open-weight,
  its hosted price varies by provider ($0.04-$0.08/1M input observed); the entry
  records the reference rate and notes the spread.

- **`qwen3.8-2.4t` aliases** pointing at the existing `qwen3.8-max` price. "2.4T"
  is that model's parameter count rather than a separate model, so the alias is
  tested to track the canonical entry exactly - a distinct entry would let the two
  drift and bill one model two ways depending on which id a caller sent.

### Audit note

A comparison against [pydantic/genai-prices](https://github.com/pydantic/genai-prices)
found 706 model ids they carry that we do not match by name. Almost none turned
out to be real gaps:

- **328 of their 386 first-party models already price correctly** through our
  longest-substring lookup, including `global.anthropic.*` Bedrock aliases. Name
  mismatch is not coverage mismatch.
- **400 are OpenRouter listings**, excluded on purpose since 1.8.0: gateways
  re-list the same underlying model at their own markup, so importing them makes
  the recorded price depend on which aggregator was indexed first.
- Of the 31 current-looking first-party models left, spot-checking against vendor
  documentation found most to be stale or retiring rather than missing: the six
  GLM-4 entries do not appear in Z.ai's current pricing at all (and carry
  identical input and output prices, which no current Z.ai model does), the
  `moonshot-v1` series has a platform sunset dated 31 August, and `mistral-saba`
  is marked deprecated with a retirement date.

They were therefore not imported. Their data is a useful cross-check, but it is a
third-party aggregate, and this release cycle has already shown three times over
what happens when aggregate pricing is trusted without a vendor check.

## [1.12.0] - 2026-08-13

### Added

- **Opt-in SIGTERM flush** (`GENAI_FLUSH_ON_SIGTERM`, default `false`), closing
  the gap where containerised shutdown silently dropped queued telemetry.

  The SDK's `atexit` hook covers a clean exit and an uncaught exception, because
  Python runs `atexit` handlers on both. It does not run them when the process is
  terminated by a signal - and `docker stop` and Kubernetes pod eviction both send
  SIGTERM. Every rolling restart therefore lost whatever was still queued in the
  batch processor, up to 5 seconds or 512 spans by default, with nothing raised
  and nothing logged.

  Verified end to end against a real collector: with the flag off the span never
  reaches OpenSearch, with it on the span arrives.

  The handler is deliberately conservative. It never replaces an existing SIGTERM
  handler - whatever was registered before is invoked after the flush - and where
  none was, it restores the default disposition and re-raises, so the process
  still exits with the conventional status instead of appearing to ignore the
  signal. An explicit `SIG_IGN` is honoured. Installing off the main thread, where
  `signal.signal` cannot work, logs a warning and continues rather than raising
  into the host application. The flush is bounded by
  `GENAI_SIGTERM_FLUSH_TIMEOUT` (default 5s) so a collector that is itself down
  cannot turn a pod's grace period into a hang.

- **`genai_otel.flush_telemetry(timeout_seconds=5.0)`** for applications that
  already own their shutdown path and would rather drain telemetry themselves than
  hand a signal slot to the library.

  SIGKILL, out-of-memory kills and segfaults remain unhandleable from inside the
  process; `OTEL_BSP_SCHEDULE_DELAY` is the lever that bounds exposure there.

## [1.11.2] - 2026-08-12

### Fixed

- **The four duration-priced entries left unverified in 1.11.1 are now checked
  against vendor pricing pages**, and three of them were wrong on rate as well as
  unit:

  | Entry | Was | Now | Source |
  |---|---|---|---|
  | `deepgram/nova-3-multilingual` | $0.0092/min | $0.0092/min | deepgram.com - rate confirmed, unit corrected |
  | `fireworks/whisper-v3` | $0.004/min | $0.0015/min | fireworks.ai - stale and per-minute |
  | `fireworks/whisper-v3-turbo` | $0.002/min | $0.0009/min | fireworks.ai - stale and per-minute |
  | `assemblyai/slam-1` | $0.0045/min | unit only | deprecated by AssemblyAI, no published rate |

  `assemblyai/slam-1` is deprecated upstream, which recommends migrating to
  `universal-3-pro`. Its unit is corrected for consistency with every other
  AssemblyAI row so it cannot act as a 60x landmine, but the rate is left alone
  because there is no published figure to check it against.

- **`deepgram/nova-3` and `deepgram/nova-3-general` were carrying Nova-2 era
  pricing.** Both sat at $0.0043/min, identical to `deepgram/nova-2` and
  `deepgram/nova`; Deepgram publishes $0.0077/min for Nova-3 pre-recorded
  pay-as-you-go. Found while confirming the multilingual rate above.

### Verified (no change required)

- **Full Deepgram sweep: 26 of 46 rows confirmed against vendor pricing, no
  corrections needed.** Nova-2 and its 13 variants are $0.0043/min, exactly the
  stored value. The nine Aura text-to-speech rows are character-priced, not
  duration-priced, at $0.015 per 1000 characters ($0.030 for `aura-2`) - also
  exactly as stored. Together with the Nova-3 rows corrected above, that is every
  Deepgram entry for which Deepgram publishes a rate.

### Known gaps

- **Deepgram no longer publishes rates for its older tiers**, so 20 rows cannot
  be checked against a vendor source: `enhanced` (5), `base` (8), the
  Deepgram-hosted `whisper` variants (6) and `nova-3-medical`. Their current
  pricing page lists Nova-3 only. The units are correct after 1.11.0 and the
  rates match what Deepgram published historically, so they are left as-is rather
  than removed - but they are legacy figures, not confirmed ones, and a customer
  on a negotiated contract should override them through custom pricing.

## [1.11.1] - 2026-08-12

### Fixed

- **Two Groq transcription models still carried the per-minute unit** that 1.11.0
  corrected everywhere else. The sweep matched entries whose value equalled the
  upstream per-second rate multiplied by exactly 60, and these two had been
  rounded (`3.083e-05 * 60 = 0.0018498` against a stored `0.00185`), so they
  slipped the filter. Read as per-second they implied $6.66/hour;
  `groq/whisper-large-v3` and `-turbo` are now $0.111/hour and $0.040/hour,
  matching Groq's published rates.

- **AssemblyAI `best` and `nano` were transposed and stale.** The
  provider-qualified keys had `best` cheaper than `nano`, inherited from
  upstream, while the bare keys had the opposite. Verified against
  assemblyai.com/pricing: `best` (Universal-3.5 Pro) is $0.21/hour and `nano`
  (Universal-2) is $0.15/hour. All four keys - `assemblyai/best`,
  `assemblyai/nano`, `best`, `nano` - plus `assemblyai/universal-3-pro` and
  `assemblyai/universal-2`, now carry the published rates.

  None of these were billing anyone, since no Deepgram, AssemblyAI, Fireworks or
  Groq-Whisper instrumentor exists yet - the same latency that hid the unit bug
  in the first place.

### Known gaps

- Four duration-priced entries remain unverified because they are absent from the
  upstream price list and would need vendor pages to confirm:
  `assemblyai/slam-1`, `deepgram/nova-3-multilingual`, `fireworks/whisper-v3` and
  `fireworks/whisper-v3-turbo`. Each looks per-minute, but that is inference, not
  a checked rate, so they are left as-is rather than guessed at.

## [1.11.0] - 2026-08-12

### Fixed

- **Duration-priced audio models were stored per minute but priced per second, a
  latent 60x overcharge.** `CostCalculator._calculate_audio_cost` documents two
  units - `characters` (price per 1000) and `seconds` (price *per second*) - but
  42 of the 111 `audio` entries had been imported as LiteLLM's
  `input_cost_per_second` multiplied by 60. Every transcription model was
  affected: all 33 Deepgram models, both AssemblyAI models, `azure/whisper`, and
  both ElevenLabs Scribe models. Only 2 entries were genuinely per-second, so the
  calculator's assumption matched almost nothing.

  This never mis-billed anyone because no audio provider was instrumented, so
  nothing ever passed `seconds` - it would have gone live the moment one was
  added. All 42 are now stored per-second, derived from the upstream value rather
  than hand-edited. `elevenlabs/scribe_v1` round-trips to $0.21996/hour against
  ElevenLabs' published $0.22/hour.

- **ElevenLabs text-to-speech rates corrected against the published API price
  list.** `eleven_multilingual_v2` / `_v1`, `eleven_monolingual_v1` and
  `eleven_english_v1` were priced at $0.24 per 1000 characters; the pay-as-you-go
  API rate is $0.10. `eleven_turbo_v2` is on the Flash/Turbo tier at $0.05. The
  two speech-to-speech models are deliberately left alone - speech-to-speech is
  not billed per input character and no published rate covers it.

### Added

- **ElevenLabs instrumentor** (`elevenlabs`), covering text-to-speech
  (`text_to_speech.convert` / `.stream`) and Scribe speech-to-text
  (`speech_to_text.convert`), on both the sync and async clients.

  Audio providers bill by media rather than tokens, so spans carry
  `gen_ai.usage.characters` for synthesis and
  `gen_ai.usage.audio_duration_seconds` for transcription instead of token
  counts, alongside `gen_ai.request.voice_id` and
  `gen_ai.response.transcript_length`. Text-to-speech returns an iterator of
  audio bytes, which is wrapped rather than consumed so `gen_ai.server.ttft`
  measures time to first audio byte - for a voice turn that is what the caller
  actually waits for, not total generation time.

  Audio payloads are never attached to spans; only sizes and durations are
  recorded, since voice audio is frequently personal data.

  The pricing table keys text-to-speech models bare (`eleven_multilingual_v2`)
  but Scribe by provider (`elevenlabs/scribe_v1`), while the SDK passes the bare
  `model_id` in both cases. Cost lookup therefore tries the provider-qualified
  key first and the bare name second - without it, `scribe_v1` matched nothing
  and every transcription would have been priced at zero.

## [1.10.0] - 2026-08-12

> Dual emission is now the DEFAULT. Spans carry both the current and the superseded
> GenAI attribute spellings unless you explicitly set
> `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai`. This reverses the 1.9.0 default, which
> dropped the superseded token names and silently zeroed any consumer still reading
> them. The default flips to current-only at 2.0, where a breaking change belongs.

### Fixed

- **1.9.0 replaced the superseded token names instead of emitting both, which zeroed
  consumers that read them.** The rename direction in 1.9.0 was right, the migration
  strategy was not. `gen_ai.usage.prompt_tokens` / `completion_tokens` simply stopped
  appearing, and an absent attribute reads as *zero tokens* and therefore *zero cost* —
  never as an error. Anything consuming the superseded names on a minor version bump
  lost its numbers without a single log line. Dual emission is now the default.

- **`gen_ai.system` -> `gen_ai.provider.name` is now emitted, closing the provider half
  of the same interop gap.** 1.9.0 fixed tokens; the provider attribute was still only
  emitted under its superseded spelling, so a backend consuming the current conventions
  saw **no provider at all**. That is not cosmetic: consumers routinely treat a missing
  provider as "not a GenAI span" and drop the record rather than showing it unlabelled.
  Roughly 29 instrumentors write `gen_ai.system` as a raw string literal in their own
  `_extract_*_attributes`, so `BaseInstrumentor._with_provider_aliases()` mirrors the two
  spellings centrally, on the two paths every one of those dicts already flows through —
  fixing all of them without editing any, and keeping the policy in one place. A value an
  instrumentor set deliberately is never overwritten, and non-GenAI spans are untouched.

- **`OTEL_SEMCONV_STABILITY_OPT_IN` was matched with substring checks.** It is a
  comma-separated list shared by *every* instrumentation area, so `"dup" in raw` was true
  for `http/dup` — an unrelated HTTP opt-in switched on GenAI dual emission. The inverse
  was worse: `"gen_ai,http/dup"` means *current GenAI names only* plus an HTTP opt-in, and
  was read as dual emission against an explicit request not to. Parsing now happens in
  `genai_otel.semconv.genai_semconv_modes()`, which tokenises the list properly.

- **A missing config silently disabled dual emission.** `_set_token_usage_attributes()`
  treated an unresolvable config as "no dual emission", so a configuration problem
  degraded into missing attributes — i.e. into a wrong number rather than a visible
  failure. Absent config now resolves the same way an unset env var does.

### Added

- **`genai_semconv_modes()` and `genai_tier_opted_in()`** in `genai_otel.semconv`. The two
  answer deliberately different questions: the first decides which *names* to use for
  attributes emitted regardless, and defaults to the safe (dual) value; the second gates
  the heavier canonical `gen_ai.input.messages` / `output.messages` payload, which carries
  message **content**, so an explicit opt-out is honoured rather than defaulted. Collapsing
  them would start emitting message content for someone who had opted out.

- **A "Renaming a span attribute" section in `Contributing.md`.** The root cause of both
  regressions is structural: the code that reads these attributes generally lives in
  another repository and fails silently, so a contributor working here cannot see the
  breakage they cause. The section states the rule — emit both spellings for at least one
  major version, never replace — and why the failure mode makes it non-negotiable.

### Upgrading

No action required. Spans now carry both spellings, so both old and new consumers work.

To emit only the current names (the 1.9.0 behaviour):

```bash
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai
```

## [1.9.0] - 2026-08-12

> Minor rather than patch: in the default `gen_ai` mode, spans now carry
> `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens` instead of
> `gen_ai.usage.prompt_tokens` / `gen_ai.usage.completion_tokens`. Anything
> querying the superseded names needs updating, or set
> `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai/dup` to emit both.

### Fixed

- **Token usage attributes now follow the current GenAI semantic conventions.**
  The two conventions were wired up backwards: `gen_ai.usage.prompt_tokens` /
  `gen_ai.usage.completion_tokens` were emitted unconditionally and treated as
  current, while `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens` were
  labelled "old semantic convention" and emitted only under
  `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai/dup`. The rename runs the other way -
  semantic-conventions v1.27.0 renamed `prompt_tokens` -> `input_tokens` and
  `completion_tokens` -> `output_tokens`
  ([semantic-conventions#1200](https://github.com/open-telemetry/semantic-conventions/pull/1200)).

  Consequence: any backend consuming the current conventions read **zero
  tokens** from our spans. Confirmed against Arize AX, which maps
  `gen_ai.usage.{input,output}_tokens` onto `llm.token_count.*` and derives cost
  from them - a default-mode span landed with `llm.token_count.total = 0` and
  `llm.cost.total = 0` despite carrying correct counts under the superseded
  names.

  **Behaviour change:** in the default `gen_ai` mode, spans now carry
  `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens` and no longer carry
  `gen_ai.usage.prompt_tokens` / `gen_ai.usage.completion_tokens`. Dashboards,
  alerts or aggregation rules querying the superseded names should either be
  updated, or keep both by setting `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai/dup`.
  Cost tracking is unaffected - the cost calculator reads the usage payload
  directly and its span-attribute fallbacks now prefer the current names.

  Emission is centralised in `BaseInstrumentor._set_token_usage_attributes()`,
  so the naming policy is applied in one place. This also fixes the HuggingFace
  and Hyperbolic instrumentors, which previously emitted only the superseded
  names and ignored the opt-in flag entirely.

### Added

- **`examples/arize_ax/`** - end-to-end example exporting to Arize AX over plain
  OTLP with no Arize or OpenInference packages installed, configured entirely
  through standard OpenTelemetry environment variables.

## [1.8.1] - 2026-08-08

### Fixed

- **Instrumenting httpx made every SYNCHRONOUS call return a wrapper instead of a
  response.** `APIInstrumentor._wrap_api_call` called `create_span_wrapper(...)` — which
  returns a `@wrapt.decorator` — as `instrumented_call(wrapped, instance, args, kwargs)`.
  Calling a wrapt decorator that way does not execute anything; it builds and returns a
  `FunctionWrapper`. So callers received the wrapper rather than the response, and any
  `resp.status_code` raised:

  ```
  AttributeError: 'function' object has no attribute 'status_code'
  ```

  Only `httpx.Client.request` is wrapped, so `client.send()` and the entire `AsyncClient`
  were unaffected — which is why this survived: nearly every caller in practice is async,
  and the synchronous ones failed silently wherever the exception was swallowed. In one
  downstream platform it meant a Jinja template loader served on-disk prompts for months
  while newer versions sat in a central prompt store, logged at `debug` where nobody saw
  it.

  Fixed by decorating first and then calling: `instrumented_call(wrapped)(*args, **kwargs)`.

  Five regression tests added in
  `tests/mcp_instrumentors/test_api_instrumentor_returns_response.py`, each verified to
  fail against the old line and pass against the new one. The load-bearing one is
  end-to-end: after `instrument()`, a real `httpx.Client().get()` over `MockTransport`
  must still return a `Response`. Every unit-level assertion about *whether the call
  happened* passes against a wrapper too — only asserting on what comes **back** catches
  this class of bug.

  Anyone on 1.6.x–1.8.0 doing synchronous httpx calls in an instrumented process is
  affected and should upgrade.

## [1.8.0] - 2026-08-07

> Ships everything in 1.7.0 as well. That version was tagged and passed
> pre-release validation, but a GitHub-wide Actions outage on 2026-08-06 meant
> its publish workflow never ran, so no 1.7.0 artifact exists on PyPI. Its
> changes are merged into main and therefore included here — upgrading from
> 1.6.1 straight to 1.8.0 loses nothing.

### Added

- **Pricing coverage sweep: 565 chat models added (858 -> 1423).** Imported from
  models.dev, restricted to first-party providers and the major clouds that
  resell under their own SKUs. Gateway/router listings (nano-gpt, kilo,
  openrouter, vercel, llmgateway and similar) are deliberately excluded: they
  re-list the same underlying model at their own markup, so importing them
  would make the recorded price depend on which aggregator happened to be
  indexed first. Coverage of models.dev's priced catalogue goes from 3614/5739
  (63%) to 4779/5739 (83%).

  This includes the **July and August 2026 releases the scheduled refresh never
  delivered** - Claude Opus 5, Kimi K3, Gemini 3.6 Flash, Gemini 3.5 Flash Lite,
  Qwen3.7 Flash, Qwen3.8-Max, Thinking Machines Inkling, Tencent Hy3 and Muse
  Spark 1.2 - together with their Bedrock, Vertex and regional aliases.

- **`gen_ai.usage.cost.pricing_source` span attribute**, reporting `table`,
  `estimated` or `unpriced`. A cost of `0.0` is otherwise ambiguous: it means
  either "this call was free" or "no price was found", and the two must not be
  summed into a spend figure alike. `estimated` marks a price inferred from the
  parameter count in the model name (indicative, not billable).

### Fixed

- **Model-name resolution rebuilt every index on every lookup.** The staleness
  guard compared the exact-index size against the pricing-dict size, but keys
  differing only by case (`MiniMax-M3` / `minimax-m3`) collapse into one
  exact-index entry, so those counts never matched. The guard therefore fired on
  every call, rebuilding all indices and clearing the memo cache each time,
  which defeated memoisation entirely. Over 4000 lookups: cold 1288ms -> 107ms,
  warm 1257ms -> 0.63ms. The guard now compares against the size the index was
  built from, and runs before the memo lookup so cached negative results cannot
  go stale.

- **Version punctuation mismatches billed against the wrong model.** Vendors
  publish `gpt-4.1` while callers and eval harnesses routinely send `gpt-4-1`.
  A dotted key cannot match a dashed name as a substring, so the dashed form
  fell through to whatever shorter key did match: `gpt-4-1` resolved to `gpt-4`
  and was billed at $30/1M instead of $2/1M (15x over), `gpt-4-1-nano` at 300x
  over, while `gpt-5-4-pro` resolved to `gpt-5` and was billed at $1.25/1M
  instead of $30/1M (24x under). Dashed aliases are now registered wherever they
  would resolve differently.

- **`setup_auto_instrumentation()` built a second `CostCalculator`** instead of
  reusing the process-wide one, rebuilding the full pricing index on every call.

- **Parameter-count fallback missed `_` and `.` delimited size tokens**
  (`MMPO_Gemma_7b_gamma1.1`, `smollm2-135M_pretrained_400k`), dropping those
  models to an unpriced $0.00 rather than the local-size tier. 145 more models
  now resolve to an estimated price.

- **Embedding, reranker and ASR models are no longer imported into the `chat`
  table.** Several (`text-embedding-3-small`, `mistral-embed`,
  `gemini-embedding-001`) are already keyed under `embeddings` with a different
  value shape, so importing them duplicated the entry and added keys to the chat
  substring index that no chat call should match.

- **Free-tier listings are skipped rather than stored as `$0`.** NVIDIA NIM,
  Groq, zai `*-flash` and llama.com quote 0/0 on models.dev. A stored zero is
  indistinguishable from a real price and, being the longer key, would win the
  substring race and shadow the paid entry for the same family - reporting real
  spend as free.

- **Bias detector fired on 76% of prompts because the pronoun "it" was listed
  as a sexual-orientation slur ([#13](https://github.com/Mandark-droid/genai_otel_instrument/issues/13)).**
  `bias_detector.py` put the bare English pronoun `it` alongside two genuine
  slurs in the sexual-orientation pattern. As a context-free `\bit\b` it matched
  almost any English text: on a 24h agentic workload the prompt-side evaluator
  fired on 516/683 prompts (76%), and every sampled detection was
  `sexual_orientation` matched on nothing but `it`. Toxicity and
  prompt-injection fired 0/974 on the same traffic, so this one was the outlier.

  A detector that fires on three quarters of traffic is worse than a missing
  one: real detections are buried, and a dashboard reporting "bias detected on
  76% of prompts" tells an operator their system is systematically biased when
  it is not — a claim someone has to answer for in a regulated setting. Nothing
  errored, so the number looked authoritative.

  The bare pronoun is removed. The dehumanising usage the pattern was reaching
  for — `it` applied to a person — now requires context
  (`called her an it`), and the genuine slurs still match. On benign agentic
  prompts the false-positive rate goes from 9/10 to 0/10 with every genuine
  detection preserved.

## [1.7.0] - 2026-08-06

### Fixed

- **MCP spans were unjoinable with the LLM spans from the same run ([#11](https://github.com/Mandark-droid/genai_otel_instrument/issues/11)).**
  MCP client spans recorded the session only as `mcp.session_id`, while LLM
  spans use `session.id` / `gen_ai.conversation.id` (set from
  `config.session_id_extractor`). With no key in common an agent session could
  not be reassembled from its own telemetry: on a real run 10 spans carried
  `session.id` and 450+ tool calls carried none, so 552 spans grouped into 209
  "sessions" of roughly 1.5 spans each where eight runs had actually happened —
  and the fragments look plausible, so anything computing cost, duration or
  tool-usage per session was silently wrong. `mcp_session()` now emits the
  conventional keys alongside `mcp.session_id`, which stays unchanged for
  existing readers.

- **LiteLLM callers lost all token and cost telemetry ([#10](https://github.com/Mandark-droid/genai_otel_instrument/issues/10)).**
  LiteLLM calls the OpenAI SDK in raw-response mode so it can read rate-limit
  headers, so `chat.completions.create` returns
  `openai._legacy_response.LegacyAPIResponse` rather than a `ChatCompletion`.
  That wrapper exposes no `.usage` — the model is reachable only via `.parse()`
  — so every `hasattr(result, "usage")` check was False and usage, cost and
  finish-reason were dropped. The failure was silent: spans were created and
  request attributes were correct, only the numbers were missing. Where an LLM
  gateway is the standard inference entry point this zeroes cost dashboards and
  spend attribution for every service behind it, and a zeroed cost reads as
  "this was free" rather than "this was not measured". Raw-response wrappers are
  now unwrapped centrally in `BaseInstrumentor._record_result_metrics`, so all
  instrumentors benefit; it also covers direct use of
  `with_raw_response.create(...)` and the same wrapper shape on Azure OpenAI.
  Degrades to prior behaviour if a wrapper cannot be parsed, and does not touch
  streaming (the only caller runs on the non-streaming branch).

- **`mcp` added to the `dev` extra.**
  `tests/mcp_semconv/test_client_instrumentor.py` asserts the instrumentor wraps
  the real `mcp.client.session.ClientSession`, but the SDK was never a dev
  dependency, so every CI test job failed on `ModuleNotFoundError: No module
  named 'mcp'`. It is pinned to `python_version >= '3.10'` (the SDK's own floor)
  and the two tests that need it skip themselves on 3.9.

### Added

- **MCP client instrumentation (`genai_otel.mcp_semconv`)** — one span per
  `callTool`, named `mcp.call_tool {server}.{tool}`. This is net-new ground:
  `openinference-instrumentation-mcp` is a pure W3C context-propagation shim
  that creates no spans and sets no attributes, and
  `openinference-instrumentation-smolagents` names every MCP tool span after
  the adapter class (`MCPAdaptTool`) with no server attribution, no protocol
  error detail, and no session key. Nothing here double-instruments either.

  Emits `mcp.server`, `mcp.tool`, `mcp.tool.raw_name`, `mcp.stage`,
  `mcp.behaviour`, `mcp.idempotent`, `mcp.session_id`,
  `mcp.error.message_raw` / `.http_status` / `.jsonrpc_code`,
  `mcp.tool_selection.candidate_count` / `.correct` / `.expected`,
  `mcp.identifier.hallucinated`, `commerce.cart_hash`,
  `commerce.order.placement_attempt` and `commerce.terminal_state`.

- **Schema-map-driven tool metadata** (`MCPToolRegistry`). Stage, behaviour and
  idempotency come from a supplied schema map, never hardcoded. Server
  attribution resolves against *known* server names rather than splitting on
  the first separator, so a real tool name that contains the separator
  (`search_restaurants_dineout`) is not mangled. A canary test fails loudly if
  the upstream composite-proxy separator assumption ever changes.

- **Deterministic terminal-state classifier** (`TerminalStateClassifier`).
  Rule-based over span attributes with no LLM judgement; every classification
  returns the evidence that produced it. Seven states, with `BLOCKED_NO_TOOL`
  sub-classified by the resolution the user actually requested — which
  separates "the agent failed" from "no tool in the surface could have
  succeeded".

- **Live validation harness** (`examples/mcp_live_validation/`). Drives a real
  MCP client over stdio against a real FastMCP composite-proxy server and
  asserts the emitted attributes, so the instrumentation is verified against a
  real transport rather than only against mocks.

### Security

- MCP user identifiers are salted-SHA-256 hashed before reaching a span
  (`GENAI_MCP_HASH_SALT`; an ephemeral per-process salt with a warning when
  unset). Request and response bodies are never written to a span by this
  module — the session id is recorded as the support-correlation key, and
  hallucinated-identifier detection records offending argument *keys* only,
  never their values.

## [1.6.1] - 2026-07-15

### Changed

- **README — Hermes integration now points to the standalone plugin repo.**
  The upstream `NousResearch/hermes-agent` PR #48184 (bundled
  `observability/otel` plugin) was approved on code review and then closed
  per the project's standing policy that observability backends ship as
  standalone plugin repos rather than in the core tree. The integration now
  lives at [Mandark-droid/hermes-otel-plugin](https://github.com/Mandark-droid/hermes-otel-plugin)
  (`hermes plugins install Mandark-droid/hermes-otel-plugin`), and the
  Ecosystem & Framework Contributions table reflects the shipped standalone
  status. Docs-only release — no code changes.

## [1.6.0] - 2026-07-10

### Security & performance hardening (BFSI / on-prem)

This is a large hardening pass ahead of regulated on-prem (bank) deployments.
Content capture stays configurable via the existing
`GENAI_ENABLE_CONTENT_CAPTURE` / `GENAI_CONTENT_MAX_LENGTH` route (RBI
tracing/auditability is preserved); the work below makes that data safer,
bounded, and cheaper to collect, and fixes correctness bugs.

### Added

- **`GENAI_PROFILE=strict` (aliases `bfsi`, `bank`)** bank hardening profile: keeps
  audit content capture ON but forces all third-party egress OFF and enables
  air-gapped mode in one switch. New settings `GENAI_ALLOW_EXTERNAL_EGRESS`
  (default true) and `GENAI_AIR_GAPPED` (default false); the strict profile sets
  egress off, air-gapped on, `co2_offline_mode` on, and Perspective API off.
- **Pre-call BLOCK enforcement**: `pii_mode=block` / `*_block_on_detection` now
  actually intercept the request — a new pre-call hook evaluates the prompt and
  raises `PolicyViolationError` (new, in `genai_otel.exceptions`) BEFORE the LLM
  call when a configured policy triggers. Gated by an active blocking policy, so
  the default path has zero extra cost.
- **Metric verbosity controls** (`GENAI_METRICS_PROFILE`, plus
  `GENAI_RECORD_TOKEN_HISTOGRAMS` / `GENAI_RECORD_GRANULAR_COST_METRICS` /
  `GENAI_RECORD_FINISH_METRICS` / `GENAI_ENABLE_CONCURRENCY_METRICS`). Full
  per-request detail is still always written to span attributes (audit is
  unaffected); the redundant aggregated-metric instruments are now opt-in.

### Changed (performance)

- **Per-call span-wrapper overhead cut ~30-80%** (measured with content capture
  ON): a priced model went ~61 us -> ~42 us and an unpriced/internal model
  ~182 us -> ~31 us. Levers: memoized/indexed cost lookup (no more O(n) scan +
  per-call `sorted()` of ~850 model keys), a process-wide shared `CostCalculator`
  (one pricing-JSON parse instead of ~29), gated metric fan-out, removal of
  per-call debug f-strings, and metric exemplar sampling now defaults OFF
  (`OTEL_METRICS_EXEMPLAR_FILTER=always_off`, override to re-enable).
- Single-thread instrumented throughput improved (~15.5k -> ~23k calls/s). Note:
  thread-level scaling remains bounded by the CPython GIL (CPU-bound work);
  scale out with multiple processes.

### Fixed (correctness)

- **Double execution of the wrapped LLM call**: the wrapper's outer fallback
  re-invoked the wrapped function after it had already run, so every errored call
  AND every sampled-out call (`GENAI_SAMPLING_RATE<1.0`) executed the underlying
  request twice (double API spend, duplicated side effects). Fixed with an
  `invoked` guard, a fast path for non-recording (sampled-out) spans, and
  defensive reads so `_record_result_metrics` never touches `.name`/`.attributes`
  on a `NonRecordingSpan`.
- **Client-breaking instrumentors**: Groq and SambaNova `__init__` wrappers
  returned a value (`TypeError` on every client construction); AWS Bedrock
  assigned the decorator factory to `invoke_model` instead of wrapping the bound
  method (`TypeError` on first call); Hyperbolic called `CostCalculator` with the
  wrong signature and re-raised JSON errors into the caller. All fixed with
  regression tests.
- LangChain `AgentExecutor` wrapper called `.get()` on a non-dict; Haystack
  wrappers passed `instance` twice into already-bound methods (corrupting
  `Pipeline.run`); pymongo metrics wrapper `NameError` on the error path.

### Security

- **No third-party egress of content by default**: the toxicity Perspective API
  path (which sends prompt/response text to Google) is now hard-gated on
  `allow_external_egress`; the API key is scrubbed from logs. Detoxify/spaCy model
  downloads are skipped in air-gapped mode.
- **Media subsystem**: the `pdf_pii_redact` redactor no longer uploads the
  original document while stamping a false `/RedactionApplied` flag - all
  built-in redactors now FAIL CLOSED (drop bytes) on error/missing-dependency;
  the env-driven redactor loader is allowlisted (built-ins only under strict
  profile); untrusted inline media is size-capped before base64 decode and before
  PIL/opencv/pypdf parsing (decompression-bomb defense); the filesystem store
  enforces path-traversal containment + `0o600` perms; http/s3 stores reject
  plaintext endpoints under strict profile and strip credentials from `media_uri`.
- **Bounded content everywhere**: previously-unbounded content on spans, span
  events, streaming buffers, enrichment-processor response copies, and MCP
  `db.statement`/Cypher capture now honor `content_max_length` (0 = unlimited for
  full audit).
- **Evaluation dedup**: detectors previously ran up to 3x per span (inline,
  `on_end`, and export); now deduped via an `evaluation.completed` marker.
- **Supply chain / CI**: removed the non-existent `azure-ai-openai` dependency
  (dependency-confusion vector; `[azure]` now uses the `openai` SDK); added
  least-privilege `permissions:` blocks and a documented OIDC trusted-publishing
  path to the workflows; added a `gitleaks` secret-scanning pre-commit hook;
  de-hardcoded the API key in `scripts/test_basic_openai.py`; corrected stale
  `SECURITY.md` claims.
- Added double-wrap idempotency guards across instrumentors, scheme validation on
  the Ollama metrics poller URL, and best-effort `uninstrument()` paths for
  several MCP instrumentors.
- **Aggregator client dedup** (folded in from the never-released 1.5.1): a client
  pointed at an aggregator `base_url` (OpenRouter, CometAPI) was wrapped twice -
  by the dedicated aggregator instrumentor and by the generic OpenAI/Anthropic
  instrumentor - producing duplicate spans and double-counted token/cost metrics.
  Aggregator instrumentors now register a `base_url` claim and the generic
  instrumentors skip claimed clients (one span, one set of metrics per call).

### Pricing

- Added `cacheReadPrice` ($0.50/1M) to the Sakana Fugu Ultra entries
  (`fugu-ultra` / `sakana/fugu-ultra` / `fugu-ultra-20260615`); the >272K-context
  premium tier is documented in each note (not representable in the flat schema).

## [1.5.0] - 2026-07-07

### Added

- **CometAPI Provider Support**
  - New `CometAPIInstrumentor` for [CometAPI](https://www.cometapi.com), an
    all-in-one aggregator exposing 500+ models (GPT, Claude, Gemini, DeepSeek,
    Qwen, and more) behind a single API key
  - Automatic detection of CometAPI clients via `base_url` checking for
    `cometapi.com` - works with BOTH the Anthropic SDK (`/v1/messages`) and the
    OpenAI SDK (`/v1/chat/completions`)
  - Token usage extraction handles both response shapes: OpenAI-compatible
    (`prompt_tokens`/`completion_tokens`) and Anthropic-compatible
    (`input_tokens`/`output_tokens`, including prompt-cache read/write tokens)
  - Spans named `cometapi.messages.create` (Anthropic SDK) and
    `cometapi.chat.completion` (OpenAI SDK) with `gen_ai.system = "cometapi"`
  - Cost tracking resolves through the requested model name against the
    existing pricing database (e.g. `claude-sonnet-5`, `gpt-5-mini`)
  - Enabled by default (`cometapi` added to `DEFAULT_INSTRUMENTORS`)
  - Install with: `pip install genai-otel-instrument[cometapi]`
  - Example: `examples/comet_api.py` (Anthropic SDK + OpenAI SDK usage)
  - 27 new unit tests in `tests/instrumentors/test_cometapi_instrumentor.py`
  - Documentation: provider table + quick example in
    `docs/guides/llm-providers.md`, install extra in
    `docs/getting-started/installation.md`, default instrumentor list in
    `docs/getting-started/configuration.md`, README provider list

## [1.4.2] - 2026-07-01

### Added

- **June 2026 model pricing sweep** in `genai_otel/llm_pricing.json`, covering
  new closed- and open-weight releases across all supported providers so cost
  tracking resolves them out of the box (prices per 1M tokens; stored per 1k):
  - **Anthropic Claude Sonnet 5** (`claude-sonnet-5`, dated + `anthropic.` Bedrock
    aliases): intro discount **$2 in / $10 out** through Sep 2026; reverts to
    standard **$5 / $15** afterward. 1M context.
  - **Moonshot Kimi K2.7 Code** (`kimi-k2.7-code`, `-highspeed`, `moonshotai/`
    prefixes): **$0.95 / $4** (HighSpeed **$1.90 / $8**). Open weights, 256K ctx.
  - **Cohere North Mini Code** (`north-mini-code-1-0`, `north-mini-code`,
    `cohere/north-mini-code`): free open-weight coding model (**$0 / $0**).
  - **Nvidia Nemotron 3 Ultra 550B A55B** (`nvidia/nemotron-3-ultra-550b-a55b`):
    **$0.50 / $2.50**. Open weights, 1M context.
  - **Xiaomi MiMo family** (previously absent entirely): `mimo-v2-flash` &
    `mimo-v2-omni` & `mimo-v2.5` at **$0.14 / $0.28**; `mimo-v2-pro` &
    `mimo-v2.5-pro` at **$0.435 / $0.87**; `mimo-v2.5-pro-ultraspeed` (Jun 8) at
    **$1.305 / $2.61**. All with `xiaomi/` prefixed aliases.
  - **Sakana AI Fugu Ultra** (`fugu-ultra`, `sakana/fugu-ultra`, dated): **$5 / $30**.
  - **Zhipu / Z.AI GLM-5.2** (`glm-5.2`, `zai/glm-5.2`, `THUDM/GLM-5.2`):
    **$1.40 / $4.40**. 1M context.
  - **Alibaba Qwen3.7 Plus** (`qwen3.7-plus`, `qwen-qwen3.7-plus`,
    `dashscope/qwen3.7-plus`): **$0.50 / $3**. 1M context.
  - **Meituan LongCat-2.0** (`longcat-2.0`, `meituan/longcat-2.0`) at standard
    **$0.75 / $2.95** (limited-time promo $0.30 / $1.20), and **LongCat Flash Chat**
    (`longcat-flash-chat`) at **$0.20 / $0.80**. Open weights.
- **`models.dev` added as a pricing data source** (see "Pricing data sources"
  below). Its structured `api.json` (per-model `cost`, `release_date`,
  `open_weights`) is now the primary cross-source for the monthly refresh.
- **Regression tests** extended in `tests/test_pricing_new_models.py`: one row per
  new model plus novel-snapshot routing rows asserting Sonnet 5 does not collapse
  onto Sonnet 4.5 and the MiMo UltraSpeed tier does not collapse onto `mimo-v2.5-pro`.

### Changed

- **DeepSeek V4 Pro** note refreshed: the 75% promotional discount
  (**$0.435 / $0.87**) remains in effect as of 2026-07-01, re-verified against
  `api-docs.deepseek.com/quick_start/pricing`; price unchanged.

### Deferred

- **DeepReinforce Ornith 1.0** (`Ornith-1.0-9B/31B/35B-MoE/397B-MoE`, MIT, Jun 25)
  intentionally NOT added: no first-party API pricing is published yet and no
  aggregator lists a rate. Adding a $0 or guessed price would make the cost
  tracker under-report, so the entry waits until an official rate exists.

### Pricing data sources

The `genai_otel/llm_pricing.json` database is refreshed monthly from the sources
below (maintained here per the update routine). Later sources are cross-checks;
first-party vendor pricing always wins on conflict.

- **First-party provider docs**: Anthropic (`platform.claude.com`), OpenAI,
  Google (`ai.google.dev`), Moonshot AI (`platform.moonshot.ai`), Xiaomi
  (`platform.xiaomimimo.com`), Zhipu / Z.AI (`docs.z.ai`), Alibaba DashScope,
  DeepSeek (`api-docs.deepseek.com`), MiniMax (`platform.minimax.io`), xAI
  (`docs.x.ai`), Cohere, Nvidia, Sakana AI, Meituan LongCat (`longcatai.org`).
- **Hyperscalers**: AWS Bedrock, Azure AI Foundry, Google Vertex AI.
- **Aggregators / cross-checks**: OpenRouter, LiteLLM, Artificial Analysis.
- **models.dev** (`https://models.dev/models/`, `api.json`) - added 2026-07-01.

## [1.4.1] - 2026-06-26

### Fixed

- **Prompt-side PII / toxicity / bias now evaluate the full message list, not just the system prompt.** `EvaluationSpanProcessor` reconstructs the prompt from the per-message `gen_ai.prompt.{idx}` span events (new `_extract_prompt_from_events`), and `BaseInstrumentor._run_evaluation_checks` reads `kwargs["messages"]`, so `evaluation.pii.prompt.detected` (and toxicity/bias) flag content in USER messages — previously only the system / first message (`gen_ai.request.first_message`) was checked.

## [1.4.0] - 2026-06-25

### Added

- **IFSC bank-code PII recognizer** (`IN_IFSC`) added to the India recognizer set
  in `PIIDetector._register_india_recognizers` and to `PIIConfig` default
  `entity_types` (regex `\b[A-Z]{4}0[A-Z0-9]{6}\b`, score 0.9). Indian bank branch
  codes (e.g. `HDFC0001234`) are now detected out of the box.
- **Custom PII recognizers via `pii_custom_patterns`.** `genai_otel.instrument(...)`
  now accepts `pii_custom_patterns={"ENTITY": r"regex"}` (or
  `{"ENTITY": {"regex": ..., "score": 0.8}}`), wired through `OTelConfig` →
  `PIIConfig.custom_patterns` → a new `PIIDetector._register_custom_recognizers`.
  Registered entity labels are tracked and included in `detect()` analysis, so
  customers can add their own PII/BFSI classes (e.g. internal customer-reference
  numbers) with no code changes. (Previously `PIIConfig.custom_patterns` existed
  but was never consumed.)
- Unit tests (`tests/evaluation/test_pii_detector.py`) for IFSC, the UPI `okhdfcbank` fix, and `custom_patterns`; the IFSC + custom recognizers also work in the no-Presidio regex fallback (parity with the other India classes).

### Fixed

- **UPI VPA recognizer NPCI handles.** The `IN_UPI` pattern used `okhdfc`, which
  failed to match real handles like `name@okhdfcbank` (word boundary after the
  truncated PSP). Corrected to the actual NPCI set
  (`ok(?:hdfcbank|axis|icici|sbi|bizaxis)`), so HDFC/Axis/ICICI/SBI UPI addresses
  are detected. Applied to both the Presidio recognizer and the no-Presidio fallback.

## [1.3.3] - 2026-06-11

### Added

- **New model pricing entry** in `genai_otel/llm_pricing.json`:
  - **Anthropic Claude Fable 5** (`claude-fable-5`): $10/1M input,
    $50/1M output. Anthropic's new top tier above Opus; dated snapshots
    and the `[1m]` long-context variant resolve to the same entry via
    the longest-substring lookup.
- **Regression test rows** in `tests/test_pricing_new_models.py` covering
  the new entry's pricing and snapshot/variant alias routing.

## [1.3.2] - 2026-06-01

### Added

- **New model pricing entries** in `genai_otel/llm_pricing.json` for recently
  released flagship models, so cost tracking resolves them out of the box:
  - **Anthropic Claude Opus 4.8** (`claude-opus-4-8`, `claude-opus-4.8`):
    $5/1M input, $25/1M output (standard mode).
  - **Google Gemini 3.5 Flash** (`gemini-3.5-flash`,
    `gemini/gemini-3.5-flash`): $1.50/1M input, $9.00/1M output
    (released May 19, 2026).
  - **MiniMax M3** (`MiniMax-M3`, `minimax-m3`, plus the `-highspeed` tier):
    $0.30/1M input, $1.20/1M output, 1M context.
  - **OpenAI GPT-5.5 series** (`gpt-5.5-mini` $0.40/$1.60, `gpt-5.5-nano`
    $0.10/$0.40, `gpt-5.5-pro` $30/$180 per 1M); the `gpt-5.5` flagship was
    already present.
- **Regression test** `tests/test_pricing_new_models.py` that loads the
  shipped pricing file and asserts the new entries resolve to the expected
  cost, including dated/preview snapshot aliases routing to the correct
  variant via the longest-substring lookup.

## [1.3.1] - 2026-05-12

### Added

- **Detailed token-usage span attributes**: when the provider supplies
  the corresponding data, LLM-call spans now emit
  `gen_ai.usage.cache_read.input_tokens`,
  `gen_ai.usage.cache_creation.input_tokens`, and
  `gen_ai.usage.reasoning.output_tokens`. Sources:
  - **Anthropic**: `usage.cache_read_input_tokens` and
    `usage.cache_creation_input_tokens` (prompt-caching feature) were
    already extracted by the Anthropic instrumentor for cost
    calculation; now also surfaced as span attributes.
  - **OpenAI**: `usage.prompt_tokens_details.cached_tokens` (prompt
    caching on chat completions) is surfaced under the same canonical
    `cache_read.input_tokens` attribute. `usage.completion_tokens_details
    .reasoning_tokens` (o1/o3-style models) is surfaced as
    `gen_ai.usage.reasoning_tokens`.

  Attribute names align with the upstream proposal at
  `semantic-conventions-genai#76` (detailed token usage: cache,
  reasoning). Zero / missing values are not emitted to avoid noisy
  zero-valued attributes on every span.

## [1.3.0] - 2026-05-12

### Changed

- **Multimodal canonical JSON: align stripped-part shape with upstream
  `semantic-conventions-genai` PR #144's design pivot.** Stripped media
  parts (instrumentation observed bytes but intentionally did not capture
  them — size cap exceeded, modality not allowed, redactor failure, etc.)
  no longer emit as a separate `{"type": "stripped", ...}` shape. Instead
  they keep the original part type (`blob` / `uri` / `file`) and `modality`,
  omit the content-bearing field (`content` / `uri` / `file_id`), and set
  `stripped_reason`. The flat `gen_ai.prompt.{n}.content.{m}.*` namespace
  is unchanged.
  - Affects `gen_ai.input.messages` / `gen_ai.output.messages` span JSON
    only; opt-in capture must be on for any consumer to see this shape.
  - Wire-format example before:
    `{"type": "stripped", "modality": "image", "stripped_reason": "size_exceeded", "byte_size": 2000000}`
  - Wire-format example after:
    `{"type": "blob", "modality": "image", "mime_type": "image/png", "byte_size": 2000000, "stripped_reason": "size_exceeded"}`
  - Companion change: `genai_otel.semconv` comment updated to reflect that
    the multimodal shape is now in active upstream review (PRs #142
    approved, #143/#144 under review).

### Added

- **Harmonized cross-framework agent attribution: `gen_ai.agent.name`** is
  now co-emitted alongside the framework-prefixed names (`crewai.*`,
  `autogen.agent.name`, `google_adk.agent.name`, `langchain.agent.name`,
  `openai.agent.name`, `pydantic_ai.agent.name`,
  `autogen_agentchat.agent.name`) on every multi-agent framework
  instrumentor that already attributes spans to a specific agent.
  Enables cross-framework "spans by agent" rollups without framework-aware
  query logic. Reference impl for upstream
  `semantic-conventions-genai#91` (proposal to standardise
  `gen_ai.agent.name`).
- **Cross-framework conversation correlation: `gen_ai.conversation.id`**
  is now co-emitted alongside the existing `session.id` (and the
  framework-prefixed `crewai.session.id` / `langgraph.session.id` /
  `langchain.session.id` / `bedrock.agent.session_id` /
  `bedrock.rag.session_id` / `bedrock.agent.response.session_id` /
  `google_adk.session_id` / `openai.agent.metadata.session_id`) on every
  framework instrumentor that derives a stable conversation identifier
  from the framework's native primitive:
  - CrewAI: `Crew.id` / kickoff-input session_id, propagated to task +
    agent child spans.
  - LangGraph: `RunnableConfig.configurable.thread_id` (or app-supplied
    session_id).
  - LangChain: resolved via the existing `_resolve_session_id` priority
    chain (kwargs / input dict / `OTelConfig.session_id_extractor`).
  - Bedrock Agents: `InvokeAgent.sessionId` (request + response sides)
    and `RetrieveAndGenerate.sessionId`. Also adds the generic
    `session.id` to these spans for consistency.
  - Google ADK: `Runner.run` `session_id` kwarg. Also adds generic
    `session.id`.
  - OpenAI Agents: `metadata.session_id` if the agent run surfaces one.
  - `BaseInstrumentor`: the `OTelConfig.session_id_extractor` callback
    (used by provider instrumentors) co-emits `gen_ai.conversation.id`
    in addition to `session.id`.

  AutoGen and PydanticAI are deliberately not touched — neither exposes
  a stable conversation primitive on the SDK boundary today (AutoGen's
  `GroupChat.messages[0].id` is not stable across replays; PydanticAI's
  `Agent.run(usage_id=...)` is app-supplied only). Reference impl for
  upstream `semantic-conventions-genai#145` (proposal to document how
  applications correlate sessions via `gen_ai.conversation.id`).

- **Harmonized cross-backend vector DB attribution: `db.collection.name`
  and `db.vector.top_k`** are now co-emitted on every vector DB
  instrumentor that previously used backend-historical names. Five
  spellings collapse to one canonical pair while preserving back-compat:
  - `vector.collection` (Weaviate/Qdrant/Chroma/Milvus) → also
    `db.collection.name`
  - `vector.table` (LanceDB) → also `db.collection.name`
  - `vector.limit` (Qdrant/Milvus) → also `db.vector.top_k`
  - `vector.n_results` (Chroma) → also `db.vector.top_k`
  - `vector.k` (FAISS) → also `db.vector.top_k`

  Reference impl for upstream `semantic-conventions-genai#5` (VectorDB
  semantic conventions). The library is positioned as the most-complete
  real-world vector-DB OTel coverage (7 backends) and the
  `db.vector.top_k` attribute is library-original, intended for upstream
  standardisation.

## [1.2.2] - 2026-05-07

### Added

- **Grok 4.3 pricing** in `genai_otel/llm_pricing.json` (input $1.25/1M,
  output $2.50/1M, 1M context). Registered under both `grok-4.3` and
  `xai/grok-4.3` keys, per `docs.x.ai/developers/models`.

## [1.2.1] - 2026-04-29

### Security

- **Bump `black` to `>=26.3.1` on Python 3.10+** to address
  [GHSA-3936-cmfr-pm3m](https://github.com/advisories/GHSA-3936-cmfr-pm3m)
  (high severity: arbitrary file writes from unsanitized user input in the
  cache file name). Black is dev-only and not in the published wheel, so
  this affects only contributor / CI environments. Python 3.9 dev
  environments stay on the latest 3.9-compatible release (`<26`) since
  the patched version dropped 3.9 support.

### Fixed

- CI matrix install on Python 3.9 — the previous unconditional
  `black>=26.3.1` pin failed `pip install -e ".[dev]"` on 3.9 because the
  fixed version requires 3.10+. Pin is now conditional on
  `python_version`.

## [1.2.0] - 2026-04-29

### Added

- **`genai_otel.cost_estimation` module — public API for token/cost estimation
  in multimodal calls.** Exports `estimate_pipeline_usage`, `estimate_chat_usage`,
  `count_images`, `audio_seconds`, `coerce_text`, `result_text`. Designed to be
  imported by external custom providers (e.g. tracesense / chaos-lab providers
  that bypass `transformers.pipeline` and the standard `ollama` entry points).
- **Token / cost estimation for multimodal calls that omit usage data.**
  - `BaseInstrumentor._estimate_usage(result, request_kwargs)` hook (default
    returns `None`). When `_extract_usage` returns nothing, this fallback
    fires and the resulting span is tagged with
    `gen_ai.usage.token_count_estimated=true` so downstream tooling can
    distinguish exact from estimated counts.
  - `OllamaInstrumentor._estimate_usage`: char-count fallback (4 chars/token)
    plus per-image token floor (256 tokens/image) for `/api/chat` and
    `/api/generate` payloads. Fixes cost being zero on multimodal Ollama
    spans whose responses omit `prompt_eval_count` / `eval_count`.
  - `HuggingFaceInstrumentor._record_pipeline_usage_and_cost`: estimates
    prompt/completion tokens for non-text pipelines including
    `image-text-to-text`, `image-to-text`, `visual-question-answering`,
    `image-classification`, `automatic-speech-recognition`,
    `audio-classification`, `audio-to-audio`, `text-to-image`,
    `text-to-speech`. Emits per-modality attributes
    `gen_ai.usage.image_count` and `gen_ai.usage.audio_seconds` for
    observability of multimodal cost drivers.
- **New image-generation pricing entries:** `gpt-image-1`, `gpt-image-2`
  (low/medium/high quality tiers), `black-forest-labs/FLUX.2-pro`,
  `FLUX.2-max`, `FLUX.2-flex`, `FLUX.2-klein-4b`, `FLUX.2-klein-9b`,
  `FLUX.2-dev`, plus a new `gemini-3-pro-image-preview` alias.

### Changed

- **Refined Gemini image pricing** to current 2026 rates: `nano-banana` /
  `gemini-2-5-flash-image` updated from $0.03/MP to per-resolution
  ($0.039 @ 1024×1024) and new `batch` quality tier (50% off).
  `nano-banana-pro` / `gemini-3-pro-image-preview` updated to $0.134
  for 1K-2K and $0.24 for 4K. `nano-banana-2` /
  `gemini-3.1-flash-image` gain a `batch` tier.
- Imagen 3.0 / 4.0 entries reshaped from scalar floats into the standard
  `{quality: {dimension: price}}` shape so the cost calculator can
  actually evaluate them.

### Fixed

- Non-chat call types (`image`, `audio`, `embedding`, ...) now also set
  `gen_ai.usage.cost.total` on the span. Previously cost was added to the
  metric counter but dropped from span attributes, so backends couldn't
  aggregate cost per image-gen / audio span.

## [1.1.1] - 2026-04-28

### Added

- **Dual-emission of OTel-canonical `gen_ai.input.messages` / `gen_ai.output.messages`**
  alongside the existing flat `gen_ai.prompt.{n}.content.{m}.*` attributes. When
  `OTEL_SEMCONV_STABILITY_OPT_IN` includes `gen_ai`, multimodal content parts are
  also serialized as a JSON blob using the upstream OTel schema's
  `BlobPart` / `FilePart` / `UriPart` / `StrippedPart` shapes
  (`docs/gen-ai/gen-ai-input-messages.json`). New module:
  `genai_otel/media/canonical.py` (public function `build_canonical_messages`).
- 9 new tests covering the canonical mapping + dual-emission gating.

### Changed

- The flat attribute namespace shipped in v1.1.0 is now documented as a
  library-specific convenience for query-friendly backends. The canonical
  upstream shape is the portable form. Both can be emitted simultaneously.

### Documentation

- Upstream PR draft redrafted around the actual gap discovered
  by reading the OTel JSON schemas: the upstream already has `BlobPart` /
  `FilePart` / `UriPart` and a `Modality` enum. The narrowed proposal adds:
  - `document` value to the `Modality` enum
  - Optional `byte_size` field on Blob/File/UriPart (cost-of-capture telemetry)
  - New `StrippedPart` type for fail-closed observability
- Issue #3672 updated with the corrected scope.

## [1.1.0] - 2026-04-28

### Added

- **Pricing for newly released models** — `genai_otel/llm_pricing.json` now covers:
  - **OpenAI GPT-5.5** (Apr 2026) — input $5/1M, output $30/1M
  - **DeepSeek V4 Flash** — input $0.14/1M, output $0.28/1M
  - **DeepSeek V4 Pro** — input $0.435/1M, output $0.87/1M (75% promotional rate until 2026-05-31; standard $1.74/$3.48 per 1M)
- **Multimodal observability** — first-class capture of image, audio, video, and document content parts on OpenAI, Anthropic, Google Gemini, and Groq spans. Defines the open standard for multimodal AI observability via an additive, OTel-compatible attribute namespace. Highlights:
  - New attribute namespace: `gen_ai.prompt.{n}.content.{m}.{type, text, media_uri, media_mime_type, media_byte_size, media_source}` plus `gen_ai.completion.*` mirror and `gen_ai.media.stripped_reason`.
  - Pluggable offload backends — `filesystem`, `s3`, `minio`, `http` — under `genai_otel/media/stores/`. Bytes never appear inline in span attributes.
  - Built-in redactors: `exif_stripper`, `face_blur`, `pdf_pii_redact` (lazy-imported, `multimodal-{images,faces,pdf}` extras).
  - New env vars: `GENAI_OTEL_MEDIA_CAPTURE_MODE` (default `off`), `GENAI_OTEL_MEDIA_STORE`, `GENAI_OTEL_MEDIA_STORE_{ENDPOINT,BUCKET,PREFIX,ACCESS_KEY,SECRET_KEY}`, `GENAI_OTEL_MEDIA_MAX_BYTES`, `GENAI_OTEL_MEDIA_ALLOWED_MODALITIES`, `GENAI_OTEL_MEDIA_REDACTOR`.
  - Default `media_capture_mode=off` keeps text-only behaviour byte-identical for existing users.
  - 41 new unit tests covering provider×modality detection matrix, offload pipeline gating, store backends, redactor graceful-degrade, and per-instrumentor wiring.
  - Live MinIO integration test (skipped unless credentials provided).
  - New examples under `examples/multimodal/` (vision, audio, document, face-blur).
  - New docs: `docs/guides/multimodal.md`.
  - New extras: `multimodal-images`, `multimodal-pdf`, `multimodal-faces`, `multimodal-s3`, umbrella `multimodal`.

## [1.0.5] - 2026-04-22

### Fixed

- **Qdrant instrumentor: avoid wrapping deprecated `search` method** - When both `query_points` (qdrant-client 1.10+) and the legacy `search` method exist, only `query_points` is wrapped. The legacy `search` method emits a `DeprecationWarning` on invocation and was removed entirely in qdrant-client 1.16+, which produced startup log noise for downstream applications. `search` is still wrapped as a fallback on older clients where `query_points` is unavailable. Wrap calls are additionally guarded against `AttributeError`/`ImportError` to survive future SDK changes silently. (`genai_otel/mcp_instrumentors/vector_db_instrumentor.py`)

## [1.0.4] - 2026-04-21

### Added

- **Pricing for 2026-Q2 model releases** - `genai_otel/llm_pricing.json` now covers models released since Feb 2026 across API, Hugging Face, and Ollama tag formats:
  - **Anthropic**: Claude Opus 4.7 (`claude-opus-4-7`, `claude-opus-4.7`)
  - **OpenAI**: GPT-5.3 family (`gpt-5.3`, `gpt-5.3-chat-latest`, `gpt-5.3-codex`) and GPT-5.4 family (`gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.4-pro`)
  - **Google**: Gemini 3.1 Flash Live Preview and Flash-Lite Preview; Gemma 4 series (`google/gemma-4-31B`, `-26B-A4B`, `-E4B`, `-E2B`) with HF + short-form + Ollama aliases
  - **xAI**: Grok 4.20 dated snapshots (`grok-4.20-0309-reasoning`, `-non-reasoning`, `-multi-agent-0309`) and Grok 4.1 Fast (`grok-4-1-fast-reasoning`, `-non-reasoning`)
  - **MiniMax**: M2.7 (`MiniMax-M2.7`, `MiniMax-M2.7-highspeed`) and M2.5 highspeed tier
  - **Zhipu / Z.ai**: GLM-5.1 and GLM-5-Turbo (with `zai/` and `THUDM/` aliases)
  - **Moonshot**: Kimi K2.6 (`kimi-k2.6`, `moonshotai/Kimi-K2.6`) covering both Moonshot first-party and OpenRouter aggregate prices
  - **Alibaba Qwen**: Qwen 3.5 Plus (`qwen3.5-plus`), Qwen 3.6 Plus (`qwen3.6-plus`), Qwen 3.6 35B MoE (`Qwen/Qwen3.6-35B-A3B`)
  - **Sarvam AI**: Sarvam-30B and Sarvam-105B (free tier per sarvam.ai/api-pricing, 22 Indic + English)
  - **Liquid AI**: LFM2-24B-A2B MoE (OpenRouter-verified pricing) and LFM2.5-350M edge model with HF, short-form, and Ollama tags

### Fixed

- **Grok 4.20 pricing correction** - `grok-4.20` prompt/completion prices corrected from the prior estimate ($3/$15 per 1M) to the xAI-documented $2/$6 per 1M tokens

## [1.0.3] - 2026-04-06

### Fixed

- **AsyncOpenAI cost/token extraction regression** - `_record_result_metrics` argument order was swapped in the AsyncOpenAI path, causing cost and token counts to be dropped from async OpenAI spans. Restored the correct argument order so async calls once again emit `gen_ai.usage.*_tokens` and `gen_ai.cost.amount` attributes (116149f)

## [1.0.2] - 2026-03-30

### Fixed

- **AsyncOpenAI client spans not exported** - `OpenAIInstrumentor` now wraps `AsyncOpenAI.__init__` in addition to `OpenAI.__init__`, so async client calls produce spans that are exported to the OTLP collector. Added `_instrument_async_client()` and `_create_async_span_wrapper()` for proper async-aware span creation (fixes #4)
- **Qdrant instrumentation crash on qdrant-client 1.16+** - `QdrantClient.search` was removed in qdrant-client 1.16 in favor of `query_points`. The Qdrant instrumentor now detects which API methods are available and instruments accordingly, supporting both old (`search`) and new (`query_points`) APIs

## [1.0.0] - 2026-03-25

### Changed

- **License changed from AGPL-3.0 to Apache-2.0** - Making the library enterprise-friendly for broader adoption
- **Production-stable release** - Promoted from Beta to Production/Stable status after 45 releases and 5+ months of development
- **Improved PyPI metadata** - Added comprehensive keywords, classifiers, and Python 3.13 support declaration
- **README overhaul** - Unified branding (TraceVerde / genai-otel-instrument), added competitive comparison table, added Users section, removed placeholder content, updated community links

### Added

- **GitHub repository topics** - 20 topics for improved discoverability on GitHub
- **Competitive comparison table** - Feature comparison vs OpenLIT, Traceloop, and Langfuse in README

### Security

- **LiteLLM version pinning** - Excluded compromised versions 1.82.7 and 1.82.8 (supply chain attack, March 2026)

## [0.1.45] - 2026-03-24

### Fixed

- **CrewAI cost enrichment** - CrewAI crew execution spans now include `gen_ai.request.model`, `gen_ai.usage.{prompt,completion,total}_tokens`, and `gen_ai.usage.cost.{total,prompt,completion}` attributes. Previously, cost enrichment was silently skipped because the instrumentor did not set `gen_ai.request.model` (required by `CostEnrichmentSpanProcessor`) and `_extract_usage()` did not properly extract token counts from CrewAI's `UsageMetrics` pydantic model
- **CrewAI token usage extraction** - `_extract_usage()` now correctly handles `CrewOutput.token_usage` (a `UsageMetrics` pydantic model with `total_tokens`, `prompt_tokens`, `cached_prompt_tokens`, `completion_tokens`), includes `cached_prompt_tokens` for Anthropic cache cost calculation, and guards against zero-token results

### Added

- **`_extract_model_from_crew()` helper** - Extracts the LLM model name from a Crew instance by inspecting agents' `llm` attribute (supports both string model names and `BaseLLM`/`LLM` objects with `.model` attr), with fallback to `manager_agent` for hierarchical processes

## [0.1.44] - 2026-03-13

### Added

- **FalkorDB graph database instrumentation** - New MCP instrumentor for FalkorDB that traces Cypher queries (`Graph.query`, `Graph.ro_query`), graph management operations (`Graph.delete`, `Graph.copy`), and graph selection (`FalkorDB.select_graph`) with span attributes for db.system, db.operation, db.name, and db.statement

## [0.1.43] - 2026-03-09

### Added

- **Auto session.id on CrewAI spans** - `crewai.crew.execution` span now sets `session.id` and `crewai.session.id` automatically. Session ID is resolved with priority: `inputs["session_id"]` > `OTelConfig.session_id_extractor` > Crew instance ID > auto-generated UUID. Child spans (`crewai.task.execution`, `crewai.agent.execution`) inherit `session.id` from the parent crew via `task.agent.crew` / `agent.crew` references
- **Auto session.id on LangGraph spans** - `langgraph.graph.invoke`, `.stream`, `.ainvoke`, `.astream` spans now set `session.id` and `langgraph.session.id`. Priority: `input_state["session_id"]` > `config["configurable"]["thread_id"]` > `OTelConfig.session_id_extractor` > auto-generated UUID
- **Auto session.id on LangChain spans** - Chain, agent, and chat model spans (`langchain.chain.*`, `langchain.agent.execute`, `langchain.chat_model.*`) now set `session.id` and `langchain.session.id`. Priority: `kwargs["session_id"]` > input dict `session_id` > `OTelConfig.session_id_extractor` > auto-generated UUID
- **CrewAI in DEFAULT_INSTRUMENTORS** - `"crewai"` added to the default instrumentor list so it is enabled automatically when the library is installed, matching all other framework instrumentors

## [0.1.42] - 2026-03-09

### Fixed

- **TimescaleDB instrumentor crash on psycopg2 C extension cursor** - `psycopg2.extensions.cursor` is an immutable C type; `wrapt` cannot monkey-patch `cursor.execute`. Switched to wrapping `psycopg2.connect` and injecting a cursor subclass with TimescaleDB-specific span creation
- **CI pipeline failures** - Guarded all optional dependency imports (httpx, requests, OTel instrumentor libraries) with `try/except` in MCP instrumentors; pinned Black to 25.11.0 for Python 3.9 compatibility; converted parenthesized context managers to `contextlib.ExitStack`; fixed test mocking for lazy `__getattr__` import pattern

### Added

- **Auto-load `.env` file** - `instrument()` now calls `dotenv.load_dotenv()` if `python-dotenv` is installed, ensuring `OTEL_EXPORTER_OTLP_ENDPOINT` and other env vars from `.env` files are picked up automatically
- **Package name and version in trace resource attributes** - Every span now includes `telemetry.auto.name` (`genai-otel-instrument`) and `telemetry.auto.version` resource attributes

## [0.1.41] - 2026-03-07

### Fixed

- **Replaced `openlit` dependency with internal `semconv.py`** - Moved all semantic convention constants from `openlit/semcov.py` into `genai_otel/semconv.py` and removed the `openlit/` directory entirely
- **Re-entrancy guard for auto-instrumentation** - Added `_INSTRUMENTATION_INITIALIZED` flag to prevent double-wrapping on repeated `instrument()` calls
- **Lazy logging (f-string to %s formatting)** - Fixed 40+ logger calls across the codebase to use `%s` lazy formatting instead of eagerly-evaluated f-strings
- **Security: PII original_text exposure** - `PIIDetectionResult` no longer stores `original_text` when PII is found in REDACT/BLOCK mode
- **Thread-safe MCP metrics initialization** - Added `threading.Lock` to prevent race conditions in `mcp_instrumentors/base.py` shared metrics setup
- **JSON serialization for first_message** - Changed `str(dict)` to `json.dumps()` in `_build_first_message()` for reliable cross-language parsing. Parser updated to try `json.loads` first with `ast.literal_eval` fallback
- **Perspective API timeout** - Added `api_timeout` config (default 30s) to `ToxicityConfig`, applied via `httplib2.Http` to prevent hanging API calls
- **Vector DB instrumentor safety** - Converted Qdrant, ChromaDB, Milvus, FAISS to use `wrapt.wrap_function_wrapper()` instead of direct class method replacement
- **API instrumentor hostname detection** - Replaced string `in` checks with `hostname.endswith()` for more accurate provider detection in `api_instrumentor.py`
- **GPU metrics tests synced to current code** - Fixed 5 GPU metrics tests (counter/gauge counts, codecarbon `stop_task`/`start_task` pattern)
- **Google AI legacy SDK tests** - Fixed 3 tests to properly block `google-genai` when testing legacy `google-generativeai` paths
- **Lazy imports for Elasticsearch/OpenSearch instrumentors** - Moved OTel instrumentor imports inside `instrument()` to prevent `ModuleNotFoundError`

### Changed

- **Reduced core dependencies** - Moved 12 DB/MQ dependencies to optional `[databases]`/`[messaging]` extras
- **Removed eager imports** - Lazy-load evaluation detectors (spacy/torch/presidio) and httpx, reducing import time from ~24s to ~3.3s
- **Deduplicated MeterProvider setup** - Extracted `_setup_meter_provider()` helper, eliminating ~120 lines of duplication
- **Removed dead code** - Cleaned up unused `_OTLP_EXPORTER_SESSIONS`, duplicate imports, emoji log characters
- **OpenInference version ranges** - Changed exact pins (`==0.1.31`) to ranges (`>=0.1.31,<1.0.0`)

### Added

- **Configurable trace sampling rate** - Added `GENAI_SAMPLING_RATE` env var (0.0-1.0, default 1.0) and `sampling_rate` config field. Applies `TraceIdRatioBased` sampler to reduce telemetry volume in high-traffic production
- **Lazy imports in `__init__.py`** - Implemented `__getattr__`-based lazy loading. `import genai_otel` now takes ~16ms instead of ~7300ms (456x improvement). Heavy modules only load on first attribute access
- **Package size CI check** - Added `tests/test_package_size.py` with package size (<5MB) and import time (<500ms) threshold tests
- **Dependency compatibility tests** - Added `tests/test_dependency_compat.py` with 5 tests for graceful degradation with missing/incompatible dependencies
- **Memory leak detection test** - Added `test_instrument_uninstrument_no_memory_leak` using `tracemalloc` across 5 instrument/uninstrument cycles
- **Performance benchmarks** - Added `benchmarks/bench_instrumentation.py` measuring import time, span wrapper overhead (0.085ms/call), and cost calculation (0.015ms/call)
- **Performance tuning guide** - Added `docs/PERFORMANCE_TUNING.md` covering sampling, batching, content capture, GPU metrics, and production checklist
- **Python version compatibility matrix** - Added `docs/PYTHON_COMPATIBILITY.md` documenting Python 3.9-3.13 support
- **Security assessment report** - Added `SECURITY.md` with SAST (bandit), SCA (pip-audit), PII handling, license compliance, and network security documentation
- **`uninstrument()` function** - Clean teardown: stops GPU collector, shuts down TracerProvider/MeterProvider, resets initialization guard for re-instrumentation
- **LanceDB vector DB instrumentation** - Added tracing for LanceDB search, add, create_table, and drop_table operations
- **TimescaleDB instrumentation** - Added tracing for TimescaleDB-specific operations (create_hypertable, time_bucket queries, compression policies, retention policies, continuous aggregates, chunk management)
- **MinIO object storage instrumentation** - Added tracing for MinIO S3-compatible operations (put_object, get_object, remove_object, list_objects, make_bucket, remove_bucket, list_buckets, stat_object, fput_object, fget_object)
- **RabbitMQ message broker instrumentation** - Added tracing for RabbitMQ via pika (basic_publish, basic_consume, basic_get, queue_declare, queue_delete, exchange_declare, exchange_delete) with proper PRODUCER/CONSUMER/CLIENT span kinds
- **OpenSearch instrumentation** - Added tracing for OpenSearch operations via opentelemetry-instrumentation-opensearch-py
- **Elasticsearch instrumentation** - Added tracing for Elasticsearch operations via opentelemetry-instrumentation-elasticsearch

## [0.1.40] - 2026-02-19

### Fixed

- **`Failed to parse first_message: unterminated string literal` warning**
  - All 13 LLM instrumentors now use a centralized `_build_first_message()` helper that truncates content text *before* building the dict string, ensuring `ast.literal_eval()` always receives syntactically valid Python
  - Previously, `str(messages[0])[:200]` truncated the serialized dict mid-string, producing invalid Python that caused parse warnings on every request
  - Added regex fallback in evaluation parsing for any remaining edge cases with truncated strings

- **0ms duration on streaming spans (`astream()`, `run_stream()`)**
  - `create_span_wrapper()` now handles async generators (`inspect.isasyncgen`) and sync generators (`inspect.isgenerator`) in addition to coroutines
  - Previously, streaming methods that return generators fell through to the sync code path, ending the span immediately (0ms) while actual iteration happened later
  - Generator wrappers keep the span open during iteration, properly recording duration, errors, and metrics

### Added

- **`GENAI_CONTENT_MAX_LENGTH` environment variable** for controlling maximum captured content length
  - Default: 200 characters (current behavior). Set to 0 for no limit (full content capture)
  - Only applies when `GENAI_ENABLE_CONTENT_CAPTURE=true`
  - Configurable via `OTelConfig(content_max_length=500)` or environment variable

- **`_build_first_message()` helper on `BaseInstrumentor`**
  - Centralized, config-aware method for building `gen_ai.request.first_message` span attributes
  - Respects `enable_content_capture` and `content_max_length` configuration
  - Handles both dict-style messages and plain string messages

## [0.1.39] - 2026-02-19

### Fixed

- **Double-wrapping bug in framework instrumentors (CrewAI, Google ADK, AutoGen, OpenAI Agents)**
  - Framework instrumentors used `wrapt.FunctionWrapper` with a callback that called `create_span_wrapper()`, resulting in double-wrapped functions (two spans per call, duplicate metrics)
  - Replaced with direct `create_span_wrapper()` application matching the pattern used by all LLM provider instrumentors
  - All 4 framework instrumentors now correctly create a single span per operation

## [0.1.38] - 2026-02-18

### Fixed

- **CrewAI trace hierarchy: spans now form proper parent-child trees**
  - Removed `_propagate_context()` method which was conflicting with `create_span_wrapper()` in the base class, causing all CrewAI spans (crew, task, agent) to be flat/disconnected instead of forming a proper trace hierarchy
  - `create_span_wrapper()` already correctly handles context via `trace.set_span_in_context()` + `otel_context.attach()`, making the extra context wrapping redundant and harmful
  - Task and agent spans now correctly appear as children of the crew span, and LLM calls appear as children of agent spans

### Added

- **Google ADK (Agent Development Kit) instrumentor**
  - New `GoogleADKInstrumentor` for Google's open-source agent framework (`google-adk` on PyPI)
  - Instruments `Runner.run_async()` and `InMemoryRunner.run_debug()` with automatic span creation
  - Captures agent name, model, app name, tools, sub-agents, user/session IDs
  - `gen_ai.system = "google_adk"` with operation names `runner.run` and `runner.run_debug`

- **AutoGen AgentChat (v0.4+) instrumentor**
  - New `AutoGenAgentChatInstrumentor` for the newer AutoGen AgentChat framework (`autogen-agentchat` package)
  - Instruments `ChatAgent.run()`, `run_stream()`, `on_messages()` for agent execution
  - Instruments `BaseGroupChat.run()`, `run_stream()` covering all team types (RoundRobinGroupChat, SelectorGroupChat, Swarm, MagenticOneGroupChat)
  - Captures agent name/type, team participants, task content, termination conditions, stop reasons
  - `gen_ai.system = "autogen_agentchat"` with operation names for agent and team execution

- **CrewAI full async support**
  - Now instruments all 6 kickoff variants: `kickoff()`, `kickoff_async()`, `akickoff()`, `kickoff_for_each()`, `kickoff_for_each_async()`, `akickoff_for_each()`
  - Added `gen_ai.system` and `gen_ai.operation.name` attributes to task and agent spans for consistency

- **24 new tests** for Google ADK instrumentor (12), AutoGen AgentChat instrumentor (12), and updated CrewAI tests

## [0.1.37] - 2026-02-18

### Fixed

- **Critical: `create_span_wrapper` does not support async functions**
  - `BaseInstrumentor.create_span_wrapper()` now detects when wrapped functions return coroutines and handles the span lifecycle asynchronously
  - Previously, wrapping async methods (e.g., LangGraph's `ainvoke()`, CrewAI's async `kickoff()`, OpenAI Agents' `Runner.run()`) caused parent spans to start and end instantly before the coroutine executed, and nested async LLM calls became disconnected root spans
  - The async path keeps the span open and OTel context attached until the coroutine completes, ensuring proper parent-child trace hierarchy for all async framework instrumentors

### Added

- **10 new async tests** for `create_span_wrapper` covering coroutine awaiting, metrics recording, exception handling, context lifecycle, evaluation checks, and server metrics cleanup

## [0.1.36] - 2026-02-18

### Fixed

- **Critical: Evaluation attributes missing on OpenInference LiteLLM spans**
  - `LiteLLMSpanEnrichmentProcessor._is_litellm_span()` failed to recognize spans from OpenInference LiteLLM instrumentor v0.1.19+ which uses bare function names (e.g., `acompletion`) instead of prefixed names (`litellm.acompletion`), and `llm.model_name` instead of `llm.model`
  - Now checks `instrumentation_scope.name` as the primary detection method, recognizes bare function names (`acompletion`, `completion`, `aembedding`, etc.), and checks `llm.model_name` attribute
  - Added scope name verification for attribute-based checks to prevent false positives

- **Critical: Evaluation attributes silently lost on ReadableSpan**
  - `EvaluationSpanProcessor.on_end()` received `ReadableSpan` (immutable) and called `span.set_attribute()` which silently failed via `except AttributeError: pass`, discarding all PII, toxicity, bias, prompt injection, and hallucination detection results
  - Fixed `safe_set_attribute()` to fall back to `_attributes` direct access on `ReadableSpan` (matching the pattern used by `LiteLLMSpanEnrichmentProcessor._set_attribute()`)
  - Created `EvaluationEnrichingSpanExporter` following the `CostEnrichingSpanExporter` pattern: wraps the exporter chain, runs all evaluation detectors, and creates new enriched `ReadableSpan` objects with evaluation attributes before export

- **LiteLLM span content extraction for OpenInference indexed attributes**
  - `_extract_request_content()` now handles `llm.input_messages.0.message.content` (indexed attribute format used by OpenInference v0.1.19+) and JSON-formatted `input.value` containing a `messages` array
  - `_extract_response_content()` now handles `llm.output_messages.0.message.content` indexed attributes

### Added

- **`trace_operation()` context manager for trace hierarchy**
  - New `genai_otel.tracing.trace_operation(name, attributes)` creates parent spans that group nested LLM calls into a single trace, solving the single-span trace problem
  - All instrumented calls (LiteLLM, OpenAI, etc.) within the context automatically become child spans
  - Exported from `genai_otel` package for easy access: `from genai_otel import trace_operation`

- **`EvaluationEnrichingSpanExporter`** - New span exporter that enriches spans with evaluation attributes (PII, toxicity, bias, prompt injection, restricted topics, hallucination) at export time, ensuring evaluation works for all span types including OpenInference spans

- **32 new tests**
  - 10 new tests for `LiteLLMSpanEnrichmentProcessor` (scope-based detection, bare function names, indexed attributes, JSON input.value, OpenInference LiteLLM spans)
  - 13 new tests for `EvaluationEnrichingSpanExporter` (all detector types, multiple detectors, exception handling, skip-if-evaluated)
  - 9 new tests for `trace_operation` (span creation, attributes, span kind, exception propagation, package import)

## [0.1.35] - 2026-02-17

### Fixed

- **Sarvam AI cost tracking for all non-chat operations**
  - Translate, transliterate, language detection, STT, and TTS operations now record character-based cost via `_record_sarvam_cost()`
  - Pricing lookup uses `speech_to_text` category in `llm_pricing.json` (per 1K characters)
  - Chat completions `start_time` fixed from `0` to `time.time()` for accurate latency measurement

- **Sarvam AI model names on all spans**
  - Translate spans now show `gen_ai.request.model = mayura:v1` (or `sarvam-translate:v1` when explicit)
  - Transliterate spans now show `gen_ai.request.model = sarvam-transliterate`
  - Language detection spans now show `gen_ai.request.model = sarvam-detect-language`
  - TTS default model corrected from `bulbul` to `bulbul-v2` to match actual SDK behavior
  - TTS model normalization: `bulbul:v3` -> `bulbul-v3` for pricing key lookup

- **Sarvam SDK OMIT sentinel handling**
  - New `_safe_kwarg()` helper detects SDK `OMIT`/`NotGiven` sentinel objects in optional parameters
  - Prevents instrumentation errors when optional params are not provided by the user

### Added

- **Comprehensive Sarvam-specific metadata capture on all spans**
  - Translate: `sarvam.translate.mode`, `sarvam.translate.speaker_gender`, `sarvam.translate.numerals_format`, `sarvam.translate.output_script`, `sarvam.translate.enable_preprocessing`
  - Transliterate: `sarvam.transliterate.numerals_format`, `sarvam.transliterate.spoken_form`, `sarvam.transliterate.spoken_form_numerals_language`
  - TTS: `sarvam.tts.pace`, `sarvam.tts.temperature`, `sarvam.tts.pitch`, `sarvam.tts.loudness`, `sarvam.tts.speech_sample_rate`, `sarvam.tts.enable_preprocessing`, `sarvam.tts.output_audio_codec`
  - STT: `sarvam.stt.mode`, `sarvam.stt.input_audio_codec`, `sarvam.stt.prompt`
  - All spans: `gen_ai.usage.characters` and `gen_ai.usage.cost.total`

- **New Sarvam AI example files**
  - `examples/sarvam/bulbul_v3_tts_example.py`: Bulbul v3 TTS with 48 speakers, pace/temperature control, all 11 languages
  - `examples/sarvam/mayura_translate_example.py`: Mayura v1 translation with modes, speaker gender, numerals format, model comparison, transliteration

- **15 new tests for Sarvam instrumentor**
  - Tests for `_safe_kwarg()`, `_normalize_sarvam_tts_model()`, `_record_sarvam_cost()`, metadata capture, model attributes, start_time validation
  - Total Sarvam tests: 37 (up from 22)

## [0.1.34] - 2026-02-11

### Fixed

- **Critical: CostEnrichingSpanExporter now plugged into the pipeline**
  - `CostEnrichingSpanExporter` was defined in `cost_enriching_exporter.py` but never used in the span export pipeline
  - The raw `OTLPSpanExporter` was passed directly to `BatchSpanProcessor`, bypassing cost enrichment for all OpenInference spans (LiteLLM, smolagents, MCP)
  - Now wraps the span exporter with `CostEnrichingSpanExporter` when cost tracking is enabled, for both OTLP and Console exporter paths
  - This immediately enables cost tracking for all OpenInference-instrumented spans

- **Critical: Context propagation fixed in BaseInstrumentor**
  - `create_span_wrapper()` used `start_span()` which creates a span but does NOT set it as the current active span in the context
  - Nested calls (e.g., tool calls within an LLM call) would not inherit the parent span, breaking trace hierarchy
  - Now uses `otel_context.attach(trace.set_span_in_context(span))` to properly activate the span, with `detach()` in all code paths (success, error, streaming)

- **CostEnrichmentSpanProcessor no longer attempts to mutate immutable ReadableSpan**
  - `on_end()` receives `ReadableSpan` which does not support `set_attribute()`, causing "Span is not mutable" warnings
  - Processor now serves as a logging/monitoring hook for cost calculations; actual attribute enrichment is handled by `CostEnrichingSpanExporter` at export time

- **LiteLLMSpanEnrichmentProcessor attribute setting made robust**
  - `_set_attribute()` was trying to assign to `ReadableSpan._attributes` which could fail depending on the underlying type
  - Now prefers `set_attribute()` when available (mutable Span), with fallback to `_attributes` direct access (works with `BoundedAttributes` in OTel SDK 1.38+)
  - Added proper error handling around both code paths

- **Evaluation processor now extracts prompts/responses from OpenInference spans**
  - `_extract_prompt()` was missing `llm.input_messages` and `input.value` attribute keys used by OpenInference LiteLLM/smolagents spans
  - `_extract_response()` was missing `llm.output_messages` and `output.value` attribute keys
  - Added JSON parsing support for OpenInference message formats, including nested `message.content` structures
  - Evaluation features (PII detection, toxicity, bias, prompt injection, hallucination) now work for LiteLLM-proxied providers

## [0.1.33] - 2026-02-10

### Fixed

- **Sarvam AI Instrumentor Fixes**
  - Fixed Sarvam instrumentor `__init__` returning instance instead of None
  - Added `sarvamai` to `DEFAULT_INSTRUMENTORS` for automatic enablement
  - Added audio playback to all Sarvam examples and fixed invalid speaker names

### Added

- Added `sarvamai` as optional dependency in `pyproject.toml`
- Added `SARVAM_API_KEY` and `SAMBANOVA_API_KEY` to `sample.env`

## [0.1.32] - 2026-02-10

### Added

- **Sarvam AI Instrumentation**
  - New `SarvamAIInstrumentor` for Sarvam AI sovereign Indian AI platform
  - Full instrumentation for Sarvam AI translate, transliterate, text-to-speech, and speech-to-text APIs
  - Token usage tracking and cost calculation for Sarvam AI models
  - Updated `llm_pricing.json` with Sarvam AI model pricing
  - Added complex multilingual pipeline example for Sarvam AI
  - Added Sarvam Arya agent orchestration example

## [0.1.31] - 2026-01-24

### Added

- **Enhanced GPU Metrics Collection**
  - Added 17 new GPU metrics for comprehensive NVIDIA GPU monitoring:
    - **Per-GPU metrics:**
      - `gen_ai.gpu.memory.utilization`: Memory controller utilization percentage
      - `gen_ai.gpu.power.limit`: GPU power limit in Watts
      - `gen_ai.gpu.clock.sm`: SM (streaming multiprocessor) clock speed in MHz
      - `gen_ai.gpu.clock.memory`: Memory clock speed in MHz
      - `gen_ai.gpu.fan.speed`: Fan speed percentage
      - `gen_ai.gpu.performance.state`: GPU P-state (0=P0 highest performance, 15=P15 lowest)
      - `gen_ai.gpu.pcie.tx`: PCIe transmit throughput in KB/s
      - `gen_ai.gpu.pcie.rx`: PCIe receive throughput in KB/s
      - `gen_ai.gpu.throttle.thermal`: Thermal throttling indicator (0/1)
      - `gen_ai.gpu.throttle.power`: Power throttling indicator (0/1)
      - `gen_ai.gpu.throttle.hw_slowdown`: Hardware slowdown indicator (0/1)
      - `gen_ai.gpu.ecc.errors.corrected`: ECC corrected memory errors count
      - `gen_ai.gpu.ecc.errors.uncorrected`: ECC uncorrected memory errors count
    - **Aggregate metrics (across all GPUs):**
      - `gen_ai.gpu.aggregate.mean_utilization`: Mean GPU utilization across all GPUs
      - `gen_ai.gpu.aggregate.total_memory_used`: Total GPU memory used across all GPUs (GiB)
      - `gen_ai.gpu.aggregate.total_power`: Total power consumption across all GPUs (W)
      - `gen_ai.gpu.aggregate.max_temperature`: Maximum temperature across all GPUs (Celsius)
  - All new metrics use pynvml (nvidia-ml-py) for data collection
  - Graceful handling for GPUs that don't support certain metrics (e.g., ECC, fan speed on passively cooled GPUs)
  - Aggregate metrics include `gpu_count` attribute for context

## [0.1.30] - 2026-01-09

### Added

- **Evaluation Metrics Enhancement - 150% Coverage Increase**
  - **Evaluation support increased from 6/31 (19%) to 15/31 (48%) providers**
  - Added 9 new providers with full evaluation metrics (PII, toxicity, bias, prompt injection, hallucination detection)
  - Total of 102 new tests added with 92% average coverage across new features

- **Span Enrichment Processors for External Instrumentors**
  - New post-processing architecture for adding evaluation support to externally-managed instrumentors
  - **LiteLLM Span Enrichment Processor**
    - Enables evaluation for all 100+ LiteLLM-proxied providers
    - Transforms OpenInference attributes to evaluation format
    - 28 unit tests, 92% coverage
    - No modifications to OpenInference library required
  - **Smolagents Span Enrichment Processor**
    - Adds evaluation support to HuggingFace Smolagents framework
    - Extracts content from agent spans for evaluation
    - 27 unit tests, 91% coverage
  - **MCP Span Enrichment Processor**
    - Enables evaluation for Model Context Protocol tools
    - Supports database, cache, vector DB, and API tool spans
    - 24 unit tests, 92% coverage
  - All processors integrated into `auto_instrument.py` and enabled by default

- **Direct Provider Evaluation Support (6 providers)**
  - **SambaNova** - Added response content capture for full evaluation support
  - **Cohere** - Added request and response capture, 12 tests (90% coverage)
  - **Mistral AI** - Added request/response capture with dict/object format support, 8 tests (38% coverage)
  - **Groq** - Added OpenAI-compatible request/response capture, 14 tests (90% coverage)
  - **Azure OpenAI** - Added support for messages and prompt formats, 15 tests (91% coverage)
  - **AWS Bedrock** - Added multi-model family support (Claude, Llama, Titan), 30 tests (92% coverage)
    - Supports multiple request formats: messages, prompt, inputText
    - Supports multiple response formats: content arrays, completion, outputText, results array

- **OpenRouter Provider Support**
  - Added comprehensive OpenRouter instrumentation for unified multi-provider LLM access
  - Automatic detection of OpenRouter clients via `base_url` checking for `openrouter.ai`
  - Captures OpenRouter-specific parameters: `provider` (routing preferences) and `route` (fallback strategy)
  - Full support for token usage tracking, cost calculation, and response attributes
  - Added 19 popular OpenRouter model pricing entries (Claude, GPT, Gemini, Llama, Mistral, DeepSeek, Perplexity)
  - OpenRouter uses OpenAI-compatible SDK with custom base_url: `https://openrouter.ai/api/v1`
  - Enabled by default in `DEFAULT_INSTRUMENTORS` list
  - Comprehensive test suite with 18 unit tests (72% coverage)
  - Install with: `pip install genai-otel-instrument[openrouter]` or use existing OpenAI SDK
  - Example: `examples/openrouter/example.py`
  - Documentation: Updated `sample.env` with `OPENROUTER_API_KEY` configuration

## [0.1.29] - 2026-01-03

### Fixed

- **Critical: Evaluation Metrics Not Captured for HuggingFace**
  - Fixed critical bug where evaluation metrics (PII, bias, toxicity detection) were not being captured for HuggingFace instrumented spans
  - HuggingFace instrumentor's custom wrapper was missing the call to `_run_evaluation_checks()`
  - Added evaluation checks before span ends in `generate_wrapper()`
  - Implemented tokenizer instrumentation with thread-local storage to preserve original text
  - All evaluation features now work correctly for HuggingFace Transformers

- **Content Capture Format Standardization**
  - Standardized `gen_ai.request.first_message` format across all instrumentors to dict-string: `{'role': 'user', 'content': '...'}`
  - Simplified BaseInstrumentor prompt extraction logic to handle single consistent format
  - Updated HuggingFace, Ollama, Anthropic, and OpenAI instrumentors for consistency
  - Set `gen_ai.response` attribute for evaluation processor in all instrumentors

### Added

- **Comprehensive Evaluation Examples**
  - Added 5 new HuggingFace evaluation examples:
    - `examples/huggingface/pii_example.py` - PII detection with Qwen model
    - `examples/huggingface/bias_example.py` - Bias detection
    - `examples/huggingface/toxicity_example.py` - Toxicity detection
    - `examples/huggingface/hallucination_example.py` - Hallucination detection with context
    - `examples/huggingface/multiple_evaluations_example.py` - Combined PII, bias, and toxicity
  - Added 4 new Ollama evaluation examples:
    - `examples/ollama/pii_detection_example.py` - PII detection with local model
    - `examples/ollama/toxicity_detection_example.py` - Toxicity detection
    - `examples/ollama/hallucination_detection_example.py` - Hallucination detection
    - `examples/ollama/multiple_evaluations_detection_example.py` - Combined evaluations
  - All examples demonstrate proper content capture configuration

## [0.1.28] - 2025-12-30

### Added

- **AMD GPU Monitoring Support**
  - Added `AMDGPUCollector` class for AMD GPU metrics via `amdsmi` library
  - Multi-vendor GPU architecture supporting both NVIDIA and AMD GPUs simultaneously
  - New installation extras:
    - `pip install genai-otel-instrument[amd-gpu]` - AMD GPU support only
    - `pip install genai-otel-instrument[all-gpu]` - Both NVIDIA and AMD GPU support
  - AMD GPU metrics collected:
    - GPU utilization (gfx_activity)
    - Memory usage (VRAM in MiB)
    - Total memory capacity
    - Temperature (junction temperature)
    - Power consumption (average power in Watts)
  - Unified observable callbacks combine metrics from both GPU vendors
  - Graceful fallback when only one vendor's GPUs are present

- **Moonshot AI Kimi Models Pricing**
  - Added pricing for 10 Moonshot AI Kimi models:
    - Kimi-K2-Instruct (flagship 1T parameters MoE)
    - Kimi-K2-Base
    - Kimi-K2-Thinking (reasoning model with thinking)
    - Kimi-Dev-72B (73B parameters)
    - Kimi-Linear-48B-A3B-Instruct & Base (MoE with 3B active, Kimi Delta Attention)
    - Kimi-VL-A3B-Instruct, Thinking, and Thinking-2506 (vision-language models, 16B parameters)

- **New Blocking Mode Examples**
  - `examples/prompt_injection/blocking_mode.py` - Demonstrates jailbreak and system override blocking
  - `examples/restricted_topics/blocking_mode.py` - Demonstrates medical/legal advice and self-harm blocking

- **Multi-Provider Evaluation Examples**
  - `examples/anthropic/pii_detection_example.py` - PII detection with Claude
  - `examples/anthropic/toxicity_detection_example.py` - Toxicity detection with Claude
  - `examples/ollama/bias_detection_example.py` - Bias detection with local Llama2
  - `examples/huggingface/prompt_injection_example.py` - Prompt injection with HF Transformers
  - `examples/mistralai/hallucination_detection_example.py` - Hallucination detection with Mistral
  - Demonstrates evaluation features work across ALL supported LLM providers

- **Environment Variable Documentation**
  - Added 4 new block-on-detection parameters to `sample.env`:
    - `GENAI_TOXICITY_BLOCK_ON_DETECTION`
    - `GENAI_BIAS_BLOCK_ON_DETECTION`
    - `GENAI_PROMPT_INJECTION_BLOCK_ON_DETECTION`
    - `GENAI_RESTRICTED_TOPICS_BLOCK_ON_DETECTION`
  - Each includes description and usage notes

- **Validation Script Updates**
  - Added multi-provider evaluation examples section
  - Now tests Anthropic, Ollama, HuggingFace, and Mistral examples
  - Validates 40+ examples across all evaluation types and providers

### Fixed

- **Critical: Missing Block-on-Detection Parameters**
  - Fixed critical bug where `*_block_on_detection` parameters were NOT exposed in `OTelConfig`
  - ALL blocking mode examples were silently failing with TypeError
  - Added missing parameters: `toxicity_block_on_detection`, `bias_block_on_detection`, `prompt_injection_block_on_detection`, `restricted_topics_block_on_detection`
  - Wired parameters through to detector configs in `auto_instrument.py`
  - Blocking mode now fully functional for all evaluation types

- **Evaluation Detection Thresholds**
  - Lowered PII detection threshold from 0.7 to 0.5 (Presidio scores 0.5-0.7 for valid PII)
  - Lowered Bias detection threshold from 0.5 to 0.4 (pattern matching scores 0.3-0.5)
  - Lowered Prompt Injection threshold from 0.7 to 0.5 (injection patterns score 0.5-0.7)
  - Updated environment variable defaults in `config.py`
  - Updated documentation in `sample.env` and `README.md`

- **Evaluation Test Thresholds**
  - Updated test expectations to match current config defaults
  - PII detection threshold test: 0.7 → 0.5
  - Bias detection threshold test: 0.5 → 0.4
  - Prompt injection threshold test: 0.7 → 0.5
  - Fixes test failures that were blocking PyPI publish

- **GPU Metrics Tests**
  - Updated tests to handle multi-vendor GPU architecture
  - Fixed mock fixtures to support 4 counters (CO2, power cost, energy consumed, total energy)
  - Updated warning messages for AMD+NVIDIA support
  - Tests now properly handle both NVIDIA and AMD GPU scenarios

- **PII Blocking Example Content**
  - Updated `examples/pii_detection/blocking_mode.py` to use reliably detectable PII
  - Changed from undetectable passport number to email + phone number
  - Now properly triggers blocked metrics in Prometheus

- **Unicode Encoding Error**
  - Fixed Unicode arrow character in `bias_detection/custom_threshold.py`
  - Changed `→` to `->` for Windows console compatibility
  - Test now passes (was failing validation)

### Changed

- Updated `gpu_metrics.py` docstring to reflect multi-vendor support
- Warning messages now mention both nvidia-ml-py and amdsmi libraries
- Installation instructions updated to recommend `[all-gpu]` extra for full GPU support
- Updated default thresholds in `genai_otel/evaluation/config.py`
- Updated default environment variables in `genai_otel/config.py`
- Enhanced validation script with multi-provider support

## [0.1.27] - 2025-12-30

### Fixed

- **PII Evaluation Attributes Export to Jaeger**
  - Fixed critical issue where PII evaluation attributes were not appearing in Jaeger traces
  - Root cause: Attributes were being set AFTER `span.end()` when span becomes immutable (ReadableSpan)
  - Solution: Added `_run_evaluation_checks()` method in `BaseInstrumentor` that runs BEFORE `span.end()`
  - PII attributes now successfully exported: `evaluation.pii.prompt.detected`, `evaluation.pii.prompt.entity_count`, `evaluation.pii.prompt.entity_types`, etc.
  - Applies to both PII and Toxicity detection attributes

- **Editable Installation Issue**
  - Fixed issue where examples were using old code due to non-editable pip install
  - Package must be installed with `pip install -e .` for development to reflect local code changes
  - Added clear documentation in scripts/README.md

### Added

- **Comprehensive Example Organization**
  - Reorganized examples into dedicated folders:
    - `examples/pii_detection/` - 10 PII detection examples (detect, redact, block modes + compliance)
    - `examples/toxicity_detection/` - 8 toxicity detection examples (Detoxify, Perspective API, categories)
    - `examples/bias_detection/` - Placeholder for future bias detection
  - All examples updated to use `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable

- **Examples Validation Script**
  - New `scripts/validate_examples.sh` for Linux/Mac
  - New `scripts/validate_examples.bat` for Windows
  - Features:
    - `--dry-run` - List all examples without running
    - `--verbose` - Show detailed output from examples
    - `--timeout N` - Configurable timeout (default: 90s)
    - `--help` - Show usage information
  - Validates all PII, Toxicity, and Bias detection examples
  - Color-coded output (PASSED/FAILED/SKIPPED)
  - Comprehensive summary with failed/skipped example lists

- **Scripts Documentation**
  - New `scripts/README.md` with comprehensive usage guide
  - Moved temporary test files to `scripts/` folder for organization
  - Moved `SOLUTION_SUMMARY.md` and `VALIDATION_REPORT.md` to `scripts/` folder

### Changed

- **Example Files Updated**
  - All PII examples now use `os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")`
  - All Toxicity examples now use `os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")`
  - Bias detection placeholder updated with env var pattern
  - Total: 19 example files updated for flexible endpoint configuration

- **File Organization**
  - Moved debug/test scripts to `scripts/` folder:
    - test_*.py files (9 temporary test scripts)
    - SOLUTION_SUMMARY.md (PII/Toxicity solution documentation)
    - VALIDATION_REPORT.md (Comprehensive validation report)

## [0.1.26] - 2025-12-09

### Added

- **Comprehensive Codecarbon Metrics Exposure**
  - Exposes ALL codecarbon EmissionsData fields as OpenTelemetry metrics
  - New metric: `gen_ai.power.consumption` - Power consumption by component (CPU/GPU/RAM) in Watts
  - New metric: `gen_ai.energy.total` - Total energy consumed (sum of CPU+GPU+RAM) in kWh
  - New metric: `gen_ai.codecarbon.task.duration` - Duration of monitoring tasks in seconds
  - Enhanced all codecarbon metrics with complete hardware and system metadata attributes:
    - Hardware: os, python_version, cpu_count, cpu_model, gpu_count, gpu_model
    - Cloud infrastructure: on_cloud, cloud_provider, cloud_region
    - Location: country, region
  - Implements full codecarbon output specification from https://mlco2.github.io/codecarbon/output.html

### Fixed

- **Codecarbon Verbose Logging Suppression**
  - Suppressed codecarbon's informational warnings by default (CPU tracking mode, multiple instances, etc.)
  - Added `GENAI_CODECARBON_LOG_LEVEL` environment variable (default: "error")
  - Users can re-enable warnings by setting `GENAI_CODECARBON_LOG_LEVEL=warning`
  - Eliminates console noise while preserving ability to enable diagnostics when needed

## [0.1.25] - 2025-12-09

### Fixed

- **Codecarbon CO2 Tracking: Task API Integration**
  - Fixed codecarbon emissions tracking by migrating from private API (_total_emissions.total) to public Task API
  - Uses stop_task/start_task cycle for continuous monitoring without accessing internals
  - Resolves compatibility issues with codecarbon 3.0.7+ where internal APIs changed
  - Properly handles task lifecycle in start() and stop() methods
  - Added enhanced logging with detailed energy breakdown

- **Codecarbon Additional Metrics**
  - New metric: `gen_ai.energy.consumed` - Energy consumption by component (CPU/GPU/RAM) in kWh
  - New metric: `gen_ai.co2.emissions_rate` - CO2 emissions rate in gCO2e/s
  - Records energy breakdown from codecarbon's EmissionsData object
  - Tracks emissions by country and region from codecarbon's location detection

### Added

- **LLM Pricing Database: Comprehensive Update - 293 New Models**
  - **DeepSeek Models** (10 new models):
    - DeepSeek V3.2 - official release with 50% cost reduction ($0.00028/$0.00042 cache hit)
    - DeepSeek V3.2-Speciale - optimized for complex reasoning, AIME 2025 96.0% accuracy
    - DeepSeek-Prover-V2-7B - formal theorem proving in Lean 4 ($0.0002/$0.0003)
    - DeepSeek-Prover-V2-671B - large-scale theorem proving ($0.0007/$0.0025)
    - DeepSeekMath V2 series - mathematical reasoning specialists
    - deepseek/ prefix variants for API compatibility
  - **xAI Grok Models** (29 new models):
    - Grok 4 - latest flagship model ($0.003/$0.015)
    - Grok 4.1 Fast - near-frontier capability ($0.0002/$0.0005)
    - Grok 3 and Grok 3 Mini - various performance tiers
    - Complete xai/ prefix variants for all Grok 2, 3, 4 models
  - **Zhipu AI GLM Models** (7 new models):
    - GLM-4.5 - 355B params, 32B active ($0.0006/$0.0022)
    - GLM-4.5-Air - 106B params, 12B active ($0.0002/$0.0011)
    - GLM-4.6 - latest version with improved performance
    - zai/ prefix variants for all GLM-4.5/4.6 models
  - **OpenAI Models** (42 new models):
    - o1-2024-12-17 - latest reasoning model ($0.015/$0.06)
    - o3-mini, o3-pro, o4-mini - next-gen reasoning models
    - GPT-5 series - gpt-5-chat, gpt-5-codex, gpt-5-mini, gpt-5-nano
    - GPT-5.1 series - latest versions with improved capabilities
    - Historical versions: gpt-3.5-turbo variants, gpt-4 32k models
  - **Anthropic Claude Models** (13 new models):
    - Claude 4 Sonnet 20250514 - latest Claude 4 Sonnet ($0.003/$0.015)
    - Claude Opus 4.5 and 4.5-20251101 - highest capability tier ($0.005/$0.025)
    - Claude 3.7 Sonnet variants - extended context versions
    - Latest aliases: claude-3-5-sonnet-latest, claude-3-opus-latest
  - **Google Gemini Models** (58 new models):
    - Gemini 2.5 Flash and Pro - latest generation models
    - Gemini 2.0 Flash variants - optimized for speed
    - Gemini 3 Pro Preview - upcoming flagship model ($0.002/$0.012)
    - Complete gemini/ prefix variants for all models
    - Experimental and preview versions with free access
  - **Mistral Models** (66 new models):
    - Mistral Large 3 - 675B total, 41B active ($0.0005/$0.0015)
    - Ministral 3 series (9 models) - compact high-performance variants
    - Codestral, Devstral, Magistral series - specialized models
    - Complete mistral/ prefix variants for all models
  - **Qwen Models** (10 new models):
    - Qwen3 Max, Qwen Plus variants via dashscope/ prefix
    - QwQ-Plus - thinking mode support
    - Qwen Turbo - cost-effective fast inference
  - **Cohere Command Models** (4 new models):
    - Command-A-03-2025 - latest flagship model
    - Command-R and Command-R-Plus 08-2024 variants
    - Command-R7B-12-2024 - compact efficient model
  - **AI21 Jamba Models** (7 new models):
    - Jamba 1.5, 1.6, 1.7 series - hybrid SSM-Transformer architecture
    - Large and Mini variants for different scale needs
  - **Additional Providers** (62 new models):
    - Moonshot Kimi models - Chinese market leaders with thinking mode
    - Together.AI and Fireworks.AI pricing tiers
    - Luminous models from Aleph Alpha
    - Morph and v0 specialized models
  - **Implementation Summary**:
    - Total additions: 293 new models (24 DeepSeek additions + 269 from comprehensive LiteLLM comparison)
    - Total model coverage: **649 chat completion models** across 25+ providers
    - Database: `genai_otel/llm_pricing.json`
    - All prices in USD per 1K tokens (prompt/completion)
    - Includes latest 2025 model releases through December

- **Embedding, Image, and Audio Models: Comprehensive Expansion - 114 New Models**
  - **Embedding Models** (47 new models):
    - Google embeddings: gemini-embedding-001, text-embedding-004/005
    - Cohere embeddings: embed-v4.0, embed-english-v3, embed-multilingual-v3
    - Cohere rerankers: rerank-v3.5, rerank-english-v3.0, rerank-multilingual-v3.0
    - Mistral embeddings: codestral-embed, codestral-embed-2505, mistral-embed
    - Amazon Titan: titan-embed-image-v1 (multimodal embedding)
    - Together.AI and Fireworks.AI embedding pricing tiers
    - Voyage rerankers: rerank-2, rerank-2-lite
    - Jina reranker: jina-reranker-v2-base-multilingual
    - Doubao (ByteDance) embedding models
    - NVIDIA reranker models
    - Total embeddings: **81 models** (was 34, +138% increase)
  - **Image Generation Models** (14 new models):
    - Google Imagen: imagen-3.0, imagen-4.0 variants (fast, standard, ultra)
    - Amazon Titan: titan-image-generator-v2
    - Recraft: recraftv2, recraftv3
    - FLUX models: FLUX-1.1-pro, FLUX.1-Kontext-pro
    - DALL-E 3 quality variants: standard and HD for various sizes
    - Total image models: **28 models** (was 14, +100% increase)
  - **Audio/Speech Models** (53 new models):
    - Deepgram models (33 models):
      * Nova 2 and 3 series - specialized for different use cases
      * Base and Enhanced tiers for various domains (finance, medical, meeting, phonecall, etc.)
      * Whisper variants (tiny, small, medium, large, base)
    - OpenAI audio models:
      * gpt-4o-audio-preview variants (2024-10-01, 2024-12-17, 2025-06-03)
      * gpt-4o-mini-audio-preview models
      * gpt-4o-transcribe and gpt-4o-mini-transcribe
      * gpt-4o-mini-tts (text-to-speech)
    - ElevenLabs: scribe_v1, scribe_v1_experimental
    - AssemblyAI: best, nano
    - Gemini TTS: gemini-2.5-flash-preview-tts, gemini-2.5-pro-preview-tts
    - Whisper-1: OpenAI's original transcription model
    - Total audio models: **64 models** (was 11, +482% increase)
  - **Pricing Structure**:
    - Embeddings: Per 1K tokens (input cost)
    - Images: Per image generation (varies by quality/size)
    - Audio: Per minute for STT, per 1K characters for TTS
  - **Implementation**:
    - Database: `genai_otel/llm_pricing.json`
    - Total non-chat additions: 114 models
    - All pricing sourced from official provider documentation and LiteLLM database

- **Ollama Instrumentor: Missing Response Attributes**
  - Added `_extract_response_attributes()` method to extract response model, finish reason, and content length
  - Added `_extract_finish_reason()` method to extract completion status from `done_reason` field
  - Fixes missing `gen_ai.response.model`, `gen_ai.response.finish_reason`, and cost tracking fields in Ollama traces
  - Handles both dict and object response formats for compatibility
  - Supports both `generate()` and `chat()` response structures
  - Enables proper cost calculation for Ollama models (previously failed due to missing response model)
  - Implementation: `genai_otel/instrumentors/ollama_instrumentor.py` (lines 199-276)
  - Tests: `tests/instrumentors/test_ollama_instrumentor.py` (6 new test functions)

- **CrewAI Instrumentor: Automatic Context Propagation**
  - **Zero-code context propagation** for complete trace continuity across threads and async execution
  - Automatic ThreadPoolExecutor patching for context propagation to worker threads
  - Enhanced instrumentation with three span types:
    - `crewai.crew.execution` - Top-level crew execution
    - `crewai.task.execution` - Individual task execution
    - `crewai.agent.execution` - Agent task execution
  - Automatic instrumentation of Task and Agent methods:
    - `Task.execute_sync()` - Synchronous task execution
    - `Task.execute_async()` - Asynchronous task execution
    - `Agent.execute_task()` - Agent task execution
  - Rich attribute extraction for better observability:
    - **Task attributes**: description, expected_output, assigned agent role, task ID
    - **Agent attributes**: role, goal, backstory, LLM model
    - **Crew attributes**: process type, agent count, task count, tools, inputs
  - Static `_propagate_context()` decorator for function-level context wrapping
  - Thread-safe context attachment/detachment using OpenTelemetry context API
  - Graceful degradation if methods don't exist (future-proof for CrewAI updates)
  - **Benefits for users**:
    - ✅ No manual context management code required
    - ✅ Complete parent-child span relationships across all execution
    - ✅ Works with FastAPI, Flask, and other async frameworks
    - ✅ Compatible with CrewAI's internal threading model
  - Implementation: `genai_otel/instrumentors/crewai_instrumentor.py` (+216 lines)
  - Example usage - before: manual `run_in_thread_with_context()` wrapper needed
  - Example usage - after: just call `crew.kickoff()` normally, context propagates automatically!

- **Codecarbon Integration for CO2 Emissions Tracking**
  - Integrated codecarbon library for accurate region-based carbon intensity calculations
  - Uses `OfflineEmissionsTracker` for offline mode (no API calls) or `EmissionsTracker` for online mode
  - Automatic region detection using country ISO codes, cloud providers, and regions
  - Fallback to manual calculation when codecarbon is not installed
  - New environment variables for codecarbon configuration:
    - `GENAI_CO2_COUNTRY_ISO_CODE` - 3-letter ISO country code (e.g., "USA", "GBR", "DEU")
    - `GENAI_CO2_REGION` - Region/state within country (e.g., "california", "texas")
    - `GENAI_CO2_CLOUD_PROVIDER` - Cloud provider name (e.g., "aws", "gcp", "azure")
    - `GENAI_CO2_CLOUD_REGION` - Cloud region (e.g., "us-east-1", "europe-west1")
    - `GENAI_CO2_OFFLINE_MODE` - Run codecarbon in offline mode (default: true)
    - `GENAI_CO2_TRACKING_MODE` - "machine" (all processes) or "process" (current only)
    - `GENAI_CO2_USE_MANUAL` - Force manual CO2 calculation using `GENAI_CARBON_INTENSITY`
  - CO2 tracking options:
    - **Automatic (codecarbon)**: Uses region-based carbon intensity data for accurate emissions
    - **Manual**: Uses `GENAI_CARBON_INTENSITY` value (gCO2e/kWh) for calculation
    - Set `GENAI_CO2_USE_MANUAL=true` to force manual calculation even when codecarbon is installed
  - Implementation in `genai_otel/gpu_metrics.py` and `genai_otel/config.py`
  - Added comprehensive tests for codecarbon integration (13 new test cases)
  - Install codecarbon with: `pip install genai-otel-instrument[co2]`

- **PII Detection and Safety Features (v0.2.0 Phase 1)**
  - Automatic PII detection with Microsoft Presidio integration
  - Three operation modes: detect (monitor only), redact (mask PII), block (prevent requests/responses)
  - GDPR compliance mode with EU-specific entity types (IBAN, UK NHS, NRP)
  - HIPAA compliance mode for healthcare data (medical licenses, PHI, dates)
  - PCI-DSS compliance mode for payment card data (credit cards, bank accounts)
  - 15+ PII entity types detected: email, phone, SSN, credit card, IP address, passport, etc.
  - Configurable confidence threshold (0.0-1.0) for detection sensitivity
  - Regex fallback patterns when Presidio library not available
  - OpenTelemetry span attributes for PII detection events:
    - `evaluation.pii.prompt.detected` - PII found in prompts
    - `evaluation.pii.response.detected` - PII found in responses
    - `evaluation.pii.*.entity_count` - Number of entities detected
    - `evaluation.pii.*.entity_types` - Array of detected entity types
    - `evaluation.pii.*.score` - Detection confidence score
    - `evaluation.pii.*.redacted` - Redacted text in redact mode
    - `evaluation.pii.*.blocked` - Whether request was blocked
  - OpenTelemetry metrics for monitoring:
    - `genai.evaluation.pii.detections` - Counter by location and mode
    - `genai.evaluation.pii.entities` - Counter by entity type
    - `genai.evaluation.pii.blocked` - Counter for blocked requests
  - Environment variable configuration:
    - `GENAI_ENABLE_PII_DETECTION` - Enable/disable PII detection
    - `GENAI_PII_MODE` - Set mode (detect/redact/block)
    - `GENAI_PII_THRESHOLD` - Confidence threshold
    - `GENAI_PII_GDPR_MODE` - Enable GDPR compliance
    - `GENAI_PII_HIPAA_MODE` - Enable HIPAA compliance
    - `GENAI_PII_PCI_DSS_MODE` - Enable PCI-DSS compliance
  - Implementation: `genai_otel/evaluation/` module
    - `config.py` - Configuration dataclasses for all evaluation features
    - `pii_detector.py` - PIIDetector with Presidio integration
    - `span_processor.py` - EvaluationSpanProcessor for span enrichment
  - Tests: `tests/evaluation/` (40+ test cases)
    - `test_pii_detector.py` - Unit tests for PII detection
    - `test_integration.py` - Integration tests with span processor
  - Example: `examples/pii_detection_example.py` (9 comprehensive scenarios)
  - Dependencies (optional): `pip install presidio-analyzer presidio-anonymizer spacy`

- **Toxicity Detection (v0.2.0 Phase 1)**
  - Automatic toxicity detection for harmful content in prompts and responses
  - Dual detection methods:
    - Google Perspective API integration (cloud-based, production-grade)
    - Detoxify local ML model (offline, privacy-friendly)
    - Automatic fallback from Perspective API to Detoxify on errors
  - Six toxicity categories detected:
    - `toxicity`: General toxic language
    - `severe_toxicity`: Extremely harmful content
    - `identity_attack`: Discrimination and hate speech
    - `insult`: Insulting or demeaning language
    - `profanity`: Swearing and obscene content
    - `threat`: Threatening or violent language
  - Configurable threshold (0.0-1.0) for detection sensitivity
  - Blocking mode to prevent toxic content processing
  - Batch processing support for efficient analysis
  - OpenTelemetry span attributes for toxicity detection:
    - `evaluation.toxicity.prompt.detected` - Toxicity in prompts
    - `evaluation.toxicity.response.detected` - Toxicity in responses
    - `evaluation.toxicity.*.max_score` - Maximum toxicity score
    - `evaluation.toxicity.*.categories` - List of toxic categories detected
    - `evaluation.toxicity.*.<category>_score` - Individual category scores
    - `evaluation.toxicity.*.blocked` - Whether content was blocked
  - OpenTelemetry metrics for monitoring:
    - `genai.evaluation.toxicity.detections` - Detection events counter
    - `genai.evaluation.toxicity.categories` - Category-specific counter
    - `genai.evaluation.toxicity.blocked` - Blocked requests counter
    - `genai.evaluation.toxicity.score` - Score distribution histogram
  - Environment variable configuration:
    - `GENAI_ENABLE_TOXICITY_DETECTION` - Enable/disable toxicity detection
    - `GENAI_TOXICITY_THRESHOLD` - Detection threshold (0.0-1.0)
    - `GENAI_TOXICITY_USE_PERSPECTIVE_API` - Use Perspective API
    - `GENAI_TOXICITY_PERSPECTIVE_API_KEY` - API key for Perspective
    - `GENAI_TOXICITY_BLOCK_ON_DETECTION` - Block toxic content
  - Implementation: `genai_otel/evaluation/` module
    - `toxicity_detector.py` - ToxicityDetector with dual detection methods
    - `span_processor.py` - Extended with toxicity detection
    - `config.py` - ToxicityConfig dataclass
  - Tests: `tests/evaluation/` (35+ test cases)
    - `test_toxicity_detector.py` - Unit tests for ToxicityDetector
    - `test_integration.py` - Integration tests with span processor
  - Example: `examples/toxicity_detection_example.py` (8 comprehensive scenarios)
  - Dependencies (optional):
    - Detoxify: `pip install detoxify`
    - Perspective API: `pip install google-api-python-client`

- **Bias Detection (v0.2.0 Phase 2)**
  - Automatic bias detection for demographic and other biases in prompts and responses
  - Pattern-based detection (always available, no external dependencies)
  - Eight bias types monitored:
    - `gender`: Gender stereotypes and discrimination
    - `race`: Racial bias and discrimination
    - `ethnicity`: Ethnic stereotypes and xenophobia
    - `religion`: Religious bias and discrimination
    - `age`: Age-based stereotypes (ageism)
    - `disability`: Disability bias and ableism
    - `sexual_orientation`: LGBTQ+ discrimination and bias
    - `political`: Political bias and partisan stereotyping
  - Comprehensive pattern matching with 50+ regex patterns and keywords
  - Score calculation based on pattern matches (0.0-1.0)
  - Configurable threshold for detection sensitivity
  - Blocking mode to prevent biased content processing
  - Batch processing support for analyzing multiple texts
  - Statistics generation for bias analysis and reporting
  - Optional ML-based detection with Fairlearn integration
  - Sensitive attributes configuration for custom monitoring
  - OpenTelemetry span attributes for bias detection:
    - `evaluation.bias.prompt.detected` - Bias in prompts
    - `evaluation.bias.response.detected` - Bias in responses
    - `evaluation.bias.*.max_score` - Maximum bias score
    - `evaluation.bias.*.detected_biases` - Array of detected bias types
    - `evaluation.bias.*.<bias_type>_score` - Individual bias type scores
    - `evaluation.bias.*.<bias_type>_patterns` - Matched patterns per type
    - `evaluation.bias.*.blocked` - Whether content was blocked
  - OpenTelemetry metrics for monitoring:
    - `genai.evaluation.bias.detections` - Detection events counter by location
    - `genai.evaluation.bias.types` - Detections by bias type
    - `genai.evaluation.bias.blocked` - Blocked requests counter
    - `genai.evaluation.bias.score` - Score distribution histogram
  - Environment variable configuration:
    - `GENAI_ENABLE_BIAS_DETECTION` - Enable/disable bias detection
    - `GENAI_BIAS_THRESHOLD` - Detection threshold (0.0-1.0, default 0.5)
    - `GENAI_BIAS_BLOCK_ON_DETECTION` - Block biased content
    - `GENAI_BIAS_TYPES` - Comma-separated list of bias types to monitor
    - `GENAI_BIAS_USE_FAIRLEARN` - Enable ML-based detection with Fairlearn
  - Implementation: `genai_otel/evaluation/` module
    - `bias_detector.py` - BiasDetector with pattern and ML-based detection
    - `span_processor.py` - Extended with bias detection support
    - `config.py` - BiasConfig dataclass
  - Tests: `tests/evaluation/` (56+ test cases)
    - `test_bias_detector.py` - Unit tests for BiasDetector (40+ test cases)
    - `test_integration.py` - Integration tests with span processor (16 test cases)
  - Example: `examples/bias_detection_example.py` (12 comprehensive scenarios)
  - Dependencies (optional):
    - Fairlearn: `pip install fairlearn scikit-learn` (for ML-based detection)

- **Prompt Injection Detection (v0.2.0 Phase 3)**
  - Automatic prompt injection detection protecting against manipulation attacks
  - 6 injection types: instruction_override, role_playing, jailbreak, context_switching, system_extraction, encoding_obfuscation
  - Pattern-based detection (always available, no dependencies)
  - Configurable threshold and blocking mode
  - Span attributes: `evaluation.prompt_injection.*` for all detection results
  - Metrics: 4 metrics (detections, types, blocked, score histogram)
  - Implementation: `prompt_injection_detector.py` (250+ lines)
  - Example: `examples/comprehensive_evaluation_example.py`

- **Restricted Topics Detection (v0.2.0 Phase 3)**
  - Topic classification for 9 sensitive categories (medical/legal/financial advice, violence, self-harm, etc.)
  - Configurable topic blacklists
  - Pattern and keyword-based detection
  - Span attributes: `evaluation.restricted_topics.*` for topic detection
  - Metrics: 4 metrics (detections, types, blocked, score histogram)
  - Implementation: `restricted_topics_detector.py` (300+ lines)
  - Example: `examples/comprehensive_evaluation_example.py`

- **Hallucination Detection (v0.2.0 Phase 3)**
  - Heuristic-based factual accuracy validation
  - Factual claim extraction, hedge word detection, citation tracking
  - Context contradiction detection
  - Span attributes: `evaluation.hallucination.*` for risk indicators
  - Metrics: 3 metrics (detections, indicators, score histogram)
  - Implementation: `hallucination_detector.py` (380+ lines)
  - Example: `examples/comprehensive_evaluation_example.py`

- **Multi-Agent & AI Framework Instrumentation (Phase 1-4)**
  - Comprehensive instrumentation for 11 AI frameworks with 13 implementations total
  - Zero-code setup with automatic tracing and cost tracking
  - Production-ready with 185+ test cases and 47 example scenarios
  - New frameworks: DSPy, Instructor, Guardrails AI

- **OpenAI Agents SDK Instrumentation**
  - Full OpenTelemetry instrumentation for OpenAI's production Agents SDK
  - Automatic tracing with `gen_ai.system="agents"` attribute
  - Agent orchestration with handoffs, sessions, and guardrails tracking
  - Implementation: `genai_otel/instrumentors/openai_agents_instrumentor.py`
  - Tests: `tests/instrumentors/test_openai_agents_instrumentor.py` (11 test cases)
  - Example: `examples/openai_agents_example.py` (4 scenarios)

- **CrewAI Multi-Agent Framework Instrumentation**
  - Full OpenTelemetry instrumentation for CrewAI framework
  - Automatic tracing with `gen_ai.system="crewai"` attribute
  - Role-based agent collaboration with crews and tasks tracking
  - Sequential and hierarchical process types supported
  - Implementation: `genai_otel/instrumentors/crewai_instrumentor.py`
  - Tests: `tests/instrumentors/test_crewai_instrumentor.py` (13 test cases)
  - Example: `examples/crewai_example.py` (3 scenarios)

- **LangGraph Stateful Workflow Instrumentation**
  - Full OpenTelemetry instrumentation for LangGraph framework
  - Automatic tracing with `gen_ai.system="langgraph"` attribute
  - Graph-based orchestration with nodes, edges, and state tracking
  - Support for sync/async execution and streaming
  - Checkpoint and state management tracking
  - Implementation: `genai_otel/instrumentors/langgraph_instrumentor.py`
  - Tests: `tests/instrumentors/test_langgraph_instrumentor.py` (12 test cases)
  - Example: `examples/langgraph_example.py` (3 scenarios)

- **AutoGen Multi-Agent Conversation Instrumentation**
  - Full OpenTelemetry instrumentation for Microsoft AutoGen framework
  - Automatic tracing with `gen_ai.system="autogen"` attribute
  - Multi-agent conversations with group chat orchestration
  - Speaker selection and manager coordination tracking
  - Support for both `autogen` and `pyautogen` package names
  - Implementation: `genai_otel/instrumentors/autogen_instrumentor.py`
  - Tests: `tests/instrumentors/test_autogen_instrumentor.py` (20 test cases)
  - Example: `examples/autogen_example.py` (4 scenarios)

- **Pydantic AI Type-Safe Agent Instrumentation**
  - Full OpenTelemetry instrumentation for Pydantic AI framework
  - Automatic tracing with `gen_ai.system="pydantic_ai"` attribute
  - Type-safe agents with Pydantic validation tracking
  - Multi-provider support (OpenAI, Anthropic, Gemini, etc.)
  - Structured outputs with Pydantic models
  - Tools/functions tracking with count and names
  - Support for sync, async, and streaming execution
  - Implementation: `genai_otel/instrumentors/pydantic_ai_instrumentor.py`
  - Tests: `tests/instrumentors/test_pydantic_ai_instrumentor.py` (24 test cases)
  - Example: `examples/pydantic_ai_example.py` (7 scenarios)

- **Haystack NLP Pipeline Instrumentation**
  - Full OpenTelemetry instrumentation for Haystack framework
  - Automatic tracing with `gen_ai.system="haystack"` attribute
  - Modular pipeline architecture with component tracking
  - RAG (Retrieval-Augmented Generation) workflow support
  - Generator, ChatGenerator, and Retriever component instrumentation
  - Pipeline graph structure tracking (nodes, edges, connections)
  - Custom metadata and configuration capture
  - Implementation: `genai_otel/instrumentors/haystack_instrumentor.py`
  - Tests: `tests/instrumentors/test_haystack_instrumentor.py` (23 test cases)
  - Example: `examples/haystack_example.py` (5 scenarios)

- **AWS Bedrock Agents Instrumentation**
  - Full OpenTelemetry instrumentation for AWS Bedrock Agents
  - Automatic tracing with `gen_ai.system="bedrock_agents"` attribute
  - Managed agent runtime with session tracking
  - Knowledge base retrieval and RAG operations
  - InvokeAgent, Retrieve, and RetrieveAndGenerate operation support
  - Session state and conversation tracking
  - Integration via boto3 BaseClient instrumentation
  - Implementation: `genai_otel/instrumentors/bedrock_agents_instrumentor.py`
  - Tests: `tests/instrumentors/test_bedrock_agents_instrumentor.py` (20 test cases)
  - Example: `examples/bedrock_agents_example.py` (4 scenarios)

- **DSPy Framework Instrumentation**
  - Full OpenTelemetry instrumentation for Stanford NLP's DSPy framework
  - Automatic tracing with `gen_ai.system="dspy"` attribute
  - Declarative language model programming with automatic optimization
  - Module execution tracking (Module.__call__, Predict, ChainOfThought, ReAct)
  - Optimizer/Teleprompter operations (COPRO, MIPROv2, BootstrapFewShot)
  - Signature and field tracking (input/output fields, rationales)
  - Tool usage and trajectory tracking for ReAct
  - Implementation: `genai_otel/instrumentors/dspy_instrumentor.py`
  - Tests: `tests/instrumentors/test_dspy_instrumentor.py` (25 test cases)
  - Example: `examples/dspy_example.py` (6 scenarios)

- **Instructor Framework Instrumentation**
  - Full OpenTelemetry instrumentation for Instructor (8K+ GitHub stars)
  - Automatic tracing with `gen_ai.system="instructor"` attribute
  - Pydantic-based structured output extraction with validation
  - Multi-provider support (OpenAI, Anthropic, Google, Ollama, etc.)
  - Automatic retry on validation failure tracking
  - Streaming partial results (Partial models)
  - Response model schema capture (fields, field count, types)
  - Implementation: `genai_otel/instrumentors/instructor_instrumentor.py`
  - Tests: `tests/instrumentors/test_instructor_instrumentor.py` (22 test cases)
  - Example: `examples/instructor_example.py` (6 scenarios)

- **Guardrails AI Framework Instrumentation**
  - Full OpenTelemetry instrumentation for Guardrails AI validation framework
  - Automatic tracing with `gen_ai.system="guardrails"` attribute
  - Input/output validation guards with risk detection
  - Validator tracking (names, on-fail actions, pass/fail status)
  - On-fail policies: reask, fix, filter, refrain, noop, exception, fix_reask
  - ValidationOutcome tracking (validation_passed, reasks count, errors)
  - Guard operations: __call__, validate, parse, use
  - Implementation: `genai_otel/instrumentors/guardrails_ai_instrumentor.py`
  - Tests: `tests/instrumentors/test_guardrails_ai_instrumentor.py` (8 test cases)

### Improved

- **Google GenAI SDK - Dual SDK Support**
  - Enhanced existing instrumentor to support BOTH legacy and new SDKs
  - Automatic SDK detection: tries new `google-genai` first, falls back to legacy `google-generativeai`
  - Deprecation warnings for legacy SDK users (support ends Nov 30, 2025)
  - Migration guidance in examples
  - Updated tests with dual SDK coverage (24 test cases)
  - Example: `examples/google_genai_example.py` with both SDK demonstrations

### Documentation

- **Framework Research Documentation**
  - Created `FRAMEWORK_RESEARCH.md` with comprehensive analysis of 9 AI frameworks
  - Tiered prioritization (Tier 1-3) based on popularity and complexity
  - Implementation estimates and recommended attributes
  - Full research report with API analysis and instrumentation strategies

- **README Updates**
  - Added "Multi-Agent Frameworks" section highlighting 6 new frameworks
  - Updated feature list with framework count
  - Comprehensive framework descriptions and capabilities

## [0.1.23] - 2025-11-13

### Added

- **SambaNova Instrumentation**
  - Full OpenTelemetry instrumentation for SambaNova AI models
  - Automatic tracing with `gen_ai.system="sambanova"` attribute
  - Token usage tracking and cost calculation
  - Support for Llama 4 Maverick and Llama 3.1 model family
  - Enabled by default in `DEFAULT_INSTRUMENTORS`
  - Example: `examples/sambanova_example.py`
  - Implementation: `genai_otel/instrumentors/sambanova_instrumentor.py`
  - Tests: `tests/instrumentors/test_sambanova_instrumentor.py`

- **Hyperbolic API Instrumentation**
  - Full OpenTelemetry instrumentation for Hyperbolic's cost-effective API
  - HTTP request-level instrumentation for raw API calls
  - Automatic tracing with `gen_ai.system="hyperbolic"` attribute
  - Token usage tracking and cost calculation
  - Support for Qwen3, DeepSeek R1/V3 models
  - **Disabled by default** - requires OTLP gRPC exporters due to requests library conflict
  - Configuration: Set `OTEL_EXPORTER_OTLP_PROTOCOL=grpc` and add "hyperbolic" to `GENAI_ENABLED_INSTRUMENTORS`
  - Example: `examples/hyperbolic_example.py` (complete working configuration)
  - Implementation: `genai_otel/instrumentors/hyperbolic_instrumentor.py`
  - Tests: `tests/instrumentors/test_hyperbolic_instrumentor.py`
  - Documentation: Added limitation section in `CLAUDE.md`

- **Nebius AI Studio Support**
  - Pricing data for Nebius models (uses OpenAI-compatible API, works automatically)
  - Model support: `openai/gpt-oss-120b` and Llama 3.1 family
  - Nebius uses OpenAI SDK with custom `base_url`, so existing OpenAI instrumentor handles it
  - Cost tracking enabled via pricing database entries

### Improved

- **Comprehensive Model Pricing Database Update**
  - Expanded pricing coverage from 240+ to 340+ models across 20+ providers
  - **DeepSeek Models** (25 new models):
    - R1 Distillations: `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` ($0.80/$2.40), `DeepSeek-R1-Distill-Qwen-1.5B` ($0.20/$0.40), `DeepSeek-R1-Distill-Llama-8B` ($0.50/$1.00), `DeepSeek-R1-Distill-Qwen-7B` ($0.40/$0.80)
    - Latest Releases: `DeepSeek-R1-0528` ($1.40/$2.80), `DeepSeek-V3.1` ($0.60/$1.70), `DeepSeek-V3-0324` ($0.60/$1.70), `DeepSeek-R1-0528-Qwen3-8B` ($0.50/$1.00)
    - Experimental: `DeepSeek-V3.2-Exp` ($0.80/$2.00), `DeepSeek-V3.1-Terminus` ($0.60/$1.70)
    - Specialized: `DeepSeek-OCR` ($1.00/$3.00 - 3.6M downloads), `deepseek-vl2` ($0.80/$2.40), `Janus-Pro-7B` ($0.80/$2.40 multimodal)
  - **Liquid AI LFM2 Series** (8 new models):
    - Edge Models: `LFM2-350M` ($0.10/$0.20), `LFM2-700M` ($0.15/$0.30), `LFM2-1.2B` ($0.20/$0.40 - 506K downloads), `LFM2-2.6B` ($0.30/$0.60)
    - MoE: `LFM2-8B-A1B` ($0.30/$0.90)
    - Vision-Language: `LFM2-VL-450M` ($0.20/$0.60), `LFM2-VL-1.6B` ($0.30/$0.90), `LFM2-VL-3B` ($0.40/$1.20)
  - **HuggingFace SmolLM Series** (8 new models):
    - SmolLM2: `SmolLM2-135M` ($0.05/$0.10 - 733K downloads), `SmolLM2-360M` ($0.10/$0.20), `SmolLM2-1.7B` ($0.20/$0.40)
    - SmolLM3: `SmolLM3-3B` ($0.30/$0.60)
    - Instruct variants for all sizes with same pricing
  - **Meta Llama Variants** (13 new models):
    - Llama 3.1/3.2: `Llama-3.1-8B-Instruct` ($0.50/$1.50 - 5M downloads), `Llama-3.1-70B-Instruct` ($2.00/$6.00), `Llama-3.2-1B-Instruct` ($0.10/$0.30), `Llama-3.2-3B-Instruct` ($0.30/$0.60)
    - Llama 3.3: `Llama-3.3-70B-Instruct` ($2.00/$6.00 - 659K downloads)
    - Vision: `Llama-3.2-11B-Vision-Instruct` ($1.00/$3.00 - 257K downloads)
    - Llama 4: `Llama-4-Scout-17B-16E-Instruct` ($1.20/$3.60 - 199K downloads)
    - Guard Models: `Llama-Guard-3-8B` ($0.50/$1.50), `Llama-Guard-3-1B` ($0.10/$0.30)
  - **Google Gemma 3 Series** (6 new models):
    - `gemma-3-1b-it` ($0.10/$0.20 - most popular), `gemma-2-2b-it` ($0.20/$0.40)
    - Vision-capable: `gemma-3-4b-it` ($0.50/$1.50), `gemma-3-12b-it` ($1.00/$3.00 - 1.5M downloads), `gemma-3-27b-it` ($1.50/$4.50)
    - Medical: `medgemma-4b-it` ($0.50/$1.50 - radiology, clinical reasoning, dermatology)
  - **ServiceNow Apriel Models** (3 new models):
    - `Apriel-5B-Instruct` ($0.50/$1.50), `Apriel-Nemotron-15b-Thinker` ($1.00/$3.00), `Apriel-1.5-15b-Thinker` ($1.00/$3.00 - 49K downloads)
  - **NVIDIA Models** (8 new models):
    - Nemotron Nano: `NVIDIA-Nemotron-Nano-9B-v2` ($0.50/$1.50), `NVIDIA-Nemotron-Nano-12B-v2` ($0.70/$2.10), `Llama-3.1-Nemotron-Nano-4B-v1.1` ($0.30/$0.90)
    - Nemotron Super: `Llama-3_3-Nemotron-Super-49B-v1_5` ($1.50/$4.50)
    - Vision: `Llama-3.1-Nemotron-Nano-VL-8B-V1` ($1.00/$3.00 - 747K downloads), `NVLM-D-72B` ($2.00/$6.00)
    - Specialized: `OpenReasoning-Nemotron-7B` ($0.40/$1.20), `Cosmos-Reason1-7B` ($0.80/$2.40 - 413K downloads)
  - **Qwen3 Series** (18 new models):
    - Base Models: `Qwen3-0.6B` ($0.05/$0.10), `Qwen3-1.7B` ($0.10/$0.20), `Qwen3-4B` ($0.30/$0.60), `Qwen3-8B` ($0.50/$1.00), `Qwen3-14B` ($0.80/$1.60), `Qwen3-32B` ($1.20/$2.40)
    - Instruct: `Qwen3-4B-Instruct-2507` ($0.30/$0.60 - 5M+ downloads), `Qwen3-4B-Thinking-2507` ($0.60/$1.80)
    - MoE: `Qwen3-30B-A3B-Instruct-2507` ($0.40/$1.20), `Qwen3-30B-A3B-Thinking-2507` ($0.80/$2.40), `Qwen3-Next-80B-A3B-Instruct` ($0.60/$1.80), `Qwen3-235B-A22B` ($0.50/$1.50)
    - Specialized: `Qwen3-Coder-30B-A3B-Instruct` ($0.40/$1.20), `Qwen3-Omni-30B-A3B-Instruct` ($0.60/$1.80 - multimodal with text-to-audio)
  - **Ollama Variants** (9 new models):
    - `gemma3:1b` ($0.10/$0.20 - 2.6M downloads), `gemma3:4b` ($0.50/$1.50), `gemma3:12b` ($1.00/$3.00)
    - `deepseek-r1:1.5b` ($0.20/$0.40 - 1.0M downloads), `deepseek-r1:671b` ($1.40/$2.80)
    - `llama3.3:70b` ($2.00/$6.00 - 659K downloads)
    - `granite3.1:1b` ($0.10/$0.30), `granite3.1:3b` ($0.30/$0.90), `granite3.1:8b` ($0.50/$1.50)
  - **Embedding Models** (4 new models):
    - Snowflake: `Snowflake/snowflake-arctic-embed-m` ($0.03/$0.03 - 496K downloads), `snowflake-arctic-embed-s` ($0.02/$0.02), `snowflake-arctic-embed-m-v2.0` ($0.03/$0.03), `snowflake-arctic-embed-xs` ($0.01/$0.01)
    - NVIDIA: `nvidia/NV-Embed-v2` ($0.05/$0.05 - 198K downloads), `llama-embed-nemotron-8b` ($0.06/$0.06), `omni-embed-nemotron-3b` ($0.04/$0.04)
    - Google: `google/embeddinggemma-300m` ($0.02/$0.02)
  - **Speech-to-Text Models** (4 new models):
    - NVIDIA Parakeet: `parakeet-tdt-0.6b-v2` ($0.15/$0.15 - 3.7M downloads), `parakeet-rnnt-0.6b` ($0.15/$0.15 - 3.1M downloads), `parakeet-tdt-0.6b-v3` ($0.15/$0.15 - 49 languages)
    - NVIDIA Canary: `canary-1b-v2` ($0.20/$0.20 - ASR + Translation, 30+ languages)
  - All pricing reflects official provider rates and HuggingFace popularity metrics as of January 2025

## [0.1.21] - 2025-11-12

### Added

- **Automatic Server Metrics for ALL Instrumentors**
  - Integrated server metrics tracking into `BaseInstrumentor` - ALL instrumentors (OpenAI, Anthropic, Ollama, etc.) now automatically track active requests
  - `gen_ai.server.requests.running` counter automatically increments/decrements during request execution
  - Works for both streaming and non-streaming requests
  - Works across success and error paths
  - Implementation in `genai_otel/instrumentors/base.py:311-391, 816-839`

- **Ollama Automatic Server Metrics Collection**
  - Created `OllamaServerMetricsPoller` that automatically polls Ollama's `/api/ps` endpoint
  - Collects per-model VRAM usage and updates `gen_ai.server.kv_cache.usage{model="llama2"}` metric
  - Extracts model details: parameter size, quantization level, format, total size
  - Updates `gen_ai.server.requests.max` based on number of loaded models
  - Runs in background daemon thread with configurable interval (default: 5 seconds)
  - Enabled by default when Ollama instrumentation is active
  - Zero configuration required - works out of the box
  - **Requires Python 3.11+** (feature is skipped on Python 3.9 and 3.10)
  - Implementation in `genai_otel/instrumentors/ollama_server_metrics_poller.py` (157 lines, 94% coverage)

- **GPU VRAM Auto-Detection**
  - Automatic GPU VRAM detection using multiple fallback methods:
    1. **nvidia-ml-py** (pynvml) - preferred method, requires `pip install genai-otel-instrument[gpu]`
    2. **nvidia-smi** - automatic fallback using command-line tool
    3. **Manual override** - `GENAI_OLLAMA_MAX_VRAM_GB` environment variable (now optional)
  - Auto-detection runs once during poller initialization
  - Logs detected VRAM: "Auto-detected GPU VRAM: 24.0GB" or "GPU VRAM not detected, using heuristic-based percentages"
  - Eliminates need for manual VRAM configuration in most cases
  - Supports multi-GPU systems (uses first GPU for Ollama)
  - Implementation in `genai_otel/instrumentors/ollama_server_metrics_poller.py:81-172`

- **Enhanced Ollama Server Metrics Configuration**
  - New environment variables for Ollama server metrics:
    - `GENAI_ENABLE_OLLAMA_SERVER_METRICS` - Enable/disable automatic metrics (default: true)
    - `OLLAMA_BASE_URL` - Ollama server URL (default: http://localhost:11434)
    - `GENAI_OLLAMA_METRICS_INTERVAL` - Polling interval in seconds (default: 5.0)
    - `GENAI_OLLAMA_MAX_VRAM_GB` - Manual VRAM override (optional, auto-detected if not set)
  - Poller integrates with OllamaInstrumentor automatically
  - Graceful error handling for offline Ollama server or missing GPU
  - Implementation in `genai_otel/instrumentors/ollama_instrumentor.py:76-104`

### Improved

- **Test Coverage Enhancements**
  - Added 31 new comprehensive tests:
    - 18 tests for `OllamaServerMetricsPoller` (metrics collection, error handling, lifecycle)
    - 8 tests for GPU VRAM auto-detection (nvidia-ml-py, nvidia-smi, fallbacks, manual override)
    - 5 tests for Ollama instrumentor integration (poller startup, configuration, error handling)
  - Total tests increased from 496 to **527** (6.25% increase)
  - Improved `ollama_server_metrics_poller.py` coverage to **94%**
  - Improved `ollama_instrumentor.py` coverage to **97%**
  - Overall coverage maintained at **84%**
  - All tests passing with zero regressions

- **Documentation Updates**
  - Added "Ollama Automatic Integration" section to `docs/SERVER_METRICS.md`
  - Documented GPU VRAM auto-detection workflow with fallback methods
  - Updated `sample.env` with detailed comments on auto-detection
  - Created comprehensive example: `examples/ollama/example_with_server_metrics.py`
  - All Ollama server metrics are now fully documented with configuration examples

### Changed

- **GENAI_OLLAMA_MAX_VRAM_GB Now Optional**
  - Environment variable is no longer required
  - Auto-detection attempts to determine GPU VRAM automatically
  - Only set this variable if you want to override auto-detection or if auto-detection fails
  - Fallback heuristic still works if both auto-detection methods fail

## [0.1.20] - 2025-11-11

### Added

- **NVIDIA NIM-Inspired Server Metrics**
  - Added KV cache usage tracking: `gen_ai.server.kv_cache.usage` (Gauge) - GPU KV-cache usage percentage per model
  - Added request queue metrics:
    - `gen_ai.server.requests.running` (Gauge) - Active requests currently executing
    - `gen_ai.server.requests.waiting` (Gauge) - Requests waiting in queue
    - `gen_ai.server.requests.max` (Gauge) - Maximum concurrent request capacity
  - New `ServerMetricsCollector` class with thread-safe manual instrumentation API
  - Exported via `genai_otel.get_server_metrics()` for programmatic access

- **Token Distribution Histograms**
  - `gen_ai.client.token.usage.prompt` (Histogram) - Distribution of prompt tokens per request
  - `gen_ai.client.token.usage.completion` (Histogram) - Distribution of completion tokens per request
  - Configurable buckets from 1 to 67M tokens for analyzing token usage patterns
  - Enables p50, p95, p99 analysis of token consumption

- **Finish Reason Tracking**
  - `gen_ai.server.request.finish` (Counter) - All finished requests by finish reason (stop, length, error, content_filter, etc.)
  - `gen_ai.server.request.success` (Counter) - Successful completions (stop/length reasons)
  - `gen_ai.server.request.failure` (Counter) - Failed requests (error/content_filter/timeout reasons)
  - `gen_ai.response.finish_reason` span attribute for detailed tracing
  - Implemented `_extract_finish_reason()` in OpenAI instrumentor (example for other providers)

### Improved

- **Test Coverage**
  - Added 16 new tests covering server metrics, token histograms, and finish reason tracking
  - Total tests increased from 480 to 496
  - Overall coverage maintained at 83%, new server_metrics.py has 100% coverage
  - All metrics are thread-safe with comprehensive concurrency tests

## [0.1.19] - 2025-01-05

### Fixed

- **LangChain Instrumentation: Standard GenAI Attributes and Cost Tracking**
  - Fixed missing standard GenAI semantic convention attributes (gen_ai.system, gen_ai.request.model, gen_ai.operation.name, gen_ai.request.message_count)
  - Fixed missing token usage metrics (gen_ai.usage.prompt_tokens, gen_ai.usage.completion_tokens, gen_ai.usage.total_tokens)
  - Fixed missing cost calculation and tracking (gen_ai.usage.cost.total and granular costs)
  - Fixed missing latency metrics recording
  - Applied fixes to all chat model methods: invoke(), ainvoke(), batch(), abatch()
  - Maintained backward compatibility with langchain.* attributes
  - Removed redundant _extract_and_record_usage() method, improved code coverage from 71% to 81%
  - LangChain instrumentation now provides the same comprehensive observability as other provider instrumentors

## [0.1.18] - 2025-11-05

### Improved

- **Test Coverage Enhancements**
  - Added comprehensive tests for GPU metrics collection (11 new tests)
  - Added comprehensive tests for cost enriching exporter (20 new tests)
  - Improved `genai_otel/gpu_metrics.py` coverage from 72% to 93%
  - Improved `genai_otel/cost_enriching_exporter.py` coverage from 20% to 100%
  - Overall test coverage improved from 81% to 83%
  - 480 total tests passing (30 new tests added)

## [0.1.17] - 2025-11-05

### Added

- **Enhanced LangChain Instrumentation**
  - Direct chat model instrumentation with support for invoke(), ainvoke(), batch(), abatch() methods
  - Captures model name, provider, message count, and token usage
  - Creates langchain.chat_model.* spans for better visibility
  - Supports both usage_metadata and response_metadata formats

- **Automated CI/CD Publishing Pipeline**
  - Full test suite execution before publishing
  - Code quality checks (black, isort validation)
  - Automated publishing to Test PyPI and production PyPI
  - Package installation verification in isolated environment
  - Release summary generation

- **Documentation Improvements**
  - Added comprehensive release documentation (.github/RELEASE_GUIDE.md, .github/RELEASE_QUICKSTART.md)
  - Enhanced environment variable documentation in sample.env
  - Added OTEL_EXPORTER_OTLP_TIMEOUT, OTEL_EXPORTER_OTLP_PROTOCOL, OTEL_SERVICE_INSTANCE_ID, OTEL_ENVIRONMENT, GENAI_GPU_COLLECTION_INTERVAL documentation
  - Cleaned up obsolete documentation files

### Fixed

- **OTLP Exporter Timeout Type Conversion Error**
  - Changed exporter_timeout from float to int in OTelConfig
  - Added _get_exporter_timeout() helper with graceful error handling
  - Invalid timeout values now default to 60 seconds with warning
  - Fixes ValueError: invalid literal for int() with base 10: '10.0'

- **Test Suite Stability**
  - Removed problematic test files that caused hanging (tests/test_cost_enriching_exporter.py, tests/test_gpu_metrics.py, tests/instrumentors/test_togetherai_instrumentor.py)
  - Test suite now completes successfully
  - Restored stable test execution for CI/CD pipeline

## [0.1.16] - 2025-11-05

### Fixed

- Reverted test coverage improvements that caused test suite hangs
  - Reverted commit 73842f5 which introduced OpenTelemetry global state pollution
  - Test suite now completes successfully (442 tests passing)
  - Eliminated hanging issues in test_vertexai_instrumentor.py and related tests
  - Restored stable test execution for CI/CD pipeline

### Note

This release focuses on stability by reverting problematic test coverage improvements. The test coverage improvements will be reintroduced in a future release with proper test isolation.

## [0.1.14] - 2025-10-29

### Changed

- **BREAKING: License changed from Apache-2.0 to AGPL-3.0-or-later**
  - Provides stronger copyleft protection for the project
  - Network provision requires sharing source code for modified versions used over network
  - Full license text in LICENSE file with Copyright (C) 2025 Kshitij Thakkar
  - Updated all license references in pyproject.toml, __init__.py, and README.md
  - Completed LICENSE template with program name, copyright, and contact information

- **Project Rebranding to TraceVerde**
  - Display name changed from "GenAI OpenTelemetry Auto-Instrumentation" to "TraceVerde"
  - Package name remains `genai-otel-instrument` for PyPI compatibility (no breaking changes)
  - Updated README.md title, branding, and license badges

### Fixed

- Removed `__version__.py` from version control (generated file, should not be tracked)
- This fixes versioning issues during builds

**⚠️ Important**: Users should review AGPL-3.0 license terms before upgrading, especially for commercial/SaaS deployments

## [0.1.12] - 2025-10-29

### Added

- **Enhanced README Documentation**
  - Added professional project logo centered at the top of README
  - Added landing page hero image showcasing the project overview
  - Added comprehensive Screenshots section with 5 embedded demonstration images:
    - OpenAI instrumentation with token usage, costs, and latency metrics
    - Ollama (local LLM) zero-code instrumentation
    - HuggingFace Transformers with automatic token counting
    - SmolAgents framework with complete agent workflow tracing
    - GPU metrics collection dashboard
  - Added links to additional screenshots (Token Cost Breakdown, OpenSearch Dashboard)
  - Added Demo Video section with placeholder for future video content
  - All images follow OSS documentation standards with professional formatting

### Changed

- **Roadmap Section Cleanup**
  - Removed Phase 4 implementation details from roadmap (Session & User Tracking, RAG/Embedding Attributes)
  - Phase 4 features are now fully implemented and documented in the Advanced Features section
  - Roadmap now focuses exclusively on future releases (v0.2.0 onwards)

### Improved

- **Comprehensive Model Pricing Database Update**
  - Expanded pricing coverage from 145+ to 240+ models across 15+ providers
  - **OpenAI GPT-5 Series** (4 new models):
    - `gpt-5` - $1.25/$10 per 1M tokens
    - `gpt-5-2025-08-07` - $1.25/$10 per 1M tokens
    - `gpt-5-mini` - $0.25/$2 per 1M tokens
    - `gpt-5-nano` - $0.10/$0.40 per 1M tokens
  - **Anthropic Claude 4/3.5 Variants** (13 new models):
    - Claude 4 Opus series: `claude-4-opus`, `claude-opus-4`, `claude-opus-4-1`, `claude-opus-4.1` - $15/$75 per 1M tokens
    - Claude 3.5 Sonnet: `claude-3-5-sonnet-20240620`, `claude-3-5-sonnet-20241022`, `claude-sonnet-4-5`, `claude-sonnet-4-5-20250929`, `claude-3-7-sonnet` - $3/$15 per 1M tokens
    - Claude 3.5 Haiku: `claude-3-5-haiku-20241022` - $0.80/$4 per 1M tokens
    - Claude Haiku 4.5: `claude-haiku-4-5` - $1/$5 per 1M tokens
  - **XAI Grok Models** (10 new models):
    - Grok 2: `grok-2-1212`, `grok-2-vision-1212` - $2/$10 per 1M tokens
    - Grok 3: `grok-3` - $3/$15 per 1M tokens, `grok-3-mini` - $0.30/$0.50 per 1M tokens
    - Grok 3 Fast: `grok-3-fast` - $5/$25 per 1M tokens, `grok-3-mini-fast` - $0.60/$4 per 1M tokens
    - Grok 4: `grok-4` - $3/$15 per 1M tokens, `grok-4-fast` - $0.20/$0.50 per 1M tokens
    - Image models: `grok-image`, `xai-grok-image` - $0.07 per image
  - **Google Gemini Variants** (2 new models):
    - `gemini-2-5-flash-image` - $0.30/$30 per 1M tokens
    - `nano-banana` - $0.30/$30 per 1M tokens
  - **Qwen Series** (6 new models):
    - `qwen3-next-80b-a3b-instruct` - $0.525/$2.10 per 1M tokens
    - `qwen3-next-80b-a3b-thinking` - $0.525/$6.30 per 1M tokens
    - `qwen3-coder-480b-a35b-instruct` - $1/$5 per 1M tokens
    - `qwen3-max`, `qwen-qwen3-max` - $1.20/$6 per 1M tokens
  - **Meta Llama 4 Scout & Maverick** (6 models with updated pricing):
    - `llama-4-scout`, `llama-4-scout-17bx16e-128k`, `meta-llama/Llama-4-Scout` - $0.15/$0.50 per 1M tokens
    - `llama-4-maverick`, `llama-4-maverick-17bx128e-128k`, `meta-llama/Llama-4-Maverick` - $0.22/$0.85 per 1M tokens
  - **IBM Granite Models** (13 new models):
    - Granite 3 series: `ibm-granite-3-1-8b-instruct`, `ibm-granite-3-8b-instruct`, `granite-3-8b-instruct` - $0.20/$0.20 per 1M tokens
    - Granite 4 series: `granite-4-0-h-small`, `granite-4-0-h-tiny`, `granite-4-0-h-micro`, `granite-4-0-micro` - $0.20/$0.20 per 1M tokens
    - Embeddings: `granite-embedding-107m-multilingual`, `granite-embedding-278m-multilingual` - $0.10/$0.10 per 1M tokens
    - Ollama variants: `granite:3b`, `granite:8b` - $0.20/$0.20 per 1M tokens
  - **Mistral AI Updates** (10 new models):
    - `mistral-large-24-11`, `mistral-large-2411` - $8/$24 per 1M tokens
    - `mistral-small-3-1`, `mistral-small-3.1` - $1/$3 per 1M tokens
    - `mistral-medium-3`, `mistral-medium-2025` - $0.40/$2 per 1M tokens
    - Magistral series: `magistral-small` - $1/$3, `magistral-medium` - $3/$9 per 1M tokens
    - Codestral: `codestral-25-01`, `codestral-2501` - $1/$3 per 1M tokens
  - **Additional Providers**:
    - **Sarvam AI**: `sarvam-m`, `sarvamai/sarvam-m`, `sarvam-chat` - Free (Open source)
    - **Liquid AI**: `lfm-7b`, `liquid/lfm-7b` - $0.30/$0.60 per 1M tokens
    - **Snowflake**: `snowflake-arctic`, `snowflake-arctic-instruct` - $0.80/$2.40 per 1M tokens, `snowflake-arctic-embed-l-v2.0` - $0.05/$0.05 per 1M tokens
    - **NVIDIA Nemotron**: `nvidia-nemotron-4-340b-instruct` - $3/$9 per 1M tokens, `nvidia-nemotron-mini` - $0.20/$0.40 per 1M tokens, `nvidia/llama-3.1-nemotron-70b-instruct` - $0.80/$0.80 per 1M tokens
    - **ServiceNow**: `servicenow-now-assist` - $1/$3 per 1M tokens
  - **Pricing Corrections**:
    - `deepseek-v3.1`: Updated to $0.56/$1.68 per 1M tokens (from $1.20/$1.20)
    - `qwen3:3b`: Renamed to `qwen3:4b` (4B parameter model)
  - All pricing reflects official provider rates as of October 2025

## [0.1.9] - 2025-01-27

### Added

- **HuggingFace AutoModelForCausalLM and AutoModelForSeq2SeqLM Instrumentation**
  - Added support for direct model usage via `AutoModelForCausalLM.generate()` and `AutoModelForSeq2SeqLM.generate()`
  - Automatic token counting from input and output tensor shapes
  - Cost calculation based on model parameter count (uses CostCalculator's local model pricing tiers)
  - Span attributes: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.operation.name`, token counts, costs
  - Metrics: request counter, token counter, latency histogram, cost counter
  - Supports generation parameters: `max_length`, `max_new_tokens`, `temperature`, `top_p`
  - Implementation in `genai_otel/instrumentors/huggingface_instrumentor.py:184-333`
  - Example usage in `examples/huggingface/example_automodel.py`
  - All 443 tests pass (added 1 new test)

### Fixed

- **CRITICAL: Cost Tracking for OpenInference Instrumentors (smolagents, litellm, mcp)**
  - Replaced `CostEnrichmentSpanProcessor` with `CostEnrichingSpanExporter` to properly add cost attributes
  - **Root Cause**: SpanProcessor's `on_end()` receives immutable `ReadableSpan` objects that cannot be modified
  - **Solution**: Custom SpanExporter that enriches span data before export, creating new ReadableSpan instances with cost attributes
  - Cost attributes now correctly appear for smolagents, litellm, and mcp spans:
    - `gen_ai.usage.cost.total`: Total cost in USD
    - `gen_ai.usage.cost.prompt`: Prompt tokens cost
    - `gen_ai.usage.cost.completion`: Completion tokens cost
  - Supports all OpenInference semantic conventions:
    - Model name: `llm.model_name`, `gen_ai.request.model`, `embedding.model_name`
    - Token counts: `llm.token_count.{prompt,completion}`, `gen_ai.usage.{prompt_tokens,completion_tokens}`
    - Span kinds: `openinference.span.kind` (LLM, EMBEDDING, CHAIN, etc.)
  - Implementation in `genai_otel/cost_enriching_exporter.py`
  - Updated `genai_otel/auto_instrument.py` to wrap OTLP and Console exporters
  - Model name normalization handles provider prefixes (e.g., `openai/gpt-3.5-turbo` → `gpt-3.5-turbo`)
  - All 442 existing tests continue to pass

- **HuggingFace AutoModelForCausalLM AttributeError Fix**
  - Fixed `AttributeError: type object 'AutoModelForCausalLM' has no attribute 'generate'`
  - Root cause: `AutoModelForCausalLM` is a factory class; `generate()` exists on `GenerationMixin`
  - Solution: Wrap `GenerationMixin.generate()` which all generative models inherit from
  - This covers all model types: `AutoModelForCausalLM`, `AutoModelForSeq2SeqLM`, `GPT2LMHeadModel`, etc.
  - Added fallback import for older transformers versions
  - Implementation in `genai_otel/instrumentors/huggingface_instrumentor.py:184-346`

## [0.1.7] - 2025-01-25

### Added

- **Phase 4: Session and User Tracking (4.1)**
  - Added `session_id_extractor` and `user_id_extractor` optional callable fields to `OTelConfig`
  - Extractor function signature: `(instance, args, kwargs) -> Optional[str]`
  - Automatically sets `session.id` and `user.id` span attributes when extractors are configured
  - Enables tracking conversations across multiple requests for the same session
  - Supports per-user analytics, cost attribution, and debugging
  - Implementation in `genai_otel/config.py:134-139` and `genai_otel/instrumentors/base.py:266-284`
  - Documented in README.md with comprehensive examples
  - Example implementation in `examples/phase4_session_rag_tracking.py`

- **Phase 4: RAG and Embedding Attributes (4.2)**
  - Added `add_embedding_attributes()` helper method to `BaseInstrumentor`
    - Sets `embedding.model_name`, `embedding.text`, `embedding.vector`, `embedding.vector.dimension`
    - Truncates text to 500 characters to avoid span size explosion
  - Added `add_retrieval_attributes()` helper method to `BaseInstrumentor`
    - Sets `retrieval.query`, `retrieval.document_count`
    - Sets per-document attributes: `retrieval.documents.{i}.document.id`, `.score`, `.content`, `.metadata.*`
    - Limits to 5 documents by default (configurable via `max_docs` parameter)
    - Truncates content and metadata to prevent excessive attribute counts
  - Enables enhanced observability for RAG (Retrieval-Augmented Generation) workflows
  - Implementation in `genai_otel/instrumentors/base.py:705-770`
  - Documented in README.md with usage examples and best practices
  - Complete RAG workflow example in `examples/phase4_session_rag_tracking.py`

- **Phase 4 Documentation and Examples**
  - Added "Advanced Features" section to README.md
  - Documented session/user tracking with extractor function patterns
  - Documented RAG/embedding attributes with helper method usage
  - Created comprehensive example file `examples/phase4_session_rag_tracking.py` demonstrating:
    - Session and user extractor functions
    - Embedding attribute capture
    - Retrieval attribute capture with document metadata
    - Complete RAG workflow with session tracking
  - Updated roadmap section to mark Phase 4 as completed
  - **Note**: Agent workflow tracking (`agent.name`, `agent.iteration`, etc.) is provided by the existing OpenInference Smolagents instrumentor, not new in Phase 4

## [0.1.5] - 2025-01-25

### Added

- **Streaming Cost Tracking and Token Usage**
  - Fixed missing cost calculation for streaming LLM requests
  - `_wrap_streaming_response()` now extracts usage from the last chunk and calculates costs
  - Streaming responses now record all cost metrics: `gen_ai.usage.cost.total`, `gen_ai.usage.cost.prompt`, `gen_ai.usage.cost.completion`, etc.
  - Token usage metrics now properly recorded for streaming: `gen_ai.usage.prompt_tokens`, `gen_ai.usage.completion_tokens`, `gen_ai.usage.total_tokens`
  - Works for all providers that include usage in final chunk (OpenAI, Anthropic, Google, etc.)
  - Streaming metrics still captured: `gen_ai.server.ttft` (histogram), `gen_ai.server.tbt` (histogram), `gen_ai.streaming.token_count` (chunk count)
  - Implementation in `genai_otel/instrumentors/base.py:551-638`
  - Resolves issue where streaming requests had TTFT/TBT but no cost/usage tracking

### Fixed

- **GPU Metrics Test Infrastructure**
  - Fixed GPU metrics test mocks to return separate Mock objects for CO2 and power cost counters
  - Updated `mock_meter` fixture in `tests/test_gpu_metrics.py` to use `side_effect` for multiple counters
  - Fixed `test_auto_instrument.py` assertions to use dynamic `config.gpu_collection_interval` instead of hardcoded values
  - All 434 tests now pass with proper GPU power cost tracking validation

## [0.1.4] - 2025-01-24

### Added

- **Custom Model Pricing via Environment Variable**
  - Added `GENAI_CUSTOM_PRICING_JSON` environment variable for custom/proprietary model pricing
  - Supports all pricing categories: chat, embeddings, audio, images
  - Custom prices merged with default `llm_pricing.json` (custom takes precedence)
  - Enables pricing for internal/proprietary models not in public pricing database
  - Format: `{"chat":{"model-name":{"promptPrice":0.001,"completionPrice":0.002}}}`
  - Added `custom_pricing_json` field to `OTelConfig` dataclass
  - Updated `CostCalculator.__init__()` to accept custom pricing parameter
  - Implemented `CostCalculator._merge_custom_pricing()` with validation and error handling
  - Added `BaseInstrumentor._setup_config()` helper to reinitialize cost calculator
  - Added 8 comprehensive tests in `TestCustomPricing` class
  - Documented in README.md with usage examples and pricing format guide
  - Documented in sample.env with multiple examples

- **GPU Power Cost Tracking**
  - Added `GENAI_POWER_COST_PER_KWH` environment variable for electricity cost tracking (default: $0.12/kWh)
  - New metric `gen_ai.power.cost` tracks cumulative electricity costs in USD based on GPU power consumption
  - Calculates cost from GPU power draw: (energy_Wh / 1000) * cost_per_kWh
  - Includes `gpu_id` and `gpu_name` attributes for multi-GPU systems
  - Works alongside existing CO2 emissions tracking (`gen_ai.co2.emissions`)
  - Added `power_cost_per_kwh` field to `OTelConfig` dataclass
  - Implemented in `GPUMetricsCollector._collect_loop()` in `gpu_metrics.py`
  - Added 2 comprehensive tests: basic tracking and custom rate validation
  - Documented in README.md, sample.env, and CHANGELOG.md
  - Common electricity rates provided as reference: US $0.12, Europe $0.20, Industrial $0.07

- **HuggingFace InferenceClient Instrumentation**
  - Added full instrumentation support for HuggingFace Inference API via `InferenceClient`
  - Enables observability for smolagents workflows using `InferenceClientModel`
  - Wraps `InferenceClient.chat_completion()` and `InferenceClient.text_generation()` methods
  - Creates child spans showing actual HuggingFace API calls under agent/tool spans
  - Extracts model name, temperature, max_tokens, top_p from API calls
  - Supports both object and dict response formats for token usage
  - Handles streaming responses with `gen_ai.server.ttft` and `gen_ai.streaming.token_count`
  - Cost tracking enabled via fallback estimation based on model parameter count
  - Implementation in `genai_otel/instrumentors/huggingface_instrumentor.py:141-222`
  - Added 10 comprehensive tests covering all InferenceClient functionality
  - Coverage increased from 85% → 98% for HuggingFace instrumentor
  - Resolves issue where only AGENT and TOOL spans were visible without LLM child spans

- **Fallback Cost Estimation for Local Models (Ollama & HuggingFace)**
  - Added 36 Ollama models to `llm_pricing.json` with parameter-count-based pricing tiers
  - Implemented intelligent fallback cost estimation for unknown local models in `CostCalculator`
  - Automatically parses parameter count from model names (e.g., "360m", "7b", "70b")
  - Supports both Ollama and HuggingFace model naming patterns:
    - Explicit sizes: `llama3:7b`, `mistral-7b-v0.1`, `smollm2:360m`
    - HuggingFace size indicators: `gpt2`, `gpt2-xl`, `bert-base`, `t5-xxl`, etc.
  - Applies tiered pricing based on parameter count:
    - Tiny (< 1B): $0.0001 / $0.0002 per 1k tokens
    - Small (1-10B): $0.0003 / $0.0006
    - Medium (10-20B): $0.0005 / $0.001
    - Large (20-80B): $0.0008 / $0.0008
    - XLarge (80B+): $0.0012 / $0.0012
  - Acknowledges that local models are free but consume GPU power and electricity
  - Provides synthetic cost estimates for carbon footprint and resource tracking
  - Added `scripts/add_ollama_pricing.py` to update pricing database with new Ollama models
  - Logs fallback pricing usage at INFO level for transparency

### Improved

- **CostEnrichmentSpanProcessor Performance Optimization**
  - Added early-exit logic to skip spans that already have cost attributes
  - Checks for `gen_ai.usage.cost.total` presence before attempting enrichment
  - Saves processing compute by avoiding redundant cost calculations
  - Eliminates warning messages for spans enriched by instrumentors
  - Benefits all instrumentors that set cost attributes directly (Mistral, OpenAI, Anthropic, etc.)
  - Implementation in `genai_otel/cost_enrichment_processor.py:69-74`
  - Added comprehensive test coverage for skip logic
  - Coverage increased from 94% → 98% for CostEnrichmentSpanProcessor

### Fixed

- **CRITICAL: Complete Rewrite of Mistral AI Instrumentor**
  - **Root problem**: Original instrumentor used instance-level wrapping which didn't work reliably
  - **Complete architectural rewrite** using class-level method wrapping with `wrapt.wrap_function_wrapper()`
  - Now properly wraps `Chat.complete`, `Chat.stream`, and `Embeddings.create` at the class level
  - All Mistral client instances now use instrumented methods automatically
  - **Streaming support** with custom `_StreamWrapper` class:
    - Iterates through streaming chunks and collects usage data
    - Records TTFT (Time To First Token) metric
    - Creates mock response objects for proper metrics recording
  - **Proper error handling** with span exception recording
  - **Cost tracking** now works correctly with BaseInstrumentor integration
  - Fixed incorrect `_record_result_metrics()` signature usage
  - Implementation in `genai_otel/instrumentors/mistralai_instrumentor.py` (180 lines, completely rewritten)
  - All 5 Mistral tests passing with proper mocking
  - Traces now collected with full details: model, tokens, costs, TTFT
  - Resolves issue where no Mistral spans were being collected

- **CRITICAL: Fixed Missing Granular Cost Counter Class Variables**
  - Fixed `AttributeError: 'OllamaInstrumentor' object has no attribute '_shared_prompt_cost_counter'`
  - **Root cause**: Granular cost counters were created in initialization but not declared as class variables
  - **Impact**: Test suite failed with 34 errors when running full suite (but passed individually)
  - Added missing class variable declarations in `BaseInstrumentor`:
    - `_shared_prompt_cost_counter`
    - `_shared_completion_cost_counter`
    - `_shared_reasoning_cost_counter`
    - `_shared_cache_read_cost_counter`
    - `_shared_cache_write_cost_counter`
  - Created instance variable references in `__init__` for all granular counters
  - Updated all references to use instance variables instead of `_shared_*` variables
  - Implementation in `genai_otel/instrumentors/base.py:85-90, 106-111`
  - All 424 tests now passing consistently
  - Affects all instrumentors using granular cost tracking

- **CRITICAL: Fixed Cost Tracking Disabled by Wrong Variable Check**
  - **Root cause**: Cost tracking checked `self._shared_cost_counter` which was always None
  - Should have checked `self.config.enable_cost_tracking` flag only
  - **Impact**: Cost attributes were never added to spans even when cost tracking was enabled
  - Removed unnecessary `cost_counter` existence check
  - Cost tracking now properly controlled by `GENAI_ENABLE_COST_TRACKING` environment variable
  - Implementation in `genai_otel/instrumentors/base.py:384`
  - Debug logging confirmed cost calculation working: "Calculating cost for model=smollm2:360m"
  - Affects all instrumentors (Ollama, Mistral, OpenAI, Anthropic, etc.)

- **CRITICAL: Fixed Token and Cost Attributes Not Being Set on Spans**
  - Fixed critical bug where `gen_ai.usage.prompt_tokens`, `gen_ai.usage.completion_tokens`, and all cost attributes were not being set on spans
  - **Root causes:**
    1. Span attributes were only set if metric counters were available, but this check was too restrictive
    2. Used wrong variable name (`self._shared_cost_counter` instead of `self.cost_counter`) in cost tracking check
  - **Impact**: Cost calculation completely failed - only `gen_ai.usage.total_tokens` was set
  - **Fixed by:**
    1. Always setting span attributes regardless of metric availability
    2. Using correct instance variables (`self.cost_counter`, `self.token_counter`)
    3. Metrics recording is now optional, but span attributes are always set
    4. Cost attributes (`gen_ai.usage.cost.total`, `gen_ai.usage.cost.prompt`, `gen_ai.usage.cost.completion`) are now always added
  - This ensures cost tracking works even if metrics initialization fails
  - Affects all instrumentors (OpenAI, Anthropic, Ollama, etc.)

- **CRITICAL: Fixed 6 Instrumentors Missing `self._instrumented = True`**
  - Ollama, Cohere, HuggingFace, Replicate, TogetherAI, and VertexAI instrumentors were completely broken
  - No traces were being collected because `self._instrumented` flag was not set after wrapping functions
  - The `create_span_wrapper()` checks this flag and skips instrumentation if False
  - Added `self._instrumented = True` after successful wrapping in all 6 instrumentors
  - All instrumentors now properly collect traces again

- **CRITICAL: CostEnrichmentSpanProcessor Now Working**
  - Fixed critical bug where `CostEnrichmentSpanProcessor` was calling `calculate_cost()` (returns float) but treating it as a dict
  - This caused all cost enrichment to silently fail with `TypeError: 'float' object is not subscriptable`
  - Now correctly calls `calculate_granular_cost()` which returns a proper dict with `total`, `prompt`, `completion` keys
  - Cost attributes (`gen_ai.usage.cost.total`, `gen_ai.usage.cost.prompt`, `gen_ai.usage.cost.completion`) will now be added to OpenInference spans (smolagents, litellm, mcp)
  - Improved error logging from `logger.debug` to `logger.warning` with full exception info for easier debugging
  - Added logging of successful cost enrichment at `INFO` level with span name, model, and token details
  - All 415 tests passing, including 20 cost enrichment processor tests

- **Fixed OpenInference Instrumentor Loading Order**
  - Corrected instrumentor initialization order to: smolagents → litellm → mcp
  - This matches the correct order found in working implementations
  - Ensures proper nested instrumentation and attribute capture

## [0.1.3] - 2025-01-23

### Added

- **Cost Enrichment for OpenInference Instrumentors**
  - **CostEnrichmentSpanProcessor**: New custom SpanProcessor that automatically adds cost tracking to spans created by OpenInference instrumentors (smolagents, litellm, mcp)
    - Extracts model name and token usage from existing span attributes
    - Calculates costs using the existing CostCalculator with 145+ model pricing data
    - Adds granular cost attributes: `gen_ai.usage.cost.total`, `gen_ai.usage.cost.prompt`, `gen_ai.usage.cost.completion`
    - **Dual Semantic Convention Support**: Works with both OpenTelemetry GenAI and OpenInference conventions
      - GenAI: `gen_ai.request.model`, `gen_ai.usage.{prompt_tokens,completion_tokens,input_tokens,output_tokens}`
      - OpenInference: `llm.model_name`, `embedding.model_name`, `llm.token_count.{prompt,completion}`
      - OpenInference span kinds: LLM, EMBEDDING, CHAIN, RETRIEVER, RERANKER, TOOL, AGENT
    - Maps operation names to call types (chat, embedding, image, audio) automatically
    - Gracefully handles missing data and errors without failing span processing
  - Enabled by default when `GENAI_ENABLE_COST_TRACKING=true`
  - Works alongside OpenInference's native instrumentation without modifying upstream code
  - 100% test coverage with 20 comprehensive test cases (includes 5 OpenInference-specific tests)

- **Comprehensive Cost Tracking Enhancements**
  - Added token usage extraction and cost calculation for **6 instrumentors**: Ollama, Cohere, Together AI, Vertex AI, HuggingFace, and Replicate
  - Implemented `create_span_wrapper()` pattern across all instrumentors for consistent metrics recording
  - Added `gen_ai.operation.name` attribute to all instrumentors for improved observability
  - Total instrumentors with cost tracking increased from 8 to **11** (37.5% increase)

- **Pricing Data Expansion**
  - Added pricing for **45+ new LLM models** from 3 major providers:
    - **Groq**: 9 models (Llama 3.1/3.3/4, Qwen, GPT-OSS, Kimi-K2)
    - **Cohere**: 5 models (Command R/R+/R7B, Command A, updated legacy pricing)
    - **Together AI**: 30+ models (DeepSeek R1/V3, Qwen 2.5/3, Mistral variants, GLM-4.5)
  - All pricing verified from official provider documentation (2025 rates)

- **Enhanced Instrumentor Implementations**
  - **Ollama**: Extracts `prompt_eval_count` and `eval_count` from response (local model usage tracking)
  - **Cohere**: Extracts from `meta.tokens` with `meta.billed_units` fallback
  - **Together AI**: OpenAI-compatible format with dual API support (client + legacy Complete API)
  - **Vertex AI**: Extracts `usage_metadata` with both snake_case and camelCase support
  - **HuggingFace**: Documented as local/free execution (no API costs)
  - **Replicate**: Documented as hardware-based pricing ($/second, not token-based)

### Improved

- **Standardization & Code Quality**
  - Standardized all instrumentors to use `BaseInstrumentor.create_span_wrapper()` pattern
  - Improved error handling with consistent `fail_on_error` support across all instrumentors
  - Enhanced documentation with comprehensive docstrings explaining pricing models
  - Added proper logging at all error points for better debugging
  - Thread-safe metrics initialization across all instrumentors

- **Test Coverage**
  - All **415 tests passing** (100% test success rate)
  - Increased overall code coverage to **89%**
  - Individual instrumentor coverage: HuggingFace (98%), OpenAI (98%), Anthropic (95%), Groq (94%)
  - Core modules at 100% coverage: config, metrics, logging, exceptions, __init__, cost_enrichment_processor
  - Updated 40+ tests to match new `create_span_wrapper()` pattern
  - Added 20 comprehensive tests for CostEnrichmentSpanProcessor (100% coverage)
    - 15 tests for GenAI semantic conventions
    - 5 tests for OpenInference semantic conventions

- **Documentation**
  - Updated all instrumentor docstrings to explain token extraction logic
  - Added comments documenting non-standard pricing models (hardware-based, local execution)
  - Improved code comments for complex fallback logic

## [0.1.2.dev0] - 2025-01-22

### Added

- **GPU Power Consumption Metric**
  - Added `gen_ai.gpu.power` observable gauge metric to track real-time GPU power consumption
  - Metric reports power usage in Watts with `gpu_id` and `gpu_name` attributes
  - Automatically collected alongside existing GPU metrics (utilization, memory, temperature)
  - Implementation in `genai_otel/gpu_metrics.py:97-102, 195-220`
  - Added test coverage in `tests/test_gpu_metrics.py:244-266`
  - Completes the GPU metrics suite with 5 total metrics: utilization, memory, temperature, power, and CO2 emissions

### Fixed

- **Test Fixes for HuggingFace and MistralAI Instrumentors**
  - Fixed HuggingFace instrumentor tests (2 failures) - corrected tracer mocking to use `instrumentor.tracer.start_span()` instead of `config.tracer.start_as_current_span()`
  - Fixed HuggingFace instrumentor tests - added `instrumentor.request_counter` mock for proper metrics assertion
  - Fixed MistralAI instrumentor test - corrected wrapt module mocking by adding to `sys.modules` instead of invalid module-level patch
  - All 395 tests now passing with zero failures
  - Tests modified: `tests/instrumentors/test_huggingface_instrumentor.py`, `tests/instrumentors/test_mistralai_instrumentor.py`

## [0.1.0] - 2025-01-20

**First Beta Release** 🎉

This is the first public release of genai-otel-instrument, a comprehensive OpenTelemetry auto-instrumentation library for LLM/GenAI applications with support for 15+ providers, frameworks, and MCP tools.

### Fixed

- **Phase 3.4 Fallback Semantic Conventions**
  - Fixed `AttributeError` when `openlit` package is not installed
  - Added missing `GEN_AI_SERVER_TTFT` and `GEN_AI_SERVER_TBT` constants to fallback `SC` class in `base.py`
  - Fixed MCP constant names in `mcp_instrumentors/base.py` to include `_METRIC` suffix
  - Library now works correctly with or without the `openlit` package

- **Third-Party Library Warnings**
  - Suppressed pydantic deprecation warnings from external dependencies
  - Added warning filters in `__init__.py` for runtime suppression
  - Added warning filters in `pyproject.toml` for pytest suppression
  - Clean output with zero warnings in both tests and production use

- **MistralAI Instrumentor Trace Collection**
  - **BREAKING**: Complete rewrite to support Mistral SDK v1.0+ properly
  - Fixed traces not being collected (was only collecting metrics)
  - Changed from class-level patching to instance-level instrumentation (Anthropic pattern)
  - Now wraps `Mistral.__init__` to instrument each client instance
  - Properly instruments: `client.chat.complete()`, `client.chat.stream()`, `client.embeddings.create()`
  - Tests: Simplified to 5 essential tests
  - Verified working with live API calls - traces now collected correctly

- **HuggingFace Instrumentor Trace Collection**
  - Fixed traces not being collected (was only collecting metrics)
  - Fixed incorrect tracer reference (`config.tracer` → `self.tracer`)
  - Properly initialize `self.config` in `instrument()` method
  - Updated to use `tracer.start_span()` instead of deprecated `start_as_current_span()`
  - Added proper span ending with `span.end()`
  - Verified working - traces now collected correctly

### Added

- **Granular Cost Tracking Tests (Phase 3.2 Coverage)**
  - Added 3 comprehensive tests for granular cost tracking functionality
  - `test_granular_cost_tracking_with_all_cost_types` - Tests all 6 cost types (prompt, completion, reasoning, cache_read, cache_write)
  - `test_granular_cost_tracking_with_zero_costs` - Validates zero-cost handling
  - `test_granular_cost_tracking_only_prompt_cost` - Tests embedding/prompt-only scenarios
  - Improved `base.py` coverage from 83% to 91%
  - Total tests: 405 → 408, all passing
  - Overall coverage maintained at 93%

- **OpenTelemetry Semantic Convention Compliance (Phase 1 & 2)**
  - Added support for `OTEL_SEMCONV_STABILITY_OPT_IN` environment variable for dual token attribute emission
  - Added `GENAI_ENABLE_CONTENT_CAPTURE` environment variable for opt-in prompt/completion content capture as span events
  - Added comprehensive span attributes to OpenAI instrumentor:
    - Request parameters: `gen_ai.operation.name`, `gen_ai.request.temperature`, `gen_ai.request.top_p`, `gen_ai.request.max_tokens`, `gen_ai.request.frequency_penalty`, `gen_ai.request.presence_penalty`
    - Response attributes: `gen_ai.response.id`, `gen_ai.response.model`, `gen_ai.response.finish_reasons`
  - Added event-based content capture for prompts and completions (disabled by default for security)
  - Added 8 new tests for Phase 2 enhancements (381 total tests, all passing)

- **Tool/Function Call Instrumentation (Phase 3.1)**
  - Added support for tracking tool/function calls in LLM responses (OpenAI function calling)
  - New span attributes:
    - `llm.tools` - JSON-serialized tool definitions from request
    - `llm.output_messages.{choice_idx}.message.tool_calls.{tc_idx}.tool_call.id` - Tool call ID
    - `llm.output_messages.{choice_idx}.message.tool_calls.{tc_idx}.tool_call.function.name` - Function name
    - `llm.output_messages.{choice_idx}.message.tool_calls.{tc_idx}.tool_call.function.arguments` - Function arguments
  - Enhanced OpenAI instrumentor to extract and record tool call information
  - Added 2 new tests for tool call instrumentation (383 total tests)

- **Granular Cost Tracking (Phase 3.2)**
  - Added granular cost breakdown with separate tracking for:
    - Prompt tokens cost (`gen_ai.usage.cost.prompt`)
    - Completion tokens cost (`gen_ai.usage.cost.completion`)
    - Reasoning tokens cost (`gen_ai.usage.cost.reasoning`) - for OpenAI o1 models
    - Cache read cost (`gen_ai.usage.cost.cache_read`) - for Anthropic prompt caching
    - Cache write cost (`gen_ai.usage.cost.cache_write`) - for Anthropic prompt caching
  - Added 5 new cost-specific metrics counters
  - Added 6 new span attributes for cost breakdown (`gen_ai.usage.cost.*`)
  - Added `calculate_granular_cost()` method to CostCalculator
  - Enhanced OpenAI instrumentor to extract reasoning tokens from `completion_tokens_details.reasoning_tokens`
  - Enhanced Anthropic instrumentor to extract cache tokens (`cache_read_input_tokens`, `cache_creation_input_tokens`)
  - Added 4 new tests for granular cost tracking (387 total tests, all passing)
  - Cost breakdown enables detailed analysis of:
    - OpenAI o1 models with separate reasoning token costs
    - Anthropic prompt caching with read/write cost separation
    - Per-request cost attribution by token type

- **MCP Metrics for Database Operations (Phase 3.3)**
  - Added `BaseMCPInstrumentor` base class with shared MCP-specific metrics
  - New MCP metrics with optimized histogram buckets:
    - `mcp.requests` - Counter for number of MCP requests
    - `mcp.client.operation.duration` - Histogram for operation duration (1ms to 10s buckets)
    - `mcp.request.size` - Histogram for request payload size (100B to 5MB buckets)
    - `mcp.response.size` - Histogram for response payload size (100B to 5MB buckets)
  - Enhanced `DatabaseInstrumentor` to use hybrid approach:
    - Keeps built-in OpenTelemetry instrumentors for full trace/span creation
    - Adds custom wrapt wrappers for MCP metrics collection
    - Instruments PostgreSQL (psycopg2), MongoDB (pymongo), and MySQL (mysql-connector)
  - Configured Views in `auto_instrument.py` to apply MCP histogram bucket boundaries
  - Added 4 new tests for BaseMCPInstrumentor (391 total tests, all passing)
  - Metrics include attributes for `db.system` and `mcp.operation` for filtering

- **Configurable GPU Collection Interval**
  - Added `gpu_collection_interval` configuration option (default: 5 seconds, down from 10)
  - New environment variable: `GENAI_GPU_COLLECTION_INTERVAL`
  - Fixes CO2 metrics not appearing for short-running scripts
  - GPU metrics and CO2 emissions now collected more frequently

- **Streaming Metrics for TTFT and TBT (Phase 3.4)**
  - Added streaming response detection and automatic metrics collection
  - New streaming metrics with optimized histogram buckets:
    - `gen_ai.server.ttft` - Time to First Token histogram (1ms to 10s buckets)
    - `gen_ai.server.tbt` - Time Between Tokens histogram (10ms to 2.5s buckets)
  - New span attribute for streaming:
    - `gen_ai.streaming.token_count` - Total number of chunks/tokens yielded
  - Enhanced `BaseInstrumentor` to detect `stream=True` parameter automatically
  - Added `_wrap_streaming_response()` helper method for streaming iterator wrapping
  - Changed span management from context manager to manual start/end for streaming support
  - Configured Views in `auto_instrument.py` to apply streaming histogram bucket boundaries
  - Added 2 new tests for streaming metrics (405 total tests, all passing)
  - Streaming metrics enable analysis of:
    - Real-time response latency (TTFT)
    - Token generation speed consistency (TBT)
    - Overall streaming performance for user experience optimization

### Changed

- **BREAKING: Metric names now use OpenTelemetry semantic conventions**
  - `genai.requests` → `gen_ai.requests`
  - `genai.tokens` → `gen_ai.client.token.usage`
  - `genai.latency` → `gen_ai.client.operation.duration`
  - `genai.cost` → `gen_ai.usage.cost`
  - `genai.errors` → `gen_ai.client.errors`
  - All GPU metrics now use `gen_ai.gpu.*` prefix (was `genai.gpu.*`)
  - Update your dashboards and alerting rules accordingly
- **Token attribute naming now supports dual emission**
  - When `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai/dup`, both old and new token attributes are emitted:
    - New (always): `gen_ai.usage.prompt_tokens`, `gen_ai.usage.completion_tokens`
    - Old (with /dup): `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`
  - Default (`gen_ai`): Only new attributes are emitted

### Fixed

- **CRITICAL: GPU metrics now use correct metric types and callbacks**
  - Changed `gpu_utilization_counter` from Counter to ObservableGauge (utilization is 0-100%, not monotonic)
  - Fixed `gpu_memory_used_gauge` and `gpu_temperature_gauge` to use callbacks instead of manual `.add()` calls
  - Added callback methods: `_observe_gpu_utilization()`, `_observe_gpu_memory()`, `_observe_gpu_temperature()`
  - Fixed CO2 metric name from `genai.co-2.emissions` to `gen_ai.co2.emissions`
  - Removed dual-thread architecture (now uses single CO2 collection thread, ObservableGauges auto-collected)
  - All GPU metrics now correctly reported with proper data types
  - Updated 19 GPU metrics tests to match new implementation
- **Histogram buckets now properly applied via OpenTelemetry Views**
  - Created View with ExplicitBucketHistogramAggregation for `gen_ai.client.operation.duration`
  - Applies `_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS` from metrics.py
  - Buckets optimized for LLM latencies (0.01s to 81.92s)
  - No longer uses default OTel buckets (which were poorly suited for GenAI workloads)
- **CRITICAL: Made OpenInference instrumentations optional to support Python 3.8 and 3.9**
  - Moved `openinference-instrumentation-smolagents`, `openinference-instrumentation-litellm`, `openinference-instrumentation-mcp`, and `litellm` to optional dependencies
  - These packages require Python >= 3.10 and were causing installation failures on Python 3.8 and 3.9
  - Added new `openinference` optional dependency group for users on Python 3.10+
  - Install with: `pip install genai-otel-instrument[openinference]` (Python 3.10+ only)
  - Package now installs cleanly on Python 3.8, 3.9, 3.10, 3.11, and 3.12
  - Conditional imports prevent errors when OpenInference packages are not installed
  - Relaxed `opentelemetry-semantic-conventions` version constraint from `>=0.58b0` to `>=0.45b0` for Python 3.8 compatibility
  - Added missing `opentelemetry-instrumentation-mysql` to core dependencies
  - Removed `mysql==0.0.3` dependency (requires system MySQL libraries not available in CI)
  - Added `sqlalchemy>=1.4.0` to core dependencies (required by sqlalchemy instrumentor)
- **CRITICAL: Fixed CLI wrapper to execute scripts in same process**
  - Changed from `subprocess.run()` to `runpy.run_path()` to ensure instrumentation hooks are active
  - Supports both `genai-instrument python script.py` and `genai-instrument script.py` formats
  - Script now runs in the same process where instrumentation is initialized, fixing ModuleNotFoundError and ensuring proper telemetry collection
  - Added tests for both CLI usage patterns (7 tests total, all passing)

- **CRITICAL: Fixed MCP dependency conflict error**
  - Removed "mcp" from `DEFAULT_INSTRUMENTORS` list to prevent dependency conflict when mcp library (>= 1.6.0) is not installed
  - Added explanatory comments in `genai_otel/config.py` - users can still enable via `GENAI_ENABLED_INSTRUMENTORS` environment variable
  - Most users don't need the specialized Model Context Protocol library for server/client development
- **Fixed test failures in instrumentor mock tests (11 total failures resolved)**
  - Fixed `test_openai_instrumentor.py::test_instrument_client` - corrected mock to return decorator function instead of wrapped function directly
  - Fixed `test_anthropic_instrumentor.py::test_instrument_client_with_messages` - applied same decorator pattern fix
  - Fixed OpenInference instrumentor tests (litellm, mcp, smolagents) - changed assertions to expect `instrument()` without config parameter, matching actual API in `auto_instrument.py:208-211`
  - Fixed 6 MCP manager test failures in `tests/mcp_instrumentors/test_manager.py` - updated setUp() to enable HTTP instrumentation for tests that expect it
- **All tests now passing: 371 passed, 0 failed, 98% coverage**
- **CRITICAL: Fixed instrumentor null check issues**
  - Added null checks for metrics (`request_counter`, `token_counter`, `cost_counter`) in all instrumentors to prevent `AttributeError: 'NoneType' object has no attribute 'add'`
  - Fixed 9 instrumentors: Ollama, AzureOpenAI, MistralAI, Groq, Cohere, VertexAI, TogetherAI, Replicate
- **CRITICAL: Fixed wrapt decorator issues in OpenAI and Anthropic instrumentors**
  - Fixed `IndexError: tuple index out of range` by properly applying `create_span_wrapper()` decorator to original methods
  - OpenAI instrumentor (`openai_instrumentor.py:82-86`)
  - Anthropic instrumentor (`anthropic_instrumentor.py:76-80`)
- **CRITICAL: Fixed OpenInference instrumentor initialization**
  - Fixed smolagents, litellm, and mcp instrumentors not being called correctly (they don't accept config parameter)
  - Added `OPENINFERENCE_INSTRUMENTORS` set to handle different instrumentation API
  - Added smolagents, litellm, mcp to `DEFAULT_INSTRUMENTORS` list
- **CRITICAL: Fixed OTLP HTTP exporter configuration issues**
  - Fixed `AttributeError: 'function' object has no attribute 'ok'` caused by requests library instrumentation conflicting with OTLP exporters
  - Disabled `RequestsInstrumentor` in MCP manager to prevent breaking OTLP HTTP exporters that use requests internally
  - Disabled requests wrapping in `APIInstrumentor` to avoid class-level Session patching
  - Fixed endpoint configuration to use environment variables so exporters correctly append `/v1/traces` and `/v1/metrics` paths
  - Updated logging to show full endpoints for both trace and metrics exporters
- Corrected indentation and patch targets in `tests/instrumentors/test_ollama_instrumentor.py` to resolve `IndentationError` and `AttributeError`.
- Fixed test failures in `tests/test_metrics.py` by ensuring proper reset of OpenTelemetry providers and correcting assertions.
- Updated `genai_otel/instrumentors/ollama_instrumentor.py` to align with corrected test logic.
- Addressed test failures in `tests/instrumentors/test_huggingface_instrumentor.py` related to missing attributes and call assertions.
- Fix HuggingFace instrumentation to correctly set span attributes and pass tests.
- Resolve `AttributeError` related to `TraceContextTextMapPropagator` in test files by correcting import paths.
- Fixed `setup_meter` function in `genai_otel/metrics.py` to correctly configure OpenTelemetry MeterProvider with metric readers and handle invalid OTLP endpoint/headers gracefully.
- Corrected `tests/test_metrics.py` to properly reset MeterProvider state between tests and accurately access metric exporter attributes, resolving `TypeError` and `AssertionError`s.
- Fixed `cost_counter` not being called in `tests/instrumentors/test_base.py` by ensuring `BaseInstrumentor._shared_cost_counter` is patched with a distinct mock before `ConcreteInstrumentor` instantiation.
- Resolved `setup_tracing` failures in `tests/test_config.py` by correcting `genai_otel/config.py`'s `setup_tracing` function and adjusting the `reset_tracer` fixture to mock `TracerProvider` correctly.
- Refined Hugging Face instrumentation tests for better attribute handling and mock accuracy.
- Improved `tests/test_metrics.py` by ensuring proper isolation of OpenTelemetry providers using `NoOp` implementations in the `reset_otel` fixture.

### Added

- **Comprehensive CI/CD improvements**
  - Added `build-and-install-test` job to test.yml workflow for package build and installation validation
  - Added pre-release-check.yml workflow that mimics manual test_release.sh script
  - Enhanced publish.yml with full test suite, code quality checks, and installation testing before publishing
  - Added workflow documentation in .github/workflows/README.md
  - CI now tests package installation and CLI functionality in isolated environments
  - Pre-release validation runs across Ubuntu, Windows, and macOS with Python 3.9 and 3.12
- **Fine-grained HTTP instrumentation control**
  - Added `enable_http_instrumentation` configuration option (default: `false`)
  - Environment variable: `GENAI_ENABLE_HTTP_INSTRUMENTATION`
  - Allows enabling HTTP/httpx instrumentation without disabling all MCP instrumentation (databases, vector DBs, Redis, Kafka)
- Support for `SERVICE_INSTANCE_ID` and environment attributes in resource creation (Issue #XXX)
- Configurable timeout for OTLP exporters via `OTEL_EXPORTER_OTLP_TIMEOUT` environment variable (Issue #XXX)
- Added openinference instrumentation dependencies: `openinference-instrumentation==0.1.31`, `openinference-instrumentation-litellm==0.1.19`, `openinference-instrumentation-mcp==1.3.0`, `openinference-instrumentation-smolagents==0.1.11`, and `openinference-semantic-conventions==0.1.17` (Issue #XXX)
- Explicit configuration of `TraceContextTextMapPropagator` for W3C trace context propagation (Issue #XXX)
- Created examples for LiteLLM and Smolagents instrumentors

### Changed

- **HTTP instrumentation now opt-in instead of opt-out**
  - HTTP/httpx instrumentation is now disabled by default (`enable_http_instrumentation=false`)
  - MCP instrumentation remains enabled by default (databases, vector DBs, Redis, Kafka all work out of the box)
  - Set `GENAI_ENABLE_HTTP_INSTRUMENTATION=true` or `enable_http_instrumentation=True` to enable HTTP tracing
- **Updated Mistral AI example for new SDK (v1.0+)**
  - Migrated from deprecated `mistralai.client.MistralClient` to new `mistralai.Mistral` API
- Updated logging configuration to allow log level via environment variable and implement log rotation (Issue #XXX)

### Tests

- Fixed tests for base/redis and auto instrument (a701603)
- Updated `test_auto_instrument.py` assertions to match new OTLP exporter configuration (exporters now read endpoint from environment variables instead of direct parameters)

[Unreleased]: https://github.com/Mandark-droid/genai_otel_instrument/compare/v1.22.0...HEAD
[1.22.0]: https://github.com/Mandark-droid/genai_otel_instrument/compare/v1.21.0...v1.22.0
[1.21.0]: https://github.com/Mandark-droid/genai_otel_instrument/compare/v1.20.2...v1.21.0
[1.8.0]: https://github.com/Mandark-droid/genai_otel_instrument/compare/v1.7.0...v1.8.0
[1.7.0]: https://github.com/Mandark-droid/genai_otel_instrument/compare/v1.6.1...v1.7.0
[0.1.2.dev0]: https://github.com/Mandark-droid/genai_otel_instrument/compare/v0.1.0...v0.1.2.dev0
[0.1.0]: https://github.com/Mandark-droid/genai_otel_instrument/releases/tag/v0.1.0
