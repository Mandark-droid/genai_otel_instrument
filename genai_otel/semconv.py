"""Semantic convention constants for GenAI and MCP metrics.

These constants define the metric and attribute names used throughout the
instrumentation library, following OpenTelemetry GenAI semantic conventions.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# OTEL_SEMCONV_STABILITY_OPT_IN
#
# Upstream defines this as a COMMA-SEPARATED list of tokens shared by every
# instrumentation area -- "http/dup", "database/dup", "gen_ai/dup" and so on
# can all appear together. Testing it with a substring check therefore reads
# other areas' opt-ins as our own: `"dup" in "http/dup"` is True, and
# `"gen_ai" in ...` matches nothing meaningful when a user writes
# "gen_ai/dup,http". Both mistakes existed before this helper.
#
# Worse case the substring check got wrong: "gen_ai,http/dup" means *current
# GenAI names only* plus an unrelated HTTP opt-in, and was read as dual GenAI
# emission because the string contains "dup".
# ---------------------------------------------------------------------------
_GENAI_CURRENT_ONLY = "gen_ai"
_GENAI_DUP = "gen_ai/dup"


def genai_semconv_modes(opt_in: str | None) -> tuple[bool, bool]:
    """Return ``(emit_current, emit_superseded)`` for a raw opt-in string.

    * ``gen_ai/dup``  -> both spellings.
    * ``gen_ai``      -> current names only; the caller has explicitly asked to
      drop the superseded ones.
    * anything else, including unset -> **both**.

    That last line is a deliberate departure from upstream's "unset means old
    only", and it is the transitional default for this library. The GenAI
    conventions are still Status: Development, so there is no stable tier to
    fall back to -- and emitting *only* the current names by default is how
    1.9.0 silently zeroed token counts for every consumer still reading the
    superseded ones. Two extra integer attributes per span is a much cheaper
    mistake than a cost dashboard that reads zero and looks measured.

    The default becomes current-only at 2.0, where a breaking change belongs.
    """
    tokens = _tokens(opt_in)
    if _GENAI_DUP in tokens:
        return True, True
    if _GENAI_CURRENT_ONLY in tokens:
        return True, False
    return True, True


def genai_tier_opted_in(opt_in: str | None) -> bool:
    """True when the caller has opted into the GenAI semconv tier at all.

    Deliberately NOT the same question as :func:`genai_semconv_modes`. That one
    decides which *names* to use for attributes we emit regardless; this one gates
    an additional and much heavier payload -- the canonical
    ``gen_ai.input.messages`` / ``gen_ai.output.messages`` JSON, which carries
    message content.

    So an explicitly empty opt-in means "do not emit that payload" and must be
    honoured, whereas for naming an empty value simply falls back to the safe
    default. Collapsing the two would silently start emitting message content for
    someone who had opted out of it -- the opposite failure to a dropped
    attribute, and a far worse one.
    """
    return bool(_tokens(opt_in) & {_GENAI_CURRENT_ONLY, _GENAI_DUP})


def _tokens(opt_in: str | None) -> set[str]:
    """Split the comma-separated opt-in list into stripped, non-empty tokens."""
    return {token.strip() for token in (opt_in or "").split(",") if token.strip()}


class SemanticConvention:
    """Semantic convention constants for metric and attribute names."""

    # GenAI Client metrics
    GEN_AI_REQUESTS = "gen_ai.requests"
    GEN_AI_CLIENT_TOKEN_USAGE = "gen_ai.client.token.usage"
    GEN_AI_CLIENT_OPERATION_DURATION = "gen_ai.client.operation.duration"
    GEN_AI_USAGE_COST = "gen_ai.usage.cost"
    GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    # Superseded by the two above (semantic-conventions v1.27.0). Named here so
    # the dual-emission policy reads from one place instead of string literals.
    GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
    GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"

    # Model provider. `gen_ai.system` was renamed to `gen_ai.provider.name`;
    # instrumentors still write the superseded spelling as a raw literal, and
    # base.py mirrors it onto the current one centrally.
    GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
    GEN_AI_SYSTEM = "gen_ai.system"
    # Upstream-standardised name (in registry at Development stability since
    # open-telemetry/semantic-conventions#3194; migrated to
    # semantic-conventions-genai). The value SHOULD also be included in
    # `gen_ai.usage.output_tokens`.
    GEN_AI_USAGE_REASONING_TOKENS = "gen_ai.usage.reasoning.output_tokens"

    # Prompt-cache token breakdown. `cache_write` is the current registry
    # spelling: semantic-conventions-genai#440 (merged 2026-08-20) renamed
    # `gen_ai.usage.cache_creation.input_tokens` to
    # `gen_ai.usage.cache_write.input_tokens`. The superseded name is still
    # emitted alongside it under the `gen_ai/dup` policy, the same way
    # prompt/completion tokens are handled above -- a backend that follows the
    # current conventions sees zero cached tokens if only the old name is
    # present, and Anthropic prompt-cache economics are the main reason anyone
    # reads these at all.
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
    GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS = "gen_ai.usage.cache_write.input_tokens"
    GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = "gen_ai.usage.cache_creation.input_tokens"

    # Per-modality token breakdown (semantic-conventions-genai#440, merged
    # 2026-08-20). Each value is a subset of the corresponding total, i.e.
    # `gen_ai.usage.text.input_tokens` is included in
    # `gen_ai.usage.input_tokens`, so consumers must not sum them alongside it.
    # Templates take .format(modality=...). Modalities are the registry's:
    # text, image, audio (video/document carry no token counts on any provider
    # we have seen, so they are not emitted rather than emitted as zero).
    GEN_AI_USAGE_MODALITY_INPUT_TOKENS = "gen_ai.usage.{modality}.input_tokens"
    GEN_AI_USAGE_MODALITY_OUTPUT_TOKENS = "gen_ai.usage.{modality}.output_tokens"
    GEN_AI_USAGE_MODALITY_CACHE_READ_INPUT_TOKENS = (
        "gen_ai.usage.{modality}.cache_read.input_tokens"
    )
    TOKEN_MODALITIES = ("text", "image", "audio")

    # Embeddings request shape. `gen_ai.embeddings.dimension.count` and the
    # plural `gen_ai.request.encoding_formats` are the registry spellings; the
    # singular/`request.dimensions` forms below are this library's older names,
    # kept under the dup policy so existing dashboards keep resolving.
    GEN_AI_EMBEDDINGS_DIMENSION_COUNT = "gen_ai.embeddings.dimension.count"
    GEN_AI_REQUEST_ENCODING_FORMATS = "gen_ai.request.encoding_formats"
    GEN_AI_REQUEST_DIMENSIONS = "gen_ai.request.dimensions"
    GEN_AI_REQUEST_ENCODING_FORMAT = "gen_ai.request.encoding_format"

    # Finish reasons are an array in the conventions. The singular form is this
    # library's older spelling and is emitted alongside it under the dup policy.
    GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
    GEN_AI_RESPONSE_FINISH_REASON = "gen_ai.response.finish_reason"

    # Request parameters that the conventions define and providers commonly
    # echo back, but which this library did not previously record.
    GEN_AI_REQUEST_SEED = "gen_ai.request.seed"
    GEN_AI_REQUEST_STREAM = "gen_ai.request.stream"
    GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"
    GEN_AI_REQUEST_CHOICE_COUNT = "gen_ai.request.choice.count"
    GEN_AI_OUTPUT_TYPE = "gen_ai.output.type"
    # Span-attribute counterpart of the time-to-first-token metric; set only on
    # streamed calls, for the same reason the metric is.
    GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK = "gen_ai.response.time_to_first_chunk"

    # Per-invocation agent budget governance
    # (semantic-conventions-genai#425). `gen_ai.request.max_tokens` is a
    # per-call limit passed to the model API; these are the invocation-level
    # envelope a framework enforces across many calls, which is a different
    # thing and is what runaway-agent alerting needs.
    GEN_AI_AGENT_TOKEN_BUDGET = "gen_ai.agent.token_budget"
    GEN_AI_AGENT_TOKEN_BUDGET_CONSUMED = "gen_ai.agent.token_budget.consumed"
    GEN_AI_AGENT_ITERATION_BUDGET = "gen_ai.agent.iteration_budget"
    GEN_AI_AGENT_ITERATION_BUDGET_CONSUMED = "gen_ai.agent.iteration_budget.consumed"
    GEN_AI_INVOKE_AGENT_TOKEN_BUDGET_UTILIZATION = "gen_ai.invoke_agent.token_budget.utilization"

    # Server identity for the model endpoint. Conditionally required on
    # inference spans; the reference scenarios in semantic-conventions-genai
    # emit both. Derived from the SDK client's base URL where the client
    # exposes one, and omitted entirely when it cannot be observed -- an
    # absent attribute reads as "unknown endpoint", whereas a guessed default
    # host silently misattributes self-hosted and proxied traffic.
    SERVER_ADDRESS = "server.address"
    SERVER_PORT = "server.port"

    # GenAI Server metrics (streaming)
    GEN_AI_SERVER_TTFT = "gen_ai.server.ttft"
    GEN_AI_SERVER_TBT = "gen_ai.server.tbt"
    # The two names above are this library's own abbreviations and predate the
    # upstream spelling; the ones below are what the GenAI conventions define
    # and what downstream platforms actually look for. Emitted as both metrics
    # and span attributes, and only ever on a streamed call -- an absent
    # attribute reads as "not measured", whereas a zero TTFT is
    # indistinguishable from a very fast first token and silently drags any
    # average that includes it.
    GEN_AI_SERVER_TIME_TO_FIRST_TOKEN = "gen_ai.server.time_to_first_token"
    GEN_AI_SERVER_TIME_PER_OUTPUT_TOKEN = "gen_ai.server.time_per_output_token"
    # Set only when TTFT was measured but the provider gave us no output-token
    # count to divide by, so TPOT is omitted rather than estimated.
    GEN_AI_STREAMING_TPOT_UNAVAILABLE_REASON = "gen_ai.streaming.tpot_unavailable_reason"

    # DB metrics
    DB_CLIENT_OPERATION_DURATION = "db.client.operation.duration"
    DB_REQUESTS = "db.requests"

    # GenAI multimodal content-part attributes — flat namespace for queryable
    # per-part attribution. Co-emitted alongside the upstream-canonical
    # `gen_ai.input.messages` / `gen_ai.output.messages` JSON (see media/canonical.py),
    # which conforms to the gen-ai message schemas in semantic-conventions-genai
    # (PR #142 merged; #143/#144 in review).
    # Templates take .format(n=<msg_idx>, m=<part_idx>).
    GEN_AI_PROMPT_ROLE = "gen_ai.prompt.{n}.role"
    GEN_AI_PROMPT_CONTENT_TYPE = "gen_ai.prompt.{n}.content.{m}.type"
    GEN_AI_PROMPT_CONTENT_TEXT = "gen_ai.prompt.{n}.content.{m}.text"
    GEN_AI_PROMPT_CONTENT_MEDIA_URI = "gen_ai.prompt.{n}.content.{m}.media_uri"
    GEN_AI_PROMPT_CONTENT_MEDIA_MIME = "gen_ai.prompt.{n}.content.{m}.media_mime_type"
    GEN_AI_PROMPT_CONTENT_MEDIA_BYTES = "gen_ai.prompt.{n}.content.{m}.media_byte_size"
    GEN_AI_PROMPT_CONTENT_MEDIA_SOURCE = "gen_ai.prompt.{n}.content.{m}.media_source"
    GEN_AI_COMPLETION_ROLE = "gen_ai.completion.{n}.role"
    GEN_AI_COMPLETION_CONTENT_TYPE = "gen_ai.completion.{n}.content.{m}.type"
    GEN_AI_COMPLETION_CONTENT_TEXT = "gen_ai.completion.{n}.content.{m}.text"
    GEN_AI_COMPLETION_CONTENT_MEDIA_URI = "gen_ai.completion.{n}.content.{m}.media_uri"
    GEN_AI_COMPLETION_CONTENT_MEDIA_MIME = "gen_ai.completion.{n}.content.{m}.media_mime_type"
    GEN_AI_COMPLETION_CONTENT_MEDIA_BYTES = "gen_ai.completion.{n}.content.{m}.media_byte_size"
    GEN_AI_COMPLETION_CONTENT_MEDIA_SOURCE = "gen_ai.completion.{n}.content.{m}.media_source"
    GEN_AI_MEDIA_STRIPPED_REASON = "gen_ai.media.stripped_reason"

    # MCP metrics
    MCP_REQUESTS = "mcp.requests"
    MCP_CLIENT_OPERATION_DURATION_METRIC = "mcp.client.operation.duration"
    MCP_REQUEST_SIZE = "mcp.request.size"
    MCP_RESPONSE_SIZE_METRIC = "mcp.response.size"
    MCP_TOOL_CALLS = "mcp.tool_calls"
    MCP_RESOURCE_READS = "mcp.resource.reads"
    MCP_PROMPT_GETS = "mcp.prompt_gets"
    MCP_TRANSPORT_USAGE = "mcp.transport.usage"
    MCP_ERRORS = "mcp.errors"
    MCP_OPERATION_SUCCESS_RATE = "mcp.operation.success_rate"
