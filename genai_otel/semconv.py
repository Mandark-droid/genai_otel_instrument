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

    # GenAI Server metrics (streaming)
    GEN_AI_SERVER_TTFT = "gen_ai.server.ttft"
    GEN_AI_SERVER_TBT = "gen_ai.server.tbt"

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
