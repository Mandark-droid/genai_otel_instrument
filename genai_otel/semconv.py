"""Semantic convention constants for GenAI and MCP metrics.

These constants define the metric and attribute names used throughout the
instrumentation library, following OpenTelemetry GenAI semantic conventions.
"""


class SemanticConvention:
    """Semantic convention constants for metric and attribute names."""

    # GenAI Client metrics
    GEN_AI_REQUESTS = "gen_ai.requests"
    GEN_AI_CLIENT_TOKEN_USAGE = "gen_ai.client.token.usage"
    GEN_AI_CLIENT_OPERATION_DURATION = "gen_ai.client.operation.duration"
    GEN_AI_USAGE_COST = "gen_ai.usage.cost"
    GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    GEN_AI_USAGE_REASONING_TOKENS = "gen_ai.usage.reasoning_tokens"

    # GenAI Server metrics (streaming)
    GEN_AI_SERVER_TTFT = "gen_ai.server.ttft"
    GEN_AI_SERVER_TBT = "gen_ai.server.tbt"

    # DB metrics
    DB_CLIENT_OPERATION_DURATION = "db.client.operation.duration"
    DB_REQUESTS = "db.requests"

    # GenAI multimodal content-part attributes (additive; not yet in upstream OTel
    # semconv as of April 2026). Templates take .format(n=<msg_idx>, m=<part_idx>).
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
