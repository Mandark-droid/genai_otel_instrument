"""Attribute names for A2A (Agent2Agent) spans.

The ``a2a.*`` keys in the first block, and the reused ``gen_ai.*`` / ``error.*`` /
``server.*`` keys, follow the OpenTelemetry A2A semantic conventions (development
status). The second block holds a few ``a2a.*`` keys of our own, for the counts the
conventions do not name yet. They are not part of the upstream convention and may be
renamed if it settles on other names.
"""

from typing import Final


class A2AAttributes:
    """Attribute names written on A2A spans."""

    # --- From the OpenTelemetry A2A conventions -------------------------------------
    METHOD_NAME: Final[str] = "a2a.method.name"
    PROTOCOL_VERSION: Final[str] = "a2a.protocol.version"
    TENANT: Final[str] = "a2a.tenant"
    MESSAGE_ID: Final[str] = "a2a.message.id"
    MESSAGE_REFERENCE_TASK_IDS: Final[str] = "a2a.message.reference_task_ids"
    TASK_ID: Final[str] = "a2a.task.id"
    TASK_STATE: Final[str] = "a2a.task.state"

    #: PROVISIONAL. Which binding carried the call: ``jsonrpc``, ``grpc`` or
    #: ``http_json``. The convention has not settled how to record this (issue #612),
    #: so the name may change.
    PROTOCOL_BINDING: Final[str] = "a2a.protocol.binding"

    # --- Reused from other conventions ----------------------------------------------
    #: The A2A ``contextId``.
    CONVERSATION_ID: Final[str] = "gen_ai.conversation.id"
    OPERATION_NAME: Final[str] = "gen_ai.operation.name"
    AGENT_NAME: Final[str] = "gen_ai.agent.name"
    AGENT_DESCRIPTION: Final[str] = "gen_ai.agent.description"
    AGENT_VERSION: Final[str] = "gen_ai.agent.version"
    ERROR_TYPE: Final[str] = "error.type"
    SERVER_ADDRESS: Final[str] = "server.address"
    SERVER_PORT: Final[str] = "server.port"

    # --- genai_otel additions (not in the upstream convention) ----------------------
    #: Parts of the request message, by kind. Counts only, never content.
    MESSAGE_TEXT_PART_COUNT: Final[str] = "a2a.message.text_part_count"
    MESSAGE_FILE_PART_COUNT: Final[str] = "a2a.message.file_part_count"
    MESSAGE_DATA_PART_COUNT: Final[str] = "a2a.message.data_part_count"
    MESSAGE_FILE_MEDIA_TYPES: Final[str] = "a2a.message.file_media_types"
    #: Parts of what the agent sent back (reply message, status message, artifacts).
    RESPONSE_TEXT_PART_COUNT: Final[str] = "a2a.response.text_part_count"
    RESPONSE_FILE_PART_COUNT: Final[str] = "a2a.response.file_part_count"
    RESPONSE_DATA_PART_COUNT: Final[str] = "a2a.response.data_part_count"
    RESPONSE_FILE_MEDIA_TYPES: Final[str] = "a2a.response.file_media_types"
    ARTIFACT_COUNT: Final[str] = "a2a.artifact.count"
    #: Agent Card skills: the count always, the names only under content capture.
    AGENT_SKILL_COUNT: Final[str] = "a2a.agent.skill_count"

    # --- Content: recorded only when content capture is on --------------------------
    MESSAGE_TEXT: Final[str] = "a2a.message.text"
    MESSAGE_DATA: Final[str] = "a2a.message.data"
    RESPONSE_TEXT: Final[str] = "a2a.response.text"
    RESPONSE_DATA: Final[str] = "a2a.response.data"
    AGENT_SKILL_NAMES: Final[str] = "a2a.agent.skill_names"


#: Every key above that can carry prompt, input, output or business text.
CONTENT_KEYS = frozenset(
    {
        A2AAttributes.MESSAGE_TEXT,
        A2AAttributes.MESSAGE_DATA,
        A2AAttributes.RESPONSE_TEXT,
        A2AAttributes.RESPONSE_DATA,
        A2AAttributes.AGENT_SKILL_NAMES,
    }
)

BINDING_JSONRPC: Final[str] = "jsonrpc"
BINDING_GRPC: Final[str] = "grpc"
BINDING_HTTP_JSON: Final[str] = "http_json"

#: Value of ``gen_ai.operation.name`` on client spans, so a backend can draw an edge
#: from the calling agent to the called one.
OPERATION_INVOKE_AGENT: Final[str] = "invoke_agent"
