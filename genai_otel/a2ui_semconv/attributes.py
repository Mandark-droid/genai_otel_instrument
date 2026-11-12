"""Attribute names for A2UI (Agent-to-UI) spans.

A2UI defines no telemetry of its own and OpenTelemetry has no convention for it, so
every ``a2ui.*`` key here is an addition of this library and is provisional: it may be
renamed if a convention settles on other names. The ``error.type`` key is the standard
one.
"""

from typing import Final

#: The A2UI extension URI for A2A, at v1.0. Older protocol versions use the same prefix.
EXTENSION_URI: Final[str] = "https://a2ui.org/a2a-extension/a2ui/v1.0"
EXTENSION_URI_PREFIX: Final[str] = "https://a2ui.org/a2a-extension/a2ui/"

#: Media types of an A2UI part. The second is deprecated and used by v0.8 and v0.9, and
#: a receiver must accept both.
MIME_TYPE: Final[str] = "application/a2ui+json"
DEPRECATED_MIME_TYPE: Final[str] = "application/json+a2ui"
MIME_TYPES = frozenset({MIME_TYPE, DEPRECATED_MIME_TYPE})

#: Message metadata keys that carry the renderer's capabilities. The first is the v1.0
#: extension's, the second is v0.8 and v0.9's (and the Python agent SDK's constant).
CAPABILITIES_KEYS = ("a2uiRendererCapabilities", "a2uiClientCapabilities")

#: Span event written once per renderer error or function error.
ERROR_EVENT: Final[str] = "a2ui.error"

SIDE_AGENT: Final[str] = "agent"
SIDE_RENDERER: Final[str] = "renderer"

VALIDATION_VALID: Final[str] = "valid"
VALIDATION_INVALID: Final[str] = "invalid"


class A2UIAttributes:
    """Attribute names written on A2UI spans and on A2A spans that carry A2UI."""

    # --- Always recorded: identifiers, counts and codes, never content --------------
    PROTOCOL_VERSION: Final[str] = "a2ui.protocol.version"
    AGENT_MESSAGE_COUNT: Final[str] = "a2ui.agent_message.count"
    RENDERER_MESSAGE_COUNT: Final[str] = "a2ui.renderer_message.count"
    MESSAGE_TYPES: Final[str] = "a2ui.message.types"
    SURFACE_ID: Final[str] = "a2ui.surface.id"
    SURFACE_COUNT: Final[str] = "a2ui.surface.count"
    CATALOG_ID: Final[str] = "a2ui.catalog.id"
    CATALOG_IDS: Final[str] = "a2ui.catalog.ids"
    COMPONENT_COUNT: Final[str] = "a2ui.component.count"
    RENDERER_SUPPORTED_CATALOG_IDS: Final[str] = "a2ui.renderer.supported_catalog_ids"
    ACTION_NAME: Final[str] = "a2ui.action.name"
    FUNCTION_NAME: Final[str] = "a2ui.function.name"
    ERROR_CODES: Final[str] = "a2ui.error.codes"
    ERROR_COUNT: Final[str] = "a2ui.error.count"
    VALIDATION_OUTCOME: Final[str] = "a2ui.validation.outcome"
    EXTENSION_ACTIVE: Final[str] = "a2ui.extension.active"
    ERROR_TYPE: Final[str] = "error.type"

    # --- Event attributes (the code and the surface are identifiers) ----------------
    EVENT_ERROR_CODE: Final[str] = "a2ui.error.code"

    # --- Content: recorded only when content capture is on --------------------------
    MESSAGES: Final[str] = "a2ui.messages"
    DATA_MODEL: Final[str] = "a2ui.data_model"
    ACTION_CONTEXT: Final[str] = "a2ui.action.context"
    ERROR_PATHS: Final[str] = "a2ui.error.paths"
    ERROR_MESSAGES: Final[str] = "a2ui.error.messages"
    COMPONENT_TYPES: Final[str] = "a2ui.component.types"


#: Every key above that can carry business text, user data or the UI itself.
CONTENT_KEYS = frozenset(
    {
        A2UIAttributes.MESSAGES,
        A2UIAttributes.DATA_MODEL,
        A2UIAttributes.ACTION_CONTEXT,
        A2UIAttributes.ERROR_PATHS,
        A2UIAttributes.ERROR_MESSAGES,
        A2UIAttributes.COMPONENT_TYPES,
    }
)
