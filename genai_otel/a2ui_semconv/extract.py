"""Read A2UI messages into span attributes.

One accumulator, :class:`A2uiObservation`, is shared by the A2A path (parts found in an
A2A message or stream) and by the agent-SDK wrappers (what ``parse_response`` produced,
what a renderer sent). It reads plain dicts in the wire format, so it imports neither
``a2a`` nor ``a2ui`` and can neither be the reason an import fails nor depend on a
version of either.

What is recorded follows one rule (decision A2UI-D3). Message types, protocol version,
surface and catalog ids, counts, action and function names and error codes are always
recorded: they describe the exchange, not what was said. The messages themselves, the
data-model values, an action's context, an error's path and message and component type
names are recorded only with content capture on, truncated to the content limit. The
content of image or file URLs inside components is never recorded.

v0.8 messages are recognised (they have no ``version`` field) and counted, and nothing
more is read from them: the version is deprecated and its shapes differ.
"""

import json
import logging
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from .attributes import (
    CAPABILITIES_KEYS,
    ERROR_EVENT,
    EXTENSION_URI_PREFIX,
    MIME_TYPES,
    SIDE_AGENT,
    SIDE_RENDERER,
    VALIDATION_INVALID,
    A2UIAttributes,
)

logger = logging.getLogger(__name__)

#: Agent-to-renderer message keys, and renderer-to-agent ones.
AGENT_KINDS = frozenset(
    {
        "createSurface",
        "updateComponents",
        "updateDataModel",
        "deleteSurface",
        "callRendererFunction",
        "agentFunctionResponse",
        "callFunction",  # v0.9
    }
)
RENDERER_KINDS = frozenset({"action", "callAgentFunction", "rendererFunctionResponse", "error"})

_REDACTED = "[redacted]"
_URL_KEY = re.compile(r"^(url|src|uri)$|(Url|URL|Uri|URI)$")
_CAMEL = re.compile(r"(?<!^)(?=[A-Z])")


def snake(name: str) -> str:
    """``updateComponents`` as ``update_components``."""
    return _CAMEL.sub("_", name).lower()


def normalise_version(value: Any) -> Optional[str]:
    """``"1.0"``, ``"v0.9"`` and ``"v0.9.1"`` as ``1.0``, ``0.9`` and ``0.9``."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text[:1] in ("v", "V"):
        text = text[1:]
    pieces = text.split(".")
    if len(pieces) < 2 or not all(p.isdigit() for p in pieces[:2]):
        return None
    return f"{int(pieces[0])}.{int(pieces[1])}"


def is_extension_uri(value: Any) -> bool:
    """True for the A2UI extension URI, at any protocol version."""
    return isinstance(value, str) and value.strip().startswith(EXTENSION_URI_PREFIX)


def has_extension(values: Iterable[Any]) -> bool:
    """True when any of ``values`` (a header's comma-separated URIs included) is A2UI's."""
    for value in values or ():
        if not isinstance(value, str):
            continue
        for item in value.split(","):
            if is_extension_uri(item):
                return True
    return False


# --- Redaction ---------------------------------------------------------------------


def _is_binding(value: Any) -> bool:
    """A data binding (``{"@path": ...}``) names where a value lives, not the value."""
    return isinstance(value, Mapping) and "@path" in value and len(value) == 1


def _scrub(value: Any, urls: bool) -> Any:
    """A copy of ``value`` with media URLs and ``data:`` URIs replaced.

    ``urls`` also replaces the value of keys that name a URL. It is set inside
    components, where those keys are image, video and audio sources.
    """
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            if urls and isinstance(key, str) and _URL_KEY.search(key) and not _is_binding(item):
                out[key] = _REDACTED
            else:
                out[key] = _scrub(item, urls)
        return out
    if isinstance(value, (list, tuple)):
        return [_scrub(item, urls) for item in value]
    if isinstance(value, str) and value[:5].lower() == "data:":
        return _REDACTED
    return value


def scrub_message(message: Mapping[str, Any]) -> Dict[str, Any]:
    """A copy of one wire-format message that is safe to record under content capture."""
    out: Dict[str, Any] = {}
    for key, item in message.items():
        if key in ("createSurface", "updateComponents") and isinstance(item, Mapping):
            out[key] = {k: _scrub(v, urls=(k == "components")) for k, v in item.items()}
        else:
            out[key] = _scrub(item, urls=False)
    return out


def scrub_payload(payload: Any) -> Any:
    """:func:`scrub_message` over one message or a list of them."""
    if isinstance(payload, list):
        return [scrub_message(m) if isinstance(m, Mapping) else m for m in payload]
    return scrub_message(payload) if isinstance(payload, Mapping) else payload


# --- Message models and parts ------------------------------------------------------


def message_to_dict(message: Any) -> Optional[Dict[str, Any]]:
    """One message (a dict, or a pydantic schema model) as a wire-format dict."""
    if isinstance(message, Mapping):
        return dict(message)
    dump = getattr(message, "model_dump", None)
    if not callable(dump):
        return None
    try:
        data = dump(mode="json", by_alias=True, exclude_unset=True)
        version = getattr(message, "version", None)
        if version == "v0.8":
            data.pop("version", None)
        elif isinstance(version, str) and "version" not in data:
            data = {"version": version, **data}
        return data if isinstance(data, dict) else None
    except Exception as exc:  # noqa: BLE001 - reading must never break the call
        logger.debug("Could not read an A2UI message model: %s", exc)
        return None


def _struct_to_dict(value: Any) -> Any:
    try:
        from google.protobuf.json_format import MessageToDict

        return MessageToDict(value)
    except Exception:  # noqa: BLE001
        return None


def read_a2a_part(part: Any) -> Optional[Any]:
    """The decoded payload of an A2A 1.x data part when it is an A2UI part, else None.

    A part is A2UI when ``metadata["mimeType"]`` or the part's own ``media_type`` is one
    of the two A2UI media types. The payload is one message or a list of messages: the
    specification says an array, the Python agent SDK writes one message per part, and
    both must be read.
    """
    mime = None
    try:
        media_type = getattr(part, "media_type", "")
        if isinstance(media_type, str) and media_type in MIME_TYPES:
            mime = media_type
        if mime is None:
            metadata = getattr(part, "metadata", None)
            if metadata is not None and "mimeType" in metadata:
                value = metadata["mimeType"]
                if isinstance(value, str) and value in MIME_TYPES:
                    mime = value
    except Exception:  # noqa: BLE001
        return None
    if mime is None:
        return None
    payload = _struct_to_dict(getattr(part, "data", None))
    return payload if isinstance(payload, (dict, list)) else None


def read_capabilities_metadata(metadata: Any) -> Optional[Mapping[str, Any]]:
    """A protobuf ``Struct`` of message metadata as a dict, only when it names capabilities."""
    try:
        if not any(key in metadata for key in CAPABILITIES_KEYS):
            return None
    except Exception:  # noqa: BLE001
        return None
    data = _struct_to_dict(metadata)
    return data if isinstance(data, Mapping) else None


# --- The accumulator ---------------------------------------------------------------


def _text(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _json(value: Any) -> str:
    try:
        return json.dumps(value, default=str, sort_keys=True, separators=(",", ":"))
    except Exception:  # noqa: BLE001
        return str(value)


class A2uiObservation:
    """What one operation put on the wire as A2UI, accumulated across messages.

    ``capture`` says whether the content lists are kept at all: with it off nothing of
    the payload is held.
    """

    def __init__(self, capture: bool = False) -> None:
        self.capture = capture
        self.versions: Set[str] = set()
        self.counts = {SIDE_AGENT: 0, SIDE_RENDERER: 0}
        self.sides_seen: Set[str] = set()
        self.types: Set[str] = set()
        self.surface_ids: Set[str] = set()
        self.catalog_ids: Set[str] = set()
        self.component_count = 0
        self.components_seen = False
        self.supported_catalog_ids: Set[str] = set()
        self.action_names: Set[str] = set()
        self.function_names: Set[str] = set()
        self.errors: List[Dict[str, Optional[str]]] = []
        self.outcome: Optional[str] = None
        self.extension_active = False
        # Content, kept only under capture.
        self.component_types: Set[str] = set()
        self.messages: List[Any] = []
        self.data_model: List[Any] = []
        self.action_contexts: List[Any] = []

    # -- feeding -------------------------------------------------------------------

    @property
    def has_data(self) -> bool:
        """True when anything A2UI was seen, so a span has something to carry."""
        return bool(
            self.sides_seen
            or self.supported_catalog_ids
            or self.extension_active
            or self.outcome is not None
        )

    def add_payload(self, payload: Any, side: str) -> None:
        """Read one message, or a list of them, from the ``side`` that sent it."""
        self.sides_seen.add(side)
        items = payload if isinstance(payload, (list, tuple)) else [payload]
        for item in items:
            message = message_to_dict(item)
            if message is not None:
                self.add_message(message, side)

    def add_messages(self, messages: Iterable[Any], side: str) -> None:
        """Read a sequence of messages (dicts or schema models) from ``side``."""
        self.add_payload(list(messages), side)

    def add_message(self, message: Mapping[str, Any], side: str) -> None:
        """Read one wire-format message from ``side``. Never raises."""
        try:
            self._add_message(message, side)
        except Exception as exc:  # noqa: BLE001 - reading must never break the call
            logger.debug("Could not read an A2UI message: %s", exc)

    def _add_message(self, message: Mapping[str, Any], side: str) -> None:
        self.sides_seen.add(side)
        self.counts[side] += 1
        version = normalise_version(message.get("version")) if "version" in message else "0.8"
        if version:
            self.versions.add(version)
        if version == "0.8":
            return  # counted and recognised, not parsed
        known = AGENT_KINDS if side == SIDE_AGENT else RENDERER_KINDS
        kind = next((k for k in message if k in known), None)
        if kind is None:
            return
        self.types.add(snake(kind))
        body = message.get(kind)
        body = body if isinstance(body, Mapping) else {}
        surface = _text(body.get("surfaceId"))
        if surface:
            self.surface_ids.add(surface)
        if self.capture:
            self.messages.append(scrub_message(message))

        if kind in ("createSurface", "updateComponents"):
            self._add_surface_message(kind, body)
        elif kind == "updateDataModel":
            if self.capture and "value" in body:
                self.data_model.append(_scrub(body.get("value"), urls=False))
        elif kind in ("callRendererFunction", "callAgentFunction"):
            call = body.get("callFunction")
            if isinstance(call, Mapping):
                name = _text(call.get("@call")) or _text(call.get("call"))
                if name:
                    self.function_names.add(name)
                catalog = _text(call.get("catalogId"))
                if catalog:
                    self.catalog_ids.add(catalog)
        elif kind == "action":
            name = _text(body.get("name"))
            if name:
                self.action_names.add(name)
            if self.capture and "context" in body:
                self.action_contexts.append(_scrub(body.get("context"), urls=False))
        elif kind == "error":
            self._add_error(body, surface)
        elif kind in ("agentFunctionResponse", "rendererFunctionResponse"):
            error = body.get("error")
            if isinstance(error, Mapping):
                self._add_error(error, surface)

    def _add_surface_message(self, kind: str, body: Mapping[str, Any]) -> None:
        catalog = _text(body.get("catalogId"))
        if catalog:
            self.catalog_ids.add(catalog)
        if kind == "updateComponents":
            self.components_seen = True
        components = body.get("components")
        if not isinstance(components, list):
            return
        self.components_seen = True
        self.component_count += len(components)
        for component in components:
            if not isinstance(component, Mapping):
                continue
            override = _text(component.get("catalogId")) or _text(component.get("@catalogId"))
            if override:
                self.catalog_ids.add(override)
            name = _text(component.get("component"))
            if self.capture and name:
                self.component_types.add(name)

    def _add_error(self, body: Mapping[str, Any], surface: Optional[str]) -> None:
        error_surface = _text(body.get("surfaceId"))
        self.errors.append(
            {
                "code": _text(body.get("code")),
                "surface": error_surface or surface,
                "path": _text(body.get("path")),
                "message": _text(body.get("message")),
            }
        )
        if error_surface:
            self.surface_ids.add(error_surface)

    def add_capabilities(self, metadata: Mapping[str, Any]) -> None:
        """Read the renderer's catalogs from message metadata (either key, any version)."""
        for key in CAPABILITIES_KEYS:
            value = metadata.get(key)
            if not isinstance(value, Mapping):
                continue
            bodies = [value] + [v for v in value.values() if isinstance(v, Mapping)]
            for body in bodies:
                ids = body.get("supportedCatalogIds")
                if isinstance(ids, list):
                    self.supported_catalog_ids.update(i for i in ids if isinstance(i, str) and i)

    def mark_extension(self, values: Iterable[Any]) -> None:
        """Note that the A2UI extension URI is among ``values`` (header values or a list)."""
        if has_extension(values):
            self.extension_active = True

    def set_outcome(self, outcome: str) -> None:
        """``valid`` or ``invalid``. Invalid is never overwritten by a later valid."""
        if outcome == VALIDATION_INVALID or self.outcome != VALIDATION_INVALID:
            self.outcome = outcome

    # -- reading out ---------------------------------------------------------------

    def attributes(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """``(always, content)`` attributes. ``content`` is empty unless capture is on."""
        always: Dict[str, Any] = {}
        content: Dict[str, Any] = {}
        A = A2UIAttributes

        # A single-valued identifier seen as several values in one operation is left
        # off: it would mislabel the span. Lists are sorted and de-duplicated.
        if len(self.versions) == 1:
            always[A.PROTOCOL_VERSION] = next(iter(self.versions))
        if SIDE_AGENT in self.sides_seen:
            always[A.AGENT_MESSAGE_COUNT] = self.counts[SIDE_AGENT]
        if SIDE_RENDERER in self.sides_seen:
            always[A.RENDERER_MESSAGE_COUNT] = self.counts[SIDE_RENDERER]
        if self.types:
            always[A.MESSAGE_TYPES] = sorted(self.types)
        if len(self.surface_ids) == 1:
            always[A.SURFACE_ID] = next(iter(self.surface_ids))
        if self.surface_ids:
            always[A.SURFACE_COUNT] = len(self.surface_ids)
        if len(self.catalog_ids) == 1:
            always[A.CATALOG_ID] = next(iter(self.catalog_ids))
        if self.catalog_ids:
            always[A.CATALOG_IDS] = sorted(self.catalog_ids)
        if self.components_seen:
            always[A.COMPONENT_COUNT] = self.component_count
        if self.supported_catalog_ids:
            always[A.RENDERER_SUPPORTED_CATALOG_IDS] = sorted(self.supported_catalog_ids)
        if len(self.action_names) == 1:
            always[A.ACTION_NAME] = next(iter(self.action_names))
        if len(self.function_names) == 1:
            always[A.FUNCTION_NAME] = next(iter(self.function_names))
        if self.errors:
            always[A.ERROR_COUNT] = len(self.errors)
            codes = sorted({e["code"] for e in self.errors if e["code"]})
            if codes:
                always[A.ERROR_CODES] = codes
        if self.outcome:
            always[A.VALIDATION_OUTCOME] = self.outcome
        if self.extension_active:
            always[A.EXTENSION_ACTIVE] = True

        if self.capture:
            if self.messages:
                content[A.MESSAGES] = _json(self.messages)
            if self.data_model:
                content[A.DATA_MODEL] = _json(self.data_model)
            if self.action_contexts:
                content[A.ACTION_CONTEXT] = _json(self.action_contexts)
            paths = [e["path"] for e in self.errors if e["path"]]
            if paths:
                content[A.ERROR_PATHS] = paths
            texts = [e["message"] for e in self.errors if e["message"]]
            if texts:
                content[A.ERROR_MESSAGES] = texts
            if self.component_types:
                content[A.COMPONENT_TYPES] = sorted(self.component_types)
        return always, content

    def events(self) -> List[Tuple[str, Dict[str, Any]]]:
        """One ``a2ui.error`` event per error, so each is visible when several share a span."""
        out: List[Tuple[str, Dict[str, Any]]] = []
        for error in self.errors:
            attrs: Dict[str, Any] = {}
            if error["code"]:
                attrs[A2UIAttributes.EVENT_ERROR_CODE] = error["code"]
            if error["surface"]:
                attrs[A2UIAttributes.SURFACE_ID] = error["surface"]
            out.append((ERROR_EVENT, attrs))
        return out


def apply_to_span(
    span: Any,
    observation: A2uiObservation,
    put_content: Callable[[Any, str, Any], None],
) -> None:
    """Write an observation's attributes and events on ``span``.

    ``put_content`` writes one content attribute, and is where capture and the length
    limit are enforced. Never raises: a span that cannot be written is not worth the call.
    """
    try:
        always, content = observation.attributes()
        for key, value in always.items():
            span.set_attribute(key, value)
        for key, value in content.items():
            put_content(span, key, value)
        for name, attrs in observation.events():
            span.add_event(name, attrs)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not write A2UI attributes: %s", exc)
