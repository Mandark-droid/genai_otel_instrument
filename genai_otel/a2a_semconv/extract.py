"""Read A2A protocol objects into span attributes.

Everything here is duck-typed over the SDK's protobuf messages and never imports
``a2a``, so it can be tested without the SDK and cannot be the reason an import fails.
Each reader tolerates a missing or odd field and returns what it could read.

What is recorded follows one rule. Ids, task states, counts of parts by kind, the
artifact count and the Agent Card's name, description and version are always recorded:
they describe the call, not what was said. Message and artifact text, data parts and
skill names are recorded only when content capture is on. File parts are counted and
their media type recorded; their bytes, URL and file name never are.
"""

import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

from .attributes import BINDING_GRPC, BINDING_HTTP_JSON, BINDING_JSONRPC, A2AAttributes

logger = logging.getLogger(__name__)

_TASK_STATES = {
    1: "submitted",
    2: "working",
    3: "completed",
    4: "failed",
    5: "canceled",
    6: "input_required",
    7: "rejected",
    8: "auth_required",
}

#: Method names on the wire (the JSON-RPC method, the gRPC RPC name).
CLIENT_METHODS = {
    "send_message": "SendMessage",
    "get_task": "GetTask",
    "list_tasks": "ListTasks",
    "cancel_task": "CancelTask",
    "create_task_push_notification_config": "CreateTaskPushNotificationConfig",
    "get_task_push_notification_config": "GetTaskPushNotificationConfig",
    "list_task_push_notification_configs": "ListTaskPushNotificationConfigs",
    "delete_task_push_notification_config": "DeleteTaskPushNotificationConfig",
    "subscribe": "SubscribeToTask",
    "get_extended_agent_card": "GetExtendedAgentCard",
}
STREAMING_SEND_METHOD = "SendStreamingMessage"

SERVER_METHODS = {
    "on_message_send": "SendMessage",
    "on_message_send_stream": STREAMING_SEND_METHOD,
    "on_get_task": "GetTask",
    "on_list_tasks": "ListTasks",
    "on_cancel_task": "CancelTask",
    "on_create_task_push_notification_config": "CreateTaskPushNotificationConfig",
    "on_get_task_push_notification_config": "GetTaskPushNotificationConfig",
    "on_list_task_push_notification_configs": "ListTaskPushNotificationConfigs",
    "on_delete_task_push_notification_config": "DeleteTaskPushNotificationConfig",
    "on_subscribe_to_task": "SubscribeToTask",
    "on_get_extended_agent_card": "GetExtendedAgentCard",
}

#: Methods whose result is a stream of events.
STREAMING_CLIENT_METHODS = frozenset({"send_message", "subscribe"})
STREAMING_SERVER_METHODS = frozenset({"on_message_send_stream", "on_subscribe_to_task"})


def state_name(state: Any) -> Optional[str]:
    """``TASK_STATE_INPUT_REQUIRED`` (or its number) as ``input_required``."""
    if isinstance(state, bool):
        return None
    if isinstance(state, int):
        return _TASK_STATES.get(state)
    if isinstance(state, str):
        name = state.upper()
        if name.startswith("TASK_STATE_"):
            name = name[len("TASK_STATE_") :]
        if name in ("", "UNSPECIFIED"):
            return None
        return name.lower()
    try:
        return _TASK_STATES.get(int(state))
    except (TypeError, ValueError):
        return None


def binding_from_transport(transport: Any) -> Optional[str]:
    """Which binding a client transport speaks, from its class name."""
    if transport is None:
        return None
    name = type(transport).__name__.lower()
    if "grpc" in name:
        return BINDING_GRPC
    if "jsonrpc" in name:
        return BINDING_JSONRPC
    if "rest" in name or "httpjson" in name:
        return BINDING_HTTP_JSON
    # A tenant decorator wraps the real transport.
    for attr in ("_base", "_delegate", "_transport"):
        inner = getattr(transport, attr, None)
        if inner is not None and inner is not transport:
            return binding_from_transport(inner)
    return None


def binding_from_card_value(value: Any) -> Optional[str]:
    """An Agent Card ``protocol_binding`` string (``JSONRPC``, ``GRPC``, ``HTTP+JSON``)."""
    text = str(value or "").upper().replace("-", "").replace("_", "").replace("+", "")
    if text == "JSONRPC":
        return BINDING_JSONRPC
    if text == "GRPC":
        return BINDING_GRPC
    if text in ("HTTPJSON", "REST"):
        return BINDING_HTTP_JSON
    return None


def binding_from_server_state(state: Any) -> Optional[str]:
    """Which binding delivered a call, from the SDK's ``ServerCallContext.state``."""
    try:
        if "grpc_context" in state:
            return BINDING_GRPC
        if "method" in state:
            return BINDING_JSONRPC
        if "headers" in state:
            return BINDING_HTTP_JSON
    except TypeError:
        pass
    return None


def pick_interface(card: Any, binding: Optional[str]) -> Any:
    """The Agent Card interface for ``binding``, else the first one."""
    try:
        interfaces = list(card.supported_interfaces)
    except Exception:  # noqa: BLE001
        return None
    if not interfaces:
        return None
    if binding:
        for interface in interfaces:
            if binding_from_card_value(getattr(interface, "protocol_binding", "")) == binding:
                return interface
    return interfaces[0]


def card_attributes(card: Any, capture: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(always, content)`` attributes for an Agent Card.

    ``content`` holds the skill names and is for the caller to write only when content
    capture is on; it is empty unless ``capture`` is true.
    """
    always: Dict[str, Any] = {}
    content: Dict[str, Any] = {}
    if card is None:
        return always, content
    try:
        for key, field in (
            (A2AAttributes.AGENT_NAME, "name"),
            (A2AAttributes.AGENT_DESCRIPTION, "description"),
            (A2AAttributes.AGENT_VERSION, "version"),
        ):
            value = getattr(card, field, "")
            if isinstance(value, str) and value:
                always[key] = value
        skills = list(getattr(card, "skills", []) or [])
        always[A2AAttributes.AGENT_SKILL_COUNT] = len(skills)
        if capture:
            names = [s.name for s in skills if isinstance(getattr(s, "name", None), str) and s.name]
            if names:
                content[A2AAttributes.AGENT_SKILL_NAMES] = names
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not read the Agent Card: %s", exc)
    return always, content


def server_address(url: Any) -> Dict[str, Any]:
    """``server.address`` / ``server.port`` from an interface URL."""
    attrs: Dict[str, Any] = {}
    if not isinstance(url, str) or not url:
        return attrs
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        if not host and "://" not in url:
            parsed = urlparse("//" + url)
            host = parsed.hostname
        if host:
            attrs[A2AAttributes.SERVER_ADDRESS] = host
            port = parsed.port
            if port is None:
                port = {"https": 443, "http": 80}.get(parsed.scheme or "")
            if port is not None:
                attrs[A2AAttributes.SERVER_PORT] = port
    except ValueError:
        pass
    return attrs


_HTTP_STATUS = re.compile(r"HTTP Error (\d{3})")
_GRPC_STATUS = re.compile(r"gRPC Error ([A-Z_]+)")


def error_type(exc: BaseException) -> str:
    """``error.type`` for a failed call.

    The A2A error name where there is one (``TaskNotFoundError``). The SDK's generic
    client error wraps HTTP and gRPC failures in its message, so for that one the
    status code is the more useful value.
    """
    name = type(exc).__name__
    if name == "A2AClientError":
        text = str(exc)
        match = _HTTP_STATUS.search(text) or _GRPC_STATUS.search(text)
        if match:
            return match.group(1)
    return name


# --- Reading messages --------------------------------------------------------------


def _part_kind(part: Any) -> Optional[str]:
    try:
        which = part.WhichOneof("content")
    except Exception:  # noqa: BLE001
        which = None
    if which is None:
        for candidate in ("text", "raw", "url", "data"):
            if _has(part, candidate):
                which = candidate
                break
    if which == "text":
        return "text"
    if which in ("raw", "url"):
        return "file"
    if which == "data":
        return "data"
    return None


def _has(obj: Any, field: str) -> bool:
    try:
        return bool(obj.HasField(field))
    except Exception:  # noqa: BLE001
        return False


def _data_json(value: Any) -> str:
    try:
        from google.protobuf.json_format import MessageToDict

        return json.dumps(MessageToDict(value), default=str, sort_keys=True)
    except Exception:  # noqa: BLE001
        return str(value)


class Observation:
    """What a call put on the wire, or got back, accumulated across stream events."""

    def __init__(self, capture: bool = False) -> None:
        self.capture = capture
        self.task_ids: Set[str] = set()
        self.context_ids: Set[str] = set()
        self.states: Set[str] = set()
        self.artifact_ids: Set[str] = set()
        self.message_id: Optional[str] = None
        self.reference_task_ids: List[str] = []
        self.tenant: Optional[str] = None
        self.part_counts = {"text": 0, "file": 0, "data": 0}
        self.file_media_types: Set[str] = set()
        self.texts: List[str] = []
        self.datas: List[str] = []
        self.has_parts = False

    # -- parts ---------------------------------------------------------------------
    def _add_parts(self, parts: Iterable[Any]) -> None:
        for part in parts:
            kind = _part_kind(part)
            if kind is None:
                continue
            self.has_parts = True
            self.part_counts[kind] += 1
            if kind == "file":
                # Counted and typed. The bytes, the URL and the file name are never read.
                media_type = getattr(part, "media_type", "")
                if isinstance(media_type, str) and media_type:
                    self.file_media_types.add(media_type)
            elif not self.capture:
                continue
            elif kind == "text":
                self.texts.append(part.text)
            else:
                self.datas.append(_data_json(part.data))

    def _add_artifact(self, artifact: Any) -> None:
        artifact_id = getattr(artifact, "artifact_id", "")
        self.artifact_ids.add(artifact_id or f"#{len(self.artifact_ids)}")
        self._add_parts(getattr(artifact, "parts", []) or [])

    def _add_ids(self, task_id: Any = None, context_id: Any = None) -> None:
        if isinstance(task_id, str) and task_id:
            self.task_ids.add(task_id)
        if isinstance(context_id, str) and context_id:
            self.context_ids.add(context_id)

    def _add_status(self, status: Any) -> None:
        if status is None:
            return
        name = state_name(getattr(status, "state", None))
        if name:
            self.states.add(name)
        if _has(status, "message"):
            self._add_parts(getattr(status.message, "parts", []) or [])

    # -- what was sent -------------------------------------------------------------
    def observe_request(self, request: Any) -> None:
        """Read a request object (the first argument of a client or handler call)."""
        try:
            kind = type(request).__name__
            tenant = getattr(request, "tenant", "")
            if isinstance(tenant, str) and tenant:
                self.tenant = tenant
            if kind == "SendMessageRequest":
                message = request.message
                self._add_ids(message.task_id, message.context_id)
                self.message_id = message.message_id or None
                self.reference_task_ids = list(message.reference_task_ids)
                self._add_parts(message.parts)
            elif kind in ("GetTaskRequest", "CancelTaskRequest", "SubscribeToTaskRequest"):
                self._add_ids(request.id)
            elif kind == "ListTasksRequest":
                self._add_ids(context_id=request.context_id)
            elif kind in (
                "TaskPushNotificationConfig",
                "GetTaskPushNotificationConfigRequest",
                "ListTaskPushNotificationConfigsRequest",
                "DeleteTaskPushNotificationConfigRequest",
            ):
                self._add_ids(request.task_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not read an A2A request: %s", exc)

    # -- what came back ------------------------------------------------------------
    def observe_result(self, result: Any) -> None:
        """Read a response, a stream event, a Task or a Message."""
        try:
            self._observe_result(result)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not read an A2A result: %s", exc)

    def _observe_result(self, result: Any) -> None:
        if result is None:
            return
        kind = type(result).__name__
        if kind in ("StreamResponse", "SendMessageResponse"):
            for field in ("task", "message", "status_update", "artifact_update"):
                if _has(result, field):
                    self._observe_result(getattr(result, field))
        elif kind == "Task":
            self._add_ids(result.id, result.context_id)
            self._add_status(result.status)
            for artifact in result.artifacts:
                self._add_artifact(artifact)
        elif kind == "Message":
            self._add_ids(result.task_id, result.context_id)
            self._add_parts(result.parts)
        elif kind == "TaskStatusUpdateEvent":
            self._add_ids(result.task_id, result.context_id)
            self._add_status(result.status)
        elif kind == "TaskArtifactUpdateEvent":
            self._add_ids(result.task_id, result.context_id)
            self._add_artifact(result.artifact)


def _single(values: Set[str]) -> Optional[str]:
    """The one value, or None when there are none or several."""
    return next(iter(values)) if len(values) == 1 else None


def build_attributes(
    request: Observation, response: Observation
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(always, content)`` attributes from what a call sent and received."""
    always: Dict[str, Any] = {}
    content: Dict[str, Any] = {}

    def put(key: str, value: Any) -> None:
        if value not in (None, ""):
            always[key] = value

    # A value seen as several distinct ones across the operation is left off: a task
    # id or state that changes mid-stream would mislabel the span.
    put(A2AAttributes.TASK_ID, _single(request.task_ids | response.task_ids))
    put(A2AAttributes.CONVERSATION_ID, _single(request.context_ids | response.context_ids))
    put(A2AAttributes.TASK_STATE, _single(response.states))
    put(A2AAttributes.TENANT, request.tenant)
    put(A2AAttributes.MESSAGE_ID, request.message_id)
    if request.reference_task_ids:
        always[A2AAttributes.MESSAGE_REFERENCE_TASK_IDS] = list(request.reference_task_ids)

    if request.has_parts:
        always[A2AAttributes.MESSAGE_TEXT_PART_COUNT] = request.part_counts["text"]
        always[A2AAttributes.MESSAGE_FILE_PART_COUNT] = request.part_counts["file"]
        always[A2AAttributes.MESSAGE_DATA_PART_COUNT] = request.part_counts["data"]
        if request.file_media_types:
            always[A2AAttributes.MESSAGE_FILE_MEDIA_TYPES] = sorted(request.file_media_types)
    if response.has_parts:
        always[A2AAttributes.RESPONSE_TEXT_PART_COUNT] = response.part_counts["text"]
        always[A2AAttributes.RESPONSE_FILE_PART_COUNT] = response.part_counts["file"]
        always[A2AAttributes.RESPONSE_DATA_PART_COUNT] = response.part_counts["data"]
        if response.file_media_types:
            always[A2AAttributes.RESPONSE_FILE_MEDIA_TYPES] = sorted(response.file_media_types)
    if response.artifact_ids:
        always[A2AAttributes.ARTIFACT_COUNT] = len(response.artifact_ids)

    if request.texts:
        content[A2AAttributes.MESSAGE_TEXT] = "\n".join(request.texts)
    if request.datas:
        content[A2AAttributes.MESSAGE_DATA] = "\n".join(request.datas)
    if response.texts:
        content[A2AAttributes.RESPONSE_TEXT] = "\n".join(response.texts)
    if response.datas:
        content[A2AAttributes.RESPONSE_DATA] = "\n".join(response.datas)
    return always, content
