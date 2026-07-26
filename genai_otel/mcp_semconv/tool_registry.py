"""Tool-metadata registry loaded from MCP tool schema maps.

Tool metadata - journey stage, behaviour, idempotency, capabilities - is looked
up from a supplied schema map rather than hardcoded, so a different tool surface
only needs different JSON.

Schema shape (as produced by the Swiggy schema sync)::

    {
      "server": "food",
      "tool_count": 14,
      "tools": [
        {
          "name": "apply_food_coupon",
          "server": "food",
          "details": {
            "**Name**": "apply_food_coupon",
            "**MCP Server**": "[Food](/docs/reference/food.md)",
            "**Endpoint**": "POST mcp.swiggy.com/food",
            "**Stage**": "Cart",
            "**Behaviour**": "mutating"
          }
        }
      ]
    }

Normalised keys (``stage``, ``behaviour``, ``idempotent``, ``placement``,
``capabilities``) are also accepted at the top level of a tool entry and take
precedence over the ``details`` block, so a schema map can be enriched without
changing its generator.

Composite-proxy prefixes
------------------------
When agents reach several servers through one FastMCP composite proxy, tool
names arrive prefixed - ``food_search_restaurants`` rather than
``search_restaurants``. :meth:`MCPToolRegistry.resolve` strips the prefix into
``mcp.server`` and keeps the raw name. This matters more than it looks: if the
prefix is left on, every ground-truth comparison against the real tool name
fails and tool-selection accuracy silently reads 0%.

Prefix detection is driven by *known* server and tool names, never by splitting
on the first separator - ``search_restaurants_dineout`` is a real unprefixed
tool name, and a naive split would mangle it.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

from .attributes import BEHAVIOUR_MUTATING, BEHAVIOUR_READ_ONLY

logger = logging.getLogger(__name__)

#: Separators a composite proxy may use between a server prefix and a tool name.
DEFAULT_PREFIX_SEPARATORS: Tuple[str, ...] = ("_", "-", ".", "/", ":")

#: Stage that, combined with a mutating behaviour, marks an order-placement tool
#: by default. Surfaces whose placement tool sits in another stage (dineout's
#: ``book_table`` is stage ``Reserve``) should pass ``placement_tools``.
DEFAULT_PLACEMENT_STAGES: FrozenSet[str] = frozenset({"Order"})

#: Default capability -> name-pattern table used by
#: :meth:`MCPToolRegistry.has_capability` when a schema map does not declare
#: capabilities explicitly. Patterns are matched case-insensitively against the
#: unprefixed tool name.
#:
#: Deliberately strict. ``report_error`` files a support report; it does not
#: cancel, refund or replace anything, and it is not an in-protocol human
#: escalation - so it maps to no capability here. Likewise
#: ``address_edit_mid_cart`` requires edit/update semantics: ``create_address``
#: and ``delete_address`` do not satisfy it.
DEFAULT_CAPABILITY_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "cancel_order": (r"cancel.*order", r"order.*cancel", r"^cancel$"),
    "refund": (r"refund",),
    "replacement": (r"replace", r"replacement"),
    "missing_item_claim": (r"missing.*item", r"item.*missing", r"claim"),
    "modify_placed_order": (r"modify.*order", r"amend.*order", r"edit.*order", r"update.*order"),
    "reschedule_booking": (r"reschedule", r"change.*slot", r"modify.*booking"),
    "cancel_booking": (r"cancel.*booking", r"booking.*cancel", r"cancel.*table"),
    "escalate_to_human": (r"escalate", r"human.*agent", r"agent.*handoff", r"live.*chat"),
    "address_edit_mid_cart": (r"(update|edit|change|modify).*address",),
    "order_history_lookup": (r"get.*orders?$", r"order.*history", r"list.*orders?"),
}

#: The resolutions a blocked session can be sub-classified by.
KNOWN_CAPABILITIES: Tuple[str, ...] = tuple(DEFAULT_CAPABILITY_PATTERNS.keys())

_DETAIL_KEY_RE = re.compile(r"[^a-z0-9]+")


def _normalise_detail_key(key: str) -> str:
    """Normalise a ``details`` key such as ``**MCP Server**`` to ``mcp server``."""
    return _DETAIL_KEY_RE.sub(" ", str(key).strip().lower()).strip()


def _coerce_bool(value: Any) -> Optional[bool]:
    """Coerce a schema value to a bool, returning None when it is not boolean-ish."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1"):
            return True
        if lowered in ("false", "no", "0"):
            return False
    return None


@dataclass(frozen=True)
class ToolMetadata:
    """Static metadata for one tool, as declared by a schema map."""

    name: str
    server: str
    stage: Optional[str] = None
    behaviour: Optional[str] = None
    idempotent: Optional[bool] = None
    placement: bool = False
    capabilities: FrozenSet[str] = field(default_factory=frozenset)
    endpoint: Optional[str] = None


@dataclass(frozen=True)
class ToolResolution:
    """Outcome of resolving a wire tool name against the registry.

    ``tool`` is always the name to compare against ground truth; ``raw_name`` is
    always what arrived on the wire.
    """

    raw_name: str
    tool: str
    server: Optional[str] = None
    stage: Optional[str] = None
    behaviour: Optional[str] = None
    idempotent: Optional[bool] = None
    placement: bool = False
    capabilities: FrozenSet[str] = field(default_factory=frozenset)
    prefix_stripped: bool = False
    ambiguous: bool = False
    known: bool = False

    @property
    def metadata_resolved(self) -> bool:
        """True when the schema map supplied stage and behaviour for this tool."""
        return self.stage is not None and self.behaviour is not None


class MCPToolRegistry:
    """Registry of tool metadata across one or more MCP servers."""

    def __init__(
        self,
        tools: Iterable[ToolMetadata],
        prefix_separators: Sequence[str] = DEFAULT_PREFIX_SEPARATORS,
        capability_patterns: Optional[Mapping[str, Sequence[str]]] = None,
    ):
        """Build a registry.

        Args:
            tools: Tool metadata entries.
            prefix_separators: Separators a composite proxy may place between a
                server prefix and a tool name.
            capability_patterns: Override for :data:`DEFAULT_CAPABILITY_PATTERNS`.
        """
        self._by_server: Dict[str, Dict[str, ToolMetadata]] = {}
        self._by_tool: Dict[str, List[ToolMetadata]] = {}
        self._prefix_separators = tuple(prefix_separators)
        self._capability_patterns = dict(
            capability_patterns if capability_patterns is not None else DEFAULT_CAPABILITY_PATTERNS
        )

        for meta in tools:
            self._by_server.setdefault(meta.server, {})[meta.name] = meta
            self._by_tool.setdefault(meta.name, []).append(meta)

        self._capability_cache: Dict[str, Tuple[ToolMetadata, ...]] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dicts(cls, documents: Iterable[Mapping[str, Any]], **kwargs: Any) -> "MCPToolRegistry":
        """Build a registry from already-parsed schema documents."""
        tools: List[ToolMetadata] = []
        for document in documents:
            tools.extend(cls._parse_document(document))
        return cls(tools, **kwargs)

    @classmethod
    def from_files(cls, paths: Iterable[str], **kwargs: Any) -> "MCPToolRegistry":
        """Build a registry from a list of JSON schema files."""
        documents = []
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    documents.append(json.load(handle))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Failed to load MCP tool schema %s: %s", path, exc)
        return cls.from_dicts(documents, **kwargs)

    @classmethod
    def from_directory(cls, directory: str, **kwargs: Any) -> "MCPToolRegistry":
        """Build a registry from every ``*.json`` file in a directory."""
        try:
            names = sorted(n for n in os.listdir(directory) if n.endswith(".json"))
        except OSError as exc:
            logger.warning("Failed to list MCP tool schema directory %s: %s", directory, exc)
            return cls([], **kwargs)
        return cls.from_files([os.path.join(directory, name) for name in names], **kwargs)

    @staticmethod
    def _parse_document(document: Mapping[str, Any]) -> List[ToolMetadata]:
        """Parse one schema document into tool metadata entries."""
        default_server = str(document.get("server") or "").strip()
        entries: List[ToolMetadata] = []

        for raw_tool in document.get("tools") or []:
            if not isinstance(raw_tool, Mapping):
                continue
            name = str(raw_tool.get("name") or "").strip()
            if not name:
                continue

            details = {
                _normalise_detail_key(k): v
                for k, v in (raw_tool.get("details") or {}).items()
                if isinstance(k, str)
            }

            server = str(raw_tool.get("server") or default_server or "").strip()
            stage = raw_tool.get("stage") or details.get("stage")
            behaviour = (
                raw_tool.get("behaviour") or raw_tool.get("behavior") or details.get("behaviour")
            )
            endpoint = raw_tool.get("endpoint") or details.get("endpoint")

            stage = str(stage).strip() if stage else None
            behaviour = str(behaviour).strip().lower() if behaviour else None

            # Idempotency is derived unless the schema states it. The default
            # rule is conservative: read-only calls are safe to repeat,
            # mutating ones are assumed not to be. That is the safe direction -
            # it never marks an order-placement tool as repeatable.
            idempotent = _coerce_bool(raw_tool.get("idempotent"))
            if idempotent is None:
                idempotent = _coerce_bool(details.get("idempotent"))
            if idempotent is None and behaviour:
                idempotent = behaviour == BEHAVIOUR_READ_ONLY

            declared_placement = _coerce_bool(raw_tool.get("placement"))
            if declared_placement is None:
                placement = bool(
                    stage in DEFAULT_PLACEMENT_STAGES and behaviour == BEHAVIOUR_MUTATING
                )
            else:
                placement = declared_placement

            declared_caps = raw_tool.get("capabilities") or []
            capabilities = frozenset(str(c).strip() for c in declared_caps if str(c).strip())

            entries.append(
                ToolMetadata(
                    name=name,
                    server=server,
                    stage=stage,
                    behaviour=behaviour,
                    idempotent=idempotent,
                    placement=placement,
                    capabilities=capabilities,
                    endpoint=str(endpoint).strip() if endpoint else None,
                )
            )

        return entries

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    @property
    def servers(self) -> Tuple[str, ...]:
        """Known server names."""
        return tuple(sorted(self._by_server))

    @property
    def candidate_count(self) -> int:
        """Total number of tools in the registration surface."""
        return sum(len(tools) for tools in self._by_server.values())

    def get(self, tool: str, server: Optional[str] = None) -> Optional[ToolMetadata]:
        """Look up one tool by unprefixed name, optionally scoped to a server."""
        if server is not None:
            return self._by_server.get(server, {}).get(tool)
        matches = self._by_tool.get(tool) or []
        return matches[0] if len(matches) == 1 else None

    def mark_placement(self, tool_names: Iterable[str]) -> None:
        """Mark named tools as order-placement tools.

        Needed for surfaces whose placement tool is not stage ``Order`` - for
        example dineout's ``book_table``, which is stage ``Reserve``.
        """
        wanted = set(tool_names)
        for server, tools in self._by_server.items():
            for name, meta in list(tools.items()):
                if name in wanted and not meta.placement:
                    updated = ToolMetadata(
                        name=meta.name,
                        server=meta.server,
                        stage=meta.stage,
                        behaviour=meta.behaviour,
                        idempotent=meta.idempotent,
                        placement=True,
                        capabilities=meta.capabilities,
                        endpoint=meta.endpoint,
                    )
                    self._by_server[server][name] = updated
                    self._by_tool[name] = [
                        updated if m is meta else m for m in self._by_tool.get(name, [])
                    ]

    def resolve(self, raw_name: str, server_hint: Optional[str] = None) -> ToolResolution:
        """Resolve a wire tool name into server + unprefixed tool + metadata.

        Args:
            raw_name: Tool name exactly as it arrived (possibly proxy-prefixed).
            server_hint: Server known from the transport, which wins over any
                prefix inference.

        Returns:
            ToolResolution: Never None - an unknown tool resolves to itself with
            ``known=False`` so the span still carries the raw name.
        """
        raw_name = str(raw_name or "").strip()
        if not raw_name:
            return ToolResolution(raw_name="", tool="")

        # An explicit server hint plus a matching tool is the strongest signal.
        if server_hint and raw_name in self._by_server.get(server_hint, {}):
            return self._resolution(raw_name, self._by_server[server_hint][raw_name], False)

        # Exact, unambiguous match on the raw name: not prefixed.
        exact = self._by_tool.get(raw_name) or []
        if len(exact) == 1:
            return self._resolution(raw_name, exact[0], False)
        if len(exact) > 1:
            # Same tool name on several servers (get_addresses, report_error).
            if server_hint and server_hint in self._by_server:
                scoped = self._by_server[server_hint].get(raw_name)
                if scoped is not None:
                    return self._resolution(raw_name, scoped, False)
            return self._ambiguous_resolution(raw_name, raw_name, exact, prefix_stripped=False)

        # No exact match - try stripping a known server prefix.
        for server in sorted(self._by_server, key=len, reverse=True):
            for separator in self._prefix_separators:
                marker = server + separator
                if not raw_name.startswith(marker):
                    continue
                remainder = raw_name[len(marker) :]
                if not remainder:
                    continue
                meta = self._by_server[server].get(remainder)
                if meta is not None:
                    return self._resolution(raw_name, meta, True)
                # Prefix matched a known server but the remainder is not a known
                # tool. Still strip it: the server attribution is real, and
                # keeping the prefix on mcp.tool would break ground truth.
                return ToolResolution(
                    raw_name=raw_name,
                    tool=remainder,
                    server=server,
                    prefix_stripped=True,
                    known=False,
                )

        return ToolResolution(raw_name=raw_name, tool=raw_name, server=server_hint, known=False)

    @staticmethod
    def _resolution(raw_name: str, meta: ToolMetadata, prefix_stripped: bool) -> "ToolResolution":
        """Build a resolution from a single unambiguous metadata match."""
        return ToolResolution(
            raw_name=raw_name,
            tool=meta.name,
            server=meta.server or None,
            stage=meta.stage,
            behaviour=meta.behaviour,
            idempotent=meta.idempotent,
            placement=meta.placement,
            capabilities=meta.capabilities,
            prefix_stripped=prefix_stripped,
            known=True,
        )

    @staticmethod
    def _ambiguous_resolution(
        raw_name: str,
        tool: str,
        candidates: Sequence[ToolMetadata],
        prefix_stripped: bool,
    ) -> "ToolResolution":
        """Build a resolution for a tool name present on several servers.

        The server stays unset, but any metadata field on which every candidate
        agrees is still reported - ``get_addresses`` is Discover/read-only on
        both food and instamart, so those are safe to emit.
        """

        def agreed(attr: str) -> Any:
            values = {getattr(c, attr) for c in candidates}
            return values.pop() if len(values) == 1 else None

        shared_caps = frozenset.intersection(*[c.capabilities for c in candidates])
        return ToolResolution(
            raw_name=raw_name,
            tool=tool,
            server=None,
            stage=agreed("stage"),
            behaviour=agreed("behaviour"),
            idempotent=agreed("idempotent"),
            placement=bool(agreed("placement")),
            capabilities=shared_caps,
            prefix_stripped=prefix_stripped,
            ambiguous=True,
            known=True,
        )

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def tools_for_capability(self, capability: str) -> Tuple[ToolMetadata, ...]:
        """Return every tool in the surface that satisfies a capability.

        A tool satisfies a capability when the schema map declares it, or when
        its name matches the capability's pattern set.
        """
        if capability in self._capability_cache:
            return self._capability_cache[capability]

        patterns = [
            re.compile(p, re.IGNORECASE) for p in self._capability_patterns.get(capability, ())
        ]
        matches: List[ToolMetadata] = []
        for tools in self._by_server.values():
            for meta in tools.values():
                if capability in meta.capabilities:
                    matches.append(meta)
                    continue
                if any(pattern.search(meta.name) for pattern in patterns):
                    matches.append(meta)

        result = tuple(matches)
        self._capability_cache[capability] = result
        return result

    def has_capability(self, capability: str) -> bool:
        """Whether any tool in the registration surface can satisfy a capability.

        This is the BLOCKED_NO_TOOL test. It asks what the *surface* can do, not
        what the agent happened to call - which is exactly what separates "the
        agent failed" from "no tool could have succeeded".
        """
        return bool(self.tools_for_capability(capability))

    def missing_capabilities(
        self, capabilities: Iterable[str] = KNOWN_CAPABILITIES
    ) -> Tuple[str, ...]:
        """Capabilities with no satisfying tool - the measured gaps in the surface."""
        return tuple(c for c in capabilities if not self.has_capability(c))
