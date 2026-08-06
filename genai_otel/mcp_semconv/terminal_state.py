"""Deterministic terminal-state classifier for MCP agent sessions.

Every session is classified into exactly one :class:`TerminalState` by
rule-based evaluation over span attributes. There is no LLM judgement anywhere
in this module: the same span set always yields the same answer, and every
classification carries the evidence that produced it so a disputed result can be
audited line by line.

The most valuable output is the ``BLOCKED_NO_TOOL`` sub-classification. A
surface that exposes tracking and error reporting but no refund, replacement or
cancellation write API will strand a class of sessions no matter how good the
agent is. Distinguishing those from sessions the agent genuinely fumbled is the
whole point of the classifier, so the ``BLOCKED_NO_TOOL`` test asks what the
*registration surface* can do - never what the agent happened to try.

Rule precedence
---------------
Rules are evaluated in a fixed order and the first match wins. The ordering
encodes what "terminal" means: the condition the session *ended on*.

1. ``BLOCKED_BY_GUARDRAIL`` - enforcement stopped it. A good outcome, and it
   dominates whatever else looks broken downstream of the block.
2. ``BLOCKED_BY_PLATFORM`` - auth, load-shedding or a genuine outage. Checked
   before the surface-gap test because a cancellation request that hit a 503 is
   a transient platform failure, not a missing tool.
3. ``BLOCKED_NO_TOOL`` - the requested resolution has no satisfying tool. Ranked
   above completion on purpose: an order that succeeded and was then followed by
   an unresolvable cancellation request ends blocked, and the gap is the finding
   that matters. ``Classification.order_placed`` still records that the order
   went through, so nothing is lost.
4. ``COMPLETED_SUCCESS`` / ``COMPLETED_DEGRADED`` - an order or booking was
   placed; degradation signals decide which.
5. ``ABANDONED_AGENT_FAULT`` - agent behaviour caused the failure.
6. ``ABANDONED_RECOVERABLE`` - an in-protocol resolution existed and was not
   taken.

A session matching no rule falls back to ``ABANDONED_RECOVERABLE`` with
``confident=False``, so a defaulted answer is never mistaken for a determined
one.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .attributes import CommerceAttributes, GuardrailAttributes, MCPAttributes
from .tool_registry import KNOWN_CAPABILITIES, MCPToolRegistry

logger = logging.getLogger(__name__)


class TerminalState(str, Enum):
    """The seven mutually exclusive session outcomes."""

    COMPLETED_SUCCESS = "COMPLETED_SUCCESS"
    COMPLETED_DEGRADED = "COMPLETED_DEGRADED"
    ABANDONED_RECOVERABLE = "ABANDONED_RECOVERABLE"
    ABANDONED_AGENT_FAULT = "ABANDONED_AGENT_FAULT"
    BLOCKED_NO_TOOL = "BLOCKED_NO_TOOL"
    BLOCKED_BY_GUARDRAIL = "BLOCKED_BY_GUARDRAIL"
    BLOCKED_BY_PLATFORM = "BLOCKED_BY_PLATFORM"


class NoToolResolution(str, Enum):
    """The resolution a ``BLOCKED_NO_TOOL`` session was actually asking for."""

    CANCEL_ORDER = "cancel_order"
    REFUND = "refund"
    REPLACEMENT = "replacement"
    MISSING_ITEM_CLAIM = "missing_item_claim"
    MODIFY_PLACED_ORDER = "modify_placed_order"
    RESCHEDULE_BOOKING = "reschedule_booking"
    CANCEL_BOOKING = "cancel_booking"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    ADDRESS_EDIT_MID_CART = "address_edit_mid_cart"
    ORDER_HISTORY_LOOKUP = "order_history_lookup"


#: HTTP statuses that mean the platform failed, not the agent.
PLATFORM_HTTP_STATUSES: Set[int] = {401, 403, 407, 429, 500, 502, 503, 504}

#: Guardrail decision values that count as a block. ``steer`` is deliberately
#: absent: a steer is a redirect with advisory guidance, not a block.
GUARDRAIL_BLOCK_DECISIONS: Set[str] = {"block", "blocked", "deny", "denied"}

#: Stages whose mutating calls can legitimately change a cart hash. Placement
#: (stage ``Order``) is excluded on purpose - it mutates the order, not the cart.
CART_MUTATION_STAGES: Set[str] = {"Cart", "Reserve"}


@dataclass(frozen=True)
class Evidence:
    """One fact that contributed to a classification."""

    rule: str
    detail: str
    span_index: Optional[int] = None
    span_name: Optional[str] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        location = "" if self.span_index is None else " [span {0}]".format(self.span_index)
        return "{0}{1}: {2}".format(self.rule, location, self.detail)


@dataclass(frozen=True)
class Classification:
    """The outcome of classifying one session."""

    state: TerminalState
    rule: str
    evidence: Tuple[Evidence, ...] = ()
    resolution: Optional[NoToolResolution] = None
    #: False when no rule matched and the fallback was used.
    confident: bool = True
    #: True when an order or booking was successfully placed, regardless of the
    #: final state - a session can complete an order and still end blocked.
    order_placed: bool = False
    #: Every signal observed, including ones that did not decide the outcome.
    signals: Mapping[str, Any] = field(default_factory=dict)

    def explain(self) -> str:
        """Render a human-readable audit trail."""
        header = "{0} (rule={1}, confident={2})".format(self.state.value, self.rule, self.confident)
        if self.resolution is not None:
            header += " resolution={0}".format(self.resolution.value)
        lines = [header]
        lines.extend("  - {0}".format(item) for item in self.evidence)
        return "\n".join(lines)


class _SpanView:
    """Uniform read-only view over a span from any supported representation.

    Accepts OpenTelemetry ``ReadableSpan`` objects and plain mappings, so the
    classifier works equally on live spans and on replayed trace records.
    """

    __slots__ = ("index", "name", "attributes", "status_error")

    def __init__(self, index: int, source: Any):
        self.index = index
        if isinstance(source, Mapping):
            self.name = str(source.get("name") or "")
            attributes = source.get("attributes") or {}
            self.status_error = self._status_is_error(source.get("status"))
        else:
            self.name = str(getattr(source, "name", "") or "")
            attributes = getattr(source, "attributes", None) or {}
            self.status_error = self._status_is_error(getattr(source, "status", None))
        self.attributes: Dict[str, Any] = dict(attributes)

    @staticmethod
    def _status_is_error(status: Any) -> bool:
        if status is None:
            return False
        if isinstance(status, bool):
            return status
        if isinstance(status, str):
            return status.strip().upper() in ("ERROR", "STATUS_CODE_ERROR")
        code = getattr(status, "status_code", status)
        name = getattr(code, "name", None)
        if isinstance(name, str):
            return name.upper() == "ERROR"
        return str(code).upper().endswith("ERROR")

    def get(self, key: str, default: Any = None) -> Any:
        return self.attributes.get(key, default)

    @property
    def failed(self) -> bool:
        """Whether this span represents a failed call."""
        return (
            self.status_error
            or MCPAttributes.ERROR_MESSAGE_RAW in self.attributes
            or MCPAttributes.ERROR_HTTP_STATUS in self.attributes
            or MCPAttributes.ERROR_JSONRPC_CODE in self.attributes
        )

    @property
    def tool(self) -> Optional[str]:
        value = self.get(MCPAttributes.TOOL)
        return str(value) if value else None


def _normalise_spans(spans: Iterable[Any]) -> List[_SpanView]:
    return [_SpanView(index, span) for index, span in enumerate(spans)]


def _truthy(value: Any) -> bool:
    """Interpret an attribute value as a boolean, tolerating string forms."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _is_false(value: Any) -> bool:
    """Whether an attribute is explicitly false (not merely absent)."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value is False
    if isinstance(value, str):
        return value.strip().lower() in ("false", "no", "0")
    if isinstance(value, (int, float)):
        return value == 0
    return False


class TerminalStateClassifier:
    """Rule-based classifier over MCP session spans."""

    def __init__(
        self,
        registry: Optional[MCPToolRegistry] = None,
        available_capabilities: Optional[Iterable[str]] = None,
    ):
        """Create a classifier.

        Args:
            registry: Tool registry describing the registration surface. Needed
                for the ``BLOCKED_NO_TOOL`` test.
            available_capabilities: Explicit capability allowlist, for callers
                that have no registry. Takes precedence over the registry.
        """
        self.registry = registry
        self._explicit_capabilities = (
            {str(c) for c in available_capabilities} if available_capabilities is not None else None
        )

    # ------------------------------------------------------------------
    # Capability lookup
    # ------------------------------------------------------------------

    def _capability_known(self) -> bool:
        """Whether the surface's capabilities can be determined at all."""
        return self._explicit_capabilities is not None or self.registry is not None

    def has_capability(self, capability: str) -> bool:
        """Whether the registration surface can satisfy a capability."""
        if self._explicit_capabilities is not None:
            return capability in self._explicit_capabilities
        if self.registry is not None:
            return self.registry.has_capability(capability)
        return False

    def _tools_for_capability(self, capability: str) -> Set[str]:
        if self.registry is None:
            return set()
        return {meta.name for meta in self.registry.tools_for_capability(capability)}

    def missing_capabilities(self) -> Tuple[str, ...]:
        """Capabilities the surface cannot satisfy - the measured tool-surface gaps."""
        if self._explicit_capabilities is not None:
            return tuple(c for c in KNOWN_CAPABILITIES if c not in self._explicit_capabilities)
        if self.registry is not None:
            return self.registry.missing_capabilities()
        return ()

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(
        self,
        spans: Iterable[Any],
        requested_resolution: Optional[str] = None,
    ) -> Classification:
        """Classify one session.

        Args:
            spans: The session's spans, in chronological order.
            requested_resolution: The resolution the user asked for. Overrides
                any value found on the spans.

        Returns:
            Classification: State, evidence, and every signal observed.
        """
        views = _normalise_spans(spans)
        resolution_name = requested_resolution or self._find_requested_resolution(views)
        signals = self._collect_signals(views, resolution_name)

        for rule in (
            self._rule_guardrail,
            self._rule_platform,
            self._rule_no_tool,
            self._rule_completed,
            self._rule_agent_fault,
            self._rule_recoverable,
        ):
            result = rule(views, resolution_name, signals)
            if result is not None:
                return result

        return Classification(
            state=TerminalState.ABANDONED_RECOVERABLE,
            rule="fallback.no_terminal_signal",
            confident=False,
            order_placed=bool(signals.get("order_placed")),
            signals=signals,
            evidence=(
                Evidence(
                    rule="fallback.no_terminal_signal",
                    detail=(
                        "No rule matched: no guardrail block, no platform error, no requested "
                        "resolution, no order placed and no agent-fault signal. Defaulted to "
                        "ABANDONED_RECOVERABLE with confident=False."
                    ),
                ),
            ),
        )

    # ------------------------------------------------------------------
    # Signal collection
    # ------------------------------------------------------------------

    @staticmethod
    def _find_requested_resolution(views: Sequence[_SpanView]) -> Optional[str]:
        """Find the requested resolution declared on the spans.

        The classifier never infers intent from free text - it reads an
        attribute the harness set. That keeps it deterministic and LLM-free.
        """
        for view in views:
            for key in (
                CommerceAttributes.REQUESTED_RESOLUTION,
                MCPAttributes.TOOL_SELECTION_REQUESTED_CAPABILITY,
            ):
                value = view.get(key)
                if value:
                    return str(value)
        return None

    def _collect_signals(
        self, views: Sequence[_SpanView], resolution_name: Optional[str]
    ) -> Dict[str, Any]:
        """Gather every signal in the session, decisive or not."""
        placement_spans = [
            v for v in views if _truthy(v.get(CommerceAttributes.ORDER_PLACED)) and not v.failed
        ]
        distinct_carts, drift_indexes = self._detect_cart_drift(views)

        hallucinated = [v for v in views if _truthy(v.get(MCPAttributes.IDENTIFIER_HALLUCINATED))]
        wrong_tool = [v for v in views if _is_false(v.get(MCPAttributes.TOOL_SELECTION_CORRECT))]
        duplicate_placement = [
            v
            for v in views
            if isinstance(v.get(CommerceAttributes.ORDER_PLACEMENT_ATTEMPT), int)
            and v.get(CommerceAttributes.ORDER_PLACEMENT_ATTEMPT) >= 2
        ]

        failed = [v for v in views if v.failed]
        repeated_failures: Dict[str, int] = {}
        for view in failed:
            if view.tool:
                repeated_failures[view.tool] = repeated_failures.get(view.tool, 0) + 1

        return {
            "span_count": len(views),
            "order_placed": bool(placement_spans),
            "placement_span_indexes": [v.index for v in placement_spans],
            "cart_hash_sequence": distinct_carts,
            "cart_drift": bool(drift_indexes),
            "cart_drift_span_indexes": drift_indexes,
            "hallucinated_span_indexes": [v.index for v in hallucinated],
            "wrong_tool_span_indexes": [v.index for v in wrong_tool],
            "duplicate_placement_span_indexes": [v.index for v in duplicate_placement],
            "failed_span_indexes": [v.index for v in failed],
            "repeated_failing_tools": {
                tool: count for tool, count in repeated_failures.items() if count >= 2
            },
            "requested_resolution": resolution_name,
            "capability_available": (
                self.has_capability(resolution_name)
                if resolution_name and self._capability_known()
                else None
            ),
        }

    @staticmethod
    def _detect_cart_drift(
        views: Sequence[_SpanView],
    ) -> Tuple[List[Any], List[int]]:
        """Find cart-hash changes that no mutating call explains.

        A cart legitimately changes hash every time an item is added, so a bare
        "the hash changed" test would mark every multi-item order as degraded.
        Real drift is a hash that changed with *no cart mutation to explain it* -
        the cart moved underneath the agent, so the cart it ordered is not the
        cart it confirmed.

        Only a cart-stage mutation counts as an explanation. Placing an order is
        mutating too, but it mutates the order, not the cart - letting it excuse
        a cart change would hide exactly the case this detects.

        Returns:
            Tuple of the observed hash sequence and the span indexes at which
            unexplained changes occurred.
        """
        sequence: List[Any] = []
        drift_indexes: List[int] = []
        previous: Optional[Any] = None
        pending_cart_mutation = False

        for view in views:
            mutates_cart = (
                view.get(MCPAttributes.BEHAVIOUR) == "mutating"
                and view.get(MCPAttributes.STAGE) in CART_MUTATION_STAGES
            )
            digest = view.get(CommerceAttributes.CART_HASH)

            if digest is None:
                pending_cart_mutation = pending_cart_mutation or mutates_cart
                continue

            if not sequence or sequence[-1] != digest:
                sequence.append(digest)

            # The change is explained by this call itself, or by any cart
            # mutation observed since the last hash reading.
            explained = mutates_cart or pending_cart_mutation
            if previous is not None and digest != previous and not explained:
                drift_indexes.append(view.index)

            previous = digest
            pending_cart_mutation = False

        return sequence, drift_indexes

    @staticmethod
    def _last_unrecovered_failure(views: Sequence[_SpanView]) -> Optional[_SpanView]:
        """The terminal failure, if the session ended on one.

        A failure followed by a successful call is one the session recovered
        from, and must not drive the classification - only the span the session
        actually ended on counts.
        """
        if not views:
            return None
        last = views[-1]
        return last if last.failed else None

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    def _rule_guardrail(
        self,
        views: Sequence[_SpanView],
        resolution_name: Optional[str],
        signals: Dict[str, Any],
    ) -> Optional[Classification]:
        """Rule 1 - enforcement stopped the session."""
        for view in views:
            decision = view.get(GuardrailAttributes.DECISION)
            blocked = view.get(GuardrailAttributes.BLOCKED)
            decision_blocks = (
                isinstance(decision, str) and decision.strip().lower() in GUARDRAIL_BLOCK_DECISIONS
            )
            if decision_blocks or _truthy(blocked):
                attributes = {
                    k: v
                    for k, v in view.attributes.items()
                    if k
                    in (
                        GuardrailAttributes.DECISION,
                        GuardrailAttributes.BLOCKED,
                        GuardrailAttributes.RULE,
                        MCPAttributes.TOOL,
                    )
                }
                return Classification(
                    state=TerminalState.BLOCKED_BY_GUARDRAIL,
                    rule="guardrail.blocked",
                    order_placed=bool(signals.get("order_placed")),
                    signals=signals,
                    evidence=(
                        Evidence(
                            rule="guardrail.blocked",
                            detail=(
                                "Enforcement blocked the call"
                                + (
                                    " (rule={0})".format(view.get(GuardrailAttributes.RULE))
                                    if view.get(GuardrailAttributes.RULE)
                                    else ""
                                )
                            ),
                            span_index=view.index,
                            span_name=view.name,
                            attributes=attributes,
                        ),
                    ),
                )
        return None

    def _rule_platform(
        self,
        views: Sequence[_SpanView],
        resolution_name: Optional[str],
        signals: Dict[str, Any],
    ) -> Optional[Classification]:
        """Rule 2 - auth failure, load shedding or upstream outage."""
        failure = self._last_unrecovered_failure(views)
        if failure is None:
            return None
        status = failure.get(MCPAttributes.ERROR_HTTP_STATUS)
        if not isinstance(status, int) or status not in PLATFORM_HTTP_STATUSES:
            return None

        return Classification(
            state=TerminalState.BLOCKED_BY_PLATFORM,
            rule="platform.http_status",
            order_placed=bool(signals.get("order_placed")),
            signals=signals,
            evidence=(
                Evidence(
                    rule="platform.http_status",
                    detail=(
                        "Session ended on an unrecovered HTTP {0}, which is a platform "
                        "condition (auth, shed or outage), not an agent or surface fault."
                    ).format(status),
                    span_index=failure.index,
                    span_name=failure.name,
                    attributes={
                        k: v
                        for k, v in failure.attributes.items()
                        if k
                        in (
                            MCPAttributes.ERROR_HTTP_STATUS,
                            MCPAttributes.ERROR_JSONRPC_CODE,
                            MCPAttributes.ERROR_MESSAGE_RAW,
                            MCPAttributes.TOOL,
                            MCPAttributes.SERVER,
                        )
                    },
                ),
            ),
        )

    def _rule_no_tool(
        self,
        views: Sequence[_SpanView],
        resolution_name: Optional[str],
        signals: Dict[str, Any],
    ) -> Optional[Classification]:
        """Rule 3 - the requested resolution has no satisfying tool in the surface."""
        if not resolution_name:
            return None
        if not self._capability_known():
            # Without a surface description this cannot be decided; say so by
            # declining rather than guessing.
            logger.debug(
                "Requested resolution %s present but no registry or capability list "
                "supplied; BLOCKED_NO_TOOL cannot be evaluated",
                resolution_name,
            )
            return None
        if self.has_capability(resolution_name):
            return None

        try:
            resolution = NoToolResolution(resolution_name)
        except ValueError:
            resolution = None

        evidence: List[Evidence] = [
            Evidence(
                rule="no_tool.capability_absent",
                detail=(
                    "The user requested '{0}' and no tool in the {1}-tool registration "
                    "surface can satisfy it, so no agent behaviour could have succeeded."
                ).format(
                    resolution_name,
                    self.registry.candidate_count if self.registry is not None else "supplied",
                ),
                attributes={CommerceAttributes.REQUESTED_RESOLUTION: resolution_name},
            )
        ]
        # Agent-fault signals are still recorded even though they did not decide
        # the outcome - the gap is upstream of anything the agent did.
        if signals.get("hallucinated_span_indexes"):
            evidence.append(
                Evidence(
                    rule="no_tool.agent_signals_present_but_not_decisive",
                    detail=(
                        "Hallucinated identifiers were also observed on spans {0}, but the "
                        "surface gap takes precedence: no tool existed to call correctly."
                    ).format(signals["hallucinated_span_indexes"]),
                )
            )

        return Classification(
            state=TerminalState.BLOCKED_NO_TOOL,
            rule="no_tool.capability_absent",
            resolution=resolution,
            order_placed=bool(signals.get("order_placed")),
            signals=signals,
            evidence=tuple(evidence),
        )

    def _rule_completed(
        self,
        views: Sequence[_SpanView],
        resolution_name: Optional[str],
        signals: Dict[str, Any],
    ) -> Optional[Classification]:
        """Rule 4 - an order or booking was placed; check for degradation."""
        placement_indexes = signals.get("placement_span_indexes") or []
        if not placement_indexes:
            return None

        placement = views[placement_indexes[-1]]
        degradations: List[Evidence] = []

        attempt = placement.get(CommerceAttributes.ORDER_PLACEMENT_ATTEMPT)
        if isinstance(attempt, int) and attempt >= 2:
            degradations.append(
                Evidence(
                    rule="degraded.duplicate_placement",
                    detail=(
                        "Placement attempt {0} against the same cart hash - the order was "
                        "submitted more than once."
                    ).format(attempt),
                    span_index=placement.index,
                    span_name=placement.name,
                    attributes={CommerceAttributes.ORDER_PLACEMENT_ATTEMPT: attempt},
                )
            )

        quoted = placement.get(CommerceAttributes.PRICE_QUOTED)
        charged = placement.get(CommerceAttributes.PRICE_CHARGED)
        if quoted is not None and charged is not None and quoted != charged:
            degradations.append(
                Evidence(
                    rule="degraded.price_mismatch",
                    detail="Quoted {0} but charged {1}; the mismatch was absorbed.".format(
                        quoted, charged
                    ),
                    span_index=placement.index,
                    span_name=placement.name,
                    attributes={
                        CommerceAttributes.PRICE_QUOTED: quoted,
                        CommerceAttributes.PRICE_CHARGED: charged,
                    },
                )
            )

        coupon_requested = any(_truthy(v.get(CommerceAttributes.COUPON_REQUESTED)) for v in views)
        coupon_applied = any(_truthy(v.get(CommerceAttributes.COUPON_APPLIED)) for v in views)
        if coupon_requested and not coupon_applied:
            degradations.append(
                Evidence(
                    rule="degraded.coupon_missed",
                    detail="A coupon was requested but never applied to the placed order.",
                    attributes={CommerceAttributes.COUPON_REQUESTED: True},
                )
            )

        for view in views:
            if _is_false(view.get(CommerceAttributes.VARIANT_MATCH)):
                degradations.append(
                    Evidence(
                        rule="degraded.wrong_variant",
                        detail="The ordered variant does not match the requested variant.",
                        span_index=view.index,
                        span_name=view.name,
                        attributes={CommerceAttributes.VARIANT_MATCH: False},
                    )
                )
                break

        if signals.get("cart_drift"):
            degradations.append(
                Evidence(
                    rule="degraded.cart_drift",
                    detail=(
                        "Cart hash changed across the session ({0}); the cart ordered is not "
                        "the cart confirmed."
                    ).format(signals.get("cart_hash_sequence")),
                    attributes={
                        CommerceAttributes.CART_HASH: signals.get("cart_hash_sequence"),
                    },
                )
            )

        if degradations:
            return Classification(
                state=TerminalState.COMPLETED_DEGRADED,
                rule=degradations[0].rule,
                order_placed=True,
                signals=signals,
                evidence=tuple(degradations),
            )

        return Classification(
            state=TerminalState.COMPLETED_SUCCESS,
            rule="completed.clean",
            order_placed=True,
            signals=signals,
            evidence=(
                Evidence(
                    rule="completed.clean",
                    detail="Order placed with no degradation signal present.",
                    span_index=placement.index,
                    span_name=placement.name,
                    attributes={CommerceAttributes.ORDER_PLACED: True},
                ),
            ),
        )

    def _rule_agent_fault(
        self,
        views: Sequence[_SpanView],
        resolution_name: Optional[str],
        signals: Dict[str, Any],
    ) -> Optional[Classification]:
        """Rule 5 - agent behaviour caused the abandonment."""
        evidence: List[Evidence] = []

        for index in signals.get("hallucinated_span_indexes") or []:
            view = views[index]
            evidence.append(
                Evidence(
                    rule="agent_fault.hallucinated_identifier",
                    detail=(
                        "Call passed identifier key(s) {0} that appeared in no prior "
                        "response in this session."
                    ).format(list(view.get(MCPAttributes.IDENTIFIER_HALLUCINATED_KEYS) or [])),
                    span_index=index,
                    span_name=view.name,
                    attributes={
                        MCPAttributes.IDENTIFIER_HALLUCINATED: True,
                        MCPAttributes.IDENTIFIER_HALLUCINATED_KEYS: view.get(
                            MCPAttributes.IDENTIFIER_HALLUCINATED_KEYS
                        ),
                        MCPAttributes.TOOL: view.get(MCPAttributes.TOOL),
                    },
                )
            )

        for index in signals.get("wrong_tool_span_indexes") or []:
            view = views[index]
            evidence.append(
                Evidence(
                    rule="agent_fault.wrong_tool",
                    detail="Called '{0}' where ground truth was '{1}'.".format(
                        view.get(MCPAttributes.TOOL),
                        view.get(MCPAttributes.TOOL_SELECTION_EXPECTED),
                    ),
                    span_index=index,
                    span_name=view.name,
                    attributes={
                        MCPAttributes.TOOL: view.get(MCPAttributes.TOOL),
                        MCPAttributes.TOOL_SELECTION_EXPECTED: view.get(
                            MCPAttributes.TOOL_SELECTION_EXPECTED
                        ),
                        MCPAttributes.TOOL_SELECTION_CORRECT: False,
                    },
                )
            )

        if signals.get("cart_drift") and signals.get("failed_span_indexes"):
            evidence.append(
                Evidence(
                    rule="agent_fault.stale_cart",
                    detail=(
                        "Cart hash drifted ({0}) and the session ended on a failure - the "
                        "agent operated on a stale cart."
                    ).format(signals.get("cart_hash_sequence")),
                )
            )

        for tool, count in (signals.get("repeated_failing_tools") or {}).items():
            evidence.append(
                Evidence(
                    rule="agent_fault.repeated_identical_failure",
                    detail="Tool '{0}' failed {1} times with no change in approach.".format(
                        tool, count
                    ),
                    attributes={MCPAttributes.TOOL: tool},
                )
            )

        if not evidence:
            return None

        return Classification(
            state=TerminalState.ABANDONED_AGENT_FAULT,
            rule=evidence[0].rule,
            order_placed=bool(signals.get("order_placed")),
            signals=signals,
            evidence=tuple(evidence),
        )

    def _rule_recoverable(
        self,
        views: Sequence[_SpanView],
        resolution_name: Optional[str],
        signals: Dict[str, Any],
    ) -> Optional[Classification]:
        """Rule 6 - an in-protocol resolution existed and the agent did not take it."""
        # Case A: the requested capability exists in the surface but no tool
        # satisfying it was ever called.
        if resolution_name and signals.get("capability_available") is True:
            satisfying = self._tools_for_capability(resolution_name)
            called = {v.tool for v in views if v.tool}
            if satisfying and not (satisfying & called):
                return Classification(
                    state=TerminalState.ABANDONED_RECOVERABLE,
                    rule="recoverable.capability_available_unused",
                    order_placed=bool(signals.get("order_placed")),
                    signals=signals,
                    evidence=(
                        Evidence(
                            rule="recoverable.capability_available_unused",
                            detail=(
                                "'{0}' was requested and tool(s) {1} could have satisfied it, "
                                "but none were called."
                            ).format(resolution_name, sorted(satisfying)),
                            attributes={CommerceAttributes.REQUESTED_RESOLUTION: resolution_name},
                        ),
                    ),
                )

        # Case B: the session ended on a non-platform failure that was never
        # retried.
        failure = self._last_unrecovered_failure(views)
        if failure is not None:
            retried = sum(1 for v in views if v.tool and v.tool == failure.tool)
            if retried <= 1:
                return Classification(
                    state=TerminalState.ABANDONED_RECOVERABLE,
                    rule="recoverable.unretried_failure",
                    order_placed=bool(signals.get("order_placed")),
                    signals=signals,
                    evidence=(
                        Evidence(
                            rule="recoverable.unretried_failure",
                            detail=(
                                "Session ended on a recoverable failure of '{0}' that was "
                                "never retried or worked around."
                            ).format(failure.tool),
                            span_index=failure.index,
                            span_name=failure.name,
                            attributes={
                                MCPAttributes.TOOL: failure.tool,
                                MCPAttributes.ERROR_MESSAGE_RAW: failure.get(
                                    MCPAttributes.ERROR_MESSAGE_RAW
                                ),
                                MCPAttributes.ERROR_JSONRPC_CODE: failure.get(
                                    MCPAttributes.ERROR_JSONRPC_CODE
                                ),
                            },
                        ),
                    ),
                )

        return None


def classify_session(
    spans: Iterable[Any],
    registry: Optional[MCPToolRegistry] = None,
    requested_resolution: Optional[str] = None,
    available_capabilities: Optional[Iterable[str]] = None,
) -> Classification:
    """Classify one session into exactly one terminal state.

    Convenience wrapper around :class:`TerminalStateClassifier`.
    """
    classifier = TerminalStateClassifier(
        registry=registry, available_capabilities=available_capabilities
    )
    return classifier.classify(spans, requested_resolution=requested_resolution)
