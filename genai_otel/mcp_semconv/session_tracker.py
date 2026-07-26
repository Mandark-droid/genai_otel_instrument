"""Per-session state needed to derive cross-call MCP attributes.

Three attributes cannot be computed from a single call in isolation:

``mcp.identifier.hallucinated``
    Requires knowing every identifier the session has actually seen in a prior
    response.

``commerce.cart_hash``
    Requires canonicalising cart contents so the same cart always hashes the
    same way.

``commerce.order.placement_attempt``
    Requires counting placement calls per cart hash - a second attempt on the
    same hash is the duplicate-order signal.

:class:`MCPSessionState` holds that state for one session;
:class:`MCPSessionRegistry` holds the sessions. Both are thread-safe.
"""

import hashlib
import json
import logging
import re
import threading
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

#: Argument/response keys treated as identifiers.
#:
#: Deliberately limited to id-shaped keys. Keys such as ``couponCode`` are
#: excluded by default: a user can legitimately supply a coupon code that
#: appeared in no prior response, and flagging it would manufacture false
#: hallucinations. Pass ``identifier_key_pattern`` to widen this.
DEFAULT_IDENTIFIER_KEY_PATTERN: str = r"(?:^|[_.\-])ids?$|(?:[a-z0-9])Ids?$|^ids?$"

#: Cap on how deep the identifier harvester walks a response payload.
DEFAULT_MAX_DEPTH: int = 8

#: Cap on how many identifiers one session retains, to bound memory on long
#: sessions that page through large catalogues.
DEFAULT_MAX_IDENTIFIERS: int = 20000


def _compile_identifier_pattern(pattern: Optional[str]) -> "re.Pattern":
    return re.compile(pattern or DEFAULT_IDENTIFIER_KEY_PATTERN)


def canonicalise_cart(cart: Any) -> Any:
    """Reduce a cart payload to a canonical, order-independent structure.

    Recognises the common shape - a mapping with an ``items``/``cartItems``
    list - and reduces it to a sorted list of ``(identifier, quantity)`` pairs
    so that reordering the same items does not change the hash. Anything
    unrecognised is passed through for generic canonical JSON hashing.
    """
    if isinstance(cart, Mapping):
        items = None
        for key in ("items", "cartItems", "cart_items", "lineItems", "line_items"):
            if isinstance(cart.get(key), (list, tuple)):
                items = cart[key]
                break
        if items is not None:
            reduced: List[Tuple[str, Any]] = []
            for item in items:
                if not isinstance(item, Mapping):
                    reduced.append((str(item), 1))
                    continue
                identifier = None
                for key in ("itemId", "item_id", "id", "productId", "product_id", "sku", "name"):
                    if item.get(key) is not None:
                        identifier = str(item[key])
                        break
                quantity = None
                for key in ("quantity", "qty", "count"):
                    if item.get(key) is not None:
                        quantity = item[key]
                        break
                reduced.append((identifier if identifier is not None else "", quantity))
            return sorted(reduced, key=lambda pair: (str(pair[0]), str(pair[1])))

    return cart


def compute_cart_hash(cart: Any, length: int = 16) -> Optional[str]:
    """Compute a stable hash of cart contents for drift detection.

    Returns None for an empty or unhashable cart rather than inventing a value.
    """
    if cart is None:
        return None
    canonical = canonicalise_cart(cart)
    if canonical in ({}, [], ""):
        return None
    try:
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError) as exc:
        logger.debug("Cart not JSON-serialisable, falling back to repr: %s", exc)
        encoded = repr(canonical)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]


class MCPSessionState:
    """Mutable state for one MCP agent session."""

    def __init__(
        self,
        session_id: str,
        seed_identifiers: Optional[Iterable[str]] = None,
        identifier_key_pattern: Optional[str] = None,
        max_identifiers: int = DEFAULT_MAX_IDENTIFIERS,
    ):
        """Create session state.

        Args:
            session_id: The support-correlation key.
            seed_identifiers: Identifiers the user supplied up front, which are
                legitimately absent from any prior response.
            identifier_key_pattern: Override for which keys count as identifiers.
            max_identifiers: Retention cap for observed identifiers.
        """
        self.session_id = session_id
        self._lock = threading.RLock()
        self._identifier_re = _compile_identifier_pattern(identifier_key_pattern)
        self._max_identifiers = max_identifiers
        self._observed: Set[str] = set()
        self._seed: Set[str] = {str(s) for s in (seed_identifiers or ()) if str(s)}
        self._identifiers_full = False
        self.cart_hash: Optional[str] = None
        self.previous_cart_hash: Optional[str] = None
        self._placement_attempts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Identifier tracking
    # ------------------------------------------------------------------

    def _is_identifier_key(self, key: str) -> bool:
        return bool(self._identifier_re.search(str(key)))

    def _walk(self, payload: Any, depth: int, out: List[Tuple[str, str]]) -> None:
        """Collect ``(key, value)`` pairs for identifier-shaped keys."""
        if depth > DEFAULT_MAX_DEPTH:
            return
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                if isinstance(value, (list, tuple)) and self._is_identifier_key(key):
                    # An id-shaped key holding a list of scalars is a batch of
                    # identifiers ({"orderIds": ["O-1", "O-2"]}), not a nested
                    # structure. Harvest the scalars, recurse into the rest.
                    for item in value:
                        if isinstance(item, (Mapping, list, tuple)):
                            self._walk(item, depth + 1, out)
                        elif item is not None:
                            out.append((str(key), str(item)))
                elif isinstance(value, (Mapping, list, tuple)):
                    self._walk(value, depth + 1, out)
                elif value is not None and self._is_identifier_key(key):
                    out.append((str(key), str(value)))
        elif isinstance(payload, (list, tuple)):
            for item in payload:
                self._walk(item, depth + 1, out)

    def observe_response(self, payload: Any) -> int:
        """Record every identifier present in a tool response.

        Returns:
            int: How many new identifiers were learned.
        """
        found: List[Tuple[str, str]] = []
        self._walk(payload, 0, found)
        with self._lock:
            if self._identifiers_full:
                return 0
            before = len(self._observed)
            for _, value in found:
                self._observed.add(value)
            if len(self._observed) >= self._max_identifiers:
                self._identifiers_full = True
                logger.debug(
                    "Session %s hit the identifier retention cap (%d); "
                    "hallucination detection disabled for the remainder",
                    self.session_id,
                    self._max_identifiers,
                )
            return len(self._observed) - before

    def seed(self, identifiers: Iterable[str]) -> None:
        """Register identifiers the user supplied, so they are never flagged."""
        with self._lock:
            self._seed.update(str(i) for i in identifiers if str(i))

    def check_request_identifiers(self, arguments: Any) -> Tuple[bool, Tuple[str, ...]]:
        """Check request arguments for identifiers never seen in a prior response.

        Returns:
            Tuple[bool, Tuple[str, ...]]: ``(hallucinated, offending_keys)``.
            Keys only - the offending values are never returned or recorded.

        Once the retention cap is hit the check returns ``(False, ())`` rather
        than guessing, so a truncated identifier set cannot manufacture a false
        positive.
        """
        found: List[Tuple[str, str]] = []
        self._walk(arguments, 0, found)
        if not found:
            return False, ()

        with self._lock:
            if self._identifiers_full:
                return False, ()
            offending = tuple(
                sorted(
                    {
                        key
                        for key, value in found
                        if value not in self._observed and value not in self._seed
                    }
                )
            )
        return bool(offending), offending

    @property
    def observed_identifier_count(self) -> int:
        """How many distinct identifiers this session has seen in responses."""
        with self._lock:
            return len(self._observed)

    # ------------------------------------------------------------------
    # Cart + placement tracking
    # ------------------------------------------------------------------

    def note_cart(self, cart: Any) -> Optional[str]:
        """Record the current cart and return its hash."""
        digest = compute_cart_hash(cart)
        if digest is None:
            return None
        with self._lock:
            if digest != self.cart_hash:
                self.previous_cart_hash = self.cart_hash
            self.cart_hash = digest
        return digest

    def note_placement(self, cart_hash: Optional[str] = None) -> int:
        """Record an order-placement attempt and return its 1-based ordinal.

        A returned value of 2 or more means this cart has been submitted before -
        the duplicate-order signal.
        """
        with self._lock:
            key = cart_hash if cart_hash is not None else (self.cart_hash or "")
            self._placement_attempts[key] = self._placement_attempts.get(key, 0) + 1
            return self._placement_attempts[key]

    def placement_attempts(self, cart_hash: Optional[str] = None) -> int:
        """How many placement attempts have been made against a cart hash."""
        with self._lock:
            key = cart_hash if cart_hash is not None else (self.cart_hash or "")
            return self._placement_attempts.get(key, 0)


class MCPSessionRegistry:
    """Thread-safe registry of :class:`MCPSessionState` keyed by session id."""

    def __init__(self, identifier_key_pattern: Optional[str] = None):
        self._lock = threading.RLock()
        self._sessions: Dict[str, MCPSessionState] = {}
        self._identifier_key_pattern = identifier_key_pattern

    def get(
        self, session_id: str, seed_identifiers: Optional[Sequence[str]] = None
    ) -> MCPSessionState:
        """Get or create state for a session."""
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = MCPSessionState(
                    session_id,
                    seed_identifiers=seed_identifiers,
                    identifier_key_pattern=self._identifier_key_pattern,
                )
                self._sessions[session_id] = state
            elif seed_identifiers:
                state.seed(seed_identifiers)
            return state

    def drop(self, session_id: str) -> None:
        """Discard state for a finished session."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def clear(self) -> None:
        """Discard all session state."""
        with self._lock:
            self._sessions.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)
