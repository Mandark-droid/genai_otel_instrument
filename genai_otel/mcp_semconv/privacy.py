"""Privacy helpers for MCP instrumentation.

Two rules are enforced here, both non-negotiable and both mirroring the
published compliance guidance for the tool surfaces this instrumentation
targets:

1. **User identifiers are hashed at rest.** A phone number, email or customer id
   never reaches a span in the clear.
2. **Session ids are logged, request/response bodies are not.** The session id
   is the support-correlation key and is sufficient for it; full payloads are
   not recorded by this module.

Salting
-------
An unsalted SHA-256 of a phone number is trivially reversible - the keyspace is
small enough to enumerate. So a salt is always applied. Set
``GENAI_MCP_HASH_SALT`` to a secret that is stable across processes if hashes
must correlate across services or runs. If it is unset, a random per-process
salt is generated and a warning is emitted once: hashes stay consistent within
the process (so a session still joins up) but deliberately do not survive a
restart, which is the safe default rather than the convenient one.
"""

import hashlib
import logging
import os
import secrets
import threading
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

#: Environment variable holding the identifier-hash salt.
HASH_SALT_ENV: str = "GENAI_MCP_HASH_SALT"

#: Default number of hex characters kept from the digest. 16 hex chars = 64 bits,
#: which is collision-safe well past any realistic session volume.
DEFAULT_HASH_LENGTH: int = 16

_salt_lock = threading.Lock()
_ephemeral_salt: Optional[str] = None
_warned_ephemeral = False


def _resolve_salt() -> str:
    """Resolve the hash salt, generating an ephemeral one if none is configured."""
    global _ephemeral_salt, _warned_ephemeral

    configured = os.getenv(HASH_SALT_ENV)
    if configured:
        return configured

    with _salt_lock:
        if _ephemeral_salt is None:
            _ephemeral_salt = secrets.token_hex(16)
        if not _warned_ephemeral:
            _warned_ephemeral = True
            logger.warning(
                "%s is not set - using an ephemeral per-process salt for identifier "
                "hashing. Hashes will not correlate across processes or restarts. "
                "Set %s to a stable secret if cross-process correlation is required.",
                HASH_SALT_ENV,
                HASH_SALT_ENV,
            )
        return _ephemeral_salt


def reset_salt_cache() -> None:
    """Clear the cached ephemeral salt. Intended for tests."""
    global _ephemeral_salt, _warned_ephemeral
    with _salt_lock:
        _ephemeral_salt = None
        _warned_ephemeral = False


def hash_identifier(
    value: Any,
    salt: Optional[str] = None,
    length: int = DEFAULT_HASH_LENGTH,
) -> Optional[str]:
    """Hash a user identifier for storage on a span.

    Args:
        value: The raw identifier. ``None`` and empty values return ``None``.
        salt: Explicit salt. Defaults to the configured or ephemeral salt.
        length: Number of hex characters to keep.

    Returns:
        Optional[str]: Salted digest prefixed with ``sha256:``, or None.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    effective_salt = salt if salt is not None else _resolve_salt()
    digest = hashlib.sha256((effective_salt + text).encode("utf-8")).hexdigest()
    return "sha256:" + digest[:length]


def hash_user_fields(
    payload: Mapping[str, Any],
    fields: Any = ("userId", "user_id", "phone", "phoneNumber", "email", "customerId"),
    salt: Optional[str] = None,
) -> dict:
    """Return a copy of ``payload`` with known user-identifier fields hashed.

    Only the named fields are touched; everything else is passed through, so the
    caller stays in control of what is recorded.
    """
    wanted = set(fields)
    result = {}
    for key, value in payload.items():
        if key in wanted:
            hashed = hash_identifier(value, salt=salt)
            if hashed is not None:
                result[key] = hashed
                continue
        result[key] = value
    return result
