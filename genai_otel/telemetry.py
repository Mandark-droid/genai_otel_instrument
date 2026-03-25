"""Optional anonymous usage telemetry for TraceVerde.

This module collects anonymous, non-identifiable usage statistics to help
the maintainers understand how the library is being used and prioritize
development efforts.

**Telemetry is DISABLED by default.** To opt in, set the environment variable:

    export TRACEVERDE_TELEMETRY=true

What is collected (all anonymous):
    - Python version (e.g., "3.11.5")
    - Package version (e.g., "1.0.0")
    - OS platform (e.g., "linux", "win32", "darwin")
    - Which instrumentors are enabled (names only, e.g., ["openai", "anthropic"])
    - Whether GPU metrics, cost tracking, and evaluation features are enabled
    - A random installation ID (UUID, NOT tied to any user identity)

What is NOT collected:
    - No API keys, tokens, or credentials
    - No IP addresses, hostnames, or user identifiers
    - No telemetry data, traces, or metrics from your application
    - No file paths, environment variables, or system details

The telemetry endpoint URL can be configured via:

    export TRACEVERDE_TELEMETRY_URL=https://your-endpoint.example.com/v1/telemetry

Telemetry calls are non-blocking (fire-and-forget in a daemon thread) with a
2-second timeout. They will never raise exceptions or affect library functionality.
"""

import json
import logging
import os
import platform
import sys
import threading
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

_TELEMETRY_ENV_VAR = "TRACEVERDE_TELEMETRY"
_TELEMETRY_URL_ENV_VAR = "TRACEVERDE_TELEMETRY_URL"
_DEFAULT_TELEMETRY_URL = ""  # Disabled by default - no default endpoint
_TELEMETRY_TIMEOUT = 2  # seconds
_ID_DIR = Path.home() / ".traceverde"
_ID_FILE = _ID_DIR / "telemetry_id"


def _get_installation_id():
    """Get or create a random installation ID (UUID).

    Stored in ~/.traceverde/telemetry_id. This is a random UUID with no
    connection to any user identity.
    """
    try:
        if _ID_FILE.exists():
            return _ID_FILE.read_text().strip()
        _ID_DIR.mkdir(parents=True, exist_ok=True)
        install_id = str(uuid.uuid4())
        _ID_FILE.write_text(install_id)
        return install_id
    except (OSError, PermissionError):
        # If we can't read/write the file, generate a transient ID
        return str(uuid.uuid4())


def _is_telemetry_enabled():
    """Check if telemetry is opted in via environment variable."""
    return os.environ.get(_TELEMETRY_ENV_VAR, "").lower() in ("true", "1", "yes")


def _build_payload(config=None):
    """Build the anonymous telemetry payload."""
    try:
        from . import __version__
    except Exception:
        __version__ = "unknown"

    payload = {
        "installation_id": _get_installation_id(),
        "package_version": str(__version__),
        "python_version": platform.python_version(),
        "platform": sys.platform,
        "arch": platform.machine(),
    }

    if config is not None:
        payload["instrumentors"] = sorted(config.enabled_instrumentors or [])
        payload["gpu_metrics_enabled"] = getattr(config, "enable_gpu_metrics", False)
        payload["cost_tracking_enabled"] = getattr(config, "enable_cost_tracking", False)
        payload["mcp_enabled"] = getattr(config, "enable_mcp_instrumentation", False)
        # Evaluation features
        payload["evaluation_enabled"] = any(
            [
                getattr(config, "enable_pii_detection", False),
                getattr(config, "enable_toxicity_detection", False),
                getattr(config, "enable_bias_detection", False),
                getattr(config, "enable_prompt_injection_detection", False),
                getattr(config, "enable_restricted_topics", False),
                getattr(config, "enable_hallucination_detection", False),
            ]
        )

    return payload


def _send_telemetry(payload, url):
    """Send telemetry data via HTTPS POST. Non-blocking, fire-and-forget."""
    try:
        import urllib.request

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=_TELEMETRY_TIMEOUT)  # nosec B310
        logger.debug("Telemetry sent successfully")
    except Exception:
        # Never let telemetry affect the library
        logger.debug("Telemetry send failed (this is non-critical)")


def report_usage(config=None):
    """Report anonymous usage telemetry if opted in.

    This function is safe to call unconditionally - it checks the opt-in
    environment variable and returns immediately if telemetry is disabled.

    Args:
        config: Optional OTelConfig instance to report enabled features.
    """
    if not _is_telemetry_enabled():
        return

    url = os.environ.get(_TELEMETRY_URL_ENV_VAR, _DEFAULT_TELEMETRY_URL)
    if not url:
        logger.debug(
            "Telemetry enabled but no endpoint configured (set %s)", _TELEMETRY_URL_ENV_VAR
        )
        return

    try:
        payload = _build_payload(config)
        thread = threading.Thread(
            target=_send_telemetry,
            args=(payload, url),
            daemon=True,
            name="traceverde-telemetry",
        )
        thread.start()
    except Exception:
        # Never let telemetry affect the library
        pass
