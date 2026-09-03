"""Resource attributes describing the host, the process, and the service instance.

Until this module existed the resource carried ``service.name`` and the SDK's
own ``telemetry.sdk.*`` keys and nothing else, so two processes of the same
service - on one host or on fifty - produced byte-identical resources. Nothing
downstream could attribute a span to a machine or to an instance.

Two rules govern what is emitted here:

1. **Every attribute name comes from the OpenTelemetry registry.** Nothing
   library-specific is invented, and the control surface is the standard
   ``OTEL_*`` variables, so an operator configures this the way they configure
   any other OTel SDK. The upstream SDK ships ``host``, ``os`` and ``process``
   detectors but runs none of them unless ``OTEL_EXPERIMENTAL_RESOURCE_DETECTORS``
   names them; we default that variable on rather than reimplementing them.

2. **An operator's explicit value always wins.** Anything supplied through
   ``OTEL_RESOURCE_ATTRIBUTES`` or a detector is left alone; this module only
   fills in what nobody else supplied.

``host.ip`` is the one registry attribute with no upstream detector, so it is
derived here.
"""

import ipaddress
import logging
import os
import re
import socket
import sys
import threading
import uuid
from typing import Dict, List, Optional, Sequence, Tuple

from opentelemetry.sdk.resources import Resource

logger = logging.getLogger(__name__)

# Registry names. Spelled out rather than imported from
# opentelemetry.semconv._incubating, which is explicitly unstable - a rename
# there should be a deliberate change here, not something we follow silently.
DEPLOYMENT_ENVIRONMENT_NAME = "deployment.environment.name"
HOST_IP = "host.ip"
HOST_NAME = "host.name"
PROCESS_COMMAND_ARGS = "process.command_args"
PROCESS_COMMAND_LINE = "process.command_line"
SERVICE_INSTANCE_ID = "service.instance.id"
SERVICE_NAME = "service.name"
TELEMETRY_DISTRO_NAME = "telemetry.distro.name"
TELEMETRY_DISTRO_VERSION = "telemetry.distro.version"

# Superseded spellings. `environment` was never a registry name at all;
# `telemetry.auto.*` was renamed to `telemetry.distro.*`. Both are still emitted
# for the same reason the GenAI token attributes are dual-emitted under
# `gen_ai/dup`: dropping a key silently breaks whatever is already reading it.
# These go at 2.0, where breaking belongs.
LEGACY_ENVIRONMENT = "environment"
LEGACY_DISTRO_NAME = "telemetry.auto.name"
LEGACY_DISTRO_VERSION = "telemetry.auto.version"

DISTRO_NAME = "genai-otel-instrument"

# The upstream detectors, all built in and registered as entry points.
_DEFAULT_DETECTORS = "host,os,process"

# `process.command_line` reproduces whatever was typed to start the process. The
# hardened profiles exist to keep exactly that class of content off the wire, so
# they do not opt in to the process detector. Host and instance identity - the
# part needed to separate traffic - are unaffected.
_HARDENED_DETECTORS = "host,os"
_HARDENED_PROFILES = frozenset({"strict", "bfsi", "bank"})

# semconv designates this namespace for a `service.instance.id` derived from an
# inherent stable ID rather than generated at random.
_INSTANCE_ID_NAMESPACE = uuid.UUID("4d63009a-8d0f-11ee-aad7-4c796ed8e320")

_PLACEHOLDER = "***REDACTED***"

# A flag whose *value* is a credential. Matched against the flag name with its
# leading dashes stripped, anchored so that `--monkey` is not read as `--key`.
_SECRET_FLAG = re.compile(
    r"(?i)(?:^|[-_.])"
    r"(?:pass(?:word|wd)?|secret|tokens?|api[-_]?keys?|keys?|auth|credentials?|creds?)$"
)

# The same, inside a single `name=value` argument.
_INLINE_SECRET = re.compile(
    r"(?i)(^|[\s,;])"
    r"((?:[\w.-]*[-_.])?"
    r"(?:pass(?:word|wd)?|secret|tokens?|api[-_]?keys?|keys?|auth|credentials?|creds?))"
    r"=[^\s,;]+"
)

# `scheme://user:password@host` - the credential sits in the URL, where no
# `name=value` pattern will ever find it.
_URL_CREDENTIALS = re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://[^\s:/@]+):[^\s@/]+@")

_generated_instance_id: Optional[str] = None
_instance_id_lock = threading.Lock()


def host_ip_addresses() -> Tuple[str, ...]:
    """Non-loopback IP addresses of this host.

    No upstream detector emits ``host.ip``, so it is resolved here. semconv
    requires loopback interfaces to be excluded, IPv4 in dotted-quad notation
    and IPv6 in RFC 5952 form - which is what ``str()`` of a parsed address
    gives. Best-effort: a host with no resolvable name yields no addresses
    rather than an error, since this must never keep an application from
    starting.
    """
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None)
    except (OSError, UnicodeError) as exc:  # pragma: no cover - host dependent
        logger.debug("host.ip detection failed: %s", exc)
        return ()

    seen = set()
    parsed_addresses = []
    for info in infos:
        # Strip any IPv6 scope id ("fe80::1%eth0") before parsing.
        candidate = str(info[4][0]).split("%", 1)[0]
        try:
            parsed = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if parsed.is_loopback or parsed.is_unspecified:
            continue
        if parsed in seen:
            continue
        seen.add(parsed)
        parsed_addresses.append(parsed)

    # getaddrinfo returns addresses in no guaranteed order, and a host with
    # virtual adapters offers plenty of them. Sort so the value is identical
    # across restarts - an array that reshuffles fragments grouping downstream -
    # and so the most useful address comes first: routable before private,
    # private before link-local.
    parsed_addresses.sort(
        key=lambda address: (
            address.is_link_local,
            address.is_private,
            address.version,
            address.packed,
        )
    )
    return tuple(str(address) for address in parsed_addresses)


def _is_secret_flag(argument: str) -> bool:
    return bool(_SECRET_FLAG.search(argument.lstrip("-")))


def _redact_inline(argument: str) -> str:
    argument = _INLINE_SECRET.sub(rf"\1\2={_PLACEHOLDER}", argument)
    return _URL_CREDENTIALS.sub(rf"\1:{_PLACEHOLDER}@", argument)


def redact_command_args(args: Sequence[str]) -> List[str]:
    """Blank out credential values in an argument vector.

    This has to happen here rather than downstream. ``process.command_args`` is
    an array, so ``--password`` and ``hunter2`` become two separate elements;
    once flattened, the value element carries no indication of which flag it
    belonged to, and a ``name=value`` pattern cannot span two elements. The SDK
    is the last point at which the flag and its value are still adjacent.

    Both spellings are handled: ``--password=hunter2`` and ``--password
    hunter2``. A following argument that itself looks like a long flag is left
    alone, so ``--password --verbose`` does not lose the second flag.
    """
    redacted: List[str] = []
    redact_next = False

    for argument in args:
        if redact_next:
            redact_next = False
            # A long flag here means no value was passed for the secret flag.
            if not argument.startswith("--"):
                redacted.append(_PLACEHOLDER)
                continue

        if argument.startswith("-"):
            name, separator, _ = argument.partition("=")
            if separator and _is_secret_flag(name):
                redacted.append(f"{name}={_PLACEHOLDER}")
                continue
            if not separator and _is_secret_flag(argument):
                redact_next = True
                redacted.append(argument)
                continue

        redacted.append(_redact_inline(argument))

    return redacted


def process_argv() -> List[str]:
    """The startup argument vector, as the SDK's process detector reports it.

    SDK 1.42 switched the process detector from ``sys.argv`` to
    ``sys.orig_argv``, which keeps the interpreter and any ``-m module`` that
    ``sys.argv`` rewrites away. Prefer the same source so a derived instance id
    is seeded from what actually lands on the span; ``sys.orig_argv`` needs
    Python 3.10, and this package supports 3.9.
    """
    return list(getattr(sys, "orig_argv", sys.argv))


def _random_instance_id() -> str:
    """A v4 UUID, stable for the life of this process.

    semconv's primary recommendation, and the only option that is guaranteed
    unique - which the spec requires of instances of one service running at the
    same time.
    """
    global _generated_instance_id
    with _instance_id_lock:
        if _generated_instance_id is None:
            _generated_instance_id = str(uuid.uuid4())
        return _generated_instance_id


def _derived_instance_id(service_name: str) -> str:
    """A v5 UUID over host name, service name, and the normalised argument list.

    Survives a restart, which a random id does not: a consumer computing
    per-instance baselines needs the same instance to keep its identity across
    deployments, or it never accumulates enough history.

    The trade-off is real and is why this is not the default. Instances that
    differ in no argument - four workers started identically on one host -
    collapse to a single id, which breaks semconv's requirement that concurrent
    instances of a service be distinguishable. Prefer setting
    ``OTEL_SERVICE_INSTANCE_ID`` to something the orchestrator already
    guarantees unique and stable, such as a pod name.
    """
    arguments = " ".join(sorted(redact_command_args(process_argv())))
    seed = "|".join((socket.gethostname(), service_name, arguments))
    return str(uuid.uuid5(_INSTANCE_ID_NAMESPACE, seed))


def _instance_id(service_name: str) -> str:
    mode = os.getenv("GENAI_SERVICE_INSTANCE_ID_MODE", "random").strip().lower()
    if mode == "derived":
        return _derived_instance_id(service_name)
    if mode not in ("random", ""):
        logger.warning(
            "Unknown GENAI_SERVICE_INSTANCE_ID_MODE %r; using 'random'. "
            "Valid values are 'random' and 'derived'.",
            mode,
        )
    return _random_instance_id()


def build_resource(service_name: str, distro_version: str, profile: str = "") -> Resource:
    """Build the resource shared by the tracer and meter providers."""
    default_detectors = (
        _HARDENED_DETECTORS if profile.strip().lower() in _HARDENED_PROFILES else _DEFAULT_DETECTORS
    )
    os.environ.setdefault("OTEL_EXPERIMENTAL_RESOURCE_DETECTORS", default_detectors)

    attributes: Dict[str, object] = {
        SERVICE_NAME: service_name,
        TELEMETRY_DISTRO_NAME: DISTRO_NAME,
        TELEMETRY_DISTRO_VERSION: distro_version,
        LEGACY_DISTRO_NAME: DISTRO_NAME,
        LEGACY_DISTRO_VERSION: distro_version,
    }

    configured_instance_id = os.getenv("OTEL_SERVICE_INSTANCE_ID")
    if configured_instance_id:
        attributes[SERVICE_INSTANCE_ID] = configured_instance_id

    environment = os.getenv("OTEL_ENVIRONMENT")
    if environment:
        attributes[DEPLOYMENT_ENVIRONMENT_NAME] = environment
        attributes[LEGACY_ENVIRONMENT] = environment

    resource = Resource.create(attributes)

    # Credential values are stripped from the detected argument vector, and the
    # joined command line is rebuilt from the stripped version so the two cannot
    # disagree.
    overrides: Dict[str, object] = {}
    detected_args = resource.attributes.get(PROCESS_COMMAND_ARGS)
    if detected_args:
        safe_args = redact_command_args([str(item) for item in detected_args])
        if safe_args != [str(item) for item in detected_args]:
            overrides[PROCESS_COMMAND_ARGS] = safe_args
            overrides[PROCESS_COMMAND_LINE] = " ".join(safe_args)
    if overrides:
        resource = resource.merge(Resource(overrides))

    # Fill in only what neither a detector nor OTEL_RESOURCE_ATTRIBUTES gave us,
    # so an operator's explicit value is never second-guessed.
    fallbacks: Dict[str, object] = {}
    if not resource.attributes.get(SERVICE_INSTANCE_ID):
        fallbacks[SERVICE_INSTANCE_ID] = _instance_id(service_name)
    if not resource.attributes.get(HOST_IP):
        addresses = host_ip_addresses()
        if addresses:
            fallbacks[HOST_IP] = list(addresses)
    if fallbacks:
        resource = Resource(fallbacks).merge(resource)

    return resource
