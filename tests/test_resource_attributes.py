"""Host, process, and instance identity on the resource.

Until now the resource carried only ``service.name`` and the SDK's own
``telemetry.sdk.*`` keys, so two processes of the same service - on one host or
on fifty - produced byte-identical resources and could not be told apart in a
backend. Per-host and per-instance attribution was impossible.

Everything asserted here is a name from the OpenTelemetry registry. No
library-specific resource key is invented, and the control surface is the
standard ``OTEL_*`` variables, so an operator configures this exactly as they
would any other OTel SDK.
"""

import contextlib
import ipaddress
import os
import re
import sys
import uuid
from unittest.mock import patch

import pytest

from genai_otel.resource import build_resource, host_ip_addresses, redact_command_args

# Registry names, spelled out rather than imported, so that a rename in the
# (explicitly unstable) semconv package surfaces here as a failure instead of
# being followed silently.
HOST_NAME = "host.name"
HOST_IP = "host.ip"
OS_TYPE = "os.type"
PROCESS_PID = "process.pid"
PROCESS_COMMAND = "process.command"
PROCESS_COMMAND_LINE = "process.command_line"
PROCESS_COMMAND_ARGS = "process.command_args"
SERVICE_INSTANCE_ID = "service.instance.id"
SERVICE_NAME = "service.name"
DEPLOYMENT_ENVIRONMENT_NAME = "deployment.environment.name"
TELEMETRY_DISTRO_NAME = "telemetry.distro.name"
TELEMETRY_DISTRO_VERSION = "telemetry.distro.version"

PLACEHOLDER = "***REDACTED***"

RESOURCE_ENV_VARS = (
    "OTEL_EXPERIMENTAL_RESOURCE_DETECTORS",
    "OTEL_RESOURCE_ATTRIBUTES",
    "OTEL_SERVICE_INSTANCE_ID",
    "OTEL_SERVICE_NAME",
    "OTEL_ENVIRONMENT",
    "GENAI_SERVICE_INSTANCE_ID_MODE",
)


@pytest.fixture
def clean_env():
    """Run with every resource-related variable unset.

    ``build_resource`` calls ``os.environ.setdefault`` on the detector list, so
    without this the first test to run would configure all the others.
    """
    with patch.dict(os.environ, {}, clear=False):
        for name in RESOURCE_ENV_VARS:
            os.environ.pop(name, None)
        yield


def _build(profile="", **env):
    with patch.dict(os.environ, env, clear=False):
        return build_resource("test-service", "9.9.9", profile=profile)


@contextlib.contextmanager
def fake_argv(argv):
    """Patch every vector the SDK's process detector might read.

    SDK <= 1.41 reads ``sys.argv``; 1.42 switched to ``sys.orig_argv``. This
    package supports the whole range, so patch both rather than pinning the
    test to whichever SDK happens to be installed.
    """
    with patch.object(sys, "argv", list(argv)):
        with patch.object(sys, "orig_argv", list(argv), create=True):
            yield


# ---------------------------------------------------------------------------
# Host identity - "which machine did this span come from"
# ---------------------------------------------------------------------------


def test_host_name_is_present_by_default(clean_env):
    """The SDK runs only the 'otel' detector unless asked; we ask for 'host'."""
    resource = build_resource("test-service", "9.9.9")
    assert resource.attributes.get(HOST_NAME), (
        "no host.name on the resource - spans from different machines are "
        "indistinguishable in the backend"
    )


def test_os_type_is_present_by_default(clean_env):
    assert build_resource("test-service", "9.9.9").attributes.get(OS_TYPE)


def test_host_ip_excludes_loopback(clean_env):
    """semconv: host.ip is the host's addresses *excluding* loopback interfaces."""
    for address in host_ip_addresses():
        parsed = ipaddress.ip_address(address)
        assert not parsed.is_loopback, f"{address} is a loopback address"
        assert not parsed.is_unspecified, f"{address} is the unspecified address"


def test_host_ip_addresses_are_valid_literals(clean_env):
    """IPv4 dotted-quad, IPv6 in RFC 5952 form - which is str() of a parsed address."""
    for address in host_ip_addresses():
        assert str(ipaddress.ip_address(address)) == address


def test_host_ip_order_is_deterministic(clean_env):
    """A reshuffling array would fragment grouping in the backend."""
    assert host_ip_addresses() == host_ip_addresses()


def test_host_ip_puts_the_useful_address_first(clean_env):
    """Routable before private, private before link-local."""
    addresses = [ipaddress.ip_address(item) for item in host_ip_addresses()]
    if not addresses:
        pytest.skip("this host has no non-loopback address")
    ranks = [(address.is_link_local, address.is_private) for address in addresses]
    assert ranks == sorted(ranks), f"link-local addresses sort ahead of routable ones: {addresses}"


def test_host_ip_is_a_sequence_on_the_resource(clean_env):
    """host.ip is defined as an array, so it must not collapse to a bare string."""
    value = build_resource("test-service", "9.9.9").attributes.get(HOST_IP)
    if value is None:
        pytest.skip("this host has no non-loopback address")
    assert not isinstance(value, str)
    assert all(isinstance(item, str) for item in value)


# ---------------------------------------------------------------------------
# Instance identity - "which of the N processes on that machine"
# ---------------------------------------------------------------------------


def test_service_instance_id_is_generated_when_unset(clean_env):
    """Without this, two workers of one service on one host are identical."""
    value = build_resource("test-service", "9.9.9").attributes.get(SERVICE_INSTANCE_ID)
    assert value, "no service.instance.id - instances cannot be separated"
    assert uuid.UUID(value).version == 4, "semconv's default recommendation is a random v4"


def test_generated_instance_id_is_stable_within_a_process(clean_env):
    """Two providers built in one process describe the same instance."""
    first = build_resource("test-service", "9.9.9").attributes[SERVICE_INSTANCE_ID]
    second = build_resource("test-service", "9.9.9").attributes[SERVICE_INSTANCE_ID]
    assert first == second


def test_explicit_instance_id_wins(clean_env):
    assert _build(OTEL_SERVICE_INSTANCE_ID="worker-7").attributes[SERVICE_INSTANCE_ID] == "worker-7"


def test_instance_id_from_otel_resource_attributes_wins(clean_env):
    """The standard variable must not be overridden by our generated UUID."""
    resource = _build(OTEL_RESOURCE_ATTRIBUTES="service.instance.id=pod-abc")
    assert resource.attributes[SERVICE_INSTANCE_ID] == "pod-abc"


def test_host_ip_from_otel_resource_attributes_wins(clean_env):
    """An operator who pins the routable address must not be second-guessed."""
    assert _build(OTEL_RESOURCE_ATTRIBUTES="host.ip=10.0.0.7").attributes[HOST_IP] == "10.0.0.7"


# A random id changes on every restart, so a consumer computing per-instance
# baselines never accumulates history for one instance. The derived mode trades
# guaranteed uniqueness for stability across restarts.


def test_derived_instance_id_is_a_v5_uuid(clean_env):
    """semconv designates v5 for an id derived from an inherent stable value."""
    resource = _build(GENAI_SERVICE_INSTANCE_ID_MODE="derived")
    assert uuid.UUID(resource.attributes[SERVICE_INSTANCE_ID]).version == 5


def test_derived_instance_id_is_reproducible(clean_env):
    first = _build(GENAI_SERVICE_INSTANCE_ID_MODE="derived").attributes[SERVICE_INSTANCE_ID]
    second = _build(GENAI_SERVICE_INSTANCE_ID_MODE="derived").attributes[SERVICE_INSTANCE_ID]
    assert first == second


def test_derived_instance_id_differs_per_service(clean_env):
    with patch.dict(os.environ, {"GENAI_SERVICE_INSTANCE_ID_MODE": "derived"}, clear=False):
        one = build_resource("service-a", "9.9.9").attributes[SERVICE_INSTANCE_ID]
        two = build_resource("service-b", "9.9.9").attributes[SERVICE_INSTANCE_ID]
    assert one != two


def test_unknown_instance_id_mode_falls_back_to_random(clean_env):
    resource = _build(GENAI_SERVICE_INSTANCE_ID_MODE="nonsense")
    assert uuid.UUID(resource.attributes[SERVICE_INSTANCE_ID]).version == 4


# ---------------------------------------------------------------------------
# Process identity - the startup command line
# ---------------------------------------------------------------------------


def test_process_attributes_are_present_by_default(clean_env):
    resource = build_resource("test-service", "9.9.9")
    assert resource.attributes.get(PROCESS_PID)
    assert resource.attributes.get(PROCESS_COMMAND_LINE) is not None
    assert resource.attributes.get(PROCESS_COMMAND) is not None


def test_command_args_are_a_sequence(clean_env):
    args = build_resource("test-service", "9.9.9").attributes.get(PROCESS_COMMAND_ARGS)
    assert args is not None and not isinstance(args, str)


def test_operator_can_drop_the_process_detector(clean_env):
    """The off-switch is the standard variable, not a GENAI_ flag."""
    resource = _build(OTEL_EXPERIMENTAL_RESOURCE_DETECTORS="host,os")
    assert PROCESS_COMMAND_LINE not in resource.attributes
    assert resource.attributes.get(HOST_NAME), "dropping 'process' must not drop 'host'"


def test_operator_can_restore_the_old_minimal_resource(clean_env):
    resource = _build(OTEL_EXPERIMENTAL_RESOURCE_DETECTORS="otel")
    assert HOST_NAME not in resource.attributes
    assert PROCESS_PID not in resource.attributes
    # Instance identity is ours, not a detector's, so it survives.
    assert resource.attributes.get(SERVICE_INSTANCE_ID)


def test_hardened_profile_omits_the_process_detector(clean_env):
    """``process.command_line`` reproduces whatever was typed to start the process.

    An operator passing a key as an argument would ship it to the backend on
    every span's resource. The bank/BFSI profile exists to stop that class of
    leak, so it does not opt in. Host and instance identity - the part that is
    mandatory for separating traffic - are unaffected.
    """
    resource = build_resource("test-service", "9.9.9", profile="bank")
    assert PROCESS_COMMAND_LINE not in resource.attributes
    assert resource.attributes.get(HOST_NAME)
    assert resource.attributes.get(SERVICE_INSTANCE_ID)


@pytest.mark.parametrize("profile", ["strict", "bfsi", "bank", "BANK", " bank "])
def test_every_hardened_profile_spelling_omits_process(clean_env, profile):
    resource = build_resource("test-service", "9.9.9", profile=profile)
    assert PROCESS_COMMAND_LINE not in resource.attributes


def test_hardened_profile_still_honours_an_explicit_opt_in(clean_env):
    """The profile picks a default; it does not overrule the operator."""
    resource = _build(profile="bank", OTEL_EXPERIMENTAL_RESOURCE_DETECTORS="host,os,process")
    assert resource.attributes.get(PROCESS_COMMAND_LINE) is not None


# ---------------------------------------------------------------------------
# Credentials in the argument vector
# ---------------------------------------------------------------------------

# `process.command_args` is an array, so "--password" and "hunter2" arrive as
# separate elements. Downstream, the value element carries no trace of which
# flag it belonged to, and a name=value pattern cannot span two elements. This
# is the last point where the flag and its value are still adjacent.

SECRETS = [
    (["--password", "hunter2"], ["--password", PLACEHOLDER]),
    (["--password=hunter2"], [f"--password={PLACEHOLDER}"]),
    (["-p", "8080"], ["-p", "8080"]),
    (["--db-pass", "hunter2"], ["--db-pass", PLACEHOLDER]),
    (["--api-key", "sk-abc123"], ["--api-key", PLACEHOLDER]),
    (["--openai-api-key=sk-abc123"], [f"--openai-api-key={PLACEHOLDER}"]),
    (["--token", "ghp_x"], ["--token", PLACEHOLDER]),
    (["--auth", "Bearer x"], ["--auth", PLACEHOLDER]),
    (["--client-secret=s3cr3t"], [f"--client-secret={PLACEHOLDER}"]),
    (["--credentials", "/etc/creds.json"], ["--credentials", PLACEHOLDER]),
]


@pytest.mark.parametrize("args,expected", SECRETS)
def test_credential_values_are_redacted(args, expected):
    assert redact_command_args(args) == expected


BENIGN = [
    ["--port", "8080"],
    ["--workers", "4"],
    ["serve", "--host", "0.0.0.0"],
    ["--monkey", "business"],  # must not be read as "--key"
    ["--keyboard-layout", "us"],
    ["python", "-m", "uvicorn", "app:main"],
]


@pytest.mark.parametrize("args", BENIGN)
def test_benign_arguments_are_untouched(args):
    assert redact_command_args(args) == args


def test_missing_value_does_not_swallow_the_next_flag():
    assert redact_command_args(["--password", "--verbose"]) == ["--password", "--verbose"]


def test_inline_name_value_pairs_are_redacted():
    assert redact_command_args(["--config", "token=abc123"]) == [
        "--config",
        f"token={PLACEHOLDER}",
    ]


def test_url_credentials_are_redacted():
    """The Telegram-token shape: the secret sits in the URL, not in a value."""
    assert redact_command_args(["postgres://user:hunter2@db:5432/app"]) == [
        f"postgres://user:{PLACEHOLDER}@db:5432/app"
    ]


def test_command_line_is_rebuilt_from_the_redacted_args(clean_env):
    """The joined string and the array must never disagree."""
    argv = ["serve.py", "--api-key", "sk-live-abcdef", "--port", "8080"]
    with fake_argv(argv):
        resource = build_resource("test-service", "9.9.9")
    args = list(resource.attributes[PROCESS_COMMAND_ARGS])
    assert "sk-live-abcdef" not in args
    assert args == ["serve.py", "--api-key", PLACEHOLDER, "--port", "8080"]
    assert resource.attributes[PROCESS_COMMAND_LINE] == " ".join(args)
    assert "sk-live-abcdef" not in resource.attributes[PROCESS_COMMAND_LINE]


def test_clean_command_line_is_left_alone(clean_env):
    with fake_argv(["serve.py", "--port", "8080"]):
        resource = build_resource("test-service", "9.9.9")
    assert list(resource.attributes[PROCESS_COMMAND_ARGS]) == ["serve.py", "--port", "8080"]


# ---------------------------------------------------------------------------
# Registry conformance - no invented keys
# ---------------------------------------------------------------------------


def test_environment_uses_the_registry_name(clean_env):
    assert _build(OTEL_ENVIRONMENT="production").attributes[DEPLOYMENT_ENVIRONMENT_NAME] == (
        "production"
    )


def test_environment_keeps_the_superseded_key(clean_env):
    """Dropping it silently would break anything already filtering on it.

    Same reasoning as the gen_ai/dup token names: emit both, break at 2.0.
    """
    assert _build(OTEL_ENVIRONMENT="production").attributes["environment"] == "production"


def test_distro_uses_the_registry_name(clean_env):
    resource = build_resource("test-service", "9.9.9")
    assert resource.attributes[TELEMETRY_DISTRO_NAME] == "genai-otel-instrument"
    assert resource.attributes[TELEMETRY_DISTRO_VERSION] == "9.9.9"


def test_distro_keeps_the_superseded_keys(clean_env):
    resource = build_resource("test-service", "9.9.9")
    assert resource.attributes["telemetry.auto.name"] == "genai-otel-instrument"
    assert resource.attributes["telemetry.auto.version"] == "9.9.9"


# Every namespace this library may write to. `environment` is the one bare key,
# kept only for compatibility. A new key must be justified against the OTel
# registry before it is added here.
ALLOWED_PREFIXES = ("host.", "os.", "process.", "service.", "telemetry.", "deployment.")
ALLOWED_EXACT = {"environment"}


def test_no_library_specific_resource_keys_are_invented(clean_env):
    """Guards the rule: resource attributes come from the registry, not from us."""
    resource = build_resource("test-service", "9.9.9")
    invented = [
        key
        for key in resource.attributes
        if key not in ALLOWED_EXACT and not key.startswith(ALLOWED_PREFIXES)
    ]
    assert not invented, f"non-registry resource keys: {sorted(invented)}"


def test_service_name_still_comes_from_config(clean_env):
    assert build_resource("test-service", "9.9.9").attributes[SERVICE_NAME] == "test-service"


def test_generated_id_is_a_plain_lowercase_uuid(clean_env):
    value = build_resource("test-service", "9.9.9").attributes[SERVICE_INSTANCE_ID]
    assert re.fullmatch(r"[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}", value)
