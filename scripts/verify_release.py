#!/usr/bin/env python3
"""Verify a published release by installing it from PyPI and inspecting it.

Run after `gh release create` has published a version, to confirm that what
PyPI actually serves matches what was intended. This is deliberately not a
pytest module: it talks to the network and builds a virtualenv, and it has to
be runnable against any already-published version.

    python scripts/verify_release.py --version 1.27.0 \
        --min-chat-entries 1495 \
        --require-model gpt-5.6-luna=0.0002/0.0012 \
        --require-note gpt-5-6-luna="Above 272000 tokens of context"

Every check here exists because it failed silently at least once.

1.  A venv directory that already exists is REUSED by `python -m venv` without
    warning, so a leftover venv from a previous release check keeps the version
    it already had. Combined with (2) this reports the previous release's
    contents as if they were the new one.

2.  `pip install pkg==X` failing does not stop a later `import pkg` from
    succeeding against whatever happens to be installed. The install's exit
    code is checked, and then the installed version is asserted independently
    through importlib.metadata, because the two can disagree.

3.  PyPI's JSON API and its simple index propagate separately. For roughly a
    minute after upload, `pypi.org/pypi/<pkg>/json` reports the new version
    while `pypi.org/simple/<pkg>/` does not yet list the file, so pip cannot
    install what the JSON API says exists. The simple index is the one pip
    reads, so that is the one worth waiting on.

4.  Running from the repository root puts the working tree ahead of site-
    packages on sys.path, so `import genai_otel` silently reads uncommitted
    local code instead of the wheel. The probe asserts its own import path is
    inside the venv.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

PACKAGE = "genai-otel-instrument"
IMPORT_NAME = "genai_otel"
SIMPLE_URL = f"https://pypi.org/simple/{PACKAGE}/"
JSON_URL = f"https://pypi.org/pypi/{PACKAGE}/json"


class VerificationError(Exception):
    """A check failed. The message is the report."""


def _fetch(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "verify-release"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def wheel_name(version):
    """The wheel filename pip will look for. Normalised per PEP 427."""
    return f"{PACKAGE.replace('-', '_')}-{version}-py3-none-any.whl"


def index_has_version(index_html, version):
    """True when the simple index lists a wheel or sdist for this version.

    Matching on the exact filename rather than a bare substring keeps 1.2 from
    matching 1.27.0.
    """
    dist = PACKAGE.replace("-", "_")
    pattern = re.escape(f"{dist}-{version}") + r"(-py3-none-any\.whl|\.tar\.gz)"
    return re.search(pattern, index_html) is not None


def json_api_version(timeout=30):
    try:
        return json.loads(_fetch(JSON_URL, timeout))["info"]["version"]
    except Exception:
        return None


def wait_for_simple_index(version, timeout_s, poll_s=15, log=print):
    """Block until pip could actually resolve this version, or give up.

    Returns the number of seconds waited. Reports JSON-vs-index skew when it
    happens, because that skew is what makes a correct release look broken.
    """
    started = time.time()
    reported_skew = False
    while True:
        try:
            if index_has_version(_fetch(SIMPLE_URL), version):
                waited = time.time() - started
                log(f"  simple index lists {version} after {waited:.0f}s")
                return waited
        except urllib.error.URLError as e:
            log(f"  simple index unreachable ({e}); retrying")

        if not reported_skew and json_api_version() == version:
            log(
                f"  NOTE: the JSON API already reports {version} but the simple "
                "index does not list it yet. pip reads the simple index, so an "
                "install right now would fail with 'No matching distribution'."
            )
            reported_skew = True

        if time.time() - started > timeout_s:
            raise VerificationError(
                f"{version} did not appear on the simple index within "
                f"{timeout_s}s. The JSON API reported "
                f"{json_api_version()!r}. Publishing may have failed."
            )
        time.sleep(poll_s)


def make_clean_venv(path, log=print):
    """Create a venv, removing any existing one first.

    `python -m venv` on an existing directory silently keeps whatever is
    installed there, which is how a stale venv reports an old release.
    """
    if os.path.exists(path):
        log(f"  removing existing venv at {path}")
        shutil.rmtree(path)
    subprocess.run([sys.executable, "-m", "venv", path], check=True, capture_output=True, text=True)
    exe = os.path.join(
        path, "Scripts" if os.name == "nt" else "bin", "python.exe" if os.name == "nt" else "python"
    )
    if not os.path.exists(exe):
        raise VerificationError(f"venv created but no interpreter at {exe}")
    return exe


PROBE = r"""
import importlib.metadata as md, json, os, sys
out = {}
try:
    out["installed_version"] = md.version(%(pkg)r)
except Exception as e:
    out["installed_version"] = None
    out["metadata_error"] = str(e)
import %(imp)s
from genai_otel.cost_calculator import CostCalculator
out["package_path"] = os.path.dirname(%(imp)s.__file__)
out["cwd"] = os.getcwd()
c = CostCalculator()
table = c.pricing_data["chat"]
out["chat_entries"] = len(table)
out["models"] = {}
for name in %(models)r:
    key = c._resolve_model_key(name, "chat")
    entry = table.get(key) or {}
    out["models"][name] = {
        "key": key,
        "promptPrice": entry.get("promptPrice"),
        "completionPrice": entry.get("completionPrice"),
        "cacheReadPrice": entry.get("cacheReadPrice"),
        "cacheWritePrice": entry.get("cacheWritePrice"),
    }
out["notes"] = {k: (table.get(k) or {}).get("note", "") for k in %(notes)r}
print("PROBE_JSON:" + json.dumps(out))
"""


def run_probe(python_exe, models, note_keys, workdir):
    src = PROBE % {
        "pkg": PACKAGE,
        "imp": IMPORT_NAME,
        "models": list(models),
        "notes": list(note_keys),
    }
    script = os.path.join(workdir, "_probe.py")
    with open(script, "w", encoding="utf-8") as f:
        f.write(src)
    # cwd is the workdir, never the repo: the repository root would put the
    # working tree ahead of site-packages on sys.path.
    proc = subprocess.run([python_exe, script], capture_output=True, text=True, cwd=workdir)
    if proc.returncode != 0:
        raise VerificationError(
            f"probe failed inside the venv (exit {proc.returncode}):\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    for line in proc.stdout.splitlines():
        if line.startswith("PROBE_JSON:"):
            return json.loads(line[len("PROBE_JSON:") :])
    raise VerificationError(f"probe produced no result:\n{proc.stdout}\n{proc.stderr}")


def parse_model_spec(spec):
    """'name=prompt/completion' -> (name, prompt, completion). Prices optional."""
    if "=" not in spec:
        return spec, None, None
    name, _, prices = spec.partition("=")
    if "/" not in prices:
        raise argparse.ArgumentTypeError(
            f"--require-model {spec!r}: expected NAME=PROMPT/COMPLETION"
        )
    p, _, comp = prices.partition("/")
    return name, float(p), float(comp)


def parse_note_spec(spec):
    key, _, text = spec.partition("=")
    if not text:
        raise argparse.ArgumentTypeError(f"--require-note {spec!r}: expected KEY=SUBSTRING")
    return key, text


def evaluate(result, version, require_model=(), require_note=(), min_chat_entries=None):
    """Compare a probe result against expectations. Returns a list of failures.

    Pure, so the checks can be tested without a network or a venv.
    """
    failures = []

    # The install's exit code is not evidence of what got installed: a stale
    # venv satisfies the import while pip's resolve failed.
    installed = result.get("installed_version")
    if installed != version:
        failures.append(
            f"installed version is {installed!r}, expected {version!r}. "
            "pip reported success, so this is a stale environment or a "
            "resolver fallback, not a network error."
        )

    pkg_path = (result.get("package_path") or "").replace("\\", "/")
    if "site-packages" not in pkg_path:
        failures.append(
            f"imported {IMPORT_NAME} from {pkg_path!r}, which is not inside the "
            "venv's site-packages. The probe read local source, not the wheel."
        )

    if min_chat_entries is not None:
        n = result.get("chat_entries", 0)
        if n < min_chat_entries:
            failures.append(f"chat table has {n} entries, expected at least {min_chat_entries}")

    for name, want_prompt, want_completion in require_model:
        got = (result.get("models") or {}).get(name, {})
        key = got.get("key")
        if key is None:
            failures.append(f"{name}: resolves to nothing (unpriced)")
            continue
        if key != name and want_prompt is not None:
            failures.append(
                f"{name}: resolved to {key!r} rather than its own entry, so it "
                "is inheriting another model's rate"
            )
        if want_prompt is not None and got.get("promptPrice") != want_prompt:
            failures.append(
                f"{name}: promptPrice is {got.get('promptPrice')}, expected {want_prompt}"
            )
        if want_completion is not None and got.get("completionPrice") != want_completion:
            failures.append(
                f"{name}: completionPrice is {got.get('completionPrice')}, "
                f"expected {want_completion}"
            )

    for key, text in require_note:
        note = (result.get("notes") or {}).get(key, "")
        if text not in note:
            failures.append(f"{key}: note does not contain {text!r}. Note reads: {note[:120]!r}")

    return failures


def verify(args, log=print):
    log(f"Verifying {PACKAGE}=={args.version}")
    log("Waiting for the simple index (the one pip reads)...")
    wait_for_simple_index(args.version, args.index_timeout, log=log)

    workdir = args.workdir or tempfile.mkdtemp(prefix="verify-release-")
    os.makedirs(workdir, exist_ok=True)
    venv_path = os.path.join(workdir, "venv")
    log(f"Building a clean venv in {venv_path}")
    python_exe = make_clean_venv(venv_path, log=log)

    log(f"Installing {PACKAGE}=={args.version}")
    proc = subprocess.run(
        [
            python_exe,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-cache-dir",
            f"{PACKAGE}=={args.version}",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise VerificationError(
            f"pip install failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        )

    models = [m[0] for m in args.require_model]
    note_keys = [n[0] for n in args.require_note]
    result = run_probe(python_exe, models, note_keys, workdir)

    failures = evaluate(
        result,
        args.version,
        require_model=args.require_model,
        require_note=args.require_note,
        min_chat_entries=args.min_chat_entries,
    )
    installed = result.get("installed_version")
    pkg_path = (result.get("package_path") or "").replace("\\", "/")

    log("")
    log(f"  installed version : {installed}")
    log(f"  imported from     : {pkg_path}")
    log(f"  chat entries      : {result.get('chat_entries')}")
    for name in models:
        m = result["models"][name]
        log(
            f"  {name:20} -> {m['key']}  {m['promptPrice']}/{m['completionPrice']} "
            f"cr={m['cacheReadPrice']} cw={m['cacheWritePrice']}"
        )

    if failures:
        raise VerificationError("release verification FAILED:\n  - " + "\n  - ".join(failures))
    log("")
    log(f"Release {args.version} verified against the published artifact.")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--version", required=True, help="published version, e.g. 1.27.0")
    p.add_argument(
        "--require-model",
        action="append",
        default=[],
        type=parse_model_spec,
        metavar="NAME[=PROMPT/COMPLETION]",
        help="model that must resolve to its own entry, optionally " "asserting its per-1k prices",
    )
    p.add_argument(
        "--require-note",
        action="append",
        default=[],
        type=parse_note_spec,
        metavar="KEY=SUBSTRING",
        help="entry whose note must contain this text",
    )
    p.add_argument("--min-chat-entries", type=int, default=None)
    p.add_argument(
        "--index-timeout",
        type=int,
        default=600,
        help="seconds to wait for the simple index (default 600)",
    )
    p.add_argument(
        "--workdir",
        default=None,
        help="where to build the venv (default: a fresh temp dir). " "Never the repository root.",
    )
    args = p.parse_args(argv)

    try:
        verify(args)
    except VerificationError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as e:
        print(f"\ncommand failed: {e}\n{e.stderr}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
