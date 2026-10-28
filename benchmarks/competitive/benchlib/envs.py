"""Create one virtual environment per library from the pinned requirement files.

Uses `uv` when it is on PATH, otherwise the standard library's venv plus pip.
Stdlib only: this module runs under the interpreter that launched the tool.
"""

import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_DIR = os.path.join(ROOT, ".venvs")
REQ_DIR = os.path.join(ROOT, "requirements")

SCRUB_PREFIXES = (
    "OTEL_",
    "GENAI_",
    "TRACELOOP_",
    "LANGFUSE_",
    "OPIK_",
    "OPENLIT_",
    "GALILEO_",
    "PHOENIX_",
    "OPENINFERENCE_",
    "OPENAI_",
    "ARIZE_",
    "COMET_",
    "ANTHROPIC_",
    "AZURE_",
    "GOOGLE_",
    "GEMINI_",
    "MISTRAL_",
    "GROQ_",
    "COHERE_",
    "LITELLM_",
    "SENTRY_",
    "POSTHOG_",
    "AWS_",
    "PYTHON",
    "VIRTUAL_ENV",
    "UV_",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "BENCH_",
)


def clean_env(keep_prefixes=()):
    """The caller's environment without anything that could configure a tracing
    library, carry a credential, or point a client at a real endpoint."""
    env = {}
    for k, v in os.environ.items():
        up = k.upper()
        if up.startswith(SCRUB_PREFIXES) and not up.startswith(tuple(keep_prefixes)):
            continue
        env[k] = v
    env.update(PYTHONHASHSEED="0", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    return env


def set_env_dir(path):
    """Put the environments somewhere else (a short path helps on Windows, see README)."""
    global ENV_DIR
    if path:
        ENV_DIR = os.path.abspath(path)


def python_of(env_name):
    sub = ("Scripts", "python.exe") if sys.platform == "win32" else ("bin", "python")
    return os.path.join(ENV_DIR, env_name, *sub)


def ensure(env_name, python_version=None, log=print):
    """Create the environment if it does not exist yet. Returns its interpreter."""
    py = python_of(env_name)
    marker = os.path.join(ENV_DIR, env_name, ".installed")
    req = os.path.join(REQ_DIR, env_name + ".txt")
    if os.path.exists(py) and os.path.exists(marker):
        return py
    if not os.path.exists(req):
        raise SystemExit("no requirement file for %r (%s)" % (env_name, req))
    os.makedirs(ENV_DIR, exist_ok=True)
    target = os.path.join(ENV_DIR, env_name)
    uv = shutil.which("uv")
    log("creating environment %s (%s)" % (env_name, "uv" if uv else "venv + pip"), flush=True)
    if uv:
        cmd = [uv, "venv", target]
        if python_version:
            cmd += ["--python", python_version]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        subprocess.check_call(
            [uv, "pip", "install", "--python", py, "-r", req],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    else:
        import venv

        venv.EnvBuilder(with_pip=True).create(target)
        subprocess.check_call([py, "-m", "pip", "install", "--quiet", "-r", req])
    with open(marker, "w", encoding="utf-8") as f:
        f.write("ok\n")
    return py
