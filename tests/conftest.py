"""Global test configuration.

Increases the Python recursion limit to handle deep import chains from heavy
SDKs (e.g., llama_index, vertexai, litellm) when running the full test suite.

Note: On Windows, the C thread stack size (default 1MB) may still cause stack
overflows even with a higher Python limit. Run tests in batches if needed:
    pytest tests/evaluation/ tests/mcp_instrumentors/ tests/test_*.py
    pytest tests/instrumentors/
"""

import importlib
import sys

import pytest

sys.setrecursionlimit(10000)


_EVALUATION_STATE = (
    "_pii_detector",
    "_toxicity_detector",
    "_bias_detector",
    "_prompt_injection_detector",
    "_restricted_topics_detector",
    "_hallucination_detector",
    "_blocking_active",
    "_blocking_policies",
)


@pytest.fixture(autouse=True)
def _evaluation_state_stays_in_its_test():
    """Keep one test's detectors, and the work they do, out of the next test.

    Detectors are registered on ``BaseInstrumentor`` itself, for the whole
    process, and nothing unregisters them: a test that builds an evaluation span
    processor leaves its detectors configured for every test that runs after it.
    That used to pass unnoticed. Now that a call with a detector configured
    hands its span to a worker thread, a later test that expects its span to
    have ended when the call returns would fail for a reason in another file.

    So what was registered before a test is put back after it, and detectors
    still running on a worker are waited for first.
    """
    base = sys.modules.get("genai_otel.instrumentors.base")
    saved = (
        {name: getattr(base.BaseInstrumentor, name) for name in _EVALUATION_STATE}
        if base is not None
        else None
    )
    yield
    base = sys.modules.get("genai_otel.instrumentors.base")
    if base is None:
        return
    base.drain_evaluations(timeout=30)
    for name in _EVALUATION_STATE:
        default = frozenset() if name == "_blocking_policies" else None
        if name == "_blocking_active":
            default = False
        setattr(base.BaseInstrumentor, name, saved[name] if saved else default)


# ---------------------------------------------------------------------------
# Keep lazily-imported third-party submodules out of patch.dict's blast radius.
#
# `patch.dict("sys.modules", {...})` -- which most instrumentor tests use to
# simulate an SDK being present or absent -- restores the dict on exit by
# *clearing it and re-applying the snapshot* taken on entry. A submodule
# imported for the first time inside such a block is therefore evicted when the
# block ends, even though nothing asked for it to be removed.
#
# That is invisible until a package imports its submodules lazily. Pydantic
# does: `pydantic.root_model` loads on first use, which happens inside the
# langchain instrumentor tests. Afterwards `pydantic` is still imported but
# `pydantic.root_model` is gone, and the next pydantic-based library to build a
# generic model dies on `sys.modules[created_model.__module__]` with a bare
# `KeyError: 'pydantic.root_model'` -- hundreds of tests away from the cause,
# and only when a package like `mcp` is installed to trigger it.
#
# Importing them here, before any test runs, puts them in the snapshot that
# patch.dict restores, so they survive. Restoring them afterwards cannot work:
# a module first imported inside the block was never in the "before" state to
# restore from.
for _lazy_module in ("pydantic.root_model", "pydantic.main", "pydantic.fields"):
    try:
        importlib.import_module(_lazy_module)
    except ImportError:
        # Pydantic is not a dependency of this library; it arrives only via
        # optional SDKs. Where it is absent there is nothing to evict, so
        # there is nothing to preload either.
        pass
