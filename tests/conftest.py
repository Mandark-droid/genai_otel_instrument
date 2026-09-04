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

sys.setrecursionlimit(10000)


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
    except ImportError:  # pragma: no cover - pydantic is a hard dependency
        pass
