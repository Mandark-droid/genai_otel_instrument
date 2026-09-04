"""Guards the conftest workaround for `patch.dict("sys.modules")` eviction.

`patch.dict` restores `sys.modules` on exit by clearing it and re-applying the
snapshot taken on entry, so a submodule imported for the first time *inside* the
block is evicted when the block ends. Most instrumentor tests in this suite use
that pattern to simulate an SDK being present or absent, so the hazard is
everywhere.

It stayed invisible for a long time because it only bites when a package imports
submodules lazily *and* something later depends on them. Pydantic does both:
`pydantic.root_model` loads on first use, and any pydantic-based library
building a generic model then reads `sys.modules[created_model.__module__]`.
With `mcp` installed, that produced a bare `KeyError: 'pydantic.root_model'`
hundreds of tests away from the langchain test that caused it -- and only in a
full-suite run, never in isolation.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

LAZY_MODULES = ("pydantic.root_model", "pydantic.main", "pydantic.fields")


@pytest.mark.parametrize("module_name", LAZY_MODULES)
def test_lazy_submodule_is_preloaded(module_name):
    """conftest imports these before any test, so they are always present.

    Being in `sys.modules` before a test starts is what puts them inside the
    snapshot `patch.dict` restores.
    """
    assert module_name in sys.modules, (
        f"{module_name} must be imported by tests/conftest.py before any test runs, "
        "or patch.dict('sys.modules') will evict it the first time a test imports it."
    )


@pytest.mark.parametrize("module_name", LAZY_MODULES)
def test_module_survives_a_patch_dict_block(module_name):
    """The eviction that broke the suite must no longer happen."""
    with patch.dict("sys.modules", {"some_fake_sdk": MagicMock()}):
        assert module_name in sys.modules

    assert module_name in sys.modules, (
        f"{module_name} was evicted by patch.dict('sys.modules'). "
        "The preload in tests/conftest.py is the guard against this."
    )


def test_pydantic_can_still_build_a_generic_model_after_patch_dict():
    """The end-to-end symptom: pydantic generics need their module in sys.modules.

    This is the operation that actually raised `KeyError: 'pydantic.root_model'`
    once the module had been evicted.
    """
    from typing import Generic, TypeVar

    from pydantic import BaseModel

    T = TypeVar("T")

    with patch.dict("sys.modules", {"another_fake_sdk": MagicMock()}):
        pass

    class Box(BaseModel, Generic[T]):
        value: T

    assert Box[int](value=1).value == 1
