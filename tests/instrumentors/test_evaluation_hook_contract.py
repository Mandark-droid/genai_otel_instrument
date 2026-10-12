"""Every instrumentor's evaluation hooks take what the shared wrapper passes.

The wrapper calls ``_extract_prompt_for_eval(kwargs, operation)``. An override
that still takes ``kwargs`` alone raises ``TypeError`` inside the pre-call
check, which is treated as a set-up fault: the request is sent unchecked.
"""

import importlib
import inspect
import pkgutil

import pytest

import genai_otel.instrumentors as instrumentors_package
from genai_otel.instrumentors.base import BaseInstrumentor


def _instrumentor_classes():
    found = {}
    for module_info in pkgutil.iter_modules(instrumentors_package.__path__):
        try:
            module = importlib.import_module(
                instrumentors_package.__name__ + "." + module_info.name
            )
        except Exception:  # an optional dependency this environment lacks
            continue
        for _, member in inspect.getmembers(module, inspect.isclass):
            if issubclass(member, BaseInstrumentor) and member is not BaseInstrumentor:
                found[member.__qualname__] = member
    return sorted(found.values(), key=lambda cls: cls.__qualname__)


CLASSES = _instrumentor_classes()


def _accepts(cls, name, *call_args):
    declared = inspect.getattr_static(cls, name)
    if isinstance(declared, (staticmethod, classmethod)):
        function, args = declared.__func__, call_args
        if isinstance(declared, classmethod):
            args = (cls,) + call_args
    else:
        function, args = declared, (None,) + call_args
    try:
        inspect.signature(function).bind(*args)
    except TypeError:
        return False
    return True


def test_instrumentor_classes_were_found():
    assert len(CLASSES) > 20


@pytest.mark.parametrize("cls", CLASSES, ids=lambda cls: cls.__qualname__)
def test_prompt_hook_takes_the_operation(cls):
    assert _accepts(cls, "_extract_prompt_for_eval", {}, "chat")


@pytest.mark.parametrize("cls", CLASSES, ids=lambda cls: cls.__qualname__)
def test_response_hook_takes_the_result(cls):
    assert _accepts(cls, "_extract_response_for_eval", object())
