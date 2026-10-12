"""Guard: every name a public ``tvtl`` (sub)package advertises must resolve.

Regression test for the phantom-export defect where ``tvtl.abcd`` listed ~17
names in ``__all__`` that were backed by a ``circuit1`` module absent from the
public tree -- so any attribute access (or ``from tvtl.abcd import *``) raised
``ModuleNotFoundError``. This walks the analytical ``tvtl`` package, imports every
public submodule, and asserts each ``__all__`` entry actually resolves, so a
phantom advertisement can never ship again.

The walk is confined to ``tvtl`` (the ADS-free analytical core); importing any of
its submodules requires no Keysight ADS, so this test runs in the default suite.
"""
from __future__ import annotations

import importlib
import pkgutil

import pytest

import tvtl


def _discover():
    """Import ``tvtl`` and every public submodule; collect import failures.

    Returns ``(modules, errors)``. Failures are collected rather than raised so a
    broken submodule surfaces as a clean test failure instead of a collection
    error. ``__main__`` entry-point modules are skipped (importing one would run
    the CLI's module-execution guard machinery, not exercise a public surface).
    """
    modules = [tvtl]
    errors: list[tuple[str, str]] = []

    def _onerror(name: str) -> None:
        errors.append((name, "ImportError while walking package"))

    for info in pkgutil.walk_packages(tvtl.__path__, prefix="tvtl.", onerror=_onerror):
        if info.name.rsplit(".", 1)[-1] == "__main__":
            continue
        try:
            modules.append(importlib.import_module(info.name))
        except Exception as exc:  # noqa: BLE001 - any import failure is a defect to report
            errors.append((info.name, f"{type(exc).__name__}: {exc}"))
    return modules, errors


_MODULES, _IMPORT_ERRORS = _discover()
_MODULES_WITH_ALL = [m for m in _MODULES if getattr(m, "__all__", None)]


def test_all_public_submodules_import():
    assert not _IMPORT_ERRORS, (
        "public tvtl submodules failed to import: "
        + "; ".join(f"{name} -> {detail}" for name, detail in _IMPORT_ERRORS)
    )


def test_walk_covered_public_surfaces():
    # Sanity floor so the guard cannot silently degrade to zero coverage if the
    # walk or import machinery breaks (observed: ~50 modules, ~18 with __all__).
    assert len(_MODULES) > 20
    assert len(_MODULES_WITH_ALL) >= 5


@pytest.mark.parametrize(
    "module", _MODULES_WITH_ALL, ids=[m.__name__ for m in _MODULES_WITH_ALL]
)
def test_public_all_names_resolve(module):
    unresolved = []
    for name in module.__all__:
        try:
            getattr(module, name)
        except Exception as exc:  # noqa: BLE001 - a name in __all__ that raises is a phantom export
            unresolved.append(f"{name} ({type(exc).__name__}: {exc})")
    assert not unresolved, (
        f"{module.__name__}.__all__ advertises names that do not resolve "
        f"(phantom exports): {unresolved}"
    )
