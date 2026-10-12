"""Concrete TVTL unit-cell topologies registered into the global registry.

Importing this package triggers registration of every topology submodule so
the global topology registry (:mod:`tvtl.floquet.topology`) gains entries for
the open-stub and lumped-shunt variants in addition to the default
``via_stub_3segment`` registered by :mod:`tvtl.floquet.uc_model`.
"""

from __future__ import annotations

from .open_stub_2segment import OPEN_STUB_2SEGMENT_DESCRIPTOR
from .lumped_shunt_varactor import LUMPED_SHUNT_VARACTOR_DESCRIPTOR

__all__ = [
    "LUMPED_SHUNT_VARACTOR_DESCRIPTOR",
    "OPEN_STUB_2SEGMENT_DESCRIPTOR",
]


