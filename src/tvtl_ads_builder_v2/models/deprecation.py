"""Deprecation of the simpler (non-distributed) termination models.

Termination-model standardization. The **standard** chain terminations are the
*distributed / physical* ADS elements:

* SHORT end -> :class:`~.via.ViagndVia` (via kind ``viagnd``): the ADS ``VIAGND``
  physical via-to-ground.
* OPEN end  -> :class:`~.open_end.MlefOpen` (open kind ``mlef``): the ADS ``MLEF``
  open-end-effect (fringe) element.

The *simpler* terminations -- the lumped, non-distributed vias
(:class:`~.via.LumpedLVia` ``lumped_L`` / :class:`~.via.RlcVia` ``rlc``) and the
ideal-open ends (:class:`~.open_end.IdealOpen` ``ideal`` /
:class:`~.open_end.MlocOpen` ``mloc``) -- are **DEPRECATED**. They remain fully
functional: the byte-identical A.3.1 anchor and the existing regression tests
still build them, and the cell's inline lumped-L/ideal-open *default* (the path
used when no model is passed) is unchanged so every existing deck stays
byte-for-byte. New designs should select the standard pair (``standard.yaml`` /
:data:`STANDARD_VIA_KIND` + :data:`STANDARD_OPEN_END_KIND`).

Instantiating a deprecated model emits a :class:`DeprecationWarning` (a warning,
not a removal). The standard models (``ViagndVia`` / ``MlefOpen``) do not warn.
"""

from __future__ import annotations

import warnings

__all__ = [
    "STANDARD_VIA_KIND",
    "STANDARD_OPEN_END_KIND",
    "DEPRECATED_VIA_KINDS",
    "DEPRECATED_OPEN_END_KINDS",
    "warn_deprecated_termination",
]

#: The standardized via (SHORT-end) model kind.
STANDARD_VIA_KIND = "viagnd"
#: The standardized open-end model kind.
STANDARD_OPEN_END_KIND = "mlef"

#: Via kinds deprecated in favour of :data:`STANDARD_VIA_KIND`.
DEPRECATED_VIA_KINDS = frozenset({"lumped_L", "rlc"})
#: Open-end kinds deprecated in favour of :data:`STANDARD_OPEN_END_KIND`.
DEPRECATED_OPEN_END_KINDS = frozenset({"ideal", "mloc"})


def warn_deprecated_termination(
    class_name: str, kind: str, *, standard_class: str, standard_kind: str
) -> None:
    """Emit the standard :class:`DeprecationWarning` for a simpler termination model.

    Called from a deprecated model's ``__post_init__``. The model stays functional;
    the warning steers new designs to the standardized distributed termination.
    """

    warnings.warn(
        f"{class_name} (kind {kind!r}) is a DEPRECATED non-distributed termination "
        f"model; the standardized termination is {standard_class} (kind "
        f"{standard_kind!r}). {class_name} remains functional for the byte-identical "
        f"anchor and legacy decks, but new designs should use the standard model.",
        DeprecationWarning,
        stacklevel=3,
    )


