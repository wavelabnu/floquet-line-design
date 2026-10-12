"""Open-end (stub far-end) models for the chain model registry.

An :class:`OpenEndModel` is the **single source** for an
OPEN-terminated cell's stub far end, consumed by:

* the **emitter** -- :meth:`emit` renders the open end into the device spec;
* the **analytical leg** -- :meth:`analytical_termination` returns the
  ``Z_load(omega)`` the harmonic-ABCD stub recursion anchors at.

Three registered variants:

* :class:`IdealOpen` -- the current reduced model: the stub MLIN end simply dangles
  (no element), an *ideal* open. :meth:`emit` returns no elements and
  :meth:`analytical_termination` returns ``None`` so the analytical leg keeps its
  unchanged open-load-limit path -- the recovery-gate behaviour is preserved exactly.
* :class:`MlefOpen` -- the *Mathematica* convention: the open-end fringe. The
  emitter renders an ADS ``MLEF`` open-end-effect element appended to the stub MLIN
  (the emitter-accuracy pass verified ``MLEF:<id> <n> Subst="…" W=… mm L=0 mm`` is the
  element that supplies the fringe). The analytical leg models the same fringe as the
  Kirschning-Franz open-end capacitance ``C_end``
  (:func:`tvtl.microstrip.discontinuities.open_end_capacitance_f`) presented as a
  shunt load ``Z_load(omega) = 1/(j*omega*C_end)``. Both derive from the **same** stub
  width + substrate.
* :class:`MlocOpen` -- the reference *ADS* convention: the ADS ``MLOC`` open-circuited
  stub. The emitter-accuracy pass verified (ADS 2026 U1.2) that flat-deck ``MLOC`` is
  electrically an **ideal open** -- it adds *no* open-end fringe over a dangling MLIN
  (S11 phase delta 0.0000 deg, 2-11 GHz). :meth:`emit` therefore appends an ``MLOC``
  element with ``L=0`` at the stub far end (so the ADS-side deck literally contains the
  ``MLOC`` the reference ADS workflow uses) but :meth:`analytical_termination` returns ``None``
  -- the analytical leg uses the same ideal-open limit as :class:`IdealOpen` (no
  ``C_end``). ``MLOC`` vs ``MLEF`` is thus a clean ideal-open-vs-fringe swap at one node.

Adding a new open-end variant = one ``@register_open_end("kind")`` registration (see
:mod:`.registry`); no call site changes.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from tvtl.materials import Substrate
from tvtl.microstrip.discontinuities import open_end_capacitance_f
from tvtl.microstrip.models import hammerstad_kirschning

from ..device_spec import Element, ElementPin, Node
from .deprecation import warn_deprecated_termination

__all__ = ["OpenEndModel", "IdealOpen", "MlefOpen", "MlocOpen"]


class OpenEndModel(ABC):
    """Interface for a stub far-end (OPEN cells)."""

    #: registry key (set by each concrete variant)
    kind: str = ""

    def extra_nodes(self, k: int, end_node: str) -> list[Node]:
        """Internal node declarations this open-end introduces for cell ``k``.

        Default: none (the fringe element is a 1-port at the existing stub end).
        """

        return []

    @abstractmethod
    def emit(self, k: int, *, end_node: str, width_m: float, substrate_name: str) -> list[Element]:
        """Render the open-end elements for cell ``k`` at the stub far end ``end_node``."""

    def analytical_termination(
        self, *, width_m: float, substrate: Substrate, f_ref_Hz: float
    ) -> Callable[[float], complex] | None:
        """``omega -> Z_load(omega)`` for the analytical stub recursion.

        ``None`` (the default, used by :class:`IdealOpen`) means *use the analytical
        leg's unchanged open-load-limit path*. A non-``None`` callable overrides it.
        """

        return None


@dataclass(frozen=True)
class IdealOpen(OpenEndModel):
    """Reduced model: an ideal open (the stub MLIN end dangles; no element).

    .. deprecated::
        A4 standardization. This ideal (no-fringe) open is DEPRECATED in favour of
        the standardized :class:`MlefOpen` (kind ``mlef``). It stays functional for
        the byte-identical anchor and legacy decks (and the cell's inline ideal-open
        *default* path, which does not instantiate this class, is unchanged).
    """

    kind: str = "ideal"

    def __post_init__(self) -> None:
        warn_deprecated_termination(
            "IdealOpen", self.kind,
            standard_class="MlefOpen", standard_kind="mlef",
        )

    def emit(self, k: int, *, end_node: str, width_m: float, substrate_name: str) -> list[Element]:
        return []


@dataclass(frozen=True)
class MlefOpen(OpenEndModel):
    """Open-end fringe: emitter renders ``MLEF``; analytical adds ``C_end``.

    **Standard OPEN-end termination (A4).** This distributed open-end-effect element
    is the standardized open-end model; the ideal :class:`IdealOpen` / :class:`MlocOpen`
    are deprecated in its favour. It does not warn.

    The emitter appends an ADS ``MLEF`` open-end-effect element (1-port) at the stub
    far end. The analytical leg models the matching Kirschning-Franz open-end
    capacitance as a shunt load.
    """

    kind: str = "mlef"

    def emit(self, k: int, *, end_node: str, width_m: float, substrate_name: str) -> list[Element]:
        return [
            Element(
                id=f"MLEF_C{k}", type="MLEF",
                pins=(ElementPin(pin=1, node=end_node),),
                parameters={"w_m": float(width_m), "l_m": 0.0,
                            "substrate_ref": substrate_name},
            )
        ]

    def analytical_termination(
        self, *, width_m: float, substrate: Substrate, f_ref_Hz: float
    ) -> Callable[[float], complex] | None:
        z0_arr, eps_eff_arr = hammerstad_kirschning(
            float(width_m), substrate.copper_thickness_m, substrate.thickness_m,
            substrate.epsilon_r, [float(f_ref_Hz)],
        )
        c_end = float(open_end_capacitance_f(
            float(width_m), float(z0_arr[0]), float(eps_eff_arr[0]), substrate,
        ))
        if c_end <= 0.0:
            return None  # degenerate -> behave as the ideal open

        def _z(omega: float) -> complex:
            w = float(omega)
            if w == 0.0:
                return complex("inf")  # capacitor is open at DC -> ideal open limit
            return 1.0 / complex(0.0, w * c_end)

        return _z


@dataclass(frozen=True)
class MlocOpen(OpenEndModel):
    """ADS ``MLOC`` open-circuited stub: an *ideal* open (no fringe), emitted explicitly.

    The emitter appends an ``MLOC`` element with ``L=0`` at the stub far end so the
    ADS-side deck literally contains the open element the reference ADS workflow uses; verified
    electrically identical to a dangling open (no open-end fringe). The analytical leg
    uses the same ideal-open limit as :class:`IdealOpen` (``analytical_termination``
    returns ``None`` -> no ``C_end``), so ``MLOC`` is the ADS-convention counterpart of
    the Mathematica-convention :class:`MlefOpen` fringe.

    .. deprecated::
        A4 standardization. This ideal-open ADS stub is DEPRECATED in favour of the
        standardized :class:`MlefOpen` (kind ``mlef``), which carries the physical
        open-end fringe. It stays functional (the ADS-convention configs still build it).
    """

    kind: str = "mloc"

    def __post_init__(self) -> None:
        warn_deprecated_termination(
            "MlocOpen", self.kind,
            standard_class="MlefOpen", standard_kind="mlef",
        )

    def emit(self, k: int, *, end_node: str, width_m: float, substrate_name: str) -> list[Element]:
        return [
            Element(
                id=f"MLOC_C{k}", type="MLOC",
                pins=(ElementPin(pin=1, node=end_node),),
                parameters={"w_m": float(width_m), "l_m": 0.0,
                            "substrate_ref": substrate_name},
            )
        ]

    # analytical_termination inherited (returns None) -> ideal-open limit, no C_end.


