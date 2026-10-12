"""Generative stub-topology generator (Stage 1).

Enumerates the stub-geometry search space as a set of *topology families* --
each a ``(n_segments, via_config, termination)`` combination -- and, per family,
a continuous per-segment ``(width, length)`` parameter space. The generator is
**decoupled from any cascade or device spec**: a topology + parameter vector
maps to a per-harmonic stub admittance via the existing analytical helper
:func:`tvtl_ads_builder_v2.analytical.floquet_stub.Y_stub_at_freq`, so Stage 1
scores a candidate from a single-cell evaluation.

Family enumeration (physical pruning of the 3x3 via x termination grid)
----------------------------------------------------------------------
A via is only meaningful as a short to ground; an open end has no via. Of the
nine ``(via, termination)`` pairs only **five are physical**::

    (NONE,   OPEN)       open-ended stub
    (SINGLE, SHORT)      1-via short to ground
    (DOUBLE, SHORT)      2-via short to ground
    (SINGLE, COMBINED)   1-via short + open shunt stublet at the via node
    (DOUBLE, COMBINED)   2-via short + open shunt stublet at the via node

Five valid ``(via, termination)`` combos x ``n_segments in {1,2,3,4}`` =
**20 families**, splitting as **12 linear (OPEN + SHORT) + 8 COMBINED**. The
ADS cross-validation path covers the 12 linear families; the 8
COMBINED families are analytical-screening-only (the COMBINED
admittance is a parallel via||open-stublet tree the linear spec chain-walker
does not yet trace -- a future extension).

> Note: an earlier sketch said "15 linear / 5 COMBINED"; that sub-split was an
> arithmetic slip. With ``n in {1,2,3,4}`` and the five physical combos the
> only consistent decomposition of 20 is **12 linear + 8 COMBINED**.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator

import numpy as np
from scipy.stats import qmc

from tvtl.microstrip.discontinuities import via_inductance_bahl_H
from tvtl.microstrip.models import hammerstad_kirschning
from tvtl_ads_builder_v2.analytical.floquet_stub import StubSegment, Y_stub_at_freq

from .substrate import SubstrateConfig

__all__ = [
    "ViaConfig",
    "Termination",
    "TopologyFamily",
    "WIDTH_MIN_M",
    "WIDTH_MAX_M",
    "LENGTH_MIN_M",
    "LENGTH_MAX_M",
    "VIA_RADIUS_M",
    "N_SEGMENTS_CHOICES",
    "enumerate_families",
    "linear_families",
    "combined_families",
    "stub_segments",
    "combined_open_segment",
    "param_bounds",
    "is_valid",
    "sample_family",
    "topology_y_stub_fn",
]


class ViaConfig(Enum):
    """Number of parallel vias to ground at the stub termination."""

    NONE = 0
    SINGLE = 1
    DOUBLE = 2


class Termination(Enum):
    """Stub far-end termination."""

    OPEN = "open"
    SHORT = "short"
    COMBINED = "combined"


# Continuous per-segment parameter box (metres). Microstrip-
# realizable widths/lengths on Rogers 4003C.
WIDTH_MIN_M: float = 0.2e-3
WIDTH_MAX_M: float = 5.0e-3
LENGTH_MIN_M: float = 0.5e-3
LENGTH_MAX_M: float = 15.0e-3

# Via barrel radius (matches the paper / Level10ViaSpec default).
VIA_RADIUS_M: float = 0.8e-3

N_SEGMENTS_CHOICES: tuple[int, ...] = (1, 2, 3, 4)

# Realizability Z0 band (Ohm) for the width gate.
_Z0_MIN_OHM: float = 15.0
_Z0_MAX_OHM: float = 130.0

# Reference frequency for the static width/Z0 realizability gate.
_Z0_GATE_FREQ_HZ: float = 5.0e9


def _valid_via_term(via: ViaConfig, term: Termination) -> bool:
    """Return whether a ``(via, termination)`` pair is physically meaningful."""

    if term is Termination.OPEN:
        return via is ViaConfig.NONE
    if term is Termination.SHORT:
        return via in (ViaConfig.SINGLE, ViaConfig.DOUBLE)
    if term is Termination.COMBINED:
        return via in (ViaConfig.SINGLE, ViaConfig.DOUBLE)
    return False


@dataclass(frozen=True)
class TopologyFamily:
    """One ``(n_segments, via, termination)`` stub-topology family.

    The continuous per-segment ``(width, length)`` parameters live outside the
    family (a separate parameter vector); the family fixes the structure.
    """

    n_segments: int
    via: ViaConfig
    termination: Termination

    def __post_init__(self) -> None:
        if self.n_segments not in N_SEGMENTS_CHOICES:
            raise ValueError(
                f"n_segments must be one of {N_SEGMENTS_CHOICES} "
                f"(got {self.n_segments})."
            )
        if not isinstance(self.via, ViaConfig):
            raise TypeError(f"via must be a ViaConfig (got {type(self.via).__name__}).")
        if not isinstance(self.termination, Termination):
            raise TypeError(
                f"termination must be a Termination (got "
                f"{type(self.termination).__name__})."
            )
        if not _valid_via_term(self.via, self.termination):
            raise ValueError(
                f"non-physical via/termination combination: {self.via.name} / "
                f"{self.termination.name} (a via is a short to ground; an open "
                f"end has no via)."
            )

    @property
    def is_linear(self) -> bool:
        """True for the OPEN/SHORT families (the ADS cross-validation path)."""

        return self.termination in (Termination.OPEN, Termination.SHORT)

    @property
    def n_via_parallel(self) -> int:
        """Number of parallel vias (0 for OPEN)."""

        return {ViaConfig.NONE: 0, ViaConfig.SINGLE: 1, ViaConfig.DOUBLE: 2}[self.via]

    @property
    def n_params(self) -> int:
        """Length of the continuous parameter vector for this family.

        ``2 * n_segments`` for the main chain, plus ``2`` for the COMBINED
        open shunt stublet ``(w, l)``.
        """

        extra = 2 if self.termination is Termination.COMBINED else 0
        return 2 * self.n_segments + extra

    @property
    def key(self) -> str:
        """Stable short label, e.g. ``"n2_double_short"``."""

        return f"n{self.n_segments}_{self.via.name.lower()}_{self.termination.name.lower()}"


def enumerate_families() -> list[TopologyFamily]:
    """All 20 physically-valid topology families (12 linear + 8 COMBINED)."""

    families: list[TopologyFamily] = []
    for n in N_SEGMENTS_CHOICES:
        for via in ViaConfig:
            for term in Termination:
                if _valid_via_term(via, term):
                    families.append(TopologyFamily(n_segments=n, via=via, termination=term))
    return families


def linear_families() -> list[TopologyFamily]:
    """The 12 linear (OPEN + SHORT) families -- the ADS cross-validation path."""

    return [f for f in enumerate_families() if f.is_linear]


def combined_families() -> list[TopologyFamily]:
    """The 8 COMBINED families -- analytical-screening-only."""

    return [f for f in enumerate_families() if not f.is_linear]


def _scale_params(unit: np.ndarray, n_params: int) -> np.ndarray:
    """Map a unit-cube vector to the (width, length) box.

    Even indices are widths, odd indices are lengths (the per-segment layout
    ``[w0, l0, w1, l1, ...]`` plus the optional COMBINED stublet ``[w, l]``).
    """

    p = np.empty(int(n_params), dtype=float)
    p[0::2] = WIDTH_MIN_M + unit[0::2] * (WIDTH_MAX_M - WIDTH_MIN_M)
    p[1::2] = LENGTH_MIN_M + unit[1::2] * (LENGTH_MAX_M - LENGTH_MIN_M)
    return p


def param_bounds(topology: TopologyFamily) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(lo, hi)`` box bounds for the family's parameter vector."""

    n = topology.n_params
    lo = np.empty(n, dtype=float)
    hi = np.empty(n, dtype=float)
    lo[0::2] = WIDTH_MIN_M
    hi[0::2] = WIDTH_MAX_M
    lo[1::2] = LENGTH_MIN_M
    hi[1::2] = LENGTH_MAX_M
    return lo, hi


def stub_segments(topology: TopologyFamily, params: np.ndarray) -> tuple[StubSegment, ...]:
    """The main series stub chain (varactor-outward), excluding the COMBINED stublet."""

    p = np.asarray(params, dtype=float)
    return tuple(
        StubSegment(width_m=float(p[2 * i]), length_m=float(p[2 * i + 1]))
        for i in range(topology.n_segments)
    )


def combined_open_segment(
    topology: TopologyFamily, params: np.ndarray
) -> StubSegment | None:
    """The COMBINED open shunt stublet ``(w, l)``, or ``None`` for OPEN/SHORT."""

    if topology.termination is not Termination.COMBINED:
        return None
    p = np.asarray(params, dtype=float)
    base = 2 * topology.n_segments
    return StubSegment(width_m=float(p[base]), length_m=float(p[base + 1]))


def is_valid(
    topology: TopologyFamily,
    params: np.ndarray,
    *,
    substrate: SubstrateConfig,
    f_ref_hz: float = _Z0_GATE_FREQ_HZ,
) -> bool:
    """Validation gate for a (topology, parameter-vector) pair.

    Checks: vector length; per-segment width/length box; microstrip Z0 of every
    segment in ``[15, 130] Ohm`` at ``f_ref_hz`` (realizability); and, for
    short/combined families, the Bahl via precondition (radius < thickness).
    """

    p = np.asarray(params, dtype=float)
    if p.shape != (topology.n_params,):
        return False
    if not np.all(np.isfinite(p)):
        return False
    widths = p[0::2]
    lengths = p[1::2]
    if np.any(widths < WIDTH_MIN_M) or np.any(widths > WIDTH_MAX_M):
        return False
    if np.any(lengths < LENGTH_MIN_M) or np.any(lengths > LENGTH_MAX_M):
        return False
    mapping = substrate.stub_substrate_mapping()
    for w in widths:
        z0_arr, _ = hammerstad_kirschning(
            float(w), mapping["t_m"], mapping["h_m"], mapping["eps_r"], [float(f_ref_hz)]
        )
        z0 = float(z0_arr[0])
        if not (_Z0_MIN_OHM <= z0 <= _Z0_MAX_OHM):
            return False
    if topology.n_via_parallel >= 1 and VIA_RADIUS_M >= mapping["h_m"]:
        return False
    return True


def sample_family(
    topology: TopologyFamily,
    n: int = 300,
    *,
    substrate: SubstrateConfig,
    seed: int = 0,
    f_ref_hz: float = _Z0_GATE_FREQ_HZ,
) -> Iterator[np.ndarray]:
    """Yield validity-filtered Sobol parameter vectors for a family.

    Draws an ``n``-point Sobol sequence (``scipy.stats.qmc.Sobol``, scrambled)
    in the unit cube, scales to the (width, length) box, and yields the vectors
    passing :func:`is_valid`. ``n`` is the sample *budget*; the valid count is
    ``<= n``. The Sobol non-power-of-two balance warning is suppressed (the
    default budget of 300 is not constrained to a power of two).
    """

    n_params = topology.n_params
    sampler = qmc.Sobol(d=n_params, scramble=True, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unit = sampler.random(int(n))
    for u in unit:
        p = _scale_params(u, n_params)
        if is_valid(topology, p, substrate=substrate, f_ref_hz=f_ref_hz):
            yield p


def _termination(
    topology: TopologyFamily,
    params: np.ndarray,
    substrate: SubstrateConfig,
    via_radius_m: float,
) -> object:
    """Build the ``Y_stub_at_freq`` termination for a family + parameters.

    OPEN -> ``"open"``; SHORT -> inductive via ``jwL_via``; COMBINED -> the
    parallel combination of the inductive via and an open shunt stublet tapped
    at the via node (a frequency-dependent callable load).
    """

    mapping = substrate.stub_substrate_mapping()
    if topology.termination is Termination.OPEN:
        return "open"

    l_via = via_inductance_bahl_H(
        substrate_thickness_m=mapping["h_m"],
        via_radius_m=float(via_radius_m),
        n_parallel=topology.n_via_parallel,
    )
    if topology.termination is Termination.SHORT:
        return lambda omega: complex(0.0, float(omega) * l_via)

    # COMBINED: Z_term = Z_via || Z_open_stublet at the via node.
    open_seg = combined_open_segment(topology, params)

    def _combined_load(omega: float) -> complex:
        w = float(omega)
        if w == 0.0:
            return complex(0.0, 0.0)  # via shorts at DC
        y_via = 1.0 / complex(0.0, w * l_via)
        y_open = Y_stub_at_freq(w, (open_seg,), mapping, termination="open")
        y_tot = y_via + y_open
        if not math.isfinite(abs(y_tot)) or abs(y_tot) == 0.0:
            return complex(0.0, 0.0)
        return 1.0 / y_tot

    return _combined_load


def topology_y_stub_fn(
    topology: TopologyFamily,
    params: np.ndarray,
    substrate: SubstrateConfig,
    *,
    via_radius_m: float = VIA_RADIUS_M,
) -> Callable[[float], complex]:
    """Return ``omega -> Y_stub(omega)`` for a (topology, parameters) pair.

    The single source of stub admittance for Stage 1 (A1) and -- via the same
    geometry emitted into a device spec -- the cross-validation cascade. Built
    on the existing :func:`...floquet_stub.Y_stub_at_freq`, so the analytical
    and spec-extracted stub admittances coincide for the same geometry.
    """

    mapping = substrate.stub_substrate_mapping()
    segs = stub_segments(topology, params)
    term = _termination(topology, params, substrate, via_radius_m)

    def _y(omega: float) -> complex:
        return Y_stub_at_freq(float(omega), segs, mapping, termination=term)

    return _y


