"""Harmonic-ABCD blocks for Stage-3 matching networks.

Builds the per-harmonic block ABCD of a distributed-microstrip matching section
(quarter-wave transformer or single-stub) so the Stage-4 analytical cross-
validation leg (:func:`.realistic_c_gain.level10_realistic_param_gain`) can
cascade the *same* matching the device-spec generator
(:func:`...parity.optimized_design.optimized_control_port_spec`) emits to ADS --
keeping the matched-cascade cross-validation honest.

A matching network is a passive, linear, time-invariant 2-port, so it is
block-diagonal in the Floquet harmonic basis: a series microstrip line is the
lossy-TL harmonic block; a shunt open/short stub is a diagonal shunt-admittance
block ``(I, 0, diag(Y_stub(omega_n)), I)``. Orientation matches the spec:

* input matching (source -> cascade): ``[shunt_stub, series_line]`` for a
  single stub (the stub sits at the 50-ohm source node), ``[series_line]`` for a
  quarter-wave transformer;
* output matching (cascade -> load): the reverse order.
"""

from __future__ import annotations

import numpy as np

from tvtl.materials import Substrate
from .floquet_stub import StubSegment, Y_stub_at_freq
from .parametric_gain import _lossy_tl_harmonic_block

__all__ = [
    "series_line_harmonic_block",
    "shunt_stub_harmonic_block",
    "matching_section_harmonic_blocks",
]


def _substrate_mapping(substrate: Substrate) -> dict[str, float]:
    """Mapping (eps_r, h_m, t_m) for ``Y_stub_at_freq`` from a tvtl Substrate."""

    return {
        "eps_r": float(substrate.epsilon_r),
        "h_m": float(substrate.thickness_m),
        "t_m": float(substrate.copper_thickness_m),
    }


def series_line_harmonic_block(
    width_m: float, length_m: float, omega_n: np.ndarray, substrate: Substrate
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Harmonic-ABCD of a series microstrip line (the lossy-TL block)."""

    return _lossy_tl_harmonic_block(float(width_m), float(length_m), omega_n, substrate)


def shunt_stub_harmonic_block(
    width_m: float,
    length_m: float,
    omega_n: np.ndarray,
    substrate: Substrate,
    *,
    termination: str = "open",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Harmonic-ABCD ``(I, 0, diag(Y_stub), I)`` of a shunt open/short stub.

    ``Y_stub(omega_n)`` is the input admittance of a single open/short microstrip
    stub segment (reusing :func:`.floquet_stub.Y_stub_at_freq`), evaluated per
    harmonic (it handles negative harmonic frequencies via ``|omega|`` + the
    odd ``tan``).
    """

    mapping = _substrate_mapping(substrate)
    seg = (StubSegment(width_m=float(width_m), length_m=float(length_m)),)
    y = np.array(
        [complex(Y_stub_at_freq(float(w), seg, mapping, termination=termination)) for w in omega_n],
        dtype=np.complex128,
    )
    size = omega_n.shape[0]
    identity = np.eye(size, dtype=np.complex128)
    zero = np.zeros((size, size), dtype=np.complex128)
    return identity, zero, np.diag(y), identity


def matching_section_harmonic_blocks(
    section,
    omega_n: np.ndarray,
    substrate: Substrate,
    *,
    side: str,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Harmonic-ABCD block list for a matching section, oriented for ``side``.

    Parameters
    ----------
    section : matching section
        Object with ``.topology`` in ``{"quarter_wave", "single_stub"}`` and
        ``.elements`` (each ``(kind, w_m, l_m, z0_ohm)``), e.g.
        :class:`tvtl.optimization.stage3_matching.MatchingSection`. ``None`` ->
        empty list.
    omega_n : numpy.ndarray
        Harmonic angular frequencies (must match the leg's truncation N).
    substrate : tvtl.materials.Substrate
    side : str
        ``"input"`` (source -> cascade) or ``"output"`` (cascade -> load); sets
        the cascade order of the single-stub (shunt-first vs series-first).

    Returns
    -------
    list of (A, B, C, D)
        Blocks to cascade (in returned order) on the given side.
    """

    if section is None:
        return []
    if side not in ("input", "output"):
        raise ValueError(f"side must be 'input' or 'output' (got {side!r}).")

    series = shunt = None
    shunt_term = "open"
    for el in section.elements:
        kind, w, l, _z0 = el
        if kind == "series_line":
            series = (float(w), float(l))
        elif kind in ("shunt_open_stub", "shunt_short_stub"):
            shunt = (float(w), float(l))
            shunt_term = "open" if kind == "shunt_open_stub" else "short"

    if section.topology == "quarter_wave":
        if series is None:
            raise ValueError("quarter_wave section has no series_line element.")
        return [series_line_harmonic_block(series[0], series[1], omega_n, substrate)]

    if section.topology == "single_stub":
        if series is None or shunt is None:
            raise ValueError("single_stub section missing series/shunt element.")
        series_block = series_line_harmonic_block(series[0], series[1], omega_n, substrate)
        shunt_block = shunt_stub_harmonic_block(
            shunt[0], shunt[1], omega_n, substrate, termination=shunt_term
        )
        # input: source -> [shunt @ source, series] -> cascade.
        # output: cascade -> [series, shunt @ load] -> load (the reverse).
        return [shunt_block, series_block] if side == "input" else [series_block, shunt_block]

    raise ValueError(f"unknown matching topology {section.topology!r}.")


