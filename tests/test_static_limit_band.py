"""Swept-band static-weld test for the pump-off harmonic cascade.

Band-swept companion to
``test_floquet_cascade.py::test_static_limit_matches_single_frequency_cascade``.
That test welds the harmonic cascade to the single-frequency static ABCD at one
operating point (f_s = 3.1 GHz). This test repeats the identical construction at
8 signal frequencies spanning 1.0-8.0 GHz (with w_m = 2*w_s at every point) and
applies the same two assertions at each point:

  * the off-diagonal ABCD entries vanish (<= 1e-12) in the harmonic basis when
    dS = 0 (block-diagonal harmonic cascade), and
  * the n = 0 block reproduces the single-frequency static cascade
    ``cascade_abcd([line, shunt(y_branch), line])`` to rel 1e-10 / abs 1e-12.

A failure at any frequency gates the manuscript wording ("at the operating
point" vs "across the band"): the test loosens nothing and reports the failing
frequency and value.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tvtl.floquet.harmonic_abcd import (
    stub_harmonic_abcd,
    tl_section_harmonic_abcd,
    unit_cell_harmonic_abcd,
)
from tvtl.materials import SPEED_OF_LIGHT_M_PER_S
from tvtl.microstrip.models import hammerstad_kirschning
from tvtl.theory.abcd import line_abcd, shunt_admittance_abcd
from tvtl.theory.cascade import cascade_abcd


# Shared parameter block: SMV1245-representative (mirrors test_floquet_cascade).
C0_REF = 7.0e-12
S0_REF = 1.0 / C0_REF
R_REF = 1.5
L_REF = 1.7e-9
N_REF = 5
PHI_REF = 0.0

SUBSTRATE = {"eps_r": 3.38, "h": 1.54e-3, "tan_delta": 0.0027, "copper_thickness": 17.5e-6}
MAIN_SEG_LENGTH = 3.055e-3   # half of paper's p = 6.11 mm
MAIN_SEG_WIDTH = 4.27e-3     # paper's w

# 8-point band sweep over [1.0, 8.0] GHz; w_m = 2*w_s at every point.
BAND_FS_HZ = tuple(float(f) for f in np.linspace(1.0e9, 8.0e9, 8))


def _z0_and_v_phase(width: float, frequency_hz: float) -> tuple[float, float]:
    z0_arr, eps_eff_arr = hammerstad_kirschning(
        width, SUBSTRATE["copper_thickness"], SUBSTRATE["h"], SUBSTRATE["eps_r"],
        np.asarray([frequency_hz], dtype=float),
    )
    z0 = float(z0_arr[0])
    eps_eff = float(eps_eff_arr[0])
    v_phase = SPEED_OF_LIGHT_M_PER_S / math.sqrt(eps_eff)
    return z0, v_phase


@pytest.mark.parametrize("fs_hz", BAND_FS_HZ)
def test_static_limit_matches_single_frequency_cascade_across_band(fs_hz: float) -> None:
    """dS = 0 collapses the harmonic cascade to the static ABCD at every f_s."""

    ws = 2.0 * math.pi * fs_hz
    wm = 2.0 * ws
    ghz = fs_hz / 1e9

    # Build the harmonic cascade for one unit cell at this frequency.
    z0, v_phase = _z0_and_v_phase(MAIN_SEG_WIDTH, fs_hz)
    tl_block = tl_section_harmonic_abcd(MAIN_SEG_LENGTH, z0, v_phase, ws, wm, N_REF)
    stub_block = stub_harmonic_abcd(S0_REF, 0.0, PHI_REF, R_REF, L_REF, ws, wm, N_REF)
    uc_block = unit_cell_harmonic_abcd(tl_block, stub_block, tl_block)
    A, B, C, D = uc_block

    # Off-diagonals must be exactly zero in the harmonic basis when dS = 0.
    size = 2 * N_REF + 1
    off_diag_mask = ~np.eye(size, dtype=bool)
    for name, block in zip("ABCD", (A, B, C, D)):
        max_off = float(np.max(np.abs(block[off_diag_mask])))
        assert max_off <= 1.0e-12, (
            f"f_s = {ghz:.3f} GHz: {name} off-diagonal {max_off:.3e} exceeds 1e-12"
        )

    # Single-frequency reference cascade at f_s.
    beta = ws / v_phase
    line_ref = line_abcd(z0, 1j * beta * MAIN_SEG_LENGTH)
    z_rlc = R_REF + 1j * (ws * L_REF - 1.0 / (ws * C0_REF))
    y_branch = 1.0 / z_rlc
    stub_ref = shunt_admittance_abcd(y_branch)
    uc_ref = cascade_abcd([line_ref, stub_ref, line_ref])

    n0 = N_REF
    assert A[n0, n0] == pytest.approx(uc_ref[0, 0], rel=1.0e-10, abs=1.0e-12), (
        f"f_s = {ghz:.3f} GHz: A n=0 block {A[n0, n0]} vs static {uc_ref[0, 0]}"
    )
    assert B[n0, n0] == pytest.approx(uc_ref[0, 1], rel=1.0e-10, abs=1.0e-12), (
        f"f_s = {ghz:.3f} GHz: B n=0 block {B[n0, n0]} vs static {uc_ref[0, 1]}"
    )
    assert C[n0, n0] == pytest.approx(uc_ref[1, 0], rel=1.0e-10, abs=1.0e-12), (
        f"f_s = {ghz:.3f} GHz: C n=0 block {C[n0, n0]} vs static {uc_ref[1, 0]}"
    )
    assert D[n0, n0] == pytest.approx(uc_ref[1, 1], rel=1.0e-10, abs=1.0e-12), (
        f"f_s = {ghz:.3f} GHz: D n=0 block {D[n0, n0]} vs static {uc_ref[1, 1]}"
    )
