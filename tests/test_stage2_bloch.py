"""Unit tests for Stage 2 (Bloch-impedance match), base Python."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tvtl.microstrip.models import hammerstad_kirschning
from tvtl.optimization.stage1_per_cell import coarse_screen
from tvtl.optimization.stage2_main_tl import (
    bloch_impedance,
    match_z_tl,
    static_cell_abcd,
)
from tvtl.optimization.substrate import ROGERS_4003C, microstrip_width_for_z0
from tvtl.optimization.topology_generator import linear_families
from tvtl.varactors import RealisticVaractor

_VAR = RealisticVaractor(
    CJ0_f=0.3e-12, VJ_v=3.5, M=1.7, CP_f=0.0, L_pkg_h=0.3e-9, R_series_ohm=0.5
)


def _screened(seed=0, n=128):
    res = coarse_screen(
        3.1e9, _VAR, substrate=ROGERS_4003C, families=linear_families(),
        n_samples=n, seed=seed, top_k=10,
    )
    return [c for c in res.scored if c.topology.is_linear and c.a1_score > 0]


@pytest.mark.parametrize("z0", [25.0, 40.0, 50.0, 75.0, 95.0])
def test_inverse_hammerstad_roundtrip(z0):
    w = microstrip_width_for_z0(z0, ROGERS_4003C, 3.1e9)
    m = ROGERS_4003C.stub_substrate_mapping()
    z0_back, _ = hammerstad_kirschning(w, m["t_m"], m["h_m"], m["eps_r"], [3.1e9])
    assert float(z0_back[0]) == pytest.approx(z0, rel=1e-4)


def test_inverse_hammerstad_out_of_band_raises():
    with pytest.raises(ValueError):
        microstrip_width_for_z0(500.0, ROGERS_4003C, 3.1e9)  # unrealizably high


def test_bloch_weak_branch_approaches_z_tl():
    """The weakest-coupling screened stub has Z_B ~ Z_TL (pure-line limit)."""
    cands = _screened()
    cands.sort(key=lambda c: c.a1_score)
    weak = cands[0]
    zb = bloch_impedance(weak, _VAR, 50.0, substrate=ROGERS_4003C)
    assert abs(zb.real - 50.0) < 10.0  # near the line impedance
    assert zb.real > 0.0


def test_match_weak_stub_matches_50():
    cands = _screened()
    cands.sort(key=lambda c: c.a1_score)
    weak = cands[0]
    d = match_z_tl(weak, _VAR, substrate=ROGERS_4003C)
    assert d.realizable
    assert d.matched
    assert abs(d.bloch_z_ohm.real - 50.0) <= 1.0


def test_match_strong_stub_realizable_but_unmatched():
    """High-A1 (heavily-loaded) stub: realizable Z_TL but Z_B != 50 (-> Stage 3)."""
    cands = _screened()
    cands.sort(key=lambda c: c.a1_score, reverse=True)
    strong = cands[0]
    d = match_z_tl(strong, _VAR, substrate=ROGERS_4003C)
    # the loaded line cannot reach 50 ohm Bloch by main width alone
    assert not d.matched
    assert d.bloch_z_ohm.real < 50.0
    assert d.note  # carries the boundary explanation


def test_static_cell_finite():
    cands = _screened()
    cell = static_cell_abcd(cands[0], _VAR, 50.0, substrate=ROGERS_4003C)
    assert cell.shape == (2, 2)
    assert np.all(np.isfinite(cell))
    # reciprocal: det(ABCD) ~ 1
    det = cell[0, 0] * cell[1, 1] - cell[0, 1] * cell[1, 0]
    assert abs(det) == pytest.approx(1.0, abs=0.05)


