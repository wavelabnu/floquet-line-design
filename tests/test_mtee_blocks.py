"""Unit tests for the reference-plane MTEE block helper.

Pins :func:`tvtl_ads_builder_v2.analytical.mtee_blocks.tee_reference_plane_single`
and :func:`...tee_harmonic_blocks`: the low-frequency reference-plane
structure, the ``|abcd_left[0,0]| >= 0.7`` validity guard at the model's
high-frequency edge, the NaN-safe fallback past the corrected-notebook
radicand breakdown, and the per-harmonic ideal-junction fallback /
conjugate symmetry in the harmonic-basis assembly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tvtl.floquet.harmonic_abcd import harmonic_frequencies
from tvtl.materials import Substrate
from tvtl_ads_builder_v2.analytical.mtee_blocks import (
    MTEE_VALIDITY_THRESHOLD,
    tee_harmonic_blocks,
    tee_reference_plane_single,
)

# Level10IdealC geometry (Rogers 4003C, paper main/branch widths).
_W_MAIN = 4.27e-3
_W_BRANCH = 1.93e-3


@pytest.fixture()
def substrate() -> Substrate:
    return Substrate(
        epsilon_r=3.38, thickness_m=1.54e-3, copper_thickness_m=17.5e-6,
        loss_tangent=0.0027, conductivity_s_per_m=5.8e7,
    )


def test_single_low_frequency_is_valid_and_near_unity(substrate):
    """At 1 GHz the reference-plane model is valid and close to the ideal junction."""
    abcd_left, abcd_right, j_bt, l_a, l_b, valid = tee_reference_plane_single(
        _W_MAIN, _W_BRANCH, 1.0e9, substrate,
    )
    assert valid is True
    assert np.all(np.isfinite(abcd_left)) and np.all(np.isfinite(abcd_right))
    # |abcd_left[0,0]| close to unity -> reference-plane scaling barely
    # departs from the ideal junction in the operating band.
    assert abs(abcd_left[0, 0]) > 0.95
    assert abs(abcd_left[0, 0]) >= MTEE_VALIDITY_THRESHOLD
    # symmetric main line (w1 == w2) and a non-zero (here inductive) shunt.
    assert l_a == pytest.approx(l_b)
    assert l_a > 0.0
    assert abs(j_bt) > 0.0


def test_single_validity_edge_falls_back_while_finite(substrate):
    """~15 GHz: |aL00| drops below 0.7 -> invalid, but the formula is still finite."""
    abcd_left, _aR, _j, _la, _lb, valid = tee_reference_plane_single(
        _W_MAIN, _W_BRANCH, 15.0e9, substrate,
    )
    assert abs(abcd_left[0, 0]) < MTEE_VALIDITY_THRESHOLD
    assert np.all(np.isfinite(abcd_left))  # the |aL00| guard fired, not NaN
    assert valid is False


def test_single_nan_radicand_is_handled_safely(substrate):
    """~21.7 GHz: the ta/tb sqrt radicand is negative -> NaN, reported invalid, no raise."""
    abcd_left, _aR, _j, _la, _lb, valid = tee_reference_plane_single(
        _W_MAIN, _W_BRANCH, 21.7e9, substrate,
    )
    assert valid is False
    assert not np.all(np.isfinite(abcd_left))  # NaN present, but guarded


def test_harmonic_blocks_all_finite_with_fallback(substrate):
    """Across valid + invalid harmonics the assembled blocks are all finite."""
    ws = 2.0 * math.pi * 5.0e9
    wm = 2.0 * math.pi * 10.0e9
    omega_n = harmonic_frequencies(ws, wm, 3)  # |f| in {5,15,25,35} GHz
    block_left, block_right, j_bt_vec, l_a, l_b = tee_harmonic_blocks(
        _W_MAIN, _W_BRANCH, omega_n, substrate,
    )
    for block in (block_left, block_right):
        for component in block:
            assert np.all(np.isfinite(component))
            # block-diagonal in the harmonic index
            assert np.allclose(component, np.diag(np.diag(component)))
    assert np.all(np.isfinite(j_bt_vec))
    assert l_a > 0.0 and l_b > 0.0


def test_harmonic_blocks_invalid_indices_are_ideal_junction(substrate):
    """Harmonics past the validity edge fall back to identity blocks + zero shunt."""
    ws = 2.0 * math.pi * 5.0e9
    wm = 2.0 * math.pi * 10.0e9
    omega_n = harmonic_frequencies(ws, wm, 3)
    a_l, b_l, c_l, d_l = tee_harmonic_blocks(
        _W_MAIN, _W_BRANCH, omega_n, substrate,
    )[0]
    j_bt_vec = tee_harmonic_blocks(_W_MAIN, _W_BRANCH, omega_n, substrate)[2]
    # n=3 (index 6) is +35 GHz -> NaN radicand -> ideal-junction fallback.
    idx_hi = 6
    assert a_l[idx_hi, idx_hi] == pytest.approx(1.0)
    assert b_l[idx_hi, idx_hi] == pytest.approx(0.0)
    assert c_l[idx_hi, idx_hi] == pytest.approx(0.0)
    assert d_l[idx_hi, idx_hi] == pytest.approx(1.0)
    assert j_bt_vec[idx_hi] == 0
    # n=0 (index 3) is +5 GHz -> valid reference-plane model (non-identity).
    idx_sig = 3
    assert abs(a_l[idx_sig, idx_sig] - 1.0) > 1e-6
    assert abs(j_bt_vec[idx_sig]) > 0.0


def test_harmonic_blocks_negative_frequency_is_conjugate(substrate):
    """A -f harmonic block is the complex conjugate of the +f block."""
    w = 2.0 * math.pi * 2.0e9  # 2 GHz: well inside validity
    omega_n = np.array([w, -w], dtype=float)
    a_l = tee_harmonic_blocks(_W_MAIN, _W_BRANCH, omega_n, substrate)[0][0]
    # index 0 is +w, index 1 is -w
    assert a_l[1, 1] == pytest.approx(np.conjugate(a_l[0, 0]))


