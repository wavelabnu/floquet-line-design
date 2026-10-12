"""Unit tests for Stage 3 (matching-network design), base Python."""

from __future__ import annotations

import math

import pytest

from tvtl.optimization.stage3_matching import (
    quarter_wave_design,
    s11_dB,
    select_matching,
    single_stub_design,
)
from tvtl.optimization.substrate import ROGERS_4003C

F0 = 3.1e9


def test_s11_matched_is_very_negative():
    assert s11_dB(50.0) < -100.0  # perfectly matched -> clamped large-negative
    assert s11_dB(50.0 + 0j) < -100.0


def test_s11_mismatch():
    # |Gamma| for 10 ohm vs 50 ohm = 40/60 = 0.667 -> ~ -3.5 dB
    assert s11_dB(10.0) == pytest.approx(20 * math.log10(40.0 / 60.0), abs=0.1)


@pytest.mark.parametrize("z_edge", [11.0 + 0j, 11.0 - 2j, 30.0 + 5j])
def test_quarter_wave_zqw_and_improves(z_edge):
    sec = quarter_wave_design(z_edge, F0, substrate=ROGERS_4003C)
    assert sec.realizable
    z_qw = math.sqrt(50.0 * z_edge.real)
    assert sec.elements[0][3] == pytest.approx(z_qw, rel=1e-6)  # Z_qw stored
    # for a near-real edge the quarter-wave should improve the match markedly
    assert sec.s11_after_dB < sec.s11_before_dB


def test_quarter_wave_exact_for_real_edge():
    sec = quarter_wave_design(11.0 + 0j, F0, substrate=ROGERS_4003C)
    assert sec.s11_after_dB < -40.0  # real edge -> near-perfect quarter-wave match


@pytest.mark.parametrize("z_edge", [11.0 - 2j, 9.0 + 3j, 25.0 - 8j, 80.0 + 10j])
def test_single_stub_matches(z_edge):
    sec = single_stub_design(z_edge, F0, substrate=ROGERS_4003C)
    assert sec.realizable
    assert sec.n_elements == 2  # series line + shunt stub
    assert sec.s11_after_dB < -40.0  # single-stub matches any passive load


def test_select_no_matching_when_already_matched():
    sm = select_matching(50.0 + 0j, "input", F0, substrate=ROGERS_4003C)
    assert not sm.needed
    assert sm.recommended == "none"
    assert sm.quarter_wave is None and sm.single_stub is None


def test_select_recommends_when_mismatched():
    sm = select_matching(11.0 - 2j, "input", F0, substrate=ROGERS_4003C)
    assert sm.needed
    assert sm.quarter_wave is not None and sm.single_stub is not None
    assert sm.recommended in ("quarter_wave", "single_stub")
    # the recommended topology must actually improve the match
    chosen = sm.quarter_wave if sm.recommended == "quarter_wave" else sm.single_stub
    assert chosen.s11_after_dB < sm.s11_unmatched_dB


def test_select_input_output_independent():
    """Different edge impedances per side -> independent decisions."""
    sm_in = select_matching(48.0 + 1j, "input", F0, substrate=ROGERS_4003C)  # ~matched
    sm_out = select_matching(11.0 - 2j, "output", F0, substrate=ROGERS_4003C)  # mismatched
    assert not sm_in.needed
    assert sm_out.needed


