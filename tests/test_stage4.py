"""Unit tests for Stage 4 (cross-validation), base Python (no ADS)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tvtl.floquet.harmonic_abcd import harmonic_frequencies
from tvtl.optimization.stage1_per_cell import coarse_screen, refine_topk
from tvtl.optimization.stage2_main_tl import match_z_tl
from tvtl.optimization.stage3_matching import (
    edge_impedance,
    quarter_wave_design,
    select_matching,
    single_stub_design,
)
from tvtl.optimization.stage4_validation import (
    CompleteDesign,
    analytical_characterization,
)
from tvtl.optimization.substrate import ROGERS_4003C
from tvtl.optimization.topology_generator import linear_families
from tvtl.varactors import RealisticVaractor
from tvtl_ads_builder_v2.analytical import level10_realistic_param_gain
from tvtl_ads_builder_v2.analytical.matching_blocks import matching_section_harmonic_blocks
from tvtl_ads_builder_v2.parity.level10_ideal_c import open_circuit_pump_voltage
from tvtl_ads_builder_v2.parity.optimized_design import optimized_control_port_spec

_VAR = RealisticVaractor(
    CJ0_f=0.3e-12, VJ_v=3.5, M=1.7, CP_f=0.0, L_pkg_h=0.3e-9, R_series_ohm=0.5
)


@pytest.fixture(scope="module")
def realizable_design():
    res = coarse_screen(3.1e9, _VAR, substrate=ROGERS_4003C, families=linear_families(),
                        n_samples=96, seed=0, top_k=10)
    top = refine_topk(res.linear_topk, 3.1e9, _VAR, substrate=ROGERS_4003C, top_n=10, maxiter=80)
    for c in top:
        mtl = match_z_tl(c, _VAR, substrate=ROGERS_4003C)
        if mtl.realizable:
            z_in, z_out = edge_impedance(mtl, _VAR, 15, substrate=ROGERS_4003C)
            sm_in = select_matching(z_in, "input", 3.1e9, substrate=ROGERS_4003C)
            sm_out = select_matching(z_out, "output", 3.1e9, substrate=ROGERS_4003C)
            return CompleteDesign(
                stub=c, main_tl=mtl, input_match=sm_in.single_stub,
                output_match=sm_out.single_stub, varactor=_VAR, n_cells=15, f0_Hz=3.1e9,
            )
    pytest.skip("no realizable candidate found in the reduced screen")


def test_matching_blocks_counts_and_order():
    qw = quarter_wave_design(11.0 + 0j, 3.1e9, substrate=ROGERS_4003C)
    ss = single_stub_design(11.0 - 2j, 3.1e9, substrate=ROGERS_4003C)
    omega_n = harmonic_frequencies(2 * math.pi * 3.1e9, 2 * math.pi * 6.2e9, 6)
    sub = ROGERS_4003C.to_tvtl_substrate()
    assert len(matching_section_harmonic_blocks(qw, omega_n, sub, side="input")) == 1
    bi = matching_section_harmonic_blocks(ss, omega_n, sub, side="input")
    bo = matching_section_harmonic_blocks(ss, omega_n, sub, side="output")
    assert len(bi) == 2 and len(bo) == 2
    # input is shunt-first, output series-first -> first blocks differ
    # (shunt block has identity A; series block has non-identity A)
    assert not np.allclose(bi[0][0], bo[0][0])
    assert matching_section_harmonic_blocks(None, omega_n, sub, side="input") == []


def test_analytical_characterization_well_formed(realizable_design):
    rep = analytical_characterization(realizable_design)
    for p in (-6.0, -3.0, 0.0):
        e = rep.analytical[p]
        for key in ("coherent_peak", "coherent_at0", "mix10", "psa_swing"):
            assert key in e and math.isfinite(e[key])
    assert set(rep.psa_curve.keys()) == {0.0, 72.0, 144.0, 216.0, 288.0}
    assert all("analytical" in v for v in rep.psa_curve.values())
    assert len(rep.landscape) > 5
    assert rep.passed is None  # analytical-only


def test_no_matching_is_bit_for_bit(realizable_design):
    """A design with no matching: Stage-4 analytical == the bare leg (None blocks)."""
    d = CompleteDesign(
        stub=realizable_design.stub, main_tl=realizable_design.main_tl,
        input_match=None, output_match=None, varactor=_VAR, n_cells=15, f0_Hz=3.1e9,
    )
    rep = analytical_characterization(d, pumps_dBm=(-3.0,), phases_deg=(0.0,), landscape_GHz=(3.1,))
    spec = optimized_control_port_spec(
        topology=d.stub.topology, params=d.stub.params_array, varactor=_VAR,
        substrate=ROGERS_4003C, n_cells=15, main_width_m=d.main_tl.w_main_m,
        f_signal_Hz=3.1e9, f_pump_Hz=6.2e9, pump_power_dBm=-3.0,
    )
    direct = level10_realistic_param_gain(
        spec, _VAR, control_port_vpump_v=open_circuit_pump_voltage(-3.0)
    )
    assert rep.analytical[-3.0]["coherent_peak"] == pytest.approx(direct["coherent_peak_dB"], abs=1e-9)
    assert rep.analytical[-3.0]["mix10"] == pytest.approx(direct["param_gain_dB"], abs=1e-9)


def test_matching_changes_result(realizable_design):
    """A matched design differs from the unmatched one (matching is folded in)."""
    d_un = CompleteDesign(
        stub=realizable_design.stub, main_tl=realizable_design.main_tl,
        input_match=None, output_match=None, varactor=_VAR, n_cells=15, f0_Hz=3.1e9,
    )
    r_un = analytical_characterization(d_un, pumps_dBm=(-3.0,), phases_deg=(0.0,), landscape_GHz=(3.1,))
    r_m = analytical_characterization(realizable_design, pumps_dBm=(-3.0,), phases_deg=(0.0,), landscape_GHz=(3.1,))
    assert r_un.analytical[-3.0]["coherent_at0"] != pytest.approx(
        r_m.analytical[-3.0]["coherent_at0"], abs=1e-6
    )


