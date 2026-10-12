"""Tests for the realistic-varactor analytical leg.

The load-bearing test is the **bit-for-bit regression** (the ideal-C reduction
contract):
``level10_realistic_param_gain`` with an ideal varactor reproduces
``level10_ideal_c_param_gain`` exactly, on every Level10IdealC geometry and in
both MTEE modes. Also pins the generalized capacitance Toeplitz's reduction to
the existing single-tone function, the package wrap's zero-parasitic no-op, and
finiteness with the realistic SMV1245.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tvtl.floquet.harmonic_abcd import (
    device_coherent_gain,
    tv_capacitor_harmonic_admittance,
    tv_capacitor_harmonic_admittance_general,
)
from tvtl_ads_builder_v2.device_spec import with_var_overrides
from tvtl.varactors import RealisticVaractor
from tvtl_ads_builder_v2.analytical import (
    ideal_c_realised_depth,
    level10_ideal_c_param_gain,
    level10_realistic_param_gain,
)
from tvtl_ads_builder_v2.parity import (
    level10_realistic_c_canonical,
    level10_realistic_c_control_port_canonical,
    make_level10_ideal_c_cascade,
    make_level10_ideal_c_family,
)
from tvtl_ads_builder_v2.parity.level10_ideal_c import open_circuit_pump_voltage
from tvtl_ads_builder_v2.device_spec.validator import validate_device_spec
from tvtl_ads_builder_v2.netlist.elements.sdd2p import sdd2p_tvc_poly_body_rows


# --------------------------------------------------------------------------- #
# Generalized Toeplitz reduces to the existing single-tone function           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("phi", [0.0, 0.7, -1.3])
@pytest.mark.parametrize("with_stub", [False, True])
def test_general_admittance_reduces_to_single_tone(phi, with_stub):
    ws, wm, N = 2 * math.pi * 3.1e9, 2 * math.pi * 6.2e9, 6
    C0, depth = 0.3e-12, 0.3
    cc = [C0, 0.5 * depth * C0 * np.exp(1j * phi)]
    z_stub_fn = (lambda w: 12.0 + 1j * w * 1.0e-9) if with_stub else None
    a = tv_capacitor_harmonic_admittance(C0, depth, phi, ws, wm, N, z_stub_fn=z_stub_fn)
    b = tv_capacitor_harmonic_admittance_general(cc, ws, wm, N, z_stub_fn=z_stub_fn)
    assert np.array_equal(a, b)


def test_general_admittance_rejects_complex_or_negative_C0():
    ws, wm, N = 2 * math.pi * 3.1e9, 2 * math.pi * 6.2e9, 4
    with pytest.raises(ValueError):
        tv_capacitor_harmonic_admittance_general([-1e-12, 0.0], ws, wm, N)
    with pytest.raises(ValueError):
        tv_capacitor_harmonic_admittance_general([1e-12 + 1e-13j, 0.0], ws, wm, N)


# --------------------------------------------------------------------------- #
# Bit-for-bit regression: realistic ideal == ideal-C leg                       #
# --------------------------------------------------------------------------- #
def _ideal_specs():
    return {
        "k0p4@3.1": make_level10_ideal_c_family(
            "k0p4", n_cells=15, f_signal_Hz=3.1e9, f_pump_Hz=6.2e9),
        "k0p5@3.1": make_level10_ideal_c_family(
            "k0p5", n_cells=15, f_signal_Hz=3.1e9, f_pump_Hz=6.2e9),
        "k0p5@5.0": make_level10_ideal_c_family(
            "k0p5", n_cells=15, f_signal_Hz=5.0e9, f_pump_Hz=10.0e9),
        "negctrl": make_level10_ideal_c_cascade(
            stub_scale=1.0, c0_f=0.3e-12, depth=0.3, n_cells=15,
            label="negctrl", f_signal_Hz=3.1e9, f_pump_Hz=6.2e9),
    }


@pytest.mark.parametrize("name", ["k0p4@3.1", "k0p5@3.1", "k0p5@5.0", "negctrl"])
@pytest.mark.parametrize("use_mtee", [True, False])
def test_realistic_ideal_matches_ideal_c_bitforbit(name, use_mtee):
    spec = _ideal_specs()[name]
    depth = ideal_c_realised_depth(spec)
    varactor = RealisticVaractor.ideal(0.3e-12)
    a = level10_ideal_c_param_gain(spec, use_mtee_parasitic=use_mtee)["param_gain_dB"]
    b = level10_realistic_param_gain(
        spec, varactor, m_branch_pump=depth, use_mtee_parasitic=use_mtee
    )["param_gain_dB"]
    assert a == b  # EXACT float equality (the Section-1f contract)


def test_realistic_ideal_matches_full_dict_fields():
    spec = _ideal_specs()["k0p4@3.1"]
    depth = ideal_c_realised_depth(spec)
    a = level10_ideal_c_param_gain(spec)
    b = level10_realistic_param_gain(
        spec, RealisticVaractor.ideal(0.3e-12), m_branch_pump=depth
    )
    assert a["abs_gain_on_dB"] == b["abs_gain_on_dB"]
    assert a["abs_gain_off_dB"] == b["abs_gain_off_dB"]
    assert a["n_cells"] == b["n_cells"]
    assert b["depth_effective"] == depth  # divider unity for ideal


# --------------------------------------------------------------------------- #
# Package wrap zero-parasitic no-op                                           #
# --------------------------------------------------------------------------- #
def test_zero_parasitic_branch_is_bare_admittance():
    # With L_pkg=R_series=0 the realistic leg must skip the inv(inv(...)) wrap.
    # We assert it via the bit-for-bit equality above; here we additionally
    # confirm a tiny but non-zero parasitic perturbs the result (the wrap is
    # actually exercised when parasitics are present).
    spec = _ideal_specs()["k0p4@3.1"]
    depth = ideal_c_realised_depth(spec)
    ideal = level10_realistic_param_gain(
        spec, RealisticVaractor.ideal(0.3e-12), m_branch_pump=depth)["param_gain_dB"]
    perturbed = level10_realistic_param_gain(
        spec,
        RealisticVaractor(CJ0_f=0.3e-12, VJ_v=3.5, M=1.7, CP_f=0.0,
                          L_pkg_h=0.3e-9, R_series_ohm=0.5),
        m_branch_pump=depth,
    )["param_gain_dB"]
    assert ideal != perturbed


# --------------------------------------------------------------------------- #
# Realistic SMV1245: finite & physical                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("f_s", [1.5e9, 3.1e9, 5.0e9])
@pytest.mark.parametrize("use_mtee", [True, False])
def test_realistic_smv1245_finite(f_s, use_mtee):
    var = RealisticVaractor.smv1245()
    spec = level10_realistic_c_canonical(
        varactor=var, stub_scale=1.0, n_cells=15,
        f_signal_Hz=f_s, f_pump_Hz=2.0 * f_s,
    )
    out = level10_realistic_param_gain(
        spec, var, m_branch_pump=0.3, use_mtee_parasitic=use_mtee
    )
    assert math.isfinite(out["param_gain_dB"])
    assert math.isfinite(out["abs_gain_on_dB"])
    assert out["label"] == "analytical_proxy_only"


def test_realistic_smv1245_paper_geometry_near_zero_gain():
    # The analytical leg agrees with the pump-shielding finding: SMV1245 on the
    # paper geometry does not parametrically amplify (the package divider
    # shields the pump). Predicted gain is small.
    var = RealisticVaractor.smv1245()
    spec = level10_realistic_c_canonical(
        varactor=var, stub_scale=1.0, n_cells=15,
        f_signal_Hz=3.1e9, f_pump_Hz=6.2e9,
    )
    out = level10_realistic_param_gain(spec, var, m_branch_pump=0.3)
    assert abs(out["param_gain_dB"]) < 0.5, out["param_gain_dB"]
    assert out["depth_effective"] < 0.3  # pump shielded


def test_favorable_geometry_can_gain_with_prescribed_depth():
    # The favorable geometry (small C0, low L_pkg, stub-scaled) IS capable of
    # parametric gain analytically when a branch-top depth is delivered -- it
    # is not a geometry limitation. (Whether a self-pumped line *delivers*
    # that depth in ADS is the pump-delivery question; see the pump-shielding
    # finding and the realistic crossval probes.)
    var = RealisticVaractor(
        CJ0_f=0.3e-12, VJ_v=3.5, M=1.7, CP_f=0.0, L_pkg_h=0.3e-9, R_series_ohm=0.5
    )
    spec = level10_realistic_c_canonical(
        varactor=var, stub_scale=0.5, n_cells=15, f_signal_Hz=3.1e9, f_pump_Hz=6.2e9
    )
    out = level10_realistic_param_gain(spec, var, m_branch_pump=0.3)
    assert out["param_gain_dB"] > 0.5, out["param_gain_dB"]


def test_lpkg_series_resonance_can_boost_effective_depth():
    # With a small C0 and a modest L_pkg the series package reactance can
    # partially resonate, so the junction sees MORE than the branch-top depth
    # (|divider| > 1) -- the opposite regime to the SMV1245 shielding.
    var = RealisticVaractor(
        CJ0_f=0.3e-12, VJ_v=3.5, M=1.7, CP_f=0.0, L_pkg_h=0.3e-9, R_series_ohm=0.5
    )
    assert var.m_effective(6.2e9, 0.3) > 0.3
    # whereas the SMV1245 (large C0, large L_pkg) is shielded below the depth
    assert RealisticVaractor.smv1245().m_effective(6.2e9, 0.3) < 0.3


def test_truncation_floor():
    spec = _ideal_specs()["k0p4@3.1"]
    with pytest.raises(ValueError):
        level10_realistic_param_gain(
            spec, RealisticVaractor.ideal(0.3e-12), m_branch_pump=0.3, truncation_N=0
        )


# --------------------------------------------------------------------------- #
# Control-port realistic realization                                           #
# --------------------------------------------------------------------------- #
def _favorable():
    return RealisticVaractor(
        CJ0_f=0.3e-12, VJ_v=3.5, M=1.7, CP_f=0.0, L_pkg_h=0.3e-9, R_series_ohm=0.5
    )


def test_requires_exactly_one_drive():
    spec = _ideal_specs()["k0p4@3.1"]
    var = _favorable()
    with pytest.raises(ValueError):  # neither
        level10_realistic_param_gain(spec, var)
    with pytest.raises(ValueError):  # both
        level10_realistic_param_gain(spec, var, m_branch_pump=0.3, control_port_vpump_v=0.5)


def test_control_port_spec_valid_and_emits_poly_body():
    var = _favorable()
    spec = level10_realistic_c_control_port_canonical(
        varactor=var, stub_scale=0.5, n_cells=3, pump_power_dBm=-3.0
    )
    validate_device_spec(spec)
    sdd = next(e for e in spec.elements if e.id == "VARACTOR_C1")
    rows = sdd2p_tvc_poly_body_rows(sdd)
    # port-1 charge has the v2-poly form; port 2 is zero-current
    assert rows[0][0] == "1" and "_v2" in rows[0][2] and "_v1" in rows[0][2]
    assert rows[1] == ["2", "0", "0"]
    # control net carries the pump (Num=2)
    assert any(s.port_node == "n_pump_ctrl" and s.num == 2 for s in spec.sources)


def test_control_port_analytical_positive_gain_at_favorable():
    var = _favorable()
    spec = level10_realistic_c_control_port_canonical(
        varactor=var, stub_scale=0.5, n_cells=15, pump_power_dBm=-3.0
    )
    Voc = open_circuit_pump_voltage(-3.0)
    out = level10_realistic_param_gain(spec, var, control_port_vpump_v=Voc)
    assert out["mode"] == "control_port"
    assert out["param_gain_dB"] > 0.5, out["param_gain_dB"]


def test_control_port_rejects_ideal_varactor():
    with pytest.raises(ValueError):
        level10_realistic_c_control_port_canonical(
            varactor=RealisticVaractor.ideal(0.3e-12), stub_scale=0.5, n_cells=2
        )


# --------------------------------------------------------------------------- #
# Coherent gain + PSA                                                         #
# --------------------------------------------------------------------------- #
def test_coherent_reduces_to_mix10_when_idler_zero():
    # No degenerate idler -> coherent peak/min/swing collapse to mix(1,0).
    N = 3
    size = 2 * N + 1
    on = np.zeros((size, size), dtype=complex)
    off = np.zeros((size, size), dtype=complex)
    on[N, N] = 2.0  # +6 dB signal transfer
    off[N, N] = 1.0
    # idler entry on[N-1, N] left zero
    d = device_coherent_gain(on, off, N)
    assert d["idler_over_signal"] == 0.0
    assert d["mix10_gain_dB"] == pytest.approx(20 * math.log10(2.0))
    assert d["coherent_peak_dB"] == pytest.approx(d["mix10_gain_dB"])
    assert d["coherent_min_dB"] == pytest.approx(d["mix10_gain_dB"])
    assert d["coherent_swing_dB"] == pytest.approx(0.0)
    assert d["coherent_at_block_phase_dB"] == pytest.approx(d["mix10_gain_dB"])


def test_coherent_peak_min_with_idler():
    N = 3
    size = 2 * N + 1
    on = np.zeros((size, size), dtype=complex)
    off = np.zeros((size, size), dtype=complex)
    on[N, N] = 1.0
    on[N - 1, N] = 0.25  # idler 25 % of signal
    off[N, N] = 1.0
    d = device_coherent_gain(on, off, N)
    assert d["idler_over_signal"] == pytest.approx(0.25)
    assert d["coherent_peak_dB"] == pytest.approx(20 * math.log10(1.25))
    assert d["coherent_min_dB"] == pytest.approx(20 * math.log10(0.75))
    assert d["coherent_swing_dB"] == pytest.approx(
        20 * math.log10(1.25) - 20 * math.log10(0.75)
    )


def test_psa_peak_is_phase_sweep_maximum():
    # The convention-free coherent peak (magnitudes formula) must equal the
    # maximum coherent gain over an explicit analytical pump-phase sweep.
    var = _favorable()
    base = level10_realistic_c_control_port_canonical(
        varactor=var, stub_scale=0.5, n_cells=15, pump_power_dBm=-3.0
    )
    Voc = open_circuit_pump_voltage(-3.0)
    peak = level10_realistic_param_gain(
        base, var, control_port_vpump_v=Voc
    )["coherent_peak_dB"]
    swept = []
    for phd in range(0, 360, 5):
        sp = with_var_overrides(base, {"pump_phase_rad": math.radians(phd)})
        swept.append(
            level10_realistic_param_gain(sp, var, control_port_vpump_v=Voc)[
                "coherent_at_phase0_dB"
            ]
        )
    assert max(swept) == pytest.approx(peak, abs=0.05)
    # the sweep is a PSA sinusoid: a real peak-to-peak swing
    assert max(swept) - min(swept) > 1.0


def test_gain_landscape_finite_and_smooth():
    # The analytical gain landscape is finite across the operating band.
    import numpy as _np
    for f in _np.arange(2.0, 5.01, 0.25):
        spec = make_level10_ideal_c_family(
            "k0p5", n_cells=15, f_signal_Hz=f * 1e9, f_pump_Hz=2 * f * 1e9
        )
        o = level10_ideal_c_param_gain(spec)
        assert math.isfinite(o["param_gain_dB"])
        assert math.isfinite(o["coherent_peak_dB"])
        assert o["coherent_peak_dB"] >= o["param_gain_dB"] - 1e-9


