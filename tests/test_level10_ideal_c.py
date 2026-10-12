"""Unit tests for the Level10IdealC stub-loaded ideal-C TVTL family (no ADS).

Covers the device-spec generator, the time-varying-C SDD emission, and the
analytical (capacitance-linear) parametric-gain leg. The ADS cross-validation
is the ADS-gated ``tests/parity/test_level10_idealc_crossval.py``.
"""

from __future__ import annotations

import math

import pytest

from tvtl_ads_builder_v2.analytical import (
    ideal_c_realised_depth,
    level10_ideal_c_param_gain,
)
from tvtl_ads_builder_v2.device_spec import validate_device_spec
from tvtl_ads_builder_v2.netlist import emit_flat_hb_netlist
from tvtl_ads_builder_v2.parity import (
    make_level10_ideal_c_cascade,
    make_level10_ideal_c_family,
)
from tvtl_ads_builder_v2.parity.level10_ideal_c import (
    control_port_k_from_depth,
    open_circuit_pump_voltage,
)


def test_open_circuit_voltage_matches_ads_calibration():
    # 10 dBm into 50 ohm -> 2.000 V open-circuit (verified against ADS HB).
    assert open_circuit_pump_voltage(10.0, 50.0) == pytest.approx(2.0, rel=1e-9)


def test_control_slope_realises_target_depth():
    c0 = 0.3e-12
    k = control_port_k_from_depth(0.3, c0, 10.0)
    voc = open_circuit_pump_voltage(10.0)
    assert k * voc / c0 == pytest.approx(0.3, rel=1e-9)


@pytest.mark.parametrize("name", ["k0p4", "k0p5", "k0p6"])
def test_family_builds_and_validates(name):
    spec = make_level10_ideal_c_family(name, n_cells=15)
    validate_device_spec(spec)
    # 15 ideal-C SDDs (one per cell), 3 vias-as-L per cell... check structure.
    sdds = [e for e in spec.elements if e.type == "SDD2P" and "tvc_c0_f" in e.parameters]
    assert len(sdds) == 15
    # no package L_pkg / R_pkg in the ideal-C branch
    assert not any(e.id.startswith("LPKG_") or e.id.startswith("RPKG_")
                   for e in spec.elements)
    # the realised depth is the design value 0.3
    assert ideal_c_realised_depth(spec) == pytest.approx(0.3, rel=1e-6)


def test_flat_hb_netlist_emits_control_port_sdd_and_port_sources():
    spec = make_level10_ideal_c_family("k0p5", n_cells=2)
    txt = emit_flat_hb_netlist(spec)
    # control-port time-varying-C SDD body (bivariate; open sense port)
    assert "I[1,1]=" in txt and "_v2*_v1" in txt and "I[2,0]=0" in txt
    # built-in Port HB sources (NOT P_1Tone); pump on the control net
    assert "Port:SIGNAL_SRC n_input" in txt
    assert "Port:PUMP_SRC n_pump_ctrl" in txt
    assert "P_1Tone" not in txt
    # no S_Param controller on the HB deck
    assert "S_Param" not in txt
    assert "HB:" in txt


def test_stub_scale_shortens_segments():
    spec = make_level10_ideal_c_family("k0p5", n_cells=1)  # scale 0.5
    lengths_mm = [round(float(e.parameters["l_m"]) * 1e3, 3) for e in
                  sorted((e for e in spec.elements if e.id.startswith("STUB_C1_")),
                         key=lambda x: x.id)]
    # paper [6.83, 5.25, 1.04] * 0.5
    assert lengths_mm == [3.415, 2.625, 0.52]


@pytest.mark.parametrize(
    "name, lo, hi",
    [("k0p4", 1.0, 2.2), ("k0p5", 2.8, 4.0), ("k0p6", 7.5, 11.0)],
)
def test_analytical_param_gain_in_expected_band(name, lo, hi):
    """Ideal-junction analytical gain in the screening-predicted band at 3.1 GHz.

    Pins the ``use_mtee_parasitic=False`` (ideal-junction) path; the
    reference-plane upgrade default is exercised by
    ``test_analytical_param_gain_mtee_upgrade``.
    """
    spec = make_level10_ideal_c_family(name, n_cells=15)
    out = level10_ideal_c_param_gain(spec, use_mtee_parasitic=False)
    assert out["label"] == "analytical_proxy_only"
    assert out["depth"] == pytest.approx(0.3, rel=1e-6)
    assert lo < out["param_gain_dB"] < hi


def test_negative_control_paper_stub_no_gain():
    """Paper stub geometry + ideal C0=0.3 pF -> ~0 analytical gain.

    Confirms stub-geometry tuning (not just a small C0) is required: the
    untuned paper stub decouples the branch even with an ideal capacitor.
    """
    spec = make_level10_ideal_c_cascade(
        stub_scale=1.0, c0_f=0.3e-12, depth=0.3, n_cells=15,
        label="level10_ideal_c_paperstub_negctrl",
    )
    out = level10_ideal_c_param_gain(spec)
    assert abs(out["param_gain_dB"]) < 0.1


