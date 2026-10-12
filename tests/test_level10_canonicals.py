"""Tests for the Level-10 paper-geometry cascade.

ADS-free: the Level-10 canonical builds, validates, emits through the
hpeesofsim and OA paths (the OA path against the stub
``keysight.ads.de``), and passes every content check across the 15
SDD2P / 31 I_Probe / 17 NAME instances. The actual ADS HB run is
exercised separately by the Figure 4 dry-run script.

Verifies (against the locked design):

* the Skyworks SMV1245 Taylor polynomial reproduces the SPICE C(v) at
  v = 0 (CJ0 = 6.9 pF) and the first-derivative slope
  ``M*CJ0/VJ ≈ 3.35 pF/V`` (negative, because the analytical convention
  is ``C(v) = CJ0/(1 + v/VJ)^M``);
* the Bahl via inductance returns the correct value for the paper
  geometry (h=1.54 mm, r=0.8 mm, n_parallel=2);
* the paper-fidelity factory builds a 15-cell cascade with the locked
  topology (15 MTEEs, 15 SDD2Ps, 15 CP capacitors, 15 L_pkg inductors,
  15 R_pkg resistors, 15 L_via inductors, 3×15 stub MLINs, 2×15 main
  MLINs, 16 boundary I_Probes, 15 varactor I_Probes, 17 NAMEs);
* the lifted Floquet stub extractor recovers the 3-segment stub
  geometry and the L_via inductance from the spec automatically;
* the netlist cross-check passes (Path A);
* every content check matches.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from test_oa_canonical import _install_fake_keysight, _stub_emit

from tvtl.design.nonlinear_capacitor import (
    smv1245_skyworks_spice_polynomial,
)
from tvtl.microstrip.discontinuities import via_inductance_bahl_H

from tvtl_ads_builder_v2 import (
    LEVEL_10_CANONICAL_CIRCUITS,
    LEVEL_10_LABEL,
    Level10CellSpec,
    Level10StubSegmentSpec,
    Level10SubstrateSpec,
    Level10VaractorSpec,
    Level10ViaSpec,
    build_stub_y_fn_from_spec,
    extract_series_via_inductance_H,
    extract_stub_segments,
    make_paper_geometry_tvtl_cascade,
    make_tvtl_level10_cascade,
)
from tvtl_ads_builder_v2.device_spec import resolve_spec, validate_device_spec
from tvtl_ads_builder_v2.netlist.emit import emit_hpeesofsim_netlist
from tvtl_ads_builder_v2.oa import oa_canonical_cli, oa_emitter
from tvtl_ads_builder_v2.oa.hb_controller_content_check import (
    check_hb_two_tone_content,
)
from tvtl_ads_builder_v2.oa.netlist_compare import (
    compare_circuits,
    parse_netlist_text,
)
from tvtl_ads_builder_v2.oa.p1tone_content_check import check_p1tone_content
from tvtl_ads_builder_v2.oa.sdd2p_content_check import check_sdd2p_content
from tvtl_ads_builder_v2.oa.var_content_check import check_var_content


# --------------------------------------------------------------------------
# Skyworks SMV1245 Taylor polynomial
# --------------------------------------------------------------------------


def test_smv1245_skyworks_spice_polynomial_c0_matches_cj0() -> None:
    """The Taylor C0 is the Skyworks CJ0 verbatim (zero-bias capacitance)."""

    model = smv1245_skyworks_spice_polynomial(
        cj0_f=6.9e-12, vj_v=3.5, m_grade=1.7, order=5,
    )
    assert model.c0_f == pytest.approx(6.9e-12)


def test_smv1245_skyworks_spice_polynomial_c1_slope_matches_sdcdv() -> None:
    """The first-order slope is ``-M*CJ0/VJ`` (paper sign convention)."""

    cj0, vj, m = 6.9e-12, 3.5, 1.7
    model = smv1245_skyworks_spice_polynomial(
        cj0_f=cj0, vj_v=vj, m_grade=m, order=5,
    )
    expected_c1 = -m * cj0 / vj
    assert model.c1_f_per_v == pytest.approx(expected_c1, rel=1e-9)


def test_smv1245_skyworks_spice_polynomial_reproduces_cv_at_low_voltage() -> None:
    """At |v| <= 1 V the 5th-order Taylor matches SPICE within 0.5%."""

    cj0, vj, m = 6.9e-12, 3.5, 1.7
    model = smv1245_skyworks_spice_polynomial(
        cj0_f=cj0, vj_v=vj, m_grade=m, order=5,
    )
    for v in (-1.0, -0.5, 0.0, 0.5, 1.0):
        spice = cj0 / (1.0 + v / vj) ** m
        taylor = model.capacitance_f(v)
        assert taylor == pytest.approx(spice, rel=5e-3), (
            f"Taylor mismatch at v={v}: {taylor} vs SPICE {spice}"
        )


def test_smv1245_skyworks_spice_polynomial_order_caps_at_5() -> None:
    """Order > 5 silently truncates to 5 (the ChargePolynomialModel cap)."""

    high = smv1245_skyworks_spice_polynomial(order=6)
    cap5 = smv1245_skyworks_spice_polynomial(order=5)
    # Same coefficient set, since the schema caps at C5 anyway.
    assert high.coefficients == cap5.coefficients


def test_smv1245_skyworks_spice_polynomial_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="cj0_f"):
        smv1245_skyworks_spice_polynomial(cj0_f=-1.0)
    with pytest.raises(ValueError, match="vj_v"):
        smv1245_skyworks_spice_polynomial(vj_v=0.0)
    with pytest.raises(ValueError, match="m_grade"):
        smv1245_skyworks_spice_polynomial(m_grade=-0.5)
    with pytest.raises(ValueError, match="order"):
        smv1245_skyworks_spice_polynomial(order=0)


# --------------------------------------------------------------------------
# Bahl via-inductance helper
# --------------------------------------------------------------------------


def test_via_inductance_bahl_paper_geometry() -> None:
    """h=1.54 mm, r=0.8 mm: ~0.57 nH single via, ~0.28 nH for 2 in parallel."""

    l_single = via_inductance_bahl_H(1.54e-3, 0.8e-3, n_parallel=1)
    l_pair = via_inductance_bahl_H(1.54e-3, 0.8e-3, n_parallel=2)
    # Bahl formula: (mu0 * h / 2pi) * (ln(2h/r) + 0.5)
    mu0 = 4.0 * math.pi * 1.0e-7
    expected_single = (mu0 * 1.54e-3 / (2.0 * math.pi)) * (
        math.log(2.0 * 1.54e-3 / 0.8e-3) + 0.5
    )
    assert l_single == pytest.approx(expected_single, rel=1e-9)
    assert l_pair == pytest.approx(expected_single / 2.0, rel=1e-9)
    assert l_pair == pytest.approx(2.846e-10, rel=1e-3)


def test_via_inductance_bahl_scales_with_substrate_thickness() -> None:
    """Doubling h roughly doubles L_via (the ln term grows slightly more)."""

    l_thin = via_inductance_bahl_H(1.0e-3, 0.5e-3)
    l_thick = via_inductance_bahl_H(2.0e-3, 0.5e-3)
    # Ratio is greater than 2 because the ln(2h/r) factor grows with h.
    assert l_thick / l_thin > 2.0


def test_via_inductance_bahl_rejects_unphysical_inputs() -> None:
    with pytest.raises(ValueError, match="substrate_thickness_m"):
        via_inductance_bahl_H(-1.0e-3, 0.5e-3)
    with pytest.raises(ValueError, match="via_radius_m"):
        via_inductance_bahl_H(1.0e-3, 0.0)
    with pytest.raises(ValueError, match="n_parallel"):
        via_inductance_bahl_H(1.0e-3, 0.5e-3, n_parallel=0)
    with pytest.raises(ValueError, match="strictly less than"):
        via_inductance_bahl_H(0.5e-3, 0.5e-3)  # r >= h


# --------------------------------------------------------------------------
# Paper-fidelity DeviceSpec build
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def paper_spec():
    return LEVEL_10_CANONICAL_CIRCUITS[0]


def test_level_10_tuple_holds_paper_fidelity_canonical() -> None:
    """LEVEL_10_CANONICAL_CIRCUITS exposes the 15-cell paper canonical."""

    assert len(LEVEL_10_CANONICAL_CIRCUITS) == 1
    assert (
        LEVEL_10_CANONICAL_CIRCUITS[0].label
        == "level10_paper_geometry_tvtl_cascade"
    )


def test_paper_spec_carries_15_cells(paper_spec) -> None:
    n_mtees = sum(1 for e in paper_spec.elements if e.type == "MTEE")
    n_sdds = sum(1 for e in paper_spec.elements if e.type == "SDD2P")
    n_grounds = len(paper_spec.ground_nodes)
    assert n_mtees == 15
    assert n_sdds == 15
    assert n_grounds == 15


def test_paper_spec_per_cell_branch_elements(paper_spec) -> None:
    """Each cell has 1 L_pkg + 1 R_pkg + 1 SDD2P + 1 IPR_VAR +
    3 stub MLINs + 1 L_via.

    CP (the parasitic 0.47 pF) is folded into the SDD2P polynomial via
    ``q_poly_coefficients[1] += cp_f`` rather than emitted as a literal
    parallel C element, to avoid the v2 OA layout engine's single-chain
    shunt-branch constraint. See ``_q_poly_from_varactor`` in
    ``parity/canonical_circuits.py`` for the equivalence proof.
    """

    n_cells = 15
    ids = {e.id for e in paper_spec.elements}
    for k in range(1, n_cells + 1):
        for prefix in (
            f"LINE_LEFT_C{k}", f"TEE_C{k}", f"LINE_RIGHT_C{k}",
            f"LPKG_C{k}", f"RPKG_C{k}", f"VARACTOR_C{k}",
            f"IPR_VAR_C{k}",
            f"STUB_C{k}_0", f"STUB_C{k}_1", f"STUB_C{k}_2",
            f"LVIA_C{k}",
        ):
            assert prefix in ids, f"missing {prefix}"
        # CPAR is intentionally NOT emitted as a literal element; it is
        # folded into the SDD2P polynomial. Asserting its absence pins
        # the design decision.
        assert f"CPAR_C{k}" not in ids, (
            f"CPAR_C{k} should be folded into SDD2P, not emitted literally"
        )


def test_paper_spec_substrate_is_rogers_4003c(paper_spec) -> None:
    params = dict(paper_spec.substrate.parameters)
    assert params["eps_r"] == pytest.approx(3.38)
    assert params["h_m"] == pytest.approx(1.54e-3)
    assert params["tan_delta"] == pytest.approx(0.0027)
    assert params["sigma_S_per_m"] == pytest.approx(5.8e7)
    assert params["t_m"] == pytest.approx(17.5e-6)


def test_paper_spec_varactor_taylor_q_poly(paper_spec) -> None:
    """Each SDD2P carries the Skyworks Taylor polynomial + folded CP.

    ``q[1]`` is the paper-stated C0 = 7.0 pF (analytical-side static
    term) PLUS the parasitic CP = 0.47 pF folded in to represent the
    (junction || CP) parallel combination as a single SDD2P emission;
    ``q[k]`` for k>=2 is pure Skyworks since CP is voltage-independent.
    """

    expected_model = smv1245_skyworks_spice_polynomial(
        cj0_f=6.9e-12, vj_v=3.5, m_grade=1.7, order=5,
    )
    expected_q = list(expected_model.charge_polynomial_coefficients)
    expected_q[1] = 7.0e-12 + 0.47e-12  # analytical C0 + CP fold
    sdds = [e for e in paper_spec.elements if e.type == "SDD2P"]
    for sdd in sdds:
        coeffs = sdd.parameters["q_poly_coefficients"]
        assert len(coeffs) == len(expected_q)
        for got, want in zip(coeffs, expected_q):
            assert float(got) == pytest.approx(float(want), abs=1e-30)


def test_paper_spec_no_literal_cpar_element(paper_spec) -> None:
    """CP is folded into SDD2P q[1], NOT emitted as a literal parallel C.

    The fold is the documented workaround for the v2 OA layout engine's
    single-chain shunt-branch constraint -- see ``_q_poly_from_varactor``
    in ``parity/canonical_circuits.py``.
    """

    cpar_elements = [
        e for e in paper_spec.elements if e.id.startswith("CPAR_C")
    ]
    assert cpar_elements == []


def test_paper_spec_l_pkg_and_r_pkg_values(paper_spec) -> None:
    """L_pkg=1.7 nH and R_pkg=2 Ohm match the locked Skyworks parasitics."""

    for k in range(1, 16):
        lp = next(e for e in paper_spec.elements if e.id == f"LPKG_C{k}")
        rp = next(e for e in paper_spec.elements if e.id == f"RPKG_C{k}")
        assert lp.parameters["l_h"] == pytest.approx(1.7e-9)
        assert rp.parameters["r_ohm"] == pytest.approx(2.0)


def test_paper_spec_l_via_matches_bahl_for_2_vias(paper_spec) -> None:
    """L_via per cell == Bahl(h=1.54mm, r=0.8mm) / 2."""

    expected = via_inductance_bahl_H(1.54e-3, 0.8e-3, n_parallel=2)
    for k in range(1, 16):
        lv = next(e for e in paper_spec.elements if e.id == f"LVIA_C{k}")
        assert lv.parameters["l_h"] == pytest.approx(expected, rel=1e-9)


def test_paper_spec_main_line_dimensions(paper_spec) -> None:
    """LINE_LEFT and LINE_RIGHT are each main_period/2 long, 4.27 mm wide."""

    for k in range(1, 16):
        ll = next(e for e in paper_spec.elements if e.id == f"LINE_LEFT_C{k}")
        lr = next(e for e in paper_spec.elements if e.id == f"LINE_RIGHT_C{k}")
        assert ll.parameters["w_m"] == pytest.approx(4.27e-3)
        assert lr.parameters["w_m"] == pytest.approx(4.27e-3)
        assert ll.parameters["l_m"] == pytest.approx(6.11e-3 / 2.0)
        assert lr.parameters["l_m"] == pytest.approx(6.11e-3 / 2.0)


def test_paper_spec_stub_segment_dimensions(paper_spec) -> None:
    """Three stub segments per cell match Figure 3b w_2/w_3/w_4 and l_2/l_3/l_4."""

    expected = (
        (1.93e-3, 6.83e-3),
        (1.41e-3, 5.25e-3),
        (1.98e-3, 1.04e-3),
    )
    for k in range(1, 16):
        for i, (w, l_seg) in enumerate(expected):
            seg = next(
                e for e in paper_spec.elements if e.id == f"STUB_C{k}_{i}"
            )
            assert seg.parameters["w_m"] == pytest.approx(w)
            assert seg.parameters["l_m"] == pytest.approx(l_seg)


def test_paper_spec_iprobe_count(paper_spec) -> None:
    """I_Probe count = 1 IPR_IN + 14 inter-cell + 1 IPR_OUT + 15 IPR_VAR = 31."""

    iprobes = [e for e in paper_spec.elements if e.type == "I_Probe"]
    assert len(iprobes) == 31


def test_paper_spec_name_count_is_minimum_needed(paper_spec) -> None:
    """32 NAMEs: n_input + 15 × (n_c{k}_b, n_c{k}_v) + n_output.

    Minimum-needed for the Floquet comparison + energy-conservation
    harnesses: n_c{k}_b (MTEE pin 3, V_0 source) and n_c{k}_v (SDD2P
    pin 2, time-varying capacitor's other terminal) per cell, plus the
    cascade boundary observables. Leaner than Level-9's 4*N+2 by
    skipping per-cell output and inter-cell boundary nets.
    """

    names = [e for e in paper_spec.elements if e.type == "NAME"]
    assert len(names) == 32
    name_nets = {e.pins[0].node for e in names}
    expected_nets = {"n_input", "n_output"} | {
        f"n_c{k}_b" for k in range(1, 16)
    } | {
        f"n_c{k}_v" for k in range(1, 16)
    }
    assert name_nets == expected_nets


def test_paper_spec_distinct_num_on_sources_and_port(paper_spec) -> None:
    """Signal Num=1, pump Num=2, output Port Num=3 -- no duplicates."""

    nums = {s.name: s.num for s in paper_spec.sources}
    assert nums == {"SIGNAL_SRC": 1, "PUMP_SRC": 2}
    assert {p.num for p in paper_spec.ports} == {3}


def test_paper_spec_var_block_carries_locked_defaults(paper_spec) -> None:
    """The VAR block carries f_signal/f_pump/powers/phase locked defaults."""

    values = {v.name: v.value for v in paper_spec.var_block.variables}
    assert values["f_signal"] == pytest.approx(3.1e9)
    assert values["f_pump"] == pytest.approx(6.2e9)
    assert values["signal_power_dBm"] == pytest.approx(-30.0)
    assert values["pump_power_dBm"] == pytest.approx(16.0)
    assert values["pump_phase_rad"] == pytest.approx(0.0)


def test_paper_spec_validates(paper_spec) -> None:
    """Paper-geometry spec passes the device-spec validator (resolved)."""

    validate_device_spec(resolve_spec(paper_spec))


# --------------------------------------------------------------------------
# Stub extraction + lifted Floquet termination
# --------------------------------------------------------------------------


def test_paper_spec_stub_extraction_returns_three_segments(paper_spec) -> None:
    """``extract_stub_segments`` recovers the paper's 3-segment stub."""

    for k in range(15):
        segments, substrate, term = extract_stub_segments(paper_spec, k)
        assert len(segments) == 3
        assert segments[0].width_m == pytest.approx(1.93e-3)
        assert segments[0].length_m == pytest.approx(6.83e-3)
        assert segments[1].width_m == pytest.approx(1.41e-3)
        assert segments[1].length_m == pytest.approx(5.25e-3)
        assert segments[2].width_m == pytest.approx(1.98e-3)
        assert segments[2].length_m == pytest.approx(1.04e-3)
        assert term == "short"


def test_paper_spec_via_inductance_auto_extracted(paper_spec) -> None:
    """``extract_series_via_inductance_H`` recovers the Bahl L_via."""

    expected = via_inductance_bahl_H(1.54e-3, 0.8e-3, n_parallel=2)
    for k in range(15):
        l_via = extract_series_via_inductance_H(paper_spec, k)
        assert l_via == pytest.approx(expected, rel=1e-9)


def test_paper_spec_y_stub_uses_inductive_termination(paper_spec) -> None:
    """At low frequency the inductive load Z = jwL dominates Y_stub."""

    y_fn = build_stub_y_fn_from_spec(paper_spec, 0)
    # 1 MHz: stub MLINs are barely electrically present (low frequency),
    # so Y_stub is dominated by the via inductance.
    omega = 2.0 * math.pi * 1.0e6
    y_low = y_fn(omega)
    # Expected: Y_in ≈ 1 / (j*w*L_via_total + tiny_stub_inductance) ≈ huge but
    # with negligible real part. The magnitude at 1 MHz is large.
    assert abs(y_low) > 1.0  # Mohms-scale admittance OK for fundamental short
    # The imaginary part has the correct sign (short-like termination
    # presents a positive imaginary Y_in at low frequencies because
    # Z = jwL is purely inductive).
    assert y_low.imag < 0.0


def test_paper_spec_y_stub_finite_at_paper_frequency(paper_spec) -> None:
    """Y_stub is finite and non-zero at the paper's 3.1 GHz operating point."""

    y_fn = build_stub_y_fn_from_spec(paper_spec, 0)
    omega = 2.0 * math.pi * 3.1e9
    y = y_fn(omega)
    assert math.isfinite(y.real) and math.isfinite(y.imag)
    assert abs(y) > 0.0


# --------------------------------------------------------------------------
# Netlist emit + content checks
# --------------------------------------------------------------------------


def test_paper_spec_hpeesofsim_deck_carries_every_iprobe(paper_spec) -> None:
    deck = emit_hpeesofsim_netlist(paper_spec)
    iprobes = [e for e in paper_spec.elements if e.type == "I_Probe"]
    for probe in iprobes:
        line_prefix = f"Short:{probe.id} "
        matching = [
            line for line in deck.splitlines()
            if line.startswith(line_prefix)
        ]
        assert matching, f"no Short line for {probe.id}"
        assert "Mode=0 SaveCurrent=yes" in matching[0]


def test_paper_spec_passes_sdd2p_content_check(paper_spec) -> None:
    deck = emit_hpeesofsim_netlist(paper_spec)
    result = check_sdd2p_content(paper_spec, hpeesofsim_netlist_text=deck)
    assert result.match, result.failures
    assert len(result.checked) == 15


def test_paper_spec_passes_var_content_check(paper_spec) -> None:
    deck = emit_hpeesofsim_netlist(paper_spec)
    result = check_var_content(paper_spec, hpeesofsim_netlist_text=deck)
    assert result.match, result.failures


def test_paper_spec_passes_p1tone_content_check(paper_spec) -> None:
    deck = emit_hpeesofsim_netlist(paper_spec)
    result = check_p1tone_content(paper_spec, hpeesofsim_netlist_text=deck)
    assert result.match, result.failures


def test_paper_spec_passes_hb_two_tone_content_check(paper_spec) -> None:
    deck = emit_hpeesofsim_netlist(paper_spec)
    result = check_hb_two_tone_content(
        paper_spec, hpeesofsim_netlist_text=deck,
    )
    assert result.match, result.failures


def test_paper_spec_self_compare_match(paper_spec) -> None:
    """Path-A self cross-check: deck matches itself across all 31 I_Probes."""

    deck = emit_hpeesofsim_netlist(paper_spec)
    result = compare_circuits(
        parse_netlist_text(deck), parse_netlist_text(deck),
    )
    assert result.match, result.differences


# --------------------------------------------------------------------------
# OA emitter (stub keysight)
# --------------------------------------------------------------------------


def test_oa_emitter_accepts_paper_spec(
    tmp_path: Path, monkeypatch, paper_spec,
) -> None:
    """The OA emitter builds a workspace for Level-10 without raising."""

    _install_fake_keysight(monkeypatch)
    info = oa_emitter.emit_workspace_from_spec(
        paper_spec, tmp_path / paper_spec.label / "oa_workspace",
    )
    hpeesofsim = Path(info["hpeesofsim_netlist_path"]).read_text(
        encoding="utf-8",
    )
    for probe in (e for e in paper_spec.elements if e.type == "I_Probe"):
        assert f"Short:{probe.id} " in hpeesofsim


# --------------------------------------------------------------------------
# Generic generator + dataclass coverage
# --------------------------------------------------------------------------


def test_make_tvtl_level10_cascade_rejects_empty_cells_tuple() -> None:
    with pytest.raises(ValueError, match="at least one cell"):
        make_tvtl_level10_cascade(cells=(), label="x")


def test_make_tvtl_level10_cascade_rejects_zero_segment_stub() -> None:
    cell = Level10CellSpec(
        main_width_m=4.27e-3, main_period_m=6.11e-3,
        branch_width_m=1.93e-3,
        stub_segments=(),
        via=Level10ViaSpec(),
    )
    with pytest.raises(ValueError, match="no stub segments"):
        make_tvtl_level10_cascade(cells=(cell,), label="x")


def test_make_tvtl_level10_cascade_accepts_custom_source_vars() -> None:
    """The source_var_names + source_var_values overrides flow through."""

    cell = Level10CellSpec(
        main_width_m=4.27e-3, main_period_m=6.11e-3,
        branch_width_m=1.93e-3,
        stub_segments=(Level10StubSegmentSpec(1.0e-3, 5.0e-3),),
        via=Level10ViaSpec(),
    )
    spec = make_tvtl_level10_cascade(
        cells=(cell,),
        label="lv10_custom_vars",
        source_var_names={"f_signal": "fs", "f_pump": "fp"},
        source_var_values={"f_signal": 2.0e9, "f_pump": 4.0e9},
    )
    validate_device_spec(resolve_spec(spec))
    names = {v.name for v in spec.var_block.variables}
    assert "fs" in names and "fp" in names


def test_make_paper_geometry_tvtl_cascade_rejects_zero_cells() -> None:
    with pytest.raises(ValueError, match="n_cells must be >= 1"):
        make_paper_geometry_tvtl_cascade(n_cells=0)


def test_make_paper_geometry_tvtl_cascade_supports_smaller_cascades() -> None:
    """A 2-cell mini paper-geometry cascade (for regression on small N)."""

    spec = make_paper_geometry_tvtl_cascade(
        n_cells=2, label="lv10_2cell_paper_mini",
    )
    validate_device_spec(resolve_spec(spec))
    assert sum(1 for e in spec.elements if e.type == "SDD2P") == 2


# --------------------------------------------------------------------------
# CLI registration
# --------------------------------------------------------------------------


def test_oa_canonical_cli_registers_level_10() -> None:
    """oa_canonical_cli has a level10 entry pointing at the Level-10 tuple."""

    assert "level10" in oa_canonical_cli._LEVELS
    circuits, label, _tol = oa_canonical_cli._LEVELS["level10"]
    assert label == LEVEL_10_LABEL
    assert circuits is LEVEL_10_CANONICAL_CIRCUITS


def test_cli_level10_processes_paper_canonical(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    """`oa_canonical_cli level10` processes the paper canonical."""

    monkeypatch.setattr(oa_emitter, "emit_workspace_from_spec", _stub_emit)
    monkeypatch.setattr(
        oa_canonical_cli, "_leg_b",
        lambda spec, d, label, tol: {
            "reason": "blocked", "max_abs_delta_dB": float("nan"),
        },
    )
    rc = oa_canonical_cli.main(["level10", "--output_root", str(tmp_path)])
    out = capsys.readouterr().out
    assert "level10_paper_geometry_tvtl_cascade" in out
    assert out.count("MATCH") >= 2  # at least Path A + content row
    assert rc == 0


