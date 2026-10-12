"""Tests for the Floquet-vs-ADS comparison.

ADS-free unit tests:

* the harmonic-band classification and per-band tolerance lookup;
* the analytical-harmonic-to-ADS-mix-tuple mapping under
  ``pump_signal_ratio = 2``;
* the SDD2P-charge-polynomial-to-``(C0, C1)`` extraction;
* the round-trip synthetic case: build an HBDataset whose varactor
  current spectrum matches the analytical solver's prediction exactly,
  and assert :func:`compare_varactor_harmonics` reports 0 dB across
  every harmonic.

The companion integration test
``tests/parity/test_level9_floquet_comparison.py`` runs the comparison
end-to-end against a live HB ``.ds`` on the Level-9 canonicals.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tvtl.floquet.harmonics import (
    current_spectrum,
    solve_harmonic_coefficients,
)
from tvtl_ads_builder_v2.analysis import (
    DEFAULT_HARMONIC_TOLERANCE_DB,
    HBDataset,
    HBHarmonic,
    HBSpectrum,
    PUMP_MIX_2TONE,
    SIGNAL_MIX_2TONE,
    VaractorModelParameters,
    build_cell_map,
    compare_cascade_harmonics,
    compare_varactor_harmonics,
    extract_varactor_parameters,
    harmonic_band,
    harmonic_tolerance_dB,
)
from tvtl_ads_builder_v2.parity.canonical_circuits import (
    LEVEL_9_CANONICAL_CIRCUITS,
)


_SPEC_L9_2CELL = LEVEL_9_CANONICAL_CIRCUITS[0]
_SPEC_L9_3CELL = LEVEL_9_CANONICAL_CIRCUITS[1]


# --------------------------------------------------------------------------
# harmonic_band / harmonic_tolerance_dB
# --------------------------------------------------------------------------


def test_harmonic_band_classification() -> None:
    assert harmonic_band(0) == "dominant"
    assert harmonic_band(1) == "dominant"
    assert harmonic_band(-1) == "dominant"
    assert harmonic_band(2) == "secondary"
    assert harmonic_band(-2) == "secondary"
    assert harmonic_band(3) == "higher"
    assert harmonic_band(-3) == "higher"
    assert harmonic_band(7) == "higher"


def test_default_tolerance_per_band() -> None:
    assert harmonic_tolerance_dB(0) == 0.5
    assert harmonic_tolerance_dB(1) == 0.5
    assert harmonic_tolerance_dB(-1) == 0.5
    assert harmonic_tolerance_dB(2) == 1.0
    assert harmonic_tolerance_dB(-2) == 1.0
    assert harmonic_tolerance_dB(3) == 2.0
    assert harmonic_tolerance_dB(10) == 2.0


def test_tolerance_lookup_honours_custom_mapping() -> None:
    custom = {"dominant": 0.1, "secondary": 0.2, "higher": 0.3}
    assert harmonic_tolerance_dB(0, custom) == 0.1
    assert harmonic_tolerance_dB(2, custom) == 0.2
    assert harmonic_tolerance_dB(5, custom) == 0.3


def test_default_tolerance_constants_match_brief() -> None:
    assert DEFAULT_HARMONIC_TOLERANCE_DB["dominant"] == 0.5
    assert DEFAULT_HARMONIC_TOLERANCE_DB["secondary"] == 1.0
    assert DEFAULT_HARMONIC_TOLERANCE_DB["higher"] == 2.0


# --------------------------------------------------------------------------
# VaractorModelParameters -- positivity guards
# --------------------------------------------------------------------------


def test_params_positivity_guards() -> None:
    with pytest.raises(ValueError, match="S0"):
        VaractorModelParameters(S0=-1.0, dS=0.0, phi=0.0)
    with pytest.raises(ValueError, match="dS"):
        VaractorModelParameters(S0=1.0, dS=2.0, phi=0.0)  # dS > S0
    with pytest.raises(ValueError, match="dS"):
        VaractorModelParameters(S0=1.0, dS=-0.1, phi=0.0)
    with pytest.raises(ValueError, match="R and L"):
        VaractorModelParameters(S0=1.0, dS=0.1, phi=0.0, R=-0.5)


def test_params_accepts_zero_modulation() -> None:
    """dS == 0 must be accepted (no pump applied)."""

    p = VaractorModelParameters(S0=1.0, dS=0.0, phi=0.0)
    assert p.S0 == 1.0
    assert p.dS == 0.0


# --------------------------------------------------------------------------
# extract_varactor_parameters -- spec + dataset to (S0, dS, phi)
# --------------------------------------------------------------------------


def _synthetic_hb_pump_only(
    cell_map,
    *,
    cell_index: int,
    v_pump: complex,
    v_signal: complex = 1.0 + 0j,
    other_pump: complex = 0j,
) -> HBDataset:
    """Build a minimal HBDataset that carries V_pump at every cell's branch.

    Only the spectra needed by extract_varactor_parameters and
    compare_varactor_harmonics are populated.
    """

    node_voltages: dict[str, HBSpectrum] = {}
    branch_currents: dict[str, HBSpectrum] = {}
    for k in range(len(cell_map.cells)):
        branch_top_net = cell_map.cells[k].varactor_signal_net
        node_voltages[branch_top_net] = HBSpectrum(
            quantity="",
            harmonics=(
                HBHarmonic(mix=(1, 0), frequency_Hz=3e9, amplitude=v_signal),
                HBHarmonic(
                    mix=(0, 1), frequency_Hz=6e9,
                    amplitude=(v_pump if k == cell_index else other_pump),
                ),
            ),
        )
        probe_id = cell_map.cells[k].varactor_probe_id
        if probe_id is not None:
            branch_currents[f"{probe_id}.i"] = HBSpectrum(
                quantity="",
                harmonics=(
                    HBHarmonic(mix=(1, 0), frequency_Hz=3e9, amplitude=0j),
                    HBHarmonic(mix=(0, 1), frequency_Hz=6e9, amplitude=0j),
                ),
            )
    return HBDataset(
        fundamental_freqs_Hz=(3e9, 6e9),
        node_voltages=node_voltages,
        branch_currents=branch_currents,
    )


def test_extract_parameters_returns_C0_from_smv1245_poly() -> None:
    """SMV1245 polynomial has C0 = 7 pF -> S0 = 1 / 7 pF."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _synthetic_hb_pump_only(cm, cell_index=0, v_pump=0.1 + 0j)
    p = extract_varactor_parameters(_SPEC_L9_2CELL, hb, cm, 0)
    assert p.C0 == pytest.approx(7.0e-12)
    assert p.S0 == pytest.approx(1.0 / 7.0e-12)


def test_extract_parameters_C1_is_signed_smv1245_slope() -> None:
    """SMV1245 polynomial has C1 = -2 pF/V."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _synthetic_hb_pump_only(cm, cell_index=0, v_pump=0.1 + 0j)
    p = extract_varactor_parameters(_SPEC_L9_2CELL, hb, cm, 0)
    assert p.C1 == pytest.approx(-2.0e-12)


def test_extract_parameters_dS_scales_with_pump_amplitude() -> None:
    """dS = |C1| * |V_p| / C0**2 -- doubling V_p doubles dS."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    hb1 = _synthetic_hb_pump_only(cm, cell_index=0, v_pump=0.1 + 0j)
    p1 = extract_varactor_parameters(_SPEC_L9_2CELL, hb1, cm, 0)
    hb2 = _synthetic_hb_pump_only(cm, cell_index=0, v_pump=0.2 + 0j)
    p2 = extract_varactor_parameters(_SPEC_L9_2CELL, hb2, cm, 0)
    assert p2.dS == pytest.approx(2.0 * p1.dS)


def test_extract_parameters_dS_zero_when_pump_zero() -> None:
    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _synthetic_hb_pump_only(cm, cell_index=0, v_pump=0j)
    p = extract_varactor_parameters(_SPEC_L9_2CELL, hb, cm, 0)
    assert p.dS == 0.0


def test_extract_parameters_phi_equals_pump_arg_for_C1_negative() -> None:
    """SMV1245 C1 < 0 -> phi = arg(V_pump)."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    # Real positive V_pump -> arg = 0.
    hb_real = _synthetic_hb_pump_only(cm, cell_index=0, v_pump=0.1 + 0j)
    p_real = extract_varactor_parameters(_SPEC_L9_2CELL, hb_real, cm, 0)
    assert p_real.phi == pytest.approx(0.0)
    # 90-degree V_pump -> arg = pi/2.
    hb_imag = _synthetic_hb_pump_only(cm, cell_index=0, v_pump=0.1j)
    p_imag = extract_varactor_parameters(_SPEC_L9_2CELL, hb_imag, cm, 0)
    assert p_imag.phi == pytest.approx(math.pi / 2.0)


def test_extract_parameters_clamps_or_raises_on_excessive_modulation() -> None:
    """When dS > S0 the linearisation has broken; the helper raises."""

    # Engineer dS / S0 > 0.999 by choosing a huge pump amplitude.
    # For SMV1245 (C0 = 7 pF, C1 = -2 pF/V):
    # dS / S0 = |C1| * |V_p| / C0 = 2e-12 * |V_p| / 7e-12 = (2/7) * |V_p|.
    # 0.999 / (2/7) ~= 3.5 V -> |V_p| > 3.5 V triggers the raise.
    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _synthetic_hb_pump_only(cm, cell_index=0, v_pump=10.0 + 0j)
    with pytest.raises(ValueError, match="dS"):
        extract_varactor_parameters(_SPEC_L9_2CELL, hb, cm, 0)


# --------------------------------------------------------------------------
# compare_varactor_harmonics -- mix mapping + round-trip identity
# --------------------------------------------------------------------------


def test_mix_mapping_pump_ratio_2() -> None:
    """For pump_signal_ratio=2 the analytical harmonic n maps to mix (1, n)."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    # Build a dataset with zero current at every mix -- the harmonic_index
    # and mix_tuples fields are populated regardless; we only care about
    # the mapping shape.
    hb = _synthetic_hb_pump_only(cm, cell_index=0, v_pump=0.1 + 0j)
    result = compare_varactor_harmonics(
        hb, cm, _SPEC_L9_2CELL, 0, truncation_N=4,
    )
    n_idx = result["harmonic_index"]
    mix_tuples = result["mix_tuples"]
    assert n_idx == tuple(range(-4, 5))
    assert mix_tuples == tuple((1, n) for n in range(-4, 5))


def test_mix_mapping_frequencies_under_pump_ratio_2() -> None:
    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _synthetic_hb_pump_only(cm, cell_index=0, v_pump=0.1 + 0j)
    result = compare_varactor_harmonics(
        hb, cm, _SPEC_L9_2CELL, 0, truncation_N=2,
    )
    # (f_signal, f_pump) = (3e9, 6e9). Analytical f_n = f_signal + n*f_pump.
    expected = (
        3e9 - 2 * 6e9,
        3e9 - 1 * 6e9,
        3e9,
        3e9 + 1 * 6e9,
        3e9 + 2 * 6e9,
    )
    for got, exp in zip(result["frequencies_Hz"], expected):
        assert got == pytest.approx(exp)


def test_round_trip_zero_dB_when_ADS_matches_analytical() -> None:
    """Seed I_n_ADS to match the analytical solver -> 0 dB everywhere.

    Build a synthetic dataset whose varactor I_Probe spectrum is exactly
    the analytical I_n at each ADS mix tuple (1, n). All ratio_dB
    entries must be 0 dB to FP noise.
    """

    cm = build_cell_map(_SPEC_L9_2CELL)
    f_signal = 3.0e9
    f_pump = 6.0e9
    ws = 2.0 * math.pi * f_signal
    wm = 2.0 * math.pi * f_pump
    v_pump = 0.1 + 0j
    v_signal = 1.0 + 0j
    # Compute analytical parameters by hand.
    c0 = 7.0e-12
    c1 = -2.0e-12
    s0 = 1.0 / c0
    d_s = abs(c1) * abs(v_pump) / (c0 ** 2)
    phi = 0.0  # arg(V_pump) = 0
    N = 5
    Qn = v_signal * solve_harmonic_coefficients(s0, d_s, phi, 0.0, 0.0, ws, wm, N)
    _f, I_n_analytical = current_spectrum(Qn, ws, wm, N)

    # Build dataset
    branch_top_net = cm.cells[0].varactor_signal_net
    v_pump_other = 0.05 + 0j  # other cells get a pump too so extract works
    node_voltages = {
        branch_top_net: HBSpectrum(
            quantity="",
            harmonics=(
                HBHarmonic(mix=(1, 0), frequency_Hz=3e9, amplitude=v_signal),
                HBHarmonic(mix=(0, 1), frequency_Hz=6e9, amplitude=v_pump),
            ),
        ),
    }
    # The cell-map only enforces per-cell V at the varactor_signal_net
    # for the cells we touch; the other cells are not consulted by
    # compare_varactor_harmonics(cell_index=0).
    for k in range(1, len(cm.cells)):
        node_voltages[cm.cells[k].varactor_signal_net] = HBSpectrum(
            quantity="",
            harmonics=(
                HBHarmonic(mix=(1, 0), frequency_Hz=3e9, amplitude=v_signal),
                HBHarmonic(mix=(0, 1), frequency_Hz=6e9, amplitude=v_pump_other),
            ),
        )
    probe_id = cm.cells[0].varactor_probe_id
    # Build I_n_ADS spectrum: one harmonic at each mix (1, n) matching
    # the analytical I_n.
    n_indices = tuple(range(-N, N + 1))
    harmonics = tuple(
        HBHarmonic(
            mix=(1, n),
            frequency_Hz=f_signal + n * f_pump,
            amplitude=I_n_analytical[k],
        )
        for k, n in enumerate(n_indices)
    )
    branch_currents = {f"{probe_id}.i": HBSpectrum(quantity="", harmonics=harmonics)}

    hb = HBDataset(
        fundamental_freqs_Hz=(f_signal, f_pump),
        node_voltages=node_voltages,
        branch_currents=branch_currents,
    )

    result = compare_varactor_harmonics(
        hb, cm, _SPEC_L9_2CELL, 0, truncation_N=N,
    )
    for n, ratio in zip(result["harmonic_index"], result["ratio_dB"]):
        if math.isnan(ratio):
            # Some harmonics may be vanishingly small; nan there is OK.
            continue
        assert abs(ratio) < 1.0e-9, (
            f"harmonic n={n}: ratio_dB={ratio} should be ~0 in the round-trip."
        )
    assert result["max_abs_ratio_dB"] < 1.0e-9
    assert result["within_tolerance"] is True


def test_round_trip_2x_amplitude_yields_six_dB() -> None:
    """If I_n_ADS = 2 * I_n_analytical at every n, ratio_dB = +20*log10(2)."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    f_signal = 3.0e9
    f_pump = 6.0e9
    ws = 2.0 * math.pi * f_signal
    wm = 2.0 * math.pi * f_pump
    v_pump = 0.1 + 0j
    v_signal = 1.0 + 0j
    c0, c1 = 7.0e-12, -2.0e-12
    s0 = 1.0 / c0
    d_s = abs(c1) * abs(v_pump) / (c0 ** 2)
    N = 3
    Qn = v_signal * solve_harmonic_coefficients(s0, d_s, 0.0, 0.0, 0.0, ws, wm, N)
    _f, I_n_analytical = current_spectrum(Qn, ws, wm, N)

    branch_top_net = cm.cells[0].varactor_signal_net
    node_voltages = {
        branch_top_net: HBSpectrum(
            quantity="",
            harmonics=(
                HBHarmonic(mix=(1, 0), frequency_Hz=3e9, amplitude=v_signal),
                HBHarmonic(mix=(0, 1), frequency_Hz=6e9, amplitude=v_pump),
            ),
        ),
    }
    for k in range(1, len(cm.cells)):
        node_voltages[cm.cells[k].varactor_signal_net] = HBSpectrum(
            quantity="",
            harmonics=(
                HBHarmonic(mix=(1, 0), frequency_Hz=3e9, amplitude=v_signal),
                HBHarmonic(mix=(0, 1), frequency_Hz=6e9, amplitude=0.05 + 0j),
            ),
        )
    probe_id = cm.cells[0].varactor_probe_id
    n_indices = tuple(range(-N, N + 1))
    harmonics = tuple(
        HBHarmonic(
            mix=(1, n),
            frequency_Hz=f_signal + n * f_pump,
            amplitude=2.0 * I_n_analytical[k],  # 2x the analytical amplitude
        )
        for k, n in enumerate(n_indices)
    )
    branch_currents = {f"{probe_id}.i": HBSpectrum(quantity="", harmonics=harmonics)}

    hb = HBDataset(
        fundamental_freqs_Hz=(f_signal, f_pump),
        node_voltages=node_voltages,
        branch_currents=branch_currents,
    )
    result = compare_varactor_harmonics(
        hb, cm, _SPEC_L9_2CELL, 0, truncation_N=N,
    )
    expected_dB = 20.0 * math.log10(2.0)
    for n, ratio in zip(result["harmonic_index"], result["ratio_dB"]):
        if math.isnan(ratio):
            continue
        assert ratio == pytest.approx(expected_dB, abs=1e-9), (
            f"harmonic n={n}: expected {expected_dB:.4f} dB; got {ratio}."
        )
    # 6 dB exceeds every default tolerance band.
    assert result["within_tolerance"] is False


def test_round_trip_returns_zero_when_ADS_has_no_harmonic_at_mix() -> None:
    """A missing ADS mix tuple should map to ``I_n_ADS = 0`` -- ratio is nan."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    f_signal = 3.0e9
    f_pump = 6.0e9
    v_pump = 0.1 + 0j
    v_signal = 1.0 + 0j

    branch_top_net = cm.cells[0].varactor_signal_net
    node_voltages = {
        branch_top_net: HBSpectrum(
            quantity="",
            harmonics=(
                HBHarmonic(mix=(1, 0), frequency_Hz=3e9, amplitude=v_signal),
                HBHarmonic(mix=(0, 1), frequency_Hz=6e9, amplitude=v_pump),
            ),
        ),
    }
    for k in range(1, len(cm.cells)):
        node_voltages[cm.cells[k].varactor_signal_net] = HBSpectrum(
            quantity="",
            harmonics=(
                HBHarmonic(mix=(1, 0), frequency_Hz=3e9, amplitude=v_signal),
                HBHarmonic(mix=(0, 1), frequency_Hz=6e9, amplitude=0.05 + 0j),
            ),
        )
    probe_id = cm.cells[0].varactor_probe_id
    # Only carry the signal harmonic (mix=(1,0)); all other (1,n) absent.
    branch_currents = {
        f"{probe_id}.i": HBSpectrum(
            quantity="",
            harmonics=(
                HBHarmonic(mix=(1, 0), frequency_Hz=3e9, amplitude=0.001 + 0j),
            ),
        ),
    }
    hb = HBDataset(
        fundamental_freqs_Hz=(f_signal, f_pump),
        node_voltages=node_voltages,
        branch_currents=branch_currents,
    )
    result = compare_varactor_harmonics(
        hb, cm, _SPEC_L9_2CELL, 0, truncation_N=3,
    )
    # I_n_ADS at n != 0 is 0; ratio is NaN at those harmonics. n=0 has a
    # finite ratio.
    n0_idx = result["harmonic_index"].index(0)
    assert math.isfinite(result["ratio_dB"][n0_idx])
    nonzero_n = [
        result["ratio_dB"][k] for k in range(len(result["harmonic_index"]))
        if result["harmonic_index"][k] != 0
    ]
    for r in nonzero_n:
        assert math.isnan(r)


def test_compare_varactor_harmonics_truncation_size() -> None:
    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _synthetic_hb_pump_only(cm, cell_index=0, v_pump=0.1 + 0j)
    result = compare_varactor_harmonics(
        hb, cm, _SPEC_L9_2CELL, 0, truncation_N=7,
    )
    n = 7
    assert len(result["harmonic_index"]) == 2 * n + 1
    assert len(result["I_n_analytical"]) == 2 * n + 1
    assert len(result["I_n_ADS"]) == 2 * n + 1
    assert len(result["ratio_dB"]) == 2 * n + 1


def test_compare_invalid_pump_ratio_raises() -> None:
    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _synthetic_hb_pump_only(cm, cell_index=0, v_pump=0.1 + 0j)
    with pytest.raises(ValueError, match="pump_signal_ratio"):
        compare_varactor_harmonics(
            hb, cm, _SPEC_L9_2CELL, 0, pump_signal_ratio=0,
        )


def test_compare_invalid_truncation_raises() -> None:
    cm = build_cell_map(_SPEC_L9_2CELL)
    hb = _synthetic_hb_pump_only(cm, cell_index=0, v_pump=0.1 + 0j)
    with pytest.raises(ValueError, match="truncation_N"):
        compare_varactor_harmonics(
            hb, cm, _SPEC_L9_2CELL, 0, truncation_N=-1,
        )


# --------------------------------------------------------------------------
# compare_cascade_harmonics -- aggregation
# --------------------------------------------------------------------------


def test_cascade_runs_compare_on_every_cell() -> None:
    cm = build_cell_map(_SPEC_L9_3CELL)
    # Every cell gets its own V_pump; choose distinct values so each
    # comparison has a meaningful operating point.
    branch_currents: dict[str, HBSpectrum] = {}
    node_voltages: dict[str, HBSpectrum] = {}
    for k in range(3):
        v_pump_k = (0.05 + 0.01 * k) + 0j
        node_voltages[cm.cells[k].varactor_signal_net] = HBSpectrum(
            quantity="",
            harmonics=(
                HBHarmonic(mix=(1, 0), frequency_Hz=3e9, amplitude=1.0 + 0j),
                HBHarmonic(mix=(0, 1), frequency_Hz=6e9, amplitude=v_pump_k),
            ),
        )
        probe_id = cm.cells[k].varactor_probe_id
        branch_currents[f"{probe_id}.i"] = HBSpectrum(
            quantity="",
            harmonics=(HBHarmonic(mix=(1, 0), frequency_Hz=3e9, amplitude=0j),),
        )
    hb = HBDataset(
        fundamental_freqs_Hz=(3e9, 6e9),
        node_voltages=node_voltages,
        branch_currents=branch_currents,
    )
    result = compare_cascade_harmonics(
        hb, cm, _SPEC_L9_3CELL, truncation_N=3,
    )
    assert isinstance(result["per_cell"], tuple)
    assert len(result["per_cell"]) == 3
    assert len(result["max_abs_ratio_dB_per_cell"]) == 3


def test_facade_exports_floquet_comparison_symbols() -> None:
    """Public symbols are reachable from the v2 facade ``__all__``."""

    import tvtl_ads_builder_v2 as v2

    for sym in (
        "compare_varactor_harmonics",
        "compare_cascade_harmonics",
        "extract_varactor_parameters",
        "VaractorModelParameters",
        "harmonic_band",
        "harmonic_tolerance_dB",
        "DEFAULT_HARMONIC_TOLERANCE_DB",
    ):
        assert hasattr(v2, sym), sym
        assert sym in v2.__all__, sym


def test_compare_with_lift_stub_abcd_changes_n0_current() -> None:
    """Enabling the harmonic ABCD lift should change |I[n=0]|.

    The local-RLC solver (lift disabled) and the lifted solver evaluate
    different physics — the lifted solver adds the per-harmonic stub
    admittance to the diagonal terms. Their I_n_analytical predictions
    therefore differ at every harmonic (most visibly at n = 0). This
    test exercises both code paths on the same dataset and asserts the
    lift produces a numerically smaller prediction at n = 0.
    """

    cm = build_cell_map(_SPEC_L9_2CELL)
    branch_top_net = cm.cells[0].varactor_signal_net
    probe_id = cm.cells[0].varactor_probe_id
    f_signal = 3.0e9
    f_pump = 6.0e9
    v_signal = 1.0 + 0j
    v_pump = 0.1 + 0j
    node_voltages = {
        branch_top_net: HBSpectrum(
            quantity="",
            harmonics=(
                HBHarmonic(mix=(1, 0), frequency_Hz=f_signal, amplitude=v_signal),
                HBHarmonic(mix=(0, 1), frequency_Hz=f_pump, amplitude=v_pump),
            ),
        ),
    }
    for k in range(1, len(cm.cells)):
        node_voltages[cm.cells[k].varactor_signal_net] = HBSpectrum(
            quantity="",
            harmonics=(
                HBHarmonic(mix=(1, 0), frequency_Hz=f_signal, amplitude=v_signal),
                HBHarmonic(mix=(0, 1), frequency_Hz=f_pump, amplitude=0.05 + 0j),
            ),
        )
    branch_currents = {f"{probe_id}.i": HBSpectrum(quantity="", harmonics=())}
    hb = HBDataset(
        fundamental_freqs_Hz=(f_signal, f_pump),
        node_voltages=node_voltages,
        branch_currents=branch_currents,
    )
    result_local = compare_varactor_harmonics(
        hb, cm, _SPEC_L9_2CELL, 0, truncation_N=3, lift_stub_abcd=False,
    )
    result_lift = compare_varactor_harmonics(
        hb, cm, _SPEC_L9_2CELL, 0, truncation_N=3, lift_stub_abcd=True,
    )
    n0_idx = result_local["harmonic_index"].index(0)
    i_local_n0 = abs(result_local["I_n_analytical"][n0_idx])
    i_lift_n0 = abs(result_lift["I_n_analytical"][n0_idx])
    assert i_lift_n0 != i_local_n0
    assert i_lift_n0 < i_local_n0
    # The lift reduces |I[n=0]| by roughly the Z_var/(Z_var+Z_stub)
    # divider — at least 6 dB on Level-9.
    reduction_dB = 20.0 * math.log10(i_lift_n0 / i_local_n0)
    assert reduction_dB < -6.0


def test_compare_with_explicit_y_stub_fn_overrides_spec_extraction() -> None:
    """A user-supplied ``y_stub_fn`` overrides spec-driven extraction."""

    cm = build_cell_map(_SPEC_L9_2CELL)
    branch_top_net = cm.cells[0].varactor_signal_net
    probe_id = cm.cells[0].varactor_probe_id
    f_signal = 3.0e9
    f_pump = 6.0e9
    v_signal = 1.0 + 0j
    v_pump = 0.1 + 0j
    node_voltages = {
        branch_top_net: HBSpectrum(
            quantity="",
            harmonics=(
                HBHarmonic(mix=(1, 0), frequency_Hz=f_signal, amplitude=v_signal),
                HBHarmonic(mix=(0, 1), frequency_Hz=f_pump, amplitude=v_pump),
            ),
        ),
    }
    for k in range(1, len(cm.cells)):
        node_voltages[cm.cells[k].varactor_signal_net] = HBSpectrum(
            quantity="",
            harmonics=(
                HBHarmonic(mix=(1, 0), frequency_Hz=f_signal, amplitude=v_signal),
                HBHarmonic(mix=(0, 1), frequency_Hz=f_pump, amplitude=0.05 + 0j),
            ),
        )
    branch_currents = {f"{probe_id}.i": HBSpectrum(quantity="", harmonics=())}
    hb = HBDataset(
        fundamental_freqs_Hz=(f_signal, f_pump),
        node_voltages=node_voltages,
        branch_currents=branch_currents,
    )

    def y_fn(omega: float) -> complex:
        return 0.01j

    result = compare_varactor_harmonics(
        hb, cm, _SPEC_L9_2CELL, 0, truncation_N=2,
        lift_stub_abcd=True, y_stub_fn=y_fn,
    )
    n0_idx = result["harmonic_index"].index(0)
    assert math.isfinite(abs(result["I_n_analytical"][n0_idx]))


def test_cascade_compare_with_lift_stub_abcd_forwards_flag() -> None:
    """``compare_cascade_harmonics`` forwards the lift flag to every cell."""

    cm = build_cell_map(_SPEC_L9_3CELL)
    f_signal = 3.0e9
    f_pump = 6.0e9
    node_voltages: dict[str, HBSpectrum] = {}
    branch_currents: dict[str, HBSpectrum] = {}
    for k in range(3):
        v_pump_k = (0.05 + 0.01 * k) + 0j
        node_voltages[cm.cells[k].varactor_signal_net] = HBSpectrum(
            quantity="",
            harmonics=(
                HBHarmonic(mix=(1, 0), frequency_Hz=f_signal, amplitude=1.0 + 0j),
                HBHarmonic(mix=(0, 1), frequency_Hz=f_pump, amplitude=v_pump_k),
            ),
        )
        probe_id = cm.cells[k].varactor_probe_id
        branch_currents[f"{probe_id}.i"] = HBSpectrum(quantity="", harmonics=())
    hb = HBDataset(
        fundamental_freqs_Hz=(f_signal, f_pump),
        node_voltages=node_voltages,
        branch_currents=branch_currents,
    )
    result_local = compare_cascade_harmonics(
        hb, cm, _SPEC_L9_3CELL, truncation_N=3, lift_stub_abcd=False,
    )
    result_lift = compare_cascade_harmonics(
        hb, cm, _SPEC_L9_3CELL, truncation_N=3, lift_stub_abcd=True,
    )
    for k in range(3):
        n0_idx = result_local["per_cell"][k]["harmonic_index"].index(0)
        i_local = abs(result_local["per_cell"][k]["I_n_analytical"][n0_idx])
        i_lift = abs(result_lift["per_cell"][k]["I_n_analytical"][n0_idx])
        assert i_lift < i_local, f"cell {k}: lift did not reduce I[n=0]"


def test_v_signal_at_branch_top_is_returned_for_diagnostic() -> None:
    cm = build_cell_map(_SPEC_L9_2CELL)
    v_signal = 0.7 - 0.3j
    branch_top_net = cm.cells[0].varactor_signal_net
    probe_id = cm.cells[0].varactor_probe_id
    node_voltages = {
        branch_top_net: HBSpectrum(
            quantity="",
            harmonics=(
                HBHarmonic(mix=(1, 0), frequency_Hz=3e9, amplitude=v_signal),
                HBHarmonic(mix=(0, 1), frequency_Hz=6e9, amplitude=0.05 + 0j),
            ),
        ),
    }
    for k in range(1, len(cm.cells)):
        node_voltages[cm.cells[k].varactor_signal_net] = HBSpectrum(
            quantity="",
            harmonics=(
                HBHarmonic(mix=(1, 0), frequency_Hz=3e9, amplitude=1.0 + 0j),
                HBHarmonic(mix=(0, 1), frequency_Hz=6e9, amplitude=0.05 + 0j),
            ),
        )
    branch_currents = {
        f"{probe_id}.i": HBSpectrum(quantity="", harmonics=()),
    }
    hb = HBDataset(
        fundamental_freqs_Hz=(3e9, 6e9),
        node_voltages=node_voltages,
        branch_currents=branch_currents,
    )
    result = compare_varactor_harmonics(
        hb, cm, _SPEC_L9_2CELL, 0, truncation_N=2,
    )
    assert result["V_signal_at_branch_top"] == pytest.approx(v_signal)


