"""ADS-free unit tests for analysis/sweep.py gain extraction.

The curves are computed from synthetic :class:`HBDataset` objects wrapped
in synthetic :class:`SweepResult` points, so no simulator is needed:

* ``source="ads"`` end-to-end gain on a Level-9 cascade (fully NAMEd
  boundaries) -- power and voltage conventions, NaN guard, x-axis sorting;
* ``source="analytical"`` harmonic-ABCD-lift proxy on a Level-10 paper
  mini-cascade (plumbing: finite output, NaN beyond the elastance limit);
* the error paths (bad convention/source, per-cell on Level-10).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tvtl_ads_builder_v2.analysis import (
    HBDataset,
    HBHarmonic,
    HBSpectrum,
    build_cell_map,
    gain_vs_pump_power,
    gain_vs_signal_frequency,
)
from tvtl_ads_builder_v2.device_spec import SweepSpec
from tvtl_ads_builder_v2.parity.canonical_circuits import (
    LEVEL_9_CANONICAL_CIRCUITS,
    make_paper_geometry_tvtl_cascade,
)
from tvtl_ads_builder_v2.simulate.sweep_runner import SweepPointResult, SweepResult

_FS = 3.1e9
_FP = 6.2e9


def _completed_point(value, dataset) -> SweepPointResult:
    return SweepPointResult(
        value=float(value), overrides={"f_signal": float(value)},
        status="completed", fundamentals_Hz=(_FS, _FP), dataset=dataset,
    )


# --------------------------------------------------------------------------
# source="ads": end-to-end boundary gain on Level-9
# --------------------------------------------------------------------------

_SPEC_L9 = LEVEL_9_CANONICAL_CIRCUITS[0]  # 2-cell probed cascade


def _boundary_dataset(*, v_in, i_in, v_out, i_out) -> HBDataset:
    """Synthesise a dataset with the L9 design-boundary V/I at mix (1,0)."""

    cm = build_cell_map(_SPEC_L9)
    in_probe = cm.cells[0].boundary_in_probe_id
    out_probe = cm.cells[-1].boundary_out_probe_id
    in_elem = next(e for e in _SPEC_L9.elements if e.id == in_probe)
    out_elem = next(e for e in _SPEC_L9.elements if e.id == out_probe)

    node_voltages: dict[str, HBSpectrum] = {}
    for pin in in_elem.pins:
        node_voltages[pin.node] = HBSpectrum(
            pin.node,
            (
                HBHarmonic((1, 0), _FS, complex(v_in)),
                HBHarmonic((0, 1), _FP, 0j),
            ),
        )
    for pin in out_elem.pins:
        node_voltages[pin.node] = HBSpectrum(
            pin.node,
            (
                HBHarmonic((1, 0), _FS, complex(v_out)),
                HBHarmonic((0, 1), _FP, 0j),
            ),
        )
    branch_currents = {
        f"{in_probe}.i": HBSpectrum(
            f"{in_probe}.i",
            (HBHarmonic((1, 0), _FS, complex(i_in)), HBHarmonic((0, 1), _FP, 0j)),
        ),
        f"{out_probe}.i": HBSpectrum(
            f"{out_probe}.i",
            (HBHarmonic((1, 0), _FS, complex(i_out)), HBHarmonic((0, 1), _FP, 0j)),
        ),
    }
    return HBDataset(
        fundamental_freqs_Hz=(_FS, _FP),
        node_voltages=node_voltages,
        branch_currents=branch_currents,
    )


def _ads_sweep_result(points):
    sweep = SweepSpec(
        canonical=_SPEC_L9, sweep_variable="f_signal",
        sweep_values=tuple(v for v, _ in points),
    )
    return SweepResult(
        sweep_spec=sweep,
        points=tuple(_completed_point(v, d) for v, d in points),
    )


def test_ads_end_to_end_power_gain_matches_formula() -> None:
    # P_in = 0.5*Re(V_in*conj(I_in)); P_out likewise; gain = 10log10(ratio).
    ds = _boundary_dataset(v_in=1.0 + 0j, i_in=0.01 + 0j,
                           v_out=2.0 + 0j, i_out=0.02 + 0j)
    result = _ads_sweep_result([(_FS, ds)])
    x, y = gain_vs_signal_frequency(result, gain_convention="power")
    p_in = 0.5 * 0.01
    p_out = 0.5 * 2.0 * 0.02
    assert x.tolist() == [_FS]
    assert y[0] == pytest.approx(10.0 * math.log10(p_out / p_in))


def test_ads_end_to_end_voltage_gain_matches_formula() -> None:
    ds = _boundary_dataset(v_in=1.0 + 0j, i_in=0.01 + 0j,
                           v_out=2.0 + 0j, i_out=0.005 + 0j)
    result = _ads_sweep_result([(_FS, ds)])
    _x, y = gain_vs_signal_frequency(result, gain_convention="voltage")
    assert y[0] == pytest.approx(20.0 * math.log10(2.0 / 1.0))


def test_ads_curve_sorted_by_signal_frequency() -> None:
    # Provide points out of order; the curve must come back ascending in x.
    pts = [
        (3.7e9, _boundary_dataset(v_in=1 + 0j, i_in=0.01 + 0j, v_out=1 + 0j, i_out=0.01 + 0j)),
        (2.5e9, _boundary_dataset(v_in=1 + 0j, i_in=0.01 + 0j, v_out=3 + 0j, i_out=0.03 + 0j)),
        (3.1e9, _boundary_dataset(v_in=1 + 0j, i_in=0.01 + 0j, v_out=2 + 0j, i_out=0.02 + 0j)),
    ]
    result = _ads_sweep_result(pts)
    x, y = gain_vs_signal_frequency(result, gain_convention="power")
    assert list(x) == sorted(x)
    assert x.tolist() == [2.5e9, 3.1e9, 3.7e9]
    assert np.all(np.isfinite(y))


def test_ads_power_gain_nan_when_power_non_positive() -> None:
    # Reverse-flowing output power (negative real part) -> NaN, not a crash.
    ds = _boundary_dataset(v_in=1.0 + 0j, i_in=0.01 + 0j,
                           v_out=1.0 + 0j, i_out=-0.01 + 0j)
    _x, y = gain_vs_signal_frequency(_ads_sweep_result([(_FS, ds)]),
                                     gain_convention="power")
    assert math.isnan(y[0])


def test_pump_power_axis_is_swept_value() -> None:
    ds = _boundary_dataset(v_in=1 + 0j, i_in=0.01 + 0j, v_out=2 + 0j, i_out=0.02 + 0j)
    sweep = SweepSpec(
        canonical=_SPEC_L9, sweep_variable="pump_power_dBm",
        sweep_values=(-10.0, 5.0),
    )
    result = SweepResult(
        sweep_spec=sweep,
        points=(
            SweepPointResult(value=5.0, overrides={"pump_power_dBm": 5.0},
                             status="completed", dataset=ds),
            SweepPointResult(value=-10.0, overrides={"pump_power_dBm": -10.0},
                             status="completed", dataset=ds),
        ),
    )
    x, _y = gain_vs_pump_power(result, gain_convention="power")
    assert x.tolist() == [-10.0, 5.0]


def test_empty_completed_points_returns_empty_arrays() -> None:
    sweep = SweepSpec(canonical=_SPEC_L9, sweep_variable="f_signal",
                      sweep_values=(_FS,))
    result = SweepResult(
        sweep_spec=sweep,
        points=(SweepPointResult(value=_FS, overrides={}, status="error"),),
    )
    x, y = gain_vs_signal_frequency(result)
    assert x.size == 0 and y.size == 0


# --------------------------------------------------------------------------
# argument validation
# --------------------------------------------------------------------------


def test_invalid_gain_convention_raises() -> None:
    result = _ads_sweep_result(
        [(_FS, _boundary_dataset(v_in=1 + 0j, i_in=0.01 + 0j,
                                 v_out=1 + 0j, i_out=0.01 + 0j))]
    )
    with pytest.raises(ValueError, match="gain_convention"):
        gain_vs_signal_frequency(result, gain_convention="bogus")


def test_invalid_source_raises() -> None:
    result = _ads_sweep_result(
        [(_FS, _boundary_dataset(v_in=1 + 0j, i_in=0.01 + 0j,
                                 v_out=1 + 0j, i_out=0.01 + 0j))]
    )
    with pytest.raises(ValueError, match="source"):
        gain_vs_signal_frequency(result, source="nope")


def test_analytical_per_cell_raises() -> None:
    result = _ads_sweep_result(
        [(_FS, _boundary_dataset(v_in=1 + 0j, i_in=0.01 + 0j,
                                 v_out=1 + 0j, i_out=0.01 + 0j))]
    )
    with pytest.raises(ValueError, match="per-cell"):
        gain_vs_signal_frequency(result, source="analytical", cell_index=0)


# --------------------------------------------------------------------------
# source="analytical": harmonic-ABCD-lift proxy on a Level-10 mini-cascade
# --------------------------------------------------------------------------

_SPEC_L10_MINI = make_paper_geometry_tvtl_cascade(n_cells=2)


def _pump_dataset(v_pump: complex) -> HBDataset:
    """Dataset providing each cell's branch-top pump voltage at mix (0,1)."""

    cm = build_cell_map(_SPEC_L10_MINI)
    node_voltages: dict[str, HBSpectrum] = {}
    for cell in cm.cells:
        node_voltages[cell.branch_top_net] = HBSpectrum(
            cell.branch_top_net,
            (
                HBHarmonic((1, 0), _FS, 0.1 + 0j),
                HBHarmonic((0, 1), _FP, complex(v_pump)),
            ),
        )
    return HBDataset(
        fundamental_freqs_Hz=(_FS, _FP),
        node_voltages=node_voltages,
        branch_currents={},
    )


def _l10_sweep_result(points):
    sweep = SweepSpec(
        canonical=_SPEC_L10_MINI, sweep_variable="pump_power_dBm",
        sweep_values=tuple(v for v, _ in points),
    )
    return SweepResult(
        sweep_spec=sweep,
        points=tuple(
            SweepPointResult(value=v, overrides={"pump_power_dBm": v},
                             status="completed", fundamentals_Hz=(_FS, _FP),
                             dataset=d)
            for v, d in points
        ),
    )


def test_analytical_source_returns_finite_gain() -> None:
    result = _l10_sweep_result(
        [(-10.0, _pump_dataset(0.1 + 0j)), (5.0, _pump_dataset(0.5 + 0j))]
    )
    x, y = gain_vs_pump_power(result, source="analytical", truncation_N=4)
    assert x.tolist() == [-10.0, 5.0]
    assert np.all(np.isfinite(y))
    # A passive-ish 2-cell proxy gain sits in a sane dB window.
    assert np.all(y > -60.0) and np.all(y < 60.0)


def test_analytical_source_nan_beyond_elastance_limit() -> None:
    # A very large branch-top pump voltage exceeds the small-signal
    # elastance linearisation; the analytical leg returns NaN (the ADS
    # leg, if present, would be unaffected).
    result = _l10_sweep_result([(20.0, _pump_dataset(50.0 + 0j))])
    _x, y = gain_vs_pump_power(result, source="analytical", truncation_N=4)
    assert math.isnan(y[0])


