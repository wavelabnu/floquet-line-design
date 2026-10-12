"""ADS-free unit tests for the SweepSpec / with_var_overrides API.

Covers the VAR-block swap mechanism, sweep-value resolution (including a
coupled override that drives ``f_pump = 2 * f_signal``), the
fixed-override merge, and the validation error paths.
"""

from __future__ import annotations

import math

import pytest

from tvtl_ads_builder_v2.device_spec import DeviceSpecError, SweepSpec, with_var_overrides
from tvtl_ads_builder_v2.parity.canonical_circuits import (
    LEVEL_1_CANONICAL_CIRCUITS,
    make_paper_geometry_tvtl_cascade,
)


def _canonical():
    # A 2-cell paper-shape mini-cascade carries the five-scalar VAR block
    # (f_signal, f_pump, signal_power_dBm, pump_power_dBm, pump_phase_rad)
    # and builds far faster than the 15-cell device.
    return make_paper_geometry_tvtl_cascade(n_cells=2)


def _var_values(spec) -> dict[str, float]:
    return {v.name: float(v.value) for v in spec.var_block.variables}


# --------------------------------------------------------------------------
# with_var_overrides
# --------------------------------------------------------------------------


def test_with_var_overrides_swaps_only_named_values() -> None:
    spec = _canonical()
    before = _var_values(spec)
    out = with_var_overrides(spec, {"f_signal": 4.0e9})
    after = _var_values(out)
    assert after["f_signal"] == 4.0e9
    # every other scalar unchanged
    for name, value in before.items():
        if name != "f_signal":
            assert after[name] == value
    # topology untouched (same element/node counts) and original unmutated
    assert len(out.elements) == len(spec.elements)
    assert len(out.nodes) == len(spec.nodes)
    assert _var_values(spec)["f_signal"] == before["f_signal"]


def test_with_var_overrides_preserves_named_references() -> None:
    # The sources / HB controller must still reference the VAR names so the
    # var-resolver picks up the swapped values at emit time.
    spec = with_var_overrides(_canonical(), {"f_signal": 4.0e9})
    assert spec.hb_two_tone.fundamental_freq_1 == "f_signal"
    assert any(s.freq == "f_signal" for s in spec.sources)


def test_with_var_overrides_rejects_unknown_variable() -> None:
    with pytest.raises(DeviceSpecError, match="not.*declared"):
        with_var_overrides(_canonical(), {"nope": 1.0})


def test_with_var_overrides_requires_var_block() -> None:
    no_var_spec = LEVEL_1_CANONICAL_CIRCUITS[0]
    assert no_var_spec.var_block is None
    with pytest.raises(DeviceSpecError, match="VAR block"):
        with_var_overrides(no_var_spec, {"x": 1.0})


def test_with_var_overrides_rejects_non_finite() -> None:
    with pytest.raises(DeviceSpecError, match="finite"):
        with_var_overrides(_canonical(), {"f_signal": math.inf})


# --------------------------------------------------------------------------
# SweepSpec validation
# --------------------------------------------------------------------------


def test_sweepspec_rejects_unknown_sweep_variable() -> None:
    with pytest.raises(DeviceSpecError, match="sweep_variable"):
        SweepSpec(
            canonical=_canonical(),
            sweep_variable="not_a_var",
            sweep_values=(1.0, 2.0),
        )


def test_sweepspec_rejects_empty_values() -> None:
    with pytest.raises(DeviceSpecError, match="non-empty"):
        SweepSpec(
            canonical=_canonical(),
            sweep_variable="f_signal",
            sweep_values=(),
        )


def test_sweepspec_rejects_canonical_without_var_block() -> None:
    with pytest.raises(DeviceSpecError, match="VAR block"):
        SweepSpec(
            canonical=LEVEL_1_CANONICAL_CIRCUITS[0],
            sweep_variable="f_signal",
            sweep_values=(1.0,),
        )


def test_sweepspec_rejects_unknown_fixed_override() -> None:
    with pytest.raises(DeviceSpecError, match="fixed_overrides"):
        SweepSpec(
            canonical=_canonical(),
            sweep_variable="f_signal",
            sweep_values=(3.0e9,),
            fixed_overrides={"bogus": 1.0},
        )


def test_sweepspec_rejects_override_callable_returning_unknown_key() -> None:
    with pytest.raises(DeviceSpecError, match="undeclared"):
        SweepSpec(
            canonical=_canonical(),
            sweep_variable="f_signal",
            sweep_values=(3.0e9,),
            point_overrides=lambda v: {"f_signal": v, "ghost": 2 * v},
        )


def test_sweepspec_label_defaults_to_variable() -> None:
    sweep = SweepSpec(
        canonical=_canonical(),
        sweep_variable="pump_power_dBm",
        sweep_values=(-10.0, 0.0, 10.0),
    )
    assert sweep.label == "pump_power_dBm"


# --------------------------------------------------------------------------
# SweepSpec resolution
# --------------------------------------------------------------------------


def test_default_point_overrides_is_identity() -> None:
    sweep = SweepSpec(
        canonical=_canonical(),
        sweep_variable="f_signal",
        sweep_values=(2.5e9, 3.1e9),
    )
    assert sweep.resolved_overrides(2.5e9) == {"f_signal": 2.5e9}


def test_coupled_point_overrides_drives_pump_at_2x() -> None:
    # Figure-6b coupling: pump frequency tracks the signal at the 2:1 ratio.
    sweep = SweepSpec(
        canonical=_canonical(),
        sweep_variable="f_signal",
        sweep_values=(2.5e9, 3.1e9),
        point_overrides=lambda fs: {"f_signal": fs, "f_pump": 2.0 * fs},
        fixed_overrides={"pump_power_dBm": 16.0},
    )
    ov = sweep.resolved_overrides(3.1e9)
    assert ov["f_signal"] == 3.1e9
    assert ov["f_pump"] == 6.2e9
    assert ov["pump_power_dBm"] == 16.0
    # spec_for must bake those into the VAR block
    spec = sweep.spec_for(3.1e9)
    vv = _var_values(spec)
    assert vv["f_signal"] == 3.1e9
    assert vv["f_pump"] == 6.2e9
    assert vv["pump_power_dBm"] == 16.0


def test_point_override_wins_over_fixed_on_collision() -> None:
    sweep = SweepSpec(
        canonical=_canonical(),
        sweep_variable="f_signal",
        sweep_values=(3.1e9,),
        point_overrides=lambda fs: {"f_signal": fs},
        fixed_overrides={"f_signal": 1.0e9, "pump_power_dBm": 16.0},
    )
    ov = sweep.resolved_overrides(3.1e9)
    assert ov["f_signal"] == 3.1e9  # point override, not the fixed 1 GHz
    assert ov["pump_power_dBm"] == 16.0


def test_var_names_property() -> None:
    sweep = SweepSpec(
        canonical=_canonical(),
        sweep_variable="f_signal",
        sweep_values=(3.1e9,),
    )
    assert "f_signal" in sweep.var_names
    assert "pump_power_dBm" in sweep.var_names


