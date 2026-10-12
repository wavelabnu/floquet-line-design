"""ADS-free unit tests for the sweep runner's control flow.

The live HB runs are exercised by the integration tests and the
validation scripts; here we test the ADS-absent skip path (the
runner must return every point blocked rather than raising) and the
:class:`SweepResult` / :class:`SweepPointResult` aggregation helpers.
"""

from __future__ import annotations

import pytest

from tvtl_ads_builder_v2.device_spec import SweepSpec
from tvtl_ads_builder_v2.parity.canonical_circuits import (
    make_paper_geometry_tvtl_cascade,
)
from tvtl_ads_builder_v2.simulate import sweep_runner as sr
from tvtl_ads_builder_v2.simulate.sweep_runner import (
    SweepPointResult,
    SweepResult,
    SweepRunner,
)


def _sweep():
    return SweepSpec(
        canonical=make_paper_geometry_tvtl_cascade(n_cells=2),
        sweep_variable="f_signal",
        sweep_values=(2.5e9, 3.1e9, 3.7e9),
        point_overrides=lambda fs: {"f_signal": fs, "f_pump": 2.0 * fs},
        fixed_overrides={"pump_power_dBm": 16.0},
    )


def test_runner_blocks_cleanly_without_ads(tmp_path, monkeypatch) -> None:
    # Force the ADS-unavailable branch regardless of the host.
    monkeypatch.setattr(sr, "ads_runtime_available", lambda: False)
    runner = SweepRunner(tmp_path / "sweep_out")
    result = runner.run(_sweep())

    assert isinstance(result, SweepResult)
    assert len(result.points) == 3
    assert result.all_blocked
    assert result.n_completed == 0
    for point in result.points:
        assert point.status == "blocked_missing_ads_runtime"
        assert point.dataset is None
        assert not point.completed
        assert point.error_message
        # The override mapping is recorded even for a blocked point.
        assert point.overrides["f_signal"] == point.value
        assert point.overrides["f_pump"] == 2.0 * point.value
        assert point.overrides["pump_power_dBm"] == 16.0
    # No workspace directory is created when the runtime is unavailable.
    assert not (tmp_path / "sweep_out").exists()


def test_runner_does_not_create_outdir_when_blocked(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sr, "ads_runtime_available", lambda: False)
    out = tmp_path / "never_made"
    SweepRunner(out).run(_sweep())
    assert not out.exists()


# --------------------------------------------------------------------------
# SweepResult / SweepPointResult aggregation
# --------------------------------------------------------------------------


class _FakeDataset:
    """Minimal stand-in -- the helpers only check identity / non-None."""


def _point(value, status, dataset=None) -> SweepPointResult:
    return SweepPointResult(value=value, overrides={}, status=status, dataset=dataset)


def test_sweeppointresult_completed_property() -> None:
    assert _point(1.0, "completed", _FakeDataset()).completed
    assert not _point(1.0, "completed", None).completed  # no dataset
    assert not _point(1.0, "error", _FakeDataset()).completed


def test_sweepresult_filters_and_orders_completed_points() -> None:
    d1, d3 = _FakeDataset(), _FakeDataset()
    points = (
        _point(1.0, "completed", d1),
        _point(2.0, "error", None),
        _point(3.0, "completed", d3),
        _point(4.0, "timeout", None),
    )
    result = SweepResult(sweep_spec=_sweep(), points=points)
    assert result.n_completed == 2
    assert result.values() == (1.0, 2.0, 3.0, 4.0)
    completed = result.completed_points()
    assert tuple(p.value for p in completed) == (1.0, 3.0)
    assert result.datasets() == (d1, d3)
    assert not result.all_blocked


def test_sweepresult_all_blocked_flag() -> None:
    points = tuple(
        _point(v, "blocked_missing_ads_runtime") for v in (1.0, 2.0)
    )
    assert SweepResult(sweep_spec=_sweep(), points=points).all_blocked


