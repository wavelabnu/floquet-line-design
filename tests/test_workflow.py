"""Unit tests for the workflow orchestrator + diversification (base Python)."""

from __future__ import annotations

import numpy as np
import pytest

from tvtl.optimization.stage1_per_cell import StubCandidate
from tvtl.optimization.topology_generator import (
    Termination,
    TopologyFamily,
    ViaConfig,
)
from tvtl.optimization.workflow import (
    WorkflowConfig,
    decoupled_design_workflow,
    param_distance,
    select_diverse_top_m,
)
from tvtl.varactors import RealisticVaractor

_VAR = RealisticVaractor(
    CJ0_f=0.3e-12, VJ_v=3.5, M=1.7, CP_f=0.0, L_pkg_h=0.3e-9, R_series_ohm=0.5
)
_FAM_A = TopologyFamily(2, ViaConfig.DOUBLE, Termination.SHORT)
_FAM_B = TopologyFamily(2, ViaConfig.NONE, Termination.OPEN)


def _cand(fam, params, a1, f0=3.1e9):
    return StubCandidate(topology=fam, params=tuple(params), a1_score=a1, f0_Hz=f0)


def test_param_distance_same_vs_diff_family():
    c1 = _cand(_FAM_A, [1e-3, 2e-3, 1.5e-3, 3e-3], 0.2)
    c2 = _cand(_FAM_A, [1e-3, 2e-3, 1.5e-3, 3e-3], 0.2)
    c3 = _cand(_FAM_B, [1e-3, 2e-3, 1.5e-3, 3e-3], 0.2)
    assert param_distance(c1, c2) == 0.0  # identical
    assert param_distance(c1, c3) == float("inf")  # different family
    c4 = _cand(_FAM_A, [4e-3, 12e-3, 4e-3, 12e-3], 0.2)  # far apart
    assert param_distance(c1, c4) > 0.3


def test_select_diverse_empty_and_single():
    assert select_diverse_top_m([]) == []
    c1 = _cand(_FAM_A, [1e-3, 2e-3, 1.5e-3, 3e-3], 0.2)
    assert select_diverse_top_m([c1], m=3) == [c1]


def test_select_diverse_rejects_near_duplicate_same_family():
    best = _cand(_FAM_A, [1e-3, 2e-3, 1.5e-3, 3e-3], 0.21)
    near = _cand(_FAM_A, [1.01e-3, 2.01e-3, 1.51e-3, 3.01e-3], 0.20)  # tiny delta
    far = _cand(_FAM_A, [4.5e-3, 13e-3, 0.5e-3, 1e-3], 0.19)  # large delta, same family
    out = select_diverse_top_m([best, near, far], m=3, param_distance_threshold=0.30)
    assert best in out
    assert near not in out  # near-duplicate rejected
    assert far in out       # far same-family accepted


def test_select_diverse_accepts_different_family():
    best = _cand(_FAM_A, [1e-3, 2e-3, 1.5e-3, 3e-3], 0.21)
    other_fam = _cand(_FAM_B, [1e-3, 2e-3, 1.5e-3, 3e-3], 0.20)
    out = select_diverse_top_m([best, other_fam], m=3, param_distance_threshold=0.30)
    assert len(out) == 2  # different family always diversifies


def test_workflow_end_to_end_analytical():
    cfg = WorkflowConfig(n_samples=96, top_k=10, top_m=3, run_ads=False)
    result = decoupled_design_workflow(3.1e9, _VAR, cfg)
    assert 1 <= len(result.reports) <= 3
    assert result.n_realizable >= result.n_positive_gain >= result.n_diverse
    # design examples are distinct
    keys = [
        (r.design.stub.topology.key, tuple(np.round(r.design.stub.params_array * 1e3, 2)))
        for r in result.reports
    ]
    assert len(set(keys)) == len(keys)
    # all selected examples have positive coherent peak gain (the min filter)
    for r in result.reports:
        assert r.analytical[-3.0]["coherent_peak"] > 0.0
        assert r.passed is None  # analytical-only run


def test_workflow_min_coherent_filter():
    """Raising min_coherent_dB drops weak designs (fewer or equal examples)."""
    base = decoupled_design_workflow(3.1e9, _VAR, WorkflowConfig(n_samples=96, top_m=3, min_coherent_dB=0.0))
    strict = decoupled_design_workflow(3.1e9, _VAR, WorkflowConfig(n_samples=96, top_m=3, min_coherent_dB=7.0))
    assert strict.n_positive_gain <= base.n_positive_gain
    for r in strict.reports:
        assert r.analytical[-3.0]["coherent_peak"] > 7.0


def test_phase5e_example_reproducible():
    """Reproducibility anchor: the workflow (SHORT + [1,5] dB window,
    seed=0) deterministically reproduces the A.3.1 design example
    (n1_double_short, the ADS-cross-validated +15.08 dB design), with its exact
    stub geometry. Analytical only -- the ADS gain is re-confirmed in the
    scratch-tree export, not here."""
    from tvtl.optimization.topology_generator import Termination, linear_families

    short = tuple(f for f in linear_families() if f.termination == Termination.SHORT)
    cfg = WorkflowConfig(n_samples=300, seed=0, top_k=10, top_m=3, n_cells=15,
                         drive_dBm=-3.0, run_ads=False, families=short,
                         coherent_window_dB=(1.0, 5.0), order_by="coherent_peak")
    wf = decoupled_design_workflow(3.1e9, _VAR, cfg)
    by_topo = {r.design.stub.topology.key: r.design for r in wf.reports}
    assert "n1_double_short" in by_topo, "reference example not reproduced"
    d = by_topo["n1_double_short"]
    p = d.stub.params_array
    assert p[0] * 1e3 == pytest.approx(1.1268, abs=2e-3)   # stub width mm
    assert p[1] * 1e3 == pytest.approx(9.405, abs=2e-3)    # stub length mm
    assert d.main_tl.z_tl_ohm == pytest.approx(54.92, abs=0.2)
    assert d.input_match.topology == "single_stub"


def test_workflow_coherent_window_selection():
    """The gain-target window selects only moderate-gain designs."""
    win = decoupled_design_workflow(
        3.1e9, _VAR, WorkflowConfig(n_samples=128, top_m=3, coherent_window_dB=(1.0, 5.0), order_by="coherent_peak"),
    )
    # every selected example's analytical coherent peak is inside the window
    for r in win.reports:
        cp = r.analytical[-3.0]["coherent_peak"]
        assert 1.0 <= cp <= 5.0
    # the window excludes the gain-max oscillators that the default would rank first
    gainmax = decoupled_design_workflow(3.1e9, _VAR, WorkflowConfig(n_samples=128, top_m=3))
    if gainmax.reports:
        assert gainmax.reports[0].analytical[-3.0]["coherent_peak"] > 5.0


