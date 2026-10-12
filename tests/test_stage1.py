"""Unit tests for Stage-1 coarse screen + refinement (base Python)."""

from __future__ import annotations

import numpy as np
import pytest

from tvtl.optimization.stage1_per_cell import coarse_screen, refine_topk
from tvtl.optimization.substrate import ROGERS_4003C
from tvtl.optimization.topology_generator import (
    Termination,
    TopologyFamily,
    ViaConfig,
    enumerate_families,
    linear_families,
)
from tvtl.varactors import RealisticVaractor

_VAR = RealisticVaractor(
    CJ0_f=0.3e-12, VJ_v=3.5, M=1.7, CP_f=0.0, L_pkg_h=0.3e-9, R_series_ohm=0.5
)

# A small family subset keeps the unit test fast (the full screen is a heavier run).
_FAMS = [
    TopologyFamily(1, ViaConfig.SINGLE, Termination.SHORT),
    TopologyFamily(2, ViaConfig.DOUBLE, Termination.SHORT),
    TopologyFamily(2, ViaConfig.NONE, Termination.OPEN),
    TopologyFamily(2, ViaConfig.SINGLE, Termination.COMBINED),
]


def test_coarse_screen_structure():
    res = coarse_screen(
        3.1e9, _VAR, substrate=ROGERS_4003C, families=_FAMS, n_samples=48,
        seed=0, top_k=5,
    )
    assert len(res.linear_topk) <= 5
    assert len(res.linear_topk) > 0
    # sorted descending by A1
    scores = [c.a1_score for c in res.linear_topk]
    assert scores == sorted(scores, reverse=True)
    # only linear families in linear_topk
    assert all(c.topology.is_linear for c in res.linear_topk)
    # all scored candidates are finite, positive
    assert all(np.isfinite(c.a1_score) for c in res.scored)
    assert res.top_linear_score == res.linear_topk[0].a1_score


def test_coarse_screen_reproducible():
    kw = dict(substrate=ROGERS_4003C, families=_FAMS, n_samples=32, seed=11, top_k=3)
    a = coarse_screen(3.1e9, _VAR, **kw)
    b = coarse_screen(3.1e9, _VAR, **kw)
    assert [c.topology.key for c in a.linear_topk] == [c.topology.key for c in b.linear_topk]
    assert np.allclose(
        [c.a1_score for c in a.linear_topk], [c.a1_score for c in b.linear_topk]
    )


def test_refine_does_not_worsen():
    res = coarse_screen(
        3.1e9, _VAR, substrate=ROGERS_4003C, families=_FAMS, n_samples=48,
        seed=0, top_k=5,
    )
    refined = refine_topk(
        res.linear_topk, 3.1e9, _VAR, substrate=ROGERS_4003C, top_n=3, maxiter=80,
    )
    assert 0 < len(refined) <= 3
    # refined scores sorted desc
    rscores = [c.a1_score for c in refined]
    assert rscores == sorted(rscores, reverse=True)
    # refinement never worsens the best coarse score it started from
    best_coarse = res.linear_topk[0].a1_score
    assert refined[0].a1_score >= best_coarse - 1e-12


def test_combined_flagging_runs():
    """COMBINED candidates are scored and the 20%-of-top-linear flag is computed."""
    res = coarse_screen(
        3.1e9, _VAR, substrate=ROGERS_4003C, families=_FAMS, n_samples=64,
        seed=1, top_k=5,
    )
    # combined_flagged is a (possibly empty) list of COMBINED candidates
    assert isinstance(res.combined_flagged, list)
    assert all(not c.topology.is_linear for c in res.combined_flagged)


def test_full_20_family_screen_smoke():
    """Sanity that the full 20-family screen runs (reduced N for test speed)."""
    res = coarse_screen(
        3.1e9, _VAR, substrate=ROGERS_4003C, families=enumerate_families(),
        n_samples=16, seed=0, top_k=10,
    )
    assert len(res.linear_topk) == 10
    assert len(linear_families()) == 12


