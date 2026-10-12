"""Shared pytest fixtures for the TVTL test suite."""
from __future__ import annotations

import pytest

# hpeesofsim writes datasets and artifacts several directory levels below the
# workspace root. On Windows the full path can exceed MAX_PATH (260) and the
# simulator then crashes opaquely rather than reporting a path error. A deep
# pytest tmp_path (``.../pytest-of-<user>/pytest-<N>/<testname><M>/...``) is the
# usual trigger, which is why the failure looks "intermittent" (it tracks the
# username and the per-run counter, not any real flakiness). This budget leaves
# headroom for the nested workspace / dataset names the ADS side appends.
_ADS_WORKSPACE_PATH_BUDGET = 200


@pytest.fixture
def ads_tmp_path(tmp_path):
    """``tmp_path`` guarded for ADS / hpeesofsim long-path safety.

    Skips with an actionable message when the base path is already long enough
    that hpeesofsim would crash on the nested workspace it creates, so a
    long-path environment surfaces as a clear skip instead of an opaque
    simulator crash. Short-path failures still run and fail normally.
    """
    length = len(str(tmp_path))
    if length > _ADS_WORKSPACE_PATH_BUDGET:
        pytest.skip(
            f"ADS workspace base path is {length} chars (> {_ADS_WORKSPACE_PATH_BUDGET}); "
            "hpeesofsim crashes on long Windows paths. Re-run with a short base temp dir "
            "(pytest --basetemp=C:\\w) or clone the repository to a short path -- see "
            "docs/REPRODUCE.md."
        )
    return tmp_path
