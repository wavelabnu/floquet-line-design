"""Drift guards for the committed manuscript figure records.

Pins the headline numbers each generator in ``examples/figures/`` reads out of
the committed records, so an imported-record change or a definition drift fails
here instead of silently re-rendering a panel:

  * **Fig 2c** (``fig2c_phase_sweep``) -- recorded peak +18.9648 dB at 210 deg,
    and the replay guard ``max|derived - recorded boost| <= 1e-12 dB`` where the
    derived trace is rebuilt from ``G_mod_in_dB`` and the passive pump-off
    reference in the same record.
  * **Fig 2d** (``fig2d_length_law``) -- the N = 15 robust optimum at 8.18 dB.
  * **SM junction profiles** (``sm_junction_profiles``) -- all four per-cell
    ``m_j`` curves are FLAT by the |slope| > 0.3 dB/cell test (no along-line
    pump attenuation), with the slopes and peak-to-trough spreads pinned.

Pure Python: no ADS, no network, no figure rendering (the generators' pure
read/compute helpers are called directly, so nothing is written).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "examples" / "figures"
if str(FIGURES_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURES_DIR))

import fig2c_phase_sweep as fig2c  # noqa: E402
import fig2d_length_law as fig2d  # noqa: E402
import sm_junction_profiles as smjp  # noqa: E402

# --------------------------------------------------------------------------- #
# Fig 2c -- certified MAVR phase sweep
# --------------------------------------------------------------------------- #
FIG2C_PEAK_DB = 18.9648
FIG2C_PEAK_PHASE_DEG = 210.0
FIG2C_TOL_DB = 1.0e-3
FIG2C_REPLAY_GUARD_DB = 1.0e-12


@pytest.fixture(scope="module")
def fig2c_derived():
    return fig2c.derive(fig2c.load_record())


def test_fig2c_recorded_peak(fig2c_derived) -> None:
    peak = fig2c_derived["peak"]
    assert peak["phase_deg"] == pytest.approx(FIG2C_PEAK_PHASE_DEG, abs=1e-9)
    assert peak["boost_dB"] == pytest.approx(FIG2C_PEAK_DB, abs=FIG2C_TOL_DB), (
        f"Fig 2c peak drifted: {peak['boost_dB']:.4f} dB vs pinned {FIG2C_PEAK_DB}"
    )


def test_fig2c_replay_guard(fig2c_derived) -> None:
    """boost = G_mod_in_dB - passive_S21_dB must reproduce the recorded trace."""
    max_dev = fig2c_derived["max_dev_dB"]
    assert max_dev <= FIG2C_REPLAY_GUARD_DB, (
        f"replay guard failed: max|derived - recorded| = {max_dev:.3e} dB "
        f"> {FIG2C_REPLAY_GUARD_DB:g}"
    )


# --------------------------------------------------------------------------- #
# Fig 2d -- length law
# --------------------------------------------------------------------------- #
FIG2D_N15_BOOST_DB = 8.18
FIG2D_TOL_DB = 0.01


def test_fig2d_n15_robust_optimum() -> None:
    groups = fig2d.classify(fig2d.load_rows())
    boost = groups["by_n"][15]
    assert boost == pytest.approx(FIG2D_N15_BOOST_DB, abs=FIG2D_TOL_DB), (
        f"Fig 2d N=15 boost drifted: {boost:.4f} dB vs pinned {FIG2D_N15_BOOST_DB}"
    )
    assert dict(groups["robust"])[15] == pytest.approx(
        FIG2D_N15_BOOST_DB, abs=FIG2D_TOL_DB), "N=15 is no longer classified robust"


# --------------------------------------------------------------------------- #
# SM junction profiles -- along-line discriminator
# --------------------------------------------------------------------------- #
SLOPE_ABS_MAX_DB_PER_CELL = 0.11
SPREAD_TOL_DB = 0.02
PINNED_SPREAD_DB = {
    "6.2 GHz, 29 dBm": 1.28,
    "6.2 GHz, 20 dBm (scaled)": 1.28,
    "10.6 GHz, 29 dBm": 3.68,
    "10.6 GHz, 20 dBm": 3.68,
}


@pytest.fixture(scope="module")
def junction_fits():
    curves = smjp.load_curves()
    return {c["label"]: smjp.fit_curve(c["cells"], c["mj"]) for c in curves}


def test_junction_profiles_cover_all_four_curves(junction_fits) -> None:
    assert set(junction_fits) == set(PINNED_SPREAD_DB), (
        f"junction-profile curve set changed: {sorted(junction_fits)}"
    )


@pytest.mark.parametrize("label", sorted(PINNED_SPREAD_DB))
def test_junction_profile_is_flat(label: str, junction_fits) -> None:
    """No along-line pump attenuation: every curve stays under the decay threshold."""
    fit = junction_fits[label]
    assert fit["verdict"] == "flat", (
        f"{label} verdict changed to {fit['verdict']} "
        f"(slope {fit['slope_dB_per_cell']:+.4f} dB/cell)"
    )
    assert abs(fit["slope_dB_per_cell"]) <= SLOPE_ABS_MAX_DB_PER_CELL, (
        f"{label} slope drifted: {fit['slope_dB_per_cell']:+.4f} dB/cell "
        f"(pinned |slope| <= {SLOPE_ABS_MAX_DB_PER_CELL})"
    )


@pytest.mark.parametrize("label", sorted(PINNED_SPREAD_DB))
def test_junction_profile_spread(label: str, junction_fits) -> None:
    want = PINNED_SPREAD_DB[label]
    got = junction_fits[label]["spread_dB"]
    assert got == pytest.approx(want, abs=SPREAD_TOL_DB), (
        f"{label} peak-to-trough spread drifted: {got:.3f} dB vs pinned {want}"
    )
