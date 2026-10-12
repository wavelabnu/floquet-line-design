"""Drift guard for the Fig 1d predicted pump-off Bloch dispersion.

Recomputes the dispersion of the fabricated SMV1245 cell through the committed
generator ``examples/figures/fig1d_dispersion.py`` (framework analytical leg at
its CORRECTED_NOTEBOOK tee default, drive ``m_branch_pump=0.0``) and pins:

  * the six band edges -- ``Re[(A+D)/2]`` crossing +1 (zone centre, beta = 0) or
    -1 (zone boundary, beta = pi) -- to within +/-5 MHz;
  * the four ``betap_over_pi`` marker values that feed the Fig 5a pair markers,
    to within +/-1e-3.

A change in the tee model, the stub table, the varactor palette entry, or the
Bloch extraction moves these numbers and fails here rather than silently
re-rendering the panel.

Pure Python: no ADS, no network. The band-edge scan uses a 10 MHz grid (the
generator ships a 4 MHz grid); both resolve every edge to well under 0.1 MHz, so
the coarser test grid keeps the suite fast with ~60x margin on the tolerance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "examples" / "figures"
if str(FIGURES_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURES_DIR))

import fig1d_dispersion as fig1d  # noqa: E402

# Pinned band edges (GHz) and their zone classification.
PINNED_BAND_EDGES = (
    (0.9953, "zone_center_beta0"),
    (3.0372, "zone_center_beta0"),
    (9.4350, "zone_boundary_betaPi"),
    (10.4754, "zone_boundary_betaPi"),
    (10.5016, "zone_center_beta0"),
    (10.7460, "zone_center_beta0"),
)
BAND_EDGE_TOL_GHZ = 0.005          # +/-5 MHz

# Pinned betap_over_pi at the Fig 5a marker frequencies.
PINNED_MARKERS = {3.1: 0.0483, 6.2: 0.4114, 5.3: 0.3259, 10.6: 0.2399}
MARKER_TOL = 1.0e-3

SCAN_STEP_HZ = 10.0e6


@pytest.fixture(scope="module")
def band_edges():
    return fig1d.scan_band_edges(step_hz=SCAN_STEP_HZ)


def test_band_edge_count(band_edges) -> None:
    assert len(band_edges) == len(PINNED_BAND_EDGES), (
        f"expected {len(PINNED_BAND_EDGES)} band edges, got "
        f"{[round(e['f_GHz'], 4) for e in band_edges]}"
    )


def test_band_edge_frequencies(band_edges) -> None:
    for got, (want_ghz, want_kind) in zip(band_edges, PINNED_BAND_EDGES):
        assert got["kind"] == want_kind, (
            f"band edge near {want_ghz} GHz changed kind: "
            f"{got['kind']} != {want_kind}"
        )
        assert got["f_GHz"] == pytest.approx(want_ghz, abs=BAND_EDGE_TOL_GHZ), (
            f"band edge drifted: {got['f_GHz']:.4f} GHz vs pinned {want_ghz} GHz "
            f"(tolerance +/-{BAND_EDGE_TOL_GHZ * 1e3:.0f} MHz)"
        )


@pytest.mark.parametrize("f_ghz,want", sorted(PINNED_MARKERS.items()))
def test_marker_betap_over_pi(f_ghz: float, want: float) -> None:
    betap_over_pi, _alphap, _half = fig1d.dispersion_at(f_ghz * 1e9)
    assert betap_over_pi == pytest.approx(want, abs=MARKER_TOL), (
        f"betap/pi at {f_ghz} GHz drifted: {betap_over_pi:.6f} vs pinned {want}"
    )


def test_pump_off_block_is_truncation_independent() -> None:
    """Drive 0 decouples the harmonics, so the central block must not depend on N."""
    import numpy as np

    diff = float(np.max(np.abs(fig1d.central_block(3.1e9, 1)
                               - fig1d.central_block(3.1e9, 6))))
    assert diff < 1.0e-12, f"pump-off central block depends on truncation: {diff:.2e}"
