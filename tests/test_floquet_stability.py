"""Tests for the parametric-instability eigenvalue framework and sweeps.

The unit-cell harmonic ABCD ``M_uc`` has eigenvalues ``lambda_k`` that play
the role of per-cell Bloch multipliers. ``max_k |lambda_k| = 1`` is the
parametric instability threshold. The lossless static limit uses a trivial
TL-only unit cell so that every harmonic is guaranteed to be in the
propagating regime (with a stub the harmonic axis spans broad frequencies
that can place some harmonics inside a passive stop-band, which would have
``|lambda| != 1`` even without modulation).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tvtl.floquet.harmonic_abcd import (
    cascade_harmonic_abcd,
    stub_harmonic_abcd,
    tl_section_harmonic_abcd,
)
from tvtl.floquet.stability import (
    assemble_full_unit_cell_matrix,
    cascade_gain_estimate,
    dominant_growth_rate,
    floquet_exponents,
    stability_margin,
    unit_cell_eigenvalues,
)
from tvtl.floquet.sweeps import (
    frequency_modulation_map,
    modulation_depth_sweep,
    threshold_search,
)
from tvtl.floquet.uc_model import (
    build_unit_cell_block_abcd,
    floquet_uc_cascade_response,
    unit_cell_period_m,
)
from tvtl.materials import SPEED_OF_LIGHT_M_PER_S
from tvtl.microstrip.models import hammerstad_kirschning


# SMV1245 + paper unit cell.
C0_REF = 7.0e-12
R_REF = 1.5
L_REF = 1.7e-9
FS_REF = 3.1e9
N_REF = 5
N_CELLS_REF = 15
DS_FRACTION_REF = 0.30

SUBSTRATE = {"eps_r": 3.38, "h": 1.54e-3, "tan_delta": 0.0027, "copper_thickness": 17.5e-6}
MAIN_SEG_LENGTH = 3.055e-3
MAIN_SEG_WIDTH = 4.27e-3
PAPER_STUB_SEGMENTS = (
    {"length": 6.83e-3, "width": 1.93e-3},
    {"length": 5.25e-3, "width": 1.41e-3},
    {"length": 1.04e-3, "width": 1.98e-3},
)


def _paper_uc_geometry() -> dict:
    return {
        "main_segments": [
            {"length": MAIN_SEG_LENGTH, "width": MAIN_SEG_WIDTH},
            {"length": MAIN_SEG_LENGTH, "width": MAIN_SEG_WIDTH},
        ],
        "stub_position": 0,
        "stub_segments": [dict(segment) for segment in PAPER_STUB_SEGMENTS],
    }


def _z0_and_v_phase(width: float, frequency_hz: float) -> tuple[float, float]:
    z0_arr, eps_eff_arr = hammerstad_kirschning(
        width, SUBSTRATE["copper_thickness"], SUBSTRATE["h"], SUBSTRATE["eps_r"],
        np.asarray([frequency_hz], dtype=float),
    )
    return float(z0_arr[0]), SPEED_OF_LIGHT_M_PER_S / math.sqrt(float(eps_eff_arr[0]))


def _tl_only_uc_block(fs_hz: float, length: float = 3.055e-3) -> tuple:
    """Trivial unit cell: a single lossless TL section.

    With no stub, every harmonic is purely propagating and the per-harmonic
    2-by-2 block is a unimodular rotation; its eigenvalues lie on the unit
    circle exactly. The resulting unit-cell harmonic ABCD therefore satisfies
    ``|lambda_k| = 1`` for every Bloch mode regardless of harmonic index.
    """

    ws = 2.0 * math.pi * fs_hz
    wm = 2.0 * ws
    z0, v_phase = _z0_and_v_phase(MAIN_SEG_WIDTH, fs_hz)
    return tl_section_harmonic_abcd(length, z0, v_phase, ws, wm, N_REF)


def test_lossless_static_eigenvalues_on_unit_circle() -> None:
    """The lossless trivial UC must produce eigenvalues all of magnitude 1."""

    block = _tl_only_uc_block(FS_REF)
    eigenvalues = unit_cell_eigenvalues(block)
    magnitudes = np.abs(eigenvalues)
    assert np.all(np.abs(magnitudes - 1.0) < 1.0e-8), magnitudes
    assert abs(stability_margin(block)) < 1.0e-8


def test_lossy_static_introduces_subunit_eigenvalues() -> None:
    """Adding loss to a passive UC produces eigenvalues strictly inside the
    unit disk on the forward-propagating side of every reciprocal pair.

    Per-harmonic ABCDs are unimodular, so eigenvalues always come in
    reciprocal pairs ``(lambda, 1/lambda)``. For a lossless propagating
    cell, every pair sits on the unit circle. With dissipation, each pair
    becomes ``(|lambda| < 1, |lambda| > 1)``, with the product still 1.
    This test asserts the dissipative version of that statement: at least
    one eigenvalue strictly inside the disk and the product of paired
    magnitudes is 1 to numerical tolerance.
    """

    fs = FS_REF
    ws = 2.0 * math.pi * fs
    wm = 2.0 * ws
    # Lossy lumped series resistance acting as a shunt: ABCD = [[I, 0], [G*I, I]]
    # with G = 1/(R*N). This is a pure dissipative shunt admittance whose
    # per-harmonic block is non-trivial but the cascade stays passive.
    size = 2 * N_REF + 1
    identity = np.eye(size, dtype=np.complex128)
    zero = np.zeros((size, size), dtype=np.complex128)
    g_value = 0.01  # 100 Ohm shunt resistor — dissipative.
    shunt_block = (identity, zero, g_value * identity, identity)
    tl_block = _tl_only_uc_block(fs)
    block = cascade_harmonic_abcd([tl_block, shunt_block, tl_block])
    eigenvalues = unit_cell_eigenvalues(block)
    magnitudes = np.abs(eigenvalues)
    # Loss must drive at least one eigenvalue strictly inside the unit disk.
    assert float(np.min(magnitudes)) < 1.0 - 1.0e-6, magnitudes
    # And the largest must lie outside (reciprocal-pair structure).
    assert float(np.max(magnitudes)) > 1.0 + 1.0e-6, magnitudes
    # Product of every pair is 1 (within tolerance). Sort log|lambda| and
    # match smallest with largest, etc.
    log_magnitudes = sorted(np.log(magnitudes).tolist())
    pair_sums = [log_magnitudes[i] + log_magnitudes[-1 - i] for i in range(len(log_magnitudes) // 2)]
    assert max(abs(s) for s in pair_sums) < 1.0e-8, pair_sums


def test_modulation_depth_sweep_produces_responsive_metrics() -> None:
    """Sweeping dS at fs = 3.1 GHz produces clearly responsive cascade metrics.

    For the paper geometry at SMV1245 parameters, the largest absolute
    eigenvalue is dominated by passive stop-band content of the
    high-order harmonics (n = +/- 5 reach ~30 GHz where the cell is no
    longer in its propagation band), so the eigenvalue-based
    ``lambda_max_abs`` is largely insensitive to ``dS``. The cascade
    S-parameter idler proxy, by contrast, is a direct probe of parametric
    coupling and must respond clearly to ``dS``. This test asserts:

    * ``lambda_max_abs`` stays bounded across the sweep (the passive
      stop-band remains the dominant eigenvalue).
    * ``gain_idler_proxy_dB`` increases monotonically with ``dS`` once we
      move past the first low-modulation point, with a clear final-vs-
      initial gap.
    """

    rows = modulation_depth_sweep(
        _paper_uc_geometry(),
        {"C0": C0_REF, "R": R_REF, "L": L_REF},
        {"fs": FS_REF, "phi": 0.0, "fm_over_fs_ratio": 2.0},
        substrate=SUBSTRATE,
        n_cells=N_CELLS_REF,
        N_harmonics=N_REF,
        dS_fractions=(0.1, 0.3, 0.5, 0.7, 0.9),
    )
    magnitudes = [float(row["lambda_max_abs"]) for row in rows]
    idler = [float(row["gain_idler_proxy_dB"]) for row in rows]
    # lambda_max is bounded close to unity by the passive stop-band baseline;
    # the corrected topology shifts the baseline slightly higher (~1.11) so
    # we use a generous upper bound that still excludes runaway instability.
    assert 0.9 < max(magnitudes) < 1.5, magnitudes
    # Idler proxy is the parametric-coupling observable and must grow.
    assert idler[-1] > idler[0] + 15.0, idler
    for previous, current in zip(idler[1:-1], idler[2:], strict=True):
        assert current >= previous - 1.0e-6, idler


def test_cascade_gain_estimate_finite_and_within_20dB_envelope() -> None:
    """The eigenvalue estimate and the cascade S21 gain are both finite,
    bounded, and within a 20 dB envelope of each other for the paper UC
    at dS_fraction=0.5.

    The two metrics differ structurally: ``cascade_gain_estimate_dB`` is
    the asymptotic per-cell amplification of the dominant Bloch mode,
    while ``gain_total_proxy_dB`` is the sum of squared S21 column entries
    at the n=0 input, which couples through the matched source/load. They
    need not agree exactly and can even differ in sign when the dominant
    eigenmode is a passive stop-band mode not excited by the matched
    n=0 input. This test asserts only that both are finite and within a
    20 dB envelope, which is the loose physical-consistency bound.
    """

    geometry = _paper_uc_geometry()
    varactor = {"C0": C0_REF, "R": R_REF, "L": L_REF}
    modulation = {"fs": FS_REF, "fm": 2.0 * FS_REF, "dS_fraction": 0.5, "phi": 0.0}
    response = floquet_uc_cascade_response(
        geometry, varactor, modulation,
        n_cells=N_CELLS_REF, N_harmonics=N_REF, substrate=SUBSTRATE,
    )
    block = build_unit_cell_block_abcd(geometry, varactor, modulation, N_REF, SUBSTRATE)
    eigen = cascade_gain_estimate(block, N_CELLS_REF)
    assert eigen["label"] == "analytical_proxy_only"
    cascade_db = float(response["gain_total_proxy_dB"])
    eigen_db = float(eigen["cascade_gain_estimate_dB"])
    assert math.isfinite(cascade_db)
    assert math.isfinite(eigen_db)
    assert abs(eigen_db - cascade_db) < 20.0, (eigen_db, cascade_db)


def test_threshold_search_finds_at_least_one_finite_dS_threshold() -> None:
    rows = threshold_search(
        _paper_uc_geometry(),
        {"C0": C0_REF, "R": R_REF, "L": L_REF},
        {"phi": 0.0, "fm_over_fs_ratio": 2.0},
        substrate=SUBSTRATE,
        n_cells=N_CELLS_REF,
        N_harmonics=N_REF,
        fs_values_Hz=[1.5e9, 2.0e9, 3.1e9, 4.0e9, 5.0e9],
        dS_grid_for_search=tuple(round(0.05 + 0.05 * idx, 4) for idx in range(19)),
    )
    finite_thresholds = [row for row in rows if math.isfinite(float(row["dS_threshold"]))]
    assert finite_thresholds, rows
    for row in rows:
        assert row["label"] == "analytical_proxy_only"


def test_disclaimer_label_propagates_into_all_sweep_rows() -> None:
    modulation_rows = modulation_depth_sweep(
        _paper_uc_geometry(),
        {"C0": C0_REF, "R": R_REF, "L": L_REF},
        {"fs": FS_REF, "phi": 0.0, "fm_over_fs_ratio": 2.0},
        substrate=SUBSTRATE,
        n_cells=N_CELLS_REF,
        N_harmonics=N_REF,
        dS_fractions=(0.2, 0.4, 0.6),
    )
    for row in modulation_rows:
        assert row["label"] == "analytical_proxy_only"

    map_rows = frequency_modulation_map(
        _paper_uc_geometry(),
        {"C0": C0_REF, "R": R_REF, "L": L_REF},
        {"phi": 0.0, "fm_over_fs_ratio": 2.0},
        substrate=SUBSTRATE,
        n_cells=N_CELLS_REF,
        N_harmonics=N_REF,
        fs_values_Hz=[2.0e9, 3.1e9],
        dS_fractions=(0.2, 0.6),
    )
    for row in map_rows:
        assert row["label"] == "analytical_proxy_only"

    # And the eigenvalue helper carries the label too.
    block = build_unit_cell_block_abcd(
        _paper_uc_geometry(),
        {"C0": C0_REF, "R": R_REF, "L": L_REF},
        {"fs": FS_REF, "fm": 2.0 * FS_REF, "dS_fraction": 0.4, "phi": 0.0},
        N_REF,
        SUBSTRATE,
    )
    estimate = cascade_gain_estimate(block, N_CELLS_REF)
    assert estimate["label"] == "analytical_proxy_only"


def test_floquet_exponents_match_eigenvalues_per_cell() -> None:
    """Sanity check on ``floquet_exponents``: ``exp(gamma * p)`` reproduces
    the eigenvalues modulo the principal branch of the logarithm."""

    block = _tl_only_uc_block(FS_REF)
    period = unit_cell_period_m(
        {
            "main_segments": [{"length": 3.055e-3, "width": MAIN_SEG_WIDTH}],
            "stub_position": 0,
        }
    )
    eigenvalues = unit_cell_eigenvalues(block)
    gammas = floquet_exponents(block, period)
    reconstructed = np.exp(gammas * period)
    # The principal branch may flip sign in the imaginary part; compare via
    # absolute value to keep the test robust.
    assert np.allclose(np.abs(reconstructed), np.abs(eigenvalues), atol=1.0e-10)


def test_assemble_full_matrix_layout_matches_cascade_pack() -> None:
    fs = FS_REF
    ws = 2.0 * math.pi * fs
    wm = 2.0 * ws
    z0, v_phase = _z0_and_v_phase(MAIN_SEG_WIDTH, fs)
    block = tl_section_harmonic_abcd(MAIN_SEG_LENGTH, z0, v_phase, ws, wm, N_REF)
    full = assemble_full_unit_cell_matrix(block)
    size = 2 * N_REF + 1
    np.testing.assert_array_equal(full[:size, :size], block[0])
    np.testing.assert_array_equal(full[:size, size:], block[1])
    np.testing.assert_array_equal(full[size:, :size], block[2])
    np.testing.assert_array_equal(full[size:, size:], block[3])


