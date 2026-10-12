"""MTEE_ADS 3-port S-matrix surrogate, calibrated from a
densified probe grid (5x5 geometries x 39 frequencies).

The surrogate provides a function ``mtee_ads_3port_smatrix(w_main_m,
w_branch_m, f_hz)`` returning the 3x3 complex S-matrix of an isolated
MTEE_ADS element at Rogers 4003C 1.524 mm substrate, evaluated via
bilinear interpolation in (w_main, w_branch) and linear interpolation
in f.

Validity bounds (clipped at the boundaries with a recoverable warning):
- w_main_m in [2.5e-3, 4.5e-3]
- w_branch_m in [0.5e-3, 2.5e-3]
- f_hz in [1.0e9, 2.0e10]

Symmetry expansion: the underlying surrogate stores 4 independent
elements (S21, S22, S31, S33). Reciprocity (S12=S21, S13=S31, S23=S32)
and left-right symmetry (S11=S22, S32=S31) expand to the full 3x3.

Used by ``TeeModelVariant.MTEE_ADS_SURROGATE_V1`` in the analytical leg.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

_SURROGATE_ENV_VAR = "TVTL_MTEE_SURROGATE_JSON"
_DEFAULT_SURROGATE_JSON = Path(__file__).with_name("data") / "surrogate_tabular.json"


def _surrogate_json_path() -> Path:
    """Return the optional surrogate table path for public-copy use."""

    env_path = os.environ.get(_SURROGATE_ENV_VAR)
    return Path(env_path) if env_path else _DEFAULT_SURROGATE_JSON


@lru_cache(maxsize=1)
def _load_surrogate() -> dict:
    """Lazy-load the surrogate table on first use."""
    path = _surrogate_json_path()
    if not path.is_file():
        raise FileNotFoundError(
            "MTEE ADS surrogate table is not bundled. Set "
            f"{_SURROGATE_ENV_VAR} to a surrogate_tabular.json path "
            "before using TeeModelVariant.MTEE_ADS_SURROGATE_V1."
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    md = data["metadata"]
    table = data["table"]
    # Convert nested lists to numpy arrays for fast slicing
    out = {
        "w_mains_mm": np.asarray(md["w_mains_mm"]),
        "w_branches_mm": np.asarray(md["w_branches_mm"]),
        "f_hz_grid": np.asarray(md["f_hz_grid"]),
        "bounds_mm": md["validity_bounds_mm"],
        "bounds_Hz": md["validity_bounds_Hz"],
        "tables": {},
    }
    for elem in md["indep_elements"]:
        re = np.asarray(table[elem]["re"], dtype=np.float64)
        im = np.asarray(table[elem]["im"], dtype=np.float64)
        # Shape: (n_wm, n_wb, n_f)
        out["tables"][elem] = re + 1j * im
    return out


def _interp_one_element(
    elem_table: NDArray,
    w_mains_mm: NDArray, w_branches_mm: NDArray, f_grid: NDArray,
    wm_mm: float, wb_mm: float, f_hz: float,
) -> complex:
    """Bilinear in (w_main, w_branch), linear in f. Edge-clipped."""
    # Clip to grid bounds
    wm_c = float(np.clip(wm_mm, w_mains_mm[0], w_mains_mm[-1]))
    wb_c = float(np.clip(wb_mm, w_branches_mm[0], w_branches_mm[-1]))
    f_c = float(np.clip(f_hz, f_grid[0], f_grid[-1]))
    # Locate indices
    i = int(np.searchsorted(w_mains_mm, wm_c, side="right") - 1)
    if i >= len(w_mains_mm) - 1:
        i = len(w_mains_mm) - 2
    j = int(np.searchsorted(w_branches_mm, wb_c, side="right") - 1)
    if j >= len(w_branches_mm) - 1:
        j = len(w_branches_mm) - 2
    k = int(np.searchsorted(f_grid, f_c, side="right") - 1)
    if k >= len(f_grid) - 1:
        k = len(f_grid) - 2
    # Bilinear weights in widths
    tm = (wm_c - w_mains_mm[i]) / (w_mains_mm[i + 1] - w_mains_mm[i])
    tb = (wb_c - w_branches_mm[j]) / (w_branches_mm[j + 1] - w_branches_mm[j])
    # Linear weight in f
    tf = (f_c - f_grid[k]) / (f_grid[k + 1] - f_grid[k])
    # 8-corner trilinear interpolation
    c000 = complex(elem_table[i, j, k])
    c100 = complex(elem_table[i + 1, j, k])
    c010 = complex(elem_table[i, j + 1, k])
    c110 = complex(elem_table[i + 1, j + 1, k])
    c001 = complex(elem_table[i, j, k + 1])
    c101 = complex(elem_table[i + 1, j, k + 1])
    c011 = complex(elem_table[i, j + 1, k + 1])
    c111 = complex(elem_table[i + 1, j + 1, k + 1])
    c00 = c000 * (1 - tf) + c001 * tf
    c10 = c100 * (1 - tf) + c101 * tf
    c01 = c010 * (1 - tf) + c011 * tf
    c11 = c110 * (1 - tf) + c111 * tf
    c0 = c00 * (1 - tb) + c01 * tb
    c1 = c10 * (1 - tb) + c11 * tb
    return c0 * (1 - tm) + c1 * tm


def mtee_ads_3port_smatrix(
    w_main_m: float, w_branch_m: float, f_hz: float
) -> NDArray[np.complex128]:
    """Return the 3x3 complex S-matrix of MTEE_ADS at the given geometry/frequency.

    Symmetry expansion:
    - S(1,1) = S(2,2)             (left-right symmetry)
    - S(1,2) = S(2,1)             (reciprocity)
    - S(1,3) = S(3,1) = S(2,3) = S(3,2)  (reciprocity + symmetry)
    - S(3,3) is independent

    Note: input widths in METERS; the surrogate is calibrated in mm so
    these are converted internally.
    """
    s = _load_surrogate()
    wm_mm = float(w_main_m) * 1.0e3
    wb_mm = float(w_branch_m) * 1.0e3
    s21 = _interp_one_element(
        s["tables"]["S21"], s["w_mains_mm"], s["w_branches_mm"], s["f_hz_grid"],
        wm_mm, wb_mm, f_hz,
    )
    s22 = _interp_one_element(
        s["tables"]["S22"], s["w_mains_mm"], s["w_branches_mm"], s["f_hz_grid"],
        wm_mm, wb_mm, f_hz,
    )
    s31 = _interp_one_element(
        s["tables"]["S31"], s["w_mains_mm"], s["w_branches_mm"], s["f_hz_grid"],
        wm_mm, wb_mm, f_hz,
    )
    s33 = _interp_one_element(
        s["tables"]["S33"], s["w_mains_mm"], s["w_branches_mm"], s["f_hz_grid"],
        wm_mm, wb_mm, f_hz,
    )
    S = np.asarray([
        [s22, s21, s31],   # S(1,1)=s22, S(1,2)=s21, S(1,3)=s31
        [s21, s22, s31],   # S(2,1)=s21, S(2,2)=s22, S(2,3)=s31
        [s31, s31, s33],   # S(3,1)=s31, S(3,2)=s31, S(3,3)=s33
    ], dtype=np.complex128)
    return S


def mtee_ads_3port_smatrix_harmonic(
    w_main_m: float, w_branch_m: float, omega_n: NDArray[np.float64]
) -> NDArray[np.complex128]:
    """Per-harmonic 3-port S-matrix. Returns shape (n_harmonics, 3, 3).

    Negative-frequency harmonics use the complex conjugate of the
    positive-frequency S (standard for real reciprocal networks).
    """
    omega_n = np.asarray(omega_n)
    out = np.zeros((omega_n.shape[0], 3, 3), dtype=np.complex128)
    for k in range(omega_n.shape[0]):
        omega = float(omega_n[k])
        f_abs = abs(omega) / (2.0 * np.pi)
        if f_abs <= 0:
            out[k] = np.eye(3, dtype=np.complex128) * 0.0
            continue
        S = mtee_ads_3port_smatrix(w_main_m, w_branch_m, f_abs)
        if omega < 0:
            S = S.conjugate()
        out[k] = S
    return out


def reduce_3port_to_2port_with_load(
    S3port: NDArray[np.complex128],
    z_branch: complex,
    z_ref: float = 50.0,
) -> NDArray[np.complex128]:
    """Terminate port 3 of a 3-port S-matrix with load impedance z_branch,
    return the resulting 2x2 S-matrix between ports 1 and 2.

    Standard formula:
        S2(i,j) = S3(i,j) + S3(i,3) * Gamma3 * S3(3,j) / (1 - S3(3,3) * Gamma3)
    where Gamma3 = (z_branch - z_ref) / (z_branch + z_ref).
    """
    Gamma3 = (z_branch - z_ref) / (z_branch + z_ref)
    denom = 1.0 - S3port[2, 2] * Gamma3
    # Numerical stability guard: cap |1/denom| at a large constant
    if abs(denom) < 1.0e-10:
        denom = 1.0e-10 * (denom / max(abs(denom), 1.0e-30) if abs(denom) > 0 else 1.0)
    S2 = np.zeros((2, 2), dtype=np.complex128)
    for i in (0, 1):
        for j in (0, 1):
            S2[i, j] = S3port[i, j] + S3port[i, 2] * Gamma3 * S3port[2, j] / denom
    return S2


def s2port_to_abcd(S: NDArray[np.complex128], z_ref: float = 50.0) -> NDArray[np.complex128]:
    """Convert a 2x2 S-matrix to a 2x2 ABCD matrix."""
    S11, S12 = S[0, 0], S[0, 1]
    S21, S22 = S[1, 0], S[1, 1]
    if abs(S21) < 1.0e-15:
        # Degenerate: return identity (best fallback)
        return np.eye(2, dtype=np.complex128)
    A = ((1 + S11) * (1 - S22) + S12 * S21) / (2 * S21)
    B = z_ref * ((1 + S11) * (1 + S22) - S12 * S21) / (2 * S21)
    C = (1 / z_ref) * ((1 - S11) * (1 - S22) - S12 * S21) / (2 * S21)
    D = ((1 - S11) * (1 + S22) + S12 * S21) / (2 * S21)
    return np.asarray([[A, B], [C, D]], dtype=np.complex128)


def tee_3port_loaded_2port_abcd(
    w_main_m: float, w_branch_m: float, f_hz: float, z_branch: complex,
    z_ref: float = 50.0,
) -> NDArray[np.complex128]:
    """Full pipeline: surrogate 3-port S -> load port 3 -> reduce to 2x2 S -> ABCD."""
    S3 = mtee_ads_3port_smatrix(w_main_m, w_branch_m, f_hz)
    S2 = reduce_3port_to_2port_with_load(S3, z_branch, z_ref)
    return s2port_to_abcd(S2, z_ref)


def build_combined_tee_branch_harmonic_abcd(
    s3_per_harm: NDArray[np.complex128],
    y_branch_full: NDArray[np.complex128],
    z_ref: float = 50.0,
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128],
           NDArray[np.complex128], NDArray[np.complex128]]:
    """Combined 2-port harmonic ABCD for the tee + branch network.

    Inputs:
        s3_per_harm: (n_harm, 3, 3) complex per-harmonic 3-port S of the
            tee (diagonal in harmonic basis - tee is linear).
        y_branch_full: (n_harm, n_harm) complex full harmonic-coupled
            branch admittance (couples harmonics via the varactor).
        z_ref: reference impedance (50 ohm).

    Output: (A, B, C, D) each (n_harm, n_harm) complex matrices, capturing
    the combined 2-port behavior of the tee with port 3 loaded by the
    branch's full admittance matrix. This replaces the existing
    [block_left, branch, block_right] cascade trio with a single combined
    block, preserving the branch's cross-harmonic coupling exactly.

    Reduction math: in the harmonic basis,
        Gamma_branch = (Y0 I - Y_branch)(Y0 I + Y_branch)^-1
        S2(i,j) = diag(S3(i,j)) + diag(S3(i,3)) Gamma_branch
                  (I - diag(S3(3,3)) Gamma_branch)^-1 diag(S3(3,j))
    where each S3 element is scalar per harmonic (diag matrix in
    harmonic basis), and the 2-port S blocks are full n_harm x n_harm
    matrices. Then 2-port S -> Y -> ABCD via standard matrix-form
    conversion.
    """
    n_harm = s3_per_harm.shape[0]
    Y0 = 1.0 / z_ref
    I_nh = np.eye(n_harm, dtype=np.complex128)

    # Gamma_branch from full Y_branch
    Y_plus = Y0 * I_nh + y_branch_full
    Y_minus = Y0 * I_nh - y_branch_full
    Gamma_branch = Y_minus @ np.linalg.inv(Y_plus)

    # Per-harmonic S3 entries as diag matrices
    def s_diag(i: int, j: int) -> NDArray:
        return np.diag(np.asarray([s3_per_harm[k, i, j] for k in range(n_harm)]))

    s11_d = s_diag(0, 0)
    s12_d = s_diag(0, 1)
    s13_d = s_diag(0, 2)
    s21_d = s_diag(1, 0)
    s22_d = s_diag(1, 1)
    s23_d = s_diag(1, 2)
    s31_d = s_diag(2, 0)
    s32_d = s_diag(2, 1)
    s33_d = s_diag(2, 2)

    # Reduction: 2-port S blocks
    denom_inv = np.linalg.inv(I_nh - s33_d @ Gamma_branch)
    common = Gamma_branch @ denom_inv
    S2_11 = s11_d + s13_d @ common @ s31_d
    S2_12 = s12_d + s13_d @ common @ s32_d
    S2_21 = s21_d + s23_d @ common @ s31_d
    S2_22 = s22_d + s23_d @ common @ s32_d

    # Convert block 2-port S to block 2-port Y via matrix formula.
    # Y = Y0 (I - S)(I + S)^-1 over the full 2*n_harm x 2*n_harm block.
    # Build the full block S matrix:
    full_size = 2 * n_harm
    S_full = np.zeros((full_size, full_size), dtype=np.complex128)
    S_full[:n_harm, :n_harm] = S2_11
    S_full[:n_harm, n_harm:] = S2_12
    S_full[n_harm:, :n_harm] = S2_21
    S_full[n_harm:, n_harm:] = S2_22
    I_full = np.eye(full_size, dtype=np.complex128)
    Y_full = Y0 * (I_full - S_full) @ np.linalg.inv(I_full + S_full)
    Y11 = Y_full[:n_harm, :n_harm]
    Y12 = Y_full[:n_harm, n_harm:]
    Y21 = Y_full[n_harm:, :n_harm]
    Y22 = Y_full[n_harm:, n_harm:]

    # 2-port Y -> ABCD per standard (matrix-form Pozar Table 4.2):
    #   A = -inv(Y21) @ Y22
    #   B = -inv(Y21)
    #   C = Y12 - Y11 @ inv(Y21) @ Y22
    #   D = -Y11 @ inv(Y21)
    # Framework convention: I1 into port 1, I2 OUT of port 2 (V2/I2 = load
    # impedance). Verified against shunt_admittance_abcd / series_impedance_abcd
    # in tvtl.abcd.matrices.  Diagonal-matrix verification cannot catch the
    # multiplication-order bug because diagonal matrices commute; the bug
    # only manifests when Y has off-diagonal harmonic coupling (Step
    # 2.5c-ii-c diagnostic: cascade exploded under SURROGATE while
    # single-cell passed).
    Y21_inv = np.linalg.inv(Y21)
    A = -Y21_inv @ Y22
    B = -Y21_inv
    C = Y12 - Y11 @ Y21_inv @ Y22
    D = -Y11 @ Y21_inv

    return (A, B, C, D)


__all__ = [
    "mtee_ads_3port_smatrix",
    "mtee_ads_3port_smatrix_harmonic",
    "reduce_3port_to_2port_with_load",
    "s2port_to_abcd",
    "tee_3port_loaded_2port_abcd",
    "build_combined_tee_branch_harmonic_abcd",
]



