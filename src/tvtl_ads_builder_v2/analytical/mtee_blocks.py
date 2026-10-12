"""Reference-plane MTEE blocks for the analytical leg (opt-in upgrade).

The v2 analytical legs (single-frequency S-parameters in
:mod:`.level10_sparams` and the Floquet harmonic-ABCD gain cascades in
:mod:`.ideal_c_gain` / :mod:`.parametric_gain`) model the MTEE T-junction
as an **ideal junction** by default: a pure shunt at the mid-node, with the
junction's geometry ignored. This module is the opt-in *reference-plane*
upgrade that wraps :func:`tvtl.microstrip.discontinuities.tee_junction`
(the Kirschning-Jansen "corrected_notebook" model) so the analytical leg
inserts the junction's reference-plane main-line transformation
(``abcd_left`` / ``abcd_right`` with the ``ta`` / ``tb`` scaling and the
``l_a`` / ``l_b`` length corrections) and the junction shunt parasitic
``j*bt`` -- the same physics ADS models for its native MTEE element.

High-frequency validity guard
-----------------------------
The corrected-notebook formula is only valid below its high-frequency edge.
For the paper-class geometry (main 4.27 mm, branch 1.93 mm) the
reference-plane main-line scaling ``|abcd_left[0,0]|`` collapses toward zero
above ~15 GHz and the ``ta`` / ``tb`` square-root radicand goes negative
(NaN) above ~21 GHz. A harmonic-ABCD gain cascade visits harmonics up to
``|ws + N*wm|`` (40-65 GHz for the design examples), so those upper
harmonics MUST be handled or they poison the cascade. The guard here falls
back to the **ideal junction** (identity main-line blocks, zero junction
shunt) for any frequency where the model is non-finite OR
``|abcd_left[0,0]| < validity_threshold`` (default 0.7, a conservative
"still close to unity" test). The fallback harmonics are deep-stopband /
evanescent and carry negligible signal/idler power, so the signal-proxy
gain is insensitive to the exact threshold (verified across 7-14 GHz cuts
in the forward prototype) while the fallback prevents the
non-physical pre-breakdown region from corrupting the result.
"""

from __future__ import annotations

import math

import numpy as np

from tvtl.config import TeeModelVariant
from tvtl.materials import Substrate
from tvtl.microstrip.discontinuities import tee_junction

__all__ = [
    "MTEE_VALIDITY_THRESHOLD",
    "tee_reference_plane_single",
    "tee_harmonic_blocks",
]


# Default validity threshold on ``|abcd_left[0, 0]|``: below this the
# corrected-notebook reference-plane model is past its high-frequency edge
# and the caller falls back to the ideal junction for that frequency.
MTEE_VALIDITY_THRESHOLD = 0.7

_IDENTITY_2X2 = np.eye(2, dtype=np.complex128)


def tee_reference_plane_single(
    width_main_m: float,
    width_branch_m: float,
    frequency_hz: float,
    substrate: Substrate,
    *,
    validity_threshold: float = MTEE_VALIDITY_THRESHOLD,
    tee_model_variant: TeeModelVariant = TeeModelVariant.CORRECTED_NOTEBOOK,
) -> tuple[np.ndarray, np.ndarray, complex, float, float, bool]:
    """Reference-plane tee at one positive frequency, with the validity guard.

    Parameters
    ----------
    width_main_m : float
        Through-line (main) microstrip width [m] (``w1 == w2``).
    width_branch_m : float
        Branch (stub-side) microstrip width [m] (``w3``).
    frequency_hz : float
        Positive frequency [Hz].
    substrate : tvtl.materials.Substrate
        Substrate carrying the junction.
    validity_threshold : float, optional
        Minimum ``|abcd_left[0, 0]|`` for the reference-plane model to be
        used; below it (or on a non-finite return) the caller should fall
        back to the ideal junction. Default :data:`MTEE_VALIDITY_THRESHOLD`.

    Returns
    -------
    tuple
        ``(abcd_left, abcd_right, j_bt, l_a, l_b, valid)`` where
        ``abcd_left`` / ``abcd_right`` are the 2x2 reference-plane main-line
        blocks, ``j_bt`` the junction shunt admittance [S], ``l_a`` / ``l_b``
        the main-line length corrections [m] (symmetric, ``l_a == l_b ==
        length_m / 2``), and ``valid`` whether the model passed the guard.
        When ``valid`` is ``False`` the matrices/shunt are still returned for
        inspection but the caller should substitute the ideal junction.
    """

    # The corrected-notebook ta/tb sqrt radicand goes negative past the
    # model's high-frequency edge; that NaN is detected by the validity guard
    # below, so the expected invalid-sqrt warning is suppressed here.
    with np.errstate(invalid="ignore", divide="ignore"):
        tee = tee_junction(
            float(width_main_m), float(width_branch_m), float(frequency_hz),
            substrate, tee_model_variant=tee_model_variant,
        )
    abcd_left = np.asarray(tee.abcd_left, dtype=np.complex128)
    abcd_right = np.asarray(tee.abcd_right, dtype=np.complex128)
    j_bt = complex(tee.shunt_admittance_siemens)
    length_tee = float(tee.length_m)
    l_a = l_b = 0.5 * length_tee  # symmetric main line (w1 == w2)
    finite = bool(
        np.all(np.isfinite(abcd_left))
        and np.all(np.isfinite(abcd_right))
        and math.isfinite(abs(j_bt))
    )
    valid = bool(finite and abs(abcd_left[0, 0]) >= float(validity_threshold))
    return abcd_left, abcd_right, j_bt, l_a, l_b, valid


def tee_harmonic_blocks(
    width_main_m: float,
    width_branch_m: float,
    omega_n: np.ndarray,
    substrate: Substrate,
    *,
    validity_threshold: float = MTEE_VALIDITY_THRESHOLD,
    tee_model_variant: TeeModelVariant = TeeModelVariant.CORRECTED_NOTEBOOK,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
    float,
    float,
]:
    """Per-harmonic reference-plane tee blocks for a harmonic-ABCD cascade.

    Evaluates :func:`tee_reference_plane_single` at each ``|omega_n| / (2*pi)``
    and assembles block-diagonal ``(A, B, C, D)`` matrices for ``abcd_left``
    and ``abcd_right``, plus the per-harmonic junction shunt vector. Negative
    harmonic frequencies use the complex conjugate of the positive-frequency
    block (``ABCD(-omega) = conj(ABCD(omega))`` for a real reciprocal
    network). Harmonics failing the validity guard fall back to the ideal
    junction (identity main-line block, zero shunt) for that index.

    Parameters
    ----------
    width_main_m, width_branch_m : float
        Main / branch microstrip widths [m].
    omega_n : numpy.ndarray
        Harmonic angular frequencies ``ws + n*wm`` [rad/s], shape ``(2N+1,)``.
    substrate : tvtl.materials.Substrate
        Substrate carrying the junction.
    validity_threshold : float, optional
        Passed to :func:`tee_reference_plane_single`.

    Returns
    -------
    tuple
        ``(block_left, block_right, j_bt_per_harmonic, l_a, l_b)``. The two
        blocks are ``(A, B, C, D)`` tuples of diagonal ``(2N+1, 2N+1)``
        complex matrices; ``j_bt_per_harmonic`` is the ``(2N+1,)`` junction
        shunt admittance to add to the branch ``C`` block; ``l_a`` / ``l_b``
        are the main-line length corrections taken at the smallest-``|omega|``
        (signal, ``n = 0``) harmonic, by which the caller shortens the
        through-line halves so the cell period is conserved.
    """

    omega = np.asarray(omega_n, dtype=float)
    if omega.ndim != 1:
        raise ValueError("omega_n must be 1-D.")
    size = omega.shape[0]
    a_l = np.zeros(size, dtype=np.complex128)
    b_l = np.zeros(size, dtype=np.complex128)
    c_l = np.zeros(size, dtype=np.complex128)
    d_l = np.zeros(size, dtype=np.complex128)
    a_r = np.zeros(size, dtype=np.complex128)
    b_r = np.zeros(size, dtype=np.complex128)
    c_r = np.zeros(size, dtype=np.complex128)
    d_r = np.zeros(size, dtype=np.complex128)
    j_bt_vec = np.zeros(size, dtype=np.complex128)

    # The length corrections come from the signal harmonic (smallest |omega|).
    n0 = int(np.argmin(np.abs(omega)))
    l_a = 0.0
    l_b = 0.0

    for k in range(size):
        w = float(omega[k])
        f_abs = abs(w) / (2.0 * math.pi)
        if f_abs <= 0.0:
            left = _IDENTITY_2X2
            right = _IDENTITY_2X2
            j_bt = 0j
            la = lb = 0.0
        else:
            left, right, j_bt, la, lb, valid = tee_reference_plane_single(
                width_main_m, width_branch_m, f_abs, substrate,
                validity_threshold=validity_threshold,
                tee_model_variant=tee_model_variant,
            )
            if not valid:
                left = _IDENTITY_2X2
                right = _IDENTITY_2X2
                j_bt = 0j
        if w < 0.0:
            left = left.conjugate()
            right = right.conjugate()
            j_bt = j_bt.conjugate()
        a_l[k], b_l[k], c_l[k], d_l[k] = left[0, 0], left[0, 1], left[1, 0], left[1, 1]
        a_r[k], b_r[k], c_r[k], d_r[k] = right[0, 0], right[0, 1], right[1, 0], right[1, 1]
        j_bt_vec[k] = j_bt
        if k == n0:
            l_a, l_b = la, lb

    block_left = (np.diag(a_l), np.diag(b_l), np.diag(c_l), np.diag(d_l))
    block_right = (np.diag(a_r), np.diag(b_r), np.diag(c_r), np.diag(d_r))
    return block_left, block_right, j_bt_vec, float(l_a), float(l_b)


