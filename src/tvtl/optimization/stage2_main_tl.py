"""Stage 2 -- main transmission-line impedance adaptation (Bloch matching).

Given a Stage-1 stub design, pick the main-line characteristic impedance
``Z_TL`` (hence the main microstrip width) so the periodic cascade's **Bloch
impedance** equals 50 ohm at the design frequency -- so the amplifier presents a
matched interface to 50 ohm terminations. Decoupled from Stage 1 (it consumes
the stub design + the varactor + the substrate) and from the optimization
objective (it solves a 1-D impedance match, not a gain optimization).

The Bloch impedance is computed on the **static (pump-off) scalar unit cell** at
``f0``, assembled from the *same* building blocks the Stage-4 analytical cascade
uses at its ``n = 0`` harmonic -- the reference-plane MTEE
(:func:`...analytical.mtee_blocks.tee_reference_plane_single`), the lossy main
line (Hammerstad-Kirschning + ``calc_gamma_legacy``), and the static branch
admittance ``1 / (Z_pkg + 1/(jw0 C0) + Z_stub(w0))`` -- so the match holds on the
cell Stage 4 actually validates.

For a symmetric reciprocal unit cell (``A = D``) the Bloch (image) impedance is
``Z_B = sqrt(B / C)`` (the forward-wave root, ``Re Z_B >= 0``).
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from tvtl.abcd.matrices import line_abcd, shunt_admittance_abcd
from tvtl.microstrip.models import calc_gamma_legacy, hammerstad_kirschning
from tvtl.theory.dispersion import bloch_dispersion
from tvtl.varactors import RealisticVaractor
from tvtl_ads_builder_v2.analytical.mtee_blocks import tee_reference_plane_single

from .stage1_per_cell import StubCandidate
from .substrate import ROGERS_4003C, SubstrateConfig, microstrip_width_for_z0
from .topology_generator import topology_y_stub_fn

__all__ = [
    "MainTLDesign",
    "bloch_impedance",
    "bloch_attenuation_per_cell",
    "static_cell_abcd",
    "match_z_tl",
    "MAIN_PERIOD_M",
    "BRANCH_WIDTH_M",
    "Z_TL_SEARCH_LO_OHM",
    "Z_TL_SEARCH_HI_OHM",
]

# Paper structural-class geometry (matches parity.optimized_design; the design
# freedom is the stub topology + the main width chosen here, not the period).
MAIN_PERIOD_M = 6.11e-3
BRANCH_WIDTH_M = 1.93e-3

# Bloch-match search bracket on Z_TL (ohm) and the realizable Z_TL band.
Z_TL_SEARCH_LO_OHM = 20.0
Z_TL_SEARCH_HI_OHM = 120.0
_Z_TL_REALIZABLE_LO_OHM = 25.0   # ~10 mm-wide microstrip on Rogers 4003C
_Z_TL_REALIZABLE_HI_OHM = 100.0  # ~0.3 mm-wide microstrip on Rogers 4003C
TARGET_BLOCH_OHM = 50.0


@dataclass(frozen=True)
class MainTLDesign:
    """Result of the Stage-2 Bloch-impedance match for one stub design.

    ``realizable`` means the chosen ``Z_TL`` is a physically-realizable Rogers-
    4003C microstrip (in band, valid width) -- *not* that the Bloch impedance
    hit 50 ohm. ``matched`` is the latter (``Re Z_B`` within tolerance of 50);
    when ``False`` the periodic line's natural Bloch impedance differs from 50
    (e.g. a heavily-loaded high-gain stub gives a low Z_B) and Stage 3 supplies
    the impedance transformation. The two are independent: a high-gain design is
    typically ``realizable=True, matched=False`` (real but low Z_B -> a clean
    quarter-wave match in Stage 3).
    """

    stub: StubCandidate
    z_tl_ohm: float
    w_main_m: float
    bloch_z_ohm: complex
    match_residual_ohm: float  # |Re Z_B - 50| at the chosen Z_TL
    realizable: bool           # Z_TL is a realizable microstrip
    matched: bool              # Re Z_B reached ~50 by main width alone
    f0_Hz: float
    note: str = ""


def _static_branch_admittance(
    stub: StubCandidate,
    varactor: RealisticVaractor,
    substrate: SubstrateConfig,
    w0: float,
) -> complex:
    """Static (pump-off) shunt-branch admittance at ``w0`` [S].

    ``Y = 1 / (Z_pkg + 1/(jw0 C0) + Z_stub(w0))`` with ``Z_pkg = R_series +
    jw0 L_pkg`` and ``Z_stub`` the topology stub (incl. via / termination).
    """

    z_pkg = float(varactor.R_series_ohm) + 1j * w0 * float(varactor.L_pkg_h)
    z_cap = 1.0 / (1j * w0 * float(varactor.C0_f))
    y_stub = topology_y_stub_fn(stub.topology, stub.params_array, substrate)(w0)
    z_stub = 1.0 / y_stub
    return 1.0 / (z_pkg + z_cap + z_stub)


def static_cell_abcd(
    stub: StubCandidate,
    varactor: RealisticVaractor,
    z_tl_ohm: float,
    *,
    substrate: SubstrateConfig = ROGERS_4003C,
    f0_Hz: float | None = None,
    use_mtee: bool = True,
    main_period_m: float | None = None,
) -> np.ndarray:
    """2x2 ABCD of the static (pump-off) unit cell at ``f0`` for a given Z_TL.

    ``line(half - l_a) @ MTEE_left @ shunt(Y_branch + j*bt) @ MTEE_right @
    line(half - l_b)`` -- the same cell the Stage-4 cascade uses at its n=0
    harmonic. Shared by :func:`bloch_impedance`, :func:`bloch_attenuation_per_cell`,
    and the Stage-3 edge impedance.

    ``main_period_m`` (default ``None`` -> :data:`MAIN_PERIOD_M`, 6.11 mm) sets the
    per-cell main-line period, hence the line-segment lengths ``half - l_{a,b}``.
    The self-pumped engine scales this period inversely with signal frequency, so
    a band-structure screen of an *off-3.1-GHz* emitted device must pass the
    engine's scaled period here to see the cell the device actually realises;
    leaving it ``None`` reproduces the historical fixed-period cell bit-for-bit
    (Stage-2 Bloch match, Stage-3 edge impedance), so every existing caller is
    byte-identical.
    """

    f0 = float(stub.f0_Hz if f0_Hz is None else f0_Hz)
    w0 = 2.0 * math.pi * f0
    w_main = microstrip_width_for_z0(z_tl_ohm, substrate, f0)

    sub_map = substrate.stub_substrate_mapping()
    tvtl_sub = substrate.to_tvtl_substrate()
    z0_arr, eps_arr = hammerstad_kirschning(
        w_main, sub_map["t_m"], sub_map["h_m"], sub_map["eps_r"], [f0]
    )
    z0_main = float(z0_arr[0])
    gamma_main = complex(calc_gamma_legacy([f0], z0_arr, eps_arr, tvtl_sub)[0])

    if use_mtee:
        left, right, j_bt, l_a, l_b, valid = tee_reference_plane_single(
            w_main, BRANCH_WIDTH_M, f0, tvtl_sub
        )
        if not valid:
            left = right = np.eye(2, dtype=np.complex128)
            j_bt = 0j
            l_a = l_b = 0.0
    else:
        left = right = np.eye(2, dtype=np.complex128)
        j_bt = 0j
        l_a = l_b = 0.0

    period = MAIN_PERIOD_M if main_period_m is None else float(main_period_m)
    half = period / 2.0
    len_left = max(half - float(l_a), 0.0)
    len_right = max(half - float(l_b), 0.0)

    line_l = line_abcd(z0_main, gamma_main * len_left)
    line_r = line_abcd(z0_main, gamma_main * len_right)
    y_branch = _static_branch_admittance(stub, varactor, substrate, w0)
    shunt = shunt_admittance_abcd(y_branch + j_bt)
    return line_l @ np.asarray(left) @ shunt @ np.asarray(right) @ line_r


def bloch_impedance(
    stub: StubCandidate,
    varactor: RealisticVaractor,
    z_tl_ohm: float,
    *,
    substrate: SubstrateConfig = ROGERS_4003C,
    f0_Hz: float | None = None,
    use_mtee: bool = True,
) -> complex:
    """Bloch (image) impedance of the static unit cell at ``f0`` [ohm].

    For a symmetric reciprocal cell ``Z_B = sqrt(B/C)`` (forward-wave root,
    ``Re >= 0``); computed from the general reciprocal form so it is robust to
    small asymmetry.
    """

    cell = static_cell_abcd(
        stub, varactor, z_tl_ohm, substrate=substrate, f0_Hz=f0_Hz, use_mtee=use_mtee
    )
    a, b = cell[0, 0], cell[0, 1]
    c, d = cell[1, 0], cell[1, 1]
    # General reciprocal Bloch impedance, reducing to sqrt(B/C) for A == D.
    disc = cmath.sqrt((a + d) ** 2 - 4.0)
    # forward-wave root: pick the sign giving Re Z_B >= 0
    zb_plus = -2.0 * b / ((a - d) - disc)
    zb_minus = -2.0 * b / ((a - d) + disc)
    zb = zb_plus if zb_plus.real >= 0.0 else zb_minus
    if zb.real < 0.0:
        zb = -zb
    return complex(zb)


def bloch_attenuation_per_cell(
    stub: StubCandidate,
    varactor: RealisticVaractor,
    z_tl_ohm: float,
    *,
    substrate: SubstrateConfig = ROGERS_4003C,
    f0_Hz: float | None = None,
    use_mtee: bool = True,
    main_period_m: float | None = None,
) -> float:
    """Per-cell passive Bloch attenuation ``alpha_p`` [Np/cell] at ``f0``.

    Reuses the static (pump-off) unit cell (:func:`static_cell_abcd`) and the
    closed-form trace-arccosh Bloch extraction
    (:func:`tvtl.theory.dispersion.bloch_dispersion`): for a reciprocal cell
    ``cosh(gamma * period) = (A + D) / 2``, so the **per-cell** attenuation is
    ``alpha_p = |Re(gamma) * period| = |Re(arccosh((A + D) / 2))|`` -- a
    period-independent measure of how strongly one cell suppresses a wave at
    ``f0``. A frequency with ``alpha_p`` above a small propagating threshold
    (~0.02 Np/cell, the net-gain propagating-threshold convention) sits in a **Bloch
    stopband**: the passive cascade deeply attenuates it, so any analytical
    parametric-gain *ratio* there is artefactually inflated (tiny pump-off
    denominator) and the realised transducer gain is poor. The self-pumped
    walk-down uses this to keep its candidate pool in the passband.

    ``main_period_m`` must be the period the device is actually emitted at (the
    engine's scaled period); see :func:`static_cell_abcd`.
    """

    cell = static_cell_abcd(
        stub, varactor, z_tl_ohm, substrate=substrate, f0_Hz=f0_Hz,
        use_mtee=use_mtee, main_period_m=main_period_m,
    )
    period = MAIN_PERIOD_M if main_period_m is None else float(main_period_m)
    a, b = cell[0, 0], cell[0, 1]
    c, d = cell[1, 0], cell[1, 1]
    gamma = bloch_dispersion((a, b, c, d), period)
    gamma_scalar = complex(np.asarray(gamma).reshape(-1)[0])
    return float(abs(gamma_scalar.real) * period)


def match_z_tl(
    stub: StubCandidate,
    varactor: RealisticVaractor,
    *,
    substrate: SubstrateConfig = ROGERS_4003C,
    f0_Hz: float | None = None,
    use_mtee: bool = True,
    z_lo_ohm: float = Z_TL_SEARCH_LO_OHM,
    z_hi_ohm: float = Z_TL_SEARCH_HI_OHM,
    target_ohm: float = TARGET_BLOCH_OHM,
) -> MainTLDesign:
    """Find Z_TL making ``Re Z_B(f0) = target`` (50 ohm) for a stub design.

    Brent root-find on ``Re Z_B(z_tl) - target`` over ``[z_lo, z_hi]``. If the
    target is not bracketed (the stub over/under-loads the line so no realizable
    Z_TL matches), returns ``realizable=False`` with the closest-achievable Z_TL
    from a grid scan (a documented framework boundary; the candidate is dropped
    or handed to Stage-3 matching). Realizability also requires Z_TL within the
    Rogers-4003C microstrip band (~25-100 ohm).
    """

    f0 = float(stub.f0_Hz if f0_Hz is None else f0_Hz)

    def re_resid(z_tl: float) -> float:
        return bloch_impedance(
            stub, varactor, z_tl, substrate=substrate, f0_Hz=f0, use_mtee=use_mtee
        ).real - float(target_ohm)

    r_lo = re_resid(z_lo_ohm)
    r_hi = re_resid(z_hi_ohm)
    note = ""
    if r_lo * r_hi <= 0.0:
        z_tl = float(brentq(re_resid, z_lo_ohm, z_hi_ohm, xtol=1e-4, rtol=1e-8))
    else:
        # Not bracketed -> closest achievable (grid scan).
        grid = np.linspace(z_lo_ohm, z_hi_ohm, 64)
        resids = np.array([abs(re_resid(z)) for z in grid])
        z_tl = float(grid[int(np.argmin(resids))])
        note = (
            f"Bloch Re(Z_B) does not reach {target_ohm:.0f} ohm in "
            f"[{z_lo_ohm:.0f},{z_hi_ohm:.0f}] ohm; closest at Z_TL={z_tl:.1f} ohm "
            f"(residual {resids.min():.1f} ohm). Stage-3 matching required."
        )

    zb = bloch_impedance(stub, varactor, z_tl, substrate=substrate, f0_Hz=f0, use_mtee=use_mtee)
    residual = abs(zb.real - float(target_ohm))
    try:
        w_main = microstrip_width_for_z0(z_tl, substrate, f0)
        w_ok = True
    except ValueError as exc:
        w_main = float("nan")
        w_ok = False
        note = (note + " " if note else "") + str(exc)

    z_realizable = _Z_TL_REALIZABLE_LO_OHM <= z_tl <= _Z_TL_REALIZABLE_HI_OHM
    realizable = bool(w_ok and z_realizable)
    matched = bool(residual <= 1.0)
    if not z_realizable and not note:
        note = (
            f"Z_TL={z_tl:.1f} ohm outside the Rogers-4003C realizable band "
            f"[{_Z_TL_REALIZABLE_LO_OHM:.0f},{_Z_TL_REALIZABLE_HI_OHM:.0f}] ohm."
        )
    if realizable and not matched and not note:
        note = (
            f"natural Bloch Z_B={zb.real:.1f}{zb.imag:+.1f}j ohm (loaded line); "
            f"Stage-3 matching transforms it to 50 ohm."
        )

    return MainTLDesign(
        stub=stub, z_tl_ohm=float(z_tl), w_main_m=float(w_main),
        bloch_z_ohm=complex(zb), match_residual_ohm=float(residual),
        realizable=realizable, matched=matched, f0_Hz=f0, note=note,
    )


