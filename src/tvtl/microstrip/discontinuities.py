"""Open-end, via, and tee-junction helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tvtl.abcd.matrices import line_abcd, series_impedance_abcd
from tvtl.config import HighFrequencyGuardrail, OpenEndCorrectionVariant, TeeModelVariant, ViaModelVariant
from tvtl.materials import FREE_SPACE_IMPEDANCE_OHM, SPEED_OF_LIGHT_M_PER_S, Substrate, VACUUM_PERMEABILITY_H_PER_M
from tvtl.microstrip.models import calc_gamma_legacy, hammerstad_kirschning


@dataclass(frozen=True, slots=True)
class TeeJunction:
    """Compact tee-junction model returned by the notebook formulas."""

    abcd_left: NDArray[np.complex128]
    abcd_right: NDArray[np.complex128]
    shunt_admittance_siemens: complex
    theta_stub: complex
    characteristic_impedance_ohm: float
    length_m: float
    target_total_admittance_siemens: complex | None = None


_NATIVE_ADS_COMPACT_SYMMETRIC_FREQ_START_HZ = 1.0e9
_NATIVE_ADS_COMPACT_SYMMETRIC_FREQ_STOP_HZ = 20.0e9
_NATIVE_ADS_COMPACT_SYMMETRIC_Z_OUTER_REAL_COEFFS = (
    12.743355861219035,
    -7.690120348377803,
    0.7587229253222988,
)
_NATIVE_ADS_COMPACT_SYMMETRIC_Z_OUTER_IMAG_COEFFS = (
    -19.411775719274818,
    0.4378755071708492,
    -0.7281520257739683,
)
_NATIVE_ADS_COMPACT_SYMMETRIC_CORE_C_BG_F = 5.61231920771344e-14
_NATIVE_ADS_COMPACT_SYMMETRIC_CORE_BRANCH_1 = (
    0.07647341372212625,
    3.5202360151312045e-09,
    6.4526419977087454e-12,
)
_NATIVE_ADS_COMPACT_SYMMETRIC_CORE_BRANCH_2 = (
    1.0913015026191426,
    2.924451619795718e-09,
    5.529707300532261e-14,
)


def _native_ads_compact_symmetric_coordinate(frequency_hz: float) -> float:
    span_hz = _NATIVE_ADS_COMPACT_SYMMETRIC_FREQ_STOP_HZ - _NATIVE_ADS_COMPACT_SYMMETRIC_FREQ_START_HZ
    return (float(frequency_hz) - _NATIVE_ADS_COMPACT_SYMMETRIC_FREQ_START_HZ) / span_hz


def _native_ads_compact_symmetric_outer_impedance_ohm(frequency_hz: float) -> complex:
    x = _native_ads_compact_symmetric_coordinate(frequency_hz)
    real_part = float(np.polyval(_NATIVE_ADS_COMPACT_SYMMETRIC_Z_OUTER_REAL_COEFFS, x))
    imag_part = float(np.polyval(_NATIVE_ADS_COMPACT_SYMMETRIC_Z_OUTER_IMAG_COEFFS, x))
    return complex(real_part, imag_part)


def _native_ads_compact_symmetric_core_admittance_siemens(frequency_hz: float) -> complex:
    omega = 2.0 * np.pi * float(frequency_hz)
    if omega == 0.0:
        raise ValueError("The native ADS compact symmetric tee law requires a non-zero frequency.")

    y_total = 1j * omega * _NATIVE_ADS_COMPACT_SYMMETRIC_CORE_C_BG_F
    for resistance_ohm, inductance_h, capacitance_f in (
        _NATIVE_ADS_COMPACT_SYMMETRIC_CORE_BRANCH_1,
        _NATIVE_ADS_COMPACT_SYMMETRIC_CORE_BRANCH_2,
    ):
        y_total += 1.0 / (resistance_ohm + 1j * omega * inductance_h + 1.0 / (1j * omega * capacitance_f))
    return complex(y_total)


def open_end_capacitance_f(
    width_m: float,
    z0_ohm: float,
    epsilon_eff: float,
    substrate: Substrate,
) -> float:
    """Kirschning-Franz open-end capacitance used in the notebook."""

    q1 = 0.434907 * ((epsilon_eff**0.81 + 0.26) / (epsilon_eff**0.81 - 0.189))
    q1 *= ((width_m / substrate.thickness_m) ** 0.8544 + 0.236) / ((width_m / substrate.thickness_m) ** 0.8544 + 0.87)
    q2 = 1.0 + ((width_m / substrate.thickness_m) ** 0.371) / ((2.358 * substrate.epsilon_r) + 1.0)
    q3 = 1.0 + (0.5274 / epsilon_eff**0.9236) * np.arctan(0.084 * (width_m / substrate.thickness_m) ** (1.9413 / q2))
    # Kirschning-Jansen-Koster open-end model (1981):
    #   q4 = 1 + 0.0377*(6 - 5*exp(0.036*(1 - eps_r)))*arctan(0.067*(W/h)^1.456)
    # The arctan multiplies ONLY the 0.0377*(...) term; the leading 1 is added.
    # (A prior `q4 = 1 + 0.0377*(...)` then `q4 *= arctan(...)` multiplied the whole
    # `1 + ...` by the arctan, collapsing q4 from ~1.0 to ~0.1 -> a ~10x oversized
    # open-end ΔL/C_end. The MLEF consistency audit
    # traced the analytical fringe to this; the corrected
    # C_end matches the ADS MLEF element to <0.5% across 2-16 GHz.)
    q4 = 1.0 + 0.0377 * (6.0 - 5.0 * np.exp(0.036 * (1.0 - substrate.epsilon_r))) * np.arctan(
        0.067 * (width_m / substrate.thickness_m) ** 1.456
    )
    q5 = 1.0 - 0.218 * np.exp(-7.5 * (width_m / substrate.thickness_m))
    delta_l = substrate.thickness_m * ((q1 * q3 * q5) / q4)
    return float((np.sqrt(epsilon_eff) / (SPEED_OF_LIGHT_M_PER_S * z0_ohm)) * delta_l)


def open_end_input_impedance_ohm(
    width_m: float,
    z0_ohm: float,
    epsilon_eff: float,
    theta: complex,
    substrate: Substrate,
    frequency_hz: float,
    open_end_correction_variant: OpenEndCorrectionVariant = OpenEndCorrectionVariant.KIRSCHNING_FRANZ,
) -> complex:
    """Return the open-end-corrected input impedance."""

    if open_end_correction_variant == OpenEndCorrectionVariant.OFF:
        return complex(z0_ohm / np.tanh(theta))
    else:
        c_end = open_end_capacitance_f(width_m, z0_ohm, epsilon_eff, substrate)
        zin_1 = -1j / (2.0 * np.pi * frequency_hz * c_end)
    return complex(z0_ohm * ((zin_1 + z0_ohm * np.tanh(theta)) / (z0_ohm + zin_1 * np.tanh(theta))))


def via_impedance_ohm(
    frequency_hz: float,
    via_radius_m: float,
    substrate: Substrate,
    preserve_legacy_thickness_bug: bool = False,
    via_model_variant: ViaModelVariant = ViaModelVariant.NOTEBOOK,
) -> complex:
    """Return the notebook via impedance model.

    Corrected thickness handling is the default. Legacy notebook matching is
    available only through `preserve_legacy_thickness_bug=True`.
    """

    z0_via, epsilon_eff_via = hammerstad_kirschning(
        width_m=via_radius_m,
        thickness_m=substrate.copper_thickness_m,
        height_m=substrate.thickness_m,
        epsilon_r=substrate.epsilon_r,
        frequency_hz=np.asarray([frequency_hz]),
        preserve_legacy_thickness_bug=preserve_legacy_thickness_bug,
    )
    gamma_via = calc_gamma_legacy(np.asarray([frequency_hz]), z0_via, epsilon_eff_via, substrate)[0]
    theta_via = gamma_via * via_radius_m

    d_via_inner = 2.0 * via_radius_m
    d_via_outer = d_via_inner + via_radius_m
    l_via = (VACUUM_PERMEABILITY_H_PER_M / (2.0 * np.pi)) * (
        substrate.thickness_m * np.log10((substrate.thickness_m + np.sqrt(via_radius_m**2 + substrate.thickness_m**2)) / via_radius_m)
    )
    l_via += 1.5 * (via_radius_m - np.sqrt(via_radius_m**2 + substrate.thickness_m**2))
    f_b = 1.0 / (np.pi * VACUUM_PERMEABILITY_H_PER_M * substrate.conductivity_s_per_m * substrate.copper_thickness_m**2)
    r_dc = substrate.thickness_m / (
        substrate.conductivity_s_per_m
        * np.pi
        * (via_radius_m**2 - (via_radius_m - substrate.copper_thickness_m) ** 2)
    )
    r_via = r_dc * np.sqrt(1.0 + (frequency_hz / f_b))
    z_series = r_via + 1j * 2.0 * np.pi * frequency_hz * l_via
    c_via = (1.41 * substrate.epsilon_r * substrate.thickness_m * d_via_inner) / (d_via_outer - d_via_inner)
    if via_model_variant == ViaModelVariant.SERIES_RL_ONLY:
        z_parallel = z_series
    else:
        z_parallel = 1.0 / ((1.0 / z_series) + 1j * 2.0 * np.pi * frequency_hz * c_via)
    z0_value = float(z0_via[0])
    return complex(z0_value * ((z_parallel + z0_value * np.tanh(theta_via)) / (z0_value + z_parallel * np.tanh(theta_via))))


def via_inductance_bahl_H(
    substrate_thickness_m: float,
    via_radius_m: float,
    *,
    n_parallel: int = 1,
) -> float:
    """Bahl simple via-to-ground inductance for a barrel-shaped via.

    Returns the lumped series inductance of a single cylindrical via
    connecting the top-side microstrip to the ground plane through the
    substrate, per Bahl, "Lumped Elements for RF and Microwave Circuits"
    (Eq. 3.50):

    .. math::
        L_\\mathrm{via} = \\frac{\\mu_0 h}{2 \\pi}
            \\left[ \\ln\\!\\left(\\frac{2 h}{r}\\right) + \\tfrac{1}{2} \\right]

    where ``h`` is the substrate thickness (the via length) and ``r`` is
    the via barrel radius. The formula assumes a thin, uniform-current
    cylinder; it is a frequency-independent quasi-static estimate, valid
    when ``h`` is small compared with the operating wavelength (i.e.,
    the via barrel is electrically short).

    The complementary frequency-dependent :func:`via_impedance_ohm`
    Goldfarb-Pucel model accumulates the same physics into a short
    transmission-line section with a parallel pad capacitance and a
    skin-effect series resistance. The two are not interchangeable: Bahl
    returns a real lumped inductance in henries, while
    :func:`via_impedance_ohm` returns the via's transformed complex input
    impedance at one frequency. The analytical Floquet stub-termination
    (the harmonic ABCD lift) consumes the lumped Bahl value because
    it needs a clean ``Z_load(omega) = j*omega*L_via`` to anchor the
    transmission-line input-impedance recursion at the stub's far end.

    Parameters
    ----------
    substrate_thickness_m : float
        Substrate thickness ``h`` in metres. Must be positive.
    via_radius_m : float
        Via barrel radius ``r`` in metres. Must be positive and strictly
        less than ``h`` (a via wider than the substrate is unphysical).
    n_parallel : int, optional
        Number of identical vias in parallel between the stub end and the
        ground plane. ``L_eff = L_single / n_parallel``. Default 1.

    Returns
    -------
    float
        Lumped series inductance in henries.
    """

    h = float(substrate_thickness_m)
    r = float(via_radius_m)
    n = int(n_parallel)
    if h <= 0.0:
        raise ValueError(f"substrate_thickness_m must be positive (got {h}).")
    if r <= 0.0:
        raise ValueError(f"via_radius_m must be positive (got {r}).")
    if n < 1:
        raise ValueError(f"n_parallel must be >= 1 (got {n}).")
    if r >= h:
        raise ValueError(
            f"via_radius_m ({r}) must be strictly less than "
            f"substrate_thickness_m ({h}); a via wider than the substrate "
            f"is unphysical."
        )
    l_single = (VACUUM_PERMEABILITY_H_PER_M * h / (2.0 * np.pi)) * (
        float(np.log(2.0 * h / r)) + 0.5
    )
    return float(l_single / n)


def tee_junction(
    width1_m: float,
    width3_m: float,
    frequency_hz: float,
    substrate: Substrate,
    preserve_legacy_thickness_bug: bool = False,
    tee_model_variant: TeeModelVariant = TeeModelVariant.CORRECTED_NOTEBOOK,
    high_frequency_guardrail: HighFrequencyGuardrail = HighFrequencyGuardrail.OFF,
) -> TeeJunction:
    """Return the tee-junction model used in the principal notebooks.

    Corrected thickness handling is the default. Legacy notebook matching is
    available only through `preserve_legacy_thickness_bug=True`.
    """

    width2_m = width1_m
    if tee_model_variant == TeeModelVariant.IDEAL_SYMMETRIC_LOSSLESS:
        # MTEE_ADS-aligned variant. Returns identity main-line ABCD
        # blocks and zero shunt admittance to match MTEE_ADS's empirically-
        # measured near-ideal junction behavior in the signal band (3-7 GHz).
        # See the variant docstring on TeeModelVariant for the empirical
        # probe context.
        identity = np.eye(2, dtype=np.complex128)
        return TeeJunction(
            abcd_left=identity,
            abcd_right=identity,
            shunt_admittance_siemens=0j,
            theta_stub=0j,
            characteristic_impedance_ohm=0.0,
            length_m=0.0,
            target_total_admittance_siemens=None,
        )
    za, epsilon_eff_a = hammerstad_kirschning(
        width1_m,
        substrate.copper_thickness_m,
        substrate.thickness_m,
        substrate.epsilon_r,
        np.asarray([frequency_hz]),
        preserve_legacy_thickness_bug=preserve_legacy_thickness_bug,
    )
    zb, epsilon_eff_b = hammerstad_kirschning(
        width2_m,
        substrate.copper_thickness_m,
        substrate.thickness_m,
        substrate.epsilon_r,
        np.asarray([frequency_hz]),
        preserve_legacy_thickness_bug=preserve_legacy_thickness_bug,
    )
    z2, epsilon_eff_2 = hammerstad_kirschning(
        width3_m,
        substrate.copper_thickness_m,
        substrate.thickness_m,
        substrate.epsilon_r,
        np.asarray([frequency_hz]),
        preserve_legacy_thickness_bug=preserve_legacy_thickness_bug,
    )
    za_v = float(za[0])
    zb_v = float(zb[0])
    z2_v = float(z2[0])
    eps_a = float(epsilon_eff_a[0])
    eps_b = float(epsilon_eff_b[0])
    eps_2 = float(epsilon_eff_2[0])

    gamma1 = calc_gamma_legacy(np.asarray([frequency_hz]), np.asarray([za_v]), np.asarray([eps_a]), substrate)[0]
    gamma2 = calc_gamma_legacy(np.asarray([frequency_hz]), np.asarray([zb_v]), np.asarray([eps_b]), substrate)[0]
    gamma3 = calc_gamma_legacy(np.asarray([frequency_hz]), np.asarray([z2_v]), np.asarray([eps_2]), substrate)[0]

    da_equiv = (FREE_SPACE_IMPEDANCE_OHM * substrate.thickness_m) / (np.sqrt(eps_a) * za_v)
    db_equiv = (FREE_SPACE_IMPEDANCE_OHM * substrate.thickness_m) / (np.sqrt(eps_b) * zb_v)
    d2_equiv = (FREE_SPACE_IMPEDANCE_OHM * substrate.thickness_m) / (np.sqrt(eps_2) * z2_v)

    fp_a = 4e5 * (za_v / substrate.thickness_m)
    fp_b = 4e5 * (zb_v / substrate.thickness_m)
    lambda_a = SPEED_OF_LIGHT_M_PER_S / (np.sqrt(eps_a) * frequency_hz)
    lambda_b = SPEED_OF_LIGHT_M_PER_S / (np.sqrt(eps_b) * frequency_hz)

    da = 0.055 * d2_equiv * (za_v / z2_v) * (1.0 - 2.0 * (za_v / z2_v) * (frequency_hz / fp_a) ** 2)
    db = 0.055 * d2_equiv * (zb_v / z2_v) * (1.0 - 2.0 * (zb_v / z2_v) * (frequency_hz / fp_b) ** 2)
    r_term = np.sqrt(za_v * zb_v) / z2_v
    q_term = frequency_hz**2 / (fp_a * fp_b)
    d2_exponential_term = 0.7 * np.exp(-1.6 * r_term)
    d2_mixed_term = 0.25 * r_term * q_term
    if tee_model_variant in {
        TeeModelVariant.CORRECTED_NOTEBOOK_NO_D2_EXP_V1,
        TeeModelVariant.CORRECTED_NOTEBOOK_NO_D2_EXP_HALF_MIXED_V1,
    }:
        d2_exponential_term = 0.0
    if tee_model_variant == TeeModelVariant.CORRECTED_NOTEBOOK_NO_D2_EXP_HALF_MIXED_V1:
        d2_mixed_term = 0.125 * r_term * q_term
    d2 = np.sqrt(da_equiv * db_equiv) * (
        0.5
        - r_term * (0.05 + d2_exponential_term + d2_mixed_term - 0.17 * np.log(r_term))
    )

    l_a = 0.5 * width3_m - da
    l_b = 0.5 * width3_m - db
    l_2 = 0.5 * width1_m - d2
    theta1 = gamma1 * l_a
    theta2 = gamma2 * l_b
    theta3 = gamma3 * l_2
    length_tee = l_a + l_b

    if tee_model_variant == TeeModelVariant.NATIVE_ADS_COMPACT_SYMMETRIC_V1:
        line_sym = 0.5 * (
            line_abcd(za_v, theta1) + line_abcd(zb_v, theta2)
        )
        z_outer_total = _native_ads_compact_symmetric_outer_impedance_ohm(frequency_hz)
        outer = np.asarray(
            series_impedance_abcd(0.5 * z_outer_total)
            @ line_sym
            @ series_impedance_abcd(0.5 * z_outer_total),
            dtype=np.complex128,
        )
        return TeeJunction(
            abcd_left=outer,
            abcd_right=outer,
            shunt_admittance_siemens=0j,
            theta_stub=complex(theta3),
            characteristic_impedance_ohm=z2_v,
            length_m=float(length_tee),
            target_total_admittance_siemens=_native_ads_compact_symmetric_core_admittance_siemens(frequency_hz),
        )

    radicand_a = (
        1.0
        - np.pi
        * (frequency_hz / fp_a) ** 2
        * ((1.0 / 12.0) * (za_v / z2_v) ** 2 + (0.5 - (d2 / da_equiv)) ** 2)
    )
    radicand_b = (
        1.0
        - np.pi
        * (frequency_hz / fp_b) ** 2
        * ((1.0 / 12.0) * (zb_v / z2_v) ** 2 + (0.5 - (d2 / db_equiv)) ** 2)
    )
    if high_frequency_guardrail == HighFrequencyGuardrail.CLIP_DISCONTINUITY_SQRT:
        radicand_a = max(float(radicand_a), 0.0)
        radicand_b = max(float(radicand_b), 0.0)
    ta = np.sqrt(radicand_a)
    tb = np.sqrt(radicand_b)
    if tee_model_variant == TeeModelVariant.IDENTITY_SCALING:
        ta = 1.0
        tb = 1.0
    elif high_frequency_guardrail == HighFrequencyGuardrail.CLIP_DISCONTINUITY_SQRT:
        ta = max(float(ta), 1e-12)
        tb = max(float(tb), 1e-12)
    bt = 5.5 * np.sqrt((da_equiv * db_equiv) / (lambda_a * lambda_b))
    bt *= ((substrate.epsilon_r + 2.0) / substrate.epsilon_r) * (1.0 / (z2_v * ta * tb))
    bt *= (np.sqrt(da * db) / d2_equiv)
    bt *= 1.0 + 0.9 * np.log10(r_term) + 4.5 * r_term * q_term - 4.4 * np.exp(-1.3 * r_term) - 20.0 * (z2_v / FREE_SPACE_IMPEDANCE_OHM) ** 2

    abcd_left = line_abcd(za_v, theta1) @ np.asarray([[ta, 0.0], [0.0, 1.0 / ta]], dtype=np.complex128)
    abcd_right = line_abcd(zb_v, theta2) @ np.asarray([[tb, 0.0], [0.0, 1.0 / tb]], dtype=np.complex128)
    y_stub = 1j * bt

    return TeeJunction(
        abcd_left=abcd_left,
        abcd_right=abcd_right,
        shunt_admittance_siemens=complex(y_stub),
        theta_stub=complex(theta3),
        characteristic_impedance_ohm=z2_v,
        length_m=float(length_tee),
    )


