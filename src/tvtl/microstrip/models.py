"""Microstrip impedance and propagation models extracted from the notebooks."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tvtl.config import ConductorLossVariant, DielectricLossVariant
from tvtl.materials import FREE_SPACE_IMPEDANCE_OHM, SPEED_OF_LIGHT_M_PER_S, Substrate, VACUUM_PERMEABILITY_H_PER_M


def hammerstad_thin_metal(width_m: float, height_m: float, epsilon_r: float) -> tuple[float, float]:
    """Return zero-thickness Hammerstad impedance and effective permittivity."""

    u = width_m / height_m
    a_term = 1.0 + (1.0 / 49.0) * np.log(((u**4) + (u / 52.0) ** 2) / ((u**4) + 0.432))
    a_term += (1.0 / 18.7) * np.log(1.0 + (u / 18.1) ** 3)
    b_term = 0.564 * (((epsilon_r - 0.9) / (epsilon_r + 3.0)) ** 0.053)
    eps_eff_0 = ((epsilon_r + 1.0) / 2.0) + (((epsilon_r - 1.0) / 2.0) * ((1.0 + (10.0 / u)) ** (-a_term * b_term)))
    f_term = 6.0 + ((2.0 * np.pi - 6.0) * np.exp(-((30.666 / u) ** 0.7528)))
    z00 = (FREE_SPACE_IMPEDANCE_OHM / (2.0 * np.pi)) * np.log((f_term / u) + np.sqrt(1.0 + (2.0 / u) ** 2))
    return float(z00), float(eps_eff_0)


def hammerstad_kirschning(
    width_m: float,
    thickness_m: float,
    height_m: float,
    epsilon_r: float,
    frequency_hz: ArrayLike,
    preserve_legacy_thickness_bug: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return characteristic impedance and effective permittivity.

    Corrected finite-thickness handling is the default. The legacy notebook bug,
    which computes finite-thickness helper widths `ul`/`ur` but still calls the
    thin-metal routine with the original width, is available only through
    `preserve_legacy_thickness_bug=True`.
    """

    frequency = np.asarray(frequency_hz, dtype=float)
    u = width_m / height_m
    t_norm = thickness_m / height_m
    if t_norm < 0.0:
        raise ValueError("Negative strip thickness is invalid")

    if t_norm > 0.0:
        delta_ul = (t_norm / np.pi) * np.log(
            1.0 + (4.0 * np.exp(1.0)) / (t_norm * np.cosh(np.sqrt(6.517 * u)) ** -2)
        )
        delta_ur = 0.5 * (1.0 + (1.0 / np.cosh(np.sqrt(epsilon_r - 1.0)))) * delta_ul
    else:
        delta_ul = 0.0
        delta_ur = 0.0

    ul = u + delta_ul
    ur = u + delta_ur

    if preserve_legacy_thickness_bug:
        z_ur, eps_ur = hammerstad_thin_metal(width_m, height_m, epsilon_r)
        z_ul, eps_ul = hammerstad_thin_metal(width_m, height_m, epsilon_r)
    else:
        z_ur, eps_ur = hammerstad_thin_metal(ur * height_m, height_m, epsilon_r)
        z_ul, eps_ul = hammerstad_thin_metal(ul * height_m, height_m, epsilon_r)

    z0_f0 = z_ur / np.sqrt(eps_ur)
    eps_eff_f0 = eps_ur * (z_ul / z_ur) ** 2

    fn_cm = 1e-7 * frequency * height_m
    p1 = 0.27488 + ((0.6315 + (0.525 / (1.0 + 0.157 * fn_cm) ** 20)) * u) - (0.065683 * np.exp(-8.7513 * u))
    p2 = 0.33622 * (1.0 - np.exp(-0.03442 * epsilon_r))
    p3 = 0.0363 * np.exp(-4.6 * u) * (1.0 - np.exp(-((fn_cm / 3.87) ** 4.97)))
    p4 = 1.0 + 2.751 * (1.0 - np.exp(-((epsilon_r / 15.916) ** 8)))
    p_all = p1 * p2 * (((0.1844 + (p3 * p4)) * 10.0 * fn_cm) ** 1.5763)
    eps_eff_f = (eps_eff_f0 + epsilon_r * p_all) / (1.0 + p_all)

    fn_mm = 1e-6 * frequency * height_m
    r1 = np.minimum(0.03891 * epsilon_r**1.4, 20.0)
    r2 = np.minimum(0.267 * u**7, 20.0)
    r3 = 4.766 * np.exp(-3.228 * u**0.641)
    r4 = 0.016 + (0.0514 * epsilon_r) ** 4.524
    r5 = (fn_mm / 28.843) ** 12
    r6 = np.minimum(22.20 * u**1.92, 20.0)
    r7 = 1.206 - 0.3144 * np.exp(-r1) * (1.0 - np.exp(-r2))
    r8 = 1.0 + 1.275 * (1.0 - np.exp(-0.004625 * r3 * epsilon_r**1.674 * (fn_mm / 18.365) ** 2.745))
    r9 = 5.086 * r4 * (r5 / (0.3838 + 0.386 * r4)) * (np.exp(-r6) / (1.0 + 1.2992 * r5))
    r9 *= ((epsilon_r - 1.0) ** 6) / (1.0 + 10.0 * (epsilon_r - 1.0) ** 6)
    r10 = 0.00044 * epsilon_r**2.136 + 0.0184
    r11 = (fn_mm / 19.47) ** 6 / (1.0 + 0.0962 * (fn_mm / 19.47) ** 6)
    r12 = 1.0 / (1.0 + 0.00245 * u**2)
    r13 = 0.9408 * eps_eff_f**r8 - 0.9603
    r14 = (0.9408 - r9) * eps_eff_f0**r8 - 0.9603
    r15 = 0.707 * r10 * (fn_mm / 12.3) ** 1.097
    r16 = 1.0 + 0.0503 * epsilon_r**2 * r11 * (1.0 - np.exp(-(u / 15.0) ** 6))
    r17 = r7 * (1.0 - 1.1241 * (r12 / r16) * np.exp(-0.026 * fn_mm**1.15656 - r15))
    z0_f = z0_f0 * (r13 / r14) ** r17

    return np.asarray(z0_f, dtype=float), np.asarray(eps_eff_f, dtype=float)


def calc_gamma_legacy(
    frequency_hz: ArrayLike,
    z0_ohm: ArrayLike,
    epsilon_eff: ArrayLike,
    substrate: Substrate,
    *,
    conductor_loss_variant: ConductorLossVariant = ConductorLossVariant.LEGACY_SURFACE,
    dielectric_loss_variant: DielectricLossVariant = DielectricLossVariant.LEGACY_TAN_DELTA,
) -> NDArray[np.complex128]:
    """Notebook propagation constant.

    The attenuation expression is preserved as written in the Mathematica
    reference, even though its conductor-loss term deserves further validation.
    """

    frequency = np.asarray(frequency_hz, dtype=float)
    z0 = np.asarray(z0_ohm, dtype=float)
    eps_eff = np.asarray(epsilon_eff, dtype=float)

    k0 = (2.0 * np.pi * frequency) / SPEED_OF_LIGHT_M_PER_S
    lambda0 = 2.0 * np.pi / k0
    beta = k0 * np.sqrt(eps_eff)
    if dielectric_loss_variant == DielectricLossVariant.OFF:
        alpha_d = np.zeros_like(beta, dtype=float)
    else:
        alpha_d = (
            (substrate.epsilon_r / np.sqrt(eps_eff))
            * ((eps_eff - 1.0) / (substrate.epsilon_r - 1.0))
            * (np.pi / lambda0)
            * substrate.loss_tangent
        )

    if conductor_loss_variant == ConductorLossVariant.OFF:
        alpha_c = np.zeros_like(beta, dtype=float)
    else:
        rs = np.sqrt((np.pi * frequency * VACUUM_PERMEABILITY_H_PER_M) / substrate.conductivity_s_per_m)
        alpha_c = (8.686 * rs) / (2.0 * z0)
    return np.asarray(alpha_d + alpha_c + 1j * beta, dtype=np.complex128)


