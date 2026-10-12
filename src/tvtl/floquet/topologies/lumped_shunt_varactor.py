"""Lumped shunt varactor topology.

The unit cell is the simplest TVTL element: a main microstrip line of width
``main_tl_w`` and period ``main_tl_p`` with the time-varying RLC connected
directly between the centre of the cell and ground (no passive stub, no via
or open-stub matching network). The shunt admittance comes directly from
the Floquet Green's function of the varactor.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from tvtl.materials import SPEED_OF_LIGHT_M_PER_S
from tvtl.microstrip.models import hammerstad_kirschning

from ..harmonic_abcd import (
    cascade_harmonic_abcd,
    stub_harmonic_admittance,
    tl_section_harmonic_abcd,
)
from ..topology import DesignVariableSpec, TopologyDescriptor, register_topology


def _z0_and_v_phase(width_m: float, substrate: Mapping[str, Any], frequency_hz: float) -> tuple[float, float]:
    eps_r = float(substrate["eps_r"])
    h = float(substrate["h"])
    copper_thickness = float(substrate.get("copper_thickness", 17.5e-6))
    z0_arr, eps_eff_arr = hammerstad_kirschning(
        float(width_m), copper_thickness, h, eps_r, np.asarray([float(frequency_hz)], dtype=float)
    )
    eps_eff = float(eps_eff_arr[0])
    v_phase = SPEED_OF_LIGHT_M_PER_S / math.sqrt(max(eps_eff, np.finfo(float).tiny))
    return float(z0_arr[0]), v_phase


def _shunt_block_from_y(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = Y.shape[0]
    identity = np.eye(size, dtype=np.complex128)
    zero = np.zeros((size, size), dtype=np.complex128)
    return identity, zero, np.asarray(Y, dtype=np.complex128), identity


def build_lumped_shunt_varactor_block_abcd(
    design: Mapping[str, Any],
    varactor_params: Mapping[str, float],
    modulation_params: Mapping[str, float],
    substrate: Mapping[str, Any],
    ws: float,
    wm: float,
    N: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Harmonic-basis ABCD of the lumped-shunt-varactor unit cell.

    Parameters
    ----------
    design : Mapping
        Flat design dict with keys ``main_tl_p`` and ``main_tl_w`` [m].
    varactor_params, modulation_params, substrate
        Same conventions as the other topology builders.
    ws, wm : float
        Signal and modulation angular frequencies [rad/s].
    N : int
        Floquet truncation order (>= 1).

    Returns
    -------
    tuple of numpy.ndarray
        ``(A, B, C, D)`` block matrices of shape ``(2N+1, 2N+1)``.
    """

    truncation = int(N)
    if truncation < 1:
        raise ValueError("N must be at least 1 to access the idler harmonic.")
    fs = float(modulation_params["fs"])
    phi = float(modulation_params["phi"])
    dS_fraction = float(modulation_params["dS_fraction"])
    if not (0.0 <= dS_fraction <= 1.0):
        raise ValueError("dS_fraction must lie in [0, 1].")
    C0 = float(varactor_params["C0"])
    R = float(varactor_params["R"])
    L = float(varactor_params["L"])
    if C0 <= 0.0 or R < 0.0 or L < 0.0:
        raise ValueError("varactor_params must satisfy C0 > 0 and R, L >= 0.")
    S0 = 1.0 / C0
    dS = dS_fraction * S0

    main_p = float(design["main_tl_p"])
    main_w = float(design["main_tl_w"])
    half_p = 0.5 * main_p

    z0_main, v_phase_main = _z0_and_v_phase(main_w, substrate, fs)
    main_block = tl_section_harmonic_abcd(half_p, z0_main, v_phase_main, ws, wm, truncation)

    Y_var = stub_harmonic_admittance(S0, dS, phi, R, L, ws, wm, truncation)
    stub_block = _shunt_block_from_y(Y_var)

    return cascade_harmonic_abcd([main_block, stub_block, main_block])


def _lumped_shunt_varactor_period(design: Mapping[str, Any]) -> float:
    return float(design["main_tl_p"])


def _lumped_shunt_varactor_static_check(design: Mapping[str, Any]) -> bool:
    return float(design.get("main_tl_p", 0.0)) > 0.0 and float(design.get("main_tl_w", 0.0)) > 0.0


def _lumped_shunt_varactor_pump_voltage_extractor(
    design: Mapping[str, Any],
    varactor_params: Mapping[str, float],
    substrate: Mapping[str, Any],
    n_cells: int,
    pump_frequency_Hz: float,
    pump_amplitude_V: float = 1.0,
    source_z_ohm: float = 50.0,
    load_z_ohm: float = 50.0,
) -> np.ndarray:
    """Per-cell varactor terminal voltage for the lumped-shunt topology.

    The varactor is connected directly between the main-line junction and
    ground. The varactor terminal voltage therefore equals the junction
    voltage.
    """

    from ..pump_analysis import _cascade_junction_voltages, _line_abcd, _z0_and_v_phase

    f_pump = float(pump_frequency_Hz)
    omega = 2.0 * math.pi * f_pump
    C0 = float(varactor_params["C0"])
    R = float(varactor_params["R"])
    L = float(varactor_params["L"])
    z_var = complex(R, omega * L - 1.0 / (omega * C0))
    y_var = 1.0 / z_var

    main_p = float(design["main_tl_p"])
    main_w = float(design["main_tl_w"])
    if main_p <= 0.0 or main_w <= 0.0:
        raise ValueError("lumped_shunt_varactor main TL geometry must be positive.")
    z0_main, v_phase_main = _z0_and_v_phase(main_w, substrate, f_pump)
    beta_main = omega / v_phase_main
    M_main_half = _line_abcd(z0_main, beta_main * 0.5 * main_p)

    junctions = _cascade_junction_voltages(
        M_main_half,
        y_var,
        int(n_cells),
        float(source_z_ohm),
        float(load_z_ohm),
        float(pump_amplitude_V),
    )
    return junctions


LUMPED_SHUNT_VARACTOR_DESCRIPTOR = TopologyDescriptor(
    name="lumped_shunt_varactor",
    version="1.0",
    description=(
        "Simplest TVTL unit cell: a main microstrip line with a time-varying RLC "
        "connected directly between the line centre and ground. No passive stub "
        "or matching network."
    ),
    design_variables=(
        DesignVariableSpec("main_tl_p", 3.0e-3, 12.0e-3, "m", "Period of the main microstrip line."),
        DesignVariableSpec("main_tl_w", 1.5e-3, 6.0e-3, "m", "Width of the main microstrip line."),
    ),
    build_unit_cell_block_abcd=build_lumped_shunt_varactor_block_abcd,
    period=_lumped_shunt_varactor_period,
    static_check=_lumped_shunt_varactor_static_check,
    pump_voltage_extractor=_lumped_shunt_varactor_pump_voltage_extractor,
)


register_topology(LUMPED_SHUNT_VARACTOR_DESCRIPTOR)


