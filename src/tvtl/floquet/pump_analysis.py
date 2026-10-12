"""Pump-delivery analysis at the varactor terminals of each cell.

The Floquet stub solver from earlier sessions applies the modulation
``S(t) = S0 + dS * cos(omega_m * t + phi)`` directly to the time-varying
RLC element without verifying that the *pump* signal applied at the
source port produces enough voltage across the varactor terminals to
realise that modulation. In real devices this verification is essential:
some topologies (notably open-stub variants) provide no low-impedance
return path for the pump current at the varactor and therefore generate
negligible terminal voltage despite strong nominal pump injection.

This module performs the static (``dS = 0``) linear network analysis at
the pump frequency, propagates the source-port pump tone through the
cascade, and reports the per-cell complex varactor terminal voltage.
The diagnosis is purely analytical and proxy in nature; the returned
dicts carry ``"label": "analytical_proxy_only"``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .topology import TopologyDescriptor, get_topology


DEFAULT_MIN_VOLTAGE_FRACTION = 0.1
"""Minimum allowed ratio ``|V_var,min| / pump_amplitude_V`` for a design to
clear the pump-coherence filter at default settings."""

DEFAULT_MAX_PHASE_VARIATION_RAD: float | None = None
"""Maximum allowed end-to-end varactor phase variation across the cascade;
``None`` (the default) disables the phase-variation check entirely."""


def _resolve_topology(topology: Any) -> TopologyDescriptor:
    if isinstance(topology, TopologyDescriptor):
        return topology
    if topology is None:
        raise ValueError("topology argument is required for pump analysis.")
    return get_topology(str(topology))


def compute_pump_voltage_profile(
    design: Mapping[str, Any],
    varactor_params: Mapping[str, float],
    substrate: Mapping[str, Any],
    pump_frequency_Hz: float,
    n_cells: int,
    *,
    topology: Any,
    pump_amplitude_V: float = 1.0,
    source_z_ohm: float = 50.0,
    load_z_ohm: float = 50.0,
) -> dict:
    """Per-cell complex voltage across the varactor at the pump frequency.

    The cascade is analysed in static linear mode (``dS = 0``) using the
    topology's registered :func:`pump_voltage_extractor`. A unit-amplitude
    pump tone is injected at the source port through ``source_z_ohm`` and
    the cascade is terminated by ``load_z_ohm`` at the load port. The
    extractor returns the complex voltage seen across the varactor
    terminals of every unit cell. From that array we report magnitude,
    phase, min/max, the minimum fraction relative to the source amplitude,
    and the cumulative phase variation across the cascade.

    Parameters
    ----------
    design : Mapping
        Topology-specific flat design dict.
    varactor_params : Mapping
        ``"C0"`` [F], ``"R"`` [Ohm], ``"L"`` [H].
    substrate : Mapping
        ``"eps_r"``, ``"h"`` [m] and optional ``"copper_thickness"`` [m].
    pump_frequency_Hz : float
        Pump tone frequency. Must be positive.
    n_cells : int
        Number of identical cells. Must be positive.
    topology : str or TopologyDescriptor
        Topology selecting the extractor.
    pump_amplitude_V : float
        Open-circuit source-port voltage at the pump frequency.
    source_z_ohm, load_z_ohm : float
        Reference impedances at the cascade source and load ports.

    Returns
    -------
    dict
        Profile dictionary with keys ``"cell_indices"``,
        ``"varactor_voltages_complex"``, ``"voltage_magnitudes_V"``,
        ``"voltage_phases_rad"`` (unwrapped), ``"voltage_min_V"``,
        ``"voltage_max_V"``, ``"voltage_min_fraction"``,
        ``"phase_variation_rad"``, ``"phase_delay_per_cell_rad"``,
        ``"pump_amplitude_at_input_V"`` (a float reproducing
        ``pump_amplitude_V`` for callers that want to scale the profile),
        and ``"label"``.
    """

    if int(n_cells) <= 0:
        raise ValueError("n_cells must be positive.")
    if float(pump_frequency_Hz) <= 0.0:
        raise ValueError("pump_frequency_Hz must be positive.")
    descriptor = _resolve_topology(topology)
    extractor = descriptor.pump_voltage_extractor
    if extractor is None:
        raise RuntimeError(
            f"Topology '{descriptor.name}' has no pump_voltage_extractor; cannot run pump analysis."
        )
    voltages = np.asarray(
        extractor(
            design,
            varactor_params,
            substrate,
            int(n_cells),
            float(pump_frequency_Hz),
            pump_amplitude_V=float(pump_amplitude_V),
            source_z_ohm=float(source_z_ohm),
            load_z_ohm=float(load_z_ohm),
        ),
        dtype=np.complex128,
    )
    if voltages.shape != (int(n_cells),):
        raise RuntimeError(
            f"Topology '{descriptor.name}' extractor returned shape {voltages.shape} for n_cells={n_cells}."
        )
    if not np.all(np.isfinite(voltages)):
        raise RuntimeError(
            f"Topology '{descriptor.name}' extractor produced non-finite voltages."
        )

    magnitudes = np.abs(voltages)
    phases = np.unwrap(np.angle(voltages))
    voltage_min = float(np.min(magnitudes))
    voltage_max = float(np.max(magnitudes))
    pump_amp = max(float(pump_amplitude_V), np.finfo(float).tiny)
    voltage_min_fraction = float(voltage_min / pump_amp)
    if int(n_cells) >= 2:
        phase_variation = float(phases[-1] - phases[0])
        increments = np.diff(phases)
        phase_delay_per_cell = float(np.median(increments))
    else:
        phase_variation = 0.0
        phase_delay_per_cell = 0.0

    return {
        "cell_indices": np.arange(int(n_cells), dtype=int),
        "varactor_voltages_complex": voltages,
        "voltage_magnitudes_V": magnitudes,
        "voltage_phases_rad": phases,
        "voltage_min_V": voltage_min,
        "voltage_max_V": voltage_max,
        "voltage_min_fraction": voltage_min_fraction,
        "phase_variation_rad": phase_variation,
        "phase_delay_per_cell_rad": phase_delay_per_cell,
        "pump_amplitude_at_input_V": float(pump_amplitude_V),
        "label": "analytical_proxy_only",
    }


def pump_power_profile(
    profile: Mapping[str, Any],
    varactor_params: Mapping[str, float],
    pump_frequency_Hz: float,
    source_z_ohm: float = 50.0,
) -> dict:
    """Per-cell pump power dissipated in the varactor's series resistance.

    For each cell ``k`` the dissipated power is
    ``P_k = 0.5 * |I_k|^2 * R`` where ``I_k = V_var,k / Z_varactor`` and
    ``Z_varactor = R + j*omega_m*L - j/(omega_m*C0)``. The pump input
    power is ``0.5 * |V_in|^2 / Z_source`` (used here as a passive
    sanity-check upper bound).

    Returns a dict with cell-indexed dissipation, totals, the assumed
    pump-input power, an "efficiency" ratio, and the proxy label.
    """

    voltages = np.asarray(profile["varactor_voltages_complex"], dtype=np.complex128)
    omega = 2.0 * math.pi * float(pump_frequency_Hz)
    C0 = float(varactor_params["C0"])
    R = float(varactor_params["R"])
    L = float(varactor_params["L"])
    if C0 <= 0.0 or R < 0.0 or L < 0.0:
        raise ValueError("varactor_params must satisfy C0 > 0 and R, L >= 0.")
    z_var = complex(R, omega * L - 1.0 / (omega * C0))
    currents = voltages / z_var
    powers = 0.5 * (np.abs(currents) ** 2) * R
    total = float(np.sum(powers))
    pump_amplitude = float(profile.get("pump_amplitude_at_input_V", 1.0))
    pump_input_power = 0.5 * (pump_amplitude ** 2) / float(source_z_ohm)
    efficiency = total / max(pump_input_power, np.finfo(float).tiny)
    return {
        "cell_indices": np.arange(voltages.shape[0], dtype=int),
        "pump_power_dissipated_W": powers,
        "total_power_dissipated_W": total,
        "pump_input_power_W": float(pump_input_power),
        "efficiency": float(efficiency),
        "label": "analytical_proxy_only",
    }


def pump_coherence_check(
    profile: Mapping[str, Any],
    min_voltage_fraction: float = DEFAULT_MIN_VOLTAGE_FRACTION,
    max_phase_variation_rad: float | None = DEFAULT_MAX_PHASE_VARIATION_RAD,
) -> dict:
    """Pass/fail check that the pump reaches every varactor.

    The check passes iff the minimum varactor terminal voltage across all
    cells is at least ``min_voltage_fraction`` of the source amplitude AND
    (if ``max_phase_variation_rad`` is not ``None``) the end-to-end phase
    variation is no more than that bound. ``failing_cells`` lists cell
    indices where the varactor voltage magnitude falls below the
    threshold.
    """

    voltages = np.asarray(profile["varactor_voltages_complex"], dtype=np.complex128)
    magnitudes = np.abs(voltages)
    pump_amplitude = float(profile.get("pump_amplitude_at_input_V", 1.0))
    threshold_V = float(min_voltage_fraction) * pump_amplitude
    failing_cells = [int(i) for i in np.where(magnitudes < threshold_V)[0]]
    voltage_pass = not failing_cells
    phase_variation = float(profile.get("phase_variation_rad", 0.0))
    if max_phase_variation_rad is None:
        phase_pass = True
    else:
        phase_pass = abs(phase_variation) <= float(max_phase_variation_rad)
    return {
        "pass": bool(voltage_pass and phase_pass),
        "voltage_min_fraction": float(profile.get("voltage_min_fraction", 0.0)),
        "min_voltage_fraction_required": float(min_voltage_fraction),
        "phase_variation_rad": phase_variation,
        "max_phase_variation_rad_required": (
            None if max_phase_variation_rad is None else float(max_phase_variation_rad)
        ),
        "failing_cells": failing_cells,
        "label": "analytical_proxy_only",
    }


def _z0_and_v_phase(width_m: float, substrate: Mapping[str, Any], frequency_hz: float) -> tuple[float, float]:
    """Hammerstad-Kirschning Z0 and effective phase velocity at a frequency."""

    from tvtl.materials import SPEED_OF_LIGHT_M_PER_S
    from tvtl.microstrip.models import hammerstad_kirschning

    eps_r = float(substrate["eps_r"])
    h = float(substrate["h"])
    copper_thickness = float(substrate.get("copper_thickness", 17.5e-6))
    z0_arr, eps_eff_arr = hammerstad_kirschning(
        float(width_m), copper_thickness, h, eps_r, np.asarray([float(frequency_hz)], dtype=float)
    )
    eps_eff = float(eps_eff_arr[0])
    v_phase = SPEED_OF_LIGHT_M_PER_S / math.sqrt(max(eps_eff, np.finfo(float).tiny))
    return float(z0_arr[0]), v_phase


def _line_abcd(z0: float, beta_l: float) -> np.ndarray:
    """Lossless transmission-line ABCD at electrical length ``beta_l``."""

    cosh = math.cos(beta_l)
    sinh = math.sin(beta_l)
    return np.asarray(
        [[cosh, 1j * z0 * sinh], [1j * sinh / float(z0), cosh]],
        dtype=np.complex128,
    )


def _shunt_abcd(y_shunt: complex) -> np.ndarray:
    """ABCD of a shunt admittance to ground."""

    return np.asarray([[1.0, 0.0], [complex(y_shunt), 1.0]], dtype=np.complex128)


def _cascade_junction_voltages(
    M_main_half: np.ndarray,
    Y_branch: complex,
    n_cells: int,
    source_z_ohm: float,
    load_z_ohm: float,
    pump_amplitude_V: float,
) -> np.ndarray:
    """Return the complex voltage at the centre-of-cell junction for each cell.

    A unit cell is modelled as ``M_main_half * M_shunt(Y_branch) * M_main_half``.
    The source generates an open-circuit voltage ``pump_amplitude_V`` through
    a series impedance ``source_z_ohm``; the load is ``load_z_ohm``. The
    cascade is solved as a single ABCD product and back-propagated cell by
    cell to read off the junction voltages.

    Returns
    -------
    numpy.ndarray
        Complex array of shape ``(n_cells,)`` ordered from input cell to
        output cell.
    """

    M_shunt = _shunt_abcd(Y_branch)
    M_cell = M_main_half @ M_shunt @ M_main_half

    M_total = np.linalg.matrix_power(M_cell, int(n_cells))
    A = complex(M_total[0, 0])
    B = complex(M_total[0, 1])
    C = complex(M_total[1, 0])
    D = complex(M_total[1, 1])
    z_s = complex(source_z_ohm)
    z_l = complex(load_z_ohm)
    denom = A + B / z_l + z_s * C + z_s * D / z_l
    v_n = complex(pump_amplitude_V) / denom
    i_n = v_n / z_l
    state_at_load = np.asarray([v_n, i_n], dtype=np.complex128)

    junctions = np.empty(int(n_cells), dtype=np.complex128)
    state_after_k_cells = state_at_load
    # Walk from the last cell back to the first, propagating state by M_cell
    # each step. After ``k`` propagations the state is the input of cell
    # ``n_cells - 1 - k``.
    for offset in range(int(n_cells)):
        # state_after_k_cells is at the right edge of cell (n_cells-1-offset),
        # which equals (V_out_k, I_out_k) for that cell. Propagate through the
        # second half-MLIN to land on (V_junction, *).
        v_junction = M_main_half[0, 0] * state_after_k_cells[0] + M_main_half[0, 1] * state_after_k_cells[1]
        cell_index = int(n_cells) - 1 - offset
        junctions[cell_index] = v_junction
        state_after_k_cells = M_cell @ state_after_k_cells
    return junctions


