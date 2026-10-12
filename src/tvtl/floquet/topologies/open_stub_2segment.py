"""Open-circuit two-segment shunt-stub topology.

Physical structure of one unit cell:

    main microstrip line of width ``main_tl_w`` and period ``main_tl_p``;
    a single shunt branch tapped at the centre of the cell containing, in
    series from the main line outward,

        time-varying RLC (the varactor) -> microstrip(w_s1, l_s1)
        -> microstrip(w_s2, l_s2) -> open circuit.

The harmonic-coupling structure comes entirely from the Floquet Green's
function of the varactor; the two passive stub sub-sections add a
diagonal (per-harmonic) series impedance to the varactor's series-RLC
impedance matrix. The open-stub recursion follows the same machinery used
by :mod:`tvtl.floquet.passband` for the static check.
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


def _open_stub_z_in_per_harmonic(
    segments: tuple[tuple[float, float], ...],
    substrate: Mapping[str, Any],
    fs: float,
    ws: float,
    wm: float,
    N: int,
) -> np.ndarray:
    """Input impedance of an open-terminated stub cascade per harmonic.

    The segments are ordered from the open end inward (i.e. the last entry
    in the cascade tuple has its outer side open, the first entry's outer
    side meets the time-varying element at the top of the stub).
    The recursion at each harmonic angular frequency ``omega_n`` is the
    standard transmission-line input-impedance formula starting from
    ``Z_load = inf`` (open).

    The microstrip Z0 and effective permittivity are evaluated at ``fs``
    (the signal frequency) and reused for every harmonic, matching the
    convention used elsewhere in :mod:`tvtl.floquet` to keep the dispersion
    treatment consistent across the harmonic block.
    """

    truncation = int(N)
    size = 2 * truncation + 1
    omega = np.asarray(
        [float(ws) + float(wm) * (k - truncation) for k in range(size)],
        dtype=float,
    )
    z_in = np.full(size, np.inf + 0.0j, dtype=np.complex128)
    # Walk inward from the open end. Spec order is "from the main line outward
    # = (w_s1, l_s1) then (w_s2, l_s2)". The recursion must start at the open
    # end (segment 2) and accumulate back toward the junction.
    for width_m, length_m in reversed(segments):
        z0, v_phase = _z0_and_v_phase(float(width_m), substrate, float(fs))
        beta = omega / v_phase
        tan_term = np.tan(beta * float(length_m))
        # Limit numerator finite when z_in is infinity at lossless open.
        # Use the algebraic equivalent that handles z_in = inf cleanly:
        #   Z_in_new = Z0 * (Z_load + j Z0 tan) / (Z0 + j Z_load tan)
        #            = Z0 * (1 + j*(Z0/Z_load) tan) / (Z0/Z_load + j tan)
        #            -> Z0 / (j tan)  as Z_load -> inf.
        new_z = np.empty(size, dtype=np.complex128)
        for index in range(size):
            zl = z_in[index]
            if not np.isfinite(zl):
                new_z[index] = z0 / (1j * tan_term[index]) if abs(tan_term[index]) > 1.0e-30 else complex(0.0, 0.0)
                continue
            new_z[index] = (
                z0 * (zl + 1j * z0 * tan_term[index])
                / (z0 + 1j * zl * tan_term[index])
            )
        z_in = new_z
    return z_in


def _shunt_block_from_y(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = Y.shape[0]
    identity = np.eye(size, dtype=np.complex128)
    zero = np.zeros((size, size), dtype=np.complex128)
    return identity, zero, np.asarray(Y, dtype=np.complex128), identity


def build_open_stub_2segment_block_abcd(
    design: Mapping[str, Any],
    varactor_params: Mapping[str, float],
    modulation_params: Mapping[str, float],
    substrate: Mapping[str, Any],
    ws: float,
    wm: float,
    N: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Harmonic-basis ABCD of the open-stub two-segment unit cell.

    Parameters
    ----------
    design : Mapping
        Flat design dict containing the six named variables (see the
        descriptor below for the schema).
    varactor_params, modulation_params, substrate
        Same conventions as for the via-stub-3-segment builder.
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
    stub_w1 = float(design["stub_w_s1"])
    stub_l1 = float(design["stub_l_s1"])
    stub_w2 = float(design["stub_w_s2"])
    stub_l2 = float(design["stub_l_s2"])

    half_p = 0.5 * main_p
    z0_main, v_phase_main = _z0_and_v_phase(main_w, substrate, fs)
    main_block = tl_section_harmonic_abcd(half_p, z0_main, v_phase_main, ws, wm, truncation)

    # Time-varying RLC harmonic admittance (full Floquet coupling).
    Y_var = stub_harmonic_admittance(S0, dS, phi, R, L, ws, wm, truncation)
    # Impedance of the varactor as the inverse of the admittance matrix.
    size = 2 * truncation + 1
    Z_var = np.linalg.solve(Y_var, np.eye(size, dtype=np.complex128))
    # Open-stub input impedance per harmonic (diagonal in the harmonic basis).
    z_open = _open_stub_z_in_per_harmonic(
        segments=((stub_w1, stub_l1), (stub_w2, stub_l2)),
        substrate=substrate,
        fs=fs,
        ws=ws,
        wm=wm,
        N=truncation,
    )
    Z_open_diag = np.diag(z_open)
    Z_branch = Z_var + Z_open_diag
    Y_branch = np.linalg.solve(Z_branch, np.eye(size, dtype=np.complex128))
    stub_block = _shunt_block_from_y(Y_branch)

    return cascade_harmonic_abcd([main_block, stub_block, main_block])


def _open_stub_2segment_period(design: Mapping[str, Any]) -> float:
    return float(design["main_tl_p"])


def _open_stub_2segment_static_check(design: Mapping[str, Any]) -> bool:
    for key in (
        "main_tl_p",
        "main_tl_w",
        "stub_w_s1",
        "stub_l_s1",
        "stub_w_s2",
        "stub_l_s2",
    ):
        if float(design.get(key, 0.0)) <= 0.0:
            return False
    return True


def _open_stub_2segment_pump_voltage_extractor(
    design: Mapping[str, Any],
    varactor_params: Mapping[str, float],
    substrate: Mapping[str, Any],
    n_cells: int,
    pump_frequency_Hz: float,
    pump_amplitude_V: float = 1.0,
    source_z_ohm: float = 50.0,
    load_z_ohm: float = 50.0,
) -> np.ndarray:
    """Per-cell varactor terminal voltage for the open-stub-2-segment topology.

    Static-mode (``dS = 0``) linear analysis at the pump frequency. In this
    topology the varactor sits at the top of the stub (closest to the
    main-line junction); the two passive microstrip sub-sections continue
    to an open termination. The branch is therefore a series chain
    ``Z_var + Z_open`` where ``Z_open`` is the open-stub microstrip
    cascade's input impedance. The varactor terminal voltage is

        V_var = V_junction * Z_var / (Z_var + Z_open).
    """

    from ..pump_analysis import _cascade_junction_voltages, _line_abcd, _z0_and_v_phase

    f_pump = float(pump_frequency_Hz)
    omega = 2.0 * math.pi * f_pump
    C0 = float(varactor_params["C0"])
    R = float(varactor_params["R"])
    L = float(varactor_params["L"])
    z_var = complex(R, omega * L - 1.0 / (omega * C0))

    main_p = float(design["main_tl_p"])
    main_w = float(design["main_tl_w"])
    if main_p <= 0.0 or main_w <= 0.0:
        raise ValueError("open_stub_2segment main TL geometry must be positive.")
    z0_main, v_phase_main = _z0_and_v_phase(main_w, substrate, f_pump)
    beta_main = omega / v_phase_main
    M_main_half = _line_abcd(z0_main, beta_main * 0.5 * main_p)

    # Open-stub input impedance via the standard recursion from the open end.
    z_load: complex = complex("inf")
    for width_m, length_m in (
        (float(design["stub_w_s2"]), float(design["stub_l_s2"])),
        (float(design["stub_w_s1"]), float(design["stub_l_s1"])),
    ):
        z0_s, v_phase_s = _z0_and_v_phase(width_m, substrate, f_pump)
        beta_s = omega / v_phase_s
        tan_term = math.tan(beta_s * length_m)
        if not math.isfinite(abs(z_load)):
            z_load = z0_s / (1j * tan_term) if abs(tan_term) > 1.0e-30 else complex(0.0, 0.0)
        else:
            z_load = z0_s * (z_load + 1j * z0_s * tan_term) / (z0_s + 1j * z_load * tan_term)
    z_open = z_load

    y_branch = 1.0 / (z_var + z_open)
    junctions = _cascade_junction_voltages(
        M_main_half,
        y_branch,
        int(n_cells),
        float(source_z_ohm),
        float(load_z_ohm),
        float(pump_amplitude_V),
    )
    v_var = junctions * z_var / (z_var + z_open)
    return v_var


OPEN_STUB_2SEGMENT_DESCRIPTOR = TopologyDescriptor(
    name="open_stub_2segment",
    version="1.0",
    description=(
        "Open-circuit shunt stub with two microstrip sub-sections following the "
        "time-varying RLC element. The stub branches off the centre of a main "
        "microstrip line; from the junction outward the order is varactor, "
        "microstrip(w_s1, l_s1), microstrip(w_s2, l_s2), open termination."
    ),
    design_variables=(
        DesignVariableSpec("main_tl_p", 4.0e-3, 10.0e-3, "m", "Period of the main microstrip line."),
        DesignVariableSpec("main_tl_w", 2.0e-3, 6.0e-3, "m", "Width of the main microstrip line."),
        DesignVariableSpec("stub_w_s1", 0.5e-3, 3.0e-3, "m", "Width of stub sub-section 1 (next to varactor)."),
        DesignVariableSpec("stub_l_s1", 2.0e-3, 10.0e-3, "m", "Length of stub sub-section 1."),
        DesignVariableSpec("stub_w_s2", 0.5e-3, 3.0e-3, "m", "Width of stub sub-section 2 (next to the open end)."),
        DesignVariableSpec("stub_l_s2", 1.0e-3, 8.0e-3, "m", "Length of stub sub-section 2."),
    ),
    build_unit_cell_block_abcd=build_open_stub_2segment_block_abcd,
    period=_open_stub_2segment_period,
    static_check=_open_stub_2segment_static_check,
    pump_voltage_extractor=_open_stub_2segment_pump_voltage_extractor,
)


register_topology(OPEN_STUB_2SEGMENT_DESCRIPTOR)


