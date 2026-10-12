"""Single-frequency passband utilities for TVTL design screening.

Used by the cascade-Floquet optimization engine to verify that a candidate
unit cell admits propagating waves at the signal, pump, and sideband
frequencies of the operating point. A design that fails this check has a
stop-band at one of those frequencies and any predicted parametric gain
there is artifactual.

All numbers returned by this module reflect passive small-signal cascade
behaviour at ``dS = 0`` and are analytical proxies only.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from tvtl.materials import SPEED_OF_LIGHT_M_PER_S
from tvtl.microstrip.models import hammerstad_kirschning
from tvtl.theory.abcd import line_abcd, shunt_admittance_abcd
from tvtl.theory.cascade import cascade_abcd
from tvtl.theory.sparams import abcd_to_sparameters

from .topology import TopologyDescriptor, get_topology


def _z0_and_v_phase(width_m: float, substrate: Mapping[str, Any], frequency_hz: float) -> tuple[float, float]:
    eps_r = float(substrate["eps_r"])
    h = float(substrate["h"])
    copper_thickness = float(substrate.get("copper_thickness", 17.5e-6))
    z0_arr, eps_eff_arr = hammerstad_kirschning(
        float(width_m),
        copper_thickness,
        h,
        eps_r,
        np.asarray([float(frequency_hz)], dtype=float),
    )
    eps_eff = float(eps_eff_arr[0])
    v_phase = SPEED_OF_LIGHT_M_PER_S / math.sqrt(max(eps_eff, np.finfo(float).tiny))
    return float(z0_arr[0]), v_phase


def _static_branch_y(
    varactor_params: Mapping[str, float],
    substrate: Mapping[str, Any],
    stub_segments: list[Mapping[str, Any]] | None,
    frequency_hz: float,
) -> complex:
    """Admittance looking into the shunt branch at a single frequency.

    Branch layout (matches :func:`floquet_uc_cascade_response`):
    ``junction -> [stub microstrip cascade] -> [series RLC] -> via short``.
    The series RLC contributes ``Z_rlc(omega) = R + j*omega*L + 1/(j*omega*C0)``.
    The terminating short is modelled as an ideal ``V = 0`` at the bottom of
    the RLC.
    """

    omega = 2.0 * math.pi * float(frequency_hz)
    C0 = float(varactor_params["C0"])
    R = float(varactor_params["R"])
    L = float(varactor_params["L"])
    z_rlc = complex(R, omega * L - 1.0 / (omega * C0))
    y_rlc = 1.0 / z_rlc

    segments = list(stub_segments) if stub_segments is not None else []
    if not segments:
        return y_rlc

    # Each stub microstrip section is block-diagonal in harmonics, so at a
    # single frequency it is just a 2x2 line ABCD. Cascade them, then load
    # by ``y_rlc`` to recover the input admittance at the junction.
    matrices = []
    for segment in segments:
        z0, v_phase = _z0_and_v_phase(float(segment["width"]), substrate, frequency_hz)
        beta = omega / v_phase
        matrices.append(line_abcd(z0, 1j * beta * float(segment["length"])))
    micro = cascade_abcd(matrices)
    a, b = complex(micro[0, 0]), complex(micro[0, 1])
    c, d = complex(micro[1, 0]), complex(micro[1, 1])
    return (c + d * y_rlc) / (a + b * y_rlc)


def _topology_static_s21_dB(
    topology: TopologyDescriptor,
    design: Mapping[str, Any],
    varactor_params: Mapping[str, float],
    substrate: Mapping[str, Any],
    n_cells: int,
    frequency_Hz: float,
) -> float:
    """Topology-agnostic passive cascade ``|S21|`` at a single frequency.

    Uses the topology's harmonic-basis builder with ``dS_fraction = 0`` and
    ``N = 1`` and reads the central (n = 0) harmonic block to obtain the
    per-cell single-frequency 2-by-2 ABCD. The cascade is then raised to
    ``n_cells`` and converted to ``|S21|`` in dB.
    """

    import math as _math

    frequency = float(frequency_Hz)
    if frequency <= 0.0:
        raise ValueError("frequency_Hz must be positive.")
    cells = int(n_cells)
    if cells < 0:
        raise ValueError("n_cells must be non-negative.")
    ws = 2.0 * _math.pi * frequency
    # fm is irrelevant when dS = 0 (no harmonic coupling); pick 2*ws so the
    # harmonic axis is well-formed and the topology builder is happy.
    wm = 2.0 * ws
    modulation_params = {
        "fs": frequency,
        "fm": 2.0 * frequency,
        "dS_fraction": 0.0,
        "phi": 0.0,
    }
    block = topology.build_unit_cell_block_abcd(
        design, varactor_params, modulation_params, substrate, ws, wm, 1
    )
    centre = 1  # index of n = 0 harmonic in the (2*1 + 1)-length axis
    a = complex(block[0][centre, centre])
    b = complex(block[1][centre, centre])
    c = complex(block[2][centre, centre])
    d = complex(block[3][centre, centre])
    unit_cell = np.asarray([[a, b], [c, d]], dtype=np.complex128)
    if cells == 0:
        total = np.eye(2, dtype=np.complex128)
    else:
        total = np.linalg.matrix_power(unit_cell, cells)
    sparams = abcd_to_sparameters(total, reference_impedance_ohm=50.0)
    magnitude = abs(complex(sparams.s21))
    tiny = np.finfo(float).tiny
    return float(20.0 * _math.log10(max(magnitude, tiny)))


def static_s21_at_frequency(
    uc_geometry: Mapping[str, Any],
    varactor_params: Mapping[str, float],
    substrate: Mapping[str, Any],
    n_cells: int,
    frequency_Hz: float,
    topology: str | TopologyDescriptor | None = None,
) -> float:
    """Passive (``dS = 0``) cascade ``|S21|`` in dB at a single frequency.

    The unit cell is built from the same fields as
    :func:`tvtl.floquet.uc_model.floquet_uc_cascade_response`: a sequence of
    main microstrip segments with a shunt branch tapped after segment
    ``stub_position``. The shunt branch is evaluated at ``dS = 0``, i.e.
    a static series-RLC with no parametric coupling.

    Parameters
    ----------
    uc_geometry : Mapping
        ``"main_segments"``, ``"stub_position"`` and optional
        ``"stub_segments"``; see ``floquet_uc_cascade_response`` for the
        exact schema.
    varactor_params : Mapping
        ``"C0"`` [F], ``"R"`` [Ohm], ``"L"`` [H].
    substrate : Mapping
        ``"eps_r"``, ``"h"`` [m] and optional ``"copper_thickness"`` [m].
    n_cells : int
        Number of identical cells in the cascade. Non-negative.
    frequency_Hz : float
        Evaluation frequency.

    Returns
    -------
    float
        ``20 * log10(|S21|)`` in dB.
    """

    if float(frequency_Hz) <= 0.0:
        raise ValueError("frequency_Hz must be positive.")
    if topology is not None:
        descriptor = topology if isinstance(topology, TopologyDescriptor) else get_topology(str(topology))
        return _topology_static_s21_dB(descriptor, uc_geometry, varactor_params, substrate, n_cells, frequency_Hz)
    main_segments = list(uc_geometry.get("main_segments", []))
    if not main_segments:
        raise ValueError("uc_geometry must define a non-empty 'main_segments' list.")
    stub_position = int(uc_geometry.get("stub_position", 0))
    if stub_position < 0 or stub_position >= len(main_segments):
        raise ValueError("stub_position must be a valid index into main_segments.")

    omega = 2.0 * math.pi * float(frequency_Hz)
    line_matrices: list = []
    for segment in main_segments:
        z0, v_phase = _z0_and_v_phase(float(segment["width"]), substrate, frequency_Hz)
        beta = omega / v_phase
        line_matrices.append(line_abcd(z0, 1j * beta * float(segment["length"])))
    y_branch = _static_branch_y(
        varactor_params, substrate, uc_geometry.get("stub_segments"), frequency_Hz
    )
    shunt = shunt_admittance_abcd(y_branch)

    cascade_inputs: list = []
    for index, line in enumerate(line_matrices):
        cascade_inputs.append(line)
        if index == stub_position:
            cascade_inputs.append(shunt)
    unit_cell = cascade_abcd(cascade_inputs)

    cells = int(n_cells)
    if cells < 0:
        raise ValueError("n_cells must be non-negative.")
    if cells == 0:
        total = np.eye(2, dtype=np.complex128)
    else:
        total = np.linalg.matrix_power(unit_cell, cells)
    sparams = abcd_to_sparameters(total, reference_impedance_ohm=50.0)
    magnitude = abs(complex(sparams.s21))
    tiny = np.finfo(float).tiny
    return float(20.0 * math.log10(max(magnitude, tiny)))


def passband_check_at_operating_point(
    uc_geometry: Mapping[str, Any],
    varactor_params: Mapping[str, float],
    substrate: Mapping[str, Any],
    n_cells: int,
    fs_Hz: float,
    fm_Hz: float,
    threshold_dB: float = -10.0,
    topology: str | TopologyDescriptor | None = None,
) -> dict:
    """Verify that signal, pump, and idler sidebands are inside the passband.

    The four frequencies are::

        f_signal      = fs_Hz
        f_pump        = fm_Hz
        f_idler_lower = fs_Hz - fm_Hz   (skipped if <= 0)
        f_idler_upper = fs_Hz + fm_Hz

    Each is checked against the passive ``|S21|`` cascade. A frequency
    passes if its dB value is at or above ``threshold_dB``. The overall
    check passes if every evaluated frequency passes.

    Parameters
    ----------
    uc_geometry, varactor_params, substrate, n_cells
        See :func:`static_s21_at_frequency`.
    fs_Hz, fm_Hz : float
        Signal and pump frequencies. Both must be positive.
    threshold_dB : float
        Minimum allowed ``|S21|`` in dB for each evaluated frequency.

    Returns
    -------
    dict
        Keys ``"pass"`` (bool), ``"s21_dB_by_freq"`` (mapping label -> dB),
        ``"failing_frequencies"`` (list of ``(label, dB)`` tuples below the
        threshold), and ``"threshold_dB"`` (float).
    """

    if float(fs_Hz) <= 0.0 or float(fm_Hz) <= 0.0:
        raise ValueError("fs_Hz and fm_Hz must be positive.")
    labels_and_freqs: list[tuple[str, float]] = [
        ("signal", float(fs_Hz)),
        ("pump", float(fm_Hz)),
        ("idler_upper", float(fs_Hz) + float(fm_Hz)),
    ]
    lower = float(fs_Hz) - float(fm_Hz)
    if lower > 0.0:
        labels_and_freqs.append(("idler_lower", lower))

    s21_by_freq: dict[str, float] = {}
    failing: list[tuple[str, float]] = []
    for label, frequency in labels_and_freqs:
        magnitude_dB = static_s21_at_frequency(
            uc_geometry, varactor_params, substrate, n_cells, frequency, topology=topology,
        )
        s21_by_freq[label] = float(magnitude_dB)
        if magnitude_dB < float(threshold_dB):
            failing.append((label, float(magnitude_dB)))
    return {
        "pass": not failing,
        "s21_dB_by_freq": s21_by_freq,
        "failing_frequencies": failing,
        "threshold_dB": float(threshold_dB),
    }


