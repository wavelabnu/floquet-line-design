"""Generic Floquet unit-cell model contracts.

This module exposes:

* The dataclasses ``FloquetBranchModel`` and ``FloquetUnitCellModel`` for
  generic unit-cell screening (unchanged in this revision).
* The high-level helper :func:`floquet_uc_cascade_response`, which composes
  the block-ABCD primitives in :mod:`tvtl.floquet.harmonic_abcd` into a
  device-level cascade and returns the harmonic ``(S11, S21)`` together
  with three analytical proxy-gain numbers. Every returned dict carries
  ``"label": "analytical_proxy_only"`` so the proxy disclaimer propagates
  into any downstream artifact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from tvtl.design.candidates import UnitCellGraph
from tvtl.materials import SPEED_OF_LIGHT_M_PER_S
from tvtl.microstrip.models import hammerstad_kirschning

from .harmonic_abcd import (
    cascade_harmonic_abcd,
    device_proxy_gain,
    harmonic_frequencies,
    harmonic_s_parameters,
    n_cells_harmonic_abcd,
    stub_harmonic_admittance,
    tl_section_harmonic_abcd,
    unit_cell_harmonic_abcd,
)
from .topology import (
    DesignVariableSpec,
    TopologyDescriptor,
    get_topology,
    register_topology,
)


@dataclass(frozen=True, slots=True)
class FloquetBranchModel:
    branch_id: str
    modulation_frequency_hz: float
    harmonic_order: int = 1
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.modulation_frequency_hz <= 0.0:
            raise ValueError("modulation_frequency_hz must be positive.")
        if self.harmonic_order < 0:
            raise ValueError("harmonic_order must be non-negative.")


@dataclass(frozen=True, slots=True)
class FloquetUnitCellModel:
    unit_cell: UnitCellGraph
    branch_models: tuple[FloquetBranchModel, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def branch_ids(self) -> tuple[str, ...]:
        return tuple(model.branch_id for model in self.branch_models)


def _microstrip_z0_and_eps_eff(
    width_m: float,
    substrate: Mapping[str, Any],
    frequency_hz: float,
) -> tuple[float, float]:
    """Evaluate ``Z0`` and ``eps_eff`` at the given frequency for a strip."""

    eps_r = float(substrate["eps_r"])
    h = float(substrate["h"])
    copper_thickness = float(substrate.get("copper_thickness", 17.5e-6))
    z0_arr, eps_eff_arr = hammerstad_kirschning(
        float(width_m), copper_thickness, h, eps_r, np.asarray([float(frequency_hz)], dtype=float)
    )
    return float(z0_arr[0]), float(eps_eff_arr[0])


def _stub_microstrip_blocks(
    stub_segments: Iterable[Mapping[str, Any]] | None,
    substrate: Mapping[str, Any],
    fs: float,
    ws: float,
    wm: float,
    N: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cascade of passive microstrip sub-sections in the shunt stub branch.

    The blocks are diagonal in the harmonic basis; the returned tuple is the
    cascade in the order ``junction -> ... -> top-of-TV-RLC``. When
    ``stub_segments`` is empty (or ``None``), the cascade is the block identity
    so the branch reduces directly to the TV-RLC terminated by the via short.
    """

    segments = list(stub_segments) if stub_segments is not None else []
    size = 2 * int(N) + 1
    identity = np.eye(size, dtype=np.complex128)
    zero = np.zeros((size, size), dtype=np.complex128)
    blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for segment in segments:
        z0, eps_eff = _microstrip_z0_and_eps_eff(float(segment["width"]), substrate, fs)
        v_phase = SPEED_OF_LIGHT_M_PER_S / math.sqrt(max(eps_eff, np.finfo(float).tiny))
        blocks.append(
            tl_section_harmonic_abcd(
                float(segment["length"]), z0, v_phase, ws, wm, N
            )
        )
    if not blocks:
        return identity, zero, zero, identity
    return cascade_harmonic_abcd(blocks)


def _stub_branch_admittance(
    Y_rlc: np.ndarray,
    micro_block: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Admittance seen from the junction looking into the shunt branch.

    The branch is the microstrip-cascade two-port (block-diagonal in
    harmonics) terminated by the 1-port admittance ``Y_rlc``. The standard
    two-port-terminated-by-load formula is

        Y_branch = (C + D * Y_rlc) @ inv(A + B * Y_rlc).
    """

    a, b, c, d = micro_block
    denom = a + b @ Y_rlc
    numerator = c + d @ Y_rlc
    return numerator @ np.linalg.solve(denom, np.eye(denom.shape[0], dtype=np.complex128))


def _shunt_block_from_y(
    Y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Block ABCD for a pure shunt branch with admittance ``Y``."""

    size = Y.shape[0]
    identity = np.eye(size, dtype=np.complex128)
    zero = np.zeros((size, size), dtype=np.complex128)
    return identity, zero, np.asarray(Y, dtype=np.complex128), identity


def _resolve_main_segments(uc_geometry: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], int]:
    segments = list(uc_geometry.get("main_segments", []))
    if not segments:
        raise ValueError("uc_geometry must define a non-empty 'main_segments' list.")
    stub_position = int(uc_geometry.get("stub_position", 0))
    if stub_position < 0 or stub_position >= len(segments):
        raise ValueError(
            f"stub_position {stub_position} must satisfy 0 <= stub_position < len(main_segments)."
        )
    return segments, stub_position


def _parse_modulation_params(
    modulation_params: Mapping[str, float],
    varactor_params: Mapping[str, float],
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """Return ``(ws, wm, fs, fm, phi, S0, dS, R, L)`` after validation."""

    fs = float(modulation_params["fs"])
    fm = float(modulation_params["fm"])
    dS_fraction = float(modulation_params["dS_fraction"])
    phi = float(modulation_params["phi"])
    if fs <= 0.0 or fm <= 0.0:
        raise ValueError("Both fs and fm must be positive frequencies.")
    if not (0.0 <= dS_fraction <= 1.0):
        raise ValueError("dS_fraction must lie in [0, 1].")

    C0 = float(varactor_params["C0"])
    R = float(varactor_params["R"])
    L = float(varactor_params["L"])
    if C0 <= 0.0 or R < 0.0 or L < 0.0:
        raise ValueError("varactor_params must satisfy C0 > 0 and R, L >= 0.")
    S0 = 1.0 / C0
    dS = dS_fraction * S0

    ws = 2.0 * math.pi * fs
    wm = 2.0 * math.pi * fm
    return ws, wm, fs, fm, phi, S0, dS, R, L


def build_unit_cell_block_abcd(
    uc_geometry: Mapping[str, Any],
    varactor_params: Mapping[str, float],
    modulation_params: Mapping[str, float],
    N_harmonics: int,
    substrate: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the harmonic-basis ``(A, B, C, D)`` for ONE TVTL unit cell.

    This is the building block used by
    :func:`floquet_uc_cascade_response` and by the parametric-instability
    eigenvalue analysis in :mod:`tvtl.floquet.stability`. The unit cell is
    composed exactly as in :func:`floquet_uc_cascade_response`: a sequence of
    series microstrip segments with a shunt branch tapped after one of them.
    See that function for the meaning of every input field.

    Returns
    -------
    tuple of numpy.ndarray
        ``(A, B, C, D)``, each of shape ``(2*N_harmonics + 1, 2*N_harmonics + 1)``.
    """

    ws, wm, _fs, _fm, _phi, _S0, _dS, _R, _L = _parse_modulation_params(modulation_params, varactor_params)
    N = int(N_harmonics)
    if N < 1:
        raise ValueError("N_harmonics must be at least 1 to access the idler harmonic.")
    # Delegate to the registered via_stub_3segment builder so the legacy
    # ``uc_geometry`` entry point and the topology-dispatched entry point
    # share a single implementation and therefore stay bit-exact under any
    # future refactor.
    return build_via_stub_3segment_block_abcd(
        uc_geometry, varactor_params, modulation_params, substrate, ws, wm, N
    )


def unit_cell_period_m(uc_geometry: Mapping[str, Any]) -> float:
    """Total physical period of one unit cell along the main line.

    The period is the sum of the ``main_segments`` lengths. The shunt branch
    runs perpendicular to the main line, so its lengths do not enter the
    period. Useful for converting per-cell eigenvalues to per-metre Floquet
    exponents.
    """

    main_segments, _ = _resolve_main_segments(uc_geometry)
    return float(sum(float(segment["length"]) for segment in main_segments))


# --------------------------------------------------------------------------- #
# Topology registration: via_stub_3segment (paper-style three-segment microstrip
# shunt stub terminated by a via short, with the time-varying RLC at the top of
# the stub closest to the main-line junction and three passive microstrip
# sub-sections cascading from the varactor's bottom terminal toward the via).
# --------------------------------------------------------------------------- #


_VIA_STUB_3SEGMENT_LEGACY_KEYS = ("main_segments", "stub_segments")


def _is_legacy_uc_geometry(design: Mapping[str, Any]) -> bool:
    """Return True if ``design`` is in the historical ``uc_geometry`` schema."""

    return any(key in design for key in _VIA_STUB_3SEGMENT_LEGACY_KEYS)


def _via_stub_3segment_geometry_from_design(design: Mapping[str, Any]) -> dict[str, Any]:
    """Translate a flat ``via_stub_3segment`` design dict to ``uc_geometry``."""

    if _is_legacy_uc_geometry(design):
        return {
            "main_segments": [dict(seg) for seg in design["main_segments"]],
            "stub_position": int(design.get("stub_position", 0)),
            "stub_segments": [dict(seg) for seg in design.get("stub_segments", [])],
        }
    main_length = float(design["main_segment_length"])
    main_width = float(design["main_width"])
    return {
        "main_segments": [
            {"length": main_length, "width": main_width},
            {"length": main_length, "width": main_width},
        ],
        "stub_position": 0,
        "stub_segments": [
            {"length": float(design["stub_segment_1_length"]), "width": float(design["stub_segment_1_width"])},
            {"length": float(design["stub_segment_2_length"]), "width": float(design["stub_segment_2_width"])},
            {"length": float(design["stub_segment_3_length"]), "width": float(design["stub_segment_3_width"])},
        ],
    }


def build_via_stub_3segment_block_abcd(
    design: Mapping[str, Any],
    varactor_params: Mapping[str, float],
    modulation_params: Mapping[str, float],
    substrate: Mapping[str, Any],
    ws: float,
    wm: float,
    N: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Harmonic-basis ABCD of the paper-style 3-segment via-stub unit cell.

    The cell is two main microstrip sections flanking a shunt branch whose
    layout from the main-line junction outward is

        main-line junction
            -> microstrip(w_s1, l_s1)
            -> microstrip(w_s2, l_s2)
            -> microstrip(w_s3, l_s3)
            -> series time-varying RLC
            -> via to ground

    ``design`` may be either the topology's flat schema (named variables
    ``main_segment_length``, ``main_width``, ``stub_segment_{1,2,3}_length``,
    ``stub_segment_{1,2,3}_width``) or the legacy ``uc_geometry`` dict with
    ``main_segments`` / ``stub_segments`` / ``stub_position`` keys. ``ws``,
    ``wm`` are angular signal/modulation frequencies and ``N`` is the
    truncation order.
    """

    if int(N) < 1:
        raise ValueError("N must be at least 1 to access the idler harmonic.")
    uc_geometry = _via_stub_3segment_geometry_from_design(design)
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

    main_segments, stub_position = _resolve_main_segments(uc_geometry)
    main_blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for segment in main_segments:
        z0, eps_eff = _microstrip_z0_and_eps_eff(float(segment["width"]), substrate, fs)
        v_phase = SPEED_OF_LIGHT_M_PER_S / math.sqrt(max(eps_eff, np.finfo(float).tiny))
        main_blocks.append(
            tl_section_harmonic_abcd(float(segment["length"]), z0, v_phase, ws, wm, int(N))
        )
    # Varactor at the top of the stub (closest to the main-line junction);
    # the three passive microstrip sub-sections cascade from the varactor's
    # bottom terminal toward the via short. The harmonic-coupled Z_var is
    # added in series with the diagonal per-harmonic Z_micro_short input
    # impedance of the microstrip cascade terminated by the via.
    Y_rlc = stub_harmonic_admittance(S0, dS, phi, R, L, ws, wm, int(N))
    micro_block = _stub_microstrip_blocks(uc_geometry.get("stub_segments"), substrate, fs, ws, wm, int(N))
    size = 2 * int(N) + 1
    A_m, B_m, _C_m, D_m = micro_block
    d_diag = np.diag(D_m)
    # Z_micro_short is diagonal because every harmonic propagates independently
    # through the passive cascade; ``B/D`` is the input impedance with the far
    # end shorted. When there are no stub segments the cascade is the identity
    # 2-port and ``B = 0``, so ``Z_micro_short`` is zero and the branch reduces
    # to the bare RLC (the same behaviour as the previous implementation).
    z_micro_short_diag = np.where(
        np.abs(d_diag) > 0.0,
        np.diag(B_m) / np.where(np.abs(d_diag) > 0.0, d_diag, 1.0),
        np.zeros_like(d_diag),
    )
    Z_micro_short = np.diag(z_micro_short_diag)
    Z_var = np.linalg.solve(Y_rlc, np.eye(size, dtype=np.complex128))
    Z_branch_total = Z_var + Z_micro_short
    Y_branch = np.linalg.solve(Z_branch_total, np.eye(size, dtype=np.complex128))
    stub_block = _shunt_block_from_y(Y_branch)
    sequence: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for index, block in enumerate(main_blocks):
        sequence.append(block)
        if index == stub_position:
            sequence.append(stub_block)
    return cascade_harmonic_abcd(sequence)


def _via_stub_3segment_period(design: Mapping[str, Any]) -> float:
    """Unit-cell period (sum of the two main-line segment lengths) [m]."""

    uc_geometry = _via_stub_3segment_geometry_from_design(design)
    return unit_cell_period_m(uc_geometry)


def _via_stub_3segment_static_check(design: Mapping[str, Any]) -> bool:
    """Reject obviously non-physical designs (non-positive geometry)."""

    uc_geometry = _via_stub_3segment_geometry_from_design(design)
    for segment in uc_geometry["main_segments"]:
        if float(segment["length"]) <= 0.0 or float(segment["width"]) <= 0.0:
            return False
    for segment in uc_geometry["stub_segments"]:
        if float(segment["length"]) <= 0.0 or float(segment["width"]) <= 0.0:
            return False
    return True


def _via_stub_3segment_pump_voltage_extractor(
    design: Mapping[str, Any],
    varactor_params: Mapping[str, float],
    substrate: Mapping[str, Any],
    n_cells: int,
    pump_frequency_Hz: float,
    pump_amplitude_V: float = 1.0,
    source_z_ohm: float = 50.0,
    load_z_ohm: float = 50.0,
) -> np.ndarray:
    """Per-cell varactor terminal voltage for the via-stub-3-segment topology.

    Static-mode (``dS = 0``) linear analysis at the pump frequency. The
    cascade is built from the single-frequency ABCD of one unit cell,
    cascaded ``n_cells`` times between the source and load. The voltage at
    each cell's centre junction is computed by back-propagating from the
    load, and the voltage across the varactor terminals is then derived
    from the topology's branch transformer.

    In this topology the varactor element sits at the top of the stub
    closest to the main-line junction; the three microstrip sub-sections
    cascade from the varactor's bottom terminal down to the via short.
    The branch from the main-line junction is therefore a series
    composition ``Z_var + Z_micro_short`` where ``Z_micro_short`` is the
    input impedance of the microstrip cascade with its far end shorted to
    ground by the via. The varactor terminal voltage is

        V_var = V_junction * Z_var / (Z_var + Z_micro_short).
    """

    from .pump_analysis import _cascade_junction_voltages, _line_abcd, _z0_and_v_phase

    uc_geometry = _via_stub_3segment_geometry_from_design(design)
    f_pump = float(pump_frequency_Hz)
    omega = 2.0 * math.pi * f_pump
    C0 = float(varactor_params["C0"])
    R = float(varactor_params["R"])
    L = float(varactor_params["L"])
    z_rlc = complex(R, omega * L - 1.0 / (omega * C0))
    y_rlc = 1.0 / z_rlc

    # Main microstrip half-cell ABCD (per cell layout uses two equal main segments).
    main_segments = uc_geometry["main_segments"]
    stub_position = int(uc_geometry.get("stub_position", 0))
    segments_equal = len(main_segments) == 2 and main_segments[0] == main_segments[1]
    if not segments_equal or stub_position != 0:
        raise ValueError(
            "via_stub_3segment pump-voltage extractor models a symmetric cell of two "
            "equal main segments with the shunt branch at stub_position 0; got "
            f"{len(main_segments)} main segment(s) (equal={segments_equal}), "
            f"stub_position={stub_position}. compute_pump_voltage_profile does not "
            "model this asymmetric legacy geometry -- pass a symmetric via_stub_3segment "
            "cell or extend the extractor."
        )
    z0_main, v_phase_main = _z0_and_v_phase(float(main_segments[0]["width"]), substrate, f_pump)
    beta_main = omega / v_phase_main
    main_half_length = float(main_segments[0]["length"])
    if main_half_length <= 0.0:
        raise ValueError("via_stub_3segment main segment length must be positive.")
    M_main_half = _line_abcd(z0_main, beta_main * main_half_length)

    # Microstrip stub-cascade ABCD at the pump frequency. The cascade
    # terminates in a via short, so its input impedance (looking down from
    # the varactor's bottom terminal) is ``B / D``.
    micro_abcd = np.eye(2, dtype=np.complex128)
    for segment in uc_geometry.get("stub_segments", []):
        z0_s, v_phase_s = _z0_and_v_phase(float(segment["width"]), substrate, f_pump)
        beta_s = omega / v_phase_s
        micro_abcd = micro_abcd @ _line_abcd(z0_s, beta_s * float(segment["length"]))
    b_micro = complex(micro_abcd[0, 1])
    d_micro = complex(micro_abcd[1, 1])
    if abs(d_micro) <= 0.0:
        z_micro_short = complex(0.0, 0.0)
    else:
        z_micro_short = b_micro / d_micro

    # Branch admittance at the junction: varactor in series with the shorted
    # microstrip cascade.
    y_branch = 1.0 / (z_rlc + z_micro_short)

    junctions = _cascade_junction_voltages(
        M_main_half,
        y_branch,
        int(n_cells),
        float(source_z_ohm),
        float(load_z_ohm),
        float(pump_amplitude_V),
    )
    v_var = junctions * z_rlc / (z_rlc + z_micro_short)
    return v_var


VIA_STUB_3SEGMENT_DESCRIPTOR = TopologyDescriptor(
    name="via_stub_3segment",
    version="1.0",
    description=(
        "Paper-style three-segment microstrip shunt stub terminated by a via short. "
        "The time-varying series RLC sits at the top of the stub closest to the "
        "main-line junction; three passive microstrip sub-sections cascade from the "
        "varactor's bottom terminal down to the via short."
    ),
    design_variables=(
        DesignVariableSpec("main_segment_length", 1.0e-3, 5.0e-3, "m", "Length of each main-line series segment (period/2)."),
        DesignVariableSpec("main_width", 2.0e-3, 5.0e-3, "m", "Width of the main microstrip line."),
        DesignVariableSpec("stub_segment_1_length", 2.0e-3, 10.0e-3, "m", "Length of stub sub-section 1 (closest to the junction)."),
        DesignVariableSpec("stub_segment_1_width", 1.0e-3, 3.0e-3, "m", "Width of stub sub-section 1."),
        DesignVariableSpec("stub_segment_2_length", 1.0e-3, 8.0e-3, "m", "Length of stub sub-section 2."),
        DesignVariableSpec("stub_segment_2_width", 1.0e-3, 2.5e-3, "m", "Width of stub sub-section 2."),
        DesignVariableSpec("stub_segment_3_length", 0.5e-3, 4.0e-3, "m", "Length of stub sub-section 3 (closest to TV-RLC)."),
        DesignVariableSpec("stub_segment_3_width", 1.0e-3, 3.0e-3, "m", "Width of stub sub-section 3."),
    ),
    build_unit_cell_block_abcd=build_via_stub_3segment_block_abcd,
    period=_via_stub_3segment_period,
    static_check=_via_stub_3segment_static_check,
    pump_voltage_extractor=_via_stub_3segment_pump_voltage_extractor,
)

register_topology(VIA_STUB_3SEGMENT_DESCRIPTOR)


def floquet_uc_cascade_response(
    uc_geometry: Mapping[str, Any],
    varactor_params: Mapping[str, float],
    modulation_params: Mapping[str, float],
    n_cells: int,
    N_harmonics: int,
    substrate: Mapping[str, Any],
    z_source: float = 50.0,
    z_load: float = 50.0,
    topology: str | TopologyDescriptor | None = None,
) -> dict:
    """Full Floquet cascade response of a TVTL.

    The unit cell is modelled as a sequence of series microstrip segments with
    a shunt branch tapped after one of them. The shunt branch consists of an
    optional cascade of passive microstrip sub-sections (block-diagonal in the
    harmonic basis) terminated by the time-varying series RLC, whose bottom
    terminal is shorted to ground (via). The harmonic ABCD blocks are cascaded
    along the unit cell, raised to ``n_cells``, and converted to block
    S-parameters with scalar source/load impedances.

    All gain numbers returned by this function are analytical proxies only and
    carry the label ``"analytical_proxy_only"``. They are NOT a validated
    device-gain claim and must never be reported as one.

    Parameters
    ----------
    uc_geometry : Mapping
        Required keys:

        * ``"main_segments"`` -- list of ``{"length": [m], "width": [m]}``
          dicts describing the per-cell series sections in order.
        * ``"stub_position"`` -- integer ``i`` such that the shunt branch
          taps after ``main_segments[i]``.

        Optional key:

        * ``"stub_segments"`` -- list of ``{"length": [m], "width": [m]}``
          microstrip sub-sections in the shunt branch, ordered from the
          junction toward the top of the TV-RLC element.
    varactor_params : Mapping
        Keys ``"C0"`` [F], ``"R"`` [Ohm], ``"L"`` [H].
    modulation_params : Mapping
        Keys ``"fs"`` [Hz], ``"fm"`` [Hz], ``"dS_fraction"`` (in ``[0, 1]``),
        ``"phi"`` [rad].
    n_cells : int
        Number of identical unit cells in the cascade. Non-negative.
    N_harmonics : int
        Floquet truncation order; the harmonic axis has length ``2*N + 1``.
    substrate : Mapping
        Keys ``"eps_r"`` and ``"h"`` [m]; optional ``"copper_thickness"`` and
        ``"tan_delta"`` (the latter is accepted but the present model uses
        the lossless TL form in the cascade).
    z_source, z_load : float
        Scalar reference impedances [Ohm].

    Returns
    -------
    dict
        Keys:

        * ``"frequencies"`` -- ndarray of harmonic frequencies in Hz.
        * ``"S11"``, ``"S21"`` -- complex ndarrays of shape ``(2N+1, 2N+1)``.
        * ``"gain_signal_proxy_dB"``, ``"gain_idler_proxy_dB"``,
          ``"gain_total_proxy_dB"`` -- floats.
        * ``"label"`` -- ``"analytical_proxy_only"``.
    """

    ws, wm, _fs, _fm, _phi, _S0, _dS, _R, _L = _parse_modulation_params(modulation_params, varactor_params)
    N = int(N_harmonics)
    if N < 1:
        raise ValueError("N_harmonics must be at least 1 to access the idler harmonic.")

    if topology is None:
        unit_cell_block = build_unit_cell_block_abcd(
            uc_geometry, varactor_params, modulation_params, N_harmonics, substrate
        )
    else:
        descriptor = topology if isinstance(topology, TopologyDescriptor) else get_topology(str(topology))
        unit_cell_block = descriptor.build_unit_cell_block_abcd(
            uc_geometry, varactor_params, modulation_params, substrate, ws, wm, N
        )
    total_block = n_cells_harmonic_abcd(unit_cell_block, int(n_cells))

    s11, s21 = harmonic_s_parameters(*total_block, z_source, z_load)
    proxy = device_proxy_gain(s21, N)

    omega_axis = harmonic_frequencies(ws, wm, N)
    return {
        "frequencies": omega_axis / (2.0 * math.pi),
        "S11": s11,
        "S21": s21,
        "gain_signal_proxy_dB": proxy["gain_signal_proxy_dB"],
        "gain_idler_proxy_dB": proxy["gain_idler_proxy_dB"],
        "gain_total_proxy_dB": proxy["gain_total_proxy_dB"],
        "label": "analytical_proxy_only",
    }


