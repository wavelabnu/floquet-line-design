"""Analytical S-parameter computation for a Level-10 paper-geometry cascade.

This is the Path-B analytical leg for the Figure 4 reproduction.
For each frequency in a sweep the helper computes the cascade's ABCD
matrix by composing per-cell sub-blocks:

* the through-line MLIN halves (``LINE_LEFT`` and ``LINE_RIGHT``) via the
  v2 framework's Hammerstad-Kirschning ``Z_0(f)`` / ``eps_eff(f)`` and
  the legacy ``calc_gamma_legacy`` propagation constant (which includes
  conductor and dielectric losses analytically);
* the shunt branch impedance ``Z_branch(f) = R_pkg + j·ω·L_pkg + 1/(j·ω·C_eff) + Z_stub(ω)``
  where ``C_eff`` is the SDD2P static capacitance ``q[1]`` (with the
  CP-fold baked in) and ``Z_stub(ω)`` is the input impedance of the
  multi-segment stub MLIN cascade with the Bahl via-inductance load
  (the lifted ``Y_stub_at_freq`` machinery, with the via L
  auto-extracted from the spec).

The per-cell ABCD is

    ABCD_cell = ABCD_left_half · shunt_admittance_ABCD(1/Z_branch) · ABCD_right_half

and the cascade ABCD is ``ABCD_cell ^ N`` (matrix power; the cells are
identical in the paper geometry). Conversion to S-parameters uses the
existing :func:`tvtl.theory.sparams.abcd_to_sparameters` helper at the
spec's port impedance.

The result is exposed as :class:`Level10SParamResult`; the helper
:func:`level10_paper_sparams` is the entry point. The matching ADS
S_Param simulation lives in
:func:`tvtl_ads_builder_v2.analytical.level10_sparams.run_ads_sparams`
(loaded lazily because it requires the ADS runtime), and the comparison
between the two is the Figure 4 dry-run's deliverable.

Linearisation caveat
--------------------
The SDD2P is a nonlinear charge-driven device. ADS's S_Param controller
small-signal-linearises it around the DC operating point (``v = 0`` for
Level-10), giving an effective static capacitance equal to ``q[1]`` (the
Q(v) polynomial's first derivative at v=0). The analytical ABCD uses
the same ``C_eff = q[1]`` so the two paths are apples-to-apples on the
linearised passive structure. They do NOT reproduce HB intermod current
(that is the Floquet-comparison harness's job, exercised separately).
The agreement between analytical ABCD and ADS S_Param is therefore the
test of how well the passive distributed structure (main line + stub +
via + package parasitics + linearised varactor) is captured by both
paths -- a quantitative check of the v2 microstrip, ABCD and via
models against ADS's own microstrip + lumped engines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from tvtl.materials import SPEED_OF_LIGHT_M_PER_S, Substrate as TvtlSubstrate
from tvtl.microstrip.models import calc_gamma_legacy, hammerstad_kirschning
from tvtl.theory.abcd import line_abcd, shunt_admittance_abcd
from tvtl.theory.sparams import abcd_to_sparameters

from ..device_spec import DeviceSpec
from .floquet_stub import build_stub_y_fn_from_spec, extract_stub_segments
from .mtee_blocks import tee_reference_plane_single


__all__ = [
    "Level10SParamResult",
    "level10_paper_sparams",
]


@dataclass(frozen=True)
class Level10SParamResult:
    """S-parameter sweep result for a Level-10 paper-geometry cascade.

    All arrays are 1-D and have the same length as ``frequencies_hz``.
    """

    frequencies_hz: np.ndarray
    s11: np.ndarray
    s21: np.ndarray
    s12: np.ndarray
    s22: np.ndarray
    reference_impedance_ohm: float

    @property
    def s11_db(self) -> np.ndarray:
        """``|S_11|`` in dB, clipped at -200 dB to keep log plots well-formed."""
        mag = np.abs(self.s11)
        with np.errstate(divide="ignore"):
            return np.where(
                mag > 1.0e-10, 20.0 * np.log10(mag), -200.0,
            )

    @property
    def s21_db(self) -> np.ndarray:
        mag = np.abs(self.s21)
        with np.errstate(divide="ignore"):
            return np.where(
                mag > 1.0e-10, 20.0 * np.log10(mag), -200.0,
            )


def _tvtl_substrate_from_spec(spec: DeviceSpec) -> TvtlSubstrate:
    """Build a :class:`tvtl.materials.Substrate` from the spec's substrate."""

    params = spec.substrate.parameters
    return TvtlSubstrate(
        epsilon_r=float(params["eps_r"]),
        thickness_m=float(params["h_m"]),
        copper_thickness_m=float(params.get("t_m", 17.5e-6)),
        loss_tangent=float(params.get("tan_delta", 0.0)),
        conductivity_s_per_m=float(params.get("sigma_S_per_m", 5.8e7)),
    )


def _line_abcd_lossy(
    width_m: float, length_m: float,
    frequency_hz: float, substrate: TvtlSubstrate,
) -> np.ndarray:
    """Lossy MLIN ABCD via Hammerstad-Kirschning Z_0 + calc_gamma_legacy."""

    freqs = np.asarray([frequency_hz], dtype=float)
    z0_arr, eps_eff_arr = hammerstad_kirschning(
        float(width_m), substrate.copper_thickness_m,
        substrate.thickness_m, substrate.epsilon_r, freqs,
    )
    z0 = float(z0_arr[0])
    eps_eff = float(eps_eff_arr[0])
    gamma = complex(
        calc_gamma_legacy(freqs, z0_arr, eps_eff_arr, substrate)[0]
    )
    theta = gamma * float(length_m)
    return line_abcd(z0, theta)


def _cell_through_geometry(
    spec: DeviceSpec, cell_index: int,
) -> tuple[float, float]:
    """Return the per-cell main-line ``(w_m, half_length_m)`` for cell k."""

    k = cell_index + 1  # element ids are 1-indexed
    left = next(e for e in spec.elements if e.id == f"LINE_LEFT_C{k}")
    width = float(left.parameters["w_m"])
    half_length = float(left.parameters["l_m"])
    return width, half_length


def _local_branch_impedance(
    spec: DeviceSpec, cell_index: int, omega: float,
) -> complex:
    """Local series impedance R + jωL + 1/(jωC) at the cell's branch.

    Reads the cell's ``LPKG_C{k}`` inductance, ``RPKG_C{k}`` resistance,
    and the SDD2P's ``q_poly_coefficients[1]`` (= ``C_eff`` after the
    CP fold). The local impedance does NOT include the stub MLIN
    cascade or the via inductance -- those are folded in separately
    via :func:`build_stub_y_fn_from_spec`.
    """

    k = cell_index + 1
    lpkg = next(e for e in spec.elements if e.id == f"LPKG_C{k}")
    rpkg = next(e for e in spec.elements if e.id == f"RPKG_C{k}")
    sdd = next(
        e for e in spec.elements
        if e.type == "SDD2P" and e.id == f"VARACTOR_C{k}"
    )
    l_pkg = float(lpkg.parameters["l_h"])
    r_pkg = float(rpkg.parameters["r_ohm"])
    c_eff = float(sdd.parameters["q_poly_coefficients"][1])
    if c_eff <= 0.0:
        raise ValueError(
            f"cell {cell_index}: SDD2P q[1] (effective C) must be positive; "
            f"got {c_eff}."
        )
    return complex(
        r_pkg + 1j * float(omega) * l_pkg
        + 1.0 / (1j * float(omega) * c_eff)
    )


def _cell_abcd(
    spec: DeviceSpec, cell_index: int, omega: float, substrate: TvtlSubstrate,
    *, use_mtee_parasitic: bool = False,
) -> np.ndarray:
    """Per-cell ABCD: half-line · shunt-Y(branch) · half-line.

    With ``use_mtee_parasitic=True`` the ideal-junction shunt is replaced by
    the reference-plane assembly ``line(half - l_a) · tee.abcd_left ·
    shunt(Y_branch + j*bt) · tee.abcd_right · line(half - l_b)``, with a
    fall-back to the ideal junction where the corrected-notebook model is past
    its high-frequency validity edge (see :mod:`.mtee_blocks`).
    """

    width, half_length = _cell_through_geometry(spec, cell_index)
    frequency_hz = float(omega) / (2.0 * math.pi)
    # Branch impedance: local R+L+C in series with the stub-with-via Z_in.
    z_local = _local_branch_impedance(spec, cell_index, omega)
    y_stub_fn = build_stub_y_fn_from_spec(spec, cell_index)
    y_stub = complex(y_stub_fn(omega))
    if abs(y_stub) < 1.0e-30:
        z_stub = complex(float("inf"))
    else:
        z_stub = complex(1.0 / y_stub)
    z_branch = z_local + z_stub
    if abs(z_branch) < 1.0e-30:
        y_branch = complex(float("inf"))
    else:
        y_branch = complex(1.0 / z_branch)

    if not use_mtee_parasitic:
        half_abcd = _line_abcd_lossy(width, half_length, frequency_hz, substrate)
        shunt = shunt_admittance_abcd(y_branch)
        return np.asarray(half_abcd @ shunt @ half_abcd, dtype=np.complex128)

    tee = next(
        e for e in spec.elements
        if e.type == "MTEE" and e.id == f"TEE_C{cell_index + 1}"
    )
    abcd_left, abcd_right, j_bt, l_a, l_b, valid = tee_reference_plane_single(
        width, float(tee.parameters["w3_m"]), frequency_hz, substrate,
    )
    if not valid:
        abcd_left = np.eye(2, dtype=np.complex128)
        abcd_right = np.eye(2, dtype=np.complex128)
        j_bt = 0j
        l_a = l_b = 0.0
    left_line = _line_abcd_lossy(width, half_length - l_a, frequency_hz, substrate)
    right_line = _line_abcd_lossy(width, half_length - l_b, frequency_hz, substrate)
    shunt = shunt_admittance_abcd(y_branch + j_bt)
    return np.asarray(
        left_line @ abcd_left @ shunt @ abcd_right @ right_line,
        dtype=np.complex128,
    )


def level10_paper_sparams(
    spec: DeviceSpec,
    frequencies_hz: Sequence[float] | np.ndarray,
    *,
    reference_impedance_ohm: float | None = None,
    use_mtee_parasitic: bool = False,
) -> Level10SParamResult:
    """Compute the cascade S-parameters analytically across a frequency sweep.

    Parameters
    ----------
    spec : DeviceSpec
        A Level-10 paper-geometry-shaped spec (from
        :func:`tvtl_ads_builder_v2.parity.canonical_circuits.make_paper_geometry_tvtl_cascade`
        or any caller of :func:`make_tvtl_level10_cascade`).
    frequencies_hz : sequence of float
        The sweep grid in Hz. The default Level-10 spec sweeps 2-17 GHz
        × 151 points to match the publication's Figure 4 grid; this
        helper accepts any positive grid.
    reference_impedance_ohm : float, optional
        Port reference impedance for the S-parameter conversion.
        Defaults to ``spec.analysis.port_impedance_ohm`` so the
        analytical and the ADS S_Param paths sit on the same reference.
    use_mtee_parasitic : bool, optional
        When ``False`` (default) the MTEE is the ideal junction (the
        validated Figure-4 baseline). When ``True`` the reference-plane
        :func:`tvtl.microstrip.discontinuities.tee_junction` is inserted per
        cell (with the high-frequency ideal-junction fallback). The Level-10
        paper-fidelity canonical keeps this ``False``; the flag is provided
        for regression-context comparison of the upgraded MTEE.

    Returns
    -------
    Level10SParamResult
        ``s11`` / ``s12`` / ``s21`` / ``s22`` arrays in linear units,
        plus the original frequency grid. Use the ``.s11_db`` and
        ``.s21_db`` properties for log-scale comparison plots.
    """

    freqs = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    if freqs.size == 0:
        raise ValueError("frequencies_hz must contain at least one entry.")
    if np.any(freqs <= 0.0):
        raise ValueError("frequencies_hz entries must be positive.")
    sdds = [e for e in spec.elements if e.type == "SDD2P"]
    if not sdds:
        raise ValueError(
            "spec carries no SDD2P elements -- not a Level-10 paper-geometry "
            "cascade."
        )
    n_cells = len(sdds)
    substrate = _tvtl_substrate_from_spec(spec)
    z_ref = float(
        reference_impedance_ohm
        if reference_impedance_ohm is not None
        else spec.analysis.port_impedance_ohm
    )

    s11 = np.zeros(freqs.size, dtype=np.complex128)
    s12 = np.zeros(freqs.size, dtype=np.complex128)
    s21 = np.zeros(freqs.size, dtype=np.complex128)
    s22 = np.zeros(freqs.size, dtype=np.complex128)

    for i, f in enumerate(freqs):
        omega = 2.0 * math.pi * float(f)
        # The paper geometry has identical cells, so build one ABCD and
        # raise it to the N-th matrix power. The generic generator
        # allows non-uniform cascades; we detect uniformity by id
        # convention and fall back to per-cell composition otherwise.
        abcd_first = _cell_abcd(
            spec, 0, omega, substrate, use_mtee_parasitic=use_mtee_parasitic,
        )
        all_uniform = True
        if n_cells > 1:
            for k in range(1, n_cells):
                abcd_k = _cell_abcd(
                    spec, k, omega, substrate,
                    use_mtee_parasitic=use_mtee_parasitic,
                )
                if not np.allclose(abcd_k, abcd_first, rtol=1e-12, atol=1e-30):
                    all_uniform = False
                    break
        if all_uniform:
            abcd_cascade = np.linalg.matrix_power(abcd_first, n_cells)
        else:
            abcd_cascade = abcd_first
            for k in range(1, n_cells):
                abcd_cascade = abcd_cascade @ _cell_abcd(
                    spec, k, omega, substrate,
                    use_mtee_parasitic=use_mtee_parasitic,
                )
        sp = abcd_to_sparameters(
            abcd_cascade, reference_impedance_ohm=z_ref,
        )
        s11[i] = sp.s11
        s12[i] = sp.s12
        s21[i] = sp.s21
        s22[i] = sp.s22

    return Level10SParamResult(
        frequencies_hz=freqs.astype(float),
        s11=s11, s21=s21, s12=s12, s22=s22,
        reference_impedance_ohm=z_ref,
    )


