"""Analytical parametric gain of a realistic-varactor stub-loaded cascade.

The analytical (TVTL) leg of the realistic-varactor cross-validation.
It is a **strict generalisation** of the ideal-C leg
(:func:`.ideal_c_gain.level10_ideal_c_param_gain`): the shunt branch is the
capacitance-linear time-varying junction (now with C-V *curvature* bands) in
series with the package ``L_pkg + R_series`` (above the junction) and the
stub-MLIN + via network (below it). When the supplied
:class:`tvtl.varactors.RealisticVaractor` is
:meth:`~tvtl.varactors.RealisticVaractor.ideal` (no parasitics, linear C-V)
this leg reproduces the ideal-C leg **bit-for-bit** (the ideal-C reduction
contract):

* the effective junction depth equals the branch-top depth (package divider
  magnitude 1, pinned exactly), so the capacitance Toeplitz is the same
  single-tone cosine the ideal-C leg builds (verified equal to the existing
  :func:`tvtl.floquet.harmonic_abcd.tv_capacitor_harmonic_admittance`);
* the package wrap is **skipped** (the ``L_pkg == R_series == 0`` branch
  returns the bare admittance directly, not ``inv(inv(...))``), so every
  downstream matrix op is identical arithmetic on identical inputs.

The geometry (through-line halves, MTEE, stub, via, frequencies, cell count,
port impedance) is read from the supplied ``spec`` exactly as the ideal-C leg
reads it -- the leg accepts either a Level10RealisticC spec or any Level-10
shaped spec. The static ``C0`` and the C-V come from the ``varactor``, not the
spec; the branch-top pump depth is passed explicitly (``m_branch_pump``), so
the leg is self-contained (no HB dataset) like the ideal-C leg. (How the
caller obtains ``m_branch_pump`` -- prescribed, or read from an ADS branch-top
pump for cross-validation -- is the caller's concern; a first-principles
pump-from-power solve is intentionally outside this leg.)

Results carry ``label="analytical_proxy_only"`` -- the analytical proxy for
the parametric-gain mechanism, the cross-validation counterpart to the ADS-HB
device gain, not a device-gain claim.

The per-cell pumped harmonic ABCD that this gain leg cascades is also exposed
directly via :func:`realistic_pumped_unit_cell_abcd` -- both share the single
cell-construction source of truth :func:`_realistic_cell_builder`, so the
Floquet/Bloch multipliers of the returned cell are the multipliers of the
*same* cell the gain leg uses (the analytical stability-margin path in
:mod:`tvtl.floquet.stability`).
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from tvtl.floquet.harmonic_abcd import (
    cascade_harmonic_abcd,
    device_coherent_gain,
    device_proxy_gain,
    harmonic_frequencies,
    harmonic_s_parameters,
    tv_capacitor_harmonic_admittance_general,
)
from tvtl.floquet.harmonics import build_stub_tridiagonal
from tvtl.varactors import RealisticVaractor

from ..device_spec import DeviceSpec
from .floquet_stub import build_stub_y_fn_from_spec, extract_stub_segments
from .ideal_c_gain import _var
from .mtee_blocks import tee_harmonic_blocks
from .parametric_gain import _lossy_tl_harmonic_block, _tvtl_substrate_from_spec

__all__ = ["level10_realistic_param_gain", "realistic_pumped_unit_cell_abcd"]


def _n_cells_sdd2p(spec: DeviceSpec) -> int:
    """Number of SDD2P shunt branches (cell count) in the spec.

    Generic over the SDD2P mode -- counts both the charge-driven
    (``q_poly_coefficients``) realistic varactor and the control-port
    (``tvc_c0_f``) ideal-C device, so the realistic leg accepts either spec
    shape (a Level10RealisticC charge cascade or a Level10IdealC tvc cascade
    for the bit-for-bit regression).
    """

    n = sum(1 for e in spec.elements if e.type == "SDD2P")
    if n < 1:
        raise ValueError("spec has no SDD2P shunt branches (not a cascade).")
    return n


_MODULATION_FORMULATIONS = ("capacitance", "elastance", "reciprocal")


def _elastance_toeplitz_from_capacitance(c_coeffs, size: int) -> np.ndarray:
    """Harmonic Toeplitz of ``S(t) = 1/C(t)`` -- the EXACT reciprocal of the
    same ``C(t)`` given by the one-sided capacitance spectrum ``c_coeffs``.

    ``C(t) = sum_d C_d e^{i d wm t}`` (``C_{-d} = conj(C_d)``) is reconstructed
    and inverted; ``S(t) = 1/C(t)`` is FFT'd to its harmonic bands and laid out
    as ``matrix[n, m] = S_{n-m}`` -- the identical band-placement convention as
    :func:`tv_capacitor_harmonic_admittance_general`, so the elastance-basis and
    capacitance-basis solvers see the same device by reciprocal routes (agreement
    is a numerical solver check, bounded by the truncation of the finite basis).
    """

    coeffs = np.asarray(c_coeffs, dtype=np.complex128)
    sample_count = 8192
    theta = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False)
    waveform = np.full(sample_count, complex(coeffs[0]), dtype=np.complex128)
    for d in range(1, coeffs.shape[0]):
        waveform = (
            waveform
            + coeffs[d] * np.exp(1j * d * theta)
            + np.conj(coeffs[d]) * np.exp(-1j * d * theta)
        )
    if np.any(waveform.real <= 0.0):
        raise ValueError("reciprocal formulation requires C(t) > 0 over the period.")
    s_fft = np.fft.fft(1.0 / waveform) / sample_count
    matrix = np.zeros((size, size), dtype=np.complex128)
    for row in range(size):
        for col in range(size):
            matrix[row, col] = s_fft[(row - col) % sample_count]
    return matrix


def _resolve_drive(modulation_formulation, m_branch_pump, control_port_vpump_v, m_s_branch):
    """Validate the formulation + depth kwargs; return ``(control_port, drive)``.

    Depth semantics: ``m_branch_pump`` (dC/C0) or ``control_port_vpump_v`` (Voc)
    for the ``"capacitance"``/``"reciprocal"`` formulations; ``m_s_branch``
    (dS/S0) for ``"elastance"``. Exactly one drive appropriate to the
    formulation is accepted; anything else is a clear ``ValueError``.
    """

    if modulation_formulation not in _MODULATION_FORMULATIONS:
        raise ValueError(
            f"modulation_formulation must be one of {_MODULATION_FORMULATIONS}; "
            f"got {modulation_formulation!r}."
        )
    if modulation_formulation == "elastance":
        if (
            m_s_branch is None
            or m_branch_pump is not None
            or control_port_vpump_v is not None
        ):
            raise ValueError(
                "modulation_formulation='elastance' requires exactly m_s_branch "
                "(dS/S0); m_branch_pump / control_port_vpump_v must be None."
            )
        return False, float(m_s_branch)
    if m_s_branch is not None:
        raise ValueError(
            "m_s_branch is only valid with modulation_formulation='elastance'."
        )
    if (m_branch_pump is None) == (control_port_vpump_v is None):
        raise ValueError(
            "provide exactly one of m_branch_pump (self-pumped) or "
            "control_port_vpump_v (control-port)."
        )
    control_port = control_port_vpump_v is not None
    return control_port, float(control_port_vpump_v if control_port else m_branch_pump)


def _realistic_cell_builder(
    spec: DeviceSpec,
    varactor: RealisticVaractor,
    *,
    control_port: bool,
    truncation_N: int,
    use_mtee_parasitic: bool,
    tee_model_variant,
    via_model,
    open_end_model,
    stub_y_fn_override,
    modulation_formulation: str = "capacitance",
):
    """Single source of truth for building ONE realistic-varactor pumped cell.

    Returns ``(make_cell, n, n_cells, z_ref)`` where ``make_cell(drive_value)``
    returns ``(cell_abcd_tuple, capacitance_coeffs)`` -- the harmonic-basis
    ``(A, B, C, D)`` of one pumped unit cell at the given drive, plus the
    capacitance Fourier coefficients used to build it. Every operation here is
    the exact arithmetic the historical inline construction performed (in the
    same order), so :func:`level10_realistic_param_gain` stays bit-for-bit and
    :func:`realistic_pumped_unit_cell_abcd` returns the identical cell the gain
    leg cascades.
    """

    n = int(truncation_N)
    z_ref = float(spec.analysis.port_impedance_ohm)
    n_cells = _n_cells_sdd2p(spec)

    f_signal = _var(spec, "f_signal")
    f_pump = _var(spec, "f_pump")
    try:
        phi = _var(spec, "pump_phase_rad")
    except KeyError:
        phi = 0.0
    ws = 2.0 * math.pi * f_signal
    wm = 2.0 * math.pi * f_pump
    omega_n = harmonic_frequencies(ws, wm, n)

    substrate = _tvtl_substrate_from_spec(spec)
    left = next(e for e in spec.elements if e.id == "LINE_LEFT_C1")
    w_m = float(left.parameters["w_m"])
    l_m = float(left.parameters["l_m"])

    # COMBINED-termination tracing. The spec-extraction stub
    # tracer (build_stub_y_fn_from_spec / extract_stub_segments) walks a single
    # series chain from the varactor to either a ground node (SHORT) or an open
    # dead-end (OPEN); it cannot trace the COMBINED parallel via||open-stublet
    # trap, where two branches hang off the stub end node. The caller therefore
    # supplies the COMBINED cell's analytical shunt admittance directly via
    # ``stub_y_fn_override`` (omega -> Y_stub), built from the SAME
    # ``tvtl.optimization.topology_generator.topology_y_stub_fn`` the Stage 1-3
    # screen already uses for COMBINED -- so the analytical leg traces the
    # identical parallel termination end-to-end. SHORT/OPEN leave it None.
    if stub_y_fn_override is not None:
        y_stub = stub_y_fn_override
    else:
        # A COMBINED spec emits per-cell OSTUB_* open stublets; without an override
        # the spec-extraction tracer would silently mis-trace the parallel
        # termination as a single open/short branch. Fail loudly instead of
        # returning a wrong gain.
        if any(str(e.id).startswith("OSTUB_") for e in spec.elements):
            raise ValueError(
                "COMBINED-termination spec (per-cell OSTUB_* open stublets present) "
                "cannot be traced by the single-chain spec-extraction stub tracer; "
                "pass stub_y_fn_override=topology_y_stub_fn(topology, params, "
                "substrate, via_radius_m=...) so the walker traces the parallel "
                "via||open-stublet admittance."
            )
        # Model-registry path: when a via / open-end model is supplied, the
        # stub far-end termination is taken from the model (the same params the emitter
        # rendered) instead of re-extracted from the spec. The default lumped_L / ideal
        # models (and via_model=open_end_model=None) leave termination_override=None, so
        # the byte-identical spec-extraction path runs unchanged.
        termination_override = None
        if via_model is not None or open_end_model is not None:
            segs_term, _sub_dict, default_term = extract_stub_segments(spec, 0)
            if default_term == "short" and via_model is not None:
                termination_override = via_model.analytical_termination(substrate=substrate)
            elif default_term == "open" and open_end_model is not None and segs_term:
                termination_override = open_end_model.analytical_termination(
                    width_m=float(segs_term[-1].width_m),
                    substrate=substrate, f_ref_Hz=float(f_signal),
                )
        y_stub = build_stub_y_fn_from_spec(
            spec, 0, termination_override=termination_override,
        )
    z_stub_fn = lambda omega: 1.0 / y_stub(omega)

    # MTEE_ADS_SURROGATE_V1: detect the variant by its string
    # value (lazy-import-friendly; avoids requiring a config-enum import
    # at module load time for callers that don't use this variant).
    _surrogate_active = (
        tee_model_variant is not None
        and str(getattr(tee_model_variant, "value", tee_model_variant))
            == "mtee_ads_surrogate_v1"
    )
    s3_per_harm_for_surrogate: np.ndarray | None = None

    if use_mtee_parasitic:
        tee = next(e for e in spec.elements if e.type == "MTEE" and e.id == "TEE_C1")
        w3_m = float(tee.parameters["w3_m"])
        if _surrogate_active:
            # MTEE_ADS surrogate path: per-harmonic 3-port S from
            # the tabular surrogate; the cascade math combines the tee +
            # branch network into a single matrix-form 2-port ABCD (full
            # harmonic coupling preserved) inside make_cell below.
            # Full line lengths are used here -- the surrogate captures the
            # reference-plane physics inside its 3-port S, so no l_a/l_b
            # shortening of the through-line is applied (unlike the
            # CORRECTED_NOTEBOOK path).
            from .mtee_ads_surrogate import mtee_ads_3port_smatrix_harmonic
            s3_per_harm_for_surrogate = mtee_ads_3port_smatrix_harmonic(
                w_m, w3_m, omega_n,
            )
            tl_left = _lossy_tl_harmonic_block(w_m, l_m, omega_n, substrate)
            tl_right = _lossy_tl_harmonic_block(w_m, l_m, omega_n, substrate)
            block_left = block_right = j_bt_vec = None
        else:
            # Default-preserving: when tee_model_variant is None, the
            # tee_harmonic_blocks default (CORRECTED_NOTEBOOK) is used and
            # the call site stays bit-for-bit backward-compatible. Callers
            # may pass TeeModelVariant.IDEAL_SYMMETRIC_LOSSLESS to
            # align with the MTEE_ADS flat-deck emitter.
            _tee_kwargs = (
                {} if tee_model_variant is None
                else {"tee_model_variant": tee_model_variant}
            )
            block_left, block_right, j_bt_vec, l_a, l_b = tee_harmonic_blocks(
                w_m, w3_m, omega_n, substrate, **_tee_kwargs,
            )
            tl_left = _lossy_tl_harmonic_block(w_m, l_m - l_a, omega_n, substrate)
            tl_right = _lossy_tl_harmonic_block(w_m, l_m - l_b, omega_n, substrate)
    else:
        tl_left = tl_right = _lossy_tl_harmonic_block(w_m, l_m, omega_n, substrate)
        block_left = block_right = None
        j_bt_vec = None

    # Package series impedance per harmonic (diagonal -- L/R are linear,
    # time-invariant, so they couple no harmonics). Above the junction.
    has_pkg = float(varactor.L_pkg_h) != 0.0 or float(varactor.R_series_ohm) != 0.0
    z_pkg = float(varactor.R_series_ohm) + 1j * omega_n * float(varactor.L_pkg_h)

    size = 2 * n + 1
    identity = np.eye(size, dtype=np.complex128)
    zero = np.zeros((size, size), dtype=np.complex128)

    def _coeffs(drive_value: float) -> list[complex]:
        if control_port:
            return varactor.capacitance_coeffs_from_junction_vpump(
                drive_value, phi=phi,
            )
        return varactor.effective_capacitance_coeffs(
            f_pump, drive_value, phi=phi,
        )

    static_c0 = (
        float(_coeffs(0.0)[0].real)
        if modulation_formulation != "capacitance"
        else 0.0
    )

    def _element_admittance(drive_value: float):
        """Modulated-element harmonic admittance (with the shared stub) plus the
        capacitance coefficients used for reporting, selected by
        ``modulation_formulation`` (documented on the two public entry points).

        The ``"capacitance"`` branch is the historical path unchanged; the
        package wrap below is shared across all three formulations.
        """

        if modulation_formulation == "capacitance":
            c_coeffs = _coeffs(drive_value)
            return (
                tv_capacitor_harmonic_admittance_general(
                    c_coeffs, ws, wm, n, z_stub_fn=z_stub_fn,
                ),
                c_coeffs,
            )
        if modulation_formulation == "elastance":
            # S-linear tridiagonal: S0 = 1/C0, dS = m_s*S0, R = L = 0 inside the
            # element (package added by the shared wrap). Y = diag(j*omega) * M^-1.
            s0 = 1.0 / static_c0
            d_s = float(drive_value) * s0
            matrix, _rhs = build_stub_tridiagonal(
                s0, d_s, phi, 0.0, 0.0, ws, wm, n, y_stub_fn=y_stub,
            )
            green = np.linalg.solve(matrix, identity)
            y_el = 1j * omega_n[:, np.newaxis] * green
            c_rep = [
                complex(static_c0),
                0.5 * float(drive_value) * static_c0 * np.exp(1j * phi),
            ]
            return y_el, c_rep
        # "reciprocal": exact S(t) = 1/C(t) of the SAME capacitance device --
        # a different-route solver cross-check (same z_stub as the C path).
        c_coeffs = _coeffs(drive_value)
        stoep = _elastance_toeplitz_from_capacitance(c_coeffs, size)
        z_stub_arr = np.array(
            [complex(z_stub_fn(float(w))) for w in omega_n], dtype=np.complex128
        )
        matrix = stoep + np.diag(1j * omega_n * z_stub_arr)
        green = np.linalg.solve(matrix, identity)
        y_el = 1j * omega_n[:, np.newaxis] * green
        return y_el, c_coeffs

    def _package_wrap(y_element: np.ndarray) -> np.ndarray:
        if not has_pkg:
            # Ideal / zero-parasitic: return the bare admittance unchanged
            # (NOT inv(inv(...))) so the ideal-C reduction is bit-for-bit.
            return y_element
        z_element = np.linalg.solve(y_element, identity)
        return np.linalg.solve(np.diag(z_pkg) + z_element, identity)

    def make_cell(drive_value: float):
        y_element, c_coeffs = _element_admittance(drive_value)
        y_branch = _package_wrap(y_element)
        if s3_per_harm_for_surrogate is not None:
            # MTEE_ADS surrogate path: combine the tee's 3-port
            # S with the full harmonic-coupled branch admittance into a
            # single 2-port harmonic ABCD (preserves cross-harmonic
            # coupling from the varactor through the branch).
            from .mtee_ads_surrogate import build_combined_tee_branch_harmonic_abcd
            combined = build_combined_tee_branch_harmonic_abcd(
                s3_per_harm_for_surrogate, y_branch, z_ref=z_ref,
            )
            cell = cascade_harmonic_abcd([tl_left, combined, tl_right])
        elif use_mtee_parasitic:
            branch = (identity, zero, y_branch + np.diag(j_bt_vec), identity)
            cell = cascade_harmonic_abcd(
                [tl_left, block_left, branch, block_right, tl_right]
            )
        else:
            branch = (identity, zero, y_branch, identity)
            cell = cascade_harmonic_abcd([tl_left, branch, tl_right])
        return cell, c_coeffs

    return make_cell, n, n_cells, z_ref


def level10_realistic_param_gain(
    spec: DeviceSpec,
    varactor: RealisticVaractor,
    *,
    m_branch_pump: float | None = None,
    control_port_vpump_v: float | None = None,
    m_s_branch: float | None = None,
    modulation_formulation: str = "capacitance",
    truncation_N: int = 6,
    use_mtee_parasitic: bool = True,
    input_match_blocks=None,
    output_match_blocks=None,
    tee_model_variant=None,
    via_model=None,
    open_end_model=None,
    stub_y_fn_override=None,
) -> Mapping[str, object]:
    """Return the analytical parametric-gain dict for a realistic-varactor cascade.

    Two pump-injection modes, mutually exclusive (provide exactly one drive):

    * **self-pumped** (``m_branch_pump``): the pump propagates down the line and
      reaches the branch top; the package divider reduces the branch-top depth
      to the effective junction depth ``varactor.m_effective(f_pump, m)``. This
      is the self-pumped cross-validation configuration. Pass
      :meth:`RealisticVaractor.ideal` here to recover the ideal-C leg
      bit-for-bit.
    * **control-port** (``control_port_vpump_v``): the junction sees the pump
      voltage ``Voc`` **directly** (the control port imposes it, bypassing the
      package divider for pump delivery), while ``L_pkg`` / ``R_series`` remain
      in the signal-path branch wrap. The capacitance spectrum is the natural
      ``C(V_bias + Voc*cos)`` Taylor content
      (:meth:`RealisticVaractor.capacitance_coeffs_from_junction_vpump`),
      matching the control-port realistic SDD body. This is the control-port
      positive-gain configuration.

    Parameters
    ----------
    spec : DeviceSpec
        A Level-10-shaped cascade spec carrying the through-line halves
        (``LINE_LEFT_C{k}`` / ``LINE_RIGHT_C{k}``), the MTEE (``TEE_C{k}``),
        the stub-MLIN + via shunt chain, and the operating-point VAR block
        (``f_signal`` / ``f_pump`` / optional ``pump_phase_rad``). The cell
        count is the number of SDD2P branches.
    varactor : RealisticVaractor
        The device model. Supplies ``C0`` (incl. CP), the C-V curvature, and
        the package ``L_pkg`` / ``R_series`` for the branch wrap.
    m_branch_pump : float, optional
        Self-pumped branch-top modulation depth (the framework's
        ``C(t) = C0*(1 + m*cos)`` convention).
    control_port_vpump_v : float, optional
        Control-port junction pump voltage ``Voc`` [V].
    truncation_N : int, optional
        Floquet / harmonic truncation; block size ``2*N+1``. Default 6. Must
        be >= 1 (the idler at ``n = -1`` is needed).
    use_mtee_parasitic : bool, optional
        Reference-plane MTEE (default ``True``, matching the Level10IdealC
        family) vs the ideal junction (``False``). Identical semantics to
        :func:`.ideal_c_gain.level10_ideal_c_param_gain`.

    Returns
    -------
    Mapping
        ``param_gain_dB`` (pumped minus unpumped), ``abs_gain_on_dB`` /
        ``abs_gain_off_dB`` (``20*log10|S21[0,0]|``), ``idler_over_signal``
        (output, pumped), ``drive`` (the passed depth or Voc), ``mode``
        (``"self_pumped"`` / ``"control_port"``), ``depth_effective`` (the
        realized fundamental depth ``2|C_1|/C_0``), ``n_cells``,
        ``label = "analytical_proxy_only"``.
    """

    control_port, drive = _resolve_drive(
        modulation_formulation, m_branch_pump, control_port_vpump_v, m_s_branch
    )
    if int(truncation_N) < 1:
        raise ValueError("truncation_N must be >= 1 (idler harmonic needed).")
    make_cell, n, n_cells, z_ref = _realistic_cell_builder(
        spec, varactor, control_port=control_port, truncation_N=truncation_N,
        use_mtee_parasitic=use_mtee_parasitic, tee_model_variant=tee_model_variant,
        via_model=via_model, open_end_model=open_end_model,
        stub_y_fn_override=stub_y_fn_override,
        modulation_formulation=modulation_formulation,
    )

    def _gain(drive_value: float) -> tuple[float, float, list[complex], np.ndarray]:
        cell, c_coeffs = make_cell(drive_value)
        cur = None
        for _ in range(n_cells):
            cur = cell if cur is None else cascade_harmonic_abcd([cur, cell])
        # Stage-3 matching sections: cascade the input/output
        # matching harmonic blocks around the n-cell core, source-side first.
        # Default None -> chain is [cur] -> identical arithmetic (the
        # ideal-reduction diff==0.0 anchor and all prior baselines are preserved
        # bit-for-bit).
        if input_match_blocks or output_match_blocks:
            chain: list = []
            if input_match_blocks:
                chain.extend(input_match_blocks)
            chain.append(cur)
            if output_match_blocks:
                chain.extend(output_match_blocks)
            cur = cascade_harmonic_abcd(chain)
        _s11, s21 = harmonic_s_parameters(*cur, z_ref, z_ref)
        g = device_proxy_gain(s21, n)
        sig = abs(s21[n, n])
        idl = abs(s21[n - 1, n])
        ios = idl / sig if sig > 0 else float("nan")
        return float(g["gain_signal_proxy_dB"]), ios, c_coeffs, s21

    g_on, idler_over_signal, coeffs_on, s21_on = _gain(drive)
    g_off, _, _, s21_off = _gain(0.0)
    coh = device_coherent_gain(s21_on, s21_off, n)
    c0_realized = abs(complex(coeffs_on[0]))
    depth_realized = (
        2.0 * abs(complex(coeffs_on[1])) / c0_realized if c0_realized > 0 else float("nan")
    )
    return {
        "param_gain_dB": float(g_on - g_off),
        "abs_gain_on_dB": float(g_on),
        "abs_gain_off_dB": float(g_off),
        "idler_over_signal": float(idler_over_signal),
        "drive": drive,
        "mode": "control_port" if control_port else "self_pumped",
        "depth_effective": float(depth_realized),
        "n_cells": int(n_cells),
        "coherent_peak_dB": coh["coherent_peak_dB"],
        "coherent_min_dB": coh["coherent_min_dB"],
        "coherent_swing_dB": coh["coherent_swing_dB"],
        "coherent_at_phase0_dB": coh["coherent_at_block_phase_dB"],
        "label": "analytical_proxy_only",
    }


def realistic_pumped_unit_cell_abcd(
    spec: DeviceSpec,
    varactor: RealisticVaractor,
    *,
    m_branch_pump: float | None = None,
    control_port_vpump_v: float | None = None,
    m_s_branch: float | None = None,
    modulation_formulation: str = "capacitance",
    truncation_N: int = 6,
    use_mtee_parasitic: bool = True,
    tee_model_variant=None,
    via_model=None,
    open_end_model=None,
    stub_y_fn_override=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the pumped harmonic-basis unit-cell ABCD ``(A, B, C, D)``.

    The single pumped unit cell whose ``n_cells``-fold cascade
    :func:`level10_realistic_param_gain` converts to a parametric gain. Built
    from the shared :func:`_realistic_cell_builder`, so the eigenvalues of the
    assembled transfer matrix are the per-cell Floquet/Bloch multipliers of the
    *same* cell the gain leg uses. Feed the returned tuple to
    :func:`tvtl.floquet.stability.stability_margin` (``1 - max_k |lambda_k|``)
    for the analytical parametric-stability margin (an analytical proxy only;
    the cell carries no proxy/realized label itself).

    Drive selection mirrors :func:`level10_realistic_param_gain` exactly:
    provide exactly one of ``m_branch_pump`` (self-pumped) or
    ``control_port_vpump_v`` (control-port). Each returned block has shape
    ``(2*truncation_N + 1, 2*truncation_N + 1)``.
    """

    control_port, drive = _resolve_drive(
        modulation_formulation, m_branch_pump, control_port_vpump_v, m_s_branch
    )
    if int(truncation_N) < 1:
        raise ValueError("truncation_N must be >= 1 (idler harmonic needed).")
    make_cell, _n, _n_cells, _z_ref = _realistic_cell_builder(
        spec, varactor, control_port=control_port, truncation_N=truncation_N,
        use_mtee_parasitic=use_mtee_parasitic, tee_model_variant=tee_model_variant,
        via_model=via_model, open_end_model=open_end_model,
        stub_y_fn_override=stub_y_fn_override,
        modulation_formulation=modulation_formulation,
    )
    cell, _coeffs = make_cell(drive)
    return cell
