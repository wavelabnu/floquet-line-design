"""Self-pumped optimized device-spec generator and design workflow.

Two public functions and a result/error type. Mirror of
:mod:`tvtl_ads_builder_v2.parity.optimized_design` for the **self-pumped**
configuration: signal (Num=1) and pump (Num=2) both attached to ``n_input``,
with the pump propagating down the cascade and reaching each varactor's
junction through the package ``L_pkg`` / ``R_series`` divider.

API
---
* :func:`optimized_self_pumped_spec` — low-level spec builder (parameterised
  by stub topology, main width, matching networks, pump operating point).
* :func:`design_optimized_self_pumped` — high-level workflow: pump
  calibration via :func:`tvtl.optimization.figure_of_merit.calibrate_self_pumped_pump_dBm`,
  Sobol screen via :class:`tvtl.optimization.figure_of_merit.ClosedFormA1SelfPumped`,
  Stage-2 Bloch match, Stage-3 distributed matching, Stage-4 analytical
  walk-down, returning a :class:`SelfPumpedDesignResult`.

The workflow was established empirically at the A.3.1 anchor
(Varactor A at 3.1 GHz: n2_single_short, +1.67 dBm calibrated pump,
analytical CN coh_pk +4.88 dB, ADS HB PSA peak +9.71 dB, eps_peak 4.83 dB
under the cross-validation ceiling). The A.3.1
selection is locked by regression test in
``tests/test_self_pumped_design.py``.

Layering
--------
* Spec emission (Element / Node / SDD2P with charge-polynomial body) lives
  here. The structure mirrors :func:`tvtl_ads_builder_v2.parity.optimized_design.optimized_control_port_spec`,
  differing only in source declaration (both Num=1 and Num=2 on ``n_input``,
  no separate ``n_pump_ctrl`` net), the SDD body (``q_poly_coefficients``
  rather than ``tvc_poly_coefficients``), and the SDD pin 3/4 self-tie
  (per-cell ``n_c{k}_ret`` return net).
* The high-level workflow imports from :mod:`tvtl.optimization` for Stage 1
  (`coarse_screen` with the self-pumped A1), Stage 2 (`match_z_tl`), and
  Stage 3 (`edge_impedance`, `select_matching`). The analytical leg
  (Stage 4) calls :func:`tvtl_ads_builder_v2.analytical.level10_realistic_param_gain`
  with the ``m_branch_pump`` kwarg.

Open question (pump-voltage convention)
---------------------------------------
The matched-load convention ``V_branch_top = sqrt(2 * Z * P_pump_watts)``
in :func:`tvtl.optimization.figure_of_merit.m_branch_from_source_pump_dBm`
is provisional. Calibration outputs a self-consistent source ``pump_dBm``
under this convention; if the canonical self-pumped V_branch_top
convention is later specified differently (e.g., Voc rather than Voc/2),
the calibrated pump and downstream gain numbers shift, but the
Stage-1 rank-equivalence finding is convention-immune.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, minimize

from tvtl.floquet.harmonic_abcd import harmonic_frequencies
from tvtl.microstrip.discontinuities import via_inductance_bahl_H
from tvtl.optimization.figure_of_merit import (
    ClosedFormA1SelfPumped,
    calibrate_self_pumped_pump_dBm,
    m_branch_from_source_pump_dBm,
)
from tvtl.optimization.stage1_per_cell import StubCandidate, coarse_screen
from tvtl.optimization.stage2_main_tl import (
    MainTLDesign,
    bloch_attenuation_per_cell,
    match_z_tl,
)
from tvtl.optimization.stage3_matching import (
    edge_impedance,
    select_matching,
    single_stub_design,
)
from tvtl.optimization.substrate import ROGERS_4003C, SubstrateConfig
from tvtl.optimization.topology_generator import (
    Termination,
    TopologyFamily,
    VIA_RADIUS_M,
    combined_families,
    is_valid,
    linear_families,
    param_bounds,
    topology_y_stub_fn,
)
from tvtl.varactors import RealisticVaractor

from ..analytical import level10_realistic_param_gain
from ..analytical.matching_blocks import matching_section_harmonic_blocks
from ..cells import SingleShuntVaractorCell
from ..device_spec import (
    Analysis,
    DeviceSpec,
    Element,
    ElementPin,
    FrequencyGrid,
    HBTwoToneSpec,
    Node,
    P1ToneSource,
    Port,
    VarBlockSpec,
    VarEntry,
)
from ..device_spec.schema import DEVICE_SPEC_VERSION
from .canonical_circuits import _DEFAULT_LEVEL10_VAR_NAMES
from .gain_objective import (
    OBJECTIVE_COH_PK,
    OBJECTIVE_G_MOD_IN,
    analytical_g_mod_in_dB,
    validate_objective,
)
from .optimized_design import (
    _MAIN_WIDTH_M,
    _emit_match_network,
    _substrate_from_config,
)

__all__ = [
    "optimized_self_pumped_spec",
    "design_optimized_self_pumped",
    "SelfPumpedDesignResult",
    "NoSelfPumpedCandidateError",
    "main_period_anchor_at_3p1GHz",
    "STOPBAND_ALPHA_THRESHOLD_NP_PER_CELL",
]

# The band-structure-aware walk-down (refinement).
#
# The Stage-1 A1 figure of merit ranks Bloch-STOPBAND geometries to the top (high
# A1 correlates with strong loading / band-edge resonance), so the raw top-K-by-A1
# walk-down pool is stopband-dominated even though ~99% of *realizable* geometries
# are in the passband. The g_mod/in proxy then selects the best of those stopband
# cells -> poor realized transducer gain (an unscreened search saw -35 dB). The opt-in
# band-structure screen (``reject_signal_stopband``) instead walks the A1-ranked
# screen keeping the candidates whose signal sits in a Bloch PASSBAND -- the
# low-passive-loss placement where the parametric boost can actually net positive.
#
# Propagating threshold on the per-cell Bloch attenuation alpha_p [Np/cell]
# (the net-gain propagating-threshold convention): alpha_p <= this is "passband"
# (propagating), above it is a stopband (deeply suppressed, ratio-inflated).
STOPBAND_ALPHA_THRESHOLD_NP_PER_CELL: float = 0.02

# Default cap on how far down the A1-ranked screen the passband walk searches for
# K passband-realizable candidates. The stopband-biased cells cluster at the top
# of the A1 ranking, so the first passband candidate can be ~A1-rank 100; this cap
# bounds the per-attempt Stage-2 cost while leaving ample margin (the empirical
# depth to collect 10 passband candidates is ~90-210 across the design set).
_BAND_SCREEN_WALK_CAP: int = 500

# Stage-2 (Nelder-Mead) refinement (``refine``): penalty [dB per Np/cell] applied
# to the refinement objective for each Np/cell the candidate's alpha_p exceeds the
# passband threshold. Large enough to dominate any in-band boost, so the simplex
# is driven OUT of the stopband and then maximises the passband g_mod/in proxy.
_STOPBAND_REFINE_PENALTY_DB_PER_NP: float = 200.0
_REFINE_INVALID_PENALTY: float = 1.0e3

# Main-line period scaling.
# Anchor is the paper-fidelity Level-10 main-line period (6.11 mm) at the
# experimental device's signal frequency (3.1 GHz). For other frequencies,
# the period is scaled inversely with signal frequency, which keeps the
# per-cell main-line electrical length (and therefore the per-cell phase
# advance for the periodic loaded line) approximately constant across the
# design set. Stage 2 Bloch realizability is sensitive to this
# per-cell phase advance; without scaling, higher-frequency design points
# place Z_Bloch outside the 50 ohm-bisectable band.
_PAPER_MAIN_PERIOD_M_AT_3P1 = 6.11e-3
_PAPER_ANCHOR_F_SIGNAL_HZ = 3.1e9


def main_period_anchor_at_3p1GHz(f_signal_Hz: float) -> float:
    """Wavelength-anchored main-line period for the self-pumped cascade.

    Returns the paper-fidelity Level-10 main-line period (6.11 mm at
    f_signal = 3.1 GHz) scaled inversely with signal frequency::

        period(f) = 6.11 mm * (3.1 GHz / f)

    At ``f_signal = 3.1 GHz`` the result is exactly 6.11 mm, preserving
    the published A.3.1 device geometry bit-for-bit. At higher signal
    frequencies the period shrinks proportionally so the per-cell
    electrical length remains comparable across the design set.
    """

    return _PAPER_MAIN_PERIOD_M_AT_3P1 * (
        _PAPER_ANCHOR_F_SIGNAL_HZ / float(f_signal_Hz)
    )


# =========================================================================== #
# Internal helpers                                                             #
# =========================================================================== #


def _q_poly_reference(topology: TopologyFamily, varactor) -> str:
    # Power-law (RealisticVaractor) keeps the historical CJ0/VJ/M reference string
    # byte-for-byte (the recovery path). A polynomial / non-power-law device (the
    # model registry's PolynomialVaractor) carries its own reference_tag.
    if not hasattr(varactor, "CJ0_f"):
        tag = getattr(varactor, "reference_tag", type(varactor).__name__)
        return f"optimized_self_pumped[{topology.key}]({tag})"
    return (
        f"optimized_self_pumped[{topology.key}]("
        f"CJ0={varactor.CJ0_f * 1e12:.3g}pF,VJ={varactor.VJ_v:.3g}V,"
        f"M={varactor.M:.3g},L_pkg={varactor.L_pkg_h * 1e9:.3g}nH,"
        f"R={varactor.R_series_ohm:.3g}ohm)"
    )


def _matching_loss_dB(section, f_signal_Hz: float, truncation_N: int) -> float:
    """Insertion loss of a matching network in dB at the signal harmonic.

    Computes ``-20 log10 |S_21|`` from the framework's
    :func:`matching_section_harmonic_blocks` evaluated at the signal-
    fundamental harmonic index. Complex scalar ABCD accumulation.
    Returns 0.0 for ``section is None`` (no matching emitted).
    """

    if section is None:
        return 0.0
    ws = 2.0 * math.pi * float(f_signal_Hz)
    wm = 2.0 * math.pi * 2.0 * float(f_signal_Hz)
    omega_n = harmonic_frequencies(ws, wm, truncation_N)
    tvtl_sub = ROGERS_4003C.to_tvtl_substrate()
    blocks = matching_section_harmonic_blocks(section, omega_n, tvtl_sub, side="input")
    if not blocks:
        return 0.0
    n_idx = truncation_N
    cA = 1.0 + 0.0j
    cB = 0.0 + 0.0j
    cC = 0.0 + 0.0j
    cD = 1.0 + 0.0j
    for (A, B, C, D) in blocks:
        a = complex(A[n_idx, n_idx])
        b = complex(B[n_idx, n_idx])
        c = complex(C[n_idx, n_idx])
        d = complex(D[n_idx, n_idx])
        nA = cA * a + cB * c
        nB = cA * b + cB * d
        nC = cC * a + cD * c
        nD = cC * b + cD * d
        cA, cB, cC, cD = nA, nB, nC, nD
    z0 = 50.0 + 0.0j
    s21 = 2.0 / (cA + cB / z0 + cC * z0 + cD)
    return -20.0 * math.log10(abs(s21)) if abs(s21) > 0 else float("inf")


# =========================================================================== #
# Public API                                                                   #
# =========================================================================== #


def optimized_self_pumped_spec(
    *,
    topology: TopologyFamily,
    params,
    varactor: RealisticVaractor,
    substrate: SubstrateConfig = ROGERS_4003C,
    n_cells: int = 15,
    main_width_m: float | None = None,
    main_period_m: float | None = None,
    f_signal_Hz: float = 3.1e9,
    f_pump_Hz: float = 6.2e9,
    signal_power_dBm: float = -30.0,
    pump_power_dBm: float = 16.0,
    pump_phase_rad: float = 0.0,
    via_radius_m: float = VIA_RADIUS_M,
    spec_id: str | None = None,
    f_start_Hz: float = 2.0e9,
    f_stop_Hz: float = 17.0e9,
    n_points: int = 151,
    port_impedance_ohm: float = 50.0,
    hb_order_signal: int = 5,
    hb_order_pump: int = 5,
    hb_max_order: int = 6,
    input_match=None,
    output_match=None,
    via_model=None,
    open_end_model=None,
    bias_return_ohm: float | None = None,
) -> DeviceSpec:
    """Build a self-pumped Level-10 cascade given pre-decided geometry.

    ``bias_return_ohm`` adds a resistor from each varactor's stub-side terminal to ground
    (``RDCRET_C{k}``), the DC return an OPEN-terminated stub lacks; without it the HB deck of an
    OPEN family is refused (``no_dc_return_for_varactor``).

    Self-pumped sibling of :func:`tvtl_ads_builder_v2.parity.optimized_design.optimized_control_port_spec`.
    Both signal (Num=1) and pump (Num=2) attach to ``n_input``; the pump
    propagates down the cascade and reaches each cell's junction through
    its ``L_pkg`` / ``R_series`` package divider. The SDD body is the
    charge polynomial ``Q(v)`` from :meth:`RealisticVaractor.charge_polynomial`;
    pins 3 and 4 are tied to a per-cell return net (KCL closure).

    Defaults match the self-pumped canonical
    (``level10_realistic_c.py``): paper-fidelity main width
    (``main_width_m=None`` -> ``_MAIN_WIDTH_M``), branch width 1.93 mm,
    +16 dBm pump, hb_order=5/5 (HB-8 cross-checks are an opt-in
    override). The small-signal anchor regime uses ``pump_power_dBm``
    near +1.67 dBm for Varactor A at f_pump=6.2 GHz; see
    :func:`tvtl.optimization.figure_of_merit.calibrate_self_pumped_pump_dBm`
    for the principled calibration.

    Parameters mirror :func:`optimized_control_port_spec` argument-for-
    argument except for the absence of a ``control_net`` parameter (the
    self-pumped path has no separate pump net).

    Raises ``ValueError`` for ``ideal_linear`` varactors (no pump
    response).

    COMBINED-termination topologies build the
    MLEF-open-stub-IN-PARALLEL-WITH-VIAGND resonant termination via
    :class:`SingleShuntVaractorCell`. The ADS deck emission + flat-HB path are
    fully supported; the analytical-leg cascade walker
    (:func:`level10_realistic_param_gain`) does not yet trace the COMBINED
    parallel via||open-stublet tree, so a COMBINED design is characterised from
    ADS HB directly (the pump-ladder/stability driver), not from the
    in-loop analytical Stage-4 score.
    """

    if int(n_cells) < 1:
        raise ValueError("n_cells must be >= 1.")
    if varactor.ideal_linear:
        raise ValueError(
            "self-pumped realisation requires a curved varactor; an ideal "
            "constant-C device has no pump response."
        )

    width_main = _MAIN_WIDTH_M if main_width_m is None else float(main_width_m)
    period_main = (
        main_period_anchor_at_3p1GHz(float(f_signal_Hz))
        if main_period_m is None else float(main_period_m)
    )
    substrate_obj = _substrate_from_config(substrate)
    q_poly = varactor.charge_polynomial().charge_polynomial_coefficients
    q_ref = _q_poly_reference(topology, varactor)
    # A via is present for SHORT and for COMBINED (the via half of MLEF∥VIAGND); OPEN
    # has none. The lumped-via fallback (no via_model) uses this Bahl inductance.
    has_via = topology.termination in (Termination.SHORT, Termination.COMBINED)
    l_via_H = (
        via_inductance_bahl_H(
            substrate_thickness_m=float(substrate.h_m),
            via_radius_m=float(via_radius_m),
            n_parallel=topology.n_via_parallel,
        )
        if has_via else 0.0
    )
    label = (
        spec_id if spec_id is not None
        else f"optimized_sp_{topology.key}_n{n_cells}"
    )

    nodes: list[Node] = [Node(id="n_input"), Node(id="n_pin")]
    elements: list[Element] = []
    ground_nodes: list[str] = []
    l_via_match_H = via_inductance_bahl_H(
        substrate_thickness_m=float(substrate.h_m),
        via_radius_m=float(via_radius_m), n_parallel=1,
    )

    if input_match is not None:
        feed = "n_mi_feed"
        nodes.append(Node(id=feed))
        mn_nodes, mn_els, mn_g = _emit_match_network(
            input_match, port_ref_net="n_input", port_casc_net=feed, tag="IN",
            substrate_name=substrate_obj.name, l_via_H=l_via_match_H,
        )
        nodes.extend(mn_nodes)
        elements.extend(mn_els)
        ground_nodes.extend(mn_g)
        elements.append(Element(
            id="IPR_IN", type="I_Probe",
            pins=(ElementPin(pin=1, node=feed),
                  ElementPin(pin=2, node="n_pin")),
            parameters={},
        ))
    else:
        elements.append(Element(
            id="IPR_IN", type="I_Probe",
            pins=(ElementPin(pin=1, node="n_input"),
                  ElementPin(pin=2, node="n_pin")),
            parameters={},
        ))

    # Cell-agnostic assembly: the single-shunt-varactor cell is built once and
    # the cascade walks it through the CellSpec interface.
    # Only this one cell exists today; enriched PT-paper cells become additional
    # CellSpec implementations and this loop is unchanged.
    cell = SingleShuntVaractorCell(
        topology=topology, params=params, varactor=varactor,
        substrate_name=substrate_obj.name, main_width_m=width_main,
        main_period_m=period_main, q_poly=q_poly, q_poly_reference=q_ref,
        l_via_H=l_via_H, via_model=via_model, open_end_model=open_end_model,
        bias_return_ohm=bias_return_ohm,
    )
    nc = int(n_cells)
    for k in range(1, nc + 1):
        cell_in = "n_pin" if k == 1 else f"n_pc{k - 1}c{k}"
        cell_out = cell.output_terminal_node(k)
        nodes.extend(cell.cell_nodes(k))
        elements.extend(cell.cell_elements(k, input_net=cell_in, output_net=cell_out))
        ground_nodes.extend(cell.ground_nodes(k))
        if k < nc:
            inter = f"n_pc{k}c{k + 1}"
            nodes.append(Node(id=inter))
            elements.append(Element(
                id=f"IPR_C{k}C{k + 1}", type="I_Probe",
                pins=(ElementPin(pin=1, node=cell_out),
                      ElementPin(pin=2, node=inter)),
                parameters={},
            ))

    nodes.append(Node(id="n_output"))
    last_out = cell.output_terminal_node(nc)
    if output_match is not None:
        casc = "n_mo_casc"
        nodes.append(Node(id=casc))
        elements.append(Element(
            id="IPR_OUT", type="I_Probe",
            pins=(ElementPin(pin=1, node=last_out),
                  ElementPin(pin=2, node=casc)),
            parameters={},
        ))
        mn_nodes, mn_els, mn_g = _emit_match_network(
            output_match, port_ref_net="n_output", port_casc_net=casc, tag="OUT",
            substrate_name=substrate_obj.name, l_via_H=l_via_match_H,
        )
        nodes.extend(mn_nodes)
        elements.extend(mn_els)
        ground_nodes.extend(mn_g)
    else:
        elements.append(Element(
            id="IPR_OUT", type="I_Probe",
            pins=(ElementPin(pin=1, node=last_out),
                  ElementPin(pin=2, node="n_output")),
            parameters={},
        ))
    elements.append(Element(
        id="P_OUT", type="TermG",
        pins=(ElementPin(pin=1, node="n_output"),),
        parameters={"z_ohm": float(port_impedance_ohm), "num": 3},
    ))

    # NAME elements for per-cell branch-top and varactor-middle nodes (used by
    # downstream diagnostic tools like the standing-wave pump-propagation
    # probe).
    name_counter = 0

    def _add_name(net: str) -> None:
        nonlocal name_counter
        elements.append(Element(
            id=f"NAME_{name_counter:02d}_{net}", type="NAME",
            pins=(ElementPin(pin=1, node=net),), parameters={},
        ))
        name_counter += 1

    _add_name("n_input")
    for k in range(1, nc + 1):
        for net in cell.name_nodes(k):
            _add_name(net)
    _add_name("n_output")

    vn = _DEFAULT_LEVEL10_VAR_NAMES
    var_block = VarBlockSpec(
        name="VAR1",
        variables=(
            VarEntry(name=vn["f_signal"], value=float(f_signal_Hz)),
            VarEntry(name=vn["f_pump"], value=float(f_pump_Hz)),
            VarEntry(name=vn["signal_power_dBm"], value=float(signal_power_dBm)),
            VarEntry(name=vn["pump_power_dBm"], value=float(pump_power_dBm)),
            VarEntry(name=vn["pump_phase_rad"], value=float(pump_phase_rad)),
        ),
    )
    sources = (
        P1ToneSource(name="SIGNAL_SRC", port_node="n_input", num=1,
                     freq=vn["f_signal"], power_dBm=vn["signal_power_dBm"]),
        P1ToneSource(name="PUMP_SRC", port_node="n_input", num=2,
                     freq=vn["f_pump"], power_dBm=vn["pump_power_dBm"],
                     phase_rad=vn["pump_phase_rad"]),
    )
    hb = HBTwoToneSpec(
        fundamental_freq_1=vn["f_signal"], fundamental_order_1=int(hb_order_signal),
        fundamental_freq_2=vn["f_pump"], fundamental_order_2=int(hb_order_pump),
        max_order=int(hb_max_order),
    )
    analysis = Analysis(
        type="s_parameters",
        frequency_grid_Hz=FrequencyGrid(
            start=float(f_start_Hz), stop=float(f_stop_Hz), points=int(n_points)
        ),
        port_impedance_ohm=float(port_impedance_ohm),
    )
    return DeviceSpec(
        device_spec_version=DEVICE_SPEC_VERSION,
        analysis=analysis,
        substrate=substrate_obj,
        nodes=tuple(nodes),
        elements=tuple(elements),
        ports=(Port(num=3, node="n_output"),),
        label=str(label),
        ground_nodes=tuple(ground_nodes),
        var_block=var_block,
        hb_two_tone=hb,
        sources=sources,
    )


# =========================================================================== #
# High-level workflow                                                          #
# =========================================================================== #


@dataclass(frozen=True)
class _WalkdownEntry:
    """One per-candidate walk-down diagnostic row.

    ``families_attempt`` is the 1-based index of the ``families_fallback``
    entry that this candidate was screened under (1 = primary, 2 = first
    fallback, ...). When fallback is triggered, the result's
    ``walkdown_trace`` is the concatenation of all attempts' traces;
    ``families_attempt`` lets callers see which families pool produced
    each entry.
    """

    families_attempt: int
    rank: int
    topology_key: str
    a1_score: float
    stage2_realizable: bool
    z_tl_ohm: float
    w_main_m: float
    z_in_real: float
    z_in_imag: float
    z_out_real: float
    z_out_imag: float
    input_match_topology: str
    output_match_topology: str
    matching_loss_input_dB: float
    matching_loss_output_dB: float
    analytical_coh_pk_dB: float
    analytical_psa_swing_dB: float
    depth_effective: float
    note: str
    # G_mod/in objective: the analytical out/in transducer-gain proxy
    # (abs_gain_off + coh_pk). Default nan keeps the coh_pk walk-down entries
    # byte-unchanged; only the g_mod_in objective populates it.
    analytical_g_mod_in_dB: float = float("nan")
    # Band-structure screen: the candidate's per-cell passive
    # Bloch attenuation alpha_p [Np/cell] at the signal frequency (nan unless
    # ``reject_signal_stopband`` computed it). alpha_p above the propagating
    # threshold means the signal sits in a Bloch stopband.
    bloch_alpha_p_np: float = float("nan")


@dataclass(frozen=True)
class SelfPumpedDesignResult:
    """Output of :func:`design_optimized_self_pumped`.

    ``families_used`` is the 1-based index of the ``families_fallback``
    entry that produced the selected candidate (1 = primary, 2 = first
    fallback, ...). The field is additive; existing callers
    can ignore it (defaults to 1 for the primary-attempt-only path).
    """

    spec: DeviceSpec
    calibrated_pump_dBm: float
    topology: TopologyFamily
    stub_params: tuple[float, ...]
    main_width_m: float
    main_period_m: float
    z_tl_ohm: float
    input_match: object | None
    output_match: object | None
    matching_loss_input_dB: float
    matching_loss_output_dB: float
    analytical_coh_pk_dB: float
    analytical_psa_swing_dB: float
    depth_effective: float
    a1_score: float
    sobol_rank: int
    families_used: int = 1
    walkdown_trace: tuple = field(default_factory=tuple)
    # Selectable objective. Default OBJECTIVE_COH_PK keeps existing
    # callers' results byte-unchanged; analytical_g_mod_in_dB is the selected
    # candidate's out/in proxy (nan under the coh_pk objective).
    objective: str = OBJECTIVE_COH_PK
    analytical_g_mod_in_dB: float = float("nan")


class NoSelfPumpedCandidateError(RuntimeError):
    """Raised when the walk-down's top-K finds no candidate that passes
    both the matching-loss and gain-window gates.

    Attribute ``walkdown_trace`` carries the per-candidate diagnostic
    info so the caller can see what failed and why (Stage-2 realizable?
    matching loss too high? coh_pk outside window?).
    """

    def __init__(self, message: str, walkdown_trace: tuple = ()):
        super().__init__(message)
        self.walkdown_trace = walkdown_trace


def _default_families_fallback() -> tuple[tuple[TopologyFamily, ...], ...]:
    """Default families policy: SHORT-only primary, SHORT+OPEN fallback."""
    short = tuple(
        f for f in linear_families() if f.termination == Termination.SHORT
    )
    open_ = tuple(
        f for f in linear_families() if f.termination == Termination.OPEN
    )
    return (short, short + open_)


# --------------------------------------------------------------------------- #
#  Shared per-candidate evaluation (band screen + refinement)                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _EvalContext:
    """Frozen bundle of the per-attempt context a single candidate needs.

    Lets the opt-in band-structure screen and the opt-in Nelder-Mead refinement
    evaluate an arbitrary ``(topology, geometry)`` through the *same* Stage 2-4
    chain the coarse walk-down uses, without threading ~20 positional args. The
    coarse walk-down itself stays inline (so the default path is byte-identical);
    this context is only consumed by the opt-in helpers below.
    """

    varactor: RealisticVaractor
    f_signal_Hz: float
    f_pump_Hz: float
    n_cells: int
    substrate: SubstrateConfig
    main_period_m: float
    pump_dBm_calibrated: float
    m_branch_pump: float
    truncation_N: int
    hb_order_signal: int
    hb_order_pump: int
    hb_max_order: int
    via_radius_m: float
    omega_n: object
    tvtl_sub: object
    reject_signal_stopband: bool
    stopband_alpha_threshold: float
    match_low_z_with_tuner: bool
    low_z_tuner_s11_target_dB: float
    via_model: object = None
    open_end_model: object = None


@dataclass(frozen=True)
class _CandidateEval:
    """Outcome of evaluating one geometry through Stage 2-4 (opt-in helpers)."""

    ok: bool
    proxy_dB: float = float("nan")
    alpha_p_np: float = float("nan")
    in_stopband: bool = False
    coh_pk_dB: float = float("nan")
    psa_swing_dB: float = float("nan")
    depth_effective: float = float("nan")
    z_tl_ohm: float = float("nan")
    w_main_m: float = float("nan")
    z_in: complex = 0j
    z_out: complex = 0j
    loss_in_dB: float = float("nan")
    loss_out_dB: float = float("nan")
    input_match_topology: str = ""
    output_match_topology: str = ""
    mtl: MainTLDesign | None = None
    im: object = None
    om: object = None
    note: str = ""


def _stage2_realizable_or_tuner(cand: StubCandidate, ctx: _EvalContext):
    """Stage-2 match with the same low-Z-tuner keep rule the coarse loop uses.

    Returns ``(mtl, keep, tuner_note)``: ``keep`` is whether the candidate
    survives Stage 2 (realizable, or low-Z kept by the opt-in edge tuner).
    """

    mtl = match_z_tl(
        cand, ctx.varactor, substrate=ctx.substrate, f0_Hz=ctx.f_signal_Hz
    )
    if mtl.realizable:
        return mtl, True, ""
    keep = False
    if ctx.match_low_z_with_tuner and math.isfinite(mtl.w_main_m):
        z_in_c, z_out_c = edge_impedance(
            mtl, ctx.varactor, ctx.n_cells, substrate=ctx.substrate
        )
        ss_in = single_stub_design(
            z_in_c, ctx.f_signal_Hz, substrate=ctx.substrate, stub_type="open"
        )
        ss_out = single_stub_design(
            z_out_c, ctx.f_signal_Hz, substrate=ctx.substrate, stub_type="open"
        )
        keep = bool(
            ss_in.realizable and ss_out.realizable
            and ss_in.s11_after_dB <= ctx.low_z_tuner_s11_target_dB
            and ss_out.s11_after_dB <= ctx.low_z_tuner_s11_target_dB
        )
    note = "Stage 2: low-Z, kept (edge tuner-matchable)" if keep else ""
    return mtl, keep, note


def _candidate_alpha_p(cand: StubCandidate, mtl: MainTLDesign, ctx: _EvalContext) -> float:
    """Per-cell passive Bloch attenuation alpha_p [Np/cell] of the emitted cell.

    Built at the engine's *scaled* main-line period so it reflects the device the
    spec actually emits (not the fixed-period Stage-2 match cell).
    """

    return bloch_attenuation_per_cell(
        cand, ctx.varactor, mtl.z_tl_ohm, substrate=ctx.substrate,
        f0_Hz=ctx.f_signal_Hz, main_period_m=ctx.main_period_m,
    )


def _evaluate_candidate_proxy(cand: StubCandidate, ctx: _EvalContext) -> _CandidateEval:
    """Run Stage 2 (match) + Stage 3 (matching) + Stage 4 (proxy) for one geometry.

    Mirrors the coarse walk-down's per-candidate chain exactly (same Stage-4 call
    and kwargs), so the proxy it returns is directly comparable to the coarse
    selection. Returns ``ok=False`` when Stage 2 drops the candidate or Stage 4
    raises (the candidate is then skipped). Only used by the opt-in band screen
    and refinement; the byte-identical default path never calls it.
    """

    mtl, keep, _tuner_note = _stage2_realizable_or_tuner(cand, ctx)
    if not keep:
        return _CandidateEval(ok=False, note="Stage 2: Z_TL not realizable")

    alpha_p = float("nan")
    in_stopband = False
    if ctx.reject_signal_stopband:
        alpha_p = _candidate_alpha_p(cand, mtl, ctx)
        in_stopband = not (
            math.isfinite(alpha_p) and alpha_p <= ctx.stopband_alpha_threshold
        )

    z_in, z_out = edge_impedance(
        mtl, ctx.varactor, ctx.n_cells, substrate=ctx.substrate
    )
    sm_in = select_matching(z_in, "input", ctx.f_signal_Hz, substrate=ctx.substrate)
    sm_out = select_matching(z_out, "output", ctx.f_signal_Hz, substrate=ctx.substrate)
    im = None
    om = None
    in_match_topo = ""
    out_match_topo = ""
    if sm_in.needed:
        im = sm_in.single_stub if sm_in.recommended == "single_stub" else sm_in.quarter_wave
        in_match_topo = im.topology
    if sm_out.needed:
        om = sm_out.single_stub if sm_out.recommended == "single_stub" else sm_out.quarter_wave
        out_match_topo = om.topology
    loss_in = _matching_loss_dB(im, ctx.f_signal_Hz, ctx.truncation_N)
    loss_out = _matching_loss_dB(om, ctx.f_signal_Hz, ctx.truncation_N)

    spec_for_ana = optimized_self_pumped_spec(
        topology=cand.topology, params=cand.params, varactor=ctx.varactor,
        substrate=ctx.substrate, n_cells=ctx.n_cells, main_width_m=mtl.w_main_m,
        main_period_m=ctx.main_period_m,
        f_signal_Hz=ctx.f_signal_Hz, f_pump_Hz=ctx.f_pump_Hz,
        pump_power_dBm=ctx.pump_dBm_calibrated, pump_phase_rad=0.0,
        via_radius_m=ctx.via_radius_m, hb_order_signal=ctx.hb_order_signal,
        hb_order_pump=ctx.hb_order_pump, hb_max_order=ctx.hb_max_order,
        input_match=im, output_match=om,
        via_model=ctx.via_model, open_end_model=ctx.open_end_model,
    )
    ib = matching_section_harmonic_blocks(im, ctx.omega_n, ctx.tvtl_sub, side="input")
    ob = matching_section_harmonic_blocks(om, ctx.omega_n, ctx.tvtl_sub, side="output")
    stub_y_override = None
    if cand.topology.termination is Termination.COMBINED:
        stub_y_override = topology_y_stub_fn(
            cand.topology, cand.params_array, ctx.substrate,
            via_radius_m=ctx.via_radius_m,
        )
    try:
        res = level10_realistic_param_gain(
            spec_for_ana, ctx.varactor, m_branch_pump=ctx.m_branch_pump,
            truncation_N=ctx.truncation_N,
            input_match_blocks=ib, output_match_blocks=ob,
            via_model=ctx.via_model, open_end_model=ctx.open_end_model,
            stub_y_fn_override=stub_y_override,
        )
    except Exception as exc:  # noqa: BLE001 -- documented diagnostic only
        return _CandidateEval(
            ok=False, alpha_p_np=alpha_p, in_stopband=in_stopband,
            note=f"Stage 4 raised: {type(exc).__name__}: {exc}",
        )

    return _CandidateEval(
        ok=True,
        proxy_dB=analytical_g_mod_in_dB(res),
        alpha_p_np=alpha_p,
        in_stopband=in_stopband,
        coh_pk_dB=float(res["coherent_peak_dB"]),
        psa_swing_dB=float(res["coherent_swing_dB"]),
        depth_effective=float(res["depth_effective"]),
        z_tl_ohm=float(mtl.z_tl_ohm),
        w_main_m=float(mtl.w_main_m),
        z_in=complex(z_in),
        z_out=complex(z_out),
        loss_in_dB=loss_in,
        loss_out_dB=loss_out,
        input_match_topology=in_match_topo,
        output_match_topology=out_match_topo,
        mtl=mtl, im=im, om=om,
    )


def _passband_pool(
    screen,
    ctx: _EvalContext,
    *,
    walk_down_top_k: int,
    walk_cap: int,
    search_combined: bool,
) -> tuple[list[StubCandidate], dict[int, float]]:
    """Band-structure-aware walk-down pool (STEP 1).

    The Stage-1 A1 ranking is stopband-biased, so the raw top-K-by-A1 pool is
    dominated by Bloch-stopband cells (poor realized gain). Instead, walk **down**
    the full A1-ranked screen and collect the first ``walk_down_top_k`` candidates
    that are Stage-2-realizable AND whose signal sits in a passband (per-cell
    ``alpha_p <= stopband_alpha_threshold``), capped at ``walk_cap`` probes. This
    keeps the A1 preference order among passband cells while excluding the
    stopband cells the proxy would otherwise mis-select. Returns the pool plus a
    ``{id(cand): alpha_p}`` map so the walk-down loop need not recompute it.
    """

    ranked = [c for c in screen.scored if search_combined or c.topology.is_linear]
    pool: list[StubCandidate] = []
    alpha_map: dict[int, float] = {}
    for cand in ranked[: max(0, int(walk_cap))]:
        mtl, keep, _note = _stage2_realizable_or_tuner(cand, ctx)
        if not keep:
            continue
        alpha_p = _candidate_alpha_p(cand, mtl, ctx)
        if not (math.isfinite(alpha_p) and alpha_p <= ctx.stopband_alpha_threshold):
            continue
        pool.append(cand)
        alpha_map[id(cand)] = alpha_p
        if len(pool) >= max(0, int(walk_down_top_k)):
            break
    return pool, alpha_map


def _refine_objective(x: np.ndarray, topology: TopologyFamily, cand_f0: float, ctx: _EvalContext) -> float:
    """Nelder-Mead objective: ``-(passband-penalised g_mod/in proxy)``.

    A band-structure-penalised version of the realized-gain proxy: invalid
    geometry / Stage-2-drop / Stage-4-raise returns a large finite penalty (so the
    simplex stays well-behaved, as in :func:`tvtl.optimization.stage1_per_cell.refine_topk`),
    and each Np/cell of stopband incursion subtracts ``_STOPBAND_REFINE_PENALTY_DB_PER_NP``
    dB. Refining the **A1** figure of merit (what ``refine_topk`` maximises on the
    control-port path) would pull toward the A1 ceiling = the Bloch stopband,
    the exact failure the band screen fixes; refining the out/in proxy under the passband
    penalty instead drives the geometry into the passband and maximises realized
    gain there.
    """

    if not is_valid(topology, x, substrate=ctx.substrate, f_ref_hz=ctx.f_signal_Hz):
        return _REFINE_INVALID_PENALTY
    cand = StubCandidate(
        topology=topology, params=tuple(float(v) for v in x),
        a1_score=0.0, f0_Hz=cand_f0,
    )
    ev = _evaluate_candidate_proxy(cand, ctx)
    if not ev.ok or not math.isfinite(ev.proxy_dB):
        return _REFINE_INVALID_PENALTY
    penalty = _STOPBAND_REFINE_PENALTY_DB_PER_NP * max(
        0.0, ev.alpha_p_np - ctx.stopband_alpha_threshold
    ) if ctx.reject_signal_stopband and math.isfinite(ev.alpha_p_np) else 0.0
    return -(ev.proxy_dB - penalty)


def _refine_candidate_geometry(
    cand: StubCandidate, ctx: _EvalContext, *, maxiter: int
) -> StubCandidate:
    """Continuous Nelder-Mead refinement of one candidate's geometry.

    Ported from :func:`tvtl.optimization.stage1_per_cell.refine_topk` (same scipy
    ``Nelder-Mead`` over the family's ``(width, length)`` box, same keep-if-not-
    worse guard) but retargeted onto the Stage-4 passband-penalised out/in proxy
    (:func:`_refine_objective`) instead of the A1 figure of merit. Returns a
    refined :class:`StubCandidate` (the coarse geometry unchanged if refinement
    did not strictly improve the objective).
    """

    topo = cand.topology
    x0 = cand.params_array
    lo, hi = param_bounds(topo)
    objective = lambda x: _refine_objective(x, topo, float(cand.f0_Hz), ctx)  # noqa: E731
    result = minimize(
        objective, x0, method="Nelder-Mead", bounds=Bounds(lo, hi),
        options={"maxiter": int(maxiter), "xatol": 1e-7, "fatol": 1e-4},
    )
    x_best = np.asarray(result.x, dtype=float)
    s_start = -objective(x0)
    s_best = -objective(x_best)
    keep = (
        x_best if (math.isfinite(s_best) and s_best >= s_start) else x0
    )
    return StubCandidate(
        topology=topo, params=tuple(float(v) for v in keep),
        a1_score=float(cand.a1_score), f0_Hz=float(cand.f0_Hz),
    )


def _attempt_design(
    *,
    varactor: RealisticVaractor,
    f_signal_Hz: float,
    f_pump_Hz: float,
    n_cells: int,
    matching_loss_threshold_dB: float,
    coh_pk_window_dB: tuple[float, float],
    walk_down_top_k: int,
    substrate: SubstrateConfig,
    sobol_n_samples: int,
    sobol_seed: int,
    families: tuple[TopologyFamily, ...],
    a1_pump_reference_dBm: float,
    truncation_N: int,
    hb_order_signal: int,
    hb_order_pump: int,
    hb_max_order: int,
    via_radius_m: float,
    main_period_m: float,
    pump_dBm_calibrated: float,
    m_branch_pump: float,
    omega_n,
    tvtl_sub,
    families_attempt: int,
    objective: str = OBJECTIVE_COH_PK,
    match_low_z_with_tuner: bool = False,
    low_z_tuner_s11_target_dB: float = -10.0,
    search_combined: bool = False,
    reject_signal_stopband: bool = False,
    stopband_alpha_threshold: float = STOPBAND_ALPHA_THRESHOLD_NP_PER_CELL,
    band_screen_walk_cap: int = _BAND_SCREEN_WALK_CAP,
    refine: bool = False,
    refine_top_n: int = 3,
    refine_maxiter: int = 50,
    via_model=None,
    open_end_model=None,
) -> tuple[
    SelfPumpedDesignResult | None,
    tuple[_WalkdownEntry, ...],
]:
    """Run one walk-down attempt against a single families pool.

    Returns ``(result, trace)`` where ``result`` is None if no candidate
    passed both gates and ``trace`` is this attempt's per-candidate
    diagnostic rows. The outer :func:`design_optimized_self_pumped`
    accumulates traces across attempts and raises
    :class:`NoSelfPumpedCandidateError` only when every attempt fails.

    Opt-in controls (default OFF -> byte-identical):

    * ``reject_signal_stopband`` -- build the walk-down pool from the A1-ranked
      screen keeping only Bloch-*passband* candidates (:func:`_passband_pool`),
      and refuse to select a stopband candidate. Fixes the stopband mis-selection
      (the A1 ranking is stopband-biased).
    * ``refine`` -- after the coarse walk-down, Nelder-Mead-refine the top
      ``refine_top_n`` candidates' geometry against the passband-penalised out/in
      proxy (:func:`_refine_candidate_geometry`) and keep the best; applies under
      the ``g_mod_in`` objective only.
    """

    a1_fom = ClosedFormA1SelfPumped(
        f_pump_Hz=f_pump_Hz, pump_power_dBm=a1_pump_reference_dBm,
        via_radius_m=via_radius_m,
    )
    # Opt-in COMBINED search. When ``search_combined`` is set the
    # 8 MLEF∥VIAGND COMBINED families are screened ALONGSIDE the linear pool; the
    # screened-but-not-linear candidates then join the walk-down (below). Default
    # OFF -> ``screen_families`` is exactly ``families``, so the screen, the
    # ``linear_topk`` ordering, and every existing caller (incl. the A.3.1 anchor)
    # are byte-identical.
    screen_families = tuple(families)
    if search_combined:
        screen_families = screen_families + tuple(combined_families())
    screen = coarse_screen(
        f_signal_Hz, varactor, fom=a1_fom, substrate=substrate,
        families=list(screen_families), n_samples=sobol_n_samples, seed=sobol_seed,
        top_k=walk_down_top_k,
    )

    # Shared per-candidate evaluation context (consumed only by the opt-in band
    # screen + refinement; the coarse loop below stays inline / byte-identical).
    eval_ctx = _EvalContext(
        varactor=varactor, f_signal_Hz=f_signal_Hz, f_pump_Hz=f_pump_Hz,
        n_cells=n_cells, substrate=substrate, main_period_m=main_period_m,
        pump_dBm_calibrated=pump_dBm_calibrated, m_branch_pump=m_branch_pump,
        truncation_N=truncation_N, hb_order_signal=hb_order_signal,
        hb_order_pump=hb_order_pump, hb_max_order=hb_max_order,
        via_radius_m=via_radius_m, omega_n=omega_n, tvtl_sub=tvtl_sub,
        reject_signal_stopband=reject_signal_stopband,
        stopband_alpha_threshold=stopband_alpha_threshold,
        match_low_z_with_tuner=match_low_z_with_tuner,
        low_z_tuner_s11_target_dB=low_z_tuner_s11_target_dB,
        via_model=via_model, open_end_model=open_end_model,
    )

    # Walk-down pool.
    #
    # Default (``reject_signal_stopband=False``, byte-identical): the top-K linear
    # candidates plus, when COMBINED search is enabled, the top-K COMBINED
    # candidates the screen scored. With ``search_combined=False`` the pool is
    # exactly ``screen.linear_topk[:K]``.
    #
    # With ``reject_signal_stopband=True``: the A1 ranking is
    # stopband-biased, so instead walk DOWN the A1-ranked screen keeping the first
    # K passband-realizable candidates (:func:`_passband_pool`) -- the
    # low-passive-loss placement where the parametric boost can net positive.
    alpha_map: dict[int, float] = {}
    if reject_signal_stopband:
        pool, alpha_map = _passband_pool(
            screen, eval_ctx, walk_down_top_k=walk_down_top_k,
            walk_cap=band_screen_walk_cap, search_combined=search_combined,
        )
    else:
        pool = list(screen.linear_topk[:walk_down_top_k])
        if search_combined:
            combined_cands = [
                c for c in screen.scored if not c.topology.is_linear
            ][:walk_down_top_k]
            pool = pool + combined_cands

    trace: list[_WalkdownEntry] = []
    selected: _WalkdownEntry | None = None
    selected_cand: StubCandidate | None = None
    selected_mtl: MainTLDesign | None = None
    selected_im = None
    selected_om = None
    # G_mod/in objective bookkeeping (unused on the coh_pk path): the walk-down
    # selects the BEST out/in proxy across all candidates (not the first in a
    # window), so track the running best and finalize the SELECTED note after the
    # loop. selected_index points at the winning trace row.
    best_g_mod_in = float("-inf")
    selected_index: int | None = None
    # Refinement bookkeeping: under the g_mod_in objective with
    # ``refine`` on, collect the Stage-4-OK, passband-eligible candidates (best
    # proxy first) so the top ``refine_top_n`` can be geometry-refined after the
    # coarse pass. Empty / unused otherwise.
    refine_candidates: list[tuple[float, StubCandidate]] = []

    for rank, cand in enumerate(pool, 1):
        entry_kwargs = dict(
            families_attempt=families_attempt,
            rank=rank,
            topology_key=cand.topology.key,
            a1_score=float(cand.a1_score),
            stage2_realizable=False,
            z_tl_ohm=float("nan"),
            w_main_m=float("nan"),
            z_in_real=float("nan"),
            z_in_imag=float("nan"),
            z_out_real=float("nan"),
            z_out_imag=float("nan"),
            input_match_topology="",
            output_match_topology="",
            matching_loss_input_dB=float("nan"),
            matching_loss_output_dB=float("nan"),
            analytical_coh_pk_dB=float("nan"),
            analytical_psa_swing_dB=float("nan"),
            depth_effective=float("nan"),
            note="",
        )

        mtl = match_z_tl(cand, varactor, substrate=substrate, f0_Hz=f_signal_Hz)
        if not mtl.realizable:
            # Default: drop a candidate whose periodic line cannot be brought to
            # a 50-ohm Bloch impedance by the main width alone (the historical
            # Stage-2 gate). Opt-in: KEEP a low-Z
            # candidate when the cascade EDGE impedance is matchable by the full
            # single-stub tuner with a realizable main-line width -- the tuner,
            # not the main width, then does the 50-ohm transformation. This is
            # default-OFF so the A.3.1 anchor (and every existing caller) is
            # byte-identical; only an explicit match_low_z_with_tuner=True
            # search admits these designs.
            keep = False
            if match_low_z_with_tuner and math.isfinite(mtl.w_main_m):
                z_in_c, z_out_c = edge_impedance(
                    mtl, varactor, n_cells, substrate=substrate
                )
                ss_in = single_stub_design(
                    z_in_c, f_signal_Hz, substrate=substrate, stub_type="open"
                )
                ss_out = single_stub_design(
                    z_out_c, f_signal_Hz, substrate=substrate, stub_type="open"
                )
                keep = bool(
                    ss_in.realizable and ss_out.realizable
                    and ss_in.s11_after_dB <= low_z_tuner_s11_target_dB
                    and ss_out.s11_after_dB <= low_z_tuner_s11_target_dB
                )
            if not keep:
                entry_kwargs["note"] = (
                    "Stage 2: Z_TL not realizable"
                    + ("; edge not tuner-matchable" if match_low_z_with_tuner else "")
                )
                trace.append(_WalkdownEntry(**entry_kwargs))
                continue
            entry_kwargs["note"] = "Stage 2: low-Z, kept (edge tuner-matchable)"
        entry_kwargs["stage2_realizable"] = True
        entry_kwargs["z_tl_ohm"] = float(mtl.z_tl_ohm)
        entry_kwargs["w_main_m"] = float(mtl.w_main_m)

        # Band-structure screen: per-cell Bloch attenuation
        # alpha_p at f_signal (reused from the passband-pool walk when available).
        # in_stopband candidates are kept in the trace but cannot be SELECTED.
        in_stopband = False
        if reject_signal_stopband:
            alpha_p = alpha_map.get(id(cand))
            if alpha_p is None:
                alpha_p = _candidate_alpha_p(cand, mtl, eval_ctx)
            entry_kwargs["bloch_alpha_p_np"] = float(alpha_p)
            in_stopband = not (
                math.isfinite(alpha_p) and alpha_p <= stopband_alpha_threshold
            )

        z_in, z_out = edge_impedance(mtl, varactor, n_cells, substrate=substrate)
        entry_kwargs["z_in_real"] = float(z_in.real)
        entry_kwargs["z_in_imag"] = float(z_in.imag)
        entry_kwargs["z_out_real"] = float(z_out.real)
        entry_kwargs["z_out_imag"] = float(z_out.imag)

        sm_in = select_matching(z_in, "input", f_signal_Hz, substrate=substrate)
        sm_out = select_matching(z_out, "output", f_signal_Hz, substrate=substrate)
        im = None
        om = None
        if sm_in.needed:
            im = sm_in.single_stub if sm_in.recommended == "single_stub" else sm_in.quarter_wave
            entry_kwargs["input_match_topology"] = im.topology
        if sm_out.needed:
            om = sm_out.single_stub if sm_out.recommended == "single_stub" else sm_out.quarter_wave
            entry_kwargs["output_match_topology"] = om.topology
        loss_in = _matching_loss_dB(im, f_signal_Hz, truncation_N)
        loss_out = _matching_loss_dB(om, f_signal_Hz, truncation_N)
        entry_kwargs["matching_loss_input_dB"] = loss_in
        entry_kwargs["matching_loss_output_dB"] = loss_out

        spec_for_ana = optimized_self_pumped_spec(
            topology=cand.topology, params=cand.params, varactor=varactor,
            substrate=substrate, n_cells=n_cells, main_width_m=mtl.w_main_m,
            main_period_m=main_period_m,
            f_signal_Hz=f_signal_Hz, f_pump_Hz=f_pump_Hz,
            pump_power_dBm=pump_dBm_calibrated, pump_phase_rad=0.0,
            via_radius_m=via_radius_m, hb_order_signal=hb_order_signal,
            hb_order_pump=hb_order_pump, hb_max_order=hb_max_order,
            input_match=im, output_match=om,
            via_model=via_model, open_end_model=open_end_model,
        )
        ib = matching_section_harmonic_blocks(im, omega_n, tvtl_sub, side="input")
        ob = matching_section_harmonic_blocks(om, omega_n, tvtl_sub, side="output")
        # COMBINED candidates: hand the Stage-4 walker the analytical shunt
        # admittance of the parallel via||open-stublet trap (the same
        # topology_y_stub_fn the Stage 1-3 screen used for this candidate), so the
        # analytical leg traces the COMBINED termination instead of the single-chain
        # spec extraction (which cannot). SHORT/OPEN pass None (byte-unchanged).
        stub_y_override = None
        if cand.topology.termination is Termination.COMBINED:
            stub_y_override = topology_y_stub_fn(
                cand.topology, cand.params_array, substrate,
                via_radius_m=via_radius_m,
            )
        try:
            res = level10_realistic_param_gain(
                spec_for_ana, varactor, m_branch_pump=m_branch_pump,
                truncation_N=truncation_N,
                input_match_blocks=ib, output_match_blocks=ob,
                via_model=via_model, open_end_model=open_end_model,
                stub_y_fn_override=stub_y_override,
            )
        except Exception as exc:  # noqa: BLE001 -- documented diagnostic only
            entry_kwargs["note"] = f"Stage 4 raised: {type(exc).__name__}: {exc}"
            trace.append(_WalkdownEntry(**entry_kwargs))
            continue
        entry_kwargs["analytical_coh_pk_dB"] = float(res["coherent_peak_dB"])
        entry_kwargs["analytical_psa_swing_dB"] = float(res["coherent_swing_dB"])
        entry_kwargs["depth_effective"] = float(res["depth_effective"])

        if objective == OBJECTIVE_COH_PK:
            # coh_pk objective (default, byte-unchanged): the first candidate in
            # Sobol-rank order whose analytical coh_pk is inside the window AND
            # whose matching insertion loss is below threshold on both sides.
            match_loss_ok = (
                loss_in < matching_loss_threshold_dB
                and loss_out < matching_loss_threshold_dB
            )
            coh = float(res["coherent_peak_dB"])
            in_window = math.isfinite(coh) and (
                coh_pk_window_dB[0] <= coh <= coh_pk_window_dB[1]
            )
            if selected is None and match_loss_ok and in_window and not in_stopband:
                entry_kwargs["note"] = "SELECTED"
                entry = _WalkdownEntry(**entry_kwargs)
                selected = entry
                selected_cand = cand
                selected_mtl = mtl
                selected_im = im
                selected_om = om
            else:
                why: list[str] = []
                if not match_loss_ok:
                    why.append(
                        f"matching loss [{loss_in:.2f}, {loss_out:.2f}] dB "
                        f"exceeds {matching_loss_threshold_dB} dB"
                    )
                if not in_window:
                    why.append(
                        f"coh_pk {coh:.2f} outside window {coh_pk_window_dB}"
                    )
                if in_stopband:
                    why.append(
                        f"signal in Bloch stopband (alpha_p="
                        f"{entry_kwargs['bloch_alpha_p_np']:.4f} > "
                        f"{stopband_alpha_threshold} Np/cell)"
                    )
                entry_kwargs["note"] = "; ".join(why) or "not selected"
                entry = _WalkdownEntry(**entry_kwargs)
            trace.append(entry)
        else:
            # G_mod/in objective: MAXIMISE the analytical out/in transducer-gain
            # proxy (abs_gain_off + coh_pk) across all Stage-4-OK candidates. The
            # passive |S21| in abs_gain_off already carries the passband/stopband
            # loss AND the matching loss, so the proxy is the complete in-loop
            # selection signal -- no separate matching-loss window gate (that gate
            # exists for coh_pk precisely because coh_pk omits the passive loss).
            g_mod_in = analytical_g_mod_in_dB(res)
            entry_kwargs["analytical_g_mod_in_dB"] = g_mod_in
            # A stopband candidate is never SELECTED (band screen). With the
            # passband pool it is already excluded; the gate is the defensive net.
            eligible = math.isfinite(g_mod_in) and not in_stopband
            if refine and eligible:
                refine_candidates.append((g_mod_in, cand))
            if eligible and g_mod_in > best_g_mod_in:
                best_g_mod_in = g_mod_in
                entry_kwargs["note"] = f"best-so-far (G_mod/in {g_mod_in:+.2f} dB)"
                entry = _WalkdownEntry(**entry_kwargs)
                selected = entry
                selected_cand = cand
                selected_mtl = mtl
                selected_im = im
                selected_om = om
                selected_index = len(trace)
            else:
                if in_stopband and math.isfinite(g_mod_in):
                    entry_kwargs["note"] = (
                        f"G_mod/in {g_mod_in:+.2f} dB but signal in Bloch stopband "
                        f"(alpha_p={entry_kwargs['bloch_alpha_p_np']:.4f} > "
                        f"{stopband_alpha_threshold} Np/cell) -- not eligible"
                    )
                else:
                    entry_kwargs["note"] = (
                        f"G_mod/in {g_mod_in:+.2f} dB (not best)"
                        if math.isfinite(g_mod_in)
                        else "G_mod/in proxy not finite"
                    )
                entry = _WalkdownEntry(**entry_kwargs)
            trace.append(entry)

    if selected is None:
        return None, tuple(trace)

    # ------------------------------------------------------------------ #
    # Nelder-Mead geometry refinement (opt-in).                           #
    # ------------------------------------------------------------------ #
    # Refine the top ``refine_top_n`` coarse-selected candidates' continuous
    # geometry against the passband-penalised out/in proxy and adopt a refined
    # design only if it STRICTLY beats the coarse selection -- so refinement is
    # purely additive (default off / no improvement -> coarse selection stands).
    refined_selection = False
    if refine and objective == OBJECTIVE_G_MOD_IN and refine_candidates:
        top = sorted(refine_candidates, key=lambda t: t[0], reverse=True)[
            : max(0, int(refine_top_n))
        ]
        best_refined_eval: _CandidateEval | None = None
        best_refined_cand: StubCandidate | None = None
        for _proxy, cand in top:
            refined_cand = _refine_candidate_geometry(
                cand, eval_ctx, maxiter=refine_maxiter
            )
            ev = _evaluate_candidate_proxy(refined_cand, eval_ctx)
            if not ev.ok or ev.in_stopband or not math.isfinite(ev.proxy_dB):
                continue
            if best_refined_eval is None or ev.proxy_dB > best_refined_eval.proxy_dB:
                best_refined_eval = ev
                best_refined_cand = refined_cand
        if (
            best_refined_eval is not None
            and best_refined_cand is not None
            and best_refined_eval.proxy_dB > best_g_mod_in
        ):
            ev = best_refined_eval
            refined_entry = _WalkdownEntry(
                families_attempt=families_attempt,
                rank=len(pool) + 1,
                topology_key=best_refined_cand.topology.key,
                a1_score=float(best_refined_cand.a1_score),
                stage2_realizable=True,
                z_tl_ohm=ev.z_tl_ohm,
                w_main_m=ev.w_main_m,
                z_in_real=float(ev.z_in.real),
                z_in_imag=float(ev.z_in.imag),
                z_out_real=float(ev.z_out.real),
                z_out_imag=float(ev.z_out.imag),
                input_match_topology=ev.input_match_topology,
                output_match_topology=ev.output_match_topology,
                matching_loss_input_dB=ev.loss_in_dB,
                matching_loss_output_dB=ev.loss_out_dB,
                analytical_coh_pk_dB=ev.coh_pk_dB,
                analytical_psa_swing_dB=ev.psa_swing_dB,
                depth_effective=ev.depth_effective,
                note=(
                    f"SELECTED (refined: G_mod/in {ev.proxy_dB:+.2f} dB vs coarse "
                    f"{best_g_mod_in:+.2f} dB)"
                ),
                analytical_g_mod_in_dB=ev.proxy_dB,
                bloch_alpha_p_np=ev.alpha_p_np,
            )
            trace.append(refined_entry)
            selected = refined_entry
            selected_cand = best_refined_cand
            selected_mtl = ev.mtl
            selected_im = ev.im
            selected_om = ev.om
            best_g_mod_in = ev.proxy_dB
            refined_selection = True

    # G_mod/in objective: stamp the winning row's note now that the full ladder
    # has been scored (coh_pk's first-in-window winner is already noted inline).
    # If refinement replaced the selection, its row is already noted/selected.
    if (
        objective == OBJECTIVE_G_MOD_IN
        and selected_index is not None
        and not refined_selection
    ):
        trace[selected_index] = dataclasses.replace(
            trace[selected_index], note="SELECTED",
        )
        selected = trace[selected_index]

    assert selected_cand is not None and selected_mtl is not None
    final_spec = optimized_self_pumped_spec(
        topology=selected_cand.topology, params=selected_cand.params,
        varactor=varactor, substrate=substrate, n_cells=n_cells,
        main_width_m=selected_mtl.w_main_m,
        main_period_m=main_period_m,
        f_signal_Hz=f_signal_Hz, f_pump_Hz=f_pump_Hz,
        pump_power_dBm=pump_dBm_calibrated, pump_phase_rad=0.0,
        via_radius_m=via_radius_m, hb_order_signal=hb_order_signal,
        hb_order_pump=hb_order_pump, hb_max_order=hb_max_order,
        input_match=selected_im, output_match=selected_om,
        via_model=via_model, open_end_model=open_end_model,
    )
    result = SelfPumpedDesignResult(
        spec=final_spec,
        calibrated_pump_dBm=pump_dBm_calibrated,
        topology=selected_cand.topology,
        stub_params=tuple(float(p) for p in selected_cand.params),
        main_width_m=float(selected_mtl.w_main_m),
        main_period_m=float(main_period_m),
        z_tl_ohm=float(selected_mtl.z_tl_ohm),
        input_match=selected_im,
        output_match=selected_om,
        matching_loss_input_dB=selected.matching_loss_input_dB,
        matching_loss_output_dB=selected.matching_loss_output_dB,
        analytical_coh_pk_dB=selected.analytical_coh_pk_dB,
        analytical_psa_swing_dB=selected.analytical_psa_swing_dB,
        depth_effective=selected.depth_effective,
        a1_score=selected.a1_score,
        sobol_rank=selected.rank,
        families_used=families_attempt,
        walkdown_trace=tuple(trace),
        objective=objective,
        analytical_g_mod_in_dB=selected.analytical_g_mod_in_dB,
    )
    return result, tuple(trace)


def design_optimized_self_pumped(
    *,
    varactor: RealisticVaractor,
    f_signal_Hz: float,
    f_pump_Hz: float | None = None,
    n_cells: int = 15,
    m_eff_target: float = 0.22,
    matching_loss_threshold_dB: float = 3.0,
    coh_pk_window_dB: tuple[float, float] = (1.0, 10.0),
    walk_down_top_k: int = 10,
    substrate: SubstrateConfig = ROGERS_4003C,
    sobol_n_samples: int = 300,
    sobol_seed: int = 0,
    families_fallback: tuple[tuple[TopologyFamily, ...], ...] | None = None,
    a1_pump_reference_dBm: float = 16.0,
    truncation_N: int = 6,
    hb_order_signal: int = 5,
    hb_order_pump: int = 5,
    hb_max_order: int = 6,
    via_radius_m: float = VIA_RADIUS_M,
    main_period_scaling_fn: Callable[[float], float] = main_period_anchor_at_3p1GHz,
    objective: str = OBJECTIVE_G_MOD_IN,
    match_low_z_with_tuner: bool = False,
    low_z_tuner_s11_target_dB: float = -10.0,
    search_combined: bool = False,
    reject_signal_stopband: bool = False,
    stopband_alpha_threshold: float = STOPBAND_ALPHA_THRESHOLD_NP_PER_CELL,
    band_screen_walk_cap: int = _BAND_SCREEN_WALK_CAP,
    refine: bool = False,
    refine_top_n: int = 3,
    refine_maxiter: int = 50,
    pump_dBm_override: float | None = None,
    via_model=None,
    open_end_model=None,
) -> SelfPumpedDesignResult:
    """High-level self-pumped design: Sobol screen + Stage 2-4 + spec build.

    Sequence (the control-port workflow transferred to self-pumped and
    generalized):

    1. **Calibrate source pump_dBm** so the junction
       ``m_eff = 2|C_1|/C_0`` matches ``m_eff_target`` (small-signal
       regime by default; m_eff_target=0.22 matches the control-port
       reference equivalent at -3 dBm).
    2. **Scale main-line period** via ``main_period_scaling_fn`` so the
       per-cell electrical length stays comparable across the
       design set (default :func:`main_period_anchor_at_3p1GHz`; anchored
       at 6.11 mm for f_signal = 3.1 GHz, inversely scaled for other
       frequencies).
    3. **For each entry in ``families_fallback``** (in order):

       a. **Stage 1 (Sobol screen)** with :class:`ClosedFormA1SelfPumped`
          over that entry's families.
       b. For each top-``walk_down_top_k`` candidate: Stage 2
          :func:`match_z_tl`, Stage 3 :func:`edge_impedance` +
          :func:`select_matching`, Stage 4 analytical
          :func:`level10_realistic_param_gain` self-pumped under
          CORRECTED_NOTEBOOK.
       c. **Selection**: governed by ``objective`` (see below). The
          default ``g_mod_in`` objective selects the candidate with the
          best analytical out/in transducer-gain proxy; the historical
          ``coh_pk`` objective selects the first candidate whose matching
          insertion loss is below ``matching_loss_threshold_dB`` on BOTH
          sides AND whose analytical coh_pk falls within ``coh_pk_window_dB``.
       d. On success: stop, return the result with
          ``families_used = <attempt index>``.
       e. On failure: append the per-candidate trace to the running
          aggregate, move to the next ``families_fallback`` entry.

    4. If every attempt fails: raise
       :class:`NoSelfPumpedCandidateError` with the full concatenated
       walk-down trace.

    5. Build the self-pumped :class:`DeviceSpec` via
       :func:`optimized_self_pumped_spec` at the calibrated pump and
       scaled period.

    Objective
    ---------
    ``objective`` selects what the walk-down maximises. The **default is**
    ``OBJECTIVE_G_MOD_IN``:
    the out/in transducer gain is the physically-meaningful selection
    signal, whereas the historical ``coh_pk`` ratio is inflated by passive
    suppression and steers the search toward Bloch-stopband placement (the
    worst configuration for net transducer gain). ``coh_pk`` remains fully
    selectable (``objective="coh_pk"``) and its selection path is
    byte-unchanged when chosen explicitly -- e.g. the A.3.1 anchor deck is
    reproduced bit-for-bit under ``objective="coh_pk"``.

    * ``"coh_pk"`` -- the historical objective: the first Sobol-rank
      candidate whose analytical coherent-peak gain falls in
      ``coh_pk_window_dB`` and whose matching loss is below threshold.
      This ratio metric is inflated by passive suppression, so it
      favours Bloch-stopband placement.
    * ``"g_mod_in"`` (default) -- the out/in **transducer** gain
      (``abs_gain_off + coh_pk`` =
      :func:`tvtl_ads_builder_v2.parity.gain_objective.analytical_g_mod_in_dB`):
      the candidate with the **best** analytical out/in proxy is
      selected (the passive |S21| already carries the passband/matching
      loss, so this favours low-loss passband placement, not stopband
      ratio inflation). The realised value + stability are then measured
      from ADS HB by
      :func:`tvtl_ads_builder_v2.parity.gain_objective.measure_g_mod_in`;
      ``coh_pk_window_dB`` is ignored under this objective.

    Defaults
    --------
    * ``coh_pk_window_dB = (1.0, 10.0)`` -- widened from the historical
      [1, 5] dB control-port window. Under
      self-pumped, the analytical leg under-predicts ADS HB by ~5 dB
      due to the uniform-pump approximation (standing-wave
      conservatism); the [1, 10] dB analytical window corresponds to
      roughly [6, 15] dB ADS-realistic gain, the target
      parametric-amplifier band.
    * ``families_fallback = None`` -> default
      ``(SHORT_FAMILIES, SHORT_FAMILIES + OPEN_FAMILIES)``: primary
      SHORT-only attempt, fallback expands to SHORT+OPEN if SHORT
      alone produces no candidate.
    * ``main_period_scaling_fn = main_period_anchor_at_3p1GHz``:
      preserves A.3.1 bit-for-bit; restores Stage 2 realizability at
      higher signal frequencies.

    COMBINED search
    ---------------
    ``search_combined`` (default ``False``) extends the search to the 8
    MLEF∥VIAGND COMBINED topology families. When enabled, the COMBINED
    families are screened alongside the linear pool (Stage 1) and the
    top-K COMBINED candidates join the Stage 2-4 walk-down. The Stage-4
    analytical leg traces each COMBINED candidate's parallel via||open-stublet
    termination via
    :func:`tvtl.optimization.topology_generator.topology_y_stub_fn` (the
    same termination admittance Stage 1-3 already screen COMBINED with).
    Default ``False`` keeps the walk-down pool exactly ``linear_topk`` so the
    A.3.1 anchor and every existing caller are byte-identical.

    Pump as a search variable
    -------------------------
    ``pump_dBm_override`` (default ``None``) makes the source pump power an
    opt-in search variable. With ``None`` the source pump is calibrated to the
    target junction ``m_eff`` exactly as before (byte-identical; the A.3.1
    anchor recovery is unchanged). With an explicit value the calibration is
    bypassed and that source pump (dBm) is used directly -- this is how the
    expanded search
    (:func:`tvtl_ads_builder_v2.parity.expanded_search.design_optimized_self_pumped_search`)
    sweeps a pump ladder. Because the analytical ``g_mod_in`` proxy rises with
    pump and cannot see oscillation, the expanded search pairs this lever with
    an in-loop stability constraint (regime classification on the finalists); a
    bare pump sweep through this argument carries no such guard.

    Band-structure screen + refinement
    ----------------------------------
    Two opt-in controls (default ``False`` -> byte-identical; under the
    ``g_mod_in`` objective they fix the stopband mis-selection):

    * ``reject_signal_stopband`` -- the Stage-1 A1 ranking is *stopband-biased*
      (high A1 correlates with the band-edge / heavily-loaded cell), so the raw
      top-K-by-A1 walk-down pool is dominated by Bloch-stopband cells whose
      passive cascade deeply suppresses the signal -> poor realized transducer
      gain (an unscreened search saw -35 dB). With this ON, the walk-down pool is built by
      walking DOWN the A1-ranked screen and keeping the first ``walk_down_top_k``
      candidates whose signal sits in a Bloch PASSBAND (per-cell attenuation
      ``alpha_p <= stopband_alpha_threshold``, default 0.02 Np/cell, the
      net-gain propagating threshold), capped at
      ``band_screen_walk_cap`` probes; a stopband candidate can never be
      SELECTED. ``alpha_p`` is recorded per walk-down row
      (``_WalkdownEntry.bloch_alpha_p_np``).
    * ``refine`` -- after the coarse walk-down, Nelder-Mead-refine the top
      ``refine_top_n`` candidates' continuous geometry (``refine_maxiter``
      iterations each) against the passband-penalised out/in proxy, adopting a
      refined design only if it strictly beats the coarse selection. The
      refinement reuses the
      :func:`tvtl.optimization.stage1_per_cell.refine_topk` Nelder-Mead machinery
      but maximises the Stage-4 realized-gain proxy under the passband penalty
      rather than the A1 figure of merit (refining A1 would pull the geometry
      toward the stopband ceiling -- the exact failure ``reject_signal_stopband``
      corrects). Applies under the ``g_mod_in`` objective only.

    The expanded search
    (:func:`tvtl_ads_builder_v2.parity.expanded_search.design_optimized_self_pumped_search`)
    forwards both flags verbatim; enabling them there is how the geometry screen
    at an arbitrary operating frequency selects a passband design with good
    realized gain.

    Reproducibility
    ---------------
    At ``(varactor=Varactor_A, f_signal_Hz=3.1e9, objective="coh_pk",
    coh_pk_window_dB=(1.0, 5.0))`` this function reproduces the
    A.3.1 selection bit-for-bit (locked by
    ``tests/test_self_pumped_design.py::test_optimized_self_pumped_spec_a3p1_reproduction``).
    The A.3.1 selection (analytical coh_pk = +4.88 dB) is a property of
    the ``coh_pk`` objective; the default ``g_mod_in`` objective selects
    by the out/in proxy and generally picks a different design. The
    period-scaling default at f_signal = 3.1 GHz is exactly 6.11 mm,
    matching the published geometry under either objective.
    """

    objective = validate_objective(objective)
    f_pump = float(f_pump_Hz) if f_pump_Hz is not None else 2.0 * float(f_signal_Hz)

    # 1. Pump operating point (families-agnostic; fixed once across all fallback
    #    attempts). By default (``pump_dBm_override is None``) the source pump is
    #    CALIBRATED to the target junction ``m_eff`` -- byte-identical to the
    #    historical path. ``pump_dBm_override`` makes pump power an opt-in
    #    SEARCH variable: a caller (the expanded search,
    #    :func:`tvtl_ads_builder_v2.parity.expanded_search.design_optimized_self_pumped_search`)
    #    supplies an explicit source pump in dBm via ``pump_dBm_override``,
    #    bypassing calibration so the operating point is the searched value. When
    #    None the calibration -- and therefore the A.3.1 anchor recovery -- is
    #    unchanged.
    if pump_dBm_override is None:
        pump_dBm_calibrated = calibrate_self_pumped_pump_dBm(
            varactor=varactor, f_pump_Hz=f_pump, m_eff_target=m_eff_target,
        )
    else:
        pump_dBm_calibrated = float(pump_dBm_override)
    m_branch_pump = m_branch_from_source_pump_dBm(pump_dBm_calibrated, varactor)

    # 2. Main-line period (constant across the design; baked into specs).
    main_period_m = float(main_period_scaling_fn(f_signal_Hz))

    # 3. Pre-compute analytical-leg matching omega grid + substrate.
    ws = 2.0 * math.pi * float(f_signal_Hz)
    wm = 2.0 * math.pi * f_pump
    omega_n = harmonic_frequencies(ws, wm, truncation_N)
    tvtl_sub = substrate.to_tvtl_substrate()

    # 4. Families fallback loop.
    if families_fallback is None:
        families_fallback = _default_families_fallback()

    aggregate_trace: list[_WalkdownEntry] = []
    for attempt_idx, fams in enumerate(families_fallback, 1):
        result, attempt_trace = _attempt_design(
            varactor=varactor, f_signal_Hz=f_signal_Hz, f_pump_Hz=f_pump,
            n_cells=n_cells,
            matching_loss_threshold_dB=matching_loss_threshold_dB,
            coh_pk_window_dB=coh_pk_window_dB,
            walk_down_top_k=walk_down_top_k, substrate=substrate,
            sobol_n_samples=sobol_n_samples, sobol_seed=sobol_seed,
            families=tuple(fams),
            a1_pump_reference_dBm=a1_pump_reference_dBm,
            truncation_N=truncation_N,
            hb_order_signal=hb_order_signal, hb_order_pump=hb_order_pump,
            hb_max_order=hb_max_order, via_radius_m=via_radius_m,
            main_period_m=main_period_m,
            pump_dBm_calibrated=pump_dBm_calibrated,
            m_branch_pump=m_branch_pump,
            omega_n=omega_n, tvtl_sub=tvtl_sub,
            families_attempt=attempt_idx,
            objective=objective,
            match_low_z_with_tuner=match_low_z_with_tuner,
            low_z_tuner_s11_target_dB=low_z_tuner_s11_target_dB,
            search_combined=search_combined,
            reject_signal_stopband=reject_signal_stopband,
            stopband_alpha_threshold=stopband_alpha_threshold,
            band_screen_walk_cap=band_screen_walk_cap,
            refine=refine, refine_top_n=refine_top_n,
            refine_maxiter=refine_maxiter,
            via_model=via_model, open_end_model=open_end_model,
        )
        aggregate_trace.extend(attempt_trace)
        if result is not None:
            # Replace the per-attempt walkdown_trace with the aggregate
            # so callers see the full history (prior failed attempts +
            # the successful attempt).
            return dataclasses.replace(
                result, walkdown_trace=tuple(aggregate_trace),
            )

    n_attempts = len(families_fallback)
    raise NoSelfPumpedCandidateError(
        f"All {n_attempts} families_fallback attempts failed. No candidate "
        f"passed the matching-loss threshold (<{matching_loss_threshold_dB} "
        f"dB on both sides) AND the gain window {coh_pk_window_dB} dB. See "
        "walkdown_trace for per-attempt per-candidate diagnostics (each "
        "entry's families_attempt field indicates which attempt produced it).",
        walkdown_trace=tuple(aggregate_trace),
    )


