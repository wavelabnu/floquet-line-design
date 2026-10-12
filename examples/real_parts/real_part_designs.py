"""Frozen real-part example designs (SMV1245 / SMV2020 / MAVR-000350).

Three worked, paper-scale varactor-loaded TVTL designs built entirely from the
public ``tvtl`` physics core and the public ``tvtl_ads_builder_v2`` ADS builder.
Every geometry literal in this module is a FROZEN design point -- the commercial
varactor SPICE parameters come from :mod:`tvtl.varactor_palette` (datasheet
provenance lives there), and the microstrip geometry / cell count / operating
point are fixed constants below. Nothing here runs a search or reads an external
spreadsheet; the designs are reproducible offline.

The three examples (in increasing termination complexity):

* ``fabricated_SMV1245`` -- an as-built fabricated board: a 3-section OPEN/MLEF
  shunt stub with TWO grounding vias spliced at the *intermediate* stub junctions
  (a non-uniform shunt the topology families cannot express directly), 15 cells,
  f_s = 3.1 GHz, self-pumped, 0 V bias.
* ``real_oscillator_SMV2020`` -- a single-via SHORT/VIAGND cell, 24 cells,
  f_s = 2.0 GHz, 16 V bias; a near-oscillation regenerative operating point.
* ``real_amplifier_MAVR`` -- a COMBINED (MLEF in parallel with VIAGND) resonant
  trap, 24 cells, f_s = 3.0 GHz, 12 V bias; a stable net-positive amplifier.

Each design is built from pre-decided geometry via
:func:`tvtl_ads_builder_v2.parity.self_pumped_design.optimized_self_pumped_spec`
(the SMV2020 / MAVR cells additionally synthesise their input/output matching
analytically through the Stage-2/3 helpers in :mod:`tvtl.optimization`). The
SMV1245 board adds its two intermediate-node vias by splicing ``VIAGND``
elements into the emitted cell.

``EXAMPLES`` lists the three designs with the library / top-cell names used by
the hierarchical OpenAccess emitter. ``REFERENCE_VOUT_SIGNAL`` records the
signal-frequency output-node voltage each design must reproduce under harmonic
balance (the parity gate; see ``build_examples.py``).
"""
from __future__ import annotations

import dataclasses

from tvtl.optimization.stage1_per_cell import StubCandidate
from tvtl.optimization.stage2_main_tl import match_z_tl
from tvtl.optimization.stage3_matching import edge_impedance, select_matching
from tvtl.optimization.substrate import ROGERS_4003C
from tvtl.optimization.topology_generator import (
    Termination,
    TopologyFamily,
    ViaConfig,
)
from tvtl.varactor_palette import get_entry
from tvtl_ads_builder_v2.device_spec import (
    DeviceSpec,
    Element,
    ElementPin,
    resolve_spec,
)
from tvtl_ads_builder_v2.models import MlefOpen, ViagndVia
from tvtl_ads_builder_v2.parity.self_pumped_design import optimized_self_pumped_spec

__all__ = [
    "SUBSTRATE",
    "build_fixed_geom",
    "spec_smv1245",
    "spec_smv2020",
    "spec_mavr",
    "resolved_spec",
    "EXAMPLES",
    "REFERENCE_VOUT_SIGNAL",
]

# --------------------------------------------------------------------------- #
# Shared build constants
# --------------------------------------------------------------------------- #
SUBSTRATE = ROGERS_4003C
_Z_TL_LO_OHM = 15.0
_Z_TL_HI_OHM = 160.0
_N_CELLS_DEFAULT = 24
_PUMP_DBM = 29.0

# Accurate distributed terminations. The fixed-geometry search cells use a
# 0.75 mm / 17.5 um / Rho=1.0 physical via; the fabricated SMV1245 board uses
# its as-built 0.8 mm via with a measured relative resistivity.
_VIA_SEARCH = ViagndVia(via_radius_m=0.75e-3, metal_thickness_m=17.5e-6, rho_relative=1.0)
_VIA_BOARD = ViagndVia(via_radius_m=0.8e-3, metal_thickness_m=17.5e-6, rho_relative=0.706613906162)
_OPEN = MlefOpen()


def _matcher(sm):
    """Pick the recommended matching block from a matching solution, or None."""
    if not sm.needed:
        return None
    return sm.single_stub if sm.recommended == "single_stub" else sm.quarter_wave


def build_fixed_geom(geom, dev, f_s, pump_dBm=_PUMP_DBM, n_cells=_N_CELLS_DEFAULT):
    """Build a self-pumped cascade for a FIXED cell geometry under accurate models.

    ``geom`` is ``{"topology": TopologyFamily, "stub_params": tuple,
    "main_width_m": float, "main_period_m": float}``. The main-line impedance is
    matched (Stage 2) and the cascade edge impedances are synthesised into
    input/output single-stub or quarter-wave matches (Stage 3) analytically; the
    geometry itself is held fixed. The varactor shunt uses the accurate physical
    VIAGND short / MLEF open-end fringe.
    """
    cand = StubCandidate(
        topology=geom["topology"], params=tuple(geom["stub_params"]),
        a1_score=0.0, f0_Hz=f_s,
    )
    mtl = match_z_tl(cand, dev, substrate=SUBSTRATE, f0_Hz=f_s,
                     z_lo_ohm=_Z_TL_LO_OHM, z_hi_ohm=_Z_TL_HI_OHM)
    z_in, z_out = edge_impedance(mtl, dev, n_cells, substrate=SUBSTRATE)
    im = _matcher(select_matching(z_in, "input", f_s, substrate=SUBSTRATE))
    om = _matcher(select_matching(z_out, "output", f_s, substrate=SUBSTRATE))
    return optimized_self_pumped_spec(
        topology=geom["topology"], params=tuple(geom["stub_params"]), varactor=dev,
        substrate=SUBSTRATE, n_cells=n_cells, main_width_m=geom["main_width_m"],
        main_period_m=geom["main_period_m"], f_signal_Hz=f_s, f_pump_Hz=2.0 * f_s,
        pump_power_dBm=pump_dBm, pump_phase_rad=0.0, input_match=im, output_match=om,
        via_model=_VIA_SEARCH, open_end_model=_OPEN)


# --------------------------------------------------------------------------- #
# Example 1 -- the fabricated SMV1245 board (OPEN/MLEF 3-section + 2 mid-stub VIAGND)
# --------------------------------------------------------------------------- #
# As-built geometry (metres). The shunt is a 3-section microstrip stub
# [w0,l0, w1,l1, w2,l2] = sec2, sec3, sec4; sections 2 and 3 each end in a
# grounding via, section 4 ends OPEN (MLEF fringe).
_SMV1245_MAIN_W_M = 4.27e-3
_SMV1245_MAIN_PERIOD_M = 6.11e-3
_SMV1245_VIA_RADIUS_M = 0.8e-3
_SMV1245_STUB_PARAMS_M = (1.93e-3, 6.83e-3, 1.41e-3, 5.25e-3, 1.98e-3, 1.04e-3)
_SMV1245_SEC_WIDTHS_M = (1.93e-3, 1.41e-3, 1.98e-3)  # sec2, sec3, sec4 widths
_SMV1245_TOPO = TopologyFamily(n_segments=3, via=ViaConfig.NONE, termination=Termination.OPEN)
_SMV1245_N_CELLS = 15
_SMV1245_F_S_HZ = 3.1e9
_SMV1245_SIGNAL_DBM = -20.0  # the measured operating input power


def _substrate_name(spec):
    for e in spec.elements:
        if e.type == "MLIN":
            return e.parameters.get("substrate_ref", "msub1")
    return "msub1"


def _viagnd(eid, node, w_m, substrate_name):
    """One ADS VIAGND (1-pin element that grounds internally), board via geometry."""
    params = {"d_m": 2.0 * _SMV1245_VIA_RADIUS_M, "w_m": float(w_m),
              "t_m": float(_VIA_BOARD.metal_thickness_m), "rho": float(_VIA_BOARD.rho_relative),
              "substrate_ref": substrate_name}
    return Element(id=eid, type="VIAGND", pins=(ElementPin(pin=1, node=node),), parameters=params)


def spec_smv1245(n_cells=_SMV1245_N_CELLS, v_bias_v=0.0, pump_dBm=_PUMP_DBM) -> DeviceSpec:
    """The faithful as-built SMV1245 board: OPEN/MLEF 3-section stub + 2 mid-stub VIAGND.

    Built from the standard OPEN ``n3_none_open`` cell, then one ``VIAGND`` spliced
    at the far end of section 2 (node ``n_c{k}_s1``) and one at the far end of
    section 3 (node ``n_c{k}_s2``) per cell -- the intermediate-node vias the
    topology families do not carry. Operating point: 0 V, f_s = 3.1 GHz,
    self-pumped 6.2 GHz, +29 dBm.
    """
    dev = get_entry("SMV1245-079LF").to_realistic_varactor(v_bias_v=float(v_bias_v))
    spec = optimized_self_pumped_spec(
        topology=_SMV1245_TOPO, params=_SMV1245_STUB_PARAMS_M, varactor=dev,
        substrate=SUBSTRATE, n_cells=n_cells, main_width_m=_SMV1245_MAIN_W_M,
        main_period_m=_SMV1245_MAIN_PERIOD_M, f_signal_Hz=_SMV1245_F_S_HZ,
        f_pump_Hz=2.0 * _SMV1245_F_S_HZ, signal_power_dBm=_SMV1245_SIGNAL_DBM,
        pump_power_dBm=pump_dBm, via_radius_m=_SMV1245_VIA_RADIUS_M,
        input_match=None, output_match=None, via_model=_VIA_BOARD, open_end_model=_OPEN)
    sub = _substrate_name(spec)
    new_els = list(spec.elements)
    for k in range(1, n_cells + 1):
        new_els.append(_viagnd(f"VIAGND1_C{k}", f"n_c{k}_s1", _SMV1245_SEC_WIDTHS_M[0], sub))
        new_els.append(_viagnd(f"VIAGND2_C{k}", f"n_c{k}_s2", _SMV1245_SEC_WIDTHS_M[1], sub))
    return dataclasses.replace(spec, elements=tuple(new_els))


# --------------------------------------------------------------------------- #
# Example 2 -- SMV2020 SHORT / single VIAGND end-via
# --------------------------------------------------------------------------- #
_SMV2020_GEOM = {
    # Single via (n1_single_short), as built and recorded. Up to 1.0.0 this entry said DOUBLE while the
    # generator emitted one via per cell; since 1.0.1 a DOUBLE family emits two.
    "topology": TopologyFamily(n_segments=1, via=ViaConfig.SINGLE, termination=Termination.SHORT),
    "stub_params": (3.207799e-3, 4.387881e-3),
    "main_width_m": 8.869238652187022e-3,
    "main_period_m": 3.987578947368421e-3,
}
_SMV2020_F_S_HZ = 2.0e9
_SMV2020_V_BIAS_V = 16.0
_SMV2020_N_CELLS = 24


def spec_smv2020(n_cells=_SMV2020_N_CELLS, pump_dBm=_PUMP_DBM) -> DeviceSpec:
    """SMV2020-079LF SHORT / single VIAGND end-via, 24 cells, f_s = 2.0 GHz, 16 V.

    A near-oscillation regenerative operating point (the SHORT-terminated sibling
    of the standard single-via cell).
    """
    dev = get_entry("SMV2020-079LF").to_realistic_varactor(v_bias_v=_SMV2020_V_BIAS_V)
    return build_fixed_geom(_SMV2020_GEOM, dev, _SMV2020_F_S_HZ, pump_dBm, n_cells=n_cells)


# --------------------------------------------------------------------------- #
# Example 3 -- MAVR-000350 COMBINED MLEF||VIAGND parallel trap
# --------------------------------------------------------------------------- #
# Frozen cell geometry for the COMBINED n4_single_combined design. The 10-element
# stub-parameter vector is [w0,l0, w1,l1, w2,l2, w3,l3, w_stublet,l_stublet]
# (four series sections + the COMBINED open shunt stublet).
_MAVR_TOPO = TopologyFamily(n_segments=4, via=ViaConfig.SINGLE, termination=Termination.COMBINED)
assert _MAVR_TOPO.key == "n4_single_combined"
_MAVR_STUB_PARAMS_MM = (2.26772476882, 14.5664964211, 3.35051290989, 6.54526167223,
                        0.422622418404, 4.50723058078, 4.47520902008, 9.81412607757,
                        4.86448738873, 13.2046549642)
_MAVR_GEOM = {
    "topology": _MAVR_TOPO,
    "stub_params": tuple(x * 1e-3 for x in _MAVR_STUB_PARAMS_MM),
    "main_width_m": 5.80429344283e-3,
    "main_period_m": 5.41171428572e-3,
}
_MAVR_F_S_HZ = 3.0e9
_MAVR_V_BIAS_V = 12.0
_MAVR_N_CELLS = 24


def spec_mavr(n_cells=_MAVR_N_CELLS, pump_dBm=_PUMP_DBM) -> DeviceSpec:
    """MAVR-000350 COMBINED MLEF||VIAGND parallel trap, 24 cells, f_s = 3.0 GHz, 12 V.

    A stable net-positive degenerate-parametric amplifier. At each cell's stub-end
    node an open stublet (MLIN + MLEF fringe) hangs in parallel with a grounding
    via (VIAGND) -- the high-Q resonant trap.
    """
    dev = get_entry("MAVR-000350").to_realistic_varactor(v_bias_v=_MAVR_V_BIAS_V)
    return build_fixed_geom(_MAVR_GEOM, dev, _MAVR_F_S_HZ, pump_dBm, n_cells=n_cells)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
# (key, spec-builder, library name, top-cell name, one-line headline)
EXAMPLES = (
    ("smv1245", spec_smv1245, "fabricated_SMV1245", "fabricated_smv1245_top",
     "Fabricated SMV1245 board -- OPEN/MLEF 3-section stub with two mid-stub VIAGND, "
     "15 cells, 0 V, 3.1 GHz"),
    ("smv2020", spec_smv2020, "real_oscillator_SMV2020", "real_oscillator_smv2020_top",
     "SMV2020-079LF SHORT / single VIAGND end-via, 24 cells, 16 V, 2.0 GHz "
     "(regenerative edge)"),
    ("mavr", spec_mavr, "real_amplifier_MAVR", "real_amplifier_mavr_top",
     "MAVR-000350 COMBINED MLEF||VIAGND parallel trap, 24 cells, 12 V, 3.0 GHz "
     "(stable net-positive amplifier)"),
)

# Signal-frequency output-node voltage magnitude each design reproduces under HB
# (the parity reference; see build_examples.py). f_s in GHz alongside.
REFERENCE_VOUT_SIGNAL = {
    "smv1245": (1.503640e-02, 3.1),
    "smv2020": (1.542714e-02, 2.0),
    "mavr": (2.615556e-02, 3.0),
}


def resolved_spec(builder) -> DeviceSpec:
    """Build a spec and resolve its VAR block to literals (idempotent w/ emitters)."""
    return resolve_spec(builder())
