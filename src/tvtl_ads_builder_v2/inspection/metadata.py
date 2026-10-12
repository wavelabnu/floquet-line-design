"""Inspection metadata for the ADS Builder v2 canonical circuits.

The device spec (``device_spec.json``) is a frozen contract and must
not carry presentational fields, so per-circuit human-readable
inspection metadata lives here in a registry keyed by circuit label
rather than on the device-spec dataclass. The inspection writers look
up :data:`CIRCUIT_INSPECTION_METADATA` by ``DeviceSpec.label``.

(A ``CanonicalCircuit`` dataclass with ``topology_description`` /
``expected_behavior`` fields was considered; the canonical circuits are
plain ``DeviceSpec`` objects and the device-spec schema is frozen, so a
label-keyed registry is used instead -- same information, no schema
change.)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CircuitInspectionMetadata:
    """Human-readable inspection metadata for one canonical circuit.

    Parameters
    ----------
    level : int
        The ladder level the circuit belongs to.
    topology_description : str
        One sentence describing the circuit topology.
    expected_behavior : str
        One sentence describing what the |S21| sweep should show.
    notes : str
        Any circuit-specific caveat; empty string when there is none.
    """

    level: int
    topology_description: str
    expected_behavior: str
    notes: str = ""


# Short human-readable description of each ladder level, used in the
# polished netlist header and the index tables.
LEVEL_DESCRIPTIONS: dict[int, str] = {
    1: "Port + MLIN + S-parameter analysis",
    2: "MLIN cascade junctions",
    3: "MTEE T-junction + short-circuited shunt stub",
    4: "Lumped R / L / C and series RLC",
    5: "Full passive 3-segment via stub",
    6: "Single-tone Harmonic Balance controller",
    7: "P_1Tone power source + integrated two-tone-HB TVTL cell",
    8: "Topology-driven N-cell TVTL cascade (design API)",
    9: "Probed N-cell TVTL cascade (boundary + varactor I_Probes)",
    10: "Paper-geometry TVTL cascade (Rogers 4003C + Skyworks SMV1245)",
}


CIRCUIT_INSPECTION_METADATA: dict[str, CircuitInspectionMetadata] = {
    # ---- Level 1 -----------------------------------------------------
    "matched_short": CircuitInspectionMetadata(
        level=1,
        topology_description=(
            "Single 2.95 mm-wide, 1.0 mm-long MLIN between two 50 Ohm ports."
        ),
        expected_behavior=(
            "Near-zero insertion loss (|S21| within a few hundredths of a dB "
            "of 0 dB) across the 1-10 GHz band."
        ),
    ),
    "matched_long": CircuitInspectionMetadata(
        level=1,
        topology_description=(
            "Single 2.95 mm-wide, 10.0 mm-long MLIN between two 50 Ohm ports."
        ),
        expected_behavior=(
            "Near-zero insertion loss, marginally more than the short line "
            "because of the longer conductor path."
        ),
    ),
    "mismatched_50ohm_port": CircuitInspectionMetadata(
        level=1,
        topology_description=(
            "Single 5.0 mm-wide (sub-50 Ohm), 5.0 mm-long MLIN between two "
            "50 Ohm ports."
        ),
        expected_behavior=(
            "Mild return loss from the impedance step at the ports; |S21| "
            "dips a few tenths of a dB."
        ),
    ),
    "paper_main_line_one_period": CircuitInspectionMetadata(
        level=1,
        topology_description=(
            "Single 4.27 mm-wide, 6.10 mm-long MLIN -- the paper main-line "
            "geometry -- between two 50 Ohm ports."
        ),
        expected_behavior=(
            "Near-zero insertion loss; this is the paper's main-line "
            "reference geometry."
        ),
    ),
    # ---- Level 2 -----------------------------------------------------
    "two_identical_5mm_each": CircuitInspectionMetadata(
        level=2,
        topology_description=(
            "Two identical 2.95 mm-wide, 5.0 mm-long MLINs in series through "
            "one shared internal node."
        ),
        expected_behavior=(
            "Electrically identical to a single 10 mm line -- |S21| matches "
            "the Level-1 'matched_long' circuit."
        ),
        notes=(
            "Cascade cross-check: splitting a uniform line into collinear "
            "segments must be exactly equivalent to the single line."
        ),
    ),
    "same_width_split_3_7mm": CircuitInspectionMetadata(
        level=2,
        topology_description=(
            "Two 2.95 mm-wide MLINs (3.0 mm then 7.0 mm) in series."
        ),
        expected_behavior=(
            "Near-zero insertion loss; an asymmetric length split of a "
            "10 mm line."
        ),
    ),
    "step_width_4p27_to_1p93mm": CircuitInspectionMetadata(
        level=2,
        topology_description=(
            "Two MLINs of different width (4.27 mm then 1.93 mm), 3.0 mm "
            "each, in series."
        ),
        expected_behavior=(
            "Near-zero insertion loss; the width step adds a negligible "
            "discontinuity at the tested widths."
        ),
        notes=(
            "MSTEP discovery circuit: passes parity without an explicit step "
            "discontinuity element."
        ),
    ),
    "three_segment_step_step": CircuitInspectionMetadata(
        level=2,
        topology_description=(
            "Three MLINs of mixed width (4.27 / 2.95 / 4.27 mm) in series "
            "with two width-step junctions."
        ),
        expected_behavior=(
            "Near-zero insertion loss across the two width-step junctions."
        ),
    ),
    # ---- Level 3 -----------------------------------------------------
    "tee_open_zero_stub": CircuitInspectionMetadata(
        level=3,
        topology_description=(
            "MTEE T-junction with a near-zero-length (0.1 mm) open shunt "
            "stub; main line 2.95 mm wide, 3.0 mm + 7.0 mm."
        ),
        expected_behavior=(
            "Near-equivalent to a plain two-MLIN cascade -- the open stub is "
            "electrically inert, so |S21| stays near 0 dB."
        ),
    ),
    "tee_short_zero_stub": CircuitInspectionMetadata(
        level=3,
        topology_description=(
            "MTEE T-junction with a short (1.0 mm) grounded shunt stub; main "
            "line 4.27 mm wide, 3.0 mm each side."
        ),
        expected_behavior=(
            "Strong reflection at low frequency (the near-short shunt); "
            "|S21| rises toward 0 dB as the stub reactance grows with "
            "frequency."
        ),
        notes=(
            "Strong-reflection circuit: parity-blocked. ADS's MTEE element "
            "returns a mildly non-passive 3-port for this geometry "
            "(|S11|^2 + |S21|^2 > 1), which a passive analytical model "
            "cannot match."
        ),
    ),
    "tee_short_paper_dims": CircuitInspectionMetadata(
        level=3,
        topology_description=(
            "MTEE T-junction with a paper-dimensioned 1.93 mm-wide, "
            "6.83 mm-long grounded shunt stub; main line 4.27 mm wide, "
            "3.05 mm each side."
        ),
        expected_behavior=(
            "Frequency-dependent |S21| set by the shorted-stub input "
            "admittance transformed to the junction."
        ),
    ),
    "tee_qw_short_3p1GHz": CircuitInspectionMetadata(
        level=3,
        topology_description=(
            "MTEE T-junction with a quarter-wave (at 3.1 GHz) shorted shunt "
            "stub and short 0.2 mm matched main arms."
        ),
        expected_behavior=(
            "Transmission peak at 3.1 GHz, where the quarter-wave shorted "
            "stub presents an open circuit at the junction."
        ),
        notes=(
            "Resonant strong-reflection circuit: parity-blocked. The "
            "ideal-junction TVTL model omits the MTEE parasitics that shift "
            "the resonance relative to ADS."
        ),
    ),
    # ---- Level 4 -----------------------------------------------------
    "rlc_pure_50ohm": CircuitInspectionMetadata(
        level=4,
        topology_description=(
            "A lossy series RLC (R = 50 Ohm, L = 1 nH, C = 1 pF) "
            "connected directly between two 50 Ohm ports."
        ),
        expected_behavior=(
            "A transmission peak at the series resonance "
            "f0 = 1/(2*pi*sqrt(LC)) ~ 5.03 GHz, where the reactance "
            "cancels and Z = R; |S21| falls away on either side as the "
            "net reactance grows."
        ),
    ),
    "rlc_lossless": CircuitInspectionMetadata(
        level=4,
        topology_description=(
            "A lossless series RLC (R = 0, L = 1 nH, C = 1 pF) between "
            "two 50 Ohm ports."
        ),
        expected_behavior=(
            "A sharp |S21| peak reaching 0 dB at the ~5.03 GHz series "
            "resonance, where Z = 0 is a through short; strong "
            "reflection off-resonance."
        ),
        notes=(
            "The resonance is sharp but falls between the 5.0 and "
            "5.1 GHz grid bins, so the swept |S21| is sampled only where "
            "the response is smooth; R = 0 still clears the 0.5 dB gate."
        ),
    ),
    "mlin_then_rlc": CircuitInspectionMetadata(
        level=4,
        topology_description=(
            "A 50-Ohm-matched 2.95 mm-wide, 5.0 mm-long MLIN in series "
            "with a lossy RLC (R = 50 Ohm, L = 1 nH, C = 1 pF)."
        ),
        expected_behavior=(
            "The RLC sets the frequency response (resonance near "
            "5.03 GHz); the matched MLIN adds only a small electrical "
            "delay."
        ),
        notes=(
            "Composition check: a lumped element cascaded with a "
            "distributed one."
        ),
    ),
    "mtee_with_rlc_stub": CircuitInspectionMetadata(
        level=4,
        topology_description=(
            "MTEE T-junction with a grounded series-RLC shunt stub "
            "(R = 50 Ohm, L = 1 nH, C = 1 pF); main line 4.27 mm wide, "
            "3.05 mm each side."
        ),
        expected_behavior=(
            "A moderate |S21| dip near the 5.03 GHz shunt resonance, "
            "where the grounded RLC presents its minimum impedance "
            "(Z = R) to the junction."
        ),
        notes=(
            "MTEE circuit: parity-blocked. The series-RLC shunt itself "
            "is unambiguous, but the TVTL ideal-junction MTEE model "
            "leaves a ~1 dB |S21| gap against ADS's MTEE element here "
            "(Path B ~0.997 dB) -- the same model-fidelity limitation "
            "as the Level-3 MTEE circuits. R = 50 Ohm damps the shunt, "
            "so this is the junction-model gap, not a strong-reflection "
            "degeneracy. Structural analogue of the paper's varactor "
            "unit cell (the SMV1245 model is itself a series R + L + C)."
        ),
    ),
    # ---- Level 5 -----------------------------------------------------
    "level5_single_sdd2p_varactor": CircuitInspectionMetadata(
        level=5,
        topology_description=(
            "A single SDD2P symbolically-defined-device varactor "
            "shunting the input port to ground; its port-1 charge "
            "equation carries the SMV1245 Q(v) polynomial."
        ),
        expected_behavior=(
            "This circuit validates the SDD2P element structurally only: "
            "the OA-vs-hpeesofsim netlist cross-check (Path A) plus the "
            "polynomial content cross-check confirm both backends emit "
            "the device spec's charge polynomial. There is no analytical "
            "SDD2P model yet, so the Path B leg does not run."
        ),
        notes=(
            "SDD2P diagnostic circuit. The polynomial content check "
            "is the gate that would have caught v1's silent "
            "(_v1)/50.0 placeholder bug. Nonlinear HB simulation is "
            "covered by the HB circuits, not this circuit."
        ),
    ),
    "level5_tvtl_unit_cell": CircuitInspectionMetadata(
        level=5,
        topology_description=(
            "A TVTL unit cell: two MLIN main-line arms across an MTEE "
            "T-junction, with an SDD2P varactor (SMV1245 Q(v) charge "
            "polynomial) as the grounded shunt branch off the MTEE "
            "branch pin."
        ),
        expected_behavior=(
            "The SDD2P element is validated structurally: the "
            "OA-vs-hpeesofsim netlist cross-check (Path A) plus the "
            "polynomial content cross-check confirm both backends emit "
            "the device spec's charge polynomial and its port-2 mirror. "
            "There is no analytical SDD2P model yet, so the Path B leg "
            "does not run."
        ),
        notes=(
            "Smallest meaningful integration of the SDD2P element in an "
            "ADS-relevant unit-cell topology. Nonlinear HB validation is "
            "ADS-dependent and is not part of the base Python test suite."
        ),
    ),
    "level5_single_var_block": CircuitInspectionMetadata(
        level=5,
        topology_description=(
            "A series RLC between two 50 Ohm ports whose R, L and C "
            "parameters are not literals but references to named "
            "scalars (R, L, Cvar) declared in the circuit's VAR block; "
            "a fourth scalar, f0, is a declared-but-unreferenced "
            "named constant."
        ),
        expected_behavior=(
            "The VAR block is resolved away before emission -- the "
            "var-resolver substitutes each named-scalar reference with "
            "its literal numeric value -- so both backends emit an "
            "identical numeric R/L/C expansion. The structural "
            "cross-check (Path A) plus the VAR content check confirm "
            "the two backends emit the same resolved literals."
        ),
        notes=(
            "Circuit exercising the VAR named-parameter "
            "resolver. VAR is not an emitted ADS element: ADS "
            "VAR/VarEqn is non-functional in ADS 2026 Update 1.2 (see "
            "docs/LIMITATIONS.md), so named scalars are resolved to "
            "literals in the device spec rather than emitted as a VAR "
            "block."
        ),
    ),
    # ---- Level 6 -----------------------------------------------------
    "level6_single_tone_hb_controller": CircuitInspectionMetadata(
        level=6,
        topology_description=(
            "A 50-Ohm series resistor between two ports, with a "
            "single-tone Harmonic Balance analysis controller "
            "(one fundamental frequency, one harmonic order)."
        ),
        expected_behavior=(
            "The HB controller is validated structurally: the "
            "OA-vs-hpeesofsim netlist cross-check (Path A) plus the HB "
            "controller content check confirm both backends emit the "
            "same HB directive (Freq[1], Order[1], MaxOrder). The "
            "signal path is a plain 50-Ohm series resistor, so the "
            "analytical Path B leg sees an ordinary frequency-flat "
            "S-parameter response."
        ),
        notes=(
            "Single-tone HB controller circuit: the HB controller element "
            "type. HB sources and the SDD2P-driven nonlinear path are "
            "covered by the two-tone and source circuits -- this "
            "circuit exercises the controller only, "
            "over a linear resistive load."
        ),
    ),
    "level6_two_tone_hb_controller": CircuitInspectionMetadata(
        level=6,
        topology_description=(
            "A 50-Ohm series resistor between two ports, with a "
            "two-tone Harmonic Balance analysis controller -- a signal "
            "fundamental and a 2x pump fundamental, each with its own "
            "harmonic order, plus a combined MaxOrder."
        ),
        expected_behavior=(
            "The HB controller is validated structurally: the "
            "OA-vs-hpeesofsim netlist cross-check (Path A) plus the HB "
            "controller content check confirm both backends emit the "
            "same two-tone HB directive (Freq[1], Freq[2], Order[1], "
            "Order[2], MaxOrder). The signal path is a plain 50-Ohm "
            "series resistor, so the analytical Path B leg sees an "
            "ordinary frequency-flat S-parameter response."
        ),
        notes=(
            "Two-tone HB controller circuit: the HB controller element "
            "type -- the controller arrangement a parametric-"
            "amplification analysis uses. HB sources and the "
            "SDD2P-driven nonlinear path are in the source circuits; "
            "this circuit "
            "exercises the controller only, over a linear resistive "
            "load."
        ),
    ),
    # ---- Level 7 -----------------------------------------------------
    "level7_two_tone_p1tone_sources": CircuitInspectionMetadata(
        level=7,
        topology_description=(
            "Two ADS P_1Tone power sources -- a signal tone and a pump "
            "tone -- on one input net, driving a 50 Ohm series load to "
            "the output port. Every source parameter (both frequencies "
            "and both powers) is a named reference into the circuit's "
            "VAR block."
        ),
        expected_behavior=(
            "The P_1Tone sources are validated structurally: the "
            "OA-vs-hpeesofsim netlist cross-check (Path A) plus the "
            "P_1Tone source content check confirm both backends emit "
            "each source with the same driven node, Num, Z, Freq and P, "
            "and that the VAR named references resolve to identical "
            "literals on both backends."
        ),
        notes=(
            "Source circuit: the isolated P_1Tone source test. v1's "
            "two-tone HB stimulus is two P_1Tone sources on one net "
            "(both Num=1) -- there is no single P_2Tone element on this "
            "ADS install. "
            "The input carries no S-parameter port: the sources are the "
            "input."
        ),
    ),
    "level7_tvtl_unit_cell_two_tone_hb": CircuitInspectionMetadata(
        level=7,
        topology_description=(
            "The integrated TVTL unit cell: two P_1Tone sources (signal "
            "+ pump) driving two MLIN main-line arms across an MTEE "
            "T-junction, with a shunt branch of a series RLC (the "
            "varactor package parasitics) feeding an SDD2P nonlinear-"
            "charge varactor to ground, terminated at the output port; "
            "a two-tone Harmonic Balance controller and a VAR block "
            "tie it together."
        ),
        expected_behavior=(
            "Validated structurally: the OA-vs-hpeesofsim netlist "
            "cross-check (Path A) plus the SDD2P, VAR, two-tone-HB and "
            "P_1Tone content checks confirm both backends emit an "
            "identical resolved circuit. A single named f_signal drives "
            "both the HB Freq[1] and the signal source's Freq from one "
            "VAR declaration; f_pump and pump_power_dBm likewise. The "
            "SDD2P varactor is nonlinear and has no closed-form linear "
            "S-parameter, so the analytical Path B leg does not run "
            "(nan dB)."
        ),
        notes=(
            "Integration circuit -- the canonical that, "
            "emitted and run in ADS, simulates parametric amplification "
            "in a TVTL unit cell from a single device spec. It composes "
            "the four element layers: VAR block, P_1Tone sources, "
            "two-tone HB controller, and the "
            "level5_tvtl_unit_cell SDD2P topology. RLC and source "
            "values are generic test values; the SDD2P charge "
            "polynomial is the shared SMV1245 model. Content MATCH is "
            "the deliverable; Path B is nan dB by SDD2P nonlinearity, "
            "not a parity failure."
        ),
    ),
    # ---- Level 8 -----------------------------------------------------
    "level8_three_cell_tvtl_cascade": CircuitInspectionMetadata(
        level=8,
        topology_description=(
            "A cascade of three identical TVTL unit cells, generated "
            "from one declarative TVTLDesign by the topology-driven design "
            "API. Each cell is a through-line MLIN and an MTEE tapping a "
            "shunt branch -- a series R - L - C chain feeding an SDD2P "
            "nonlinear-charge varactor to ground; two P_1Tone sources "
            "(signal + pump) drive the first cell's input and a two-tone "
            "HB controller and a VAR block tie it together."
        ),
        expected_behavior=(
            "Validated structurally: the OA-vs-hpeesofsim netlist "
            "cross-check (Path A) plus the SDD2P, VAR, two-tone-HB and "
            "P_1Tone content checks confirm both backends emit an "
            "identical resolved circuit for all three cells. Every cell "
            "carries an SDD2P nonlinear varactor with no closed-form "
            "linear S-parameter, so the analytical Path B leg does not "
            "run (nan dB)."
        ),
        notes=(
            "The first canonical built from the "
            "topology-driven design API. The DeviceSpec -- per-cell "
            "element instances, per-cell net names, multi-cell wiring -- "
            "is generated mechanically by to_device_spec(); nothing is "
            "hand-assembled. Content MATCH is the deliverable; Path B is "
            "nan dB by SDD2P nonlinearity, not a parity failure."
        ),
    ),
    "level8_five_cell_tvtl_cascade": CircuitInspectionMetadata(
        level=8,
        topology_description=(
            "A cascade of five identical TVTL unit cells, generated from "
            "the SAME TVTLDesign template as the three-cell canonical "
            "with only n_cells changed. Each cell is a through-line "
            "MLIN, an MTEE, and a shunt R - L - C - SDD2P-varactor "
            "branch to ground."
        ),
        expected_behavior=(
            "Validated structurally, exactly as the three-cell canonical "
            "(Path A cross-check plus the content checks across all five "
            "cells). The analytical Path B leg does not run (nan dB) -- "
            "every cell carries an SDD2P nonlinear varactor."
        ),
        notes=(
            "Scaling check: it shares the cell template with "
            "level8_three_cell_tvtl_cascade and differs only in the cell "
            "count, so a clean build confirms the design API expands an "
            "arbitrary N with no hardcoded cell count. Content MATCH is "
            "the deliverable; Path B is nan dB by SDD2P nonlinearity."
        ),
    ),
    # ---- Level 9 -----------------------------------------------------
    "level9_two_cell_probed_tvtl_cascade": CircuitInspectionMetadata(
        level=9,
        topology_description=(
            "A two-cell TVTL cascade with I_Probes at every cell boundary "
            "(input, between cells, output) and in series with every "
            "varactor. Each cell is two MLIN halves across an MTEE, with a "
            "shunt branch off the MTEE pin 3: SDD2P varactor -> I_Probe -> "
            "stub MLIN -> per-cell ground. Two P_1Tone sources drive the "
            "input (signal Num=1, pump Num=2) and a two-tone HB controller "
            "ties it together; a VAR block declares the source/HB scalars."
        ),
        expected_behavior=(
            "Validated structurally: the OA-vs-hpeesofsim netlist "
            "cross-check (Path A) plus the SDD2P, VAR, two-tone-HB and "
            "P_1Tone content checks confirm both backends emit an "
            "identical resolved circuit including the five I_Probe Short "
            "lines (Mode=0 SaveCurrent=yes). Every cell carries an SDD2P "
            "nonlinear varactor with no closed-form linear S-parameter, so "
            "the analytical Path B leg does not run (nan dB)."
        ),
        notes=(
            "Probed-cascade canonical: the first cascade "
            "exercising the platform's I_Probe element type with both "
            "boundary and varactor probes. Two refinements over Level-7/8 "
            "make this deck HB-runnable end-to-end: distinct Num on the "
            "P_1Tone sources (Num=1 / Num=2) avoids ADS's S_Param "
            "Duplicate-Port error, and the I_Probes expose per-varactor "
            "and per-boundary currents the per-cell gain "
            "analysis reads as ``<id>.i`` from the HB1.HB varblock. "
            "I_Probe count: 5 (3 boundary + 2 varactor)."
        ),
    ),
    "level9_three_cell_probed_tvtl_cascade": CircuitInspectionMetadata(
        level=9,
        topology_description=(
            "A three-cell TVTL cascade with the same per-cell probed "
            "topology as level9_two_cell_probed_tvtl_cascade (MLIN halves "
            "+ MTEE + SDD2P varactor + I_Probe + stub MLIN to ground). "
            "Boundary I_Probes at input, between every adjacent cell pair, "
            "and at output. Two distinct-Num P_1Tone sources + two-tone HB."
        ),
        expected_behavior=(
            "Validated structurally, exactly as the two-cell canonical "
            "(Path A cross-check plus the content checks across all "
            "three cells and seven I_Probes). The analytical Path B leg "
            "does not run (nan dB) -- every cell carries an SDD2P "
            "nonlinear varactor."
        ),
        notes=(
            "Probed-cascade scaling canonical: shares the "
            "_probed_cell template with the two-cell canonical and differs "
            "only in the cell count. I_Probe count: 7 (4 boundary + 3 "
            "varactor)."
        ),
    ),
    # ---- Level 10 ----------------------------------------------------
    "level10_paper_geometry_tvtl_cascade": CircuitInspectionMetadata(
        level=10,
        topology_description=(
            "The paper-fidelity 15-cell TVTL cascade. Each cell carries "
            "two main-line MLIN halves (w=4.27 mm) across an MTEE, with a "
            "shunt branch off MTEE pin 3: an external package inductor "
            "L_pkg=1.7 nH, a series RS=2 Ohm, an SDD2P varactor with the "
            "Skyworks SMV1245-011LF SPICE Taylor expansion (CJ0=6.9 pF, "
            "VJ=3.5 V, M=1.7) in parallel with a CP=0.47 pF package "
            "capacitor, a varactor I_Probe, a three-MLIN stub cascade "
            "(1.93/1.41/1.98 mm widths, 6.83/5.25/1.04 mm lengths), and a "
            "Bahl-formula via-to-ground inductor for two parallel 0.8 mm "
            "vias. Boundary I_Probes splice between cells and at the "
            "design boundaries; two P_1Tone sources (signal Num=1, pump "
            "Num=2) drive n_input; the output TermG is Num=3. Substrate "
            "is Rogers 4003C (eps_r=3.38, h=1.54 mm, tan_d=0.0027, "
            "copper 17.5 um)."
        ),
        expected_behavior=(
            "Validated structurally and quantitatively. The OA <-> "
            "hpeesofsim netlist cross-check (Path A) plus the SDD2P, "
            "VAR, two-tone-HB and P_1Tone content checks confirm both "
            "backends emit an identical resolved circuit across all 15 "
            "cells including the 31 I_Probe Short lines (15 varactor + "
            "14 inter-cell + 2 boundary). The analytical Floquet harness "
            "(harmonic ABCD lift) reads the same SMV1245 "
            "polynomial via the SDD2P q_poly_coefficients parameter and "
            "auto-detects the via inductance from the L_VIA lumped "
            "element in each cell's stub chain. The Path B analytical "
            "S-parameter leg compares against ADS S_Param on the "
            "linearised passive structure (SDD2P frozen at C0); the lift "
            "is exercised at the locked +16 dBm pump operating point "
            "via the Floquet comparison."
        ),
        notes=(
            "Paper-fidelity canonical: the publication "
            "Figure 4 reference device. Locked to the paper's fabricated "
            "geometry (Figure 3b + SI Methodology) with all five "
            "operating-point scalars in the VAR block (f_signal, f_pump, "
            "signal_power_dBm, pump_power_dBm, pump_phase_rad). NAMEs: 32 "
            "(n_input + 15 n_c{k}_b + 15 n_c{k}_v + n_output -- the n_c{k}_b "
            "branch-top is needed for the Floquet comparison harness's V_0 "
            "extraction, n_c{k}_v for the energy-conservation cross-check). "
            "I_Probe count: 31 (15 varactor + 16 boundary). 15-cell cascade "
            "is the paper's fabricated count; the make_tvtl_level10_cascade "
            "generic generator builds arbitrary-N cascades for ablation "
            "studies."
        ),
    ),
}


def metadata_for(label: str) -> CircuitInspectionMetadata:
    """Return inspection metadata for a circuit label.

    Parameters
    ----------
    label : str
        The ``DeviceSpec.label`` of a canonical circuit.

    Returns
    -------
    CircuitInspectionMetadata
        Registered metadata, or a generic placeholder when the label is
        not in the registry (so the inspection writers never crash on a
        new, not-yet-described circuit).
    """

    found = CIRCUIT_INSPECTION_METADATA.get(label)
    if found is not None:
        return found
    return CircuitInspectionMetadata(
        level=0,
        topology_description="(no topology description registered)",
        expected_behavior="(no expected-behavior description registered)",
    )


