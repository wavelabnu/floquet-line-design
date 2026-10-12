"""Clean-room ADS builder rewrite -- the active platform for ADS validation.

Architectural principles
------------------------
1. A canonical device-spec file (JSON or YAML) is the contract between
   TVTL (analytical truth) and the ADS-side simulator. Nothing flows
   between them except via the spec; the schema is version-pinned
   (``DEVICE_SPEC_VERSION = 1``).
2. TVTL is the source of truth. Every parity check compares an ADS
   ``hpeesofsim`` run against the TVTL analytical result for the same
   spec, not the other way around.
3. Generated netlists are flat ``hpeesofsim`` ``.ckt`` decks. The
   OpenAccess workspace emitter (``oa.emit_workspace_from_spec``) is
   the ADS-side companion; the netlist cross-check
   (``oa.compare_circuits``) confirms the two paths agree.
4. One element type per level. Levels 1-10 are implemented (Level 9
   adds probed cascades; Level 10, the canonical circuits).
5. Every level concludes with an automated parity gate
   (``parity.parity_check``). The tolerance is documented in
   ``parity.tolerance`` and never loosened to make a test pass.

Public API
----------
Symbols listed in ``__all__`` below form the stable public surface;
see ``STABILITY.md`` for the public/internal split. Internal modules
under this package are not part of the contract and may change.
"""

from .analysis import (
    DEFAULT_ENERGY_TOLERANCE_DB_REL_INPUT,
    DEFAULT_HARMONIC_TOLERANCE_DB,
    HBDataset,
    HBHarmonic,
    HBSpectrum,
    IDLER_MIX_2TONE,
    PUMP_MIX_2TONE,
    SIGNAL_MIX_2TONE,
    VaractorModelParameters,
    build_cell_map,
    cascade_energy_residuals,
    cascade_power_gain_dB,
    cascade_voltage_ratio_gain_dB,
    cell_boundary_current,
    cell_boundary_power,
    cell_branch_current,
    cell_count,
    cell_energy_residual,
    cell_net,
    cell_node_voltage,
    cell_power_gain_dB,
    cell_varactor_current,
    cell_voltage_ratio_gain_dB,
    compare_cascade_harmonics,
    compare_varactor_harmonics,
    extract_varactor_parameters,
    gain_vs_pump_power,
    gain_vs_signal_frequency,
    harmonic_band,
    harmonic_tolerance_dB,
    hb_dataset_from_raw,
    read_hb_dataset,
    varactor_power,
)
from .analytical import (
    Level10SParamResult,
    StubSegment,
    Y_stub_at_freq,
    build_stub_y_fn_from_spec,
    extract_level9_stub_segments,
    extract_series_via_inductance_H,
    extract_stub_segments,
    level10_paper_sparams,
    level10_parametric_gain_proxy,
    level10_signal_gain_proxy_dB,
)
from .ads_oa_writer import (
    ADSBuildResult,
    ADSComponentRef,
    build_self_pumped_minimal_schematic,
)
from .circuit_spec import SelfPumpedCircuitSpec, default_self_pumped_spec
from .design import (
    AnalysisSpec,
    BranchSpec,
    CellSpec,
    ElementSpec,
    MSUBSpec,
    MicrostripSpec,
    SParamSpec,
    SourceSpec,
    TVTLDesign,
    to_device_spec,
    validate_design,
)
from .device_spec import (
    DEVICE_SPEC_FILE_EXTENSION,
    Analysis,
    DeviceSpec,
    Element,
    ElementPin,
    HBSingleToneSpec,
    HBTwoToneSpec,
    Node,
    P1ToneSource,
    Port,
    Substrate,
    SweepSpec,
    VarBlockSpec,
    VarEntry,
    dump_device_spec,
    load_device_spec,
    resolve_spec,
    validate_device_spec,
    with_var_overrides,
)
from .netlist import emit_hpeesofsim_netlist
from .oa.netlist_compare import compare_circuits, format_report
from .oa.oa_emitter import emit_workspace_from_spec
from .parity.canonical_circuits import (
    LEVEL_1_CANONICAL_CIRCUITS,
    LEVEL_1_LABEL,
    LEVEL_2_CANONICAL_CIRCUITS,
    LEVEL_2_LABEL,
    LEVEL_3_CANONICAL_CIRCUITS,
    LEVEL_3_LABEL,
    LEVEL_4_CANONICAL_CIRCUITS,
    LEVEL_4_LABEL,
    LEVEL_5_CANONICAL_CIRCUITS,
    LEVEL_5_LABEL,
    LEVEL_6_CANONICAL_CIRCUITS,
    LEVEL_6_LABEL,
    LEVEL_7_CANONICAL_CIRCUITS,
    LEVEL_7_LABEL,
    LEVEL_10_CANONICAL_CIRCUITS,
    LEVEL_10_LABEL,
    Level10CellSpec,
    Level10StubSegmentSpec,
    Level10SubstrateSpec,
    Level10VaractorSpec,
    Level10ViaSpec,
    make_paper_geometry_sparam_only,
    make_paper_geometry_tvtl_cascade,
    make_tvtl_level10_cascade,
)
from .simulate.parser import parse_s_parameter_results, read_raw_dataset
from .simulate.runner import run_hpeesofsim
from .simulate.sweep_runner import (
    SweepPointResult,
    SweepResult,
    SweepRunner,
    ads_runtime_available,
)


__all__ = [
    # ---- device_spec ----
    "Analysis",
    "DEVICE_SPEC_FILE_EXTENSION",
    "DeviceSpec",
    "Element",
    "ElementPin",
    "HBSingleToneSpec",
    "HBTwoToneSpec",
    "Node",
    "P1ToneSource",
    "Port",
    "Substrate",
    "SweepSpec",
    "VarBlockSpec",
    "VarEntry",
    "dump_device_spec",
    "load_device_spec",
    "resolve_spec",
    "validate_device_spec",
    "with_var_overrides",
    # ---- netlist ----
    "emit_hpeesofsim_netlist",
    # ---- self-pumped validation ----
    "ADSBuildResult",
    "ADSComponentRef",
    "SelfPumpedCircuitSpec",
    "build_self_pumped_minimal_schematic",
    "default_self_pumped_spec",
    # ---- oa ----
    "compare_circuits",
    "emit_workspace_from_spec",
    "format_report",
    # ---- simulate ----
    "SweepPointResult",
    "SweepResult",
    "SweepRunner",
    "ads_runtime_available",
    "parse_s_parameter_results",
    "read_raw_dataset",
    "run_hpeesofsim",
    # ---- parity (Level constants) ----
    "LEVEL_1_CANONICAL_CIRCUITS",
    "LEVEL_1_LABEL",
    "LEVEL_2_CANONICAL_CIRCUITS",
    "LEVEL_2_LABEL",
    "LEVEL_3_CANONICAL_CIRCUITS",
    "LEVEL_3_LABEL",
    "LEVEL_4_CANONICAL_CIRCUITS",
    "LEVEL_4_LABEL",
    "LEVEL_5_CANONICAL_CIRCUITS",
    "LEVEL_5_LABEL",
    "LEVEL_6_CANONICAL_CIRCUITS",
    "LEVEL_6_LABEL",
    "LEVEL_7_CANONICAL_CIRCUITS",
    "LEVEL_7_LABEL",
    "LEVEL_10_CANONICAL_CIRCUITS",
    "LEVEL_10_LABEL",
    "Level10CellSpec",
    "Level10StubSegmentSpec",
    "Level10SubstrateSpec",
    "Level10VaractorSpec",
    "Level10ViaSpec",
    "make_paper_geometry_sparam_only",
    "make_paper_geometry_tvtl_cascade",
    "make_tvtl_level10_cascade",
    # ---- design ----
    "AnalysisSpec",
    "BranchSpec",
    "CellSpec",
    "ElementSpec",
    "MSUBSpec",
    "MicrostripSpec",
    "SParamSpec",
    "SourceSpec",
    "TVTLDesign",
    "to_device_spec",
    "validate_design",
    # ---- analysis (HB dataset reader, I_Probe-aware
    # cell_boundary_current and cell_varactor_current, and
    # per-cell power and gain helpers in both peak-phasor and
    # voltage-ratio conventions) ----
    "DEFAULT_ENERGY_TOLERANCE_DB_REL_INPUT",
    "DEFAULT_HARMONIC_TOLERANCE_DB",
    "HBDataset",
    "HBHarmonic",
    "HBSpectrum",
    "IDLER_MIX_2TONE",
    "PUMP_MIX_2TONE",
    "SIGNAL_MIX_2TONE",
    "VaractorModelParameters",
    "build_cell_map",
    "cascade_energy_residuals",
    "cascade_power_gain_dB",
    "cascade_voltage_ratio_gain_dB",
    "cell_boundary_current",
    "cell_boundary_power",
    "cell_branch_current",
    "cell_count",
    "cell_energy_residual",
    "cell_net",
    "cell_node_voltage",
    "cell_power_gain_dB",
    "cell_varactor_current",
    "cell_voltage_ratio_gain_dB",
    "compare_cascade_harmonics",
    "compare_varactor_harmonics",
    "extract_varactor_parameters",
    "gain_vs_pump_power",
    "gain_vs_signal_frequency",
    "harmonic_band",
    "harmonic_tolerance_dB",
    "hb_dataset_from_raw",
    "read_hb_dataset",
    "varactor_power",
    # ---- analytical (harmonic ABCD lift + Level-10 helpers) ----
    "Level10SParamResult",
    "StubSegment",
    "Y_stub_at_freq",
    "build_stub_y_fn_from_spec",
    "extract_level9_stub_segments",
    "extract_series_via_inductance_H",
    "extract_stub_segments",
    "level10_paper_sparams",
    "level10_parametric_gain_proxy",
    "level10_signal_gain_proxy_dB",
]


