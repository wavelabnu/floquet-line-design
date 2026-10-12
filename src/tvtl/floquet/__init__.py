"""Generic Floquet proxy models for TVTL design search."""

from .convergence import FloquetConvergenceCheck, check_floquet_truncation_convergence, compare_truncation_orders
from .coupling_metrics import (
    FloquetCouplingMetrics,
    analytical_stub_gain_proxy,
    coupling_metrics_from_admittance,
    coupling_metrics_from_solution,
)
from .harmonics import (
    build_stub_tridiagonal,
    current_spectrum,
    elastance_coefficients_from_capacitance_waveform,
    harmonic_frequencies,
    harmonic_indices,
    sinusoidal_elastance_coefficients,
    solve_harmonic_coefficients,
)
from .harmonic_abcd import (
    cascade_harmonic_abcd,
    device_proxy_gain,
    harmonic_s_parameters,
    n_cells_harmonic_abcd,
    stub_green_function,
    stub_harmonic_abcd,
    stub_harmonic_admittance,
    tl_section_harmonic_abcd,
    unit_cell_harmonic_abcd,
)
from .stability import (
    StabilityAssessment,
    assemble_full_unit_cell_matrix,
    assess_stability_proxy,
    cascade_gain_estimate,
    dominant_growth_rate,
    floquet_exponents,
    stability_margin,
    unit_cell_eigenvalues,
)
from .passband import passband_check_at_operating_point, static_s21_at_frequency
from .pump_analysis import (
    compute_pump_voltage_profile,
    pump_coherence_check,
    pump_power_profile,
)
from .sweeps import (
    classify_design_spectrum,
    design_gain_spectrum,
    find_spectrum_peaks,
    frequency_modulation_map,
    modulation_depth_sweep,
    threshold_search,
)
from .topology import (
    DesignVariableSpec,
    TopologyDescriptor,
    design_bounds_dict,
    get_topology,
    list_topologies,
    register_topology,
)
from .uc_model import (
    FloquetBranchModel,
    FloquetUnitCellModel,
    VIA_STUB_3SEGMENT_DESCRIPTOR,
    build_unit_cell_block_abcd,
    build_via_stub_3segment_block_abcd,
    floquet_uc_cascade_response,
    unit_cell_period_m,
)
from .uc_matrix import FloquetBranchParameters, FloquetMatrixSolution, build_local_floquet_matrix, solve_local_floquet_admittance

# Importing the topologies subpackage registers the additional topology
# descriptors (open-stub, lumped-shunt) into the global registry.
from . import topologies as _topologies  # noqa: F401
from .topologies import (
    LUMPED_SHUNT_VARACTOR_DESCRIPTOR,
    OPEN_STUB_2SEGMENT_DESCRIPTOR,
)

__all__ = [
    "DesignVariableSpec",
    "FloquetBranchModel",
    "FloquetBranchParameters",
    "FloquetCouplingMetrics",
    "FloquetConvergenceCheck",
    "FloquetMatrixSolution",
    "FloquetUnitCellModel",
    "LUMPED_SHUNT_VARACTOR_DESCRIPTOR",
    "OPEN_STUB_2SEGMENT_DESCRIPTOR",
    "StabilityAssessment",
    "TopologyDescriptor",
    "VIA_STUB_3SEGMENT_DESCRIPTOR",
    "analytical_stub_gain_proxy",
    "assemble_full_unit_cell_matrix",
    "assess_stability_proxy",
    "build_local_floquet_matrix",
    "build_stub_tridiagonal",
    "build_unit_cell_block_abcd",
    "build_via_stub_3segment_block_abcd",
    "cascade_gain_estimate",
    "cascade_harmonic_abcd",
    "check_floquet_truncation_convergence",
    "classify_design_spectrum",
    "compare_truncation_orders",
    "compute_pump_voltage_profile",
    "coupling_metrics_from_admittance",
    "coupling_metrics_from_solution",
    "current_spectrum",
    "design_bounds_dict",
    "design_gain_spectrum",
    "device_proxy_gain",
    "dominant_growth_rate",
    "elastance_coefficients_from_capacitance_waveform",
    "find_spectrum_peaks",
    "floquet_exponents",
    "floquet_uc_cascade_response",
    "frequency_modulation_map",
    "get_topology",
    "harmonic_frequencies",
    "harmonic_indices",
    "harmonic_s_parameters",
    "list_topologies",
    "modulation_depth_sweep",
    "n_cells_harmonic_abcd",
    "passband_check_at_operating_point",
    "pump_coherence_check",
    "pump_power_profile",
    "register_topology",
    "sinusoidal_elastance_coefficients",
    "solve_harmonic_coefficients",
    "solve_local_floquet_admittance",
    "stability_margin",
    "static_s21_at_frequency",
    "stub_green_function",
    "stub_harmonic_abcd",
    "stub_harmonic_admittance",
    "threshold_search",
    "tl_section_harmonic_abcd",
    "unit_cell_eigenvalues",
    "unit_cell_harmonic_abcd",
    "unit_cell_period_m",
]


