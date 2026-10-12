"""Decoupled TVTL design-workflow package.

A four-stage, *decoupled* distributed-parametric-amplifier design workflow
(there is no unified cross-stage objective -- the decoupling is the methodology
contribution):

1. :mod:`~tvtl.optimization.stage1_per_cell` -- per-cell stub optimization
   (the :mod:`~tvtl.optimization.topology_generator` + the
   :mod:`~tvtl.optimization.figure_of_merit` A1 screen + hybrid pruning);
2. ``stage2_main_tl`` -- Bloch-impedance matching of the main line;
3. ``stage3_matching`` -- distributed matching-network design;
4. ``stage4_validation`` -- coherent-gain / PSA cross-validation.

This package orchestrates *above* both the :mod:`tvtl` analytical core and the
:mod:`tvtl_ads_builder_v2` ADS builder (it imports from both); it is never
imported by either, so there is no dependency cycle.
"""

from __future__ import annotations

from .figure_of_merit import (
    ClosedFormA1,
    FigureOfMerit,
    LorentzianA1,
    SingleCellFloquetA2,
    a1_factors,
    standardized_drive_voltage,
)
from .stage1_per_cell import (
    CoarseScreenResult,
    StubCandidate,
    coarse_screen,
    refine_topk,
)
from .stage2_main_tl import MainTLDesign, bloch_impedance, match_z_tl
from .stage3_matching import (
    MatchingSection,
    SideMatching,
    edge_impedance,
    quarter_wave_design,
    select_matching,
    single_stub_design,
)
from .stage4_validation import (
    CompleteDesign,
    CrossValReport,
    analytical_characterization,
    cross_validate,
)
from .substrate import ROGERS_4003C, SubstrateConfig, microstrip_width_for_z0
from .workflow import (
    WorkflowConfig,
    WorkflowResult,
    decoupled_design_workflow,
    select_diverse_top_m,
)
from .topology_generator import (
    Termination,
    TopologyFamily,
    ViaConfig,
    combined_families,
    enumerate_families,
    linear_families,
    sample_family,
    topology_y_stub_fn,
)

__all__ = [
    # substrate
    "SubstrateConfig",
    "ROGERS_4003C",
    # topology generator
    "TopologyFamily",
    "ViaConfig",
    "Termination",
    "enumerate_families",
    "linear_families",
    "combined_families",
    "sample_family",
    "topology_y_stub_fn",
    # figure of merit
    "FigureOfMerit",
    "ClosedFormA1",
    "LorentzianA1",
    "SingleCellFloquetA2",
    "a1_factors",
    "standardized_drive_voltage",
    # stage 1
    "StubCandidate",
    "CoarseScreenResult",
    "coarse_screen",
    "refine_topk",
    # stage 2
    "MainTLDesign",
    "bloch_impedance",
    "match_z_tl",
    "microstrip_width_for_z0",
    # stage 3
    "MatchingSection",
    "SideMatching",
    "edge_impedance",
    "quarter_wave_design",
    "single_stub_design",
    "select_matching",
    # stage 4
    "CompleteDesign",
    "CrossValReport",
    "analytical_characterization",
    "cross_validate",
    # workflow
    "WorkflowConfig",
    "WorkflowResult",
    "decoupled_design_workflow",
    "select_diverse_top_m",
]


