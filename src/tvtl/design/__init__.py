"""Generic design-space contracts for TVTL candidate search."""

from .candidates import BranchSpec, CandidateScorecard, ComponentSpec, DesignCandidate, UnitCellGraph
from .constraints import ConstraintEvaluation, ConstraintSpec, ConstraintStatus, constraint_from_mapping, evaluate_constraints
from .nonlinear_capacitor import (
    AdsAnalysisMode,
    AdsRealizationMode,
    AdsRealizationPolicy,
    ChargePolynomialModel,
    NonlinearCapacitorModel,
    StaticEquivalentModel,
    charge_polynomial_from_coefficients,
    select_ads_realization_policy,
    static_ads_realization_report,
)
from .variables import DesignVariable, VariableKind, design_variable_from_mapping, sample_variable_midpoint

__all__ = [
    "AdsAnalysisMode",
    "AdsRealizationMode",
    "AdsRealizationPolicy",
    "BranchSpec",
    "CandidateScorecard",
    "ChargePolynomialModel",
    "ComponentSpec",
    "ConstraintEvaluation",
    "ConstraintSpec",
    "ConstraintStatus",
    "DesignCandidate",
    "DesignVariable",
    "NonlinearCapacitorModel",
    "StaticEquivalentModel",
    "UnitCellGraph",
    "VariableKind",
    "charge_polynomial_from_coefficients",
    "constraint_from_mapping",
    "design_variable_from_mapping",
    "evaluate_constraints",
    "sample_variable_midpoint",
    "select_ads_realization_policy",
    "static_ads_realization_report",
]


