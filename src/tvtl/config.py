"""Shared configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SweepConfig:
    """Uniform frequency sweep in SI units."""

    start_hz: float
    stop_hz: float
    step_hz: float

    def frequencies_hz(self) -> NDArray[np.float64]:
        count = int(round((self.stop_hz - self.start_hz) / self.step_hz)) + 1
        return self.start_hz + self.step_hz * np.arange(count, dtype=float)


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    """Optimization targets and weights for analytical fitting."""

    target_frequencies_hz: tuple[float, ...]
    s_weight: float = 1.0
    beta_weight: float = 1.0
    target_beta_ratio: float = 2.0
    use_tee: bool = True
    legacy_inactive_beta_term: bool = False
    differential_evolution_seed: int = 0
    polish: bool = True
    workers: int = 1
    design_bounds_m: tuple[tuple[float, float], ...] = (
        (0.0010, 0.0025),  # pl1
        (0.0010, 0.0070),  # pl3
        (0.0020, 0.0070),  # pl4
        (0.0020, 0.0070),  # pl5
        (0.0015, 0.0050),  # width1
        (0.0015, 0.0020),  # width3
        (0.0015, 0.0020),  # width4
        (0.0015, 0.0020),  # width5
    )  


class TeeModelVariant(StrEnum):
    """Switchable tee/discontinuity sub-models for diagnosis passes."""

    CORRECTED_NOTEBOOK = "corrected_notebook"
    CORRECTED_NOTEBOOK_NO_D2_EXP_V1 = "corrected_notebook_no_d2_exp_v1"
    CORRECTED_NOTEBOOK_NO_D2_EXP_HALF_MIXED_V1 = "corrected_notebook_no_d2_exp_half_mixed_v1"
    IDENTITY_SCALING = "identity_scaling"
    NATIVE_ADS_COMPACT_SYMMETRIC_V1 = "native_ads_compact_symmetric_v1"
    # MTEE_ADS-aligned ideal-junction variant. Empirical probe at 5
    # geometries on Rogers 4003C 1.524 mm substrate
    # (private calibration run) shows MTEE_ADS
    # S(2,1) sits at the lossless symmetric tee value (-3.52 dB = 20*log10(2/3)
    # with 50 ohm branch termination) across 1-7 GHz. NOTE: subsequent cascade
    # diagnostics refuted the hypothesis that
    # an ideal junction analytical matches MTEE_ADS cascade behavior; the
    # mechanism is geometry-dependent cross-coupling not captured by the
    # IDEAL_SYMMETRIC representation. This variant is preserved as scaffolding;
    # MTEE_ADS_SURROGATE_V1 is the working MTEE_ADS-aligned variant.
    IDEAL_SYMMETRIC_LOSSLESS = "ideal_symmetric_lossless"
    # MTEE_ADS surrogate variant. Numerical surrogate fit to MTEE_ADS
    # 3-port S-matrix on a 5x5 (w_main, w_branch) grid x 39 frequencies (1-20 GHz)
    # at Rogers 4003C 1.524 mm substrate. Cascade math: per-harmonic 3-port S
    # is loaded with the branch network's impedance (Z = B/D from the existing
    # branch ABCD chain) and reduced to a combined tee-plus-branch 2-port via
    # standard 3-port termination formula, replacing the existing
    # [block_left, branch_network, block_right] chain elements.
    # Default path (CORRECTED_NOTEBOOK) is preserved bit-for-bit; this variant
    # is opt-in for MTEE_ADS-flat-deck-emitter cross-validation.
    # Calibration source: optional public-copy surrogate table;
    # surrogate_tabular.json. Loaded lazily by
    # tvtl_ads_builder_v2.analytical.mtee_ads_surrogate.
    MTEE_ADS_SURROGATE_V1 = "mtee_ads_surrogate_v1"


class ViaModelVariant(StrEnum):
    """Switchable via sub-models for diagnosis passes."""

    NOTEBOOK = "notebook"
    SERIES_RL_ONLY = "series_rl_only"


class ConductorLossVariant(StrEnum):
    """Switchable conductor-loss assumptions for diagnosis passes."""

    LEGACY_SURFACE = "legacy_surface"
    OFF = "off"


class DielectricLossVariant(StrEnum):
    """Switchable dielectric-loss assumptions for diagnosis passes."""

    LEGACY_TAN_DELTA = "legacy_tan_delta"
    OFF = "off"


class ThicknessCorrectionVariant(StrEnum):
    """Finite-thickness handling variants."""

    CORRECTED = "corrected"
    LEGACY_BUG = "legacy_bug"


class OpenEndCorrectionVariant(StrEnum):
    """Open-end correction variants for stub termination diagnosis."""

    KIRSCHNING_FRANZ = "kirschning_franz"
    OFF = "off"


class PadTransformVariant(StrEnum):
    """Pad effective-transform variants for no-tee branch diagnosis."""

    BASELINE = "baseline"
    NATIVE_ADS_NO_TEE_V1 = "native_ads_no_tee_v1"


class FullCellOuterEmbeddingVariant(StrEnum):
    """Opt-in outer-embedding variants for full-cell diagnosis only."""

    BASELINE = "baseline"
    NATIVE_ADS_UNIT_CELL_OUTER_V1 = "native_ads_unit_cell_outer_v1"


class BlochExtractionStrategy(StrEnum):
    """Bloch/gamma extraction formulas for the unit-cell matrix."""

    TRACE_ARCCOSH = "trace_arccosh"
    EIGENVALUE_LOG = "eigenvalue_log"


class BetaBranchStrategy(StrEnum):
    """Frequency-sweep branch-selection strategies for Bloch beta."""

    POINTWISE_PRINCIPAL = "pointwise_principal"
    CONTINUOUS = "continuous"


class CascadeStrategy(StrEnum):
    """Ways of raising the unit-cell matrix to a cascade power."""

    MATRIX_POWER = "matrix_power"
    EIGEN_POWER = "eigen_power"


class HighFrequencyGuardrail(StrEnum):
    """Optional high-frequency stabilizers for diagnosis only."""

    OFF = "off"
    CLIP_DISCONTINUITY_SQRT = "clip_discontinuity_sqrt"


@dataclass(frozen=True, slots=True)
class AnalyticalModelConfig:
    """Explicit analytical-model assumptions for same-geometry diagnosis.

    The defaults preserve the current corrected analytical behavior. Any legacy
    or alternative path must be selected explicitly.
    """

    tee_model_variant: TeeModelVariant = TeeModelVariant.CORRECTED_NOTEBOOK
    via_model_variant: ViaModelVariant = ViaModelVariant.NOTEBOOK
    conductor_loss_variant: ConductorLossVariant = ConductorLossVariant.LEGACY_SURFACE
    dielectric_loss_variant: DielectricLossVariant = DielectricLossVariant.LEGACY_TAN_DELTA
    thickness_correction_variant: ThicknessCorrectionVariant = ThicknessCorrectionVariant.CORRECTED
    open_end_correction_variant: OpenEndCorrectionVariant = OpenEndCorrectionVariant.KIRSCHNING_FRANZ
    pad_transform_variant: PadTransformVariant = PadTransformVariant.BASELINE
    full_cell_outer_embedding_variant: FullCellOuterEmbeddingVariant = FullCellOuterEmbeddingVariant.BASELINE
    bloch_extraction_strategy: BlochExtractionStrategy = BlochExtractionStrategy.TRACE_ARCCOSH
    beta_branch_strategy: BetaBranchStrategy = BetaBranchStrategy.POINTWISE_PRINCIPAL
    cascade_strategy: CascadeStrategy = CascadeStrategy.MATRIX_POWER
    high_frequency_guardrail: HighFrequencyGuardrail = HighFrequencyGuardrail.OFF

    @property
    def preserve_legacy_thickness_bug(self) -> bool:
        """Compatibility helper for older analytical call sites."""

        return self.thickness_correction_variant == ThicknessCorrectionVariant.LEGACY_BUG


class ModulationMode(IntEnum):
    """Legacy capacitance-modulation options from the MATLAB FDTD code."""

    NONE = 0
    TIME_ONLY = 1
    SPACE_ONLY = 2
    SPACE_TIME = 3
    POLYNOMIAL_VARACTOR = 4


@dataclass(frozen=True, slots=True)
class FDTDConfig:
    """One-dimensional TL/FDTD configuration in SI units."""

    freq_max_hz: float
    tl_frequency_hz: float
    source_frequency_hz: float
    source_amplitude_v: float
    modulation_mode: ModulationMode = ModulationMode.NONE
    modulation_depth: float = 0.0
    modulation_voltage_v: float = 8.0
    base_l_h_per_m: float = 3.73e-7
    base_c_f_per_m: float = 7.91e-11
    base_r_ohm_per_m: float = 1.14
    base_g_s_per_m: float = 2.56e-4
    cells_per_wavelength: int = 30
    total_line_wavelengths: float = 3.0
    total_time_velocity_lengths: float = 3.0
    slab_wavelengths: float = 1.0
    source_position_fraction: float = 0.2
    slab_start_fraction: float = 0.5
    left_probe_fraction: float = 0.2
    right_probe_fraction: float = 0.5
    varactor_spacing_m: float = 0.005
    num_time_steps: int | None = None
    record_full_history: bool = False
    preserve_legacy_probe_locations: bool = False
    log_every: int = 0
    legacy_modulation_frequency_multiplier: float = 2.0
    varactor_polynomial_coefficients_f: tuple[float, ...] = field(
        default_factory=lambda: (3.14e-11, 1.88e-11, 6.43e-12, 1.24e-12, 1.23e-13, 4.8e-15)
    )



