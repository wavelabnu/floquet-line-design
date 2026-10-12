"""HB result analysis -- dataset structuring and per-cell queries.

This layer post-processes a Harmonic-Balance ``.ds`` dataset: it
structures the raw varblock columns into per-quantity harmonic spectra
(:mod:`.hb_dataset`) and resolves per-cell quantities of an N-cell TVTL
cascade (:mod:`.per_cell`).

It is platform-level, not design-layer. It imports from ``simulate/``
and ``device_spec/`` and operates on an *expanded* :class:`DeviceSpec`;
it does NOT import from ``design/`` and carries no design-layer naming
templates -- per-cell resolution walks the DeviceSpec's element list
(the k-th SDD2P is cell k's varactor). Any cascade DeviceSpec can be
analysed, whoever built it.

This module is the reader infrastructure and the per-cell
query API; the gain analysis builds on top of it.
"""

from .hb_dataset import (
    HBDataset,
    HBDatasetError,
    HBHarmonic,
    HBSpectrum,
    IDLER_MIX_2TONE,
    PUMP_MIX_2TONE,
    SIGNAL_MIX_2TONE,
    hb_dataset_from_raw,
    read_hb_dataset,
)
from .per_cell import (
    CellInfo,
    CellMap,
    CellResolutionError,
    build_cell_map,
    cell_boundary_current,
    cell_branch_current,
    cell_count,
    cell_net,
    cell_node_voltage,
    cell_varactor_current,
)
from .power import (
    cascade_power_gain_dB,
    cascade_voltage_ratio_gain_dB,
    cell_boundary_power,
    cell_power_gain_dB,
    cell_voltage_ratio_gain_dB,
    varactor_power,
)
from .energy_conservation import (
    DEFAULT_ENERGY_TOLERANCE_DB_REL_INPUT,
    cascade_energy_residuals,
    cell_energy_residual,
)
from .floquet_comparison import (
    DEFAULT_HARMONIC_TOLERANCE_DB,
    VaractorModelParameters,
    compare_cascade_harmonics,
    compare_varactor_harmonics,
    extract_varactor_parameters,
    harmonic_band,
    harmonic_tolerance_dB,
)
from .sweep import gain_vs_pump_power, gain_vs_signal_frequency

__all__ = [
    "CellInfo",
    "CellMap",
    "CellResolutionError",
    "DEFAULT_ENERGY_TOLERANCE_DB_REL_INPUT",
    "DEFAULT_HARMONIC_TOLERANCE_DB",
    "HBDataset",
    "HBDatasetError",
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
]


