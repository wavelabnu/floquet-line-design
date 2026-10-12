"""Parity gate between TVTL analytical and ADS hpeesofsim.

Every level concludes with one of these gates. The tolerance is
documented here (and only here), and is never loosened to make a test
pass.
"""

from .canonical_circuits import (
    LEVEL_1_CANONICAL_CIRCUITS,
    LEVEL_1_LABEL,
    LEVEL_2_CANONICAL_CIRCUITS,
    LEVEL_2_LABEL,
    LEVEL_3_CANONICAL_CIRCUITS,
    LEVEL_3_LABEL,
    LEVEL_4_CANONICAL_CIRCUITS,
    LEVEL_4_LABEL,
    make_mlin_then_rlc,
    make_rlc_series_two_port,
    make_termg_mlin_termg,
    make_tee_with_open_zero_length_stub,
    make_tee_with_quarter_wave_short_stub_at_3p1GHz,
    make_tee_with_rlc_stub,
    make_tee_with_short_stub_paper_dims,
    make_tee_with_short_zero_length_stub,
    make_three_mlins_series,
    make_two_mlins_series,
    quarter_wave_length_m,
)
from .level10_ideal_c import (
    make_level10_ideal_c_cascade,
    make_level10_ideal_c_family,
)
from .level10_realistic_c import (
    level10_realistic_c_canonical,
    level10_realistic_c_control_port_canonical,
    realistic_q_poly_reference,
)
from .check import parity_check
from .schematic_check import (
    compare_to_schematic,
    run_schematic_validation,
    schematic_dataset_path,
)
from .tolerance import (
    PARITY_TOLERANCE_DB_S21_LEVEL_1,
    PARITY_TOLERANCE_DB_S21_LEVEL_2,
    PARITY_TOLERANCE_DB_S21_LEVEL_3,
    PARITY_TOLERANCE_DB_S21_LEVEL_4,
)

__all__ = [
    "LEVEL_1_CANONICAL_CIRCUITS",
    "LEVEL_1_LABEL",
    "LEVEL_2_CANONICAL_CIRCUITS",
    "LEVEL_2_LABEL",
    "LEVEL_3_CANONICAL_CIRCUITS",
    "LEVEL_3_LABEL",
    "LEVEL_4_CANONICAL_CIRCUITS",
    "LEVEL_4_LABEL",
    "PARITY_TOLERANCE_DB_S21_LEVEL_1",
    "PARITY_TOLERANCE_DB_S21_LEVEL_2",
    "PARITY_TOLERANCE_DB_S21_LEVEL_3",
    "PARITY_TOLERANCE_DB_S21_LEVEL_4",
    "level10_realistic_c_canonical",
    "level10_realistic_c_control_port_canonical",
    "make_level10_ideal_c_cascade",
    "make_level10_ideal_c_family",
    "make_mlin_then_rlc",
    "make_rlc_series_two_port",
    "make_termg_mlin_termg",
    "make_tee_with_open_zero_length_stub",
    "make_tee_with_quarter_wave_short_stub_at_3p1GHz",
    "make_tee_with_rlc_stub",
    "make_tee_with_short_stub_paper_dims",
    "make_tee_with_short_zero_length_stub",
    "make_three_mlins_series",
    "make_two_mlins_series",
    "parity_check",
    "realistic_q_poly_reference",
    "compare_to_schematic",
    "run_schematic_validation",
    "schematic_dataset_path",
    "quarter_wave_length_m",
]


