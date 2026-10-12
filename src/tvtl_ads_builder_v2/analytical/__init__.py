"""Analytical helpers that lift the v2 framework's static ABCD machinery.

The harmonic ABCD lift is used by the Floquet solver: at each
analytical harmonic frequency ``omega_n = ws + n*wm``, the per-harmonic
stub admittance is evaluated via the same static transmission-line
formula the v2 framework already uses for single-frequency analysis
(Hammerstad-Kirschning Z_0/eps_eff + the standard input-impedance
recursion).
"""

from .floquet_stub import (
    StubSegment,
    Y_stub_at_freq,
    Y_stub_at_freq_with_shunts,
    build_stub_y_fn_from_spec,
    build_stub_y_fn_with_mid_shunts,
    extract_level9_stub_segments,
    extract_series_via_inductance_H,
    extract_stub_segments,
)
from .ideal_c_gain import ideal_c_realised_depth, level10_ideal_c_param_gain
from .level10_sparams import Level10SParamResult, level10_paper_sparams
from .mtee_blocks import (
    MTEE_VALIDITY_THRESHOLD,
    tee_harmonic_blocks,
    tee_reference_plane_single,
)
from .parametric_gain import (
    level10_parametric_gain_proxy,
    level10_signal_gain_proxy_dB,
)
from .realistic_c_gain import (
    level10_realistic_param_gain,
    realistic_pumped_unit_cell_abcd,
)

__all__ = [
    "Level10SParamResult",
    "MTEE_VALIDITY_THRESHOLD",
    "StubSegment",
    "Y_stub_at_freq",
    "Y_stub_at_freq_with_shunts",
    "build_stub_y_fn_from_spec",
    "build_stub_y_fn_with_mid_shunts",
    "extract_level9_stub_segments",
    "extract_series_via_inductance_H",
    "extract_stub_segments",
    "ideal_c_realised_depth",
    "level10_ideal_c_param_gain",
    "level10_paper_sparams",
    "level10_parametric_gain_proxy",
    "level10_realistic_param_gain",
    "level10_signal_gain_proxy_dB",
    "realistic_pumped_unit_cell_abcd",
    "tee_harmonic_blocks",
    "tee_reference_plane_single",
]


