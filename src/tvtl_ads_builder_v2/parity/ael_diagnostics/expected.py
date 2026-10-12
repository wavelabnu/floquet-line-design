"""TVTL analytical expected response for the AEL diagnostics.

For each diagnostic spec this computes the analytical |S21| across the
1-10 GHz sweep, reusing the main parity gate's TVTL simulator
(:func:`tvtl_ads_builder_v2.tvtl_simulate.simulate_s_parameters`). The
result is the Path-1 reference each diagnostic's hpeesofsim run -- and
ultimately each AEL-built schematic -- must agree with.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...device_spec import DeviceSpec
from ...tvtl_simulate import simulate_s_parameters
from ...tvtl_simulate.s_parameters import s_parameters_to_dB


def compute_tvtl_expected(spec: DeviceSpec) -> dict[str, Any]:
    """Compute the TVTL analytical |S21| (dB + phase) for ``spec``.

    Returns
    -------
    dict
        ``{"freq_Hz": ndarray, "S21_dB": ndarray, "S21_phase_deg":
        ndarray}`` -- all finite, real-valued arrays over the spec's
        frequency grid.
    """

    sim = simulate_s_parameters(spec)
    s21 = np.asarray(sim["S21"], dtype=np.complex128)
    return {
        "freq_Hz": np.asarray(sim["frequencies_Hz"], dtype=float),
        "S21_dB": s_parameters_to_dB(s21),
        "S21_phase_deg": np.degrees(np.angle(s21)),
    }


def write_tvtl_expected_csv(spec: DeviceSpec, path: str | Path) -> Path:
    """Write the TVTL expected response to ``tvtl_expected.csv``.

    Columns: ``freq_Hz``, ``S21_dB``, ``S21_phase_deg``.
    """

    expected = compute_tvtl_expected(spec)
    freq = expected["freq_Hz"]
    s21_dB = expected["S21_dB"]
    phase = expected["S21_phase_deg"]
    lines = ["freq_Hz,S21_dB,S21_phase_deg\n"]
    for index in range(freq.size):
        lines.append(
            f"{float(freq[index]):.6e},"
            f"{float(s21_dB[index]):.6f},"
            f"{float(phase[index]):.6f}\n"
        )
    path = Path(path)
    path.write_text("".join(lines), encoding="utf-8")
    return path


