"""Local Floquet coupling proxy metrics.

These metrics are local branch and unit-cell screening proxies. They are not
full-device gain and do not validate amplifier performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .harmonics import current_spectrum
from .uc_matrix import FloquetMatrixSolution


@dataclass(frozen=True, slots=True)
class FloquetCouplingMetrics:
    signal_admittance_mag: float
    upper_sideband_mag: float
    lower_sideband_mag: float
    phase_matching_proxy: float = 0.0
    condition_number: float | None = None
    signal_frequency_hz: float | None = None
    pump_frequency_hz: float | None = None
    idler_frequency_hz: float | None = None
    upper_sideband_frequency_hz: float | None = None
    sideband_coupling_proxy: float | None = None
    degenerate_coupling_proxy: float | None = None
    convergence_status: str = "not_evaluated"
    metric_scope: str = "local_proxy"
    full_device_gain_available: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def sideband_balance(self) -> float:
        denom = max(self.upper_sideband_mag + self.lower_sideband_mag, np.finfo(float).tiny)
        return float(1.0 - abs(self.upper_sideband_mag - self.lower_sideband_mag) / denom)

    @property
    def coupling_proxy(self) -> float:
        if self.sideband_coupling_proxy is not None:
            return float(self.sideband_coupling_proxy)
        denom = max(self.signal_admittance_mag, np.finfo(float).tiny)
        return float((self.upper_sideband_mag + self.lower_sideband_mag) / denom)


def coupling_metrics_from_admittance(
    admittance_matrix: Any,
    *,
    signal_index: int,
    upper_index: int,
    lower_index: int,
    condition_number: float | None = None,
    phase_matching_proxy: float = 0.0,
    signal_frequency_hz: float | None = None,
    pump_frequency_hz: float | None = None,
    idler_frequency_hz: float | None = None,
    convergence_status: str = "computed",
) -> FloquetCouplingMetrics:
    matrix = np.asarray(admittance_matrix, dtype=np.complex128)
    if matrix.ndim != 2:
        raise ValueError("admittance_matrix must be two-dimensional.")
    idler = abs(float(pump_frequency_hz) - float(signal_frequency_hz)) if idler_frequency_hz is None and pump_frequency_hz is not None and signal_frequency_hz is not None else idler_frequency_hz
    upper_frequency = None if pump_frequency_hz is None or signal_frequency_hz is None else float(pump_frequency_hz) + float(signal_frequency_hz)
    coupling_proxy = float((abs(matrix[upper_index, signal_index]) + abs(matrix[lower_index, signal_index])) / max(abs(matrix[signal_index, signal_index]), np.finfo(float).tiny))
    degenerate_error = (
        None
        if pump_frequency_hz is None or signal_frequency_hz is None
        else abs(float(pump_frequency_hz) - 2.0 * float(signal_frequency_hz)) / max(float(signal_frequency_hz), np.finfo(float).tiny)
    )
    metrics = FloquetCouplingMetrics(
        signal_admittance_mag=float(abs(matrix[signal_index, signal_index])),
        upper_sideband_mag=float(abs(matrix[upper_index, signal_index])),
        lower_sideband_mag=float(abs(matrix[lower_index, signal_index])),
        phase_matching_proxy=float(phase_matching_proxy),
        condition_number=None if condition_number is None else float(condition_number),
        signal_frequency_hz=None if signal_frequency_hz is None else float(signal_frequency_hz),
        pump_frequency_hz=None if pump_frequency_hz is None else float(pump_frequency_hz),
        idler_frequency_hz=None if idler is None else float(idler),
        upper_sideband_frequency_hz=upper_frequency,
        sideband_coupling_proxy=coupling_proxy,
        degenerate_coupling_proxy=None if degenerate_error is None else float(1.0 / (1.0 + degenerate_error)),
        convergence_status=convergence_status,
    )
    if not np.all(np.isfinite([metrics.signal_admittance_mag, metrics.upper_sideband_mag, metrics.lower_sideband_mag])):
        raise FloatingPointError("Floquet coupling metrics contain non-finite values.")
    return metrics


def analytical_stub_gain_proxy(
    Qn: np.ndarray,
    ws: float,
    wm: float,
    N: int,
) -> dict:
    """Return the idler-to-signal current ratio as an analytical proxy.

    The metric is defined as

        gain_proxy_dB = 20 * log10( |I_{n=-1}| / |I_{n=0}| )

    with ``I_k = j*(ws + k*wm)*Q_k``. It is the ratio of the parametric idler
    current to the signal current produced by the analytical Floquet
    tridiagonal solve.

    This is an analytical proxy only. It is NOT equivalent to ADS-HB gain and
    must never be reported as a validated device-gain claim. Callers should
    propagate the ``"label"`` field to make the proxy nature explicit in any
    downstream artifact.

    Parameters
    ----------
    Qn : numpy.ndarray
        Charge harmonics ordered ``n = -N, ..., +N``.
    ws : float
        Signal angular frequency [rad/s].
    wm : float
        Modulation angular frequency [rad/s].
    N : int
        Truncation order; ``Qn`` must have length ``2*N + 1``.

    Returns
    -------
    dict
        Keys: ``"gain_proxy_dB"`` (float), ``"I_signal_amp"`` (complex,
        ``n = 0``), ``"I_idler_amp"`` (complex, ``n = -1``), and ``"label"``
        equal to ``"analytical_proxy_only"``.
    """

    truncation = int(N)
    if truncation < 1:
        raise ValueError("N must be at least 1 to access the n = -1 idler harmonic.")
    _, currents = current_spectrum(Qn, ws, wm, truncation)
    signal = complex(currents[truncation])
    idler = complex(currents[truncation - 1])
    signal_mag = abs(signal)
    idler_mag = abs(idler)
    if signal_mag <= 0.0:
        gain_db = float("-inf")
    elif idler_mag <= 0.0:
        gain_db = float("-inf")
    else:
        gain_db = float(20.0 * np.log10(idler_mag / signal_mag))
    return {
        "gain_proxy_dB": gain_db,
        "I_signal_amp": signal,
        "I_idler_amp": idler,
        "label": "analytical_proxy_only",
    }


def coupling_metrics_from_solution(
    solution: FloquetMatrixSolution,
    *,
    signal_harmonic: int = 0,
    upper_harmonic: int = 1,
    lower_harmonic: int = -1,
    phase_matching_proxy: float = 0.0,
    signal_frequency_hz: float | None = None,
    pump_frequency_hz: float | None = None,
) -> FloquetCouplingMetrics:
    index_to_row = {int(harmonic): row for row, harmonic in enumerate(solution.harmonic_indices)}
    return coupling_metrics_from_admittance(
        solution.admittance_matrix_s,
        signal_index=index_to_row[signal_harmonic],
        upper_index=index_to_row[upper_harmonic],
        lower_index=index_to_row[lower_harmonic],
        condition_number=solution.condition_number,
        phase_matching_proxy=phase_matching_proxy,
        signal_frequency_hz=signal_frequency_hz,
        pump_frequency_hz=pump_frequency_hz,
        convergence_status="computed",
    )


