"""Diagnostic Floquet sweeps for parametric-instability analysis.

This module composes the harmonic-basis cascade builder
(:func:`tvtl.floquet.uc_model.build_unit_cell_block_abcd`) with the
eigenvalue-based stability helpers in :mod:`tvtl.floquet.stability` and the
device-level proxy gain in
:func:`tvtl.floquet.uc_model.floquet_uc_cascade_response`. Every sweep
produces dicts whose gain or stability-margin numbers are analytical
proxies only; the ``"label"`` field on each returned row always reads
``"analytical_proxy_only"``.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from .passband import passband_check_at_operating_point
from .stability import (
    cascade_gain_estimate,
    dominant_growth_rate,
)
from .uc_model import build_unit_cell_block_abcd, floquet_uc_cascade_response


def _validate_dS_fractions(values: Iterable[float]) -> list[float]:
    fractions = [float(value) for value in values]
    if not fractions:
        raise ValueError("dS_fractions must contain at least one value.")
    for value in fractions:
        if not (0.0 < value <= 0.95):
            raise ValueError("dS_fraction values must lie in (0, 0.95].")
    return fractions


def _row_at_operating_point(
    *,
    uc_geometry: Mapping[str, Any],
    varactor_params: Mapping[str, float],
    modulation_params: Mapping[str, Any],
    n_cells: int,
    N_harmonics: int,
    substrate: Mapping[str, Any],
) -> dict:
    """Evaluate cascade response + eigenvalue gain estimate at one point."""

    response = floquet_uc_cascade_response(
        uc_geometry,
        varactor_params,
        modulation_params,
        n_cells=int(n_cells),
        N_harmonics=int(N_harmonics),
        substrate=substrate,
    )
    block = build_unit_cell_block_abcd(
        uc_geometry, varactor_params, modulation_params, int(N_harmonics), substrate
    )
    eigen = cascade_gain_estimate(block, int(n_cells))
    return {
        "dS_fraction": float(modulation_params["dS_fraction"]),
        "lambda_max_abs": float(eigen["lambda_max_abs"]),
        "stability_margin": float(eigen["stability_margin"]),
        "gain_signal_proxy_dB": float(response["gain_signal_proxy_dB"]),
        "gain_idler_proxy_dB": float(response["gain_idler_proxy_dB"]),
        "gain_total_proxy_dB": float(response["gain_total_proxy_dB"]),
        "cascade_gain_estimate_dB": float(eigen["cascade_gain_estimate_dB"]),
        "label": "analytical_proxy_only",
    }


def _resolve_fm_ratio(modulation_params_base: Mapping[str, Any]) -> float:
    return float(modulation_params_base.get("fm_over_fs_ratio", 2.0))


def modulation_depth_sweep(
    uc_geometry: Mapping[str, Any],
    varactor_params: Mapping[str, float],
    modulation_params_base: Mapping[str, Any],
    substrate: Mapping[str, Any],
    n_cells: int,
    N_harmonics: int,
    dS_fractions: Iterable[float],
) -> list[dict]:
    """Sweep the modulation depth ``dS / S0`` at fixed signal frequency.

    Parameters
    ----------
    uc_geometry, varactor_params, modulation_params_base, substrate
        Same conventions as :func:`floquet_uc_cascade_response`.
        ``modulation_params_base`` must provide ``"fs"`` and ``"phi"``;
        ``"fm"`` is set to ``fm_over_fs_ratio * fs`` (default ratio 2.0).
    n_cells : int
        Number of cascade cells.
    N_harmonics : int
        Floquet truncation order.
    dS_fractions : iterable of floats
        Values of ``dS / S0`` to evaluate. Each must lie in ``(0, 0.95]``
        so that ``S(t)`` stays positive throughout the modulation cycle.

    Returns
    -------
    list of dict
        One row per ``dS_fraction``; keys are documented in
        :func:`_row_at_operating_point`.
    """

    fractions = _validate_dS_fractions(dS_fractions)
    fs = float(modulation_params_base["fs"])
    phi = float(modulation_params_base.get("phi", 0.0))
    fm = float(modulation_params_base.get("fm", _resolve_fm_ratio(modulation_params_base) * fs))
    rows: list[dict] = []
    for fraction in fractions:
        modulation_params = {"fs": fs, "fm": fm, "dS_fraction": fraction, "phi": phi}
        rows.append(
            _row_at_operating_point(
                uc_geometry=uc_geometry,
                varactor_params=varactor_params,
                modulation_params=modulation_params,
                n_cells=n_cells,
                N_harmonics=N_harmonics,
                substrate=substrate,
            )
        )
    return rows


def frequency_modulation_map(
    uc_geometry: Mapping[str, Any],
    varactor_params: Mapping[str, float],
    modulation_params_base: Mapping[str, Any],
    substrate: Mapping[str, Any],
    n_cells: int,
    N_harmonics: int,
    fs_values_Hz: Iterable[float],
    dS_fractions: Iterable[float],
) -> list[dict]:
    """2D sweep over ``(fs, dS / S0)`` with ``fm = ratio * fs`` per point.

    Parameters
    ----------
    uc_geometry, varactor_params, modulation_params_base, substrate
        See :func:`floquet_uc_cascade_response`. The base modulation params
        contribute ``"phi"`` (default 0) and ``"fm_over_fs_ratio"`` (default
        2.0). ``"fs"`` / ``"fm"`` in the base are ignored.
    n_cells, N_harmonics : int
        Cascade size and Floquet truncation order.
    fs_values_Hz, dS_fractions : iterables
        Sweep axes. ``fs_values_Hz`` values must be positive; ``dS_fractions``
        must lie in ``(0, 0.95]``.

    Returns
    -------
    list of dict
        One row per ``(fs, dS_fraction)`` cell. Each row carries the keys of
        :func:`modulation_depth_sweep` plus ``"fs_Hz"`` and ``"fm_Hz"``.
    """

    fractions = _validate_dS_fractions(dS_fractions)
    fs_values = [float(value) for value in fs_values_Hz]
    if not fs_values:
        raise ValueError("fs_values_Hz must contain at least one frequency.")
    if any(value <= 0.0 for value in fs_values):
        raise ValueError("All entries of fs_values_Hz must be positive.")
    ratio = _resolve_fm_ratio(modulation_params_base)
    phi = float(modulation_params_base.get("phi", 0.0))
    rows: list[dict] = []
    for fs in fs_values:
        fm = ratio * fs
        for fraction in fractions:
            modulation_params = {"fs": fs, "fm": fm, "dS_fraction": fraction, "phi": phi}
            row = _row_at_operating_point(
                uc_geometry=uc_geometry,
                varactor_params=varactor_params,
                modulation_params=modulation_params,
                n_cells=n_cells,
                N_harmonics=N_harmonics,
                substrate=substrate,
            )
            row["fs_Hz"] = fs
            row["fm_Hz"] = fm
            rows.append(row)
    return rows


def threshold_search(
    uc_geometry: Mapping[str, Any],
    varactor_params: Mapping[str, float],
    modulation_params_base: Mapping[str, Any],
    substrate: Mapping[str, Any],
    n_cells: int,
    N_harmonics: int,
    fs_values_Hz: Iterable[float],
    dS_grid_for_search: Iterable[float],
    margin_target: float = 0.0,
) -> list[dict]:
    """Find, for each ``fs``, the smallest ``dS / S0`` reaching a margin.

    The dominant per-cell amplification ``max_k |lambda_k|`` from
    :func:`tvtl.floquet.stability.dominant_growth_rate` is evaluated on the
    grid. Starting from the smallest ``dS``, the search reports the lowest
    ``dS_fraction`` at which ``stability_margin <= margin_target`` (i.e.,
    ``|lambda_max| >= 1 - margin_target``). The threshold is linearly
    interpolated between the bracketing grid points; if no point reaches it,
    ``dS_threshold`` is ``NaN``.

    Parameters
    ----------
    uc_geometry, varactor_params, modulation_params_base, substrate
        See :func:`frequency_modulation_map`.
    n_cells, N_harmonics : int
        Cascade size and Floquet truncation order. ``n_cells`` does not
        enter the threshold (eigenvalues do not depend on ``n_cells``); it
        is accepted only to keep the signature symmetric with the other
        sweeps.
    fs_values_Hz : iterable
        Signal frequencies at which to search.
    dS_grid_for_search : iterable
        Monotonically increasing grid of ``dS / S0`` values in ``(0, 0.95]``.
    margin_target : float
        Threshold definition: ``0.0`` is the instability boundary
        (``|lambda_max| = 1``); ``0.1`` is a 10% safety margin
        (``|lambda_max| = 0.9``).

    Returns
    -------
    list of dict
        One row per ``fs`` with keys ``"fs_Hz"``, ``"dS_threshold"`` (NaN if
        not reached), ``"lambda_max_at_threshold"``, and ``"label"``.
    """

    grid = _validate_dS_fractions(dS_grid_for_search)
    sorted_grid = sorted(grid)
    if any(b - a <= 0.0 for a, b in zip(sorted_grid, sorted_grid[1:])):
        raise ValueError("dS_grid_for_search must contain strictly increasing values.")
    fs_values = [float(value) for value in fs_values_Hz]
    if not fs_values:
        raise ValueError("fs_values_Hz must contain at least one frequency.")
    target = float(margin_target)
    threshold_lambda = 1.0 - target

    ratio = _resolve_fm_ratio(modulation_params_base)
    phi = float(modulation_params_base.get("phi", 0.0))

    rows: list[dict] = []
    for fs in fs_values:
        fm = ratio * fs
        magnitudes: list[float] = []
        for fraction in sorted_grid:
            modulation_params = {"fs": fs, "fm": fm, "dS_fraction": fraction, "phi": phi}
            block = build_unit_cell_block_abcd(
                uc_geometry, varactor_params, modulation_params, int(N_harmonics), substrate
            )
            magnitudes.append(float(dominant_growth_rate(block)))
        threshold: float = float("nan")
        lambda_at_threshold: float = float("nan")
        for index in range(len(sorted_grid)):
            if magnitudes[index] >= threshold_lambda:
                if index == 0:
                    threshold = sorted_grid[0]
                    lambda_at_threshold = magnitudes[0]
                else:
                    lower_frac = sorted_grid[index - 1]
                    upper_frac = sorted_grid[index]
                    lower_mag = magnitudes[index - 1]
                    upper_mag = magnitudes[index]
                    span = upper_mag - lower_mag
                    if abs(span) < 1.0e-15:
                        threshold = upper_frac
                        lambda_at_threshold = upper_mag
                    else:
                        weight = (threshold_lambda - lower_mag) / span
                        threshold = float(lower_frac + weight * (upper_frac - lower_frac))
                        lambda_at_threshold = float(lower_mag + weight * span)
                break
        rows.append(
            {
                "fs_Hz": fs,
                "dS_threshold": threshold,
                "lambda_max_at_threshold": lambda_at_threshold,
                "label": "analytical_proxy_only",
            }
        )
    return rows


def design_gain_spectrum(
    uc_geometry: Mapping[str, Any],
    varactor_params: Mapping[str, float],
    substrate: Mapping[str, Any],
    fs_grid_Hz: Iterable[float],
    fm_over_fs: float,
    dS_fraction: float,
    phi: float,
    n_cells: int,
    N_harmonics: int,
    passband_threshold_dB: float = -10.0,
    topology: Any = None,
) -> dict:
    """Evaluate the cascade-Floquet gain spectrum of a single design.

    For each ``fs`` in ``fs_grid_Hz`` the pump is set to ``fm = fm_over_fs * fs``
    and the cascade response is evaluated. The output combines per-fs
    cascade proxy gains with a passive passband check at every operating
    point so that downstream peak detection can flag artifactual gain.

    Parameters
    ----------
    uc_geometry : Mapping
        ``"main_segments"``, ``"stub_position"`` and optional
        ``"stub_segments"`` (same schema as
        :func:`tvtl.floquet.uc_model.floquet_uc_cascade_response`).
    varactor_params : Mapping
        ``"C0"`` [F], ``"R"`` [Ohm], ``"L"`` [H].
    substrate : Mapping
        ``"eps_r"``, ``"h"`` [m]; optional ``"copper_thickness"`` [m].
    fs_grid_Hz : iterable of float
        Strictly positive signal frequencies.
    fm_over_fs : float
        Modulation/signal frequency ratio (typical: 2.0).
    dS_fraction : float
        Modulation depth ``dS / S0`` in ``(0, 0.95]``.
    phi : float
        Modulation phase [rad].
    n_cells : int
        Cascade size.
    N_harmonics : int
        Floquet truncation order.
    passband_threshold_dB : float
        Minimum passive ``|S21|`` (dB) at each operating-point frequency to
        be considered "in the passband".

    Returns
    -------
    dict
        Keys ``"fs_grid_Hz"`` (ndarray), ``"gain_signal_proxy_dB"``,
        ``"gain_idler_proxy_dB"``, ``"gain_total_proxy_dB"`` (ndarrays),
        ``"passband_pass_by_fs"`` (bool ndarray),
        ``"passband_details_by_fs"`` (list of dicts) and ``"label"``.
    """

    if float(fm_over_fs) <= 0.0:
        raise ValueError("fm_over_fs must be positive.")
    if not (0.0 < float(dS_fraction) <= 0.95):
        raise ValueError("dS_fraction must lie in (0, 0.95].")
    fs_values = [float(value) for value in fs_grid_Hz]
    if not fs_values:
        raise ValueError("fs_grid_Hz must contain at least one frequency.")
    if any(value <= 0.0 for value in fs_values):
        raise ValueError("All entries of fs_grid_Hz must be positive.")

    signal = np.empty(len(fs_values), dtype=float)
    idler = np.empty(len(fs_values), dtype=float)
    total = np.empty(len(fs_values), dtype=float)
    pass_flags = np.empty(len(fs_values), dtype=bool)
    details: list[dict] = []
    for index, fs in enumerate(fs_values):
        fm = float(fm_over_fs) * fs
        modulation_params = {"fs": fs, "fm": fm, "dS_fraction": float(dS_fraction), "phi": float(phi)}
        response = floquet_uc_cascade_response(
            uc_geometry,
            varactor_params,
            modulation_params,
            n_cells=int(n_cells),
            N_harmonics=int(N_harmonics),
            substrate=substrate,
            topology=topology,
        )
        signal[index] = float(response["gain_signal_proxy_dB"])
        idler[index] = float(response["gain_idler_proxy_dB"])
        total[index] = float(response["gain_total_proxy_dB"])
        # Passband detection is intrinsically a per-cell property: it tests
        # whether a harmonic is propagating or evanescent in the unit cell.
        # Cumulative stop-band attenuation over many cells would otherwise
        # mask propagating frequencies at the edge of a passband.
        check = passband_check_at_operating_point(
            uc_geometry,
            varactor_params,
            substrate,
            1,
            fs,
            fm,
            threshold_dB=float(passband_threshold_dB),
            topology=topology,
        )
        pass_flags[index] = bool(check["pass"])
        details.append(check)

    return {
        "fs_grid_Hz": np.asarray(fs_values, dtype=float),
        "gain_signal_proxy_dB": signal,
        "gain_idler_proxy_dB": idler,
        "gain_total_proxy_dB": total,
        "passband_pass_by_fs": pass_flags,
        "passband_details_by_fs": details,
        "label": "analytical_proxy_only",
    }


def find_spectrum_peaks(
    spectrum: Mapping[str, Any],
    gain_key: str = "gain_idler_proxy_dB",
    min_prominence_dB: float = 2.0,
    min_peak_dB: float = -30.0,
) -> list[dict]:
    """Locate local maxima in a gain spectrum with prominence filtering.

    A peak is a strict local maximum of the chosen gain array whose absolute
    value is at or above ``min_peak_dB`` and whose prominence relative to its
    immediate neighbours is at least ``min_prominence_dB``. Boundary points
    are excluded so the prominence check is well-defined. Per-fs passband
    membership is copied onto each peak's ``"classification"`` field:
    ``"valid"`` iff the passband check passed at that fs, otherwise
    ``"artifactual"``.

    Parameters
    ----------
    spectrum : Mapping
        Output of :func:`design_gain_spectrum`.
    gain_key : str
        Spectrum array used to detect peaks.
    min_prominence_dB : float
        Minimum drop on each side of the candidate peak for it to be
        accepted.
    min_peak_dB : float
        Minimum absolute peak value.

    Returns
    -------
    list of dict
        One entry per accepted peak, sorted by descending gain.
    """

    fs_grid = np.asarray(spectrum["fs_grid_Hz"], dtype=float)
    gains = np.asarray(spectrum[gain_key], dtype=float)
    passes = np.asarray(spectrum["passband_pass_by_fs"], dtype=bool)
    if fs_grid.shape != gains.shape or fs_grid.shape != passes.shape:
        raise ValueError("spectrum fields must share the same shape.")
    if fs_grid.size < 3:
        return []
    peaks: list[dict] = []
    for index in range(1, fs_grid.size - 1):
        value = float(gains[index])
        left = float(gains[index - 1])
        right = float(gains[index + 1])
        if value <= left or value <= right:
            continue
        if value < float(min_peak_dB):
            continue
        prominence = float(min(value - left, value - right))
        if prominence < float(min_prominence_dB):
            continue
        peaks.append(
            {
                "fs_Hz": float(fs_grid[index]),
                "fs_index": int(index),
                "gain_dB": value,
                "prominence_dB": prominence,
                "passband_pass": bool(passes[index]),
                "classification": "valid" if bool(passes[index]) else "artifactual",
            }
        )
    peaks.sort(key=lambda item: item["gain_dB"], reverse=True)
    return peaks


def _interp_value_at(fs_grid: np.ndarray, values: np.ndarray, target_fs: float) -> float:
    if fs_grid.size == 0:
        return float("nan")
    if float(target_fs) <= float(fs_grid[0]):
        return float(values[0])
    if float(target_fs) >= float(fs_grid[-1]):
        return float(values[-1])
    return float(np.interp(float(target_fs), fs_grid, values))


def _nearest_index(fs_grid: np.ndarray, target_fs: float) -> int:
    return int(np.argmin(np.abs(fs_grid - float(target_fs))))


def classify_design_spectrum(
    spectrum: Mapping[str, Any],
    target_fs_Hz: float,
    target_window_fractional: float = 0.05,
    min_prominence_dB: float = 2.0,
    min_peak_dB: float = -30.0,
) -> dict:
    """Diagnose a design's gain spectrum relative to a target frequency.

    Calls :func:`find_spectrum_peaks` and labels each peak as valid or
    artifactual. Then extracts the highest valid peak; whether that peak
    lies inside a fractional window around ``target_fs_Hz``; the
    interpolated gain at the target; whether the target operating point is
    feasible; and whether any artifactual peak dwarfs the largest valid
    peak by more than 6 dB.

    Parameters
    ----------
    spectrum : Mapping
        Output of :func:`design_gain_spectrum`.
    target_fs_Hz : float
        Operating frequency of interest. Must be positive.
    target_window_fractional : float
        Fractional half-width around ``target_fs_Hz`` defining "near target".
        Must be positive.
    min_prominence_dB, min_peak_dB : float
        Peak-detection thresholds passed to :func:`find_spectrum_peaks`.

    Returns
    -------
    dict
        Diagnostic dictionary; the ``"label"`` field carries the proxy
        disclaimer.
    """

    if float(target_fs_Hz) <= 0.0:
        raise ValueError("target_fs_Hz must be positive.")
    if float(target_window_fractional) <= 0.0:
        raise ValueError("target_window_fractional must be positive.")

    peaks = find_spectrum_peaks(
        spectrum,
        gain_key="gain_idler_proxy_dB",
        min_prominence_dB=float(min_prominence_dB),
        min_peak_dB=float(min_peak_dB),
    )
    valid_peaks = [peak for peak in peaks if peak["classification"] == "valid"]
    artifactual_peaks = [peak for peak in peaks if peak["classification"] != "valid"]

    fs_grid = np.asarray(spectrum["fs_grid_Hz"], dtype=float)
    idler_array = np.asarray(spectrum["gain_idler_proxy_dB"], dtype=float)
    passes = np.asarray(spectrum["passband_pass_by_fs"], dtype=bool)

    window_half = float(target_window_fractional) * float(target_fs_Hz)

    def _in_window(peak: Mapping[str, Any]) -> bool:
        return abs(float(peak["fs_Hz"]) - float(target_fs_Hz)) <= window_half

    peak_in_target_window: dict | None = None
    for peak in valid_peaks:
        if _in_window(peak):
            peak_in_target_window = peak
            break

    highest_valid_peak: dict | None = valid_peaks[0] if valid_peaks else None
    highest_valid_peak_in_target = bool(highest_valid_peak is not None and _in_window(highest_valid_peak))

    has_dominant_artifact = False
    if artifactual_peaks:
        highest_artifact_gain = artifactual_peaks[0]["gain_dB"]
        reference = highest_valid_peak["gain_dB"] if highest_valid_peak is not None else float("-inf")
        if highest_artifact_gain > reference + 6.0:
            has_dominant_artifact = True

    gain_at_target_dB = _interp_value_at(fs_grid, idler_array, float(target_fs_Hz))
    nearest = _nearest_index(fs_grid, float(target_fs_Hz))
    feasible_at_target = bool(passes[nearest])

    return {
        "peaks": peaks,
        "valid_peaks": valid_peaks,
        "artifactual_peaks": artifactual_peaks,
        "peak_in_target_window": peak_in_target_window,
        "highest_valid_peak": highest_valid_peak,
        "highest_valid_peak_in_target": highest_valid_peak_in_target,
        "has_dominant_artifact": has_dominant_artifact,
        "gain_at_target_dB": gain_at_target_dB,
        "feasible_at_target": feasible_at_target,
        "label": "analytical_proxy_only",
    }



