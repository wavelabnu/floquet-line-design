"""Run TVTL analytical + ADS hpeesofsim on the same spec and compare.

The check always populates the TVTL side (it is deterministic and does
not require ADS); the ADS side is populated only when the runner can
find a simulator. If ADS is unavailable, ``pass=False`` with
``reason='ads_blocked'`` -- the parity gate is not satisfied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..device_spec import DeviceSpec, dump_device_spec
from ..netlist import emit_hpeesofsim_netlist
from ..simulate import parse_s_parameter_results, run_hpeesofsim
from ..tvtl_simulate import simulate_s_parameters
from ..tvtl_simulate.s_parameters import s_parameters_to_dB
from .tolerance import PARITY_TOLERANCE_DB_S21_LEVEL_1


def parity_check(
    spec: DeviceSpec,
    output_dir: str | Path,
    *,
    ads_environment: Mapping[str, Any] | None = None,
    tolerance_dB_S21: float = PARITY_TOLERANCE_DB_S21_LEVEL_1,
    level_label: str = "parity_level_1",
) -> dict[str, Any]:
    """Run TVTL + ADS on ``spec`` and report the comparison.

    Parameters
    ----------
    spec : DeviceSpec
        Device spec for the level under test (an MLIN cascade for
        Levels 1-2).
    output_dir : str or Path
        Workspace directory for the run. Files written:
        ``device_spec.json``, ``hpeesofsim.ckt``, ``tvtl_s_parameters.csv``,
        plus the simulator's ``.ds`` and log tails on success.
    ads_environment : Mapping or None
        Optional ADS environment override; otherwise the runner reads
        ``HPEESOF_DIR`` from the environment.
    tolerance_dB_S21 : float
        Max permitted ``|delta_dB|`` across the band. Defaults to
        :data:`PARITY_TOLERANCE_DB_S21_LEVEL_1` (0.5 dB).
    level_label : str
        Identifier written into the result's ``label`` field
        (``"parity_level_1"``, ``"parity_level_2"``, ...).

    Returns
    -------
    dict
        See module docstring of :mod:`tvtl_ads_builder_v2.parity` for
        the field list.
    """

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tvtl_result: dict[str, Any] | None = None
    tvtl_error: str | None = None
    try:
        tvtl_result = simulate_s_parameters(spec)
    except Exception as exc:
        tvtl_error = repr(exc)

    spec_path = dump_device_spec(spec, output_dir / "device_spec.json")
    netlist_path = output_dir / "hpeesofsim.ckt"
    netlist_path.write_text(emit_hpeesofsim_netlist(spec), encoding="utf-8")
    if tvtl_result is not None:
        _write_tvtl_csv(output_dir / "tvtl_s_parameters.csv", tvtl_result)

    ads_run = run_hpeesofsim(netlist_path, ads_environment=ads_environment)
    ads_result: dict[str, Any]
    if ads_run["status"] == "completed" and ads_run["dataset_path"]:
        ads_result = parse_s_parameter_results(ads_run["dataset_path"])
    else:
        ads_result = {
            "frequencies_Hz": np.zeros(0, dtype=float),
            "S11": np.zeros(0, dtype=np.complex128),
            "S21": np.zeros(0, dtype=np.complex128),
            "S12": np.zeros(0, dtype=np.complex128),
            "S22": np.zeros(0, dtype=np.complex128),
            "analysis_label": ads_run["status"],
        }

    if tvtl_result is None:
        return {
            "tvtl": {"error_message": tvtl_error},
            "ads": ads_result,
            "frequencies_Hz": np.zeros(0, dtype=float),
            "tvtl_S21_dB": np.zeros(0, dtype=float),
            "ads_S21_dB": np.zeros(0, dtype=float),
            "delta_dB_grid": np.zeros(0, dtype=float),
            "max_abs_delta_dB": float("nan"),
            "tolerance_dB": float(tolerance_dB_S21),
            "pass": False,
            "reason": "tvtl_error",
            "label": level_label,
            "spec_path": str(spec_path),
            "netlist_path": str(netlist_path),
            "ads_run": ads_run,
        }

    frequencies = np.asarray(tvtl_result["frequencies_Hz"], dtype=float)
    tvtl_S21_dB = s_parameters_to_dB(tvtl_result["S21"])
    if (
        ads_result["analysis_label"] != "ads_hpeesofsim"
        or ads_result["S21"].size == 0
    ):
        return {
            "tvtl": tvtl_result,
            "ads": ads_result,
            "frequencies_Hz": frequencies,
            "tvtl_S21_dB": tvtl_S21_dB,
            "ads_S21_dB": np.full_like(tvtl_S21_dB, float("nan")),
            "delta_dB_grid": np.full_like(tvtl_S21_dB, float("nan")),
            "max_abs_delta_dB": float("nan"),
            "tolerance_dB": float(tolerance_dB_S21),
            "pass": False,
            "reason": "ads_blocked",
            "label": level_label,
            "spec_path": str(spec_path),
            "netlist_path": str(netlist_path),
            "ads_run": ads_run,
        }

    ads_S21_dB = _resample_dB(
        source_freq=np.asarray(ads_result["frequencies_Hz"], dtype=float),
        source_S21=ads_result["S21"],
        target_freq=frequencies,
    )
    delta = ads_S21_dB - tvtl_S21_dB
    max_abs = float(np.nanmax(np.abs(delta))) if delta.size > 0 else float("nan")
    passed = bool(max_abs <= float(tolerance_dB_S21))
    return {
        "tvtl": tvtl_result,
        "ads": ads_result,
        "frequencies_Hz": frequencies,
        "tvtl_S21_dB": tvtl_S21_dB,
        "ads_S21_dB": ads_S21_dB,
        "delta_dB_grid": delta,
        "max_abs_delta_dB": max_abs,
        "tolerance_dB": float(tolerance_dB_S21),
        "pass": passed,
        "reason": "within_tolerance" if passed else "exceeds_tolerance",
        "label": level_label,
        "spec_path": str(spec_path),
        "netlist_path": str(netlist_path),
        "ads_run": ads_run,
    }


def _write_tvtl_csv(path: Path, tvtl_result: Mapping[str, Any]) -> None:
    freqs = np.asarray(tvtl_result["frequencies_Hz"], dtype=float)
    s21 = np.asarray(tvtl_result["S21"], dtype=np.complex128)
    s11 = np.asarray(tvtl_result["S11"], dtype=np.complex128)
    lines = ["freq_Hz,S21_mag_dB,S21_phase_deg,S11_mag_dB,S11_phase_deg\n"]
    s21_db = s_parameters_to_dB(s21)
    s11_db = s_parameters_to_dB(s11)
    s21_phase = np.degrees(np.angle(s21))
    s11_phase = np.degrees(np.angle(s11))
    for index, f in enumerate(freqs):
        lines.append(
            f"{float(f):.6e},{float(s21_db[index]):.6f},{float(s21_phase[index]):.6f},"
            f"{float(s11_db[index]):.6f},{float(s11_phase[index]):.6f}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def _resample_dB(
    source_freq: np.ndarray, source_S21: np.ndarray, target_freq: np.ndarray
) -> np.ndarray:
    """Linearly interpolate ``|S21|_dB`` from source freqs onto target.

    The ADS simulator's frequency grid is set to match the spec's, so
    in practice the two grids are identical and this is a no-op. The
    fallback is here so a small grid mismatch (last-bin rounding) does
    not silently shift the delta.
    """

    if source_freq.size == 0 or source_S21.size == 0:
        return np.full(target_freq.shape, float("nan"))
    if (
        source_freq.size == target_freq.size
        and np.allclose(source_freq, target_freq, rtol=0.0, atol=1.0)
    ):
        return s_parameters_to_dB(source_S21)
    magnitude_dB = s_parameters_to_dB(source_S21)
    order = np.argsort(source_freq)
    return np.interp(target_freq, source_freq[order], magnitude_dB[order])


