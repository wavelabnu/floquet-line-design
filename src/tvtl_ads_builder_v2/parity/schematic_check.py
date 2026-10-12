"""Path 3 -- compare the AEL-built schematic's ADS simulation to TVTL.

Three validation paths reach the same circuit:

* **Path 1** -- device spec -> flat hpeesofsim ``.ckt`` -> ``.ds``.
  This is the parity gate; it currently passes at sub-percent.
* **Path 2** -- device spec -> AEL deck -> schematic cellview (visual).
* **Path 3** -- the AEL deck ends with ``de_analyze()``, which simulates
  that schematic and writes a second ``.ds``. Comparing Path 3 to TVTL
  (and to Path 1) confirms the schematic's *wiring* is electrically
  what the device spec describes -- numerically, not by eyeballing
  pin markers.

Path 3 is **opt-in and purely additive**. The parity CLIs' optional
``--with-schematic-sim`` flag drives it. When Path 3 is not run, or the
schematic dataset is absent, the parity gate is exactly Path 1. Path 3
can fail the gate only on a *real numerical disagreement* of a dataset
that was actually produced -- never on a missing dataset or absent ADS.

ADS writes a simulated schematic's dataset into the workspace's
``data/`` directory as ``<cell>.ds`` (the cellview-name convention --
confirmed by the reference workspace's ``data/reference_circuit.ds``).
"""

from __future__ import annotations

import csv
import os
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..simulate import parse_s_parameter_results
from .check import _resample_dB

# Optional reference ADS workspace. In the public copy this is supplied by
# environment variable when Path-3 schematic comparison is used.
REFERENCE_WORKSPACE_PATH = Path(
    os.environ.get("TVTL_AEL_REFERENCE_WORKSPACE", Path.cwd() / "ads_reference_workspace")
)


def schematic_dataset_path(
    cell: str,
    reference_workspace: str | Path = REFERENCE_WORKSPACE_PATH,
) -> Path:
    """Return the ``.ds`` path ADS writes for a simulated schematic.

    Parameters
    ----------
    cell : str
        The cellview (canonical circuit) name.
    reference_workspace : str or Path
        The ADS workspace the schematic cellview lives in.

    Returns
    -------
    Path
        ``<reference_workspace>/data/<cell>.ds``.
    """

    return Path(reference_workspace) / "data" / f"{cell}.ds"


def compare_to_schematic(
    parity_result: Mapping[str, Any],
    schematic_ds_path: str | Path,
    tolerance_dB: float,
) -> dict[str, Any]:
    """Three-way comparison columns for one circuit.

    Parameters
    ----------
    parity_result : Mapping
        A :func:`tvtl_ads_builder_v2.parity.parity_check` result -- the
        Path-1 result, supplying ``frequencies_Hz``, ``tvtl_S21_dB`` and
        ``ads_S21_dB``.
    schematic_ds_path : str or Path
        The Path-3 schematic dataset (``data/<cell>.ds``).
    tolerance_dB : float
        The level's ``|delta_S21|`` tolerance.

    Returns
    -------
    dict
        New keys -- ``ads_schematic_S21_dB``, ``delta_schem_dB_grid``,
        ``max_abs_delta_schem_dB``, ``delta_paths_dB_grid``,
        ``max_abs_delta_paths_dB``, ``schematic_pass``,
        ``schematic_status``. ``parity_result`` is not mutated.

    Notes
    -----
    ``schematic_status`` is ``"compared"`` only when a non-empty dataset
    was read; otherwise ``"no_dataset"`` / ``"empty_dataset"`` and
    ``schematic_pass`` is ``False`` -- but a non-``"compared"`` status
    must never be treated as a gate failure (it is missing
    infrastructure, not a numerical disagreement).
    """

    freqs = np.asarray(parity_result.get("frequencies_Hz", []), dtype=float)
    tvtl_dB = np.asarray(parity_result.get("tvtl_S21_dB", []), dtype=float)
    ads_flat_dB = np.asarray(parity_result.get("ads_S21_dB", []), dtype=float)

    path = Path(schematic_ds_path)
    if not path.is_file():
        return _blank_schematic("no_dataset", freqs.shape)

    parsed = parse_s_parameter_results(str(path))
    source_freq = np.asarray(parsed.get("frequencies_Hz", []), dtype=float)
    source_S21 = np.asarray(parsed.get("S21", []))
    if source_freq.size == 0 or source_S21.size == 0:
        return _blank_schematic("empty_dataset", freqs.shape)

    schem_dB = _resample_dB(source_freq, source_S21, freqs)
    delta_schem = schem_dB - tvtl_dB
    delta_paths = schem_dB - ads_flat_dB
    max_schem = _finite_max_abs(delta_schem)
    max_paths = _finite_max_abs(delta_paths)
    passed = math.isfinite(max_schem) and max_schem <= float(tolerance_dB)
    return {
        "ads_schematic_S21_dB": schem_dB,
        "delta_schem_dB_grid": delta_schem,
        "max_abs_delta_schem_dB": max_schem,
        "delta_paths_dB_grid": delta_paths,
        "max_abs_delta_paths_dB": max_paths,
        "schematic_pass": passed,
        "schematic_status": "compared",
    }


def run_schematic_validation(
    output_root: str | Path,
    level_number: int,
    specs: Sequence[Any],
    results: Sequence[Mapping[str, Any]],
    tolerance_dB: float,
    *,
    reference_workspace: str | Path = REFERENCE_WORKSPACE_PATH,
) -> dict[str, Any]:
    """Collect Path-3 schematic datasets for a level and three-way compare.

    For each circuit it looks for the schematic dataset ADS would have
    written (``<reference_workspace>/data/<label>.ds``). Circuits whose
    dataset is present are three-way compared; absent ones are reported
    and never fail the gate. A three-way CSV is written to
    ``<output_root>/ads_builder_v2_level_<N>_three_way.csv``.

    This function does **not** launch ADS. The user runs each circuit's
    ``open_workspace.bat`` (the AEL deck self-bootstraps and now ends
    with ``de_analyze()``); this then collects what ADS produced.

    Returns
    -------
    dict
        ``{"status": "compared" | "no_datasets", "rows": list[dict],
        "n_compared": int, "any_schematic_fail": bool,
        "csv_path": Path}``.
    """

    output_root = Path(output_root)
    rows: list[dict[str, Any]] = []
    n_compared = 0
    any_fail = False
    for spec, result in zip(specs, results):
        label = str(getattr(spec, "label", result.get("label", "")))
        ds_path = schematic_dataset_path(label, reference_workspace)
        schem = compare_to_schematic(result, ds_path, tolerance_dB)
        status = str(schem["schematic_status"])
        if status == "compared":
            n_compared += 1
            if not schem["schematic_pass"]:
                any_fail = True
        rows.append(
            {
                "label": label,
                "max_abs_delta_flat_dB": _fmt(result.get("max_abs_delta_dB")),
                "max_abs_delta_schem_dB": _fmt(schem["max_abs_delta_schem_dB"]),
                "max_abs_delta_paths_dB": _fmt(schem["max_abs_delta_paths_dB"]),
                "tolerance_dB": _fmt(tolerance_dB),
                "schematic_status": status,
                "schematic_pass": schem["schematic_pass"],
                "schematic_dataset": str(ds_path),
            }
        )

    csv_path = output_root / f"ads_builder_v2_level_{level_number}_three_way.csv"
    _write_three_way_csv(csv_path, rows)
    return {
        "status": "compared" if n_compared else "no_datasets",
        "rows": rows,
        "n_compared": n_compared,
        "any_schematic_fail": any_fail,
        "csv_path": csv_path,
    }


def _blank_schematic(status: str, shape: tuple[int, ...]) -> dict[str, Any]:
    """Path-3 columns for a circuit with no usable schematic dataset."""

    nan_grid = np.full(shape, float("nan"))
    return {
        "ads_schematic_S21_dB": nan_grid,
        "delta_schem_dB_grid": nan_grid,
        "max_abs_delta_schem_dB": float("nan"),
        "delta_paths_dB_grid": nan_grid,
        "max_abs_delta_paths_dB": float("nan"),
        "schematic_pass": False,
        "schematic_status": status,
    }


def _finite_max_abs(grid: np.ndarray) -> float:
    """Max ``|grid|`` over finite entries, or NaN if there are none."""

    arr = np.asarray(grid, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.max(np.abs(finite))) if finite.size else float("nan")


def _fmt(value: Any) -> str:
    """Format a delta value for the three-way CSV."""

    try:
        v = float(value)
    except (TypeError, ValueError):
        return "NaN"
    return "NaN" if not math.isfinite(v) else f"{v:.6f}"


def _write_three_way_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the per-level three-way comparison CSV."""

    fieldnames = [
        "label",
        "max_abs_delta_flat_dB",
        "max_abs_delta_schem_dB",
        "max_abs_delta_paths_dB",
        "tolerance_dB",
        "schematic_status",
        "schematic_pass",
        "schematic_dataset",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})



