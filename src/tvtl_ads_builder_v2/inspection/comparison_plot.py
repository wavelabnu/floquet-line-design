"""TVTL-vs-ADS comparison plot and CSV for one canonical-circuit workspace.

The CSV (``comparison_plot.csv``) is written unconditionally: it holds
the |S21| sweep for TVTL and ADS plus their dB difference, so the
comparison data is always available even without a plotting library.

The PNG (``comparison_plot.png``) is written only when ``matplotlib``
is importable in the running interpreter (it is imported lazily); the
PNG is therefore a best-effort extra. When it is unavailable the CSV
stands in for it and the workspace README says so.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def write_comparison_plot(
    workspace_dir: str | Path, result: Mapping[str, Any]
) -> dict[str, Any]:
    """Write ``comparison_plot.csv`` and, if possible, ``comparison_plot.png``.

    Parameters
    ----------
    workspace_dir : str or Path
        The canonical-circuit workspace directory.
    result : Mapping
        A parity-check result dict (see
        :func:`tvtl_ads_builder_v2.parity.parity_check`). The keys used
        are ``frequencies_Hz``, ``tvtl_S21_dB``, ``ads_S21_dB``,
        ``delta_dB_grid``, ``tolerance_dB``, ``label`` and ``pass``.

    Returns
    -------
    dict
        ``{"csv_path": Path, "png_path": Path | None,
        "png_written": bool, "ads_available": bool}``.
    """

    workspace_dir = Path(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    frequencies = np.asarray(result.get("frequencies_Hz", []), dtype=float)
    tvtl_dB = np.asarray(result.get("tvtl_S21_dB", []), dtype=float)
    ads_dB = np.asarray(result.get("ads_S21_dB", []), dtype=float)
    delta_dB = np.asarray(result.get("delta_dB_grid", []), dtype=float)
    tolerance_dB = float(result.get("tolerance_dB", float("nan")))
    label = str(result.get("label", "circuit"))

    ads_available = bool(ads_dB.size) and bool(np.any(np.isfinite(ads_dB)))

    csv_path = workspace_dir / "comparison_plot.csv"
    _write_csv(csv_path, frequencies, tvtl_dB, ads_dB, delta_dB)

    png_path = workspace_dir / "comparison_plot.png"
    png_written = _write_png(
        png_path,
        frequencies=frequencies,
        tvtl_dB=tvtl_dB,
        ads_dB=ads_dB,
        delta_dB=delta_dB,
        tolerance_dB=tolerance_dB,
        label=label,
        passed=bool(result.get("pass", False)),
        max_abs_delta_dB=float(result.get("max_abs_delta_dB", float("nan"))),
        ads_available=ads_available,
    )
    return {
        "csv_path": csv_path,
        "png_path": png_path if png_written else None,
        "png_written": png_written,
        "ads_available": ads_available,
    }


def _write_csv(
    path: Path,
    frequencies: np.ndarray,
    tvtl_dB: np.ndarray,
    ads_dB: np.ndarray,
    delta_dB: np.ndarray,
) -> None:
    """Write the four-column comparison CSV."""

    n = frequencies.size
    lines = ["freq_GHz,tvtl_S21_dB,ads_S21_dB,abs_delta_dB\n"]
    for index in range(n):
        f_ghz = float(frequencies[index]) / 1.0e9
        tvtl = tvtl_dB[index] if index < tvtl_dB.size else float("nan")
        ads = ads_dB[index] if index < ads_dB.size else float("nan")
        delta = delta_dB[index] if index < delta_dB.size else float("nan")
        lines.append(
            f"{f_ghz:.6f},{_fmt(tvtl)},{_fmt(ads)},{_fmt(abs(delta) if math.isfinite(delta) else delta)}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def _fmt(value: float) -> str:
    return "NaN" if not math.isfinite(float(value)) else f"{float(value):.6f}"


def _write_png(
    path: Path,
    *,
    frequencies: np.ndarray,
    tvtl_dB: np.ndarray,
    ads_dB: np.ndarray,
    delta_dB: np.ndarray,
    tolerance_dB: float,
    label: str,
    passed: bool,
    max_abs_delta_dB: float,
    ads_available: bool,
) -> bool:
    """Render the two-panel comparison PNG if matplotlib is importable.

    Returns ``True`` if the PNG was written, ``False`` if matplotlib is
    unavailable (the CSV already carries the same data).
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    if frequencies.size == 0:
        return False

    f_ghz = frequencies / 1.0e9
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True)

    ax_top.plot(f_ghz, tvtl_dB, color="tab:blue", linestyle="-", label="TVTL")
    if ads_available:
        ax_top.plot(f_ghz, ads_dB, color="tab:red", linestyle="--", label="ADS")
    ax_top.set_ylabel("|S21| [dB]")
    ax_top.legend(loc="upper right")
    ax_top.grid(True, alpha=0.3)

    if ads_available:
        ax_bot.plot(f_ghz, np.abs(delta_dB), color="tab:purple", linestyle="-")
        if math.isfinite(tolerance_dB):
            ax_bot.axhline(
                tolerance_dB, color="tab:gray", linestyle=":",
                label=f"tolerance {tolerance_dB:g} dB",
            )
            ax_bot.legend(loc="upper right")
        ax_bot.set_ylabel("|TVTL - ADS| [dB]")
    else:
        ax_bot.text(
            0.5, 0.5, "ADS unavailable -- no delta to plot",
            ha="center", va="center", transform=ax_bot.transAxes,
        )
        ax_bot.set_ylabel("|TVTL - ADS| [dB]")
    ax_bot.set_xlabel("Frequency [GHz]")
    ax_bot.grid(True, alpha=0.3)

    if ads_available:
        verdict = "PASS" if passed else "FAIL"
        title = f"{label} -- max |delta| = {max_abs_delta_dB:.3f} dB [{verdict}]"
    else:
        title = f"{label} -- TVTL only (ADS unavailable)"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


