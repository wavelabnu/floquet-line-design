"""Shared manuscript figure style + provenance helpers.

Single-column manuscript look shared by every generator in this directory, so
the panels render identically:

* Arial (Helvetica, then DejaVu Sans as fallback);
* axis labels 7 pt, tick labels 6 pt at the FINAL saved size;
* single column = 89 mm (3.504 in) wide unless a panel notes otherwise;
* data lines 1.0 pt, axis spines 0.6 pt;
* NO in-panel titles, NO grids, tight layout;
* panel letters are OFF here (added at assembly time).

Palette
-------
The curve palette is PINNED here as named constants (the matplotlib ``tab10``
values the manuscript panels use). Generators must reference these names rather
than inlining hex literals, so a palette change is one edit in one place and the
sidecar records exactly which colors produced a panel.

Output
------
:func:`resolve_outdir` sends panels to ``results/figures/`` by default and
honours the ``TVTL_FIG_OUTDIR`` environment variable (the repo-wide convention
used by the other committed figure scripts) so tests can render headless into a
temporary directory. Sidecars record repo-relative paths and repo-relative
commands only.
"""
from __future__ import annotations

import json
import os
import subprocess
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# Sizing
# --------------------------------------------------------------------------- #
MM_PER_IN = 25.4
SINGLE_COL_MM = 89.0
SINGLE_COL_IN = SINGLE_COL_MM / MM_PER_IN  # 3.5039 in

LABEL_PT = 7.0    # axis labels at final size
TICK_PT = 6.0     # tick labels at final size
DATA_LW = 1.0     # data-line width (pt)
AXIS_LW = 0.6     # spine / tick width (pt)

# --------------------------------------------------------------------------- #
# Palette -- PINNED (matplotlib tab10). Reference these names, not raw hex.
# --------------------------------------------------------------------------- #
PALETTE_PINNED = True
C_PRIMARY = "#1f77b4"    # tab:blue   -- primary data curve
C_ACCENT = "#d62728"     # tab:red    -- peak / annotation accent, second series
C_GREY = "#7f7f7f"       # tab:gray   -- secondary / trough / open markers
C_GREY_DK = "#555555"    # darker grey -- secondary annotation text
C_REF = "#9e9e9e"        # reference / zero lines

PALETTE: dict[str, str] = {
    "primary": C_PRIMARY,
    "accent": C_ACCENT,
    "grey": C_GREY,
    "grey_dark": C_GREY_DK,
    "reference": C_REF,
}


def apply_style() -> None:
    """Install the manuscript rcParams (idempotent)."""
    warnings.filterwarnings("ignore", message="findfont")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": LABEL_PT,
        "axes.labelsize": LABEL_PT,
        "axes.titlesize": LABEL_PT,
        "xtick.labelsize": TICK_PT,
        "ytick.labelsize": TICK_PT,
        "legend.fontsize": TICK_PT,
        "lines.linewidth": DATA_LW,
        "axes.linewidth": AXIS_LW,
        "xtick.major.width": AXIS_LW,
        "ytick.major.width": AXIS_LW,
        "xtick.minor.width": AXIS_LW * 0.8,
        "ytick.minor.width": AXIS_LW * 0.8,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.minor.size": 1.4,
        "ytick.minor.size": 1.4,
        "axes.grid": False,
        "axes.titlepad": 3.0,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.3,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # embed TrueType (editable text in the vector PDF)
        "ps.fonttype": 42,
    })


def new_panel(width_in: float = SINGLE_COL_IN, height_in: float = 2.55):
    """A single-panel figure at the manuscript width: no title, no grid."""
    apply_style()
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LW)
    ax.tick_params(width=AXIS_LW)
    return fig, ax


def resolve_outdir() -> Path:
    """Panel output directory: ``$TVTL_FIG_OUTDIR`` if set, else results/figures."""
    env = os.environ.get("TVTL_FIG_OUTDIR")
    out = Path(env) if env else (REPO_ROOT / "results" / "figures")
    out.mkdir(parents=True, exist_ok=True)
    return out


def repo_relative(path: Path | str) -> str:
    """Repo-relative POSIX path, or the bare file name when outside the repo.

    Keeps sidecars free of machine-specific absolute paths even when a caller
    redirects output (e.g. a test rendering into a temporary directory).
    """
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.name


def git_short_hash(rel_path: str) -> str:
    """``git log -1 --format=%h -- <rel_path>`` (read-only). '' when unavailable."""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%h", "--", rel_path],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def save_panel(fig, out_base: Path, *, dpi_png: int = 600) -> dict[str, str]:
    """Write vector PDF + 600 dpi PNG. Returns repo-relative paths."""
    out_base.parent.mkdir(parents=True, exist_ok=True)
    pdf = out_base.with_suffix(".pdf")
    png = out_base.with_suffix(".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=dpi_png)
    plt.close(fig)
    return {"pdf": repo_relative(pdf), "png": repo_relative(png)}


def write_sidecar(
    out_base: Path,
    *,
    sources: Sequence[str],
    script: str,
    command: str,
    key_values: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write the JSON provenance sidecar next to the panel.

    ``sources`` are repo-relative paths; each gets its own ``git log -1`` short
    hash. Records the generating script, the exact (repo-relative) command, the
    pinned palette, and the key plotted values, so the panel is reconstructible
    from the committed tree.
    """
    payload: dict[str, Any] = {
        "panel": out_base.name,
        "generated_by": script,
        "command": command,
        "palette_pinned": PALETTE_PINNED,
        "palette": dict(PALETTE),
        "style": {
            "width_mm": SINGLE_COL_MM,
            "label_pt": LABEL_PT,
            "tick_pt": TICK_PT,
            "data_lw_pt": DATA_LW,
            "axis_lw_pt": AXIS_LW,
            "titles": "off",
            "grid": "off",
        },
        "sources": [
            {"path": s, "git_short_hash": git_short_hash(s)} for s in sources
        ],
        "key_values": dict(key_values),
    }
    if extra:
        payload.update(extra)
    sc = out_base.with_suffix(".sidecar.json")
    sc.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return sc
