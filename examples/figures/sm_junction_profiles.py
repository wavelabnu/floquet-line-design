"""SM panel -- per-cell junction-depth (m_j) profiles along the line.

Route-B junction depth vs cell index from the committed boundary records:

  * ``examples/real_parts/results/smv1245_mstar_arc_v2.json``
    ``step4_delivered_junction_depth/per_cell`` -- 6.2 GHz pump. The 29 dBm leg
    is solved; the 20 dBm leg is a sqrt(P) voltage rescale of it (recorded as
    ``note_20dBm``: "flagged, not re-solved"), so its profile SHAPE is identical.
  * ``examples/real_parts/results/smv1245_lobe2_prediction.json``
    ``junction_at_10p6_pump/{29dBm,20dBm}/per_cell`` -- 10.6 GHz pump, both
    powers independently solved.

Route B is the junction-side readout ``V(n_c{k}_rp) - V(n_c{k}_v)`` at the
(0, 1) mix, i.e. below the package divider.

For each curve a straight line is fitted to ``20*log10(m_j)`` vs cell index; the
slope is the along-line discriminator (|slope| > 0.3 dB/cell would indicate
along-line pump attenuation; flat indicates uniform arrival). The fit R^2,
peak-to-trough spread and the cell of minimum m_j are reported alongside,
because a single slope is a poor summary of a non-monotonic standing-wave
profile: the 10.6 GHz legs dip mid-line rather than decaying.

Outputs (into ``$TVTL_FIG_OUTDIR`` if set, else ``results/figures/``):
``sm_junction_profiles.pdf`` / ``.png`` / ``.sidecar.json``.

Usage:
    python examples/figures/sm_junction_profiles.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle  # noqa: E402

MSTAR = "examples/real_parts/results/smv1245_mstar_arc_v2.json"
LOBE2 = "examples/real_parts/results/smv1245_lobe2_prediction.json"

SLOPE_THRESHOLD_DB_PER_CELL = 0.3

LABEL_6P2_29 = "6.2 GHz, 29 dBm"
LABEL_6P2_20 = "6.2 GHz, 20 dBm (scaled)"
LABEL_10P6_29 = "10.6 GHz, 29 dBm"
LABEL_10P6_20 = "10.6 GHz, 20 dBm"


def load_curves() -> list[dict]:
    """Return one dict per (pump frequency, power) curve."""
    mstar = json.loads((REPO_ROOT / MSTAR).read_text(encoding="utf-8"))
    lobe2 = json.loads((REPO_ROOT / LOBE2).read_text(encoding="utf-8"))

    per_cell = mstar["step4_delivered_junction_depth"]["per_cell"]
    cells = [int(r["cell"]) for r in per_cell]
    curves = [
        {"label": LABEL_6P2_29, "fpump_GHz": 6.2, "power_dBm": 29, "solved": True,
         "cells": cells,
         "Vj": [float(r["Vj_direct_V"]) for r in per_cell],
         "mj": [float(r["m_j_29dBm"]) for r in per_cell]},
        {"label": LABEL_6P2_20, "fpump_GHz": 6.2, "power_dBm": 20, "solved": False,
         "cells": cells,
         "Vj": [float(r["Vj_direct_V"]) * float(r["m_j_20dBm_scaled"])
                / float(r["m_j_29dBm"]) for r in per_cell],
         "mj": [float(r["m_j_20dBm_scaled"]) for r in per_cell]},
    ]
    for key, power, label in (("29dBm", 29, LABEL_10P6_29), ("20dBm", 20, LABEL_10P6_20)):
        rows = lobe2["junction_at_10p6_pump"][key]["per_cell"]
        curves.append({"label": label, "fpump_GHz": 10.6, "power_dBm": power,
                       "solved": True, "cells": [int(r["cell"]) for r in rows],
                       "Vj": [float(r["Vj_direct_V"]) for r in rows],
                       "mj": [float(r["m_j"]) for r in rows]})
    return curves


def fit_curve(cells, mj) -> dict:
    """Least-squares line through 20*log10(m_j) vs cell index."""
    x = np.asarray(cells, dtype=float)
    y = 20.0 * np.log10(np.asarray(mj, dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    slope = float(slope)
    return {
        "slope_dB_per_cell": slope,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "spread_dB": float(y.max() - y.min()),
        "cell_min": int(np.argmin(mj)) + 1,
        "cell_max": int(np.argmax(mj)) + 1,
        "verdict": "decaying" if abs(slope) > SLOPE_THRESHOLD_DB_PER_CELL else "flat",
    }


def main() -> int:
    curves = load_curves()
    for curve in curves:
        curve["fit"] = fit_curve(curve["cells"], curve["mj"])

    print(f"{'curve':<26}{'slope dB/cell':>14}{'R^2':>8}{'spread dB':>11}"
          f"{'min@cell':>9}  verdict")
    for curve in curves:
        f = curve["fit"]
        print(f"{curve['label']:<26}{f['slope_dB_per_cell']:>+14.4f}{f['r2']:>8.3f}"
              f"{f['spread_dB']:>11.3f}{f['cell_min']:>9}  {f['verdict']}")

    styles = {
        LABEL_6P2_29: dict(color=figstyle.C_ACCENT, marker="o", ls="-",
                           mfc=figstyle.C_ACCENT),
        LABEL_6P2_20: dict(color=figstyle.C_ACCENT, marker="s", ls="--", mfc="none"),
        LABEL_10P6_29: dict(color=figstyle.C_PRIMARY, marker="o", ls="-",
                            mfc=figstyle.C_PRIMARY),
        LABEL_10P6_20: dict(color=figstyle.C_PRIMARY, marker="s", ls="--", mfc="none"),
    }

    fig, ax = figstyle.new_panel(height_in=2.7)
    for curve in curves:
        st = styles[curve["label"]]
        ax.plot(curve["cells"], curve["mj"], st["ls"], color=st["color"],
                marker=st["marker"], mfc=st["mfc"], mec=st["color"], mew=0.9,
                ms=4.0, lw=figstyle.DATA_LW, label=curve["label"])
    ax.set_yscale("log")
    ax.set_xlabel("cell index")
    ax.set_ylabel(r"junction depth $m_j$ (route B)")
    ax.set_xticks(range(1, 16, 2))
    # Legend below the axes: the four curves fill the data field, so an in-axes
    # legend would sit on top of the 10.6 GHz traces.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
              columnspacing=1.2, handletextpad=0.6)
    fig.tight_layout()

    out_base = figstyle.resolve_outdir() / "sm_junction_profiles"
    paths = figstyle.save_panel(fig, out_base)
    sc = figstyle.write_sidecar(
        out_base, sources=[MSTAR, LOBE2],
        script="examples/figures/sm_junction_profiles.py",
        command="python examples/figures/sm_junction_profiles.py",
        key_values={
            "x_axis": "cell index (1..15)",
            "y_axis": "m_j (route B), log scale",
            "slope_threshold_dB_per_cell": SLOPE_THRESHOLD_DB_PER_CELL,
            "curves": [{"label": c["label"], "fpump_GHz": c["fpump_GHz"],
                        "power_dBm": c["power_dBm"], "solved": c["solved"],
                        **c["fit"]} for c in curves],
            "per_cell_mj": {c["label"]: c["mj"] for c in curves},
            "per_cell_Vj_route_B": {c["label"]: c["Vj"] for c in curves},
        },
        extra={
            "route_B": "V(n_c{k}_rp) - V(n_c{k}_v) at the (0,1) mix (below the package divider)",
            "note_6p2_20dBm": ("sqrt(P) voltage-scaled from the 29 dBm leg "
                               "(flagged in the source record, not re-solved)"),
            "interpretation": ("all four curves are flat by the |slope| > 0.3 dB/cell "
                               "test (no along-line pump attenuation); the 10.6 GHz legs "
                               "are a mid-line standing-wave dip, not a decay"),
        },
    )
    print(f"wrote {paths['pdf']} / {paths['png']} / {figstyle.repo_relative(sc)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
