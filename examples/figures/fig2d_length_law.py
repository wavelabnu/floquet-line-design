"""Fig 2d -- realized pump-on/off boost vs cell count N (2.0 GHz length sweep).

Source (committed): ``results/length_sweep/sweep_table_primary.csv`` (primary
candidate ``proxy_n3_single_short``).

Rendering follows the study's own row classification (CSV column
``sharpness_class``):

  * ``robust`` rows form the connected curve (smooth pump walk-up operating
    points);
  * ``SPIKE`` rows are isolated OPEN grey markers with NO connecting line
    (fragile Fabry-Perot resonance spikes that collapse within one 3 dB pump
    step -- artifacts of the idealized proxy-varactor Q);
  * any other class (e.g. ``moderate``) is drawn as isolated grey squares.

N = 15 is annotated as the robust optimum with its boost read from the data, and
the committed anchors (N=15 8.18, N=12 26.49, N=18 28.20 dB) are asserted so an
imported-record drift fails loudly.

Tier note: the study's realized numbers use the idealized proxy varactor (a
faithfulness record, not a real-world device gain); see
``results/length_sweep/PROVENANCE.md`` and ``results/length_sweep/REPORT.md``.

Outputs (into ``$TVTL_FIG_OUTDIR`` if set, else ``results/figures/``):
``fig2d_length_law.pdf`` / ``.png`` / ``.sidecar.json``.

Usage:
    python examples/figures/fig2d_length_law.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle  # noqa: E402

SOURCE = "results/length_sweep/sweep_table_primary.csv"
PROVENANCE = "results/length_sweep/PROVENANCE.md"

COL_N = "N"
COL_BOOST = "boost_peak_dB"
COL_CLASS = "sharpness_class"

LEGEND_ROBUST = "robust (smooth walk-up)"
LEGEND_SPIKE = "SPIKE (fragile resonance)"

# Committed anchors from the imported study table.
SPOT_CHECKS = {15: 8.18, 12: 26.49, 18: 28.20}

TIER_SENTENCE = ("Realized numbers use the idealized proxy varactor: a "
                 "faithfulness RECORD, not a real-world device gain.")


def load_rows() -> list[dict]:
    rows = []
    with open(REPO_ROOT / SOURCE, newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            rows.append({"N": int(float(raw[COL_N])),
                         "boost": float(raw[COL_BOOST]),
                         "cls": (raw[COL_CLASS] or "").strip()})
    return rows


def classify(rows: list[dict]) -> dict:
    robust = sorted((r["N"], r["boost"]) for r in rows if r["cls"] == "robust")
    spike = [(r["N"], r["boost"]) for r in rows if r["cls"] == "SPIKE"]
    other = [(r["N"], r["boost"], r["cls"]) for r in rows
             if r["cls"] not in ("robust", "SPIKE")]
    classes = sorted({c for _, _, c in other})
    return {"robust": robust, "spike": spike, "other": other,
            "other_label": "/".join(classes) if classes else "other",
            "by_n": {r["N"]: r["boost"] for r in rows}}


def main() -> int:
    rows = load_rows()
    groups = classify(rows)
    by_n = groups["by_n"]

    for n, expect in SPOT_CHECKS.items():
        got = round(by_n[n], 2)
        status = "OK" if got == expect else "MISMATCH"
        print(f"spot-check N={n}: {got:.2f} dB (expect {expect:.2f})  {status}")
        assert got == expect, f"N={n} boost {got} != committed {expect}"

    fig, ax = figstyle.new_panel(height_in=2.55)

    if groups["robust"]:
        ax.plot([n for n, _ in groups["robust"]], [b for _, b in groups["robust"]],
                "-o", color=figstyle.C_PRIMARY, ms=3.2, mew=0.0,
                label=LEGEND_ROBUST, zorder=3)
    if groups["spike"]:
        ax.plot([n for n, _ in groups["spike"]], [b for _, b in groups["spike"]],
                linestyle="none", marker="o", mfc="none", mec=figstyle.C_GREY,
                mew=1.0, ms=6, label=LEGEND_SPIKE, zorder=2)
    if groups["other"]:
        ax.plot([n for n, _, _ in groups["other"]], [b for _, b, _ in groups["other"]],
                linestyle="none", marker="s", color=figstyle.C_GREY, alpha=0.6,
                ms=5, mew=0.0, label=groups["other_label"], zorder=2)

    n15 = by_n.get(15)
    if n15 is not None:
        ax.annotate(f"N = 15 robust optimum ({n15:.2f} dB)", xy=(15, n15),
                    xytext=(16.5, n15 + 9.0), ha="left", fontsize=figstyle.TICK_PT,
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.6))

    ax.set_xlabel("cell count N")
    ax.set_ylabel("pump-on/off boost (dB)")
    ax.set_xticks([r["N"] for r in rows])
    ax.legend(loc="upper right")
    fig.tight_layout()

    out_base = figstyle.resolve_outdir() / "fig2d_length_law"
    paths = figstyle.save_panel(fig, out_base)
    sc = figstyle.write_sidecar(
        out_base, sources=[SOURCE, PROVENANCE],
        script="examples/figures/fig2d_length_law.py",
        command="python examples/figures/fig2d_length_law.py",
        key_values={
            "y_label": "pump-on/off boost (dB)",
            "x_label": "cell count N",
            "robust": [{"N": n, "boost_dB": b} for n, b in groups["robust"]],
            "SPIKE": [{"N": n, "boost_dB": b} for n, b in groups["spike"]],
            groups["other_label"]: [{"N": n, "boost_dB": b}
                                    for n, b, _ in groups["other"]],
            "N15_robust_optimum_dB": n15,
            "spot_checks": SPOT_CHECKS,
        },
        extra={
            "legend_wording": {"robust": LEGEND_ROBUST, "SPIKE": LEGEND_SPIKE,
                               "other": groups["other_label"]},
            "tier_sentence_path": PROVENANCE,
            "tier_sentence": TIER_SENTENCE,
        },
    )
    print(f"wrote {paths['pdf']} / {paths['png']} / {figstyle.repo_relative(sc)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
