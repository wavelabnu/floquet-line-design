"""Fig 2d - realized pump-on/off boost vs cell count N (2.0 GHz length sweep).

Reads the imported study table ``results/length_sweep/sweep_table_primary.csv``
(primary candidate ``proxy_n3_single_short``) and plots the realized
pump-on/off boost (CSV column ``boost_peak_dB``) against cell count N (6..30).

Rendering follows the study's own row classification (CSV column
``sharpness_class``):
  * ``robust`` rows form the connected curve (smooth pump walk-up operating
    points);
  * ``SPIKE`` rows are drawn as isolated open grey markers with NO connecting
    line (fragile Fabry-Perot resonance spikes that collapse within one 3 dB
    pump step - artifacts of the idealized proxy-varactor Q);
  * any other class (e.g. ``moderate``) is drawn as distinct isolated markers.
N=15 is annotated as the robust optimum with its boost read from the data.

This is the 2.0 GHz length sweep only - there is no 1.7 GHz content anywhere.

Tier note: the study's realized numbers use the idealized proxy varactor
(a faithfulness record, not a real-world device gain); see
``results/length_sweep/REPORT.md`` and ``results/length_sweep/PROVENANCE.md``.

Usage:
    %ADSPY% examples/fig2d_length_sweep.py [--with-proxy]

Output: ``$TVTL_FIG_OUTDIR/fig2d_length_sweep.png`` when TVTL_FIG_OUTDIR is set,
else ``results/length_sweep/fig2d_length_sweep.png``.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "results" / "length_sweep" / "sweep_table_primary.csv"

# Column names as they appear in the study CSV header.
COL_N = "N"
COL_BOOST = "boost_peak_dB"      # realized pump-on/off boost (dB) -> y-axis
COL_CLASS = "sharpness_class"    # robust / SPIKE / moderate / ...
COL_PROXY = "pred_coh_pk_dB"     # analytical optimism proxy (optional overlay)

# Axis label taken from the source column (realized pump-on/off boost).
BOOST_AXIS_LABEL = "Realized pump-on/off boost, boost_peak_dB (dB)"


def _load_rows(path: Path):
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        for needed in (COL_N, COL_BOOST, COL_CLASS, COL_PROXY):
            if needed not in header:
                raise KeyError(f"column {needed!r} not in {path.name} header: {header}")
        rows = []
        for raw in reader:
            proxy_raw = raw.get(COL_PROXY, "")
            rows.append(
                {
                    "N": int(float(raw[COL_N])),
                    "boost": float(raw[COL_BOOST]),
                    "cls": (raw[COL_CLASS] or "").strip(),
                    "proxy": float(proxy_raw) if proxy_raw not in ("", None) else None,
                }
            )
    return header, rows


def build_figure(with_proxy: bool = False):
    _header, rows = _load_rows(CSV_PATH)

    robust = sorted((r["N"], r["boost"]) for r in rows if r["cls"] == "robust")
    spike = [(r["N"], r["boost"]) for r in rows if r["cls"] == "SPIKE"]
    other = [(r["N"], r["boost"], r["cls"]) for r in rows if r["cls"] not in ("robust", "SPIKE")]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    # Robust rows: connected curve.
    if robust:
        ax.plot(
            [n for n, _ in robust], [b for _, b in robust],
            "-o", color="#1f77b4", label="robust (smooth walk-up)", zorder=3,
        )

    # SPIKE rows: isolated open grey markers, no connecting line.
    if spike:
        ax.plot(
            [n for n, _ in spike], [b for _, b in spike],
            linestyle="none", marker="o", markerfacecolor="none",
            markeredgecolor="grey", markersize=8,
            label="SPIKE (fragile resonance)", zorder=2,
        )

    # Any other class (e.g. moderate): distinct isolated markers, no line.
    if other:
        classes = sorted({c for _, _, c in other})
        ax.plot(
            [n for n, _, _ in other], [b for _, b, _ in other],
            linestyle="none", marker="s", color="grey", alpha=0.6, markersize=7,
            label="/".join(classes) if classes else "other", zorder=2,
        )

    # Optional analytical-proxy overlay (default OFF).
    if with_proxy:
        px = [r["N"] for r in rows if r["proxy"] is not None]
        py = [r["proxy"] for r in rows if r["proxy"] is not None]
        ax.plot(px, py, ":", color="#d62728", alpha=0.75,
                label="analytical proxy (pred_coh_pk_dB)", zorder=1)

    # Annotate N=15 robust optimum with its boost read from the data.
    n15 = next((r for r in rows if r["N"] == 15), None)
    if n15 is not None:
        ax.annotate(
            f"N=15 robust optimum ({n15['boost']:.2f} dB)",
            xy=(15, n15["boost"]),
            xytext=(16.5, n15["boost"] + 7.0),
            ha="left", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black"),
        )

    ax.set_xlabel("Cell count N")
    ax.set_ylabel(BOOST_AXIS_LABEL)
    ax.set_title("Fig 2d - realized boost vs length (2.0 GHz, proxy_n3_single_short)")
    ax.set_xticks([r["N"] for r in rows])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig, rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fig 2d - realized boost vs N (2.0 GHz length sweep)."
    )
    parser.add_argument(
        "--with-proxy", action="store_true",
        help="overlay the analytical optimism proxy pred_coh_pk_dB (default off).",
    )
    args = parser.parse_args()

    fig, rows = build_figure(with_proxy=args.with_proxy)

    env_outdir = os.environ.get("TVTL_FIG_OUTDIR")
    outdir = Path(env_outdir) if env_outdir else (REPO_ROOT / "results" / "length_sweep")
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "fig2d_length_sweep.png"
    fig.savefig(outpath, dpi=150)

    print(f"wrote {outpath}")
    print(f"y-axis column: {COL_BOOST}  |  class column: {COL_CLASS}")
    print("plotted values (N, boost_peak_dB, class):")
    for r in rows:
        print(f"  N={r['N']:>2}  boost={r['boost']:+8.3f} dB  class={r['cls']}")


if __name__ == "__main__":
    main()
