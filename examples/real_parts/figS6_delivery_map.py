"""SM Fig S6 -- SMV1245 branch junction-delivery map.

Two stacked panels vs frequency (2-14 GHz), from
``results/branch_delivery_vs_freq.json``:

* top: junction-delivery transfer ``T(f) = |V_j/V_bt|`` in dB -- the fraction of
  the branch-top drive that reaches the varactor junction;
* bottom: the corrected-stub magnitude ``|Z_stub(f)|`` (ohm, log scale).

The signal / pump-relevant spots 3.1 / 5.3 / 6.2 / 8.4 / 10.6 GHz are marked: the
6.2 GHz notch (deep |Z_stub|) starves delivery, while near 10.6 GHz the stub
turns low-impedance and delivery peaks -- the mechanism behind the second-lobe
prediction.

Base Python (matplotlib, Agg), no ADS. Writes results/figures/figS6_delivery_map.{pdf,png}.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
FIGDIR = Path(os.environ.get("TVTL_FIG_OUTDIR", str(RESULTS / "figures")))
SPOTS = ("3.1", "5.3", "6.2", "8.4", "10.6")


def main() -> int:
    d = json.loads((RESULTS / "branch_delivery_vs_freq.json").read_text(encoding="utf-8"))
    f = d["grid_GHz"]
    T = d["T"]
    Zabs = d["Z_stub_abs_ohm"]
    T_dB = [20.0 * math.log10(max(t, 1e-30)) for t in T]
    spot = d["spot_table"]

    # internal check: global extrema agree with the recorded scalars
    tmax = d["T_global_max"]; tmin = d["T_global_min"]
    assert abs(max(T) - tmax["T"]) < 1e-9 and abs(min(T) - tmin["T"]) < 1e-9
    print(f"T_max {tmax['T']:.3e} @ {tmax['f_GHz']} GHz ; T_min {tmin['T']:.3e} @ {tmin['f_GHz']} GHz")
    for k in SPOTS:
        print(f"  spot {k:>4} GHz: T_dB={spot[k]['T_dB']:+.2f}  |Z_stub|={spot[k]['Z_stub_abs_ohm']:.1f} ohm")

    # ---- figure ----------------------------------------------------------- #
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.6, 5.6), sharex=True,
                                   height_ratios=[1, 1])

    ax0.plot(f, T_dB, color="#1f77b4", lw=1.2)
    ax0.set_ylabel("delivery  T(f) = |V$_j$/V$_{bt}$|  (dB)")
    ax0.set_title("SMV1245 branch junction-delivery map (Fig S6)\n"
                  "package R-L + junction C + element-accurate corrected stub")
    ax0.grid(True, alpha=0.3)

    ax1.semilogy(f, Zabs, color="#d62728", lw=1.2)
    ax1.set_ylabel("|Z$_{stub}$(f)|  (ohm)")
    ax1.set_xlabel("frequency (GHz)")
    ax1.grid(True, alpha=0.3, which="both")

    for k in SPOTS:
        fq = float(k)
        for ax in (ax0, ax1):
            ax.axvline(fq, color="#555555", lw=0.7, ls=":")
        ax0.annotate(f"{k}", (fq, spot[k]["T_dB"]), textcoords="offset points",
                     xytext=(2, 4), fontsize=8, color="#333333")
        ax0.plot(fq, spot[k]["T_dB"], "o", color="#1f77b4", ms=4)
        ax1.plot(fq, spot[k]["Z_stub_abs_ohm"], "o", color="#d62728", ms=4)

    ax1.set_xlim(min(f), max(f))
    fig.tight_layout()

    base = FIGDIR / "figS6_delivery_map"
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print("wrote", base.with_suffix(".pdf"), "and .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
