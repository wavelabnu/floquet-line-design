"""Plot SM Fig S1 from ``results/c_vs_s_formulations.json``: coherent-peak gain
vs modulation depth for the capacitance-linear, elastance-linear, and reciprocal
modulated-element formulations, plus the reciprocal solver residual.

Base Python (matplotlib), no ADS. Run:

    python examples/real_parts/plot_c_vs_s_formulations.py

Writes ``results/c_vs_s_formulations.png`` alongside the JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"


def main() -> int:
    data = json.loads((RESULTS / "c_vs_s_formulations.json").read_text(encoding="utf-8"))
    rows = data["fig_s1"]["rows"]
    m = [r["m"] for r in rows]
    cap = [r["capacitance_dB"] for r in rows]
    ela = [r["elastance_dB"] for r in rows]
    rec = [r["reciprocal_dB_N6"] for r in rows]
    resid6 = [r["reciprocal_residual_N6"] for r in rows]
    resid10 = [r["reciprocal_residual_N10"] for r in rows]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.4, 6.4), height_ratios=[2, 1])
    ax0.plot(m, cap, "o-", label="capacitance-linear  Q = C(t) v", color="#1f77b4")
    ax0.plot(m, ela, "s--", label="elastance-linear  S(t) Q = v", color="#d62728")
    ax0.plot(m, rec, "x:", label="reciprocal  S(t) = 1/C(t)  (solver check)", color="#2ca02c")
    ax0.set_ylabel("coherent-peak gain (dB)")
    ax0.set_title("C-linear vs S-linear modulated element (SM Fig S1)\n"
                  "k0p4 ideal canonical, 15 cells, N=6")
    ax0.legend(loc="upper left", fontsize=8)
    ax0.grid(True, alpha=0.3)

    ax1.semilogy(m, [max(v, 1e-16) for v in resid6], "x:", label="reciprocal residual, N=6", color="#2ca02c")
    ax1.semilogy(m, [max(v, 1e-16) for v in resid10], "d-", label="reciprocal residual, N=10", color="#9467bd")
    ax1.axhline(1e-9, color="grey", lw=0.8, ls=":", label="1e-9 dB")
    ax1.set_xlabel("modulation depth  m  (dC/C0)")
    ax1.set_ylabel("|C - reciprocal| (dB)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    out = RESULTS / "c_vs_s_formulations.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
