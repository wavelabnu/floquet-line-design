"""SM Fig S7 -- the model boundary for the fabricated SMV1245 board.

Three panels, one per boundary probe, all telling the same story: the frozen
fixed-C fabricated board sits far inside the model's coherent-gain boundary.

* (a) bias sweep (``smv1245_bias_sweep.json``): pumped boost stays < 0.01 dB
  across 0-3 V of effective reverse bias at both pump powers -- a modest bias
  does NOT recover double-digit boost on the frozen geometry.
* (b) junction-depth m-sweep (``smv1245_mstar_arc_v2.json`` step 2): the
  analytical coherent peak grows with the delivered junction depth, but the
  board delivers only m_j ~ 0.006, whereas +16 dB (D16, at the null) needs a
  junction depth ~ 0.38 -- a ~60x gap.
* (c) second-lobe prediction (``smv1245_lobe2_prediction.json``): even where the
  passive floor is most favorable (4.8-5.7 GHz, near the 10.6 GHz delivery peak),
  the HB pumped boost stays ~ 0.01 dB.

Base Python (matplotlib, Agg), no ADS. Writes results/figures/figS7_model_boundary.{pdf,png}.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
FIGDIR = Path(os.environ.get("TVTL_FIG_OUTDIR", str(RESULTS / "figures")))


def main() -> int:
    bs = json.loads((RESULTS / "smv1245_bias_sweep.json").read_text(encoding="utf-8"))
    mv = json.loads((RESULTS / "smv1245_mstar_arc_v2.json").read_text(encoding="utf-8"))
    lo = json.loads((RESULTS / "smv1245_lobe2_prediction.json").read_text(encoding="utf-8"))

    # ---- panel (a): bias sweep ------------------------------------------- #
    by_pump: dict[float, list[tuple[float, float]]] = {}
    for r in bs["rows"]:
        if r.get("all_phases_converged"):
            by_pump.setdefault(r["pump_power_dBm"], []).append((r["v_bias_V"], r["peak_boost_dB"]))
    for p in by_pump:
        by_pump[p].sort()
    max_boost = max(b for pts in by_pump.values() for _, b in pts)
    print(f"(a) max pumped boost over all bias/pump = {max_boost:.4f} dB")
    assert max_boost < 0.05, "bias sweep boundary story broken: boost not ~0"

    # ---- panel (b): junction-depth m-sweep ------------------------------- #
    step2 = mv["step2_msweep_3p1GHz"]
    de = [r["depth_eff_junction"] for r in step2]
    coh = [r["coherent_peak_dB"] for r in step2]
    d16n = mv["D16"]["at_notch"]
    delivered = mv["step4_delivered_junction_depth"].get("m_j_29_mean")
    assert d16n.get("bracketed"), "D16 at notch not bracketed"
    print(f"(b) D16 junction depth = {d16n['D16_depth_eff_junction']:.4f} for +{d16n['coherent_peak_dB']:.1f} dB ; "
          f"delivered m_j(29dBm) = {delivered:.4f}")

    # ---- panel (c): second lobe ------------------------------------------ #
    ana = lo["analytical_S21_off_4p8_5p7"]
    fa = [r["f_GHz"] for r in ana]
    s21 = [r["S21_off_dB"] for r in ana]
    hb = [r for r in lo["hb_leg"]["29"] if r.get("converged")]
    fh = [r["f_GHz"] for r in hb]
    boost_h = [r["peak_boost_dB"] for r in hb]
    print(f"(c) 2nd-lobe HB boost range = [{min(boost_h):.4f}, {max(boost_h):.4f}] dB ; "
          f"passive S21_off in [{min(s21):.2f}, {max(s21):.2f}] dB")

    # ---- figure ---------------------------------------------------------- #
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(12.4, 3.9))

    # (a)
    styles = {29.0: ("o-", "#1f77b4"), 20.0: ("s--", "#ff7f0e")}
    for p in sorted(by_pump, reverse=True):
        xs, ys = zip(*by_pump[p])
        st, col = styles.get(p, ("o-", "#333333"))
        axA.plot(xs, ys, st, color=col, label=f"pump {p:.0f} dBm")
    axA.axhline(0.0, color="grey", lw=0.8, ls=":")
    axA.set_xlabel("effective reverse bias (V)")
    axA.set_ylabel("peak pumped boost (dB)")
    axA.set_title("(a) bias sweep @ 3.1 GHz\nfrozen geometry: boost stays ~0")
    axA.grid(True, alpha=0.3)
    axA.legend(fontsize=8, loc="upper left")

    # (b)
    axB.plot(de, coh, "o-", color="#2ca02c", label="analytical coherent peak")
    if delivered is not None:
        axB.axvline(delivered, color="#d62728", lw=1.0, ls="--",
                    label=f"delivered m$_j$ = {delivered:.3f}")
    axB.set_xlabel("junction depth  m$_j$ (depth_effective)")
    axB.set_ylabel("coherent peak (dB)")
    axB.set_title("(b) junction-depth m-sweep\n+16 dB needs m$_j$ "
                  f"~{d16n['D16_depth_eff_junction']:.2f} (D16, at null)")
    axB.grid(True, alpha=0.3)
    axB.legend(fontsize=8, loc="upper left")
    axB.text(0.97, 0.05, f"D16 gap: {d16n['D16_depth_eff_junction'] / delivered:.0f}x",
             transform=axB.transAxes, ha="right", va="bottom", fontsize=8,
             bbox=dict(boxstyle="round", fc="#fff0f0", ec="#d62728", alpha=0.9))

    # (c)
    axC.plot(fa, s21, "o-", color="#9467bd", label="passive S21$_{off}$")
    axC.set_xlabel("frequency (GHz)")
    axC.set_ylabel("passive S21$_{off}$ (dB)", color="#9467bd")
    axC.tick_params(axis="y", labelcolor="#9467bd")
    axC.set_title("(c) second-lobe prediction\n4.8-5.7 GHz (pump 2f$_s$ ~ 10.6 GHz)")
    axC.grid(True, alpha=0.3)
    axC2 = axC.twinx()
    axC2.plot(fh, boost_h, "s--", color="#d62728", label="HB pumped boost (29 dBm)")
    axC2.set_ylabel("HB pumped boost (dB)", color="#d62728")
    axC2.tick_params(axis="y", labelcolor="#d62728")
    axC2.set_ylim(-0.02, max(0.05, max(boost_h) * 1.4))
    lines = axC.get_lines() + axC2.get_lines()
    axC.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="center right")

    fig.suptitle("Model boundary of the fixed-C fabricated SMV1245 board (Fig S7)", y=1.02)
    fig.tight_layout()

    base = FIGDIR / "figS7_model_boundary"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", base.with_suffix(".pdf"), "and .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
