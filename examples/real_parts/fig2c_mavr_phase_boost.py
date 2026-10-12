"""Main-text Fig 2c -- MAVR pump-on/off boost vs pump phase.

Per-phase boost of the pumped modulated line over its own passive (pump-off)
reference, from ``results/headline_mavr.json``:

    boost_dB(phi) = G_mod_in_dB(phi) - passive_S21_dB
    passive_S21_dB = 10*log10( 0.5 * Vout_off^2 / 50 / P_avs )

where ``Vout_off`` is the pump-off output voltage and ``P_avs`` the available
source power (both recorded). The passive reference is a single phase-independent
number, so this simply lifts the recorded net out/in curve onto the pump-off
floor. Internal check: boost(210 deg) = +18.96 dB.

Base Python (matplotlib, Agg), no ADS. Writes results/figures/fig2c_mavr_phase_boost.{pdf,png}.
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


def main() -> int:
    data = json.loads((RESULTS / "headline_mavr.json").read_text(encoding="utf-8"))
    curve = data["curve"]
    vout_off = data["passive_pumpoff_Vout_V"]
    p_avs = data["P_avs_W"]

    passive_S21_dB = 10.0 * math.log10(0.5 * vout_off ** 2 / 50.0 / p_avs)

    phi = [r["phase_deg"] for r in curve]
    g_mod = [r["G_mod_in_dB"] for r in curve]
    boost = [g - passive_S21_dB for g in g_mod]                 # derived per the formula
    boost_recorded = [r["boost_dB"] for r in curve]             # cross-check against JSON

    # ---- internal checks -------------------------------------------------- #
    formula = ("boost_dB(phi) = G_mod_in_dB(phi) - passive_S21_dB ; "
               "passive_S21_dB = 10*log10(0.5*Vout_off^2/50 / P_avs)")
    print("FORMULA:", formula)
    print(f"passive_S21_dB = {passive_S21_dB:+.4f} dB  (Vout_off={vout_off:.6e} V, P_avs={p_avs:.1e} W)")
    i210 = phi.index(210.0)
    b210 = boost[i210]
    print(f"boost(210 deg) computed = {b210:+.4f} dB ; recorded = {boost_recorded[i210]:+.4f} dB")
    assert round(b210, 2) == 18.96, f"internal check failed: boost(210)={b210:.4f} != +18.96"
    max_dev = max(abs(a - b) for a, b in zip(boost, boost_recorded))
    print(f"max |derived - recorded boost| over all phases = {max_dev:.2e} dB")
    assert max_dev < 5e-3, f"derived boost disagrees with recorded by {max_dev:.2e} dB"

    # ---- figure ----------------------------------------------------------- #
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(phi, boost, "o-", color="#1f77b4", label="pump-on/off boost")
    ax.axhline(0.0, color="grey", lw=0.8, ls=":", label="passive floor (0 dB)")

    ipk = boost.index(max(boost))
    ax.plot(phi[ipk], boost[ipk], "*", color="#d62728", ms=14, zorder=5)
    ax.annotate(f"+{boost[ipk]:.2f} dB @ {phi[ipk]:.0f}$\\degree$",
                (phi[ipk], boost[ipk]), textcoords="offset points", xytext=(8, -14),
                color="#d62728", fontsize=9)
    itr = boost.index(min(boost))
    ax.plot(phi[itr], boost[itr], "v", color="#7f7f7f", ms=8)
    ax.annotate(f"{boost[itr]:.2f} dB @ {phi[itr]:.0f}$\\degree$",
                (phi[itr], boost[itr]), textcoords="offset points", xytext=(6, 6),
                color="#555555", fontsize=8)

    ax.set_xlabel("pump phase  $\\phi$  (deg)")
    ax.set_ylabel("boost over passive pump-off (dB)")
    ax.set_title("MAVR PSA boost vs pump phase (Fig 2c)\n"
                 f"f$_s$=3 GHz, 2f$_s$ pump; passive S21 = {passive_S21_dB:+.2f} dB reference")
    ax.set_xticks(range(0, 361, 45))
    ax.set_xlim(-10, 355)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", fontsize=8)
    fig.tight_layout()

    base = FIGDIR / "fig2c_mavr_phase_boost"
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print("wrote", base.with_suffix(".pdf"), "and .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
