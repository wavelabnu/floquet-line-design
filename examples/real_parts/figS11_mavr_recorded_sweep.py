"""SM Fig S11 -- recorded MAVR net out/in G_mod_in_dB(phi), the audit companion
to main-text Fig 2c.

Where Fig 2c shows the *derived* pump-on/off boost, this shows the *recorded*
net available-power gain G_mod_in_dB(phi) straight from ``headline_mavr.json``,
with the certification flags that qualify the sweep annotated on the axes:
single-lobe, all-phases-converged, peak finite, bounded, and the reference-level
line (+14.14 dB -- a historical reference level, not a gate output). It is the
traceable record behind the headline number.

Base Python (matplotlib, Agg), no ADS. Writes results/figures/figS11_mavr_recorded_sweep.{pdf,png}.
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
    d = json.loads((RESULTS / "headline_mavr.json").read_text(encoding="utf-8"))
    curve = d["curve"]
    phi = [r["phase_deg"] for r in curve]
    g = [r["G_mod_in_dB"] for r in curve]

    peak = d["peak"]
    trough = d["trough"]
    cert = d["certified_reference_dB"]

    # ---- certification flags (the audit checklist) ------------------------ #
    flags = {
        "single_lobe": d["single_lobe"],
        "all_phases_converged": d["all_phases_converged"],
        "peak_is_finite": d["peak_is_finite"],
        "bounded": d["bounded"],
        "n_local_maxima": d["n_local_maxima"],
        "n_local_minima": d["n_local_minima"],
    }
    print("certification flags:", flags)
    print(f"peak G_mod_in_dB = {peak['G_mod_in_dB']:.3f} dB @ {peak['phase_deg']:.0f} deg ; "
          f"reference_level = {cert:.2f} dB")
    # internal consistency with the recorded scalars
    assert abs(max(g) - peak["G_mod_in_dB"]) < 1e-9
    assert abs(min(g) - trough["G_mod_in_dB"]) < 1e-9
    assert flags["single_lobe"] and flags["all_phases_converged"]
    assert flags["n_local_maxima"] == 1 and flags["n_local_minima"] == 1

    # ---- figure ----------------------------------------------------------- #
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(phi, g, "s-", color="#2ca02c", label="recorded G$_{mod,in}$(phi)")
    ax.axhline(0.0, color="grey", lw=0.8, ls=":")
    ax.axhline(cert, color="#d62728", lw=1.0, ls="--", label=f"reference level ({cert:+.2f} dB)")

    ax.plot(peak["phase_deg"], peak["G_mod_in_dB"], "*", color="#d62728", ms=14, zorder=5)
    ax.annotate(f"peak {peak['G_mod_in_dB']:.2f} dB @ {peak['phase_deg']:.0f}$\\degree$",
                (peak["phase_deg"], peak["G_mod_in_dB"]), textcoords="offset points",
                xytext=(-12, 10), ha="right", color="#d62728", fontsize=9)
    ax.plot(trough["phase_deg"], trough["G_mod_in_dB"], "v", color="#7f7f7f", ms=8)
    ax.annotate(f"trough {trough['G_mod_in_dB']:.2f} dB",
                (trough["phase_deg"], trough["G_mod_in_dB"]), textcoords="offset points",
                xytext=(10, 6), color="#555555", fontsize=8)

    checklist = "\n".join([
        "certification:",
        f"  single lobe: {flags['single_lobe']}",
        f"  all phases converged: {flags['all_phases_converged']}",
        f"  peak finite: {flags['peak_is_finite']}",
        f"  bounded: {flags['bounded']}",
        f"  local maxima/minima: {flags['n_local_maxima']}/{flags['n_local_minima']}",
    ])
    ax.text(0.36, 0.03, checklist, transform=ax.transAxes, fontsize=7.5,
            va="bottom", ha="left", family="monospace",
            bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#bbbbbb", alpha=0.9))

    ax.set_xlabel("pump phase  $\\phi$  (deg)")
    ax.set_ylabel("net out/in  G$_{mod,in}$  (dB)")
    ax.set_title("MAVR recorded net gain vs pump phase (Fig S11)\n"
                 "audit companion to Fig 2c (degenerate full-family out/in)")
    ax.set_xticks(range(0, 361, 45))
    ax.set_xlim(-10, 355)
    ax.set_ylim(-16, 17)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    base = FIGDIR / "figS11_mavr_recorded_sweep"
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print("wrote", base.with_suffix(".pdf"), "and .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
