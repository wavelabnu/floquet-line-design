"""Fig 2c -- certified MAVR PSA boost vs pump phase.

Source (committed): ``examples/real_parts/results/headline_mavr.json``.

The panel plots the RECORDED per-phase boost trace (``curve[i].boost_dB``); the
physics is not re-derived here. As a replay guard the boost is independently
recomputed from the same record,

    passive_S21_dB   = 10*log10(0.5 * Vout_off^2 / 50 / P_avs)
    boost_derived[i] = G_mod_in_dB[i] - passive_S21_dB

and the maximum deviation from the recorded trace is asserted at or below
1e-12 dB, so a silent drift in either the record or the definition fails loudly.

Outputs (into ``$TVTL_FIG_OUTDIR`` if set, else ``results/figures/``):
``fig2c_phase_sweep.pdf`` / ``.png`` / ``.sidecar.json``.

Usage:
    python examples/figures/fig2c_phase_sweep.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle  # noqa: E402

SOURCE = "examples/real_parts/results/headline_mavr.json"
GUARD_DB = 1.0e-12


def load_record() -> dict:
    return json.loads((REPO_ROOT / SOURCE).read_text(encoding="utf-8"))


def derive(record: dict) -> dict:
    """Recorded trace + the recomputed replay-guard quantities."""
    curve = record["curve"]
    phase = [float(r["phase_deg"]) for r in curve]
    recorded = [float(r["boost_dB"]) for r in curve]
    g_mod = [float(r["G_mod_in_dB"]) for r in curve]

    passive_s21_db = 10.0 * math.log10(
        0.5 * float(record["passive_pumpoff_Vout_V"]) ** 2 / 50.0
        / float(record["P_avs_W"])
    )
    derived = [g - passive_s21_db for g in g_mod]
    max_dev = max(abs(a - b) for a, b in zip(derived, recorded))

    i_peak = max(range(len(recorded)), key=lambda i: recorded[i])
    i_trough = min(range(len(recorded)), key=lambda i: recorded[i])
    return {
        "phase_deg": phase,
        "boost_recorded_dB": recorded,
        "passive_S21_dB": passive_s21_db,
        "max_dev_dB": max_dev,
        "peak": {"phase_deg": phase[i_peak], "boost_dB": recorded[i_peak]},
        "trough": {"phase_deg": phase[i_trough], "boost_dB": recorded[i_trough]},
        "phase_step_deg": phase[1] - phase[0],
    }


def main() -> int:
    record = load_record()
    d = derive(record)

    print(f"passive_S21_dB (derived from Vout) = {d['passive_S21_dB']:+.12f} dB")
    print(f"max|derived - recorded boost| = {d['max_dev_dB']:.3e} dB "
          f"(guard <= {GUARD_DB:g})")
    assert d["max_dev_dB"] <= GUARD_DB, (
        f"replay guard failed: {d['max_dev_dB']:.3e} > {GUARD_DB:g}")
    print(f"PEAK   (from data): {d['peak']['boost_dB']:+.4f} dB "
          f"@ {d['peak']['phase_deg']:.0f} deg")
    print(f"TROUGH (from data): {d['trough']['boost_dB']:+.4f} dB "
          f"@ {d['trough']['phase_deg']:.0f} deg")

    fig, ax = figstyle.new_panel(height_in=2.45)
    ax.axhline(0.0, color=figstyle.C_REF, lw=0.8, ls=":", zorder=1)
    ax.plot(d["phase_deg"], d["boost_recorded_dB"], "-o",
            color=figstyle.C_PRIMARY, ms=3.0, mew=0.0, zorder=3)

    pk, tr = d["peak"], d["trough"]
    ax.plot(pk["phase_deg"], pk["boost_dB"], "*", color=figstyle.C_ACCENT,
            ms=9, mew=0.0, zorder=5)
    ax.annotate(f"+{pk['boost_dB']:.2f} dB @ {pk['phase_deg']:.0f}°",
                (pk["phase_deg"], pk["boost_dB"]), textcoords="offset points",
                xytext=(5, -9), color=figstyle.C_ACCENT, fontsize=figstyle.TICK_PT)
    ax.plot(tr["phase_deg"], tr["boost_dB"], "v", color=figstyle.C_GREY,
            ms=5, mew=0.0, zorder=5)
    ax.annotate(f"{tr['boost_dB']:.2f} dB @ {tr['phase_deg']:.0f}°",
                (tr["phase_deg"], tr["boost_dB"]), textcoords="offset points",
                xytext=(6, 4), color=figstyle.C_GREY_DK, fontsize=figstyle.TICK_PT)

    ax.set_xlabel("pump phase (deg)")
    ax.set_ylabel("boost over passive pump-off (dB)")
    ax.set_xticks(range(0, 361, 45))
    ax.set_xlim(-12, 357)
    fig.tight_layout()

    out_base = figstyle.resolve_outdir() / "fig2c_phase_sweep"
    paths = figstyle.save_panel(fig, out_base)

    caption = {
        "candidate_id": "ABSENT (no candidate/candidate_id/id field in the record)",
        "headline": record.get("headline"),
        "definition": record.get("definition"),
        "f_signal_GHz": record.get("f_signal_GHz"),
        "f_pump_GHz": record.get("f_pump_GHz"),
        "pump_power_dBm": record.get("pump_power_dBm"),
        "signal_power_dBm": record.get("signal_power_dBm"),
        "P_avs_W": record.get("P_avs_W"),
        "passive_pumpoff_Vout_V": record.get("passive_pumpoff_Vout_V"),
        "passive_S21_dB_derived": d["passive_S21_dB"],
        "phase_grid_deg": {"start": d["phase_deg"][0], "stop": d["phase_deg"][-1],
                           "step": d["phase_step_deg"], "n": len(d["phase_deg"])},
        "varactor_part": "ABSENT (no varactor part field in the record)",
        "varactor_bias": "ABSENT (no bias field in the record)",
        "reference_level": {
            "note": ("no field literally named 'reference_level'; the recorded "
                     "reference quantity is 'certified_reference_dB' (recorded "
                     "for completeness, NOT annotated on the panel)"),
            "certified_reference_dB": record.get("certified_reference_dB"),
        },
        "design_reference": ("MAVR-000350, 12 V bias, 24 cells, n4_single_combined "
                            "(COMBINED MLEF||VIAGND trap) -- "
                            "examples/real_parts/real_part_designs.py:206-230"),
    }
    sc = figstyle.write_sidecar(
        out_base, sources=[SOURCE],
        script="examples/figures/fig2c_phase_sweep.py",
        command="python examples/figures/fig2c_phase_sweep.py",
        key_values={
            "y_label": "boost over passive pump-off (dB)",
            "x_label": "pump phase (deg)",
            "peak_from_data": d["peak"],
            "trough_from_data": d["trough"],
            "passive_S21_dB_derived": d["passive_S21_dB"],
            "replay_guard_max_dev_dB": d["max_dev_dB"],
            "replay_guard_threshold_dB": GUARD_DB,
            "n_phase_points": len(d["phase_deg"]),
        },
        extra={"caption_material": caption},
    )
    print(f"wrote {paths['pdf']} / {paths['png']} / {figstyle.repo_relative(sc)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
