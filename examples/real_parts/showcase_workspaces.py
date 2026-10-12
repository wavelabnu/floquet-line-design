"""Regenerate one self-contained, openable ADS GUI workspace PER EXAMPLE (Part 3).

Each real-part example becomes its OWN portable hierarchical workspace under
``outputs/``, baked to the operating point that demonstrates its headline so that
opening the workspace in ADS and running *Simulate* shows it directly:

* ``real_amplifier_MAVR`` -- the pump is baked at the OPTIMAL phase (the PSA-peak,
  ~210 deg) so a Simulate reads ``G_mod_in_dB`` ~ +14.35 dB. The gain-vs-pump-phase
  figure sits alongside and the pump phase is documented as THE tunable parameter
  (0 deg = the +3.55 dB phase-0 parity point, ~210 deg = the +14.35 dB peak,
  ~30 deg = the -14 dB anti-phase trough).
* ``real_oscillator_SMV2020`` -- the signal is driven OFF (-200 dBm) and the pump is
  on at the PSA-peak phase, so a Simulate shows the autonomous signal-off limit cycle
  ``|V(n_output)|`` ~ 30.9 mV: the oscillation signature (output persists with no input).
* ``fabricated_SMV1245`` -- the as-built 0 V board at its 3.1 GHz operating point with
  the net out/in readout, and the Part 1 boost-vs-frequency figure alongside (the honest
  delivery-starved result -- the modulation barely registers across the band).

Every workspace is the same hierarchical design as the parity build: a ``unit_cell``
subcircuit cascaded as ``unit_cell:Xk`` on the top schematic, with the ``MEAS_GAIN``
``MeasEqn`` block embedded and the boundary nets named ``n_input`` / ``n_output`` so a
human Simulate reads the gain definitions straight off the schematic. ONLY the showcase
pump phase / signal level differ from the phase-0 parity point; the geometry and the flat
phase-0 deck are byte-identical to the validated reference (see ``results/parity_<key>.json``
and ``results/decks/<key>_flat_hb_deck.ckt``). Before emitting each workspace the baked
operating point is verified headlessly against its recorded headline.

Examples-only, no ``src/`` change. Run under the ADS Python interpreter:

    $env:HPEESOF_DIR = "<ADS_INSTALL_DIR>"
    $env:PYTHONPATH  = "src"
    & $env:ADS_PYTHON showcase_workspaces.py            # all three
    & $env:ADS_PYTHON showcase_workspaces.py mavr       # one (mavr | smv2020 | smv1245)
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_examples import _embed_gain_equations  # noqa: E402  (reuse the committed embed)
from real_part_designs import (  # noqa: E402
    spec_mavr,
    spec_smv1245,
    spec_smv2020,
)

_HERE = Path(__file__).resolve().parent
OUTPUTS = _HERE / "outputs"            # git-ignored (regenerated workspaces)
RESULTS = _HERE / "results"            # committed headline JSON + figures + parity proofs
SIGNAL_OFF_DBM = -200.0                # the oscillator signal-off probe level (DEFAULT_SIGNAL_OFF_DBM)
_TIMEOUT_S = 600

# Per-example showcase configuration. ``demo`` overrides bake the headline operating
# point onto the spec's VAR block; ``p_avs_dBm`` is the absolute-gain denominator baked
# into MEAS_GAIN (the oscillator uses the -30 dBm reference so the ratio stays sane while
# the signal is off -- its real readout is the output amplitude, not the ratio).
_GITIGNORE = """\
# Generated OpenAccess workspace internals -- regenerate with
#   <ADS_PYTHON> showcase_workspaces.py <key>
# The reviewable text deliverables are README.md and <lib>/flat_hb_deck.ckt; the
# phase-0 parity proof is in ../../results/ (parity_<key>.json + decks/).
*.oa
*.db
*.dm
*.tag
*.cfg
*.prf
*.log
*.opt
*.wrkstate
cds.lib
lib.defs
data/
"""


def _headline_phase(key, default_deg):
    """The recorded PSA-peak pump phase from the committed headline JSON (deg)."""
    path = RESULTS / f"headline_{key}.json"
    if not path.exists():
        return default_deg
    d = json.loads(path.read_text(encoding="utf-8"))
    if "peak" in d and isinstance(d["peak"], dict) and "phase_deg" in d["peak"]:
        return float(d["peak"]["phase_deg"])
    if "peak_phase_deg" in d:
        return float(d["peak_phase_deg"])
    return default_deg


def _dbmtow(dbm):
    return 10.0 ** ((float(dbm) - 30.0) / 10.0)


def _demo_specs():
    """Build the three baked demo specs + their showcase metadata (phases from headlines)."""
    from tvtl_ads_builder_v2.device_spec import with_var_overrides

    mavr_phi = _headline_phase("mavr", 210.0)
    osc_phi = _headline_phase("smv2020", 300.0)
    smv1245_phi = _headline_phase("smv1245", 255.0)  # boost ~0, phase immaterial; baked for definiteness

    return {
        "mavr": {
            "spec": with_var_overrides(spec_mavr(), {"pump_phase_rad": math.radians(mavr_phi)}),
            "lib": "real_amplifier_MAVR", "top": "real_amplifier_mavr_top",
            "ws": "TVTL_real_amplifier_MAVR_wrk",
            "baked_phase_deg": mavr_phi, "p_avs_W": _dbmtow(-30.0),
            "figure": "mavr_gain_vs_pump_phase.png",
            "metric": "g_mod_in_dB", "expect": 14.35, "tol": 0.5,
        },
        "smv2020": {
            "spec": with_var_overrides(spec_smv2020(),
                                       {"pump_phase_rad": math.radians(osc_phi),
                                        "signal_power_dBm": SIGNAL_OFF_DBM}),
            "lib": "real_oscillator_SMV2020", "top": "real_oscillator_smv2020_top",
            "ws": "TVTL_real_oscillator_SMV2020_wrk",
            "baked_phase_deg": osc_phi, "p_avs_W": _dbmtow(-30.0),
            "figure": None,
            "metric": "out_mV", "expect": 30.9, "tol": 3.0,
        },
        "smv1245": {
            "spec": with_var_overrides(spec_smv1245(), {"pump_phase_rad": math.radians(smv1245_phi)}),
            "lib": "fabricated_SMV1245", "top": "fabricated_smv1245_top",
            "ws": "TVTL_fabricated_SMV1245_wrk",
            "baked_phase_deg": smv1245_phi, "p_avs_W": _dbmtow(-20.0),
            "figure": "smv1245_boost_vs_freq.png",
            "metric": "g_mod_in_dB", "expect": -6.46, "tol": 0.5,
        },
    }


def _verify_headless(cfg, run_dir):
    """Run the baked demo deck once at its baked phase; return (on_V, g_mod_in_dB, off_V)."""
    from tvtl_ads_builder_v2.oa.psa_gain_artifact import run_full_sum_gain_sweep

    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    sw = run_full_sum_gain_sweep(cfg["spec"], run_dir, phases_deg=(cfg["baked_phase_deg"],),
                                 off_pump_power_dBm=-100.0, timeout_s=_TIMEOUT_S)
    on_V = math.sqrt(2.0 * 50.0 * sw.pout_mod_W) if sw.pout_mod_W > 0 else 0.0
    return on_V, sw.g_mod_in_dB, abs(sw.v_sig_off)


def _lib_defs_portable(ws_dir):
    """True if lib.defs references its libraries by RELATIVE path only (no drive letters)."""
    libdefs = ws_dir / "lib.defs"
    if not libdefs.exists():
        return True, "(no lib.defs)"
    txt = libdefs.read_text(encoding="utf-8", errors="replace")
    bad = [ln for ln in txt.splitlines()
           if (":\\" in ln or ":/" in ln or ln.strip().startswith("/"))]
    return (not bad), txt.strip()


def _readme(key, cfg, on_V, g_mod_in_dB, off_V):
    lib, top = cfg["lib"], cfg["top"]
    phi = cfg["baked_phase_deg"]
    common_tail = (
        "## How it is built\n\n"
        f"A clean hierarchical design: a `unit_cell` subcircuit (one cell's elements behind an\n"
        f"IN/OUT symbol) cascaded as `unit_cell:Xk` on the **{lib} > {top} > schematic** top sheet,\n"
        f"on the accurate distributed terminations (ADS `VIAGND` physical via + `MLEF` open-end\n"
        f"fringe). The `MEAS_GAIN` `MeasEqn` block is on the top sheet and the boundary nets are\n"
        f"named `n_input` / `n_output`, so a GUI **Simulate** reads the gain definitions directly.\n"
        f"`{lib}/flat_hb_deck.ckt` is the equivalent standalone `hpeesofsim` deck for a headless run.\n\n"
        "## Parity (build fidelity)\n\n"
        "This build is **byte-identical to the validated reference at the phase-0 parity point** --\n"
        "the geometry and the flat phase-0 deck are unchanged; only the showcase pump phase / signal\n"
        f"level differ here. The proof travels with this workspace as [`parity_{key}.json`](parity_{key}.json)\n"
        "(flat-vs-hierarchical `V(n_output)` agreement + the reference-value match); the canonical copy\n"
        f"and the phase-0 deck `{key}_flat_hb_deck.ckt` live in the repository at\n"
        "`examples/real_parts/results/` (and `results/decks/`).\n"
    )
    if key == "mavr":
        return (
            f"# {lib} -- amplifier (PSA-peak pump phase)\n\n"
            f"**MAVR-000350 COMBINED MLEF∥VIAGND parallel-trap degenerate parametric amplifier, "
            f"24 cells, 12 V, f_s = 3.0 GHz.**\n\n"
            f"This workspace opens at the **optimal pump phase ({phi:.0f}°)**. Run *Simulate*, then\n"
            f"plot `G_mod_in_dB` (the degenerate full-family out/in gain): it reads "
            f"**{g_mod_in_dB:+.2f} dB** here\n"
            f"(the reference level ~+14.14 dB PSA peak). The gain is **phase-sensitive** -- the pump phase is\n"
            f"THE tunable parameter:\n\n"
            f"| pump phase | `G_mod_in_dB` |\n|---|---|\n"
            f"| 0° (phase-0 parity point) | ~ +3.55 dB |\n"
            f"| ~{phi:.0f}° (this workspace -- the peak) | **{g_mod_in_dB:+.2f} dB** |\n"
            f"| ~30° (anti-phase) | ~ -14 dB (deamplification) |\n\n"
            f"The full lobe is in `mavr_gain_vs_pump_phase.png` (alongside). To move along it, edit the\n"
            f"`PUMP_SRC` source phase on the top schematic (it is baked as `polar(dbmtow(29),{phi:.0f})`).\n\n"
            + common_tail
        )
    if key == "smv2020":
        return (
            f"# {lib} -- oscillator (autonomous signal-off limit cycle)\n\n"
            f"**SMV2020-079LF SHORT / single-VIAGND end-via cascade, 24 cells, 16 V, f_s = 2.0 GHz "
            f"(regenerative edge).**\n\n"
            f"This workspace opens with the **signal driven off** ({SIGNAL_OFF_DBM:.0f} dBm) and the pump\n"
            f"on at the PSA-peak phase ({phi:.0f}°). Run *Simulate*, then plot the signal-frequency\n"
            f"output at `n_output` (f_s = 2.0 GHz) -- e.g. `mag(Vout_mod)` from `MEAS_GAIN`, the\n"
            f"on-f_s folding-family sum. It reads **~{on_V*1e3:.1f} mV with the signal off**: a finite\n"
            f"output that **persists with no input** is the self-oscillation signature (an autonomous\n"
            f"limit cycle), NOT a transducer gain.\n\n"
            f"For an oscillator the relevant readout is this **output amplitude**, not a gain ratio:\n"
            f"with the signal off the out/in ratio (`G_mod_in_dB`) is not meaningful (the `P_avs`\n"
            f"denominator is referenced to a nominal -30 dBm only so the equation stays finite). Raise\n"
            f"the `SIGNAL_SRC` power to inject a signal and watch the output stay pinned (input-\n"
            f"independent) -- the loss-invariant oscillation test.\n\n"
            + common_tail
        )
    # smv1245
    boost = _smv1245_framing()
    return (
        f"# {lib} -- fabricated 0 V board (pump-delivery lesson)\n\n"
        f"**As-built SMV1245-079LF board: OPEN/MLEF 3-section stub with two mid-stub VIAGND, "
        f"15 cells, 0 V, f_s = 3.1 GHz.**\n\n"
        f"This workspace opens at the board's **3.1 GHz operating point**. Run *Simulate*, then plot\n"
        f"`G_mod_in_dB` (net out/in): it reads **{g_mod_in_dB:+.2f} dB** -- net negative, equal to the\n"
        f"passive insertion loss. At 0 V the pump is **delivery-starved across the package**, so the\n"
        f"modulation adds essentially nothing to the output.\n\n"
        f"{boost}\n\n"
        f"The boost-vs-frequency sweep is in `smv1245_boost_vs_freq.png` (alongside): boost (pump-on/off)\n"
        f"and net out/in across the band, with the passive transmission marking where the stopband is.\n\n"
        + common_tail
    )


def _smv1245_framing():
    """One-paragraph honest framing of the Part 1 boost-vs-frequency result, read from its JSON."""
    path = RESULTS / "headline_smv1245_boost_vs_freq.json"
    if not path.exists():
        return ("The Part 1 boost-vs-frequency result is recorded in "
                "`../../results/headline_smv1245_boost_vs_freq.json`.")
    d = json.loads(path.read_text(encoding="utf-8"))
    peak = d.get("peak_boost_meaningful") or d.get("peak_boost_overall") or {}
    null = d.get("stopband_null") or {}
    reaches = d.get("peak_boost_reaches_measured")
    pb = peak.get("boost_dB")
    pf = peak.get("f_signal_GHz")
    if reaches:
        return (f"**Part 1 (boost vs frequency):** the pump-on/off boost reaches **{pb:+.2f} dB** near "
                f"{pf:.3f} GHz -- the modulation lifts the deep-stopband signal even though the net "
                f"stays negative (delivery starved). That boost-vs-frequency curve is this board's headline.")
    return (
        f"**Part 1 (boost vs frequency):** across the physical passband the pump-on/off boost stays "
        f"**negligible** (peak ~{pb:+.2f} dB near {pf:.3f} GHz); the passive transmission is a smooth "
        f"high-pass rolloff with no localized stopband null that the modulation lifts. The ~+16 dB lift "
        f"the fabricated board measured does **not** reproduce in simulation: at 0 V this board is "
        f"deeply pump-delivery-starved and the modulation barely registers."
    )


def _build_one(key, cfg):
    from tvtl_ads_builder_v2.oa.oa_hier_emitter import emit_hierarchical_workspace_from_spec

    lib, top, ws_name = cfg["lib"], cfg["top"], cfg["ws"]
    ws_dir = OUTPUTS / ws_name
    print(f"\n=== showcase: {key} -> {ws_name} (baked phase {cfg['baked_phase_deg']:.0f} deg) ===",
          flush=True)

    # 1. verify the baked operating point reproduces the headline headlessly
    on_V, g_mod_in_dB, off_V = _verify_headless(cfg, OUTPUTS / "_showcase_verify" / key)
    if cfg["metric"] == "out_mV":
        got, exp = on_V * 1e3, cfg["expect"]
        ok = abs(got - exp) <= cfg["tol"]
        print(f"  headless: |V(n_output)| signal-off = {got:.2f} mV  (expect ~{exp} mV)  "
              f"{'OK' if ok else 'REVIEW'}", flush=True)
    else:
        got, exp = g_mod_in_dB, cfg["expect"]
        ok = abs(got - exp) <= cfg["tol"]
        print(f"  headless: G_mod_in_dB @ {cfg['baked_phase_deg']:.0f} deg = {got:+.2f} dB  "
              f"(expect ~{exp:+.2f})  {'OK' if ok else 'REVIEW'}", flush=True)

    # 2. emit the self-contained hierarchical workspace
    res = emit_hierarchical_workspace_from_spec(
        cfg["spec"], ws_dir, library_name=lib, top_cell_name=top, uc_cell_name="unit_cell")

    # 3. embed MEAS_GAIN + name n_input / n_output
    eq = _embed_gain_equations(ws_dir, lib, top, p_avs_W=cfg["p_avs_W"])

    # 4. text deliverables: flat demo deck, per-example README, headline figure, .gitignore
    from tvtl_ads_builder_v2.netlist import emit_flat_hb_netlist
    (ws_dir / lib / "flat_hb_deck.ckt").write_text(emit_flat_hb_netlist(cfg["spec"]), encoding="utf-8")
    (ws_dir / "README.md").write_text(_readme(key, cfg, on_V, g_mod_in_dB, off_V), encoding="utf-8")
    (ws_dir / ".gitignore").write_text(_GITIGNORE, encoding="utf-8")
    fig_copied = None
    if cfg["figure"]:
        src = RESULTS / cfg["figure"]
        if src.exists():
            shutil.copy2(src, ws_dir / cfg["figure"])
            fig_copied = cfg["figure"]
    # bundle the phase-0 parity proof so the workspace is self-documenting standalone
    psrc = RESULTS / f"parity_{key}.json"
    if psrc.exists():
        shutil.copy2(psrc, ws_dir / f"parity_{key}.json")

    # 5. portability + self-containment checks
    portable, libdefs_txt = _lib_defs_portable(ws_dir)
    print(f"  emitted n_cells={res['n_cells']}; gain eqns {eq.get('n_gain_eqs')} "
          f"(nets {eq.get('renamed_nets')}); figure={fig_copied}", flush=True)
    print(f"  lib.defs portable (relative paths only): {'YES' if portable else 'NO -- REVIEW'}", flush=True)
    return {
        "key": key, "workspace": str(ws_dir), "library": lib, "top": top,
        "baked_phase_deg": cfg["baked_phase_deg"],
        "headless_on_mV": on_V * 1e3, "headless_g_mod_in_dB": g_mod_in_dB,
        "metric_ok": bool(ok), "figure": fig_copied,
        "lib_defs_portable": bool(portable),
        "gain_eqs": eq.get("n_gain_eqs"), "renamed_nets": eq.get("renamed_nets"),
    }


def main(argv):
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    specs = _demo_specs()
    keys = [k for k in ("mavr", "smv2020", "smv1245") if not argv or k in argv]
    summary = []
    for k in keys:
        summary.append(_build_one(k, specs[k]))
    shutil.rmtree(OUTPUTS / "_showcase_verify", ignore_errors=True)

    print("\n=== SHOWCASE SUMMARY ===", flush=True)
    all_ok = True
    for s in summary:
        all_ok = all_ok and s["metric_ok"] and s["lib_defs_portable"]
        print(f"  {s['key']:8s}: {Path(s['workspace']).name}  "
              f"metric_ok={s['metric_ok']}  portable={s['lib_defs_portable']}  "
              f"figure={s['figure']}", flush=True)
    (RESULTS / "showcase_workspaces.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n  >>> showcase {'OK' if all_ok else 'REVIEW'}; wrote {RESULTS / 'showcase_workspaces.json'}",
          flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
