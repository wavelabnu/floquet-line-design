# TVTL example designs

**Six** worked, paper-scale varactor-loaded time-varying transmission-line designs,
built end-to-end from the public `tvtl` physics core and the public
`tvtl_ads_builder_v2` ADS builder: **three real-part designs** (commercial varactor
SPICE, frozen geometry — `real_part_designs.py`) and **three optimizer-picked proxy
designs** (the public optimizer's diversified picks on the low-parasitic proxy varactor
— `proxy_designs.py`). Each is a frozen design point; nothing here runs a search at
build time or reads an external spreadsheet, so the designs reproduce offline.

## Three real-part designs (commercial varactor SPICE)

| design (library) | varactor | termination | cells | operating point |
|---|---|---|---|---|
| `fabricated_SMV1245` | SMV1245-079LF | OPEN/MLEF 3-section stub + two mid-stub VIAGND | 15 | 0 V, 3.1 GHz, +29 dBm |
| `real_oscillator_SMV2020` | SMV2020-079LF | single VIAGND short | 24 | 16 V, 2.0 GHz, +29 dBm |
| `real_amplifier_MAVR` | MAVR-000350 | COMBINED MLEF ∥ VIAGND trap | 24 | 12 V, 3.0 GHz, +29 dBm |

All three are self-pumped (degenerate, pump at twice the signal frequency).

## What "hierarchical" means here

Each design is emitted through the hierarchical OpenAccess emitter
(`tvtl_ads_builder_v2.oa.oa_hier_emitter`): one `unit_cell` subcircuit holds a
single cell's elements behind a two-terminal (IN/OUT) symbol, and the top-level
schematic cascades it as `unit_cell:X1 … unit_cell:XN` on one tidy left-to-right
rail, with the harmonic-balance controller and the signal/pump sources at the left.

This is electrically identical to the flat single-sheet deck but far easier to read,
and it resolves a flat-emitter limitation for the MAVR design: the COMBINED
`MLEF ∥ VIAGND` fork collapses in the flat OA schematic (with a source present, ADS
recomputes connectivity from pin geometry and merges the fork), but renders correctly
in the hierarchical form because the unit-cell subcircuit carries no source.

## Run it

Build the three specs and check their structure (no ADS required):

```powershell
python build_examples.py specs
```

This validates that each regenerated spec matches its recorded design fingerprint
(element/node counts and per-cell body) and writes the runnable flat `hpeesofsim`
decks to `results/decks/`.

Under ADS Python (the `keysight.ads.de` OpenAccess bridge and the `hpeesofsim`
harmonic-balance engine), prove parity, render screenshots, and assemble one
openable workspace:

```powershell
$env:PYTHONPATH = "..\..\src"   # run from examples\real_parts; points ADS Python at the repo src
& $env:ADS_PYTHON build_examples.py parity all     # emit flat + hier, HB both, compare Vout (all six)
& $env:ADS_PYTHON build_examples.py shots  all     # render unit-cell + top-level PNGs + OA overlap scan
& $env:ADS_PYTHON build_examples.py assemble       # one openable ADS workspace (six libraries)
```

(`$env:ADS_PYTHON` is your ADS Python interpreter,
`<ADS_INSTALL_DIR>\tools\python\python.exe` — see the top-level README.)

## The parity gate

`parity` is the correctness gate. For each design it emits the hierarchical workspace
and the flat reference deck, runs harmonic balance on both, and compares the
signal-frequency output-node voltage `V(n_output)`. The hierarchical and flat results
must agree, and each must reproduce the recorded reference value:

| design | reference `V(n_output)` @ signal | flat vs hierarchical |
|---|---|---|
| `fabricated_SMV1245` | 1.503640e-02 @ 3.1 GHz | bit-identical |
| `real_oscillator_SMV2020` | 1.542714e-02 @ 2.0 GHz | equal to the HB convergence floor |
| `real_amplifier_MAVR` | 2.615556e-02 @ 3.0 GHz | bit-identical |

SMV2020 sits at a near-oscillation regenerative operating point, where the flat and
hierarchical solver paths agree only to the harmonic-balance convergence floor
(~6e-9 V) rather than bit-for-bit; that floating-point-floor difference is expected,
not a structural difference. The three proxy designs are each bit-identical flat-vs-
hierarchical at their operating point (`V(n_output)` = 4.115031e-03 / 6.151059e-03 /
4.114838e-03 V @ 2.0 GHz). The per-design proofs are written to
`results/parity_<design>.json`. `shots` additionally runs the OA overlap scan on every
emitted drawing — unit cell and top level, including the matched proxy designs — and
records **0 strict overlaps** to `results/overlap_scan.json`.

## Gain readout

`assemble` bakes a `MEAS_GAIN` `MeasEqn` block onto each top schematic and names the
boundary nets `n_input` / `n_output`. Open the assembled workspace in ADS, run Simulate,
and plot `G_mod_in_dB` (the degenerate full-family out/in) or `Gain_coherent_dB` (the
2-term coherent gain); the block also carries the per-term building blocks and the
`Gain_ref_dB` / `Gain_4f_dB` comparison definitions (see
`docs/GAIN_DEFINITIONS.md`). The equations evaluate on the ADS harmonic-balance dataset
— the realized record, not the analytical prediction. `P_avs` is set per example from the
signal drive level (−20 dBm for SMV1245, −30 dBm for the others). Per-cell branch-power
equations are not added in the hierarchical form, because those nodes live inside the
`unit_cell` subcircuits and are not referenceable from the top level by a flat name.

## Headline readouts

Each example also earns the one number that actually describes its role, layered on top
of the byte-identical phase-0 build (which stays put as the parity evidence). All three
run through the committed ADS machinery — one resolved deck per point, no `src/` change.
Run under ADS Python:

```powershell
& $env:ADS_PYTHON headline_measurements.py all     # mavr + smv2020 + smv1245
```

- **MAVR (amplifier)** — a pump-**phase** sweep of the degenerate full-family out/in gain
  `G_mod_in_dB(φ)`. The phase-sensitive lobe is the headline: it peaks at **+14.35 dB**
  (φ = 210°, the ~+14 dB PSA peak), passes the **+3.55 dB** phase-0 parity point,
  and dips to **−14.05 dB** deamplification near the anti-phase (φ = 30°). The peak is a
  BOUNDED single lobe — every phase converges — not a runaway. Figure:
  `results/mavr_gain_vs_pump_phase.png`. The certified quantities are the recorded
  gate-output fields in `results/headline_mavr.json` (`peak`, `single_lobe`,
  `bounded`, `all_phases_converged`); the +14.14 dB line is a historical *reference
  level* (it matches the φ = 240° sweep sample), not a gate output. The
  `certified_reference_dB` field name in that JSON is retained for historical continuity.
- **SMV2020 (oscillator)** — the autonomous signal-**off** output `|V(n_output)|` at f_s
  (pump on, signal driven to ≈ off). It self-oscillates: a finite **30.9 mV** limit cycle
  persists with the input off (`r_A = |off|/|on| = 0.87 → 1`) and the output is pinned
  regardless of the input (input-dependence slope ≈ 0, regime *lasing*). That is an
  oscillation signature, NOT a transducer gain.
- **SMV1245 (fabricated 0 V board)** — the pump-delivery lesson, shown two ways. At the
  3.1 GHz operating point the pump-on/off boost is just **+0.001 dB** and the net out/in
  `G_mod_in_dB` = **−6.46 dB** (= the passive insertion loss). Sweeping the drive across the
  band (`smv1245_boost_sweep.py`, the SMV1245 analog of the MAVR phase sweep, one resolved
  deck per point) confirms the lesson: the boost stays **≤ +0.3 dB everywhere resolvable** —
  even at the ~2.46 GHz point where the passive insertion loss matches the measured −44.7 dB
  notch the boost is only **+0.05 dB**: the model records no amplification at this point, consistent with the 2026 bench controls, which attribute the earlier pump-on/off difference to the pump path rather than to the injected signal. The
  passive response is a smooth high-pass rolloff with no localized stopband null the modulation
  lifts; at 0 V the pump is delivery-starved across the package. Figure:
  `results/smv1245_boost_vs_freq.png`.

The per-example readouts are written to `results/headline_<key>.json` (and the frequency sweep
to `results/headline_smv1245_boost_vs_freq.json`).

## Three optimizer-picked proxy designs (matching networks)

The public optimizer picks these three — they cover `(stub-count × termination)` cells the
real-part trio does not, and they are the FIRST examples whose edge impedances require
input/output **matching networks** (the trio's need none), so they confirm the hierarchical
emitter places matching networks cleanly outside the generalization test. The pick is
deterministic:

```powershell
floquet-line-design optimize-amplifier --freq-ghz 2.0 --out <dir>   # the three diversified picks
```

All use the low-parasitic proxy varactor (CJ0 0.3 pF, L_pkg 0.3 nH, R 0.5 Ω, `proxy_designs.py`),
15 cells, f_s = 2.0 GHz, single-stub input/output matching, Z_TL = 61.27 Ω. Each is baked at its
**convergent, bounded operating point** (HB converges, the degenerate-PSA phase lobe is bounded —
not an oscillation-edge runaway). Predicted (analytical, optimistic ranking proxy) and realized
(ADS HB) gain are reported separately; the idealized proxy's realized number is a faithfulness
record, not a real-world device-gain claim:

| design (library) | cell | op. pump | predicted | realized boost (peak φ) | abs. out/in |
|---|---|---|---|---|---|
| `proxy_n2_double_short` | n2 SHORT, double via | +6 dBm | +10.63 dB | +2.86 dB @45° | −3.92 dB |
| `proxy_n3_single_short` | n3 SHORT, single via | +15 dBm | +10.11 dB | +8.18 dB @45° | **+1.18 dB** (net-pos) |
| `proxy_n4_double_short` | n4 SHORT, double via | +9 dBm | +10.40 dB | +2.51 dB @90° | −4.51 dB |

For the net-positive `proxy_n3_single_short`, **N = 15 is validated as the robust optimum across N = 6–30**.

**`proxy_n2_double_short` and `proxy_n4_double_short`:** ADS record generated with the pre-1.0.1 generator
(one via per cell); analytical leg with two vias; re-simulation scheduled after 1.0.1. Since 1.0.1 the
generated decks carry two `VIAGND` per cell (`results/decks/`).

The per-design analysis (pump-power scan, phase-lobe stability, predicted vs realized) is in
`analyze_proxy.py`, written to `results/proxy_analysis_<key>.json`:

```powershell
& $env:ADS_PYTHON analyze_proxy.py all
```

## Showcase workspaces (one openable workspace per example)

`showcase_workspaces.py` regenerates one self-contained, portable ADS workspace per example
under `outputs/`, each baked to the operating point that demonstrates its headline so that
opening it and running *Simulate* shows the result:

```powershell
& $env:ADS_PYTHON showcase_workspaces.py     # all three (or: mavr | smv2020 | smv1245)
```

- `TVTL_real_amplifier_MAVR_wrk` — opens at the optimal pump phase (210°); plot `G_mod_in_dB`
  → **+14.35 dB**, with `mavr_gain_vs_pump_phase.png` alongside (pump phase = the tunable parameter).
- `TVTL_real_oscillator_SMV2020_wrk` — signal driven off, pump on at the PSA-peak phase;
  plot `mag(Vout_mod)` → the autonomous **~30.9 mV** limit cycle.
- `TVTL_fabricated_SMV1245_wrk` — the 3.1 GHz operating point (net out/in **−6.46 dB**) with the
  Part 1 `smv1245_boost_vs_freq.png` alongside.

Each workspace carries a per-example `README.md`, a portable relative `lib.defs`, the embedded
`MEAS_GAIN` block (boundary nets named `n_input` / `n_output`), and the phase-0 `parity_<key>.json`
proof — byte-identical to the validated reference at the phase-0 parity point.

## Outputs

- `results/` (committed): `specs.json`, `parity_<design>.json`, `decks/`,
  `screenshots/` (`<design>_unit_cell.png` and `<design>_top_level.png`),
  `headline_<design>.json`, `headline_smv1245_boost_vs_freq.json`,
  `mavr_gain_vs_pump_phase.png`, and `smv1245_boost_vs_freq.png`.
- `outputs/` (git-ignored, regenerated): the three per-example showcase workspaces
  (`TVTL_real_amplifier_MAVR_wrk`, `TVTL_real_oscillator_SMV2020_wrk`,
  `TVTL_fabricated_SMV1245_wrk`), plus harmonic-balance run scratch.

## Files

- `real_part_designs.py` — the three frozen real-part designs (geometry literals + spec builders).
- `proxy_designs.py` — the three optimizer-picked proxy designs (frozen picks + spec builders).
- `build_examples.py` — the `specs` / `parity` / `shots` / `assemble` driver (all six examples).
- `analyze_proxy.py` — the proxy predicted-vs-realized + phase-lobe stability driver.
- `oa_overlap_scan.py` — the OA-geometry strict-overlap scan used by `shots`.
- `headline_measurements.py` — the `mavr` / `smv2020` / `smv1245` headline-readout driver.
- `smv1245_boost_sweep.py` — the SMV1245 boost-vs-frequency sweep (Part 1).
- `showcase_workspaces.py` — regenerates one openable ADS workspace per real-part example, baked to its headline.
- `_render.py` — renders a saved OpenAccess schematic to PNG.
