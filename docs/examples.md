# TVTL Examples — a tour of the six designs

Six worked designs ship under `examples/real_parts/`, built end to end from the
public core. They split into **three real-part designs** (commercial varactor
SPICE, frozen as-built geometry) and **three optimizer-picked proxy designs** (the
public optimizer's diversified picks on the idealized proxy varactor). All six are
self-pumped degenerate parametric structures (pump at twice the signal frequency).

This page is the *why* — what each design demonstrates and the lesson it teaches.
For the build/analyze commands see [workflow.md §5](workflow.md#5-the-six-example-designs-build_examplespy)
and `examples/real_parts/README.md`. **Every number below is traced to a committed
artifact under `examples/real_parts/results/`, and labeled predicted (analytical
ranking proxy) or realized (the ADS harmonic-balance record)** — the distinction
explained in [theory §5](theory.md#5-predicted-vs-realized-the-honesty-rule).

| design | varactor | termination | cells | operating point | the lesson |
|---|---|---|---|---|---|
| `real_amplifier_MAVR` | MAVR-000350 | COMBINED `MLEF ∥ VIAGND` trap | 24 | 12 V, 3.0 GHz | a stable high-gain amplifier |
| `real_oscillator_SMV2020` | SMV2020-079LF | single `VIAGND` short | 24 | 16 V, 2.0 GHz | the self-oscillation regime |
| `fabricated_SMV1245` | SMV1245-079LF | OPEN/`MLEF` 3-stub + 2 mid-stub `VIAGND` | 15 | 0 V, 3.1 GHz | pump-delivery starvation |
| `proxy_n2_double_short` | proxy "Varactor A" | SHORT, double via + matching | 15 | 2.0 GHz | framework range |
| `proxy_n3_single_short` | proxy "Varactor A" | SHORT, single via + matching | 15 | 2.0 GHz | the net-positive record |
| `proxy_n4_double_short` | proxy "Varactor A" | SHORT, double via + matching | 15 | 2.0 GHz | framework range |

> **Via count (v1.0.1).** Two designs use a `DOUBLE` via family, `proxy_n2_double_short` and
> `proxy_n4_double_short`. Release 1.0.0 emitted one `VIAGND` per cell for them; 1.0.1 emits two, and their
> golden decks under `results/decks/` are regenerated.
>
> For these two: ADS record generated with the pre-1.0.1 generator (one via per cell); analytical leg with
> two vias; re-simulation scheduled after 1.0.1.
>
> `real_oscillator_SMV2020` is a single-via design. Its registry entry said `n1_double_short` up to 1.0.0
> while the deck had one via; 1.0.1 relabels it `n1_single_short`. Its deck circuit is unchanged (only the
> deck's label comment changes) and its records stay valid. The deck hashes are in `CHANGELOG.md`.

---

## The three real-part designs

These use real datasheet SPICE parameters from `tvtl.varactor_palette` and frozen
microstrip geometry — no search, no spreadsheet. All run at a +29 dBm pump.

### `real_amplifier_MAVR` — a stable high-gain amplifier

**Demonstrates** that, with pump delivery solved and bias/frequency co-selected, a
real catalog part gives a genuine bounded degenerate amplifier.

- **Topology** — MAVR-000350, COMBINED `MLEF ∥ VIAGND` parallel trap, 24 cells.
- **Operating point** — 12 V bias, 3.0 GHz signal (6.0 GHz pump), −30 dBm signal.
- **Realized (the record)** — the pump-**phase** sweep of `G_mod_in_dB(φ)` is a
  single bounded lobe: **peak +14.35 dB at φ = 210°**, **+3.55 dB at φ = 0°**, and
  a deamplification dip to **−14.05 dB at φ = 30°**. One maximum, one minimum,
  **every phase converges** — a bounded PSA lobe, not a runaway (reference level
  +14.14 dB). Source: `results/headline_mavr.json`,
  `results/mavr_gain_vs_pump_phase.png`. The *certified* quantities are the recorded
  gate-output fields in that JSON (`peak`, `single_lobe`, `bounded`,
  `all_phases_converged`); the +14.14 dB line is only a historical reference level
  (it matches the φ = 240° sweep sample), not a gate output.
- **Predicted** — the analytical coherent gain ranks this COMBINED-trap design
  positive; it is the optimistic ranking proxy, and the +14.35 dB above is the
  realized HB record, not the prediction.
- **Lesson / quirk** — the COMBINED fork is the design that motivated the
  hierarchical emitter: in a flat OpenAccess schematic with a source present, ADS
  recomputes connectivity from pin geometry and merges the `MLEF ∥ VIAGND` fork;
  inside the source-free `unit_cell` subcircuit it renders correctly. This is the
  [certified stability methodology](theory.md#7-stability-methodology-how-a-stable-operating-point-is-certified)
  in practice.

### `real_oscillator_SMV2020` — the self-oscillation regime

**Demonstrates** what the [regime detector](theory.md#the-loss-invariant-regime-detector)
calls *lasing*, and why a big "gain" reading is not always an amplifier.

- **Topology** — SMV2020-079LF, single `VIAGND` short, 24 cells.
- **Operating point** — 16 V bias, 2.0 GHz signal (4.0 GHz pump).
- **Realized (the record)** — drive the input to ≈ off and a **30.9 mV** limit
  cycle persists at the signal frequency. The input-independence ratio
  `r_A = 0.87` and the gain-linearity slope `≈ 0.08` both say the output is pinned
  regardless of the input → regime **lasing**, classification **oscillation**.
  That 30.9 mV is an oscillation signature, **not a transducer gain**. Source:
  `results/headline_smv2020.json`.
- **Lesson** — this design sits at a near-oscillation regenerative point. Its
  parity check agrees flat-vs-hierarchical only to the harmonic-balance
  convergence floor (~6e-9 V) rather than bit-for-bit — expected at a regenerative
  edge, not a structural difference
  ([troubleshooting](troubleshooting.md#the-smv2020-near-oscillation-parity-floor)).

### `fabricated_SMV1245` — the pump-delivery lesson

**Demonstrates** the central cautionary result of the whole study: the parametric
mechanism can work while the device is still net-negative, because the pump never
reaches the junction.

- **Topology** — the as-built SMV1245-079LF board: OPEN/`MLEF` 3-section stub with
  two mid-stub `VIAGND`, 15 cells.
- **Operating point** — **0 V bias**, 3.1 GHz signal (6.2 GHz pump), −20 dBm signal.
- **Realized (the record)** — the pump-on/off boost is just **+0.001 dB** and the
  net out/in `G_mod_in_dB` is **−6.46 dB**, equal to the passive insertion loss
  (`passive_S21 = −6.457 dB`). Sweeping the drive across the band, the boost stays
  **≤ +0.3 dB everywhere resolvable** — even at the ~2.46 GHz point where the
  passive loss matches the measured −44.7 dB notch, the boost is only **+0.05 dB**.
  The analyzer-side pump-on/off difference on the board is not reproduced by this ADS model. Source:
  `results/headline_smv1245.json`, `results/headline_smv1245_boost_vs_freq.json`,
  `results/smv1245_boost_vs_freq.png`.
- **Lesson** — this model gives only a tiny positive on/off difference at 0 V and
  predicts a **delivery-starved junction** ([theory §6](theory.md#6-pump-delivery-why-catalog-parts-are-delivery-limited)).
  Net-positive operation on a catalog part needs bias and frequency
  co-selected so the junction stays above starvation.

---

## The three optimizer-picked proxy designs

The public optimizer's deterministic `--freq-ghz 2.0` run selects exactly these
three (verified: `floquet-line-design optimize-amplifier --freq-ghz 2.0`):

```text
Candidates: 3247 screened, top-10 refined, 7 realizable, 3 selected
rank 1  n2_double_short   coherent_peak = 10.63 dB
rank 2  n4_double_short   coherent_peak = 10.40 dB
rank 3  n3_single_short   coherent_peak = 10.11 dB
```

They cover `(stub-count × termination)` cells the real-part trio does not, and
they are the **first examples whose edge impedances require input/output
single-stub matching networks** — which is why they are included: they confirm the
hierarchical emitter places matching networks cleanly. All use the low-parasitic
proxy *Varactor A* (`CJ0 0.3 pF, L_pkg 0.3 nH, R 0.5 Ω`), 15 cells, 2.0 GHz,
`Z_TL = 61.27 Ω`, each baked at a convergent, bounded operating point.

**Predicted vs realized**, side by side (predicted = optimizer/analytical coherent
peak, `candidates.csv`; realized = ADS HB, `results/proxy_analysis_summary.json`):

| design | op. pump | predicted | realized boost (peak φ) | realized abs. out/in |
|---|---|---|---|---|
| `proxy_n2_double_short` | +6 dBm | +10.63 dB | +2.86 dB @ 45° | −3.92 dB |
| `proxy_n3_single_short` | +15 dBm | +10.11 dB | +8.18 dB @ 45° | **+1.18 dB** (net-positive) |
| `proxy_n4_double_short` | +9 dBm | +10.40 dB | +2.51 dB @ 90° | −4.51 dB |

Every one has a **bounded PSA lobe** and passes parity **IDENTICAL**.

**Lesson** — the prediction ranks all three in a tight 10.1–10.6 dB band; harmonic
balance separates them into a 2.5–8.2 dB realized boost. `proxy_n3_single_short`
is the **framework's net-positive record** (+1.18 dB absolute out/in): the
cleanest demonstration that the modeled mechanism can exceed unity in ADS HB on
the idealized proxy. It is a *faithfulness* result on an idealized device — not a
real-world device-gain claim. The real parts above show why: a catalog part is
delivery-limited, so the same mechanism lands net-negative until bias and
frequency are co-selected. That +1.18 dB is the proxy's net-positive *ceiling*, and
15 cells is its robust optimum length — both characterized in
[LIMITATIONS.md](LIMITATIONS.md#characterized-performance-limits).

---

## What ties the six together: the gates

Every design passes the same two gates (see
[workflow §5](workflow.md#5-the-six-example-designs-build_examplespy)):

- **Parity** — the hierarchical design's signal-frequency output voltage equals
  the flat reference deck's: bit-identical for all but the near-oscillation
  SMV2020 point (which agrees to the HB convergence floor). Reference
  `V(n_output)`: MAVR `2.615556e-02 @ 3.0 GHz`, SMV1245 `1.503640e-02 @ 3.1 GHz`,
  SMV2020 `1.542714e-02 @ 2.0 GHz`; proxies `4.115031e-03 / 6.151059e-03 /
  4.114838e-03 @ 2.0 GHz`. Source: `results/parity_<key>.json`.
- **Overlap scan** — every emitted drawing (unit cell and top level, all six)
  records **0 strict overlaps**. Source: `results/overlap_scan.json`.

Open any of them: `build_examples.py assemble` writes one ADS workspace with a
library per design (`examples/real_parts/outputs/TVTL_examples_workspace/`), and
`showcase_workspaces.py` writes one self-contained workspace per real-part design
baked to its headline operating point. Plot `G_mod_in_dB` after **Simulate** to see
the realized number.

---

## Where to go next

- The physics each design exercises → [theory.md](theory.md)
- How to build and analyze them → [workflow.md](workflow.md) and
  `examples/real_parts/README.md`
- Add your own varactor or topology → [extending.md](extending.md)
- The gain definitions behind `G_mod_in_dB` → [GAIN_DEFINITIONS.md](GAIN_DEFINITIONS.md)
