# Public Supplement Limitations

This repository is intended to make the reusable TVTL analytical core and v2 ADS
builder inspectable and testable. It is not a complete private research archive.

## ADS Dependency

The base Python package imports and tests do not require Keysight ADS. Workflows
that launch `hpeesofsim`, inspect OpenAccess workspaces, or read ADS harmonic
balance datasets require a local ADS installation and Python bridge.

Use local placeholders in scripts and notes:

```powershell
$env:HPEESOF_DIR = "<ADS_INSTALL_DIR>"
$env:TVTL_AEL_REFERENCE_WORKSPACE = "<ADS_REFERENCE_WORKSPACE>"
$env:TVTL_OUTPUT_DIR = "<OUTPUT_DIR>"
```

## Unsupported ADS Constructs

The v2 builder resolves named scalar variables before emission. It does not emit
ADS `VAR`, `Vars`, `Eqn`, or OA `VarEqn` constructs for portable public
workflows because those forms were not reliable across the ADS version used for
the paper work.

For nonlinear varactors, the public ADS realization uses `SDD2P` charge
expressions. Those decks still require local ADS validation before making final
gain, stability, or convergence claims.

## Optional Evidence And Calibration Data

Some evidence/calibration artifacts are optional and are not bundled, including
recorded AEL macro workspaces and dense MTEE surrogate tables. Code paths that
need them accept environment-variable overrides and fail with a clear message
when the artifacts are unavailable.

## Generated Outputs

Generated workspaces, datasets, figures, reports, and diagnostic logs are not
included. Write regenerated artifacts under an explicit output directory such as
`<OUTPUT_DIR>` or a repository-local ignored `runs/` directory.

## Paper model boundaries

The `runs/…` reports cited in this file are in the companion data deposit, which holds the measured data set
and the R1–R7 run records. The measured data set is archived separately on Zenodo; its DOI is given in the
paper's Data availability statement.

- R3's extracted via inductance is **1.90 nH** for the evaluated board/deck, versus **0.195 nH** in the design-tier model. The change materially affects the passive boundary; see the R3 report, data deposit `runs/r3_via_fit/REPORT.md`.
- R4b estimates about **90%** pump absorption for the board and **22.5%** for the evaluated deck. These are distinct paths and should not be treated as one validated pump delivery; see the R4b report, data deposit `runs/r4b_fulldiode_hb/REPORT.md`.
- The full-diode R4b channel produces at most **0.007 dB** in the stated checks. It does not account for the pump-on/off difference recorded on the analyzer, which the 2026 controls attribute to the pump path rather than to amplification of the injected signal; see the R4b report (data deposit, `runs/r4b_fulldiode_hb/REPORT.md`).
- Across **936** R5 lumped-via realizations, the recorded upper absolute out/in is **+0.035 dB** under that search and model. The bound is model- and screen-specific; see the R5 report, data deposit `runs/r5_gain_search/REPORT.md`.
- R5/R6's accepted proxy calibration bias is bounded by **0.12 dB** for the stated anchor; geometry and bias transfer remain unvalidated. See the R5 report, data deposit `runs/r5_gain_search/REPORT.md`, and R6 report, data deposit `runs/r6_delivery_core/REPORT.md`.
- R7 shows a transient-versus-harmonic-balance transport mismatch and limits direct transfer of a threshold from one solver to the other. See the R7 threshold report, data deposit `runs/r7_subharmonic_threshold/REPORT.md`, and R7 bias-taper report, data deposit `runs/r7_bias_taper/REPORT.md`.
- The paper has **no measured device threshold** and no measured amplification: the two-tone controls (signal off, device removed, signal added) bound the response of the injected signal to the linear-superposition window (−1.9…+1.6 dB expected; −2.3…+1.2 dB observed at 11 points). The 2025 two-tone spectra overdrive the analyzer front end, so their levels are relative and those spectra are not in the cited data deposit.

## Release 1.0.1: enforced limits and open items

Enforced in 1.0.1, so that the release refuses instead of returning a number:

- **Tee-model range.** Above the MTEE reference-plane validity edge (11.64 GHz for the SMV1245 cell), cell
  samples raise `TeeModelOutOfRange`, and the gates refuse with `tee_model_out_of_range`. The reduced model
  (no tee correction) is available only when requested explicitly. The Fig. 7 tables use it, and
  `scripts/dense_map --mode tee` lists where: pump samples from `f_s = 5.85 GHz`, the proxy's `3 f_s` from
  3.9 GHz.
- **Via count.** Double-via families emit two `VIAGND` per cell. For the two double-via proxy examples,
  `proxy_n2_double_short` and `proxy_n4_double_short`: ADS record generated with the pre-1.0.1 generator
  (one via per cell); analytical leg with two vias; re-simulation scheduled after 1.0.1. The
  `real_oscillator_SMV2020` example is a single-via family (`n1_single_short`, relabelled in 1.0.1). Its
  deck and ADS records are those of 1.0.0.
- **DC return.** A varactor terminal without a DC path to ground is refused (`no_dc_return_for_varactor`).
  Open-stub families need a declared bias-return element.
- **Depth ceiling.** A candidate whose delivered depth exceeds the C(V) ceiling at its bias is refused before
  harmonic balance (`delivered_depth_above_ceiling`).

Known open items, not fixed in 1.0.1:

- **Harmonic-block tee fallback.** Two tee models still fall back to the ideal junction above their
  validity edge: the harmonic-block MTEE model (`mtee_blocks.tee_harmonic_blocks`, used by
  `realistic_c_gain`) and the single-frequency tee of `level10_sparams`. This is a documented fallback, not
  a refusal. The two-filter sweep behind Fig. 2(c) uses the harmonic-block path for pump samples up to
  16 GHz.
- **Transient versus harmonic-balance transport.** The mismatch reported by R7 limits the transfer of a
  threshold between the two solvers.
- **Absorption.** The measured absorption at 6.2 GHz is 89.7 %; the evaluated deck absorbs 22.5 % (R4b). The
  analytical and circuit cells do not reproduce the board's loss.
- **Full-wave parasitics.** Pad and neck parasitics from full-wave models are not in the analytical screen.
  The stage-0 gate and the cascade layer use the circuit cell, so their verdicts can differ from an EM
  model of the same layout.
- **Proxy calibration anchors.** The anchors of the delivered-modulation proxy sit at `f_s = 3.0-3.1 GHz`.
  Elsewhere the proxy ranks candidates, and harmonic-balance records decide.

## Characterized Performance Limits

The numbers in this section are **ADS harmonic-balance records on the idealized
low-parasitic proxy varactor** ("Varactor A"), at the 2.0 GHz signal / 4.0 GHz pump
operating point with the accurate `VIAGND`/`MLEF` terminations. They are faithfulness
records of the modeled mechanism on an idealized device — **not** real-world device-gain
claims, since catalog parts are
[pump-delivery limited](theory.md#6-pump-delivery-why-catalog-parts-are-delivery-limited).
Analytical predictions and ADS-realized records are kept separate throughout, per the
[predicted-vs-realized rule](theory.md#5-predicted-vs-realized-the-honesty-rule).

### Gain ceiling at 2 GHz (idealized proxy)

How high does a *stable* proxy design go?

- **The best stable linear (OPEN/SHORT) design** (`n3_single_short`, 15 cells) records a
  **+8.18 dB** pump-on/off boost over a **−7.00 dB** passive-insertion-loss baseline, for
  **+1.18 dB** absolute out/in (ADS-realized; the analytical proxy predicted a +10.11 dB boost).
- **COMBINED (`MLEF ∥ VIAGND`) families are analytical-screening-only** in this release; the
  optimizer does not rank them as amplifier candidates.
- **A boost ratio is never gain (the canonical caution).** A Bloch *transmission zero* can
  show a very large pump-on/off boost *ratio* only because its pump-off output sits near a
  zero floor, while its absolute out/in is deep loss. A large ratio over a near-zero passive
  floor is an artifact, not amplification — judge designs on **absolute out/in**, never the
  on/off ratio, and treat only a harmonic-balance record certified as in
  [GAIN_DEFINITIONS.md](GAIN_DEFINITIONS.md#certification-record-gainrecord) as a gain claim.

### Optimal length and its mechanism

The shipped n3 proxy example uses **15 cells**. A length sweep over N = 6–30 (the
input/output match frozen at the N = 15 solution, so only the length varies) shows why:

- **N = 15 is the robust optimum.** It gives the best boost (+8.18 dB) and the best
  net-positive absolute out/in (+1.18 dB) of any length reached by a smooth pump walk-up.
  Larger absolute gains do appear at other lengths (N = 18 records +21.07 dB absolute
  out/in), but they are **fragile resonance spikes that collapse within one 3 dB pump
  step** — an artifact of the idealized proxy's high Q, not a robust operating point.
- **The oscillation onset is not a 1/length law.** Onset pump versus length scales at
  roughly **−5 dB/decade**, not the **−20 dB/decade** of a lossless 1/length critical-depth
  law, and is non-monotone across the sweep: the physical `VIAGND` damps the end-to-end
  standing wave, so the oscillation edge does not track a clean critical depth.
- **Mechanism (corrected).** The signal operates at a **slow-wave Bloch band edge** —
  per-cell attenuation ≈ 4–9× the
  [propagating threshold](theory.md#why-operating-points-sit-where-they-do) and per-cell
  phase ≈ 133–156°, far from the ~30° a simple half-wave resonance would require. The robust
  optimum length equals the degenerate [phase-matching](theory.md#phase-matching) coherence
  length N_coh = π / |θ(2f_s) − 2θ(f_s)|. Predicting N_coh from the dispersion *before*
  measuring, at two design frequencies, confirms it:

  | design frequency | predicted N_coh | measured robust optimum |
  |---|---|---|
  | 2.0 GHz (shipped) | 15.3 cells | **N = 15** |
  | 1.7 GHz (re-derived) | 18.5 cells | **N = 18** |

  So the robust optimum in cells tracks the design frequency through the coherence length,
  staying ≈ 15–18 cells as the band edge relocates.
- **A deeper edge buys more gain at the cost of fragility.** The 1.7 GHz re-derivation sits
  deeper at its band edge (attenuation ≈ 9× vs ≈ 4× the threshold). Its N = 18 optimum
  records a higher **+2.74 dB** absolute out/in (ADS-realized; +12.57 dB boost over a
  −9.83 dB passive baseline, at +12 dBm pump, pump phase 0°) with a **6 dB** stability
  margin to the oscillation onset — but it is only a *moderate*-robustness point (stepping
  the pump down 3 dB drops the boost ≈ 12 dB), in a regime where eight of the nine swept
  lengths are fragile spikes. A deeper band edge therefore raises the realized net-positive
  mark above +1.18 dB, but trades robustness to do it; the shipped 2.0 GHz / 15-cell design
  remains the most robust net-positive point.

The procedural guards these results imply for any future search-space extension are in
[OPTIMIZER_METHODOLOGY.md](methodology/OPTIMIZER_METHODOLOGY.md#search-space-extension-guards).

### Analytical-vs-ADS peak error is conditional on the signal path (2026-08-09)

The analytical parametric-gain leg has carried a blanket caveat that its coherent peak
`coh_pk` over-predicts the ADS harmonic-balance peak by roughly 20–30 dB. Evidence from
The R1 report, data deposit `runs/r1_redesign_under_law/REPORT.md`, narrows that: **the error tracks signal-path pathology
rather than being an unconditional model offset.**

Across the twelve designs certified in that run, the parity error
`ε_peak = |analytical coh_pk − ADS coh peak|` splits cleanly by whether the design
satisfies a signal-transport budget (per-cell attenuation at the signal frequency within
the same 0.02 Np/cell used for the pump):

| design class | ε_peak | absolute out/in |
|---|---|---|
| fails the signal-transport budget (11 of 12) | 26–149 dB | −2.3 to −30.6 dB |
| satisfies it (1 of 12) | **1.39 dB** | **+0.57 dB** |

The single design that meets the budget is also the only one in that run with a positive
absolute out/in and a pump-off reference well clear of the harmonic-balance numerical
floor. Where the line is low-loss, matched and propagating at the signal frequency, the
analytical cascade agrees with ADS to ~1.4 dB; the large errors are concentrated on
designs whose signal path is degenerate — lines in which the pump-off reference has
collapsed, or which the regime detector classifies as oscillators rather than amplifiers.

Two practical consequences:

- Read the 20–30 dB caveat as **conditional**. It is a statement about a class of
  candidates, not a correction to subtract from every analytical number.
- A large ε_peak is usable as an early **pathology detector**: it flags a degenerate
  candidate before any ADS run is spent on it.

This narrows the caveat; it does not retire it. One design satisfying the budget is a
single point, and the boundary between the two classes has not been mapped. Full evidence,
including the per-candidate table and the certification records behind it, is in
the R1 report §6, data deposit `runs/r1_redesign_under_law/REPORT.md`.

---

**Related:** interpreting failures and expected warnings →
[troubleshooting.md](troubleshooting.md); the physics these caveats bound →
[theory.md](theory.md).
