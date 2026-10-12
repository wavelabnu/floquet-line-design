# Optimizer Methodology

This note summarizes the public optimizer and design-search methodology used by
the TVTL supplement. It is intended as a methods companion to the code, not as a
record of private development history.

## Scope

The public code supports analytically guided design of varactor-loaded
time-varying transmission lines. The optimizer ranks candidate unit-cell
geometries with inexpensive physics proxies, promotes a small set of candidates
through deterministic matching and Floquet analyses, and leaves ADS harmonic
balance as a separate validation step.

The central public entry points are:

- `tvtl.optimization`: generic topology, figure-of-merit, and staged workflow
  helpers.
- `tvtl_ads_builder_v2.parity.self_pumped_design`: self-pumped paper-scale
  design generation.
- `tvtl_ads_builder_v2.parity.expanded_search`: optional searches over cell
  count and pump settings with a stability/regime gate.
- `tvtl_ads_builder_v2.analytical`: harmonic-ABCD and supporting analytical
  gain calculations.

## Design Pipeline

The default flow is an analytical screen followed by optional ADS validation:

1. Define a varactor, substrate, target signal/pump frequencies, and an allowed
   set of unit-cell topology families.
2. Generate a quasi-random pool of stub geometries within public fabrication and
   microstrip constraints.
3. Score each candidate with a closed-form single-cell figure of merit that
   combines modulation depth, branch coupling, and resonance detuning.
4. Walk the top-ranked candidates through a static unit-cell ABCD model and a
   Bloch impedance target to obtain a realizable main-line width.
5. Synthesize distributed input/output matching sections.
6. Evaluate the matched cascade with a harmonic-ABCD Floquet model.
7. Select either by coherent peak gain (`coh_pk`) or by the net out/in
   transducer-gain proxy (`g_mod_in`).
8. Export the selected design as a v2 device specification and optional ADS
   netlist/workspace artifacts.

The Python loop is deliberately ADS-free. ADS/OpenAccess/HB validation can be
run later when a local Keysight ADS installation and Python bridge are
available.

## Objectives

Two public objectives are supported:

- `coh_pk`: selects a candidate whose coherent pump-on/pump-off gain lies in a
  target window after matching constraints are met.
- `g_mod_in`: maximizes an analytical out/in proxy, computed as passive
  matched-cascade gain plus coherent parametric boost.

`coh_pk` is useful for phase-sensitive-amplifier behavior because it isolates
the pump-induced boost. `g_mod_in` is more conservative for design ranking
because it penalizes passive stopband loss instead of letting a small pump-off
denominator inflate the ratio.

## Search Variables

The base screen samples:

- topology family;
- per-segment microstrip width;
- per-segment microstrip length.

The deterministic stages solve or synthesize:

- main-line impedance/width;
- per-side matching sections;
- cascade response over the harmonic basis.

The expanded search can also sweep cell count and pump setting. When ADS
measurements are available, the expanded flow can reject candidates classified
as oscillatory, non-convergent, or otherwise outside the amplifier regime.

## Analytical Models

The optimizer uses standard network models that remain runnable in ordinary
Python:

- 2-port ABCD cascades for static and periodic unit-cell analysis;
- Bloch dispersion and impedance estimates for repeated cells;
- microstrip and discontinuity approximations for geometry-dependent elements;
- harmonic-ABCD Floquet blocks for time-varying capacitors;
- varactor charge/capacitance models derived from public device parameters.

The public Floquet utilities include both elastance-style harmonic helpers and
capacitance-linear harmonic-admittance helpers. The realistic-varactor design
path uses the capacitance-linear formulation because the ADS SDD representation
is a charge-polynomial device.

## ADS Validation

ADS is not required to import the package or run the base test suite. ADS is
needed only for workflows that emit or execute Keysight-specific artifacts:

- OpenAccess workspace emission and GUI inspection;
- `hpeesofsim` execution;
- harmonic-balance datasets;
- ADS-simulated gain, regime, and convergence checks.

Use local placeholders such as `<ADS_INSTALL_DIR>` and `<OUTPUT_DIR>` in scripts
and reports. Do not rely on host-specific development paths.

## Limitations

The analytical gain values are ranking proxies, not publication-grade device
gain measurements by themselves. ADS/HB or an equivalent circuit simulator
should be used to validate final gain, stability, and convergence claims.

Some optional calibration and evidence artifacts, such as MTEE surrogate tables
or recorded ADS AEL macro workspaces, are not bundled in the public supplement.
The corresponding modules accept environment-variable overrides and fail
gracefully when those optional artifacts are absent.

## Search-space extension guards

The default optimizer searches the 12 linear (OPEN/SHORT) topology families and screens
the 8 COMBINED (`MLEF ∥ VIAGND`) families analytically only. Extending the search —
admitting the COMBINED families as first-class candidates, for instance — requires three
guards (see also
[LIMITATIONS.md](../LIMITATIONS.md#gain-ceiling-at-2-ghz-idealized-proxy)):

1. **Screen realizability and passband before ranking.** Ranked purely by the single-cell
   pump-delivery proxy, the top-scoring COMBINED cells can be unrealizable Bloch deep-zeros
   (high per-cell pump attenuation, main-line impedance pinned at the matching ceiling).
   Apply the realizability and passband screens up front, or the ranking fills with
   transmission zeros.
2. **Characterize the full pump × phase surface, not one phase.** A single pump-phase slice
   can misjudge a design badly. Sweep pump power *and* pump phase before classifying a
   candidate.
3. **Judge on absolute out/in, never the on/off ratio.** A Bloch-zero cell can post a very
   large pump-on/off boost *ratio* over a near-zero pump-off floor while its absolute out/in
   is deep loss. Prefer the conservative out/in objective
   (`g_mod_in`), which penalizes passive loss, over the boost ratio (`coh_pk`), which a small
   pump-off denominator can inflate.

### Length selection — N ≈ N_coh(f_design)

A related characterization shows the robust optimum cell count equals the degenerate
phase-matching coherence length, N_coh = π / |θ(2f_s) − 2θ(f_s)|, and that it tracks the
design frequency (predicted 15.3 → measured 15 cells at 2.0 GHz; predicted 18.5 → 18 cells at
1.7 GHz from an analytical re-derivation only, not confirmed by sweep since no
1.7 GHz length sweep exists —
[LIMITATIONS.md](../LIMITATIONS.md#optimal-length-and-its-mechanism)). This rule is
**empirically validated but not implemented**: the shipped cell count stays fixed at 15,
because the rigorous evaluation point of the phase walk-off at a slow-wave band edge — where
the propagating-Bloch-wave coupled-mode picture is singular — is still open theory. Treat
N ≈ N_coh(f_design) as a heuristic to confirm by sweep, not an automated selection step.

## Reproduction

Base Python reproduction is documented in `docs/REPRODUCE.md`. ADS-dependent
reproduction requires a local ADS installation and should write generated
workspaces, datasets, and reports under an explicit output directory.

---

**Related:** the physics the proxies approximate → [../theory.md](../theory.md);
running the optimizer CLI → [../workflow.md](../workflow.md#3-the-optimizer-in-full-optimize-amplifier).
