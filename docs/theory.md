# TVTL Theory — the physics the code implements

This page explains the physics of a varactor-loaded **time-varying transmission
line (TVTL)** and ties each idea to the routine that computes it, so you can read
the theory and the code side by side. It is deliberately operational: every
formula here corresponds to something the public `tvtl` core actually evaluates.
Full derivations belong to the accompanying manuscript (listed under "Used in" in the
[README](../README.md#used-in)); the precise gain algebra lives in
[`GAIN_DEFINITIONS.md`](GAIN_DEFINITIONS.md).

One rule runs through everything below and is worth stating first: **analytical
numbers are predictions, ADS harmonic-balance numbers are the record.** They are
never merged. [Section 5](#5-predicted-vs-realized-the-honesty-rule) explains why.

---

## 1. The device

### Unit-cell anatomy (as built)

A TVTL is a cascade of identical unit cells. One cell, exactly as the builder
emits it (`src/tvtl_ads_builder_v2/cells/single_shunt_varactor.py`,
`SingleShuntVaractorCell`), is:

- **Through-line** — two microstrip half-sections (`MLIN`, each `main_period/2`
  long) split by a tee junction (`MTEE`). The tee's third arm taps off the shunt
  branch. The through arms carry `main_width`; the branch arm carries
  `branch_width` (default `1.93 mm`).
- **Shunt branch**, from the tap downward:
  `package inductance L_pkg` → `series resistance R_series` →
  **`SDD2P` varactor** (the nonlinear capacitor) → a current probe (`I_Probe`,
  for per-branch current readout) → one or more microstrip **stub** sections
  (`MLIN`) → a **termination**.
- **Termination** — one of three, and the choice is electrically important:
  - **OPEN** — an open-ended stub. Realized with the `MLEF` open-end microstrip
    model (its fringing capacitance is part of the resonance).
  - **SHORT** — a via to ground, realized with the physical `VIAGND` element (a
    one-terminal model that grounds internally). A lumped inductor-to-ground
    (`LVIA`) is the deprecated stand-in.
  - **COMBINED** — an open stublet (`MLEF`) **in parallel with** a via
    (`VIAGND`) at the same node: a parallel **resonant trap**.

The cell interface (`cells/cell_spec.py`, `CellSpec`) exposes the two varactor
terminals so the junction voltage `V(top) − V(bottom)` — the quantity the pump
modulates — is always well defined.

### N-cell cascade

The assembler chains cell *k*'s output to cell *k+1* through an inter-cell
current probe, and owns the input port, the output port and its termination, the
input/output matching, the sources, and the harmonic-balance controller. The
**number of cells is simply the number of `SDD2P` varactors** in the deck
(`analytical/realistic_c_gain.py`). Analytically, the cascade is the single-cell
harmonic-ABCD matrix raised to the *N*-th power
(`src/tvtl/floquet/harmonic_abcd.py`, `n_cells_harmonic_abcd`).

### Matching networks

The cell's Bloch impedance rarely equals 50 Ω, so edge cells are matched. The
optimizer synthesizes single-stub input/output matching sections when the edge
impedance needs them. In the shipped examples the three real-part designs need
no matching (their edge impedances are already near the port), while the three
proxy designs are the first that **do** require single-stub matching — which is
exactly why they are included (see [examples.md](examples.md)). A representative
main-line impedance is `Z_TL ≈ 61.3 Ω`.

---

## 2. Operating principle

### Capacitance modulation

The varactor is a voltage-controlled capacitor. The reverse-bias C–V law the
code uses (`src/tvtl/design/realistic_varactor.py`, `RealisticVaractor.C`) is the
Skyworks SPICE form, **hyperabrupt by default**:

```text
C(v) = CJ0 / (1 + v/VJ)^M          # default exponent M = 1.7
```

A pump tone swings the junction voltage, so the capacitance becomes
time-periodic: `C(t) = C0 + C1·cos(ω_p t + φ) + …`. What the ADS `SDD2P` element
actually integrates is the **charge** `Q(v)` — the Taylor expansion of `C(v)`
about the bias point (`src/tvtl/design/nonlinear_capacitor.py`) — emitted as
`I[1,1] = Q(_v1)` so that the port current is `dQ/dt`
(`netlist/elements/sdd2p.py`). The static package capacitance `CP` is folded into
the `C0` term exactly, not approximated.

The optimizer's default device is the **low-parasitic proxy "Varactor A"**:

```text
power_law(CJ0 = 0.3 pF, VJ = 3.5, M = 1.7, CP = 0, L_pkg = 0.3 nH, R = 0.5 Ω)
```

A real catalog part such as SMV1245-079LF is far more parasitic
(`CJ0 = 6.9 pF, L_pkg = 1.7 nH, R = 1.5 Ω, CP = 0.47 pF`) — the gap between those
two is most of [Section 6](#6-pump-delivery-why-catalog-parts-are-delivery-limited).

### Degenerate parametric amplification

These designs are **self-pumped and degenerate**: a single pump at

```text
f_pump = 2·f_signal
```

is injected on the same RF line as the signal (no separate pump lane — see the
[ADS drawing guide](ADS_OA_DRAWING_GUIDE.md)). A small signal at `f_s` beats
against the `2·f_s` modulation to create an idler at `f_pump − f_s = f_s`. Because
the idler lands **on the signal frequency**, pump energy can be pumped directly
into the signal band. In harmonic-mix bookkeeping a product of signal index `m`
and pump index `n` lands on `f_s` whenever

```text
m + 2n = 1
```

and the truncated family the gain equations sum is
`FOLDING_FAMILY = (1,0), (-1,1), (3,-1), (-3,2)`
(`src/tvtl_ads_builder_v2/oa/gain_equations.py`).

### Phase-sensitive gain (the lobe)

Because signal and idler coincide in frequency, the result of their interference
depends on the **pump phase** `φ`: one quadrature is amplified, the orthogonal one
is deamplified. This is the defining signature of a degenerate parametric
amplifier. The analytical gain is built from the signal transfer `H_s` (mix
`(1,0)`), the idler transfer `H_i` (mix `(-1,1)`), and the pump-off reference
`H_s,off` (`src/tvtl/floquet/harmonic_abcd.py`, `device_coherent_gain`):

```text
coherent_peak_dB = 20·log10( (|H_s| + |H_i|) / |H_s,off| )   # best quadrature
coherent_min_dB  = 20·log10( ||H_s| − |H_i|| / |H_s,off| )   # deamplified quadrature
coherent_swing   = peak − min                                 # the PSA lobe depth
```

A healthy degenerate PSA traces a **single bounded lobe** as `φ` goes 0→360°. The
shipped MAVR amplifier shows exactly this in ADS harmonic balance (realized,
committed `examples/real_parts/results/headline_mavr.json`): the full-family
out/in gain `G_mod_in_dB(φ)` peaks at **+14.35 dB @ φ = 210°**, passes **+3.55 dB**
at φ = 0°, and dips to **−14.05 dB @ φ = 30°** — one maximum, one minimum, every
phase converged. That bounded single lobe is the amplifier's certificate, not a
runaway.

### Phase matching

Gain only accumulates down the cascade if signal, idler, and pump stay in phase
as they propagate; otherwise contributions from successive cells cancel. Whether
that holds is a property of the line's dispersion, which is the next section.

---

## 3. Dispersion (Bloch–Floquet)

### The passive backbone

For the un-modulated cell, the propagation per period follows from the cell's
ABCD matrix (`src/tvtl/theory/dispersion.py`, `bloch_dispersion`):

```text
cosh(γ·period) = (A + D) / 2,    γ = α + jβ
```

`β` is the Bloch wavenumber and `α` the per-cell attenuation. Where
`|(A+D)/2| ≤ 1` the cell **passes** (a passband, `α ≈ 0`); outside that range it
is evanescent (a **stopband**). Passbands and stopbands are what make some
operating frequencies usable and others dead.

### The modulated line

With the pump on, the cell couples a ladder of harmonics, and the analysis moves
to the **Floquet harmonic basis**. The time-varying elastance
`S(t) = S0 + dS·cos(ω_m t + φ)` builds a tridiagonal system
(`src/tvtl/floquet/harmonics.py`, `build_stub_tridiagonal`): the **diagonal** is
`S0 + jR·ω_n − L·ω_n²` at each harmonic `ω_n = ω_s + n·ω_m`, and the
**off-diagonal** `dS·e^{±jφ}/2` couples each harmonic to its neighbors. **That
off-diagonal coupling is the parametric interaction** — set `dS = 0` and the
harmonics decouple back to the passive line.

Cascading the per-cell harmonic-ABCD block and taking its eigenvalues `λ_k`
(`src/tvtl/floquet/stability.py`) gives per-cell Bloch multipliers; the cascade
amplifies harmonic mode `k` like `|λ_k|^N`. This is the bridge from one cell's
physics to the N-cell device response.

### Two equivalent modulated-element formulations (C-linear and S-linear)

The cascade can build the modulated shunt element in **either** basis, selected by
`modulation_formulation` on `level10_realistic_param_gain` /
`realistic_pumped_unit_cell_abcd`:

- **`"capacitance"`** (default) — the capacitance-linear device `Q = C(t)·v` with
  `C(t) = C0·(1 + m·cos)`; depth `m_branch_pump = dC/C0`. This is the historical
  path (byte-identical) and the one the ADS-realized cross-check rests on.
- **`"elastance"`** — the elastance-linear device `S(t)·Q = v` with
  `S(t) = S0·(1 + m_s·cos)`, `S0 = 1/C0`, and `R = L = 0` inside the element
  (`build_stub_tridiagonal`); depth `m_s_branch = dS/S0`. The package wrap and the
  stub stacking are shared with the capacitance path.
- **`"reciprocal"`** — the *same* capacitance device solved through its exact
  elastance `S(t) = 1/C(t)`; a solver cross-check (not new physics) that agrees
  with `"capacitance"` to numerical precision, bounded only by the finite harmonic
  truncation.

Exactly one depth kwarg is accepted per call. The two physical formulations **agree
to first order in the modulation depth** — the `O(m)` idler terms are identical —
and **diverge at deep modulation**, because `C0·(1 + m·cos)` and the elastance
`S0·(1 + m_s·cos)` are reciprocals only to first order. `SM Fig S1`
(`examples/real_parts/plot_c_vs_s_formulations.py`) plots both gains and the
reciprocal residual versus depth.

### Why operating points sit where they do

An operating frequency is only useful if the signal **and** its pump **and** its
idler all fall in passbands. The screen
`src/tvtl/floquet/passband.py` (`passband_check_at_operating_point`) checks the
static cascade `|S21|` at each of those frequencies against a `−10 dB` floor, and
the peak classifier (`src/tvtl/floquet/sweeps.py`, `classify_design_spectrum`)
flags any "gain" peak that actually sits inside a stopband as **artifactual**.
That is the discipline behind the chosen operating points in the examples — they
are passband-clean, not stopband artifacts.

---

## 4. The two regimes

A TVTL lives in one of two regimes, and telling them apart is the whole game.

### Stable amplification vs self-oscillation

From the eigenvalues above (`src/tvtl/floquet/stability.py`):

```text
stability_margin = 1 − max_k |λ_k|
```

`margin > 0` (all `|λ_k| < 1`) is a **stable, bounded amplifier**; `margin < 0`
(some `|λ_k| > 1`) is a **self-oscillator**; `|λ| = 1` is the
parametric-instability **threshold**.

### Regenerative buildup and the oscillation edge

Approaching the threshold, the device behaves like a regenerative amplifier: with
a round-trip loop factor `L → 1`,

```text
G ≈ G0 / (1 − L)
```

so gain grows without bound as you near the edge. The useful operating point is
deliberately **one step below collapse** — high gain, still convergent.

### What HB non-convergence means physically

When the driven harmonic-balance solve **stops converging**, it is not a numerical
nuisance — it is the simulator reporting that the output diverges with the signal
on, i.e. the device has crossed the oscillation edge
(`src/tvtl_ads_builder_v2/analysis/amplification_check.py`). The toolchain records
non-convergence as the data point it is (`non-convergent` = instability onset),
never silently dropping it. See also
[troubleshooting.md](troubleshooting.md#hb-non-convergence-is-the-oscillation-edge).

### The loss-invariant regime detector

Because a near-threshold amplifier and a true oscillator can show similar raw
gain, the classifier
(`src/tvtl_ads_builder_v2/analysis/regime_detector.py`,
`classify_regime_loss_invariant`) uses two loss-invariant tests:

- **Input-independence** `r_A = |out(signal off)| / |out(signal on)|`, pump on in
  both. `r_A → 1` means the output persists with no input → **lasing**;
  `r_A → 0` means it follows the input → **stable**. Threshold `0.5`.
- **Gain linearity** `slope = d·log|out| / d·log|in|`. `slope → 1` is a
  well-behaved amplifier; `slope → 0` means the output is pinned regardless of
  drive → laser. Threshold `0.5`.

The categories are `passive | amplifier | lasing | non-convergent`.

The shipped **SMV2020 oscillator** is the textbook lasing case (realized,
committed `headline_smv2020.json`): with the input driven off, a **30.9 mV** limit
cycle persists (`r_A = 0.87`, slope `≈ 0.08`), regime `lasing`. That output is an
**oscillation signature, not a transducer gain** — and the detector says so.

---

## 5. Predicted vs realized (the honesty rule)

This is the repository's central convention.

- **Predicted** = the analytical Floquet coherent gain
  (`device_coherent_gain`, `level10_realistic_param_gain`). It is a fast
  **ranking** indicator and is **systematically optimistic** — it omits loss,
  mismatch, and large-signal saturation. The code marks it as such mechanically:
  every analytical gain dict carries `label = "analytical_proxy_only"`, and the
  sweep CLI prints `ANALYTICAL PROXY ONLY -- not a validated ADS-HB or
  device-gain claim`. Use it to order candidates, never as a device spec.
- **Realized** = the ADS harmonic-balance result. This is **the record**. The
  baked-in measurement block (`src/tvtl_ads_builder_v2/oa/gain_equations.py`,
  `meas_gain_equations`) evaluates on the HB dataset:

  ```text
  Pout_mod    = 0.5·|Σ_family V_out|² / 50          # full m+2n=1 family on f_s
  G_mod_in_dB = 10·log10( Pout_mod / P_avs )         # P_avs = dbmtow(-30) = 1e-6 W
  ```

The two are kept in **separate fields** in every report; a number that has not
been computed is written `null` with a note, never back-filled from the
prediction (see [GAIN_DEFINITIONS.md](GAIN_DEFINITIONS.md) and
[LIMITATIONS.md](LIMITATIONS.md)). When you want to know *why* they differ at a
specific operating point, the cross-check
`src/tvtl_ads_builder_v2/analysis/floquet_comparison.py` recomputes the analytical
harmonic currents **given the pump actually delivered in HB**, so a disagreement
is a model-vs-simulator gap rather than a guess about pump depth.

The proxy designs make the gap concrete (predicted vs realized, committed
`proxy_analysis_summary.json`):

| design | predicted (analytical) | realized boost (ADS HB) | realized abs. out/in |
|---|---|---|---|
| `proxy_n2_double_short` | +10.63 dB | +2.86 dB | −3.92 dB |
| `proxy_n3_single_short` | +10.11 dB | +8.18 dB | **+1.18 dB** (net-positive) |
| `proxy_n4_double_short` | +10.40 dB | +2.51 dB | −4.51 dB |

The prediction ranks them in a tight band; harmonic balance separates them. The
net-positive `n3_single_short` is the idealized-proxy record — a faithfulness
result, not a real-world device-gain claim. How high this proxy ceiling reaches,
and why the example is 15 cells long, are characterized in
[LIMITATIONS.md](LIMITATIONS.md#characterized-performance-limits).

---

## 6. Pump delivery (why catalog parts are delivery-limited)

The junction only modulates as hard as the pump that actually reaches it across
the package. The transfer from the branch top to the junction is a simple divider
(`src/tvtl/design/realistic_varactor.py`):

```text
pump_divider(f) = Z_C / (Z_C + jω·L_pkg + R_series)
m_effective     = m_branch · |pump_divider(f_pump)|
```

The series branch `L_pkg`–`C0`–`R` **self-resonates** (its SRF). Below SRF the
branch is capacitive and the junction sees most of the pump; **above SRF it turns
inductive and starves the junction** — at high pump frequency with a real package
inductance the junction can see only a few percent of the branch-top pump. That is
why a catalog varactor needs its **bias and frequency co-selected**: bias moves
`C0` (hence SRF), frequency sets where on the divider you sit.

The single-cell delivery figure of merit captures this
(`src/tvtl/optimization/figure_of_merit.py`, `a1_factors` / `ClosedFormA1`):

```text
A1 = m_eff · kappa_branch · rho_para
kappa_branch = |Z_ref| / |Z_ref + Z_branch(f)|        # pump current-divider coupling, Z_ref = 50 Ω
Z_branch     = (R_series + jω·L_pkg) + 1/(jω·C0) + Z_stub(f)
```

`m_eff` is the modulation depth at a standardized drive, `kappa_branch` is the
pump coupling, and `rho_para` is a resonance-detuning guard that rewards sitting
*near but not on* the branch resonance (which would tip into the non-convergent
oscillation band). The commercial-part screen lives in the device palette
(`src/tvtl/varactor_palette.py`), which carries datasheet SPICE parameters and the
SRF/bias reasoning per part (e.g. a large `CJ0` lowers SRF; a larger `L_pkg`
lowers it further).

The **fabricated SMV1245 board at 0 V, 3.1 GHz** is the cautionary realization
(committed `headline_smv1245.json`): the pump-on/off boost is just **+0.001 dB**
and the net out/in is **−6.46 dB** — exactly the passive insertion loss. The
modulation mechanism works, but at 0 V the pump is **delivery-starved across the
package**, so the large measured lab lift does not reproduce. Delivery, not the
parametric mechanism, is the limiter for catalog parts.

---

## 7. Stability methodology (how a stable operating point is certified)

Putting Sections 4–6 together, a design is promoted to "stable amplifier" only
after two sweeps, both run in ADS harmonic balance (template:
`examples/real_parts/analyze_proxy.py`):

1. **Pump-power scan.** Sweep pump power and find the **boost-optimal point that
   still converges** — the largest gain in the convergent window, one step below
   the non-convergence (oscillation) onset. On the analytical side the equivalent
   choice is the maximum proxy gain among points with `stability_margin ≥ 0.10`.
2. **Pump-phase sweep → bounded-lobe check.** At that operating power, sweep the
   pump phase. A stable amplifier shows a **single bounded PSA lobe** — finite
   `swing = max − min`, every phase converged
   (`src/tvtl_ads_builder_v2/oa/psa_gain_artifact.py`) — whereas an oscillator
   runs away or refuses to converge.

The certified-amplifier signature is therefore: converges at every phase, bounded
single lobe, net enhancement. The MAVR design meets all three (realized, committed
`headline_mavr.json`: +14.35 dB peak, single lobe, all phases converged). The
SMV2020 design fails the input-independence test and is reported as an
oscillator, not an amplifier.

---

## Where to go next

- The exact gain algebra and normalization conventions → [GAIN_DEFINITIONS.md](GAIN_DEFINITIONS.md)
- How to run any of this → [workflow.md](workflow.md)
- These numbers in their full design context → [examples.md](examples.md)
- The optimizer's ranking method → [methodology/OPTIMIZER_METHODOLOGY.md](methodology/OPTIMIZER_METHODOLOGY.md)
  and the staged design flow → [PARAMETRIC_AMPLIFIER_DESIGN_FLOW.md](PARAMETRIC_AMPLIFIER_DESIGN_FLOW.md)
- What the public supplement does and does not claim → [LIMITATIONS.md](LIMITATIONS.md)
- Interpreting failures (HB non-convergence, the SMV2020 floor) → [troubleshooting.md](troubleshooting.md)
