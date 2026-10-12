# Gain Definitions

This repository distinguishes analytical gain estimates, direct
`hpeesofsim` metrics, and ADS OpenAccess generated-netlist metrics. Missing
numerical quantities are reported as `null` with notes; they are not fabricated.

## Notation

- `fs`: signal frequency.
- `fp = 2 fs`: pump frequency.
- `k`: harmonic or HB mix index.
- `V_in(fs)`: complex input/RF_MAIN phasor at the signal frequency.
- `V_out(fs)`: complex output/load phasor at the signal frequency.
- `P_L(fs)`: power delivered to the load at the signal frequency.
- `P_in(fs)`: power entering the DUT at the RF input plane.
- `P_avs(fs)`: available signal-source power at `fs`.
- `P_pump,av`: available pump-source power at `2fs`.
- `K_out`: output harmonics included in full-sum gain.

ADS `P_1Tone.P` is treated as available source power in transducer-gain
reports only when a matched passive reference verifies that convention for the
local ADS setup. Until then, reports keep unavailable transducer quantities as
`null` with a note.

## Voltage Gain At Signal Frequency

```text
G_V(fs) = |V_out(fs)| / |V_in(fs)|
G_V,dB(fs) = 20 log10(G_V(fs))
```

This is a phasor-amplitude ratio at the signal-frequency HB component.
RMS/peak convention cancels if both voltages use the same convention.

## Load Power Gain At Signal Frequency

```text
G_P(fs) = P_L(fs) / P_in(fs)
G_P,dB(fs) = 10 log10(G_P(fs))
```

Here `P_in(fs)` means power entering the DUT at the RF input plane. If a
dataset does not expose enough voltage/current information to compute that
input-plane power, `G_P` is reported as `null`.

## Transducer Gain

```text
G_T(fs) = P_L(fs) / P_avs(fs)
G_T,dB(fs) = 10 log10(G_T(fs))
```

This is relative to available signal-source power, not pump power. ADS
`P_1Tone.P` is treated as available source power only where a matched passive
reference verifies that convention for the local comparison mode.

## S-Like / Insertion Gain

For equal reference impedances or a linearized S-like comparison:

```text
G_S21(fs) = |S21(fs)|^2
G_S21,dB(fs) = 20 log10(|S21(fs)|)
```

Large-signal HB power gain is not equated with small-signal `S21` unless the
comparison mode explicitly supports that interpretation.

## Coherent Signal Gain

Coherent gain means output only at the original signal frequency:

```text
G_coh,V = |V_out(fs)| / |V_ref(fs)|
G_coh,V,dB = 20 log10(G_coh,V)
```

or, when normalized to input power:

```text
G_coh,P = P_out(fs) / P_in(fs)
G_coh,P,dB = 10 log10(G_coh,P)
```

The JSON report states which normalization is used. Optimizer
`coherent_peak` values are carried as preliminary analytical coherent-gain
metrics for amplifier candidates.

## Full-Sum / Total Harmonic Output Gain

```text
G_sum = (sum_{k in K_out} P_out(k fs)) / P_in(fs)
G_sum,dB = 10 log10(G_sum)
```

This is not the same as coherent signal gain. It includes converted, idler, or
harmonic output power. `K_out` is written into JSON reports as
`harmonics_in_sum`.

## Pump-Normalized Conversion Metrics

```text
eta_sig = P_out(fs) / P_pump,av
eta_sum = (sum_{k in K_out} P_out(k fs)) / P_pump,av
```

These are pump-normalized conversion or efficiency metrics, not ordinary gain.

## Emitted `MEAS_GAIN` Equation Names

The OA-workspace `MeasEqn` block (`src/tvtl_ads_builder_v2/oa/gain_equations.py`)
bakes these named equations onto each assembled schematic; they evaluate on the
realized ADS harmonic-balance dataset. Each maps to the abstract definitions
above. All powers use `P = 0.5*|V|^2/50`, `P_avs` is the available signal-source
power at `fs`, and the degenerate folding family on `fs` is
`(1,0),(-1,1),(3,-1),(-3,2)`.

| Emitted name | Definition | Maps to |
|---|---|---|
| `Gain_sig_dB` | `10 log10(P_out(1,0) / P_avs)` | Transducer Gain restricted to the signal term `(1,0)` |
| `Gain_coherent_dB` | `10 log10(P_out{(1,0)+(-1,1)} / P_avs)` | Coherent Signal Gain (2-term signal+idler, normalized to `P_avs`) |
| `G_mod_in_dB` | `10 log10(P_out{full folding family} / P_avs)` | absolute out/in over the full on-`fs` family (the SA-comparable headline) |
| `Gain_ref_dB` | `10 log10(P_out{(1,0)+(-1,1)} / P_avs)` | the reference/comparison convention; algebraically equals `Gain_coherent_dB` |
| `Gain_4f_dB` | `10 log10(P_out,4f / P_in,4f)` | out/in over the 4-frequency total `(1,0)+(0,1)+(-1,1)+(1,-1)` — includes the pump term `(0,1)` at `2fs`; used to reconcile gain-definition conventions |

The pump-on/off headline `G_mod_unmod` is a two-run quantity: `G_mod_in_dB` (or
`Gain_sig_dB`) at the baked optimal pump phase minus its pump-off value (see the
per-workspace README).

## Stage-0 feasibility gate

`tvtl.design.feasibility.feasibility` evaluates one cell at `f_s` and `f_p = 2 f_s`.
Its outcome is tier `ANALYTICAL`, a passive screen, not a gain claim.

| Gate | Quantity | Pass rule | Refusal reason |
|---|---|---|---|
| A, delivery | `T(2 f_s) = abs(Z_C / (Z_C + Z_series))`, the junction share of the branch-top pump voltage | `T(2 f_s) >= T_req` | `pump_delivery_below_threshold` |
| B, pump transport | `alpha(2 f_s) = Re(gamma p)`, Np/cell | `alpha(2 f_s) <= alpha_max` | `pump_attenuation_above_threshold` |
| D, signal transport | `alpha(f_s)`, Np/cell | `alpha(f_s) <= alpha_max` | `signal_transport_above_budget` |

`gamma p = arccosh((A + D)/2)` is the per-cell Bloch exponent of the cell's ABCD matrix. The gate also keeps
the passband flags `in_band_p` / `in_band_s` (`abs(Re (A + D)/2) <= 1`) and the reason `signal_stopband`.
A lossy cell can keep that real part inside `[-1, 1]` while its signal decays by several Np per cell, and
only gate D refuses it.

Each gate reports its margin; a negative margin fails the gate:

- delivery: `margin_delivery_dB = 20 log10(T(2 f_s) / T_req)`;
- pump transport: `margin_pump_transport_dB = 8.686 N (alpha_max - alpha(2 f_s))`;
- signal transport: `margin_signal_transport_dB = 8.686 N (alpha_max - alpha(f_s))`.

The transport margins are also given per cell in Np. Without `N`, the dB values are `null`.

### Delivered-depth ceiling before harmonic balance

The pump may swing the junction only between 0 V (no forward-biased interval) and the part's maximum reverse
voltage `V_max`. At DC reverse bias `V_bias` the largest allowed junction pump amplitude is
`A_max = min(V_bias, V_max - V_bias)`. In the tool's linear depth convention `m = (M/V_j) abs(V_j)`, the C(V)
ceiling is `m_ceiling = (M/V_j) A_max`.

A candidate is refused with the reason `delivered_depth_above_ceiling` when its largest junction pump
amplitude from the linear pump solve exceeds `A_max`. The check runs in three places
(`tvtl.design.depth_ceiling`):

- in `feasibility(..., depth_ceiling={'bias_V': ..., 'v_max_V': ...})`;
- in the optimiser's `cascade` gate, by default at the varactor's bias;
- in the workflow, before any HB job (`run_ads=True`), at the largest requested pump power.

For a palette part, `V_max` defaults to the datasheet-backed reverse ceiling: the largest voltage of its
C(V) table, else its polynomial validity limit, else 12 V.

### Thresholds

| Constant | Value | Definition |
|---|---|---|
| `T_REQ` (default) | 0.17378 | `m*_D / (k V_bt^max)`, cascade-anchored; see below |
| `T_REQ_SCREEN_2026_09` | 0.1438 | earlier screening threshold, reverse-engineered from the fabricated board (measured +16 dB at +29 dBm); the 2026-09 screens (R1 leg 1, R6 maps, Fig. 2b,c) used it |
| `ALPHA_MAX` | 0.02 Np/cell | per-cell loss budget for pump and signal transport |

The adopted threshold uses these inputs:

- `m*_D = 0.235625` is the smallest uniform junction depth at which the `N = 20` cascade of the as-built
  SMV1245 cell gives a 1.0 dB on/off boost at `f_s`.
  - The cell is at 0 V with stub scale 1.
  - The cascade assumes ideal phase matching, `phi_k = 2 k theta(f_s)`, and takes the best of 24 global pump
    phases.
  - The minimum is taken over `f_s = 2.0-6.0 GHz` on a 0.1 GHz grid, restricted to frequencies whose signal
    passes the signal-transport budget `alpha(f_s) <= alpha_max`. It is reached at the first such frequency,
    3.3 GHz.
- `k = 0.4547 /V` is `|C1|/C0` per volt of the SMV1245 junction at 0 V, `(M / V_J)(C_J0 / C_0)`.
- `V_bt^max = 2.98197 V` is the largest branch-top pump amplitude over the 20 cells (it occurs at cell 4).
  It comes from the linear pump solve at +19.5 dBm incident from a 50-ohm source at the board plane, at
  `f_p = 6.2 GHz`.

`T_req = 0.235625 / (0.4547 x 2.98197) = 0.17378`, which is 1.645 dB above 0.1438.

## Cascade layer: phase matching and the exceptional point

`tvtl_ads_builder_v2.analytical.cascade_budget` computes the per-cell budget of a uniform line of `N` cells.
All rates are per cell, and every quantity is tier `ANALYTICAL`.

| Quantity | Definition |
|---|---|
| `alpha`, `theta` | `Re` and `Im` of `gamma p = arccosh((A + D)/2)`, numpy principal branch |
| `Delta_theta` | `theta(2 f_s) - 2 theta(f_s)`, wrapped to `(-pi, pi]`; `N_coh = pi / abs(Delta_theta)` |
| `kappa/m` | small-depth coupling per unit junction depth (see below) |
| `g` | `Re sqrt(kappa^2 - (Delta_theta/2)^2) - alpha(f_s)` with `kappa = (kappa/m) m` |
| `m_EP` | `abs(Delta_theta) / (2 kappa/m)`, the exceptional point: the depth at which coupling first overcomes the phase mismatch |
| `m_del` | largest delivered junction depth `(M/V_j) abs(V_j)` over the cells, from the linear pump solve at the board-plane pump power |
| `phase_margin` | `0.5 m_del - m_EP` |

`kappa/m` is a fit to the boost of the harmonic cascade (basis (1, 3)):

- The cascade has a uniform prescribed depth `m`.
- The pump phases are ideal, `phi_k = 2 k theta_fwd(f_s)`, where `theta_fwd` is the forward-eigenwave phase.
- The boost is the best of 24 global pump phases.
- The fit is through the origin, `boost_dB = 8.686 N (kappa/m) m`, on `m = 1e-3, 2e-3, 5e-3, 1e-2`.

`m_star` is the smallest uniform depth with a 1.0 dB boost, bisected on a 0-0.60 ladder; it supplies `m*_D`
in `T_req`.

The measured-dispersion variant takes a table of `beta p / pi` against frequency and uses the model
`kappa/m`.

As-built SMV1245 cell at 3.1 GHz (`N = 20`):

| Quantity | Value |
|---|---|
| `Delta_theta` | 55.55 deg |
| `kappa/m` | 0.0541 |
| `m_EP`, model dispersion | 8.96 |
| `m_EP`, measured dispersion (`Delta_theta` 24.75 deg) | 3.99 |

**Phase-margin gate.** `feasibility(..., phase_margin=True)` adds the condition `phase_margin >= 0`. A
failing cell gets the reason `phase_margin_negative`.

- The gate is off by default in the stage-0 gate.
- It is on in the optimiser's cascade gate (`--gate cascade`, `gate="cascade"`).
- Its coupling fit samples `3 f_s`. Above the cell's tee-model edge the gate refuses with
  `tee_model_out_of_range`.

## Certification record (`GainRecord`)

`tvtl_ads_builder_v2.oa.certification.GainRecord` holds one candidate's ADS harmonic-balance evidence
(tier `ADS-REALIZED`).

| Field | Recorded quantity | Pass rule |
|---|---|---|
| `order_convergence` | best-phase absolute out/in at HB orders 5, 7, 9 | consecutive orders within 0.5 dB |
| `device_tier_agreement` | absolute out/in, SDD (charge polynomial) and full diode model, same phase | within 1 dB |
| `signal_off_pump_on` | output at `f_s` with the signal on and off, pump on | signal on at least 20 dB above signal off |
| `phase_lobe` | absolute out/in at eight pump phases, 45 deg apart, and the pump-off reference | one maximum and one minimum, 180 +- 15 deg apart, minimum below the pump-off reference |
| `idler` | signal and idler outputs for a signal detuned by 50 MHz | within 3 dB |
| `soft_start` | pump ladder with convergence flags | ascending, converged, steps of at most 3 dB |
| `pump_slope` | absolute out/in at the operating pump minus 3 dB | finite |
| `r_A` (1.0.1) | input-independence ratio, defined below | `r_A < 0.5` |
| `input_scaling_slope` (1.0.1) | input-scaling slope, defined below | `>= 0.5` |
| `peak_parity` (1.0.1) | best absolute out/in, analytical proxy (`delivered_gain`) and HB | `abs(analytical - HB) <= 5 dB` |
| `pump_sweep` (1.0.1) | absolute out/in and convergence over ascending pump powers, with the operating pump | bounded (not SPIKE), defined below |
| `junction_voltage` (1.0.1) | smallest and largest reverse junction voltage over the pump cycle, all cells, and the part's rating | smallest `>= 0` (no forward-biased interval) and largest `<= rating` |

**`r_A`, input independence:**

`r_A = abs(V_out(f_s; signal off)) / abs(V_out(f_s; signal on))`.

- `V_out(f_s)` is the output-node phasor at the signal frequency, mix (1, 0).
- Both runs have the pump on, at the operating pump power and phase.
- "Signal on" is the operating small-signal drive. "Signal off" sets the signal source to -100 dBm.
- An output that persists without input gives `r_A` near 1 (self-oscillation). An output that follows the
  input gives `r_A` near 0.

**`input_scaling_slope`:** the least-squares slope of the output power at `f_s` (dBm) against the incident
signal power (dBm). The fit runs over a small-signal input sweep at fixed pump power and phase; the
published records used -52, -46 and -40 dBm.

- A linear (input-proportional) amplifier gives a slope of 1.
- A pump-set output gives a slope near 0.

Both thresholds are the physical midpoints of these definitions (`analysis.regime_detector`).

**`pump_sweep`, bounded:**

- Every rung up to the operating pump `P_op` has converged.
- The lower 3 dB shoulder is below 15 dB: `absolute(P_op) - absolute(P_op - 3 dB) < 15 dB`.
- A turnover inside the swept range is not required.

**Verdicts:**

| Output | Meaning |
|---|---|
| `certified` | The 1.0.0 rule: the first seven fields present and passing. It also fails if any present 1.0.1 field fails. |
| `not_run` | The 1.0.1 fields left at `None`. A not-run check is never a pass, and it is optional on older records. |
| `methods_complete` | True only when every check in the table is present and passes. |

## Bench controls for an absolute record

The measurement-side counterpart of `GainRecord` requires three paired controls at the same reference planes and drive settings:

1. **Signal off, pump on:** measure the output in the signal band to bound self-oscillation, pump leakage and background. The signal-on response must stand above this control under the stated detection rule.
2. **Device removed or bypassed:** measure the source, cables and coupler path without the active device to establish the incident-signal and transfer reference. Record the plane and any mismatch correction rather than assuming the generator setting is the power entering the device.
3. **Signal added:** compare pump-on records with and without a small added input signal. Identify a linear-superposition window in which the incremental output follows the added input and the pump-only background remains bounded.

A pump-on/off ratio in a degenerate experiment is a relative enhancement, not an absolute gain measurement. A certified bench record also needs phase, idler, stability and order/device-tier evidence appropriate to the claimed mechanism; absent controls remain explicit missing fields.

## ADS Element Policy

Deprecated or inaccurate ADS implementations are not used in the public OA
amplifier exporter:

- `MLOC` is replaced by validated `MLEF` where open-end microstrip termination
  is needed.
- `LVIA` / lumped inductor-to-ground via emulation is replaced by validated
  `VIAGND` where via-to-ground shorting is needed.
- If `MLEF` or `VIAGND` are unavailable in a local ADS install, the exporter
  reports the limitation instead of silently substituting an unvalidated model.

Do not claim physical-layout accuracy beyond the validated schematic,
netlist, and component-mapping checks.

---

**Related:** the predicted-vs-realized rule these definitions serve →
[theory.md §5](theory.md#5-predicted-vs-realized-the-honesty-rule); where the
`MEAS_GAIN` numbers are read → [workflow.md](workflow.md#the-meas_gain-numbers-what-each-means).
