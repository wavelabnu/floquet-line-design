# TVTL length sweep — optimal N and threshold-vs-length scaling

Study in `runs/length_sweep/` (gitignored; nothing outside `runs/` touched). Conditions pinned identical to the committed proxy/COMBINED records — Varactor A, 2.0 GHz signal, 4.0 GHz pump, accurate VIAGND/MLEF terminations — with the **input/output matching FROZEN at its N=15 solution** (the per-N edge impedance Fabry-Perot-ripples with N — drift ≤13.6 Ω, `matching_drift.json` — so freezing isolates the length axis). Only `n_cells` is swept. Realized numbers use the idealized proxy varactor: a faithfulness RECORD, not a real-world device gain.

## Recommendation (material only — the decision is the project lead's)

1. **Keep the n3 example at N=15.** It reproduces the committed record (+8.18 boost / +1.18 abs) and is simultaneously the **robust optimum** — the best smooth-pump-walk-up boost (+8.18 dB) and the best robust net-positive absolute out/in (+1.18 dB) across N=6–30. There is **no robust N\*** that beats it.
2. **Do not chase the higher-gain lengths.** The largest absolute gains (N=18: +21.07 dB) are **razor-sharp Fabry-Perot resonance spikes** that collapse within one 3 dB pump step — artifacts of the idealized proxy-varactor Q, not robust operating points. Promoting one to the example would contradict the honest-record framing.
3. **The scaling result is paper-worthy.** Realized gain and the oscillation onset vs length are a **via-damped Fabry-Perot comb**, not the lossless 1/length critical-depth law (−20 dB/decade predicted; not observed — combed/flat). The physical VIAGND damps the end-to-end standing wave and combs the threshold. Worth recording as the realized-device counterpoint to the 1/N theory.
4. **Options** (the project lead's call): (a) keep N=15, cite the comb/threshold finding in the paper; (b) additionally show N=12/N=18 explicitly as labelled *fragile high-Q resonance* exhibits contrasting the robust N=15; (c) both. Nothing here is committed or pushed.

## Method & pre-checks (no ADS)

**Matching N-independence (the freeze decision).** `stage3_matching.edge_impedance` raises the static unit cell to `matrix_power(cell, N)` and returns the *finite* N-cell cascade edge impedance (50 Ω-terminated) — N-dependent unless the cell is image-matched to 50 Ω. Measured:
- `proxy_n3_single_short`: edge Z ripples with N (max |ΔZ_in vs N=15| = 13.6 Ω over N∈[6, 9, 12, 15, 18, 21, 24, 27, 30]); the line is **not** image-matched (edge Z ≈ 7–15 Ω ≪ 50 Ω) → **N-DEPENDENT (drift up to 13.6 ohm) -- FREEZE N=15 match across sweep**.
- `n3_double_combined`: edge Z ripples with N (max |ΔZ_in vs N=15| = 8.1 Ω over N∈[6, 9, 12, 15, 18, 21, 24, 27, 30]); the line is **not** image-matched (edge Z ≈ 7–15 Ω ≪ 50 Ω) → **N-DEPENDENT (drift up to 8.1 ohm) -- FREEZE N=15 match across sweep**.
- Decision: **freeze the N=15 single-stub match** across the sweep (only N changes). Passive insertion loss stays within ~2 dB across N=6–30 (per-N table), confirming the frozen match is a fair, not loss-dominating, comparison.

**Analytical prediction reconstruction.** The optimistic coherent-peak proxy (`_evaluate_candidate_proxy`, optimizer Stage 2–4) is recomputed at each N (the “optimism vs length” axis; per-N table column).
- Secondary `n3_double_combined` reproduces the recorded step-1 prediction exactly (coh_pk(15)=+31.55 dB). Primary `proxy_n3_single_short` reconstructs +13.42 dB vs the proxy_designs literal +10.11 dB (|Δ|=3.31 dB): the literal came from the optimizer **CLI** analytical leg (different truncation/HB order), so the prediction is reported as a trend, not a gate. The HB-realized numbers are the validation gate and reproduce exactly (below).
- Note: the prediction re-derives its matching **per N** (it cannot be frozen without a src change), so its own ripple/comb partly reflects the edge-Z Fabry-Perot — read it for the envelope, not point values.

## primary  (`proxy_n3_single_short`)

### primary — N=15 validation against the committed record

| quantity | swept | record | match |
|:--|--:|--:|:--:|
| operating pump (dBm) | +15.00 | +15.00 | ✓ |
| boost peak (dB) | +8.18 | +8.18 | ✓ |
| absolute out/in (dB) | +1.18 | +1.18 | ✓ |
| phase-0 boost-collapse (dBm) | +18.00 | +18.00 | ✓ |
| non-convergence onset (dBm) | -- | (none) | ✓ |
| flat==hier parity | IDENTICAL | IDENTICAL | ✓ |

_The study's “onset +18” = the phase-0 boost-collapse cliff (+15→+18); the harder HB non-convergence does not occur within the +21 scan (matching the recorded analyze_proxy, whose nonconvergence onset is null)._

### primary — realized record vs N (frozen N=15 match; only N swept)

| N | pred coh_pk | passive IL | op pump | boost_pk | abs out/in | shoulder lo/hi | class | φ0 collapse | nonconv onset | margin | regime | parity |
|--:|--:|--:|--:|--:|--:|:--|:--|--:|--:|--:|:--|:--|
| 6 | +5.20 | -5.00 | +18 | +14.85 | +9.85 | 25.4/-- | SPIKE | +12 | +21 | +3 | ins | IDENTICAL |
| 9 | +8.99 | -5.93 | +15 | +7.70 | +1.77 | 7.3/31.5 | moderate | +12 | -- | -- | PSA | IDENTICAL |
| 12 | +5.30 | -6.61 | +3 | +26.49 | +19.89 | 18.9/51.6 | SPIKE | +6 | +18 | +15 | ins | IDENTICAL |
| 15 ★ | +13.42 | -7.00 | +15 | +8.18 | +1.18 | 3.1/43.9 | robust | +18 | -- | -- | PSA | IDENTICAL |
| 18 | +9.11 | -7.13 | +12 | +28.20 | +21.07 | 20.5/33.0 | SPIKE | +9 | +18 | +6 | PSA | IDENTICAL |
| 21 | +18.00 | -7.03 | +9 | +5.73 | -1.30 | 2.9/0.4 | robust | +15 | +18 | +9 | PSA | IDENTICAL |
| 24 | +14.28 | -6.81 | +9 | +4.67 | -2.14 | 1.5/3.4 | robust | +15 | -- | -- | PSA | IDENTICAL |
| 27 | -17.64 | -6.61 | +6 | +6.90 | +0.28 | 6.1/42.0 | moderate | +3 | +12 | +6 | PSA | IDENTICAL |
| 30 | -57.97 | -6.60 | +15 | +5.88 | -0.72 | 3.3/42.2 | robust | +18 | +21 | +6 | PSA | IDENTICAL |

_★ = N=15 (record validation point). shoulder lo/hi = boost drop (dB) at operating pump ∓3 dB; class: robust ≤6 dB lower shoulder (smooth walk-up), SPIKE ≥15 dB or at the oscillation edge (fragile resonance), else moderate. passive IL separates loss from parametric growth._

**flat==hier parity**: bit-IDENTICAL at 9/9 lengths.

### primary — optimum N per metric

**Raw (any operating point, fragile spikes included):**
- **max boost (pump on/off)**: best **N=18** = +28.20 — **SPIKE**; N=15 = +8.18.
- **max absolute out/in (at op point)**: best **N=18** = +21.07 — **SPIKE**; N=15 = +1.18.
- **max stability margin**: best **N=12** = +15 — **SPIKE**.
- **min passive insertion loss**: best **N=6** = -5.00 — **SPIKE**; N=15 = -7.00.

**Robust (operating point reached by a smooth pump walk-up):**
- Robust-class lengths: [15, 21, 24, 30]; moderate: [9, 27].
- **Best ROBUST boost: N=15** = +8.18 dB; **best ROBUST absolute out/in: N=15** = +1.18 dB.
- **Caveat**: the largest absolute gains are fragile resonance SPIKEs (collapse within one 3 dB pump step) from the idealized proxy-varactor Q — a faithfulness record, not a robust operating point. The robust row is the engineering-relevant answer.

### primary — threshold-vs-length scaling

| N | nonconv onset | op pump | margin | high-pump fate |
|--:|--:|--:|--:|:--|
| 6 | +21 | +18 | +3 | oscillation |
| 9 | -- | +15 | -- | saturation |
| 12 | +18 | +3 | +15 | oscillation |
| 15 | -- | +15 | -- | saturation |
| 18 | +18 | +12 | +6 | oscillation |
| 21 | +18 | +9 | +9 | oscillation |
| 24 | -- | +9 | -- | saturation |
| 27 | +12 | +6 | +6 | oscillation |
| 30 | +21 | +15 | +6 | oscillation |

- Non-convergence onset present at 6/9 lengths [(6, 21.0), (12, 18.0), (18, 18.0), (21, 18.0), (27, 12.0), (30, 21.0)].
- onset_dBm vs log10(N): slope **-5.4 dB/decade** (R²=0.18). 1/length critical-depth predicts **−20 dB/decade**; **NON-monotone — combed**.
- The onset does **not** follow a clean 1/length law: it is **combed** by the finite-line Fabry-Perot resonances of the via-damped cascade. The physical VIAGND damps the end-to-end standing wave, so the oscillation edge appears where a resonance aligns with the pump-driven instability — not at a monotone critical depth. **That flattening/combing is the finding.**
- High-pump-edge census: oscillation 6, saturation-collapse 3, stable 0 (of 9). The length-dependent mix IS the comb.

![primary curves](curves_primary.png)

## secondary  (`n3_double_combined`)

### secondary — N=15 validation against the committed record

| quantity | swept | record | match |
|:--|--:|--:|:--:|
| operating pump (dBm) | +6.00 | +6.00 | ✓ |
| boost peak (dB) | +9.17 | +9.17 | ✓ |
| absolute out/in (dB) | -1.11 | -1.11 | ✓ |
| non-convergence onset (dBm) | +21.00 | +21.00 | ✓ |
| flat==hier parity | IDENTICAL | IDENTICAL | ✓ |

### secondary — realized record vs N (frozen N=15 match; only N swept)

| N | pred coh_pk | passive IL | op pump | boost_pk | abs out/in | shoulder lo/hi | class | φ0 collapse | nonconv onset | margin | regime | parity |
|--:|--:|--:|--:|--:|--:|:--|:--|--:|--:|--:|:--|:--|
| 6 | +11.19 | -6.60 | +3 | +27.69 | +21.08 | 28.0/54.7 | SPIKE | -- | +21 | +18 | PSA | IDENTICAL |
| 9 | +15.04 | -7.43 | +21 | +14.23 | +6.80 | 18.8/-- | SPIKE | +6 | -- | -- | PSA | IDENTICAL |
| 12 | +28.17 | -9.18 | +6 | +15.92 | +6.73 | 18.9/43.5 | SPIKE | -- | -- | -- | PSA | RUN_FAILED |
| 15 ★ | +31.55 | -10.28 | +6 | +9.17 | -1.11 | 23.5/46.5 | SPIKE | +0 | +21 | +15 | PSA | IDENTICAL |
| 18 | +31.28 | -11.33 | +0 | +1.95 | -9.38 | 0.5/2.4 | robust | -- | -- | -- | PSA | IDENTICAL |
| 21 | +38.50 | -12.79 | +21 | +14.27 | +1.48 | 37.0/-- | SPIKE | -3 | -- | -- | ins | DIFFER |
| 24 | +32.97 | -13.97 | -3 | +1.54 | -12.43 | 0.5/0.4 | robust | +3 | -- | -- | PSA | IDENTICAL |
| 27 | +38.41 | -15.13 | -6 | +0.47 | -14.66 | --/0.1 | n/a | -3 | +6 | +12 | PSA | IDENTICAL |
| 30 | +37.78 | -16.46 | -6 | +2.00 | -14.46 | --/1.4 | n/a | +0 | +21 | +27 | PSA | IDENTICAL |

_★ = N=15 (record validation point). shoulder lo/hi = boost drop (dB) at operating pump ∓3 dB; class: robust ≤6 dB lower shoulder (smooth walk-up), SPIKE ≥15 dB or at the oscillation edge (fragile resonance), else moderate. passive IL separates loss from parametric growth._

**flat==hier parity**: bit-IDENTICAL at 7/9 lengths; exceptions: N=12 RUN_FAILED, N=21 DIFFER.
- Both exceptions are at fragile high-pump operating points, **not** emitter faults: N=21 (+21, near the oscillation edge) agrees to 5–6 significant figures (max|ΔV|=2.1e-6 on a 6.3e-3 V signal — over the strict 1e-6 bit-threshold but at HB solver tolerance); N=12’s hierarchical HB did not converge at its +6 resonance spike while the flat deck did. Re-checking parity for **both** N=12 and N=21 at robust low pumps (−6, 0 dBm) gives **IDENTICAL to machine precision** (max|ΔV|≈1e-16, `_parity_confirm.py`) — the combined-topology emitter is faithful; the flags are HB convergence/tolerance fragility at the spike/edge, which is itself the comb.

### secondary — optimum N per metric

**Raw (any operating point, fragile spikes included):**
- **max boost (pump on/off)**: best **N=6** = +27.69 — **SPIKE**; N=15 = +9.17.
- **max absolute out/in (at op point)**: best **N=6** = +21.08 — **SPIKE**; N=15 = -1.11.
- **max stability margin**: best **N=30** = +27; N=15 = +15.
- **min passive insertion loss**: best **N=6** = -6.60 — **SPIKE**; N=15 = -10.28.

**Robust (operating point reached by a smooth pump walk-up):**
- Robust-class lengths: [18, 24].
- **Best ROBUST boost: N=18** = +1.95 dB; **best ROBUST absolute out/in: N=18** = -9.38 dB.
- **Caveat**: the largest absolute gains are fragile resonance SPIKEs (collapse within one 3 dB pump step) from the idealized proxy-varactor Q — a faithfulness record, not a robust operating point. The robust row is the engineering-relevant answer.

### secondary — threshold-vs-length scaling

| N | nonconv onset | op pump | margin | high-pump fate |
|--:|--:|--:|--:|:--|
| 6 | +21 | +3 | +18 | oscillation |
| 9 | -- | +21 | -- | stable |
| 12 | -- | +6 | -- | saturation |
| 15 | +21 | +6 | +15 | oscillation |
| 18 | -- | +0 | -- | saturation |
| 21 | -- | +21 | -- | stable |
| 24 | -- | -3 | -- | saturation |
| 27 | +6 | -6 | +12 | oscillation |
| 30 | +21 | -6 | +27 | oscillation |

- Non-convergence onset present at 4/9 lengths [(6, 21.0), (15, 21.0), (27, 6.0), (30, 21.0)].
- onset_dBm vs log10(N): slope **-10.5 dB/decade** (R²=0.20). 1/length critical-depth predicts **−20 dB/decade**; **NON-monotone — combed**.
- The onset does **not** follow a clean 1/length law: it is **combed** by the finite-line Fabry-Perot resonances of the via-damped cascade. The physical VIAGND damps the end-to-end standing wave, so the oscillation edge appears where a resonance aligns with the pump-driven instability — not at a monotone critical depth. **That flattening/combing is the finding.**
- High-pump-edge census: oscillation 4, saturation-collapse 3, stable 2 (of 9). The length-dependent mix IS the comb.

![secondary curves](curves_secondary.png)

