# Changelog

The entries below identify the evidence rounds documented in the companion data deposit, which holds the measured data set and the R1–R7 run records cited by their `runs/…` paths. The measured data set is archived separately on Zenodo; its DOI is given in the paper's Data availability statement. This repository starts at release 1.0.1; the 1.0.0 entry and the earlier entries record the development history before it.

## 1.0.1-paper — prepared 2026-10-07

The code release cited by the paper. Every number and rule the paper states is what this release does. Each
change F1-F10 has a test that fails on 1.0.0 and passes here.

- **F1, adopted gate constants.** The stage-0 gate defaults to `T_req = 0.17378`, cascade-anchored as
  `m*_D / (k V_bt^max)`. The earlier screening threshold remains available as `T_REQ_SCREEN_2026_09 = 0.1438`.
  `ALPHA_MAX = 0.02` Np/cell is unchanged. The definitions are in `docs/GAIN_DEFINITIONS.md`.
- **F2, gate D in the stage-0 gate.** `feasibility` refuses `alpha(f_s) > alpha_max` with the reason
  `signal_transport_above_budget`. It reports a margin for each gate.
- **F3, cascade layer.** A new module, `tvtl_ads_builder_v2.analytical.cascade_budget`, provides:
  - the Bloch phase mismatch `Delta_theta`;
  - `kappa/m` from the harmonic cascade;
  - the growth `g`, `m_EP`, `m*` and `m_del`;
  - the measured-dispersion `m_EP`.

  The optional phase-margin gate `0.5 m_del - m_EP >= 0` (reason `phase_margin_negative`) is off in the
  stage-0 gate and on in the optimiser's `--gate cascade`.
- **F4, tee-model range guard.** Above the MTEE reference-plane validity edge (11.64 GHz for the SMV1245 cell),
  `cell_functions_from_spec` used to drop the tee correction silently. Now:
  - It raises `TeeModelOutOfRange` (a `ValueError`; reason `tee_model_out_of_range`).
  - The stage-0 gate refuses with that reason.
  - `delivered_gain` withholds a harmonic-sensitivity value it cannot compute (`None`, with the reason).
  - The 1.0.0 behaviour remains available only on request, as `tee_out_of_range="reduced"`.
- **F5, depth ceiling before harmonic balance.** A candidate is refused with the reason
  `delivered_depth_above_ceiling` when its delivered junction pump amplitude exceeds the reverse-only swing
  `min(V_bias, V_max - V_bias)`, i.e. when its depth exceeds the C(V) ceiling at the bias point. The check is:
  - optional in `feasibility`;
  - on in the optimiser's `cascade` gate;
  - applied by the workflow before any HB job is emitted.
- **F6, via count in the generated deck.** `ViagndVia.emit` used to emit one `VIAGND` for every family. It
  now emits `n_parallel` per cell: `VIAGND_C{k}` and `VIAGND_C{k}_2` for `DOUBLE` families.
  - Single-via decks are byte-identical. The R1 SMV2019 gate-D record c3 decks reproduce their deposited
    SHA-256.
  - The analytical via termination of the gate's candidate cell divides by the via count, so the gate
    and the deck describe the same circuit. On the 0 V SMV1245 screen this gives 2,034 evaluable,
    A∧B = 0 at both thresholds, and corr +0.883 (+0.881 in 1.0.0).
  - Two worked examples are double-via families (`proxy_n2_double_short`, `proxy_n4_double_short`), so
    their golden decks change (see below). For them: ADS record generated with the pre-1.0.1 generator
    (one via per cell); analytical leg with two vias; re-simulation scheduled after 1.0.1.
  - The `smv2020` worked example is relabelled `n1_single_short`. It was registered as `n1_double_short`
    while its deck, headline and ADS records had one via. Its deck circuit is unchanged; only the deck's
    label comment changes. Its records stay valid.
- **F7, SMV2020 registry entry.** `SMV2020-079LF` no longer carries a polynomial C(V). The transcribed one
  was the SMV1405 device-of-record polynomial (C0 2.66 pF against the part's C_jo 3.20 pF).
  - Every C(V) path now uses the junction law.
  - The SPICE junction parameters are unchanged.
  - The rejected transcription is kept as `_POLY_XLSX_SMV2020_REJECTED`, for provenance only.
- **F8, certification fields of the Methods.** `GainRecord` gains five fields, each with a pass rule:
  - `r_A` < 0.5;
  - `input_scaling_slope` >= 0.5;
  - `peak_parity`, analytical vs HB within 5 dB;
  - `pump_sweep`, bounded: lower 3 dB shoulder < 15 dB, converged;
  - `junction_voltage`, reverse only and within the rating.

  On older records the new fields are optional: `None` is listed as `not_run` and never counts as a pass.
  `methods_complete` requires every check. The 1.0.0 checks and their tolerances are unchanged.
- **F9, figure inputs from the release.** Fig. 2 and Fig. 7 inputs now regenerate from the release alone.
  - `floquet_stub` gains `Y_stub_at_freq_with_shunts` and `build_stub_y_fn_with_mid_shunts`, the stub with
    the board's mid-stub vias. The change is additive.
  - `scripts/two_filter_sweep` is the Fig. 2(c) sweep. It regenerates the published task-2 arrays
    identically.
  - `scripts/dense_map` is the Fig. 7 tables. They regenerate byte-for-byte.
- **F10, DC return of the varactor.** A varactor terminal must have a DC path to ground; otherwise the
  release refuses with the reason `no_dc_return_for_varactor`.
  - The flat HB deck generator checks this before emitting.
  - The stage-0 gate's candidate cell refuses OPEN-stub families.
  - A bias-return resistor (`bias_return_ohm`, element `RDCRET_C{k}`, CLI `--bias-return-ohm`) restores
    them.
  - The 0 V SMV1245 screen keeps its 2,034 evaluable geometries with a declared 100 kOhm return; without
    one, its 1,081 open-stub geometries are refused.
- **F11, version and documentation.**
  - Version 1.0.1, and the `CITATION.cff` version and release date.
  - `docs/LIMITATIONS.md` lists the limits 1.0.1 enforces and the open items.
  - README gains a "Reproduce the paper" table, with the adopted `T_req` and the cascade layer.
  - `docs/REPRODUCE.md` and `STABILITY.md` are updated.
  - `scripts/check_paper_numbers.py` recomputes every analytical number the paper quotes and prints PASS or
    FAIL.
- Public release tree: development-process wording removed from comments and docs; unused combined-gain study removed; licence holder named; citation metadata completed.

Open items, not changed in 1.0.1 (`docs/LIMITATIONS.md`):

- the ideal-junction fallback of the harmonic-block tee model above its edge (used by the Fig. 2(c) sweep);
- the transient versus harmonic-balance transport mismatch (R7);
- measured absorption 89.7 % against 22.5 % for the evaluated deck (R4b);
- full-wave pad and neck parasitics, which the analytical screen does not include;
- the two double-via proxy examples: their ADS records were generated with the pre-1.0.1 generator (one via
  per cell), and re-simulation is scheduled after 1.0.1.

### Changed expectations

Existing tests whose expected value changed on purpose, with the old and new value:

| Test | Item | 1.0.0 | 1.0.1 |
|---|---|---|---|
| `tests/test_feasibility.py::test_fabricated_nominal_two_physical_exclusion_reasons`, SMV1245 tabulated cell at 3.1 GHz | F2 | `reasons == ['pump_delivery_below_threshold']` | `reasons == ['pump_delivery_below_threshold', 'signal_transport_above_budget']` (`alpha(f_s) = 0.03765` Np/cell) |
| `tests/test_amplifier_oa_export.py::test_unit_cell_cascade_is_connected`, `n2_double_short` unit-cell component counts | F6 | `VIAGND: 1` | `VIAGND: 2` |
| `tests/test_direct_amplifier_hpeesofsim.py::test_direct_amplifier_deck_has_public_self_pumped_policy`, 15-cell `n2_double_short` deck | F6 | 15 `VIAGND:` lines | 30 |
| `tests/test_hb_convergence_ladder.py::test_hb_convergence_ladder_dry_run_writes_reports_and_policy_decks`, one-cell deck | F6 | 1 `VIAGND:` line | 2 |
| `tests/test_deck_byte_identity.py`, golden `smv2020_flat_hb_deck.ckt` (SHA-256 of LF text) | relabel `n1_single_short` | `82f120fc69cd06bfcf1cb25b187908fb0c719f011d60036cd3263e9e3d0ea23d` | `d6ba15f4262a2f8c1b40aa1a503ffb2be1baa960e730f66214234c2fc774ab6e`; only line 2 changes (`; label = optimized_sp_n1_single_short_n24`); every circuit line is identical |
| `tests/test_deck_byte_identity.py`, golden `proxy_n2_double_short_flat_hb_deck.ckt` | F6 | `42e1af43096d27a996e675bf7ab7007b28e98133c4b3223e4ad76035a74a1fb9` | `25f743815bb7aa356b1debf106e29c2c3a64ff18bcde82584d6e1acef1b813dc` |
| `tests/test_deck_byte_identity.py`, golden `proxy_n4_double_short_flat_hb_deck.ckt` | F6 | `ffd0c5d5dc54d99d1f0f6c2ae85fead153bd5504b3f52f250e757c11ad60f341` | `bd40a39edf334017f91daa56d74301bf24720548d769e7d28ff79d509d8a9a59` |
| `tests/test_a4_standardization.py::test_standard_palette_emits_distributed_terminations`, OPEN-family deck | F10 | emitted without a DC return | emitted with `bias_return_ohm=1e5` (refused without) |
| `tests/test_direct_amplifier_hpeesofsim.py::test_direct_open_termination_uses_mlef_not_mloc_or_lvia`, `n2_none_open` deck | F10 | 15 `MLEF:` lines, no DC return | refused without a return; with 100 kOhm: 15 `MLEF:` and 15 `R:RDCRET_C` lines |
| `examples/real_parts/build_examples.py` `EXPECTED_STRUCTURE`, elements and per-cell `VIAGND` | F6 | proxy_n2 203 / 1; proxy_n4 233 / 1 | 218 / 2; 248 / 2 (smv2020 unchanged at 292 / 1; `results/specs.json` regenerated) |
| `results/theoretical_leg/candidates.csv` (pin in `DETERMINISM.md`, `tests/test_theoretical_leg_digest.py`) | regenerated | 14 columns, SHA-256 `7ad31bfb…35cb8b` | same 3 rows in the same order (`n2_double_short`, `n4_double_short`, `n3_single_short`), all 14 columns unchanged; 9 columns added: `absolute_outin_dB`, `T_2fs`, `T_fs`, `alpha_2fs_Np_per_cell`, `alpha_fs`, `kappa_branch`, `rho_para`, `A1_prime`, `feasible` (all three `True`); SHA-256 `57942633…08c6755` |
| `results/theoretical_leg/candidates.json` | regenerated | SHA-256 `a6dbe7c0…963562` | each candidate gains `feasibility` (the 1.0.1 stage-0 gate record: thresholds, T, alpha, margins, reasons, A1') and `absolute_outin_dB` in `metrics_at_drive` and `analytical["-3"]`; the predicted-gain disclaimer is reworded without study-specific figures; all other values unchanged; SHA-256 `0674693f…3a92bda` |
| `results/theoretical_leg/summary.json` | regenerated | SHA-256 `fb195bce…82743b` | `best_candidate` gains the same three fields; `objective` text now reads "ranked by the selected order_by gain metric" (was "ranked by positive coherent gain"); the predicted-gain disclaimer is reworded; SHA-256 `f7e668c7…551960c` |
| `results/theoretical_leg/report.md` and the three PNGs | regenerated | — | PNGs byte-identical to 1.0.0; `report.md` differs only in the reworded predicted-gain disclaimer line, SHA-256 `d9bae823…fea4beb` |
| README quickstart, `optimize-amplifier --freq-ghz 2.0 --quick --gate stage0` | F1, F2, F10 | 569 screened, 8 realizable, best `n4_single_short` 15.4 dB | 352 screened, 7 realizable, best `n4_single_short` 15.4 dB |

No tolerance was loosened.

## 1.0.0-paper — prepared 2026-09-24

Prepared the tool-only code release on `origin/main`, with the R1–R7 evidence and measurements staged in a separate data deposit. Scientific source code and worked-example outputs remain unchanged from the approved private branch.

## R7 — 2026-09-22 (threshold); 2026-09-10 (bias taper)

Recorded subharmonic threshold checks, transient/HB consistency limits and three native passive taper sweeps. See the R7 threshold report, data deposit `runs/r7_subharmonic_threshold/REPORT.md`, and R7 bias-taper report, data deposit `runs/r7_bias_taper/REPORT.md`.

## R6b — 2026-09-10

Restored K7 routing geometry and replayed the historical attenuation gate. See the R6b report, data deposit `runs/r6b_geometry_pool/REPORT.md`.

## R6 — 2026-09-09

Added delivered-gain quantities, quantitative feasibility filtering, certification requirements and the analytical design map. See the R6 report, data deposit `runs/r6_delivery_core/REPORT.md`.

## R5 — 2026-09-08

Screened 936 lumped-via realizations and recorded the bounded HB and calibration evidence. See the R5 report, data deposit `runs/r5_gain_search/REPORT.md`.

## R4b — 2026-09-07

Registered pump transport and full-family HB checks for the full-diode model. See the R4b report, data deposit `runs/r4b_fulldiode_hb/REPORT.md`.

## R4 — 2026-09-07

Recorded the full-diode capacitance-parity halt. See the R4 report, data deposit `runs/r4_fulldiode_hb/REPORT.md`.

## R3 — 2026-09-07

Recorded via-fit limits and bounded HB diagnostics. See the R3 report, data deposit `runs/r3_via_fit/REPORT.md`.

## R2 — 2026-08-09

Rebuilt the harmonic ladder and Fig. N6 under the accepted law. See the R2 report, data deposit `runs/r2_cascade_fign6/REPORT.md`.

## R1 — 2026-08-08

Installed the pump-delivery law as a Stage 0 gate and recorded the two-leg feasibility study. See the R1 report, data deposit `runs/r1_redesign_under_law/REPORT.md`.
