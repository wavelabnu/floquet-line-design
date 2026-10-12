# Paper reproduction scripts

These scripts regenerate the analytical inputs of the paper's figures from this release alone. Every
output is tier `ANALYTICAL`. Outputs go to git-ignored folders under `runs/` unless `--out` / `--out-dir`
says otherwise.

| Script | Regenerates | Runtime (16-core desktop) |
|---|---|---|
| `two_filter_sweep/task2.py` | Fig. 2(c) "realized" fractions: `frac_usable_measured` of `task2_summary.json`, 0.6593 / 0.3396 / 0.0904 / 0.0139 at C0 = 0.3 / 1 / 3 / 7.37 pF | about 30 min (7.5 min per C0; `--c0-pf` runs one C0) |
| `two_filter_sweep/task1.py` | the geometry sweep of the same study (nominal and W-matched via inductance) | about 15 min |
| `dense_map/dense_map.py` | Fig. 7 inputs: `dense_stub.csv`, `dense_plain.csv`, `proxy.csv`, byte-identical to the published tables (`--mode check`) | under 1 min with 15 workers |
| `check_paper_numbers.py` | every analytical number of the paper that the code produces, printed next to its quoted value with PASS/FAIL | about 4 min (`--skip-slow`: seconds) |

## Fig. 7

```powershell
python scripts\dense_map\dense_map.py --mode gates --family stub --workers 15
python scripts\dense_map\dense_map.py --mode gates --family plain --workers 15
python scripts\dense_map\dense_map.py --mode proxy --workers 15
python scripts\dense_map\dense_map.py --mode check
python scripts\dense_map\dense_map.py --mode counts
```

`--mode counts` re-gates the stub table at the adopted constants (`T_req = 0.17378`, `alpha_max = 0.02`,
gates A, B, D and C). It reports 140,049 points, 7,848 that pass the law and 1,776 admissible. The tool-verdict
columns `feasible_tool` / `admissible` of the tables keep the 2026-09 convention (`T_req = 0.1438`, no gate
D); Fig. 7 does not read them. The script docstring gives the details.

The tables use the reduced tee model above the cell's tee-model edge, as the published tables did:

| Samples | Reduced tee model from |
|---|---|
| pump columns, `2 f_s` | `f_s = 5.85 GHz` |
| proxy harmonics `3 f_s` | `f_s = 3.9 GHz` |
| proxy harmonics `5 f_s` | `f_s = 2.35 GHz` |
| proxy harmonics `7 f_s` | `f_s = 1.7 GHz` |

The release default refuses such samples (`tee_model_out_of_range`). The script requests the reduced model
explicitly and `--mode tee` lists the affected frequencies. None of the 1,776 admissible points uses the
reduced model for its pump sample.

## Fig. 2(c)

```powershell
python scripts\two_filter_sweep\task2.py --out runs\two_filter_sweep
```

The sweep uses the 2026-09 screening threshold `T_REQ_SCREEN_2026_09 = 0.1438` and the passband
form of the transport test, as the published panel does. Fig. 2 (a) and (b) are drawn from committed files:

- `examples/real_parts/results/branch_delivery_vs_freq.json`;
- `results/figures/fig1d_analytical.csv`;
- the closed form `sweep.difficulty_law_fraction`.

The stub model is the analytical three-section stub with the board's two mid-stub vias,
`tvtl_ads_builder_v2.analytical.floquet_stub.Y_stub_at_freq_with_shunts`. Above 11.64 GHz the cell's MTEE
blocks fall back to the ideal junction inside `mtee_blocks.tee_harmonic_blocks` (a documented fallback;
`docs/LIMITATIONS.md`).
