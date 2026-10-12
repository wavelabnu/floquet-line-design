# TVTL Workflow — the design loop and CLI reference

This page is the hands-on companion to [theory.md](theory.md): how to install the
tools, run a design end to end, and read what comes out. Every command here has
been run as written; the fenced **expected output** blocks are real (numbers may
shift in the last digits with library versions).

Two interpreters appear throughout:

- **Base Python** (an ordinary `.venv`) — runs the analytical core, the
  optimizer, structure/placement checks, and report scaffolding. **No Keysight
  ADS required.**
- **ADS Python** — the interpreter shipped with Keysight ADS
  (`<ADS_INSTALL_DIR>\tools\python\python.exe`). Required only for harmonic
  balance (`hpeesofsim`), OpenAccess workspace emission, and anything that reads
  an ADS dataset.

> Throughout, `<ADS_INSTALL_DIR>` is your local ADS install (for the machine this
> was authored on, `C:\Program Files\Keysight\ADS2026_Update1.2`). Never hard-code
> a private path into committed files.

---

## 1. Install and environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Confirm the base install:

```powershell
python -c "import tvtl, tvtl_ads_builder_v2; print('imports ok')"
floquet-line-design --help
```

```text
imports ok
```

The editable install puts one console script on your PATH:
**`floquet-line-design`** (defined in `pyproject.toml` →
`tvtl.optimization.cli:main`). Everything in the v2 builder is reached with
`python -m tvtl_ads_builder_v2 <command>`. `requires-python = ">=3.11`; the core
deps are NumPy, SciPy, pandas, PyYAML, and Matplotlib.

### ADS environment (only for HB / OpenAccess steps)

```powershell
$env:ADS_INSTALL_DIR = "<ADS_INSTALL_DIR>"
$env:ADS_PYTHON      = "<ADS_INSTALL_DIR>\tools\python\python.exe"
$env:HPEESOF_DIR     = "<ADS_INSTALL_DIR>"
$env:PYTHONPATH      = "src"     # so ADS Python finds the packages from a source checkout
```

`PYTHONPATH=src` matters because ADS Python is a separate interpreter that has not
had `pip install -e` run into it; pointing it at `src` lets it import `tvtl` and
`tvtl_ads_builder_v2` from the checkout. The HB runner
(`src/tvtl_ads_builder_v2/simulate/runner.py`) locates the simulator from
`HPEESOF_DIR\bin` (preferring `adssim_win.exe` on Windows); if it finds none, it
returns the status `blocked_missing_ads_runtime` rather than pretending to
succeed.

### What needs ADS, and what does not

| Runs on base Python (no ADS) | Needs ADS Python |
|---|---|
| `optimize-amplifier` (analytical ranking) | `build_examples.py parity / shots / assemble` |
| `build_examples.py specs` (structure + decks) | `--simulate` on any v2 command (HB) |
| `compare-gain`, `sweep-pump-gain`, `validate-self-pumped` **without** `--simulate` | `--draw-oa` (OpenAccess workspace) |
| `export-gui-project` (portable scaffold) | `analyze_proxy.py`, `headline_measurements.py` |
| the bulk of `pytest` | OpenAccess emission / dataset reads |

ADS-dependent tests skip cleanly when ADS is absent (they call
`pytest.importorskip("keysight.ads.de")`), so `python -m pytest -q` is green on a
plain machine.

---

## 2. Quickstart (about 10 minutes)

### a. Search for a 2 GHz amplifier

```powershell
.\.venv\Scripts\python.exe -m tvtl.optimization optimize-amplifier --freq-ghz 2.0 --out runs\amp_2ghz
```

```text
Target frequency: 2 GHz
Candidates: 3247 screened, top-10 refined, 7 realizable, 3 selected
Best candidate: n2_double_short (coherent_peak=10.6 dB, Z_TL=61.3 ohm, main_width=2.49 mm)
Predicted-leg figures: runs\amp_2ghz\fig_topology.png, runs\amp_2ghz\fig_predicted_sparams.png, runs\amp_2ghz\fig_predicted_nsweep.png
Output path: runs\amp_2ghz
Predictions only (analytical, not ADS-verified); ADS validation is optional and separate, and no .env file is required for this pure Python optimizer.
```

Add `--quick` for a ~2-second smoke run (smaller budget, one pump phase, a single
pick). The console-script form `floquet-line-design optimize-amplifier ...` is
identical.

### b. Read the report (predicted, not yet realized)

```powershell
type runs\amp_2ghz\report.md
```

```text
# Optimizer Run — Theoretical (Predicted) Leg

Target signal frequency (f_s): 2 GHz
Degenerate pump frequency (2 f_s): 4 GHz
Workflow: optimize-amplifier
Cells (N): 15
Screened candidates: 3247
Refined top-k: 10
Selected candidates: 3

> **PREDICTION ONLY — analytical, NOT ADS-verified.** Every number and every figure in this report is an analytical prediction from the pure-Python optimizer (`run_ads=False`).
> ANALYTICAL PREDICTION (optimistic; not ADS-verified). This is an analytical ranking, not a gain claim: the coherent-gain proxy can over-state the ADS harmonic-balance record by a family-dependent margin, and a boost ratio over a near-zero pump-off floor is never gain. Only a harmonic-balance record certified as in docs/GAIN_DEFINITIONS.md is a gain claim. See docs/LIMITATIONS.md 'Characterized Performance Limits'.
> ADS validation is optional and separate; no `.env` file is required for this pure-Python optimizer run.

## Best Candidate (predicted)

- Topology: n2_double_short
- Predicted coherent peak (analytical proxy, optimistic; see note above): 10.6 dB — PREDICTION, not ADS-verified
- Main line Z_TL: 61.3 Ω
- Main line width: 2.49 mm
- ... (predicted operating capacitance, analytical stability margin, per-candidate coherence-length heuristic, predicted S-parameters, embedded figures, and a "Numerical verification (ADS): pending" section follow)
```

The optimizer report carries **only the analytical prediction** — "coherent peak"
is a ranking proxy ([theory §5](theory.md#5-predicted-vs-realized-the-honesty-rule)).
A realized number requires ADS harmonic balance (Section 4). The run also writes
`summary.json`, `candidates.json`, and `candidates.csv`.

### c. Open a worked example

You do not need to run anything to learn the workflow — six designs are already
built and analyzed under `examples/real_parts/`. Read
[examples.md](examples.md) for the tour, or check one design's structure offline:

```powershell
.\.venv\Scripts\python.exe examples\real_parts\build_examples.py specs
```

That validates each design's element/node fingerprint and writes its runnable
flat HB deck — all without ADS.

---

## 3. The optimizer in full (`optimize-amplifier`)

The optimizer ranks unit-cell geometries with inexpensive physics proxies and
promotes the best few through matching and Floquet analysis. The ranking method
is documented in [OPTIMIZER_METHODOLOGY.md](methodology/OPTIMIZER_METHODOLOGY.md);
this is the command surface.

```powershell
.\.venv\Scripts\python.exe -m tvtl.optimization optimize-amplifier --freq-ghz <GHz> --out <dir> [flags]
```

| Flag | Default | Meaning |
|---|---|---|
| `--freq-ghz` | *required* | Target signal frequency in GHz. |
| `--out` | *required* | Output directory for summary and candidate files. |
| `--quick` | off | Small smoke-run budget (64/family, 1 phase). |
| `--samples-per-family` | 300 | Sobol budget per topology family (or use `--max-candidates`). |
| `--top-k` | 10 | Candidates refined after screening. |
| `--top-m` | 3 | Diverse designs reported. |
| `--n-cells` | 15 | Cells in the analytical cascade. |
| `--drive-dbm` | −3.0 | Control-port drive used for scoring. |
| `--seed` | 0 | Sobol scramble seed (runs are deterministic). |
| `--min-coherent-db` | 0.0 | Minimum coherent peak for selection. |

**The 12 linear families.** Topologies are enumerated as
`(via, termination) × n_segments ∈ {1,2,3,4}`
(`src/tvtl/optimization/topology_generator.py`). Of the 20 physical families, the
**12 linear** ones (OPEN and SHORT terminations) are the ADS-cross-validatable
path the optimizer searches by default; the 8 COMBINED families are
analytical-screening-only. A family key looks like `n2_double_short` (2 stub
segments, double via, SHORT termination).

**The default device.** Every optimizer run uses the low-parasitic proxy
*Varactor A* — `RealisticVaractor(CJ0=0.3 pF, VJ=3.5, M=1.7, CP=0, L_pkg=0.3 nH,
R=0.5 Ω)` (`src/tvtl/optimization/cli.py`, `_default_varactor`). It is idealized
on purpose, so optimizer picks are a clean test of the *framework*, not a catalog
part. ADS validation is always left off here (`run_ads=False`).

**Outputs** (`--out`):

- `report.md` — human summary, predicted only (shown above).
- `summary.json` — the selected design(s), the input to every downstream command.
- `candidates.json` — full per-candidate record: topology, `a1_score`, stub
  geometry (mm), main line `z_tl_ohm`/width, matching sections, and the analytical
  per-pump metrics. The `ads` field is `null` for a pure-Python run.
- `candidates.csv` — the same ranked, one row per candidate.

The default `--freq-ghz 2.0` run deterministically selects three diversified
designs — `n2_double_short`, `n4_double_short`, `n3_single_short`
(`Z_TL ≈ 61.27 Ω`, `w ≈ 2.49 mm`) — which are exactly the three proxy examples in
[examples.md](examples.md).

---

## 4. From a pick to a realized number

A `summary.json` feeds three kinds of downstream work. All three keep predicted
and realized in separate fields ([GAIN_DEFINITIONS.md](GAIN_DEFINITIONS.md)).

### Gain comparison

```powershell
# base Python: analytical column only
.\.venv\Scripts\python.exe -m tvtl_ads_builder_v2 compare-gain --input runs\amp_2ghz\summary.json --out runs\amp_2ghz_gain_compare_no_ads
```

```text
metric              analytical       direct HB        ADS-OA HB        status
coherent_gain_db    10.63 dB         NaN              NaN              comparison_unavailable
voltage_gain_db     NaN              NaN              NaN              unavailable
...
```

The HB columns are `NaN` until you run with ADS. The table layout —
**analytical | direct HB | ADS-OA HB** — is the predicted-vs-realized contract made
visible. With ADS Python, add `--simulate --draw-oa` to fill the HB columns:

```powershell
$env:PYTHONPATH = "src"
& $env:ADS_PYTHON -m tvtl_ads_builder_v2 compare-gain --input runs\amp_2ghz\summary.json --out runs\amp_2ghz_gain_compare_ads --simulate --draw-oa
```

### Pump-power sweep

```powershell
.\.venv\Scripts\python.exe -m tvtl_ads_builder_v2 sweep-pump-gain --input runs\amp_2ghz\summary.json --out runs\amp_2ghz_pump_sweep_no_ads --pump-start-dbm -20 --pump-stop-dbm 10 --pump-step-dbm 2
```

```text
Pump power sweep summary
metric: analytical coherent gain
best point: 40.47 dB at Ppump = 6.00 dBm
flagged points: 1 null/singular
numerical sweep: analytical_only
```

Writes `pump_sweep.csv`, `pump_sweep.json`, `pump_power_sweep_report.md`, and
`gain_vs_pump_power.png`. Raw analytical coherent gain is preserved in the
CSV/JSON; the plot is clamped to a bounded `-20…50 dB` axis with
null/singular points flagged, so a pathological analytical value cannot dominate
the scale. Add `--simulate --draw-oa` under ADS Python to add the realized HB
curve.

### Self-pumped circuit validation

```powershell
.\.venv\Scripts\python.exe -m tvtl_ads_builder_v2 validate-self-pumped --freq-ghz 2.0 --out runs\validation_self_pumped_no_ads
```

```text
Analytical: passed
Topology comparison: passed
Numerical comparison: not_available
Report: runs\validation_self_pumped_no_ads\comparison_report.json
```

This drives one `SelfPumpedCircuitSpec` through analytical → direct `hpeesofsim`
→ ADS OA schematic → ADS OA `hpeesofsim`, proving the same minimal circuit (signal
at `f_s` and pump at `2·f_s` on the **same** RF line). Without `--simulate
--draw-oa` it reports the analytical/topology legs and honestly marks the numerical
comparison `not_available`.

### The `MEAS_GAIN` numbers (what each means)

When an ADS run produces a dataset, gain is read with the equation block baked onto
the schematic (`src/tvtl_ads_builder_v2/oa/gain_equations.py`); these are the
**realized** numbers:

- **`G_mod_in_dB`** — `10·log10(Pout_mod / P_avs)`, the absolute out/in gain over
  the full degenerate folding family on `f_s` (`(1,0),(-1,1),(3,-1),(-3,2)`). This
  is the headline.
- **`Gain_coherent_dB`** — the 2-term (signal + idler) coherent gain.
- **`Gain_sig_dB`** — the signal-band transducer gain.
- **`Gain_ref_dB` / `Gain_4f_dB`** — comparison/reference definitions used to
  reconcile conventions.

`P_avs` is the available signal power (e.g. `-30 dBm → 1e-6 W`). Full algebra and
normalization: [GAIN_DEFINITIONS.md](GAIN_DEFINITIONS.md).

---

## 5. The six example designs (`build_examples.py`)

`examples/real_parts/build_examples.py` builds, proves, and assembles all six
shipped designs. It dispatches on its first argument (it is a script, not an
argparse CLI):

```powershell
# base Python — structure check + flat HB decks, no ADS
.\.venv\Scripts\python.exe examples\real_parts\build_examples.py specs

# ADS Python — the three ADS-gated stages
$env:PYTHONPATH = "src"
& $env:ADS_PYTHON examples\real_parts\build_examples.py parity all     # HB flat vs hierarchical, compare V(n_output)
& $env:ADS_PYTHON examples\real_parts\build_examples.py shots  all     # render PNGs + run the OA overlap scan
& $env:ADS_PYTHON examples\real_parts\build_examples.py assemble       # one openable workspace, six libraries
```

| Stage | ADS? | What it does | Key output |
|---|---|---|---|
| `specs` | no | Rebuilds each spec, checks element/node fingerprint, emits the flat deck. | `results/decks/<key>_flat_hb_deck.ckt`, `results/specs.json` |
| `parity` | yes | HB on flat **and** hierarchical decks; compares the signal-frequency `V(n_output)`; verdict IDENTICAL/CLOSE/DIFFER. | `results/parity_<key>.json` |
| `shots` | yes | Renders unit-cell and top-level PNGs; runs the strict **OA overlap scan** on every drawing. | `results/screenshots/*.png`, `results/overlap_scan.json` |
| `assemble` | yes | Bakes the `MEAS_GAIN` block, builds one workspace with a library per design. | `outputs/TVTL_examples_workspace/` |

**The parity gate** is the correctness gate: the hierarchical design must give the
same signal-frequency output voltage as the flat reference deck (bit-identical, or
to the HB convergence floor for the near-oscillation SMV2020 point). **The overlap
scan** is the placement gate: it measures geometric overlaps on each emitted
drawing and expects `strict_total == 0`. The proxy characterization
(`analyze_proxy.py all`, ADS) writes the predicted-vs-realized
`results/proxy_analysis_<key>.json` consumed by the per-library READMEs.

`results/` is committed (proofs, screenshots, JSONs); `outputs/` is git-ignored
(regenerated workspaces). See [examples.md](examples.md) for what each design
demonstrates and [extending.md](extending.md) to add your own.

---

## 6. Opening an assembled workspace in ADS (GUI)

After `build_examples.py assemble`, open the workspace in the ADS GUI:

1. Launch **Advanced Design System**.
2. **File → Open → Workspace…**
3. Select `examples\real_parts\outputs\TVTL_examples_workspace` (the directory
   containing `lib.defs`). ADS loads all six libraries.
4. In the workspace tree, expand a design's library (e.g. `real_amplifier_MAVR`)
   and open **`<top> > schematic`** for the full cascade (the HarmonicBalance
   controller is on it) or **`unit_cell > schematic`** for a single cell.
5. **Simulate** (the gear / F7).
6. In the dataset display, plot **`G_mod_in_dB`** (the full-family out/in gain) or
   **`Gain_coherent_dB`**. These evaluate on the harmonic-balance dataset — the
   realized record, not the analytical prediction.

To export your own optimizer pick for GUI inspection, use the portable scaffold
(works without ADS) and materialize the OpenAccess workspace under ADS Python:

```powershell
# portable scaffold (no ADS): device spec, flat deck, ads_native.net, topology.svg, OPEN_IN_ADS.txt
.\.venv\Scripts\python.exe -m tvtl_ads_builder_v2 export-gui-project --input runs\amp_2ghz\summary.json --out ads_projects\amp_2ghz

# connected self-pumped amplifier OA scaffold with gain-report breadcrumbs (ADS Python)
$env:PYTHONPATH = "src"
& $env:ADS_PYTHON -m tvtl_ads_builder_v2 export-amplifier-oa --input runs\amp_2ghz\summary.json --out ads_projects\amp_2ghz_amplifier_oa
```

The detailed ADS Python / OpenAccess automation API — placing validated symbols,
reading snap points, drawing wires, the `MLEF`/`VIAGND` element policy, and the
known pitfalls — is in [ADS_OA_DRAWING_GUIDE.md](ADS_OA_DRAWING_GUIDE.md).

---

## 7. The full `tvtl_ads_builder_v2` command set

```powershell
.\.venv\Scripts\python.exe -m tvtl_ads_builder_v2 <command> --help
```

| Command | Purpose |
|---|---|
| `export-gui-project` | Portable ADS scaffold (and OA workspace under ADS Python). |
| `export-amplifier-oa` | Connected self-pumped amplifier OA scaffold with gain-report breadcrumbs. |
| `validate-self-pumped` | The shared-line self-pumped circuit check across all four legs. |
| `compare-gain` | Analytical vs direct-HB vs OA-HB gain table. |
| `validate-amplifier-hb` | Direct deck + OA netlist + topology comparison for a pick. |
| `hb-convergence-ladder` | HB convergence vs cell count / order / pump power. |
| `sweep-pump-gain` | Gain vs pump power (CSV/JSON/MD/PNG). |

Commands that launch HB take a strictness pair: `--strict-convergence` returns a
nonzero exit when a requested HB run fails to converge, while the default
`--allow-nonconvergence` records non-convergence as data (because, per
[theory §4](theory.md#what-hb-non-convergence-means-physically), non-convergence
is itself a physical result — the oscillation edge).

---

## Reproduction and deeper reading

- Exact reproduction of the shipped results, base-Python and ADS:
  [REPRODUCE.md](REPRODUCE.md).
- The physics behind every step: [theory.md](theory.md).
- The optimizer's scoring internals:
  [OPTIMIZER_METHODOLOGY.md](methodology/OPTIMIZER_METHODOLOGY.md) and the staged
  design flow [PARAMETRIC_AMPLIFIER_DESIGN_FLOW.md](PARAMETRIC_AMPLIFIER_DESIGN_FLOW.md).
- When something fails: [troubleshooting.md](troubleshooting.md).
