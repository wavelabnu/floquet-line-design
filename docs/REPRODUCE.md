# Reproducing the Public Supplement

This document separates workflows that run with ordinary Python from workflows that require Keysight ADS. It is the exact-command companion to [workflow.md](workflow.md), which explains the design loop and what each number means; this page is the minimal sequence to reproduce the shipped results.

## Base Python Reproduction

Create an environment and install the package in editable mode:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the base validation suite:

```powershell
python -m pytest -q
python -c "import tvtl; print('tvtl import ok')"
python -c "import tvtl_ads_builder_v2; print('tvtl_ads_builder_v2 import ok')"
```

The base suite covers Python-side analytical models, schema validation, netlist text generation, OpenAccess emission logic that can be exercised without ADS, and selected optimization/design helpers.

## Recorded Environment

The committed results (`results/theoretical_leg/`, `examples/real_parts/results/`)
were produced under a recorded environment. `requirements-freeze.txt` pins the exact
base-Python package versions — a RECORD only; `pyproject.toml` stays the installable
source of truth (floating lower bounds). Wherever this page states an expected number,
it was produced here:

- **Base Python (analytical leg, committed `results/`):** Python 3.12.10, numpy 2.4.6,
  scipy 1.17.1, matplotlib 3.10.9, pandas 3.0.3, PyYAML 6.0.3, on Windows 11
  (10.0.26200). The Sobol screen is deterministic given a fixed seed **and** these
  library versions; on other minor versions expect agreement to ~1e-6 rather than
  bit-for-bit.
- **ADS leg:** Keysight ADS 2026 Update 1.2 (its bundled Python 3.13.2, numpy 2.2.3,
  scipy 1.16.0). `hpeesofsim` harmonic-balance and OpenAccess results were recorded
  under this install.

Historical note: an earlier, superseded environment description named a Python 3.11.9 / numpy 1.26.4 / scipy 1.13.1 freeze; the committed results and current `requirements-freeze.txt` use the versions above.

**Windows long-path note.** `hpeesofsim` crashes on workspace paths near the
260-character Windows limit. Clone to a short path (e.g. `C:\w\...`) or run the
ADS-dependent tests with a short temp dir (`pytest --basetemp=C:\w`); the ADS-gated
tests skip with this reason when the projected path is too long rather than crashing
opaquely.

## Paper Numbers and Figure Inputs

`scripts/` regenerates, from this release alone, the analytical numbers the paper quotes and the inputs of
its Figs. 2 and 7. `scripts/README.md` gives the commands and runtimes.

```powershell
python scripts\check_paper_numbers.py
python scripts\dense_map\dense_map.py --mode gates --family stub --workers 15
python scripts\two_filter_sweep\task2.py --out runs\two_filter_sweep
```

- `check_paper_numbers.py` prints every number with the value the paper quotes and PASS/FAIL. A FAIL is
  reported, never adjusted.
- The dense-map tables reproduce the published Fig. 7 inputs byte-for-byte (`--mode check`).
- The task-2 sweep reproduces the Fig. 2(c) arrays identically.

## Pure Python Optimizer

Run a quick optimizer search without ADS and without a `.env` file:

```powershell
.\.venv\Scripts\python.exe -m tvtl.optimization optimize-amplifier --freq-ghz 2.0 --out runs\amp_2ghz --quick
```

```text
Target frequency: 2 GHz
Candidates: 696 screened, top-8 refined, 3 realizable, 1 selected
Best candidate: n2_single_short (coherent_peak=12 dB, Z_TL=61.3 ohm, main_width=2.49 mm)
```

The optimizer writes `summary.json`, `candidates.json`, `candidates.csv`, and `report.md` under the requested `runs\` directory. These generated outputs are local evidence artifacts and are not committed. Drop `--quick` for the full deterministic run (3247 screened, three diversified picks — the three proxy examples).

### The committed theoretical-leg artifact

The predicted optimizer output shipped under `results/theoretical_leg/` (report,
JSONs, CSV, and figures) is exactly the documented default run at 2.0 GHz.
Regenerate it in place with:

```powershell
.\.venv\Scripts\python.exe -m tvtl.optimization optimize-amplifier --freq-ghz 2.0 --out results\theoretical_leg
```

The run is deterministic (seed 0): `report.md`, `summary.json`, `candidates.json`,
`candidates.csv`, and the three figures reproduce byte-for-byte (only the recorded
`--out` path varies), and the best predicted candidate is `n2_double_short` at a
~10.6 dB coherent-peak proxy. The tree was regenerated for release 1.0.1. The candidates gained their
stage-0 gate fields, and the 1.0.0 values are unchanged. `results/theoretical_leg/DETERMINISM.md` holds
the pins.

**`results/` vs `runs/`.** `results/` holds committed, citable artifacts that are
regenerated only deliberately by the documented commands above; `runs/`,
`ads_projects/`, and the other output roots are git-ignored scratch and are never
committed. Point `--out` at `runs\...` for throwaway searches, and at
`results\theoretical_leg` only when deliberately refreshing the shipped artifact.

## Gain Comparison Reports

The gain equations and normalization conventions are documented in
`docs/GAIN_DEFINITIONS.md`. Reports distinguish analytical, direct
`hpeesofsim`, and ADS-OA `hpeesofsim` metrics. Unavailable metrics are reported
as `null` with notes, not silently fabricated.
Gain-vs-pump-power sweeps produce CSV, JSON, Markdown, and PNG outputs. Raw
analytical coherent gain is preserved in CSV/JSON; plot values default to a
bounded `-20` to `50` dB axis, with points below `--invalid-gain-floor-db -100`
flagged as `null_or_singular`.

Run the ADS-free gain-definition and analytical report:

```powershell
.\.venv\Scripts\python.exe -m tvtl_ads_builder_v2 compare-gain --input runs\amp_2ghz\summary.json --out runs\amp_2ghz_gain_compare_no_ads
```

Run the ADS-free pump-power sweep:

```powershell
.\.venv\Scripts\python.exe -m tvtl_ads_builder_v2 sweep-pump-gain --input runs\amp_2ghz\summary.json --out runs\amp_2ghz_pump_sweep_no_ads --pump-start-dbm -20 --pump-stop-dbm 10 --pump-step-dbm 2
```

Run the ADS path with component validation, optional simulation, and ADS OA
netlist audit:

```powershell
$env:PYTHONPATH = "src"
& $env:ADS_PYTHON -m tvtl_ads_builder_v2 compare-gain --input runs\amp_2ghz\summary.json --out runs\amp_2ghz_gain_compare_ads --simulate --draw-oa
& $env:ADS_PYTHON -m tvtl_ads_builder_v2 sweep-pump-gain --input runs\amp_2ghz\summary.json --out runs\amp_2ghz_pump_sweep_ads --pump-start-dbm -20 --pump-stop-dbm 10 --pump-step-dbm 2 --simulate --draw-oa
```

The report files are:

```text
runs\amp_2ghz_gain_compare_ads\analytical_gain.json
runs\amp_2ghz_gain_compare_ads\numerical_gain_direct.json
runs\amp_2ghz_gain_compare_ads\numerical_gain_oa.json
runs\amp_2ghz_gain_compare_ads\gain_comparison_report.json
runs\amp_2ghz_gain_compare_ads\gain_comparison_report.md
runs\amp_2ghz_pump_sweep_ads\pump_sweep.csv
runs\amp_2ghz_pump_sweep_ads\pump_sweep.json
runs\amp_2ghz_pump_sweep_ads\pump_power_sweep_report.md
runs\amp_2ghz_pump_sweep_ads\gain_vs_pump_power.png
```

## ADS GUI Export

Export the optimizer result to a public ADS project package:

```powershell
.\.venv\Scripts\python.exe -m tvtl_ads_builder_v2 export-gui-project --input runs\amp_2ghz\summary.json --out ads_projects\amp_2ghz
```

The top-level console script exposes the same workflow:

```powershell
.\.venv\Scripts\floquet-line-design.exe ads-export --input runs\amp_2ghz\summary.json --out ads_projects\amp_2ghz
```

Normal Python writes a portable scaffold containing the rebuilt DeviceSpec, flat `hpeesofsim` deck, ADS-native netlist, topology/audit aids, a manifest, and `OPEN_IN_ADS.txt`. This scaffold is useful for review and handoff even when ADS is not installed.

With ADS Python, export the connected self-pumped amplifier OA scaffold with
gain-report breadcrumbs:

```powershell
$env:PYTHONPATH = "src"
& $env:ADS_PYTHON -m tvtl_ads_builder_v2 export-amplifier-oa --input runs\amp_2ghz\summary.json --out ads_projects\amp_2ghz_amplifier_gain_annotated_oa
```

The amplifier exporter auto-discovers `runs\amp_2ghz_gain_compare(_ads)` and
`runs\amp_2ghz_pump_sweep(_ads)`, copies available report/plot artifacts into
`gain_reports\`, and records them in `OPEN_IN_ADS.txt`, the schematic plan, and
the manifest. Use `--gain-comparison-report`, `--gain-comparison-json`,
`--pump-sweep-report`, `--pump-sweep-json`, and `--pump-sweep-plot` to provide
explicit artifact paths.

To materialize an ADS OpenAccess workspace, run the export under ADS Python or run the generated `build_oa_workspace.bat` after setting `ADS_PYTHON` or `ADS_INSTALL_DIR`. Then open the generated `oa_workspace` directory from ADS GUI with `File > Open > Workspace`.

## Self-Pumped ADS OA Validation

The validation command starts from one
`tvtl_ads_builder_v2.circuit_spec.SelfPumpedCircuitSpec` and compares the same
minimal circuit through:

```text
Analytical -> direct hpeesofsim -> ADS OA schematic -> ADS OA hpeesofsim -> comparison
```

The pump source at `2*fs` is on the same RF/main line as the signal source at
`fs`. Both source terminal-1 pins connect to `RF_MAIN`, both terminal-2 pins
are grounded, and no separate `PUMP`, `CONTROL`, or `BIAS` lane is used.

Run the ADS-free analytical/topology report:

```powershell
.\.venv\Scripts\python.exe -m tvtl_ads_builder_v2 validate-self-pumped --freq-ghz 2.0 --out runs\validation_self_pumped_no_ads
```

Run the full ADS path with ADS Python:

```powershell
& "$env:HPEESOF_DIR\tools\python\python.exe" -m tvtl_ads_builder_v2 validate-self-pumped --freq-ghz 2.0 --out runs\validation_self_pumped_ads --simulate --draw-oa
```

If `HPEESOF_DIR` is not set, point at the installed ADS Python directly (replace
`<ADS_INSTALL_DIR>` with your install path):

```powershell
& "<ADS_INSTALL_DIR>\tools\python\python.exe" -m tvtl_ads_builder_v2 validate-self-pumped --freq-ghz 2.0 --out runs\validation_self_pumped_ads --simulate --draw-oa
```

The report files are:

```text
runs\validation_self_pumped_ads\comparison_report.json
runs\validation_self_pumped_ads\comparison_report.md
```

The numerical comparison is attempted only after both `.ds` datasets exist and
the HB dataset parser can expose a robust `RF_MAIN` phasor at `fs`. Otherwise
the report honestly marks the result as topology-only / numerical comparison
not available.

The public OA amplifier exporter uses validated ADS distributed terminations
for active amplifier unit cells. `MLOC` is replaced by `MLEF` where open-end
microstrip termination is needed. `LVIA` and lumped inductor-to-ground via
emulation are replaced by `VIAGND` where via-to-ground shorting is needed. If
`MLEF` or `VIAGND` are unavailable in a local ADS install, the exporter reports
the limitation instead of silently substituting an unvalidated model.

## ADS-Dependent Reproduction

ADS workflows require a local Keysight ADS installation and the ADS Python bridge. Use placeholders in scripts and notes rather than machine-specific paths:

```powershell
$env:ADS_INSTALL_DIR = "<ADS_INSTALL_DIR>"
$env:ADS_PYTHON = "<ADS_INSTALL_DIR>\tools\python\python.exe"
$env:HPEESOF_DIR = "<ADS_INSTALL_DIR>"
$env:TVTL_AEL_REFERENCE_WORKSPACE = "<ADS_REFERENCE_WORKSPACE>"
$env:TVTL_MTEE_SURROGATE_JSON = "<SURROGATE_JSON>"
```

Optional ADS validation can be run with the ADS Python interpreter when available:

```powershell
& "<ADS_INSTALL_DIR>\tools\python\python.exe" -m pytest -q
```

Optional ADS project export with ADS Python:

```powershell
$env:PYTHONPATH = "src"
& "<ADS_INSTALL_DIR>\tools\python\python.exe" -m tvtl_ads_builder_v2 export-gui-project --input runs\amp_2ghz\summary.json --out ads_projects\amp_2ghz_ads --backend oa
```

Generated optimizer runs, ADS project packages, OpenAccess workspaces, datasets, and reports should be written outside the source tree or under explicit local output roots such as `runs\`, `ads_projects\`, or `<OUTPUT_DIR>`. These artifacts are intentionally not committed.

## Figure And Report Regeneration

The public copy does not include private diagnostic pipelines or generated result archives. Regeneration scripts should be run from documented public inputs and should write to an explicit output directory:

```powershell
$env:TVTL_OUTPUT_DIR = "<OUTPUT_DIR>"
```

If a report or figure requires ADS-generated datasets, regenerate those datasets with the local ADS installation first, then run the Python-side parser/analysis step against the generated files.

### Shipped tool result records

The following committed `results/` tables and worked examples remain in the code release. The paper figure map and R1–R7 evidence index are in the companion data deposit. The measured data set is archived separately on Zenodo; its DOI is given in the paper's Data availability statement.

| Record | Reproduction entry point | Pinned output | Recorded runtime |
|---|---|---|---|
| `results/theoretical_leg/` | optimizer command above without `--quick` | 3,247 screened, three selected; best analytical proxy `n2_double_short` at 10.6 dB coherent peak | Not recorded for the historical full run; R8 gated quick run took about 18 s |
| `results/length_sweep/` | its `REPORT.md` and frozen N=15 matching record | Robust proxy N=15 record +1.18 dB absolute out/in; N=18 spike is fragile | ADS runtime not recorded |

At the R8 clean-branch checkpoint, the base validation suite took 238.99 s (1,116 passed, three skipped) and the ADS-Python selection took 12.52 s (46 passed). These are test runtimes, not timings for the historical ADS research runs.

## Worked Example Reproduction

The six shipped designs reproduce offline from frozen design points (no search, no
spreadsheet). Check their structure and emit the flat decks with base Python:

```powershell
.\.venv\Scripts\python.exe examples\real_parts\build_examples.py specs
```

Under ADS Python, regenerate the parity proofs, screenshots, and the assembled
workspace, and the predicted-vs-realized proxy analysis:

```powershell
$env:PYTHONPATH = "src"
& "<ADS_INSTALL_DIR>\tools\python\python.exe" examples\real_parts\build_examples.py parity all
& "<ADS_INSTALL_DIR>\tools\python\python.exe" examples\real_parts\build_examples.py shots all
& "<ADS_INSTALL_DIR>\tools\python\python.exe" examples\real_parts\build_examples.py assemble
& "<ADS_INSTALL_DIR>\tools\python\python.exe" examples\real_parts\analyze_proxy.py all
```

The committed proofs are under `examples/real_parts/results/` (parity, overlap
scan, headline and proxy-analysis JSONs). What each design demonstrates and the
exact predicted/realized numbers are in [examples.md](examples.md); the build
stages are documented in [workflow.md](workflow.md#5-the-six-example-designs-build_examplespy)
and `examples/real_parts/README.md`.

## Notes

- No private development paths are required to import the package.
- The legacy v1 ADS builder is intentionally absent.
- Some optional calibration/evidence artifacts are not bundled; set the environment variables above when using those optional paths.
- A local `.env` may be used by a developer's shell, but it is optional and untracked. Start from `.env.example` if desired.


