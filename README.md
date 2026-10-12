# floquet-line-design

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23315340.svg)](https://doi.org/10.5281/zenodo.23315340)

Physics-guided design and verification of varactor-loaded time-varying transmission lines.

It models the passive pump and signal paths of varactor-loaded time-varying transmission lines.
It screens candidate designs with a quantitative feasibility gate before scoring them.
It builds analytical predictions and ADS circuit records, with the evidence tier kept explicit.
It includes tool documentation, tests, configs, six worked designs, and the scripts that regenerate the paper's analytical numbers and figure inputs.
The companion data deposit holds measurements and the R1–R7 evidence rounds.

## The pump-delivery law

- At the junction, pump transfer must reach `T(2f_s) ≥ T_req = 0.17378` (cascade-anchored; the 2026-09 screens used 0.1438, kept as `T_REQ_SCREEN_2026_09`).
- Pump attenuation must satisfy `α(2f_s) ≤ α_max = 0.02 Np/cell`, and the signal must meet the same budget, `α(f_s) ≤ α_max` (gate D).
- The same shunt reactance faces opposite demands: a strong pump voltage division and low through-line loss.
- The available modulation contracts with `ω_p C₀`, so a prescribed Floquet depth does not imply delivery at the junction.
- The gate evaluates a specified cell, bias, frequency, geometry, and drive model; its outcome is tier `ANALYTICAL`.

## Included

- Analytical TVTL support under `src/tvtl/` for ABCD matrices, Floquet/harmonic models, microstrip estimates, varactor models, and staged design/optimization helpers.
- The clean-room v2 ADS builder under `src/tvtl_ads_builder_v2/` for device specs, flat `hpeesofsim` decks, OpenAccess workspace emission, netlist checks, and selected parity workflows.
- Public model-registry configs under `configs/model_registry/` for reduced and standard paper conventions.
- Public-facing methodology and reproduction notes under `docs/`.
- A Python test suite covering the copied public core and v2 surface.

## What is not here

- Legacy v1 ADS builder code (`src/tvtl_ads_builder/`).
- Private diagnostics, generated outputs, ADS scratch workspaces, and R1–R7 evidence rounds; their paths and hashes are documented in the companion data deposit.
- Measurements. The repository contains one small measured-dispersion file used by the tests (`tests/fixtures/measured_dispersion/`); the measured data set is archived separately on Zenodo; its DOI is given in the paper's Data availability statement.
- Collaborator-specific configs and unpublished local reconstruction material.
- Host-specific paths from private development machines.

## Predicted vs. Realized Performance

Two kinds of gain number appear throughout this workflow, and they are always
reported **separately** — never combined into a single headline figure:

- The **analytical / Floquet figure is a PREDICTION**: a fast ranking indicator used
  to screen and order candidate designs. It is *optimistic* — it omits loss, mismatch,
  and large-signal effects, so it generally over-states realized gain. Treat it as a
  relative score for choosing between designs, not as a device specification.
- The **ADS harmonic-balance (HB) figure is the RECORD**: the device figure of merit.
  Every gain, stability, or convergence claim about a final design rests on the ADS
  result, not on the analytical one.

Reporting keeps these in distinct fields (analytical, direct `hpeesofsim`, and ADS-OA
`hpeesofsim`). A metric that has not been computed is written as `null` with a note —
it is never back-filled from the prediction. `docs/GAIN_DEFINITIONS.md` gives the
precise definitions and `docs/LIMITATIONS.md` the caveats.

## Certification record

`GainRecord` stores `absolute_outin_dB`, `order_convergence`,
`device_tier_agreement`, `signal_off_pump_on`, `phase_lobe`, `idler`,
`soft_start`, `pump_slope`, and the evidence `tier`. It also stores the 1.0.1 checks `r_A`,
`input_scaling_slope`, `peak_parity`, `pump_sweep` and `junction_voltage`. Its evaluator reports
missing, failed and not-run controls explicitly. **No gain is recorded without them.** The pass rules are in
[GAIN_DEFINITIONS.md](docs/GAIN_DEFINITIONS.md#certification-record-gainrecord).

## Evidence tiers

`ANALYTICAL` labels model predictions and feasibility screens.
`ADS-REALIZED` labels circuit-simulation records and their controls.
`MEASURED` labels instrument data at the stated reference plane. A pump-on/off difference recorded on a spectrum analyzer is a relative observation; in degenerate operation it is not a gain measurement, and on the fabricated line the 2026 controls attribute it to the pump path (present with the signal off, partly without the board).

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Requires Python ≥ 3.11. The editable install adds one console script,
`floquet-line-design`; everything in the ADS builder is reached with
`python -m tvtl_ads_builder_v2 <command>`. On Linux/macOS, activate with
`source .venv/bin/activate` and use forward-slash paths — the analytical leg is
OS-independent; the ADS workflows are validated on Windows.

## Quickstart (about 10 minutes)

**1. Confirm the base install (no ADS needed).**

```powershell
python -m pytest -q
python -c "import tvtl, tvtl_ads_builder_v2; print('imports ok')"
```

These tests run without Keysight ADS; ADS-dependent tests skip cleanly.

**2. Run the feasibility gate on the SMV1245 nominal cell at 3.1 GHz.** This uses the tracked 80 GHz stub table and the 0 V varactor record:

```powershell
python -c "import sys; sys.path[:0]=['examples/real_parts','examples/real_parts/boundary']; from real_part_designs import spec_smv1245; from _stub_table import STUB_80GHz,load_s1p,make_y_override; from tvtl.varactor_palette import get_entry; from tvtl_ads_builder_v2.analytical.delivered_gain import cell_functions_from_spec; from tvtl.design.feasibility import feasibility; v=get_entry('SMV1245-079LF').to_realistic_varactor(v_bias_v=0); c=cell_functions_from_spec(spec=spec_smv1245(n_cells=1),varactor=v,stub_y_fn=make_y_override(*load_s1p(STUB_80GHz)))[0]; print('reasons='+str(feasibility(c,3.1e9,N=20)['reasons']))"
```

Expected: `reasons=['pump_delivery_below_threshold', 'signal_transport_above_budget']` (analytical model; no measured threshold claim).

**3. Search for a 2 GHz design with the Stage 0 gate.**

```powershell
.\.venv\Scripts\python.exe -m tvtl.optimization optimize-amplifier --freq-ghz 2.0 --out runs\amp_2ghz --quick --gate stage0
```

```text
Target frequency: 2 GHz
Candidates: 352 screened, top-8 refined, 7 realizable, 1 selected
Best candidate: n4_single_short (coherent_peak=15.4 dB, Z_TL=61.3 ohm, main_width=2.49 mm)
```

The Stage 0 gate uses the adopted `T_req = 0.17378`, the signal-transport gate D and the DC-return rule, which
refuses open-stub families without `--bias-return-ohm` (1.0.0: 569 screened, 8 realizable). `--gate cascade`
adds the phase margin `0.5 m_del - m_EP >= 0` and the delivered-depth ceiling.

The run writes `summary.json`, `candidates.json`, `candidates.csv`, and
`report.md`. The quick run above took about 18 seconds in the R8 environment. No `.env` is required; the
optimizer is pure Python and leaves ADS validation off.

**4. Read the report and open a worked example.** `report.md` carries the
*predicted* (analytical) ranking; a *realized* number needs ADS harmonic balance.
Six designs are already built and analyzed under `examples/real_parts/` — start
with the [examples tour](docs/examples.md), or check one design offline with
`python examples\real_parts\build_examples.py specs`.

The full design loop, every CLI command with expected output, and the ADS steps
are in **[docs/workflow.md](docs/workflow.md)**.

## Documentation

| Page | What it covers |
|---|---|
| [docs/theory.md](docs/theory.md) | The physics the code implements: unit cell, parametric gain, dispersion, the two regimes, pump delivery. |
| [docs/workflow.md](docs/workflow.md) | The full design loop and CLI reference, every command verified with expected output. |
| [docs/examples.md](docs/examples.md) | A tour of the six worked designs, predicted and realized numbers side by side. |
| [docs/extending.md](docs/extending.md) | Add a varactor, a topology family, or an analysis. |
| [docs/troubleshooting.md](docs/troubleshooting.md) | How to read HB non-convergence and other expected "failures". |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | Exact reproduction of the shipped results, base-Python and ADS. |

Reference depth: [GAIN_DEFINITIONS.md](docs/GAIN_DEFINITIONS.md) (gain algebra),
[LIMITATIONS.md](docs/LIMITATIONS.md) (scope and caveats),
[OPTIMIZER_METHODOLOGY.md](docs/methodology/OPTIMIZER_METHODOLOGY.md) and
[PARAMETRIC_AMPLIFIER_DESIGN_FLOW.md](docs/PARAMETRIC_AMPLIFIER_DESIGN_FLOW.md)
(design method), [ADS_OA_DRAWING_GUIDE.md](docs/ADS_OA_DRAWING_GUIDE.md) (ADS
Python / OpenAccess automation), and [STABILITY.md](STABILITY.md) (the stable
public API).

## ADS-Dependent Workflows

ADS / OpenAccess / HB workflows require a local Keysight ADS installation and the
ADS Python bridge (validated against **Keysight ADS 2026 Update 1.2**). Configure
your local install path and point ADS Python at the source checkout:

```powershell
$env:HPEESOF_DIR     = "<ADS_INSTALL_DIR>"
$env:ADS_INSTALL_DIR = "<ADS_INSTALL_DIR>"
$env:ADS_PYTHON      = "<ADS_INSTALL_DIR>\tools\python\python.exe"
$env:PYTHONPATH      = "src"
```

What runs without ADS (the optimizer, structure checks, analytical reports,
portable scaffolds) versus what needs it (harmonic balance, OpenAccess emission,
dataset reads) is tabulated in [docs/workflow.md](docs/workflow.md#what-needs-ads-and-what-does-not).
Generated workspaces, datasets, and reports are written under git-ignored output
roots such as `runs/` and `ads_projects/` and are not committed.

## Reproduce the paper

Every analytical number and rule the paper states comes from this release. A single command recomputes them
and prints each with the value the paper quotes:

```powershell
python scripts\check_paper_numbers.py
```

| Paper item | Release source | Value |
|---|---|---|
| Delivery threshold | `tvtl.design.feasibility.T_REQ`, recomputed as `m*_D / (k V_bt^max)` | `0.17378` (`m*_D` 0.235625 at 3.3 GHz, `k` 0.4547 /V, `V_bt^max` 2.98197 V) |
| Earlier screening threshold (Fig. 2b,c) | `T_REQ_SCREEN_2026_09` | `0.1438` |
| Transport budgets B and D | `ALPHA_MAX`, gates on `alpha(2 f_s)` and `alpha(f_s)` | `0.02` Np/cell |
| 0 V SMV1245 screen | `coarse_screen(gate="stage0")` | 0 of 2,034 pass delivery and transport; corr(log T, log alpha) +0.88 |
| Phase matching, exceptional point (Results 6, Fig. 6) | `tvtl_ads_builder_v2.analytical.cascade_budget` | at 3.1 GHz: `Delta_theta` 55.55 deg, `kappa/m` 0.0541, `m_EP` 8.96 (model) and 3.99 (measured dispersion) |
| Fig. 2(c) realized fractions | `scripts/two_filter_sweep/task2.py` | 0.6593 / 0.3396 / 0.0904 / 0.0139 |
| Fig. 7 dense map | `scripts/dense_map/dense_map.py` | 140,049 points, 1,776 admissible |
| Certification fields (Methods) | `GainRecord` | definitions in `docs/GAIN_DEFINITIONS.md` |

`scripts/README.md` gives the commands and runtimes. `docs/GAIN_DEFINITIONS.md` gives the definitions, and
`CHANGELOG.md` lists what changed since 1.0.0.

## Used in

*Physics-guided design of varactor-loaded time-varying transmission lines: pump delivery as the binding constraint*, Aiganym Abzaliyeva, Constantinos Valagiannopoulos, and Bakhtiyar Orazbayev; submitted to *Scientific Reports*.

The measured data set is archived separately on Zenodo; its DOI is given in the paper's Data availability statement. That companion data deposit also holds the R1–R7 evidence rounds and maps every paper figure to its evidence file. This repository's `v1.0.1-paper` GitHub release is the code release cited by the paper, archived on Zenodo as [10.5281/zenodo.23315340](https://doi.org/10.5281/zenodo.23315340) (concept DOI for all versions: [10.5281/zenodo.23315339](https://doi.org/10.5281/zenodo.23315339)). The six worked example records under `examples/real_parts/results/` and the committed `results/` tables remain drift-guard inputs in this code release.

## Citation

`CITATION.cff` identifies this software release, with its release date and its Zenodo DOI, [10.5281/zenodo.23315340](https://doi.org/10.5281/zenodo.23315340). The paper that uses it is listed under "Used in" above.

## License

See `LICENSE`.


