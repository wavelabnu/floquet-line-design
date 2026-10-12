# Troubleshooting

Most "failures" in this workflow are physics being reported honestly, not bugs.
This page explains how to read them, plus the handful of genuine environment
quirks. Deeper background: [LIMITATIONS.md](LIMITATIONS.md) for what the supplement
does and does not claim, and [ADS_OA_DRAWING_GUIDE.md](ADS_OA_DRAWING_GUIDE.md) for
the ADS Python pitfalls.

---

## HB non-convergence is the oscillation edge

**Symptom.** A `--simulate` run reports a non-convergent harmonic-balance point,
or `analyze_proxy.py` records a non-convergence onset as pump power rises.

**What it means.** This is physical, not numerical noise. When the driven HB solve
stops converging, the device output is diverging with the signal on — the
[parametric-instability / oscillation edge](theory.md#what-hb-non-convergence-means-physically).
The toolchain records it as the data point it is (`non-convergent`), never silently
dropping it.

**What to do.** Nothing, if you wanted to find the edge — that is the answer. If you
want a stable amplifier, back off: the pump-power scan deliberately picks the
**boost-optimal point that still converges**, one step below collapse
([theory §7](theory.md#7-stability-methodology-how-a-stable-operating-point-is-certified)).
Lower the pump power until HB converges again. By default the CLIs treat
non-convergence as recorded data and still exit success (`--allow-nonconvergence`);
pass `--strict-convergence` if you want a nonzero exit instead.

---

## The SMV2020 near-oscillation parity floor

**Symptom.** The `real_oscillator_SMV2020` parity check returns verdict **CLOSE**,
not **IDENTICAL** — flat and hierarchical agree only to about `6e-9 V`, while the
other five designs are bit-identical.

**What it means.** SMV2020 sits at a near-oscillation **regenerative** operating
point ([examples.md](examples.md#real_oscillator_smv2020--the-self-oscillation-regime)).
At a regenerative edge, the loop factor approaches 1 and the gain `≈ G0/(1−L)`
magnifies the tiny numerical differences between the flat and hierarchical solver
paths up to the harmonic-balance **convergence floor** (~6e-9 V). That floating-
point-floor difference is expected; it is not a structural difference in the
circuit. The parity gate accepts it as CLOSE (`max_abs_dV < 1e-6`) rather than
demanding bit-equality there.

---

## "comparison_unavailable" / NaN gain columns

**Symptom.** `compare-gain` on base Python prints `NaN` in the direct-HB and ADS-OA
columns with status `comparison_unavailable`; `sweep-pump-gain` reports
`numerical sweep: analytical_only`.

**What it means.** Expected — there is no ADS dataset because you did not run with
`--simulate`/`--draw-oa` (or ADS is not installed). The analytical column is the
prediction; the HB columns stay `null`/`NaN` until ADS fills them, and are
**never** back-filled from the prediction ([theory §5](theory.md#5-predicted-vs-realized-the-honesty-rule)).
Re-run under ADS Python with `--simulate --draw-oa` to populate them.

---

## ADS runtime not found (`blocked_missing_ads_runtime`)

**Symptom.** A `--simulate` run returns status `blocked_missing_ads_runtime`, or HB
never launches.

**Fix.** The simulator binary was not located. Set `HPEESOF_DIR` to your ADS
install and use the ADS Python interpreter:

```powershell
$env:HPEESOF_DIR = "<ADS_INSTALL_DIR>"
$env:PYTHONPATH  = "src"
& "<ADS_INSTALL_DIR>\tools\python\python.exe" -m tvtl_ads_builder_v2 compare-gain --input ... --simulate --draw-oa
```

On Windows the runner prefers `adssim_win.exe`; calling `hpeesofsim.exe` directly
can fail with `STATUS_DLL_NOT_FOUND` because the ADS `bin` directories are not on
PATH. The runner adds them, which is why it is the supported entry point.

---

## ADS element availability (`MLEF` / `VIAGND`)

**Symptom.** The exporter reports an element limitation instead of drawing a
termination.

**What it means.** The accurate distributed terminations are the ADS `MLEF`
(open-end fringe) and `VIAGND` (physical via). If they are unavailable in your
local install, the exporter **reports the limitation rather than substituting an
unvalidated model** ([GAIN_DEFINITIONS.md](GAIN_DEFINITIONS.md#ads-element-policy)).
The deprecated `MLOC` (open) and `LVIA` (lumped via) stand-ins exist only for the
legacy/anchor configs and warn on instantiation — do not use them for accurate
work. On ADS 2026 Update 1.2 both `MLEF` and `VIAGND` validate as one-terminal
`ads_tlines` symbols.

---

## Collinear wires and the flat-schematic fork collapse

**Symptom 1.** Exporting a scaffold prints, e.g.,
`layout: P_1Tone sources drive 2 distinct nets ... grid fallback`.

**Symptom 2.** A COMBINED `MLEF ∥ VIAGND` design looks merged in a flat OpenAccess
schematic.

**What it means.** Both come from ADS recomputing connectivity from pin geometry.
With a source present, ADS can merge a parallel fork or collapse collinear/distinct
source nets, so the flat single-sheet emitter falls back to a grid layout and the
MAVR fork collapses. The fix already in the repo is the **hierarchical emitter**:
the `unit_cell` subcircuit carries no source, so ADS does not recompute its
connectivity and the fork renders correctly
([examples.md](examples.md#real_amplifier_mavr--a-stable-high-gain-amplifier)).
Use `build_examples.py assemble` (hierarchical) for COMBINED designs; the flat deck
remains valid as the parity reference.

---

## Benign warnings you can ignore

- **`vtb.defs` include warning** — this ADS install prints a non-fatal `vtb.defs`
  include warning because the SystemVue include path is absent. The schematic,
  netlist, and simulation flow still complete.
- **Layout-layer / purpose warnings** during OA drawing — cosmetic; they do not
  affect netlist connectivity, which is established from pins/terms/nets, not
  layout shapes.
- **`P_1Tone` power token** — source power is written as
  `P=polar(dbmtow(<dBm>),0)`; that is the validated ADS convention, not an error.

---

## Windows file locks on `runs/` and `outputs/`

**Symptom.** A re-run fails to write a workspace or dataset under `runs\`,
`ads_projects\`, or `examples\real_parts\outputs\` with a permission/lock error.

**Fix.** An open ADS session (or a dataset still open in the data display) holds a
lock on the generated files. Close the workspace/dataset in ADS, or write to a
fresh `--out` directory. These directories are git-ignored and meant to be
regenerated, so deleting and rebuilding them is always safe.

---

## Licensing note

Importing `keysight.ads.de` checks out Schematic and Layout licenses for the
lifetime of the Python process. Long-lived automation holds those licenses until
it exits — run ADS-dependent steps as discrete scripts and let them finish rather
than keeping one interpreter open. See
[ADS_OA_DRAWING_GUIDE.md](ADS_OA_DRAWING_GUIDE.md#known-pitfalls) for the full list.

---

## Still stuck?

- Is this a base-Python step or an ADS step? Re-check the table in
  [workflow.md §1](workflow.md#what-needs-ads-and-what-does-not).
- Is the number predicted or realized? A predicted number that looks too good is
  expected — it is the optimistic ranking proxy
  ([theory §5](theory.md#5-predicted-vs-realized-the-honesty-rule)).
- What the supplement intentionally omits (private diagnostics, generated
  archives, optional calibration data) → [LIMITATIONS.md](LIMITATIONS.md).
