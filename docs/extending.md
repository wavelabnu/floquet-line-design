# Extending TVTL

The `tvtl` physics core and the `tvtl_ads_builder_v2` ADS builder are generic
production code; the shipped designs are **data** on top of them. So extending the
work usually means adding data — a new device, a new frozen design, a new analysis
— not changing the engine. This page covers the four most common extensions.

**Build on the stable surface.** Prefer the documented package-level and
subpackage exports listed in [STABILITY.md](../STABILITY.md); internal modules can
change between releases. The unsupported-construct rule below is the one
engine-level constraint you must respect.

---

## 1. Add a varactor (frozen literals — the palette pattern)

Commercial parts live in `src/tvtl/varactor_palette.py` as frozen `PaletteEntry`
records. The discipline is strict and is the point of the pattern:

> **Every numeric value is transcribed verbatim from the cited datasheet. Nothing
> is recalled, approximated, or back-filled from a family default. Any parameter
> the vendor does not publish is `None` and listed in `unknown_flags` — never
> guessed.**

Add an entry to the `PALETTE` mapping:

```python
# src/tvtl/varactor_palette.py
PaletteEntry(
    part_number="SMVxxxx-079LF",
    device_class="Si hyperabrupt",        # or "Si abrupt" | "GaAs hyperabrupt"
    vendor="Skyworks",
    cj0_pf=<from datasheet>, vj_v=<from datasheet>, m=<from datasheet>,
    cp_pf=<from datasheet>, r_series_ohm=<from datasheet>, l_package_nh=<from datasheet>,
    cv_table={0.0: <pF>, 2.0: <pF>, ...},  # datasheet C(V) table, or None
    v_bias_v=<nominal reverse bias>,
    source="<vendor doc number + table>",
    cv_source="<doc + table>", law_ok_within_5pct=None, law_max_dev_pct=None,
    unknown_flags=(),                      # name every unpublished field here
)
```

Then consume it:

```python
from tvtl.varactor_palette import get_entry
entry  = get_entry("SMVxxxx-079LF")
device = entry.to_realistic_varactor(v_bias_v=4.0)   # SI RealisticVaractor the engine uses
```

The C–V law is the hyperabrupt power law `C(v) = CJ0 / (1 + v/VJ)^M`
([theory §2](theory.md#capacitance-modulation)). The SPICE parameters are
bias-independent — `to_realistic_varactor(v_bias_v=…)` just moves the operating
point. If a part is published only as a **polynomial** C–V fit, carry
`cv_poly_coeffs` plus a `cv_poly_valid_max_v` ceiling; the guarded accessor uses
the polynomial only inside its validity window and falls back to the junction law
beyond it (the fits go non-physical past their window).

**No datasheet, just a device?** Skip the palette and build a device directly with
`tvtl_ads_builder_v2.models.varactor`: `build_power_law_varactor(...)` for a
CJ0/VJ/M device, or `PolynomialVaractor` from C(V)/Q(V) coefficients. The same
coefficients feed both the ADS `SDD2P` charge body and the analytical small-signal
leg — one definition, never two.

---

## 2. Register it as a named chain model (model registry)

`configs/model_registry/*.yaml` names a complete chain model so a run can select
it by name. A config has three axes:

```yaml
varactor:
  kind: power_law        # power_law | effective_c | polynomial
  CJ0_f: 0.3e-12
  VJ_v: 3.5
  M: 1.7
  CP_f: 0.0
  L_pkg_h: 0.3e-9
  R_series_ohm: 0.5
  V_bias_v: 0.0
via:
  kind: viagnd           # lumped_L | rlc | viagnd  (viagnd = distributed/physical)
  via_radius_m: 0.8e-3
open_end:
  kind: mlef             # ideal | mloc | mlef      (mlef = distributed open-end fringe)
```

Copy `standard.yaml` (the accurate distributed-termination config — `VIAGND` +
`MLEF`) and swap the `varactor` block to your device. `reduced.yaml` is the
simpler lumped-via / ideal-open anchor; the lumped/ideal terminations are
**deprecated** (still functional, but they warn on instantiation). Prefer
`viagnd` + `mlef` for any new accurate work.

---

## 3. Add a topology family

Families are enumerated in `src/tvtl/optimization/topology_generator.py` as
`TopologyFamily(n_segments, via, termination)`:

- `Termination ∈ {OPEN, SHORT, COMBINED}`, `ViaConfig ∈ {NONE, SINGLE, DOUBLE}`.
- Only **five** `(via, termination)` pairs are physical (a via only means a short;
  an open end has no via). Five combos × `n_segments ∈ {1,2,3,4}` = **20 families**
  = **12 linear (OPEN+SHORT) + 8 COMBINED**. The optimizer searches the 12 linear
  families; the 8 COMBINED ones are analytical-screening-only today (the parallel
  via∥open-stublet tree is not yet traced by the linear ADS chain-walker — a
  documented limitation, not a silent gap).
- The continuous per-segment box is `WIDTH 0.2–5.0 mm`, `LENGTH 0.5–15 mm`, gated to
  a `15–130 Ω` realizable impedance band.

For most new designs you do **not** need a new enum value — pick an existing family
and freeze its per-segment `(width, length)` parameters (Section 4). To add a
genuinely new termination or segment behavior, extend `Termination`/`ViaConfig`,
the `_valid_via_term` gate, the segment builders (`stub_segments`,
`combined_open_segment`), and `topology_y_stub_fn`, and confirm the ADS emitter can
render it before claiming a realized number.

---

## 4. Freeze a worked design (the example pattern)

The template is `examples/real_parts/real_part_designs.py`. `build_fixed_geom`
takes a fixed geometry and produces a self-pumped cascade spec:

```python
geom = {
    "topology": TopologyFamily(n_segments=3, via=ViaConfig.SINGLE, termination=Termination.SHORT),
    "stub_params": (w0, l0, w1, l1, w2, l2),   # metres, per segment
    "main_width_m": 2.49e-3,
    "main_period_m": 6.0e-3,
}
spec = build_fixed_geom(geom, device, f_s=3.0e9)   # matches Z_TL + synthesises I/O matching
```

Internally it matches the main-line impedance (Stage 2 `match_z_tl`), synthesizes
input/output matching from the cascade edge impedances (Stage 3 `edge_impedance` +
`select_matching`), and emits through
`tvtl_ads_builder_v2.parity.self_pumped_design.optimized_self_pumped_spec` with the
accurate `ViagndVia` / `MlefOpen` termination models. Add your design to the
`EXAMPLES` list and record its harmonic-balance parity reference in
`REFERENCE_VOUT_SIGNAL` so `build_examples.py` can gate it. `proxy_designs.py` is
the same pattern for optimizer-picked frozen designs.

---

## 5. Rerun or add an analysis (`analyze_proxy.py` as the template)

`examples/real_parts/analyze_proxy.py` is the reusable predicted-vs-realized +
stability driver. Per design it:

1. runs a **pump-power scan** at phase 0 to find the convergent, boost-optimal
   operating point and the non-convergence (oscillation) onset;
2. sweeps **pump phase** at that power and checks for a bounded PSA lobe;
3. reports the analytical **prediction** next to the ADS-realized **boost** and
   absolute out/in — in separate fields, never merged;
4. confirms flat-vs-hierarchical **parity** at the operating point.

Copy it, point it at your spec builder, and keep the predicted/realized fields
separate (a missing realized number stays `null` with a note — never back-filled
from the prediction). Run it under ADS Python:

```powershell
$env:PYTHONPATH = "src"
& $env:ADS_PYTHON examples\real_parts\analyze_proxy.py all
```

This is the same two-sweep
[stability methodology](theory.md#7-stability-methodology-how-a-stable-operating-point-is-certified)
the shipped examples use.

---

## The one engine constraint: resolve scalars before emission

The v2 builder **resolves named scalar variables to numbers before it emits**. It
deliberately does **not** emit ADS `VAR` / `Vars` / `Eqn` or OpenAccess `VarEqn`
constructs, because those forms were unreliable across the ADS version used for the
paper work ([LIMITATIONS.md](LIMITATIONS.md)). So pass **resolved numeric values**
into your spec, not symbolic ADS variables. And remember that `SDD2P` charge
expressions still require local ADS validation before you make a final gain,
stability, or convergence claim — the analytical side is a prediction until HB
records it.

---

## Where to go next

- The stable public API you are building on → [STABILITY.md](../STABILITY.md)
- The physics your extension must respect → [theory.md](theory.md)
- How the optimizer ranks candidates → [OPTIMIZER_METHODOLOGY.md](methodology/OPTIMIZER_METHODOLOGY.md)
- Building and validating what you added → [workflow.md](workflow.md)
