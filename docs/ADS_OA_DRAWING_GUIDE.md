# ADS OA Drawing Guide

This guide documents the ADS Python/OpenAccess path audited on this machine for
TVTL / floquet-line-design. The architecture is self-pumped: the ADS schematic
must put the signal source at `fs` and the pump source at `2*fs` on the same
RF/main path. Do not create a separate pump, control, or bias lane.

Generated amplifier packages also carry gain-report breadcrumbs in
`OPEN_IN_ADS.txt`, `amplifier_export_manifest.json`, and
`amplifier_schematic_plan.json`. Full equations live in
`docs/GAIN_DEFINITIONS.md` and generated `gain_comparison_report.md` /
`pump_power_sweep_report.md`; the schematic itself should only carry compact
status text, not long formulas.

## Audited Environment

- Install root: `C:\Program Files\Keysight\ADS2026_Update1.2`
- ADS Python: `%HPEESOF_DIR%\tools\python\python.exe`
- Python version reported by that interpreter: `3.13.2`
- Required package path: `%HPEESOF_DIR%\tools\python\packages`
- `keysight.ads.de.product_version()`: `Design Environment (*) 637.update1.2 Jan 22 2026 (64-bit)`
- `keysight.ads.de.version()`: `637`
- Local HTML docs title: `ADS Advanced Design System 2026 Update 1 (635)`
- Wheelhouse includes ADS Python wheels such as `keysight_ads_de-637-cp313-cp313-win_amd64.whl`

The useful local references are:

- `%HPEESOF_DIR%\doc\python\de\examples`
- `%HPEESOF_DIR%\doc\python\de\html\pydocs`
- `%HPEESOF_DIR%\tools\python\packages\keysight\ads\de`

## Execution Model

Use the ADS Python interpreter for standalone automation:

```bat
set HPEESOF_DIR=C:\Program Files\Keysight\ADS2026_Update1.2
"%HPEESOF_DIR%\tools\python\python.exe" tests\ads_oa_smoke.py --simulate
```

Inside the ADS application, open `Tools > Python Console...`. In that context
`keysight.ads.de.is_pde_app()` is `True`, and ADS UI APIs such as
`keysight.ads.de.app` are available.

In standalone automation, this audit measured:

```text
de.running_automation() == True
de.is_pde_app() == False
```

The ADS docs note that `running_automation()` is misnamed: it means the script
is running in a Python application context rather than ADS or DDS application
context. In automation mode, do not use UI-only APIs such as windows, message
boxes, palettes, menus, or toolbars.

Licensing pitfall: importing `keysight.ads.de` checks out Schematic and Layout
licenses for the lifetime of the Python session.

## Workspace, Library, Cell, View

Use this sequence for deterministic automation:

```python
from keysight.ads import de
from keysight.ads.de import db_uu

if de.workspace_is_open():
    de.close_workspace()

de.create_workspace(workspace_path)          # path must not already exist
workspace = de.open_workspace(workspace_path)

de.create_new_library("tvtl_ads_oa_lib", library_path)
workspace.add_library("tvtl_ads_oa_lib", library_path, de.LibraryMode.NON_SHARED)

design = db_uu.create_schematic(("tvtl_ads_oa_lib", "self_pumped_minimal", "schematic"))
with db_uu.Transaction(design, "Build schematic") as transaction:
    ...
    transaction.commit()
design.save_design()
```

If the workspace already exists, use `de.directory_is_workspace(path)` and then
`de.open_workspace(path, force=True)`. If the schematic exists, use
`db_uu.open_design((lib, cell, "schematic"), de.db.DesignMode.WRITE)` and clear
inside a transaction.

Views are ordinary ADS/OA cellviews:

- Schematic: `schematic`, created with `db_uu.create_schematic(...)`
- Symbol: `symbol`, created with `db_uu.create_symbol(...)`
- Layout: `layout`, created with `db_uu.create_layout(...)`

Use `design.save_design()` after the transaction commits. `db_uu.design_saving`
is a convenience context manager that saves on exit, but it does not replace the
need to commit transactions.

## Component Discovery

Do not hard-code unverified component names. Validate every built-in component
before drawing:

```python
assert de.cellview_exists("ads_sources", "P_1Tone", "symbol")
inst = design.add_instance(("ads_sources", "P_1Tone", "symbol"), (0, 0), name="PORT_SIG")
print(list(inst.parameters.keys()))
```

Validated on ADS API version 637:

| Role | Cellview | Important parameters |
| --- | --- | --- |
| Signal and pump source | `ads_sources:P_1Tone:symbol` | `Num`, `Z`, `P`, `Freq`, `Pac`, `FundIndex`, `Vdc` |
| Load port | `ads_simulation:Term:symbol` | `Num`, `Z`, `Noise`, `Vdc`, `Temp` |
| Harmonic balance | `ads_simulation:HarmonicBalance:symbol` | repeated `Freq`, repeated `Order`, `MaxOrder` |
| Variables | `ads_datacmps:VAR:symbol` | `inst.vars[...]` |
| Ground | `ads_rflib:GROUND:symbol` | numbered terminal 1 on `gnd!` |
| Ideal line | `ads_tlines:TLIN:symbol` | `Z`, `E`, `F` |
| Microstrip line | `ads_tlines:MLIN:symbol` | `Subst`, `W`, `L`, `Wall1`, `Wall2` |
| Microstrip open-end effect | `ads_tlines:MLEF:symbol` | `Subst`, `W`, `L`, `Wall1`, `Wall2`, `Temp`, `Mod` |
| Via to ground | `ads_tlines:VIAGND:symbol` | `Subst`, `D`, `W`, `T`, `Rho`, `Cond1Layer`, `HoleLayer`, `ViaLayout` |

## Schematic Drawing Procedure

Think of the OA schematic drawing task as four separate operations:

1. Place validated ADS symbols on a simple coordinate grid.
2. Set every electrical parameter by its exact ADS parameter name.
3. Read each symbol's numbered terminal snap point from ADS.
4. Draw wires between snap points and label only the intentional named nets.

Do not draw by guessing where a pin should be. ADS symbols may have different
pin offsets after rotation or mirroring, and a visually plausible wire can still
miss the real electrical pin. Place the instance first, then ask ADS where each
numbered terminal is.

For the minimal self-pumped circuit, draw the main RF path left to right:

```text
          VAR1(fs, Psig_dBm, Ppump_dBm)     HB1

gnd!            gnd!                         gnd!
 |               |                            |
PORT_SIG ---- PORT_PUMP ---- TL_RF ---- TERM_LOAD
  fs             2*fs          line       50 Ohm
        named RF_MAIN
```

The important drawing rule is that `PORT_SIG` terminal 1 and `PORT_PUMP`
terminal 1 are physically wired onto the same node. That shared node is labeled
`RF_MAIN`. The pump is not drawn as a separate control line; it is a second
`P_1Tone` port injecting `2*fs` into the same RF path. Terminal 2 of each source
and terminal 2 of the load are grounded.

## ADS OA Amplifier Unit-Cell Elements

The active public OA amplifier exporter uses validated ADS distributed
termination elements for unit-cell internals:

- Use `MLEF` for open-end microstrip termination, appended after the MLIN or
  MLIN2 stub section that carries the physical stub length.
- Use `VIAGND` for short/via-to-ground termination. `VIAGND` is a one-terminal
  ADS element that grounds internally, so the schematic should not add a
  separate lumped inductor-to-ground via approximation at that node.

Deprecated or inaccurate ADS implementations are not used in the public OA
amplifier exporter:

- `MLOC` is replaced by validated `MLEF` where open-end microstrip termination
  is needed.
- `LVIA` / lumped inductor-to-ground via emulation is replaced by validated
  `VIAGND` where via-to-ground shorting is needed.
- If `MLEF` or `VIAGND` are unavailable in a local ADS install, the exporter
  reports the limitation instead of silently substituting an unvalidated model.

On ADS 2026 Update 1.2, both `MLEF` and `VIAGND` validated as one-terminal
`ads_tlines` symbols after opening a temporary workspace. The discovered
parameter keys were `L`, `Mod`, `Subst`, `Temp`, `W`, `Wall1`, `Wall2` for
`MLEF`, and `Cond1Layer`, `D`, `HoleLayer`, `PO`, `Pad`, `Rho`, `Subst`, `T`,
`Temp`, `ViaLayout`, `W` for `VIAGND`.

### 1. Place Symbols

Use `design.add_instance` with a library/cell/view tuple that was already
validated with `de.cellview_exists`. The example below uses compact schematic
coordinates in user units; they are not physical layout dimensions.

```python
sig = design.add_instance(
    ("ads_sources", "P_1Tone", "symbol"),
    (0.0, 0.0),
    name="PORT_SIG",
    angle=-90.0,
)
pump = design.add_instance(
    ("ads_sources", "P_1Tone", "symbol"),
    (2.0, 0.0),
    name="PORT_PUMP",
    angle=-90.0,
)
line = design.add_instance(
    ("ads_tlines", "TLIN", "symbol"),
    (3.0, 0.0),
    name="TL_RF",
)
load = design.add_instance(
    ("ads_simulation", "Term", "symbol"),
    (5.0, 0.0),
    name="TERM_LOAD",
    angle=-90.0,
)
```

Keep the schematic sparse and readable. Put simulation/control components such
as `HB1` and `VAR1` above the circuit instead of inline with the RF path:

```python
hb = design.add_instance(("ads_simulation", "HarmonicBalance", "symbol"), (0.0, 2.0), name="HB1")
var = design.add_instance(("ads_datacmps", "VAR", "symbol"), (3.0, 2.0), name="VAR1", angle=90.0)
```

After placing the RF components, place grounds directly at the snap point of
each grounded terminal. This keeps the ground symbol electrically coincident
with the terminal instead of merely close to it:

```python
for index, point in enumerate(
    [pin_snap_point(sig, 2), pin_snap_point(pump, 2), pin_snap_point(load, 2)],
    start=1,
):
    design.add_instance(
        ("ads_rflib", "GROUND", "symbol"),
        (point.x, point.y),
        name=f"GND{index}",
        angle=-90.0,
    )
```

### 2. Set Parameters

Set parameters after placement and before netlisting. Use exact ADS parameter
keys discovered from `inst.parameters.keys()`.

```python
sig.parameters["Num"].value = "1"
sig.parameters["Z"].value = "100 Ohm"
sig.parameters["Freq"].value = "fs"
sig.parameters["P"].value = "polar(dbmtow(Psig_dBm),0)"

pump.parameters["Num"].value = "2"
pump.parameters["Z"].value = "100 Ohm"
pump.parameters["Freq"].value = "2*fs"
pump.parameters["P"].value = "polar(dbmtow(Ppump_dBm),0)"

line.parameters["Z"].value = "50 Ohm"
line.parameters["E"].value = "90"
line.parameters["F"].value = "fs"

load.parameters["Num"].value = "3"
load.parameters["Z"].value = "50 Ohm"
```

For VAR components, write variables through `inst.vars`:

```python
var.vars["fs"] = "1 GHz"
var.vars["Psig_dBm"] = "-20"
var.vars["Ppump_dBm"] = "0"
```

For Harmonic Balance, repeated parameters may need the first repeat slot:

```python
hb.parameters["Freq"].repeats[0].value = "fs"
hb.parameters["Order"].repeats[0].value = "5"
```

### 3. Read Snap Points

Use numbered terminals to find the real pin location. This helper is the safest
way to draw a wire because it works after rotation and uses ADS's own symbol
metadata:

```python
def pin_snap_point(instance, term_number):
    for inst_term in instance.inst_terms:
        if inst_term.is_bound and inst_term.is_numbered and inst_term.term_number == term_number:
            return list(inst_term.inst_pins)[0].snap_point
    raise RuntimeError(f"missing terminal {term_number} on {instance.name}")
```

For this topology, the terminal mapping is:

| Instance | Terminal 1 | Terminal 2 |
| --- | --- | --- |
| `PORT_SIG` | shared `RF_MAIN` node | `gnd!` |
| `PORT_PUMP` | shared `RF_MAIN` node and line input | `gnd!` |
| `TL_RF` | line input from `RF_MAIN` | line output to load |
| `TERM_LOAD` | line output node | `gnd!` |

### 4. Draw Wires and Labels

Only draw wires between snap points returned by ADS:

```python
rf_wire = design.add_wire([pin_snap_point(sig, 1), pin_snap_point(pump, 1)])
rf_wire.add_wire_label("RF_MAIN")
design.add_wire([pin_snap_point(pump, 1), pin_snap_point(line, 1)])
design.add_wire([pin_snap_point(line, 2), pin_snap_point(load, 1)])
```

Label the first shared source-to-pump wire as `RF_MAIN`. Downstream nets can be
left anonymous unless the simulation or report needs a measurement label. Avoid
labels named `PUMP`, `CONTROL`, or `BIAS` because those names imply a separate
lane that this self-pumped topology must not have.

Finish the drawing inside a transaction and commit it:

```python
with db_uu.Transaction(design, "Build schematic") as transaction:
    # place symbols, set parameters, place grounds, draw wires
    transaction.commit()
design.save_design()
```

The transaction commit is part of the drawing process. It lets ADS check and
repair connectivity before the design is saved and netlisted.

## Schematic Connectivity

Always wire from ADS-reported snap points, not guessed offsets:

```python
def pin_snap_point(instance, term_number):
    for inst_term in instance.inst_terms:
        if inst_term.is_bound and inst_term.is_numbered and inst_term.term_number == term_number:
            return list(inst_term.inst_pins)[0].snap_point
    raise RuntimeError(f"missing terminal {term_number} on {instance.name}")

with db_uu.Transaction(design, "Wire schematic") as transaction:
    sig = design.add_instance(("ads_sources", "P_1Tone", "symbol"), (0, 0), name="PORT_SIG", angle=-90)
    pump = design.add_instance(("ads_sources", "P_1Tone", "symbol"), (2, 0), name="PORT_PUMP", angle=-90)
    wire = design.add_wire([pin_snap_point(sig, 1), pin_snap_point(pump, 1)])
    wire.add_wire_label("RF_MAIN")
    transaction.commit()
```

Committing the transaction is important. The ADS examples state that
connectivity edits should be done in a transaction; committing checks and can
repair connectivity. This avoids schematics that look connected but netlist as
separate nets.

Use `angle` for rotation and `mirror` for mirroring in `design.add_instance`.
For instance parameters, set by validated name:

```python
inst.parameters["Freq"].value = "2*fs"
inst.parameters["P"].value = "polar(dbmtow(Ppump_dBm),0)"
hb.parameters["Freq"].repeats[0].value = "fs"
hb.parameters["Order"].repeats[0].value = "5"
```

## Layout Drawing

Prefer `keysight.ads.de.db_uu` for generator code unless you intentionally need
raw integer database units:

- `db_uu` converts geometric values to/from user units.
- `db_dbu` uses database units directly.
- A measured schematic design reported `design.dbu_to_uu_factor == 0.00625`.
- A measured design reported `design.meter_to_uu_factor == 39.37007874015748`.

Useful primitives:

```python
layer = design.create_layer_id("cond")
design.add_path(layer, [(0.0, 0.0), (100.0, 0.0)], 10.0)
design.add_trace(layer, [(0.0, 0.0), (100.0, 0.0)], 10.0)
design.add_polygon(layer, [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])
design.add_rectangle(layer, (0.0, 0.0), (10.0, 10.0))
design.add_text(layer, "label", (0.0, 0.0), "Arial", 0.225, db_uu.TextAlignment.LOWER_LEFT)
```

For layout pins and nets:

```python
net = design.find_or_add_net("P1")
term = design.add_term(net, "P1")
dot = design.add_dot(layer, (0.0, 5.0))
design.add_pin(term, dot)
```

Layer/purpose can be created by name with `design.create_layer_id("cond")` or
with numeric `db_uu.LayerId(layer, purpose)`. Layout shapes alone do not imply
schematic electrical connectivity. Bind pins/terms/nets explicitly when
electrical meaning matters.

## Self-Pumped ADS Topology

The simulation-safe minimal topology is:

```text
gnd!        gnd!         gnd!
 |           |            |
PORT_SIG -- PORT_PUMP -- TL_RF -- TERM_LOAD
  fs          2*fs        RF main path
```

Both sources are `ads_sources:P_1Tone:symbol`, both first numbered terminals
are wired to `RF_MAIN`, and both second numbered terminals are grounded. There
is no `PUMP`, `CONTROL`, or `BIAS` net/lane.

The guide script defaults each P_1Tone source to `Z=100 Ohm`. Because the two
finite source impedances are in parallel at the common RF injection node, their
equivalent source impedance is 50 Ohm. If you set both to 50 Ohm, the common
node sees 25 Ohm and the pump/source loading changes. Do not use ideal voltage
sources in parallel for this topology.

For HB, use a single fundamental `fs` because `2*fs` is the second harmonic:

```python
hb.parameters["Freq"].repeats[0].value = "fs"
hb.parameters["Order"].repeats[0].value = "5"
```

The verified generated netlist contains:

```text
Port:PORT_SIG  RF_MAIN 0 Num=1 Z=100 Ohm P[1]=polar(dbmtow(Psig_dBm),0) Freq[1]=fs
Port:PORT_PUMP  RF_MAIN 0 Num=2 Z=100 Ohm P[1]=polar(dbmtow(Ppump_dBm),0) Freq[1]=2*fs
TLIN:TL_RF  RF_MAIN N__5 Z=50 Ohm E=90 F=fs
Port:TERM_LOAD  N__5 0 Num=3 Z=50 Ohm
HB:HB1 MaxOrder=4 Freq[1]=fs Order[1]=5
```

## Public Validation Workflow

The public repository implements this guide as a validation workflow driven by
one source-of-truth object:

```text
src/tvtl_ads_builder_v2/circuit_spec.py
```

`SelfPumpedCircuitSpec` is consumed by all three paths:

```text
1. analytical passive RF-path check
2. direct hpeesofsim/ADS netlist
3. ADS OA schematic writer and generated netlist
```

Run the ADS-free topology report with normal Python:

```bat
python -m tvtl_ads_builder_v2 validate-self-pumped --freq-ghz 2.0 --out runs\validation_self_pumped_no_ads
```

Run the full OA/netlist/simulation check with ADS Python:

```bat
"%HPEESOF_DIR%\tools\python\python.exe" -m tvtl_ads_builder_v2 validate-self-pumped --freq-ghz 2.0 --out runs\validation_self_pumped_ads --simulate --draw-oa
```

The ADS OA schematic is generated from the same `SelfPumpedCircuitSpec` as the
analytical and direct hpeesofsim circuits. The pump source at `2*fs` is on the
same RF/main line as the signal source at `fs`; no separate pump, control, or
bias lane is used.

## Netlist and Simulation

Netlist OA schematics with:

```python
netlist = design.generate_netlist()
```

The verified automation simulation path is:

```python
from keysight.edatoolbox import ads, util

util.safe_makedirs(output_dir)
ads.CircuitSimulator().run_netlist(netlist, output_dir=output_dir)
```

This wrapper writes a temporary `.ckt` and invokes:

```text
%HPEESOF_DIR%\bin\hpeesofsim <netlist_file>
```

In this audit, the minimal self-pumped netlist simulated and produced:

```text
self_pumped.ds
self_pumped_data
spectra.raw
```

Run the repository smoke test where ADS is installed:

```bat
set HPEESOF_DIR=C:\Program Files\Keysight\ADS2026_Update1.2
"%HPEESOF_DIR%\tools\python\python.exe" tests\ads_oa_smoke.py --simulate
```

Run static tests anywhere:

```bat
python -m pytest tests\test_ads_oa_writer_static.py
```

## Known Pitfalls

- `de.create_workspace(path)` fails if `path` already exists. Use
  `de.directory_is_workspace(path)` and `de.open_workspace(path, force=True)`
  for existing workspaces.
- Importing `keysight.ads.de` consumes Schematic and Layout licenses until the
  Python process exits.
- `de.cellview_exists` takes `(library, cell, view)`, not one colon-separated
  string.
- Component parameter names are case-sensitive. Discover them from
  `inst.parameters.keys()` or `de.ItemInfo`, then set by exact key.
- VAR names are case-sensitive and a VAR instance must always keep at least one
  variable.
- Use transactions for connectivity edits and call `transaction.commit()`.
  A `with` block without commit rolls back.
- Wire to `inst.inst_pins[*].snap_point` or numbered `inst_terms`, not guessed
  symbol coordinates.
- `db_uu` uses user units and converts to database units. `db_dbu` uses raw DBU.
- Layout polygons/paths/traces are shapes. They are not sufficient for
  schematic electrical connectivity unless pins/terms/nets are created and
  bound.
- Gate UI-only code with `de.is_pde_app()`.
- ADS automation cannot use ADS application UI APIs. Use
  `keysight.edatoolbox.ads.CircuitSimulator` for standalone simulation.
- This install prints a non-fatal `vtb.defs` include warning because the
  SystemVue include path is absent. The ADS schematic/netlist/sim flow above
  still completed.

---

**Related:** the user-facing ADS steps and the GUI click path →
[workflow.md](workflow.md#6-opening-an-assembled-workspace-in-ads-gui); reading these
warnings as expected behavior → [troubleshooting.md](troubleshooting.md).
