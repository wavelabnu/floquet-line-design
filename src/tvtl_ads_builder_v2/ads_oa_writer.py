"""ADS OpenAccess writer for the minimal self-pumped validation schematic.

Keysight ADS modules are imported lazily so normal Python can import this
module and run static tests without ADS installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from .circuit_spec import SelfPumpedCircuitSpec, ads_ghz, ads_number


@dataclass(frozen=True)
class ADSComponentRef:
    """A library/cell/view reference validated with ``de.cellview_exists``."""

    library: str
    cell: str
    view: str = "symbol"

    def as_lcv(self) -> tuple[str, str, str]:
        return (self.library, self.cell, self.view)

    def as_string(self) -> str:
        return f"{self.library}:{self.cell}:{self.view}"


@dataclass(frozen=True)
class ADSBuildResult:
    """Serializable summary from an ADS OA schematic build."""

    workspace_path: str
    library_name: str
    cell_name: str
    design_name: str
    product_version: str
    api_version: int
    running_automation: bool
    is_pde_app: bool
    netlist_path: str
    output_dir: str | None
    simulation_output_files: list[str]
    required_cellviews: dict[str, str]
    net_names: list[str]
    instance_cells: dict[str, str]
    rf_net: str = "RF_MAIN"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


REQUIRED_COMPONENTS: dict[str, ADSComponentRef] = {
    "signal_source": ADSComponentRef("ads_sources", "P_1Tone"),
    "pump_source": ADSComponentRef("ads_sources", "P_1Tone"),
    "load_term": ADSComponentRef("ads_simulation", "Term"),
    "harmonic_balance": ADSComponentRef("ads_simulation", "HarmonicBalance"),
    "var": ADSComponentRef("ads_datacmps", "VAR"),
    "ground": ADSComponentRef("ads_rflib", "GROUND"),
    "main_line": ADSComponentRef("ads_tlines", "TLIN"),
}


def self_pumped_schematic_contract() -> dict[str, Any]:
    """Pure-Python contract used by static tests and docs."""

    return {
        "architecture": "self-pumped",
        "rf_net": "RF_MAIN",
        "source_instances": ["PORT_SIG", "PORT_PUMP"],
        "source_cell": "ads_sources:P_1Tone:symbol",
        "signal_frequency": "fs",
        "pump_frequency": "2*fs",
        "forbidden_separate_lanes": ["PUMP", "CONTROL", "BIAS"],
        "snap_point_wiring": True,
        "transaction_commit": True,
    }


def import_ads_modules(ads_home: str | os.PathLike[str] | None = None):
    """Import and return ``(de, db_uu)`` from ADS Python."""

    if ads_home is not None:
        os.environ.setdefault("HPEESOF_DIR", str(Path(ads_home)))

    try:
        from keysight.ads import de
        from keysight.ads.de import db_uu

        return de, db_uu
    except ModuleNotFoundError:
        hpeesof_dir = os.environ.get("HPEESOF_DIR")
        if not hpeesof_dir:
            raise RuntimeError(
                "keysight.ads.de is unavailable and HPEESOF_DIR is not set. "
                "Run with ADS Python or set HPEESOF_DIR."
            )
        package_dir = Path(hpeesof_dir) / "tools" / "python" / "packages"
        if str(package_dir) not in sys.path:
            sys.path.insert(0, str(package_dir))
        from keysight.ads import de
        from keysight.ads.de import db_uu

        return de, db_uu


def validate_component_refs(
    de: Any,
    refs: dict[str, ADSComponentRef],
) -> dict[str, str]:
    """Validate every built-in component cellview before drawing."""

    missing: list[str] = []
    found: dict[str, str] = {}
    for role, ref in refs.items():
        if not de.cellview_exists(ref.library, ref.cell, ref.view):
            missing.append(f"{role}={ref.as_string()}")
        else:
            found[role] = ref.as_string()
    if missing:
        raise RuntimeError(
            "ADS component discovery failed; missing required cellviews: "
            + ", ".join(missing)
        )
    return found


def _open_or_create_workspace(
    de: Any,
    workspace_path: Path,
    library_name: str,
) -> tuple[Any, Any]:
    workspace_path = workspace_path.resolve()
    if de.workspace_is_open():
        de.close_workspace()

    if workspace_path.exists():
        if not de.directory_is_workspace(str(workspace_path)):
            raise RuntimeError(
                f"Path exists but is not an ADS workspace: {workspace_path}"
            )
        workspace = de.open_workspace(str(workspace_path), force=True)
    else:
        de.create_workspace(str(workspace_path))
        workspace = de.open_workspace(str(workspace_path))

    library_path = workspace_path / library_name
    if library_name in set(workspace.library_names):
        library = workspace.open_library(library_name)
    else:
        if not de.library_exists_at_path(str(library_path)):
            de.create_new_library(library_name, str(library_path))
        library = workspace.add_library(
            library_name, str(library_path), de.LibraryMode.NON_SHARED
        )
    return workspace, library


def _pin_snap_point(instance: Any, term_number: int) -> Any:
    """Return a numbered terminal snap point from ``inst.inst_terms``."""

    for inst_term in instance.inst_terms:
        if not inst_term.is_bound or not inst_term.is_numbered:
            continue
        if inst_term.term_number != term_number:
            continue
        pins = list(inst_term.inst_pins)
        if not pins or pins[0].snap_point is None:
            raise RuntimeError(
                f"Instance {instance.name} term {term_number} has no snap point."
            )
        return pins[0].snap_point
    raise RuntimeError(
        f"Instance {instance.name} has no numbered terminal {term_number}."
    )


def _set_param(instance: Any, name: str, value: str) -> None:
    if name not in instance.parameters:
        available = ", ".join(instance.parameters.keys())
        raise RuntimeError(
            f"{instance.name} has no parameter {name!r}. Available: {available}"
        )
    instance.parameters[name].value = value


def _set_first_repeated_param(instance: Any, name: str, value: str) -> None:
    if name not in instance.parameters:
        available = ", ".join(instance.parameters.keys())
        raise RuntimeError(
            f"{instance.name} has no parameter {name!r}. Available: {available}"
        )
    param = instance.parameters[name]
    if hasattr(param, "repeats") and len(param.repeats) > 0:
        param.repeats[0].value = value
    else:
        param.value = [value]


def _create_or_clear_schematic(
    de: Any,
    db_uu: Any,
    library_name: str,
    cell_name: str,
) -> Any:
    design_ref = (library_name, cell_name, "schematic")
    if de.cellview_exists(library_name, cell_name, "schematic"):
        design = db_uu.open_design(design_ref, de.db.DesignMode.WRITE)
    else:
        design = db_uu.create_schematic(design_ref)
    return design


def build_self_pumped_minimal_schematic(
    workspace_path: str | os.PathLike[str],
    *,
    spec: SelfPumpedCircuitSpec | None = None,
    library_name: str = "tvtl_ads_oa_lib",
    cell_name: str = "self_pumped_minimal",
    simulate: bool = False,
    ads_home: str | os.PathLike[str] | None = None,
) -> ADSBuildResult:
    """Create, netlist, and optionally simulate the OA self-pumped circuit.

    The schematic is a single RF/main path:

    ``PORT_SIG(fs) -- PORT_PUMP(2*fs) -- TL_RF -- TERM_LOAD``.

    Both ``P_1Tone`` source terminal-1 pins are wired to ``RF_MAIN`` and both
    terminal-2 pins are grounded. No separate pump, control, or bias lane is
    drawn.
    """

    spec = spec or SelfPumpedCircuitSpec()
    de, db_uu = import_ads_modules(ads_home)
    workspace, _library = _open_or_create_workspace(
        de, Path(workspace_path), library_name
    )
    discovered = validate_component_refs(de, REQUIRED_COMPONENTS)
    design = _create_or_clear_schematic(de, db_uu, library_name, cell_name)

    with db_uu.Transaction(design, "Build TVTL self-pumped schematic") as transaction:
        design.clear_design()

        sig = design.add_instance(
            REQUIRED_COMPONENTS["signal_source"].as_lcv(),
            (0.0, 0.0),
            name="PORT_SIG",
            angle=-90.0,
        )
        pump = design.add_instance(
            REQUIRED_COMPONENTS["pump_source"].as_lcv(),
            (2.0, 0.0),
            name="PORT_PUMP",
            angle=-90.0,
        )
        line = design.add_instance(
            REQUIRED_COMPONENTS["main_line"].as_lcv(),
            (3.0, 0.0),
            name="TL_RF",
            angle=0.0,
        )
        load = design.add_instance(
            REQUIRED_COMPONENTS["load_term"].as_lcv(),
            (5.0, 0.0),
            name="TERM_LOAD",
            angle=-90.0,
        )

        for index, point in enumerate(
            [
                _pin_snap_point(sig, 2),
                _pin_snap_point(pump, 2),
                _pin_snap_point(load, 2),
            ],
            start=1,
        ):
            design.add_instance(
                REQUIRED_COMPONENTS["ground"].as_lcv(),
                (point.x, point.y),
                name=f"GND{index}",
                angle=-90.0,
            )

        hb = design.add_instance(
            REQUIRED_COMPONENTS["harmonic_balance"].as_lcv(),
            (0.0, 2.0),
            name="HB1",
        )
        var = design.add_instance(
            REQUIRED_COMPONENTS["var"].as_lcv(),
            (3.0, 2.0),
            name="VAR1",
            angle=90.0,
        )

        var.vars["fs"] = ads_ghz(spec.fs_hz)
        var.vars["Psig_dBm"] = ads_number(spec.psig_dbm)
        var.vars["Ppump_dBm"] = ads_number(spec.ppump_dbm)
        try:
            del var.vars["X"]
        except Exception:
            pass

        _set_param(sig, "Num", "1")
        _set_param(sig, "Z", f"{ads_number(spec.source_z_ohm)} Ohm")
        _set_param(sig, "Freq", "fs")
        _set_param(sig, "P", "polar(dbmtow(Psig_dBm),0)")

        _set_param(pump, "Num", "2")
        _set_param(pump, "Z", f"{ads_number(spec.source_z_ohm)} Ohm")
        _set_param(pump, "Freq", "2*fs")
        _set_param(pump, "P", "polar(dbmtow(Ppump_dBm),0)")

        _set_param(load, "Num", "3")
        _set_param(load, "Z", f"{ads_number(spec.load_z_ohm)} Ohm")

        _set_param(line, "Z", f"{ads_number(spec.line_z_ohm)} Ohm")
        _set_param(line, "E", ads_number(spec.line_e_deg))
        _set_param(line, "F", "fs")

        _set_param(hb, "MaxOrder", str(int(spec.hb_max_order)))
        _set_first_repeated_param(hb, "Freq", "fs")
        _set_first_repeated_param(hb, "Order", str(int(spec.hb_order)))

        rf_wire = design.add_wire([_pin_snap_point(sig, 1), _pin_snap_point(pump, 1)])
        rf_wire.add_wire_label("RF_MAIN")
        design.add_wire([_pin_snap_point(pump, 1), _pin_snap_point(line, 1)])
        design.add_wire([_pin_snap_point(line, 2), _pin_snap_point(load, 1)])

        transaction.commit()

    design.save_design()
    netlist = design.generate_netlist()
    netlist_path = Path(workspace.path) / f"{cell_name}.ckt"
    netlist_path.write_text(netlist, encoding="utf-8")

    _assert_self_pumped_netlist(netlist)

    output_dir: str | None = None
    output_files: list[str] = []
    if simulate:
        from keysight.edatoolbox import ads, util

        output_path = Path(workspace.path) / "output"
        util.safe_makedirs(str(output_path))
        before = _mtimes(output_path)
        started = time.time()
        ads.CircuitSimulator().run_netlist(netlist, output_dir=str(output_path))
        output_dir = str(output_path)
        output_files = _new_or_updated(output_path, before, started)

    instance_cells = {instance.name: instance.cell_name for instance in design.instances}
    net_names = sorted(str(net.name) for net in design.nets if str(net.name))

    return ADSBuildResult(
        workspace_path=str(workspace.path),
        library_name=library_name,
        cell_name=cell_name,
        design_name=f"{library_name}:{cell_name}:schematic",
        product_version=de.product_version().strip(),
        api_version=de.version(),
        running_automation=de.running_automation(),
        is_pde_app=de.is_pde_app(),
        netlist_path=str(netlist_path),
        output_dir=output_dir,
        simulation_output_files=output_files,
        required_cellviews=discovered,
        net_names=net_names,
        instance_cells=instance_cells,
    )


def _mtimes(path: Path) -> dict[str, float]:
    return {entry.name: entry.stat().st_mtime for entry in path.glob("*")}


def _new_or_updated(path: Path, before: dict[str, float], started: float) -> list[str]:
    out: list[str] = []
    for entry in path.glob("*"):
        previous = before.get(entry.name)
        current = entry.stat().st_mtime
        if previous is None or current != previous or current >= started:
            out.append(entry.name)
    return sorted(out)


def _assert_self_pumped_netlist(netlist: str) -> None:
    if "Port:PORT_SIG  RF_MAIN 0" not in netlist:
        raise RuntimeError("Signal P_1Tone source is not netlisted on RF_MAIN.")
    if "Port:PORT_PUMP  RF_MAIN 0" not in netlist:
        raise RuntimeError("Pump P_1Tone source is not netlisted on RF_MAIN.")
    if "Freq[1]=fs" not in netlist:
        raise RuntimeError("Signal source/HB fundamental does not use fs.")
    if "Freq[1]=2*fs" not in netlist:
        raise RuntimeError("Pump source is not set to 2*fs in the netlist.")
    forbidden = {"PUMP", "CONTROL", "BIAS"}
    tokens = {token.strip(" ,;()") for token in netlist.replace("\n", " ").split()}
    bad = sorted(forbidden & tokens)
    if bad:
        raise RuntimeError(f"Forbidden separate lane net(s) in netlist: {bad}")
