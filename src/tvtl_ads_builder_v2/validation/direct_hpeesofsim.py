"""Direct hpeesofsim netlist path for the self-pumped validation circuit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any

from ..circuit_spec import SelfPumpedCircuitSpec, ads_ghz, ads_number


@dataclass(frozen=True)
class HpeesofsimRunResult:
    """Serializable summary for a CircuitSimulator.run_netlist invocation."""

    status: str
    netlist_path: str
    output_dir: str
    dataset_files: list[str]
    output_files: list[str]
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def direct_self_pumped_netlist(spec: SelfPumpedCircuitSpec) -> str:
    """Return the direct ADS HB deck for the minimal self-pumped circuit."""

    fs = ads_ghz(spec.fs_hz)
    zsrc = ads_number(spec.source_z_ohm)
    zline = ads_number(spec.line_z_ohm)
    zload = ads_number(spec.load_z_ohm)
    line_e = ads_number(spec.line_e_deg)
    psig = ads_number(spec.psig_dbm)
    ppump = ads_number(spec.ppump_dbm)
    hb_order = int(spec.hb_order)
    hb_max_order = int(spec.hb_max_order)
    return (
        "; tvtl_ads_builder_v2 direct self-pumped HB deck\n"
        "; same source spec as analytical and ADS OA validation paths\n"
        "\n"
        "; --- same-line sources and passive RF path ---\n"
        f"Port:PORT_SIG  RF_MAIN 0 Num=1 Z={zsrc} Ohm "
        "P[1]=polar(dbmtow(Psig_dBm),0) Freq[1]=fs Noise=no\n"
        f"Port:PORT_PUMP  RF_MAIN 0 Num=2 Z={zsrc} Ohm "
        "P[1]=polar(dbmtow(Ppump_dBm),0) Freq[1]=2*fs Noise=no\n"
        f"TLIN:TL_RF  RF_MAIN N_LOAD Z={zline} Ohm E={line_e} F=fs\n"
        f"Port:TERM_LOAD  N_LOAD 0 Num=3 Z={zload} Ohm\n"
        "\n"
        "; --- analysis ---\n"
        f"HB:HB1 MaxOrder={hb_max_order} Freq[1]=fs Order[1]={hb_order}\n"
        "\n"
        "; --- equations from shared SelfPumpedCircuitSpec ---\n"
        f"fs={fs}\n"
        f"Psig_dBm={psig}\n"
        f"Ppump_dBm={ppump}\n"
    )


def write_direct_netlist(
    spec: SelfPumpedCircuitSpec,
    output_dir: str | Path,
    *,
    filename: str = "self_pumped_direct.ckt",
) -> Path:
    """Write the direct hpeesofsim netlist and return its path."""

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    netlist_path = out_path / filename
    netlist_path.write_text(direct_self_pumped_netlist(spec), encoding="utf-8")
    return netlist_path


def run_direct_hpeesofsim(
    spec: SelfPumpedCircuitSpec,
    output_dir: str | Path,
) -> HpeesofsimRunResult:
    """Run the direct deck with Keysight's CircuitSimulator wrapper."""

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    netlist_path = write_direct_netlist(spec, out_path)
    netlist = netlist_path.read_text(encoding="utf-8")
    before = _mtimes(out_path)
    started = time.time()

    try:
        from keysight.edatoolbox import ads, util

        util.safe_makedirs(str(out_path))
        ads.CircuitSimulator().run_netlist(netlist, output_dir=str(out_path))
    except Exception as exc:  # noqa: BLE001 - report ADS/runtime failures
        return HpeesofsimRunResult(
            status="error",
            netlist_path=str(netlist_path),
            output_dir=str(out_path),
            dataset_files=[],
            output_files=_new_or_updated(out_path, before, started),
            error_message=str(exc),
        )

    output_files = _new_or_updated(out_path, before, started)
    dataset_files = sorted(name for name in output_files if name.endswith(".ds"))
    return HpeesofsimRunResult(
        status="passed" if dataset_files else "error",
        netlist_path=str(netlist_path),
        output_dir=str(out_path),
        dataset_files=dataset_files,
        output_files=output_files,
        error_message=None if dataset_files else "hpeesofsim wrote no .ds dataset",
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
