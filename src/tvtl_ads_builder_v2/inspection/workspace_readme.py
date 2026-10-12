"""Per-workspace ``inspection_README.md`` writer.

Each canonical-circuit workspace gets a human-readable README: the
circuit topology, the device summary read from the device spec, the
parity verdict, the list of files actually present in the directory,
ADS-import instructions, the expected sweep behaviour, and any
circuit-specific caveat. It is the entry point a reviewer opens to
confirm a generated circuit matches intent.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..device_spec import DeviceSpec
from .metadata import CircuitInspectionMetadata, metadata_for


def write_workspace_readme(
    workspace_dir: str | Path,
    spec: DeviceSpec,
    result: Mapping[str, Any],
    *,
    png_written: bool,
) -> Path:
    """Write ``inspection_README.md`` into a canonical-circuit workspace.

    Parameters
    ----------
    workspace_dir : str or Path
        The workspace directory for this circuit.
    spec : DeviceSpec
        The device spec for the circuit (device summary source).
    result : Mapping
        The parity-check result dict for the circuit.
    png_written : bool
        Whether ``comparison_plot.png`` was generated (matplotlib
        present). When ``False`` the README says the PNG is
        unavailable and points at the CSV.

    Returns
    -------
    Path
        The path of the written README.
    """

    workspace_dir = Path(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    meta = metadata_for(spec.label)

    lines: list[str] = []
    lines.append(f"# {spec.label}\n\n")
    lines.append(f"**Level**: {meta.level}\n\n")
    lines.append(f"**Topology**: {meta.topology_description}\n\n")
    tolerance_dB = float(result.get("tolerance_dB", float("nan")))
    lines.append(f"**Parity tolerance applied**: {tolerance_dB:g} dB\n\n")
    lines.append(f"**Parity result**: {_verdict_line(result)}\n\n")

    lines.append("## Device summary\n\n")
    params = spec.substrate.parameters
    lines.append(
        f"- Substrate: {spec.substrate.name}, eps_r = "
        f"{float(params['eps_r']):g}, h = {float(params['h_m']) * 1e3:g} mm\n"
    )
    lines.append("- Elements:\n")
    for element in spec.elements:
        lines.append(f"  - {element.id} ({element.type}): {_element_summary(element)}\n")
    port_bits = ", ".join(
        f"{port.node} (Num={port.num})" for port in sorted(spec.ports, key=lambda p: p.num)
    )
    lines.append(f"- Ports: {port_bits}\n")
    if spec.ground_nodes:
        lines.append(f"- Ground nodes: {', '.join(spec.ground_nodes)}\n")
    lines.append("\n")

    lines.append("## Files in this directory\n\n")
    for entry in _file_listing(workspace_dir, png_written):
        lines.append(f"- {entry}\n")
    lines.append("\n")

    netlist_name = Path(str(result.get("netlist_path", "hpeesofsim.ckt"))).name
    lines.append("## How to inspect in ADS\n\n")
    lines.append("1. Launch ADS.\n")
    lines.append("2. `File -> Import -> ADS Netlist` (or the equivalent path in your ADS version).\n")
    lines.append(f"3. Select `{netlist_name}` in this directory.\n")
    lines.append("4. ADS will construct a schematic; open it in the schematic editor.\n")
    lines.append(
        "5. Compare the schematic against the topology description above to "
        "confirm the generated circuit matches intent.\n\n"
    )

    lines.append("## Expected behavior\n\n")
    lines.append(meta.expected_behavior + "\n\n")

    lines.append("## Notes\n\n")
    if meta.notes:
        lines.append(meta.notes + "\n")
    else:
        lines.append("(no circuit-specific caveats)\n")

    path = workspace_dir / "inspection_README.md"
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _verdict_line(result: Mapping[str, Any]) -> str:
    """Render the one-line parity verdict for the README header."""

    reason = str(result.get("reason", ""))
    if reason == "ads_blocked":
        return "TVTL only (ADS unavailable) -- parity not evaluated"
    if reason == "tvtl_error":
        return "TVTL error -- parity not evaluated"
    passed = bool(result.get("pass", False))
    verdict = "PASS" if passed else "FAIL"
    max_abs = float(result.get("max_abs_delta_dB", float("nan")))
    freqs = np.asarray(result.get("frequencies_Hz", []), dtype=float)
    delta = np.asarray(result.get("delta_dB_grid", []), dtype=float)
    where = ""
    if freqs.size and delta.size and np.any(np.isfinite(delta)):
        idx = int(np.nanargmax(np.abs(delta)))
        where = f" at {freqs[idx] / 1e9:.3f} GHz"
    max_str = "NaN" if not math.isfinite(max_abs) else f"{max_abs:.3f}"
    return f"{verdict} with max |delta_S21| = {max_str} dB{where}"


def _element_summary(element: Any) -> str:
    """Compact one-line parameter summary for one element."""

    p = element.parameters
    if element.type == "TermG":
        return f"Z = {float(p.get('z_ohm', 50.0)):g} Ohm, Num = {int(p.get('num', 0))}"
    if element.type == "MLIN":
        return (
            f"W = {float(p['w_m']) * 1e3:g} mm, L = {float(p['l_m']) * 1e3:g} mm"
        )
    if element.type == "MTEE":
        return (
            f"W1 = {float(p['w1_m']) * 1e3:g} mm, "
            f"W2 = {float(p['w2_m']) * 1e3:g} mm, "
            f"W3 = {float(p['w3_m']) * 1e3:g} mm"
        )
    return ", ".join(f"{k} = {v}" for k, v in p.items())


def _file_listing(workspace_dir: Path, png_written: bool) -> list[str]:
    """Return README bullet lines for the files present in the workspace."""

    known = {
        "device_spec.json": "canonical device-spec definition",
        "tvtl_s_parameters.csv": "TVTL analytical |S21|/|S11| across the frequency grid",
        "comparison_plot.csv": "TVTL vs ADS |S21| and |delta_dB| (plot data as CSV)",
    }
    entries: list[str] = []
    for name, description in known.items():
        if (workspace_dir / name).exists():
            entries.append(f"`{name}` -- {description}")
    for ckt in sorted(workspace_dir.glob("*.ckt")):
        entries.append(
            f"`{ckt.name}` -- flat hpeesofsim netlist (also ADS "
            "schematic-importable; see below)"
        )
    for ds in sorted(workspace_dir.glob("*.ds")):
        entries.append(f"`{ds.name}` -- ADS hpeesofsim dataset")
    if png_written and (workspace_dir / "comparison_plot.png").exists():
        entries.append(
            "`comparison_plot.png` -- overlay of TVTL vs ADS |S21| with "
            "|delta_dB| on a second panel"
        )
    else:
        entries.append(
            "`comparison_plot.png` -- NOT generated (matplotlib is not "
            "installed; `comparison_plot.csv` carries the same data)"
        )
    return entries


