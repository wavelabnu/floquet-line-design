"""Inspection / visual-review artifacts for ADS Builder v2.

The parity gate produces machine outputs (a summary text, a results
CSV, per-circuit workspaces). This package layers human-reviewable
artifacts on top of them, level-agnostically, so every present and
future ladder level inherits the same inspection surface:

* ``inspection_README.md`` per circuit workspace -- topology, device
  summary, parity verdict, file listing, ADS-import steps, expected
  behaviour and caveats.
* ``comparison_plot.csv`` per circuit workspace (always) and
  ``comparison_plot.png`` (when ``matplotlib`` is importable) -- the
  TVTL-vs-ADS |S21| overlay.
* a polished leading comment header on each emitted ``.ckt`` deck
  (presentational only; the simulated circuit is untouched).
* ``inspection_index.md`` + ``inspection_status.json`` per level.
* ``ads_builder_v2_inspection_index.md`` -- the cross-level index,
  rebuilt monotonically from the on-disk per-level status files.
* a per-circuit ADS schematic builder -- an element-placement AEL
  deck (``build_workspace.ael``), an ``open_workspace.bat`` launcher
  and an ADS-native ``.net`` netlist -- plus a per-level
  ``ads_inspection_workspace/README.md``.

:func:`emit_inspection_artifacts` is the single public entry point;
each level's parity CLI calls it once, after the parity check
completes. The PNG is best-effort (``matplotlib`` is imported lazily);
the CSV always carries the same data, so the artifacts are complete even
when the plot is skipped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..device_spec import DeviceSpec
from .ael_templates import render_ael_script
from .comparison_plot import write_comparison_plot
from .global_index import update_global_index
from .level_index import write_level_index
from .metadata import (
    CIRCUIT_INSPECTION_METADATA,
    CircuitInspectionMetadata,
    LEVEL_DESCRIPTIONS,
    metadata_for,
)
from .polish import netlist_element_graph, polish_netlist
from .workspace_readme import write_workspace_readme

__all__ = [
    "CIRCUIT_INSPECTION_METADATA",
    "CircuitInspectionMetadata",
    "LEVEL_DESCRIPTIONS",
    "emit_inspection_artifacts",
    "generate_ads_workspace",
    "metadata_for",
    "netlist_element_graph",
    "polish_netlist",
    "render_ael_script",
    "update_global_index",
    "write_comparison_plot",
    "write_level_index",
    "write_workspace_readme",
]


def __getattr__(name: str) -> Any:
    """Lazily expose ``generate_ads_workspace`` (PEP 562).

    ``ads_workspace`` is also a ``python -m`` entry point; importing it
    eagerly here would make the runtime warn about double import when
    the module is run as ``-m``. The lazy hook keeps
    ``tvtl_ads_builder_v2.inspection.generate_ads_workspace`` available
    without the eager import.
    """

    if name == "generate_ads_workspace":
        from .ads_workspace import generate_ads_workspace

        return generate_ads_workspace
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def emit_inspection_artifacts(
    output_root: str | Path,
    level_number: int,
    specs: Sequence[DeviceSpec],
    results: Sequence[Mapping[str, Any]],
    *,
    level_title: str | None = None,
) -> dict[str, Any]:
    """Emit every inspection artifact for one completed parity run.

    For each canonical circuit this polishes the emitted ``.ckt``
    header, writes ``comparison_plot.csv`` / ``.png`` and
    ``inspection_README.md`` into the circuit workspace; it then writes
    the per-level ``inspection_index.md`` / ``inspection_status.json``
    and rebuilds the cross-level ``ads_builder_v2_inspection_index.md``.

    Parameters
    ----------
    output_root : str or Path
        The directory holding the per-level parity roots
        (``<OUTPUT_DIR>`` in normal runs). The
        level's workspaces live under
        ``<output_root>/ads_builder_v2_level_<N>/<circuit label>``.
    level_number : int
        The ladder level number.
    specs : sequence of DeviceSpec
        The canonical circuits of the level, in run order.
    results : sequence of Mapping
        The parity-check results, aligned with ``specs`` (the i-th
        result is for ``specs[i]``). Each result carries the keys
        documented for :func:`tvtl_ads_builder_v2.parity.parity_check`,
        with ``label`` set to the device-spec (circuit) label.
    level_title : str or None
        The level's human-readable parity-gate title, copied into the
        level index. Falls back to a generic title.

    Returns
    -------
    dict
        ``{"level_dir": Path, "circuit_count": int, "png_count": int,
        "readme_paths": list[Path], "index_path": Path,
        "status_path": Path, "global_index_path": Path}``.

    Notes
    -----
    ``specs`` and ``results`` must be the same length; the shorter of
    the two bounds the loop, so a truncated input simply emits fewer
    workspaces rather than raising.
    """

    output_root = Path(output_root)
    level_dir = output_root / f"ads_builder_v2_level_{level_number}"
    level_dir.mkdir(parents=True, exist_ok=True)

    readme_paths: list[Path] = []
    png_count = 0
    for spec, result in zip(specs, results):
        workspace_dir = level_dir / spec.label
        workspace_dir.mkdir(parents=True, exist_ok=True)

        netlist_path = Path(str(result.get("netlist_path", "")))
        if netlist_path.is_file():
            polish_netlist(netlist_path, label=spec.label, level=level_number)

        plot_info = write_comparison_plot(workspace_dir, result)
        png_written = bool(plot_info.get("png_written", False))
        if png_written:
            png_count += 1

        readme_paths.append(
            write_workspace_readme(
                workspace_dir, spec, result, png_written=png_written
            )
        )

    level_index = write_level_index(
        level_dir, level_number, results, level_title=level_title
    )
    # The per-level ADS inspection README reads the level index's
    # inspection_status.json for its parity column, so the inspection
    # workspace is generated after write_level_index.
    from .ads_workspace import generate_ads_workspace

    ads_workspace = generate_ads_workspace(level_dir, list(specs))
    global_index_path = update_global_index(output_root)

    return {
        "level_dir": level_dir,
        "circuit_count": len(readme_paths),
        "png_count": png_count,
        "readme_paths": readme_paths,
        "index_path": level_index["index_path"],
        "status_path": level_index["status_path"],
        "ads_workspace_dir": ads_workspace["workspace_dir"],
        "ads_workspace_deck_count": len(ads_workspace["deck_paths"]),
        "global_index_path": global_index_path,
    }


