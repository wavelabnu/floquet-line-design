"""Emit an ADS-native netlist (``.net``) from a device spec.

This is a *second* netlist form, separate from and additive to the
hpeesofsim flat deck produced by :func:`emit_hpeesofsim_netlist`. The
hpeesofsim deck is the simulator input the parity gate uses and is left
untouched. The ADS-native form here mirrors the netlist ADS itself
writes for a drawn schematic -- captured from the reference workspace's
``netlist.log`` (see ``ADS reference-workspace inventory``).

The two forms differ in model names and parameter sets:

* ``MLIN``  -> ``MLIN2`` with ``Wall1``/``Wall2``/``Mod`` parameters
* ``MTEE``  -> ``MTEE_ADS``
* ``Port``  carries ``Noise=yes``
* the ``MSUB`` model carries the full ADS default parameter set
* ``S_Param`` carries the full controller parameter set and an
  ``OutputPlan`` block

The ADS-native ``.net`` documents the canonical ADS form of each
canonical circuit. Note that ADS exposes no AEL path from a netlist to
a drawn schematic cellview (see
``src/tvtl_ads_builder_v2/inspection/verified_ael_functions.py``); the
``.net`` is a reference/handoff artifact, not an automation input.
"""

from __future__ import annotations

from typing import Callable

from ..device_spec import DeviceSpec, resolve_spec, validate_device_spec

# Fixed ADS sentinels reproduced verbatim from the reference netlist.
_WALL_SENTINEL = "Wall1=1.0E+30 mil Wall2=1.0E+30 mil"
_MSUB_DEFAULT_TAIL = (
    "Hu=1e+33 mm DielectricLossModel=1 FreqForEpsrTanD=1.0 GHz "
    "LowFreqForTanD=1.0 kHz HighFreqForTanD=1.0 THz RoughnessModel=2"
)


def emit_ads_native_netlist(spec: DeviceSpec) -> str:
    """Emit the ADS-native ``.net`` text for ``spec``.

    Parameters
    ----------
    spec : DeviceSpec
        Device spec describing a Level-1/2/3 circuit. Re-validated here.

    Returns
    -------
    str
        The ADS-native netlist: a leading comment block, the substrate
        ``model`` line, every element in order (ground nodes
        translated to ``0``), and the ``S_Param`` / ``SweepPlan`` /
        ``OutputPlan`` analysis block.

    Raises
    ------
    ValueError
        If any element has a ``type`` outside ``TermG``/``MLIN``/``MTEE``.
    """

    spec = resolve_spec(spec)
    validate_device_spec(spec)
    ground = set(spec.ground_nodes)
    resolve: Callable[[str], str] = lambda node: "0" if node in ground else node

    lines: list[str] = []
    lines.append("// tvtl_ads_builder_v2 ADS-native netlist (.net)\n")
    if spec.label:
        lines.append(f"// label = {spec.label}\n")
    lines.append(
        "// canonical ADS form -- see "
        "ADS reference-workspace inventory\n\n"
    )

    lines.append(_emit_msub(spec) + "\n\n")

    for element in spec.elements:
        lines.append(_emit_element(element, spec.substrate.name, resolve) + "\n")
    lines.append("\n")

    lines.append(_emit_sp_block(spec) + "\n")
    return "".join(lines)


def _emit_element(element, substrate_name: str, resolve: Callable[[str], str]) -> str:
    """Emit one ADS-native element line."""

    if element.type == "TermG":
        pin = element.pins[0]
        z_ohm = float(element.parameters["z_ohm"])
        num = int(element.parameters["num"])
        return (
            f"Port:{element.id} {resolve(pin.node)} 0 "
            f"Num={num} Z={z_ohm:.12g} Ohm Noise=yes"
        )
    if element.type == "MLIN":
        pins = {pin.pin: resolve(pin.node) for pin in element.pins}
        w_mm = float(element.parameters["w_m"]) * 1.0e3
        l_mm = float(element.parameters["l_m"]) * 1.0e3
        return (
            f"MLIN2:{element.id} {pins[1]} {pins[2]} "
            f'Subst="{substrate_name}" '
            f"W={w_mm:.12g} mm L={l_mm:.12g} mm {_WALL_SENTINEL} Mod=1"
        )
    if element.type == "MTEE":
        pins = {pin.pin: resolve(pin.node) for pin in element.pins}
        w1_mm = float(element.parameters["w1_m"]) * 1.0e3
        w2_mm = float(element.parameters["w2_m"]) * 1.0e3
        w3_mm = float(element.parameters["w3_m"]) * 1.0e3
        return (
            f"MTEE_ADS:{element.id} {pins[1]} {pins[2]} {pins[3]} "
            f'Subst="{substrate_name}" '
            f"W1={w1_mm:.12g} mm W2={w2_mm:.12g} mm W3={w3_mm:.12g} mm"
        )
    raise ValueError(
        f"emit_ads_native_netlist: element '{element.id}' has unsupported "
        f"type '{element.type}' (handles TermG, MLIN, MTEE)."
    )


def _emit_msub(spec: DeviceSpec) -> str:
    """Emit the ADS-native ``model <name> MSUB ...`` line."""

    substrate = spec.substrate
    if substrate.model != "hammerstad_kirschning":
        raise ValueError(
            f"emit_ads_native_netlist does not implement substrate model "
            f"'{substrate.model}'."
        )
    p = substrate.parameters
    h_mm = float(p["h_m"]) * 1.0e3
    t_mm = float(p["t_m"]) * 1.0e3
    rough_mm = float(p.get("rough_m", 0.0)) * 1.0e3
    return (
        f"model {substrate.name} MSUB "
        f"H={h_mm:.12g} mm "
        f"Er={float(p['eps_r']):.12g} "
        f"Mur={float(p.get('mur', 1.0)):.12g} "
        f"Cond={float(p['sigma_S_per_m']):.12g} "
        f"T={t_mm:.12g} mm "
        f"TanD={float(p['tan_delta']):.12g} "
        f"Rough={rough_mm:.12g} mm "
        f"{_MSUB_DEFAULT_TAIL}"
    )


def _emit_sp_block(spec: DeviceSpec) -> str:
    """Emit the ADS-native S_Param / SweepPlan / OutputPlan block."""

    grid = spec.analysis.frequency_grid_Hz
    start_GHz = float(grid.start) * 1.0e-9
    stop_GHz = float(grid.stop) * 1.0e-9
    points = int(grid.points)
    step_GHz = (stop_GHz - start_GHz) / (points - 1) if points > 1 else (
        stop_GHz - start_GHz
    )
    controller = (
        "S_Param:SP1 CalcS=yes CalcY=no CalcZ=no GroupDelayAperture=1e-4 "
        "FreqConversion=no FreqConversionPort=1 StatusLevel=2 CalcNoise=no "
        "SortNoise=0 BandwidthForNoise=1.0 Hz DevOpPtLevel=0 SPSS_WSP=yes \\\n"
        'SweepVar="freq" SweepPlan="SP1_stim" OutputPlan="SP1_Output"'
    )
    plan = (
        f"SweepPlan: SP1_stim Start={start_GHz:.12g} GHz "
        f"Stop={stop_GHz:.12g} GHz Step={step_GHz:.12g} GHz"
    )
    output_plan = (
        "OutputPlan:SP1_Output \\\n"
        "      Type=\"Output\" \\\n"
        "      UseEquationNestLevel=yes \\\n"
        "      EquationNestLevel=2 \\\n"
        "      UseSavedEquationNestLevel=yes \\\n"
        "      SavedEquationNestLevel=2"
    )
    return controller + "\n\n" + plan + "\n\n" + output_plan


