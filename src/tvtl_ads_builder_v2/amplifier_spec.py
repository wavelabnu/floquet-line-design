"""Shared public amplifier circuit specification.

This module is the source of truth for public amplifier validation. It
rebuilds the selected optimizer candidate once, expands it into the same
single-shunt-varactor unit cells used by the ADS-OA exporter, and exposes a
flat :class:`DeviceSpec` for direct hpeesofsim validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from tvtl.microstrip.discontinuities import via_inductance_bahl_H
from tvtl.optimization.substrate import ROGERS_4003C
from tvtl.optimization.topology_generator import Termination, VIA_RADIUS_M

from .cells.single_shunt_varactor import SingleShuntVaractorCell
from .device_spec import (
    Analysis,
    DeviceSpec,
    Element,
    ElementPin,
    FrequencyGrid,
    HBTwoToneSpec,
    Node,
    P1ToneSource,
    Port,
    VarBlockSpec,
    VarEntry,
)
from .gui_export import (
    ExportError,
    _matching_from_candidate,
    _params_from_candidate,
    _topology_from_candidate,
    _varactor_from_summary,
    load_optimizer_candidates,
    select_candidates,
)
from .models.registry import standard_chain_models
from .parity.optimized_design import _substrate_from_config
from .parity.self_pumped_design import _q_poly_reference, main_period_anchor_at_3p1GHz

RF_MAIN_NET = "RF_MAIN"
DIRECT_OUTPUT_NET = "N_LOAD"
SIGNAL_SOURCE_NAME = "PORT_SIG"
PUMP_SOURCE_NAME = "PORT_PUMP"
DEFAULT_SOURCE_Z_OHM = 100.0
DEFAULT_LOAD_Z_OHM = 50.0
DEFAULT_SIGNAL_POWER_DBM = -3.0
DEFAULT_HB_ORDER = 5
DEFAULT_HB_MAX_ORDER = 6


@dataclass(frozen=True)
class AmplifierCircuitSpec:
    """Public amplifier candidate and its expanded direct-HB spec."""

    candidate_name: str
    topology_family: str
    fs_hz: float
    pump_hz: float
    psig_dbm: float
    ppump_dbm: float
    source_z_ohm: float
    load_z_ohm: float
    z_tl_ohm: float | None
    main_width_m: float
    main_period_m: float
    n_cells: int
    stub_parameters_m: tuple[float, ...]
    varactor_parameters: dict[str, float]
    package_parameters: dict[str, float]
    matching_sections: dict[str, Any]
    matching_status: str
    termination_type: str
    via_model: str
    open_end_model: str
    direct_output_net: str
    device_spec: DeviceSpec
    analytical_metrics: dict[str, Any]
    candidate_summary: dict[str, Any]

    def to_dict(self, *, include_device_spec: bool = True) -> dict[str, Any]:
        data = asdict(self)
        if include_device_spec:
            data["device_spec"] = self.device_spec.to_dict()
        else:
            data.pop("device_spec", None)
        return data


def load_amplifier_circuit_spec(
    input_summary: str | Path,
    *,
    candidate_index: int | None = None,
    candidate_name: str | None = None,
    signal_power_dbm: float | None = None,
    pump_power_dbm: float | None = None,
    pump_phase_rad: float = 0.0,
    n_cells: int | None = None,
    hb_order: int | None = None,
    bias_return_ohm: float | None = None,
) -> AmplifierCircuitSpec:
    """Load optimizer output and return the selected public amplifier spec.

    ``bias_return_ohm`` adds the varactor bias-return resistor per cell (required for an HB deck of an
    OPEN-stub family; reason ``no_dc_return_for_varactor``).
    """

    summary_path = Path(input_summary).resolve()
    summary, candidates = load_optimizer_candidates(summary_path)
    if not candidates:
        raise ExportError(f"optimizer summary has no selected candidates: {summary_path}")
    selection = select_candidates(
        candidates,
        candidate_index=candidate_index,
        candidate_name=candidate_name,
    )[0]
    return build_amplifier_circuit_spec(
        summary,
        selection.candidate,
        selected_count=len(candidates),
        signal_power_dbm=signal_power_dbm,
        pump_power_dbm=pump_power_dbm,
        pump_phase_rad=pump_phase_rad,
        n_cells=n_cells,
        hb_order=hb_order,
        bias_return_ohm=bias_return_ohm,
    )


def build_amplifier_circuit_spec(
    summary: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    selected_count: int = 1,
    signal_power_dbm: float | None = None,
    pump_power_dbm: float | None = None,
    pump_phase_rad: float = 0.0,
    n_cells: int | None = None,
    hb_order: int | None = None,
    bias_return_ohm: float | None = None,
) -> AmplifierCircuitSpec:
    """Build the shared public spec from an optimizer candidate mapping."""

    raw_summary = dict(summary)
    raw_candidate = dict(candidate)
    config = raw_summary.get("config") if isinstance(raw_summary.get("config"), Mapping) else {}
    main_tl = raw_candidate.get("main_tl")
    if not isinstance(main_tl, Mapping):
        raise ExportError("candidate is missing main_tl")
    if main_tl.get("width_m") is None:
        raise ExportError("candidate is missing main_tl.width_m")

    fs_hz = _target_frequency_hz(raw_summary)
    psig_dbm = float(
        signal_power_dbm
        if signal_power_dbm is not None
        else config.get("drive_dBm", DEFAULT_SIGNAL_POWER_DBM)
    )
    ppump_dbm = float(pump_power_dbm if pump_power_dbm is not None else psig_dbm)
    n_cells = int(n_cells if n_cells is not None else config.get("n_cells", 15))
    if n_cells <= 0:
        raise ExportError("optimizer summary config.n_cells must be positive")
    hb_order = int(hb_order if hb_order is not None else DEFAULT_HB_ORDER)
    if hb_order <= 0:
        raise ExportError("HB order must be positive")
    hb_max_order = max(DEFAULT_HB_MAX_ORDER, hb_order)

    topology = _topology_from_candidate(raw_candidate)
    params = tuple(float(value) for value in _params_from_candidate(raw_candidate))
    varactor = _varactor_from_summary(raw_summary)
    chain_models = standard_chain_models(varactor)
    substrate = _substrate_from_config(ROGERS_4003C)
    main_width_m = float(main_tl["width_m"])
    main_period_m = main_period_anchor_at_3p1GHz(fs_hz)
    l_via_h = _l_via_for_topology(topology)
    cell = SingleShuntVaractorCell(
        topology=topology,
        params=params,
        varactor=varactor,
        substrate_name=substrate.name,
        main_width_m=main_width_m,
        main_period_m=main_period_m,
        q_poly=varactor.charge_polynomial().charge_polynomial_coefficients,
        q_poly_reference=_q_poly_reference(topology, varactor),
        l_via_H=l_via_h,
        via_model=chain_models.via,
        open_end_model=chain_models.open_end,
        bias_return_ohm=bias_return_ohm,
    )

    nodes: list[Node] = [Node(id=RF_MAIN_NET)]
    elements: list[Element] = []
    ground_nodes: list[str] = []
    for index in range(1, n_cells + 1):
        input_net = RF_MAIN_NET if index == 1 else cell.output_terminal_node(index - 1)
        output_net = cell.output_terminal_node(index)
        nodes.extend(cell.cell_nodes(index))
        elements.extend(cell.cell_elements(index, input_net=input_net, output_net=output_net))
        ground_nodes.extend(cell.ground_nodes(index))

    output_net = cell.output_terminal_node(n_cells)
    elements.append(
        Element(
            id="TERM_LOAD",
            type="TermG",
            pins=(ElementPin(pin=1, node=output_net),),
            parameters={"z_ohm": DEFAULT_LOAD_Z_OHM, "num": 3},
        )
    )
    elements.extend(_name_annotations([RF_MAIN_NET, output_net]))

    var_block = VarBlockSpec(
        name="VAR1",
        variables=(
            VarEntry(name="fs", value=float(fs_hz)),
            VarEntry(name="fpump", value=float(2.0 * fs_hz)),
            VarEntry(name="Psig_dBm", value=float(psig_dbm)),
            VarEntry(name="Ppump_dBm", value=float(ppump_dbm)),
            VarEntry(name="pump_phase_rad", value=float(pump_phase_rad)),
        ),
    )
    device_spec = DeviceSpec(
        device_spec_version=1,
        label=f"amplifier_{_candidate_name(raw_candidate)}",
        analysis=Analysis(
            type="s_parameters",
            frequency_grid_Hz=FrequencyGrid(start=fs_hz, stop=2.0 * fs_hz, points=2),
            port_impedance_ohm=DEFAULT_LOAD_Z_OHM,
        ),
        substrate=substrate,
        nodes=tuple(_dedupe_nodes(nodes)),
        elements=tuple(elements),
        ports=(Port(num=3, node=output_net),),
        ground_nodes=tuple(dict.fromkeys(ground_nodes)),
        var_block=var_block,
        hb_two_tone=HBTwoToneSpec(
            fundamental_freq_1="fs",
            fundamental_order_1=hb_order,
            fundamental_freq_2="fpump",
            fundamental_order_2=hb_order,
            max_order=hb_max_order,
        ),
        sources=(
            P1ToneSource(
                name=SIGNAL_SOURCE_NAME,
                port_node=RF_MAIN_NET,
                num=1,
                freq="fs",
                power_dBm="Psig_dBm",
                z_ohm=DEFAULT_SOURCE_Z_OHM,
                phase_rad=0.0,
            ),
            P1ToneSource(
                name=PUMP_SOURCE_NAME,
                port_node=RF_MAIN_NET,
                num=2,
                freq="fpump",
                power_dBm="Ppump_dBm",
                z_ohm=DEFAULT_SOURCE_Z_OHM,
                phase_rad="pump_phase_rad",
            ),
        ),
    )

    matching_sections = {
        "input_match": raw_candidate.get("input_match"),
        "output_match": raw_candidate.get("output_match"),
    }
    return AmplifierCircuitSpec(
        candidate_name=_candidate_name(raw_candidate),
        topology_family=topology.key,
        fs_hz=fs_hz,
        pump_hz=2.0 * fs_hz,
        psig_dbm=psig_dbm,
        ppump_dbm=ppump_dbm,
        source_z_ohm=DEFAULT_SOURCE_Z_OHM,
        load_z_ohm=DEFAULT_LOAD_Z_OHM,
        z_tl_ohm=_float_or_none(main_tl.get("z_tl_ohm")),
        main_width_m=main_width_m,
        main_period_m=main_period_m,
        n_cells=n_cells,
        stub_parameters_m=params,
        varactor_parameters=_varactor_dict(varactor),
        package_parameters={
            "L_pkg_h": float(varactor.L_pkg_h),
            "R_series_ohm": float(varactor.R_series_ohm),
        },
        matching_sections=matching_sections,
        matching_status=_matching_status(raw_candidate),
        termination_type=topology.termination.value,
        via_model="viagnd",
        open_end_model="mlef",
        direct_output_net=output_net,
        device_spec=device_spec,
        analytical_metrics=_analytical_metrics(raw_candidate),
        candidate_summary=_candidate_summary(
            raw_summary,
            raw_candidate,
            selected_count=selected_count,
            fs_hz=fs_hz,
            psig_dbm=psig_dbm,
            ppump_dbm=ppump_dbm,
        ),
    )


def _name_annotations(nodes: list[str]) -> list[Element]:
    return [
        Element(
            id=f"NAME_AMP_{index:02d}_{node}",
            type="NAME",
            pins=(ElementPin(pin=1, node=node),),
            parameters={},
        )
        for index, node in enumerate(nodes)
    ]


def _dedupe_nodes(nodes: list[Node]) -> list[Node]:
    out: dict[str, Node] = {}
    for node in nodes:
        out.setdefault(node.id, node)
    return list(out.values())


def _target_frequency_hz(summary: Mapping[str, Any]) -> float:
    if summary.get("target_frequency_Hz") is not None:
        return float(summary["target_frequency_Hz"])
    if summary.get("target_frequency_GHz") is not None:
        return float(summary["target_frequency_GHz"]) * 1.0e9
    raise ExportError("optimizer summary is missing target frequency")


def _candidate_name(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("topology") or candidate.get("label") or "candidate")


def _candidate_summary(
    summary: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    selected_count: int,
    fs_hz: float,
    psig_dbm: float,
    ppump_dbm: float,
) -> dict[str, Any]:
    main_tl = candidate.get("main_tl") if isinstance(candidate.get("main_tl"), Mapping) else {}
    return {
        "candidate_name": _candidate_name(candidate),
        "rank": candidate.get("rank"),
        "target_frequency_Hz": fs_hz,
        "target_frequency_GHz": fs_hz / 1.0e9,
        "pump_frequency_Hz": 2.0 * fs_hz,
        "pump_frequency_GHz": 2.0 * fs_hz / 1.0e9,
        "Psig_dBm": psig_dbm,
        "Ppump_dBm": ppump_dbm,
        "source_z_ohm": DEFAULT_SOURCE_Z_OHM,
        "load_z_ohm": DEFAULT_LOAD_Z_OHM,
        "Z_TL_ohm": main_tl.get("z_tl_ohm"),
        "main_width_m": main_tl.get("width_m"),
        "main_width_mm": main_tl.get("width_mm"),
        "n_cells": int(summary.get("config", {}).get("n_cells", 15))
        if isinstance(summary.get("config"), Mapping)
        else 15,
        "topology_family": candidate.get("topology"),
        "n_segments": candidate.get("n_segments"),
        "via": candidate.get("via"),
        "termination": candidate.get("termination"),
        "selected_candidates": selected_count,
        "coherent_peak": _best_metric(candidate, "coherent_peak"),
    }


def _best_metric(candidate: Mapping[str, Any], key: str) -> float | None:
    metrics = candidate.get("metrics_at_drive")
    if isinstance(metrics, Mapping) and metrics.get(key) is not None:
        return float(metrics[key])
    analytical = candidate.get("analytical")
    if isinstance(analytical, Mapping):
        for value in analytical.values():
            if isinstance(value, Mapping) and value.get(key) is not None:
                return float(value[key])
    return None


def _analytical_metrics(candidate: Mapping[str, Any]) -> dict[str, Any]:
    metrics = candidate.get("metrics_at_drive")
    return {
        "status": "available" if isinstance(metrics, Mapping) else "unavailable",
        "metrics_at_drive": dict(metrics) if isinstance(metrics, Mapping) else {},
        "coherent_gain_db": _best_metric(candidate, "coherent_peak"),
        "normalization": "Stage-4 coherent_peak normalized to pump-off signal reference",
    }


def _matching_status(candidate: Mapping[str, Any]) -> str:
    input_match = _matching_from_candidate(dict(candidate), "input_match")
    output_match = _matching_from_candidate(dict(candidate), "output_match")
    if input_match is None and output_match is None:
        return "none"
    return "recorded_excluded_to_match_current_oa_export"


def _varactor_dict(varactor: Any) -> dict[str, float]:
    return {
        "CJ0_f": float(varactor.CJ0_f),
        "VJ_v": float(varactor.VJ_v),
        "M": float(varactor.M),
        "CP_f": float(varactor.CP_f),
        "L_pkg_h": float(varactor.L_pkg_h),
        "R_series_ohm": float(varactor.R_series_ohm),
    }


def _l_via_for_topology(topology: Any) -> float:
    if topology.termination not in (Termination.SHORT, Termination.COMBINED):
        return 0.0
    return float(
        via_inductance_bahl_H(
            substrate_thickness_m=float(ROGERS_4003C.h_m),
            via_radius_m=float(VIA_RADIUS_M),
            n_parallel=topology.n_via_parallel,
        )
    )


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
