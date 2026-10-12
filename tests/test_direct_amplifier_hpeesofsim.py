"""Direct amplifier hpeesofsim deck tests."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tvtl_ads_builder_v2.amplifier_spec import load_amplifier_circuit_spec
from tvtl_ads_builder_v2.validation.direct_amplifier_hpeesofsim import (
    direct_amplifier_netlist,
    write_direct_amplifier_deck,
)


def _summary(topology: str = "n2_double_short", *, termination: str = "short", via: str = "double") -> dict:
    return {
        "workflow": "optimize-amplifier",
        "target_frequency_GHz": 2.0,
        "target_frequency_Hz": 2.0e9,
        "config": {"n_cells": 15, "drive_dBm": -3.0},
        "varactor": {
            "CJ0_f": 3.0e-13,
            "VJ_v": 3.5,
            "M": 1.7,
            "CP_f": 0.0,
            "L_pkg_h": 3.0e-10,
            "R_series_ohm": 0.5,
        },
        "best_candidate": {
            "rank": 1,
            "topology": topology,
            "n_segments": 2,
            "via": via,
            "termination": termination,
            "metrics_at_drive": {"coherent_peak": 10.6},
            "main_tl": {
                "z_tl_ohm": 61.3,
                "width_m": 2.49e-3,
                "width_mm": 2.49,
            },
            "stub_parameters": [
                {"segment": 1, "width_m": 2.0e-3, "length_m": 7.0e-3},
                {"segment": 2, "width_m": 4.0e-3, "length_m": 14.0e-3},
            ],
            "input_match": None,
            "output_match": None,
        },
    }


def _write_summary(tmp_path: Path, summary: dict) -> Path:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    (tmp_path / "candidates.json").write_text(
        json.dumps([summary["best_candidate"]]),
        encoding="utf-8",
    )
    return summary_path


def test_direct_amplifier_deck_has_public_self_pumped_policy(tmp_path):
    summary_path = _write_summary(tmp_path, _summary())
    spec = load_amplifier_circuit_spec(summary_path)
    deck = direct_amplifier_netlist(spec)

    assert "Port:PORT_SIG RF_MAIN 0" in deck
    assert "Port:PORT_PUMP RF_MAIN 0" in deck
    assert " PUMP " not in f" {deck} "
    assert " CONTROL " not in f" {deck} "
    assert " BIAS " not in f" {deck} "
    assert "LVIA" not in deck
    assert "MLOC:" not in deck
    assert deck.count("VIAGND:") == 30  # n2_double_short: two vias per cell since v1.0.1 (was 15)
    assert deck.count("MTEE_ADS:TEE_C") == 15
    assert deck.count("SDD:VARACTOR_C") == 15
    assert re.search(r"[A-Za-z]:\\", deck) is None


def test_direct_open_termination_uses_mlef_not_mloc_or_lvia(tmp_path):
    summary_path = _write_summary(
        tmp_path,
        _summary("n2_none_open", termination="open", via="none"),
    )
    # v1.0.1: the open stub needs a bias return (no_dc_return_for_varactor without one)
    from tvtl_ads_builder_v2.validation.dc_return import NoDCReturn
    with pytest.raises(NoDCReturn):
        direct_amplifier_netlist(load_amplifier_circuit_spec(summary_path))
    spec = load_amplifier_circuit_spec(summary_path, bias_return_ohm=1e5)
    deck = direct_amplifier_netlist(spec)

    assert deck.count("R:RDCRET_C") == 15
    assert deck.count("MLEF:") == 15
    assert "MLOC:" not in deck
    assert "LVIA" not in deck
    assert "VIAGND:" not in deck


def test_direct_amplifier_outputs_are_written_under_requested_dir(tmp_path):
    summary_path = _write_summary(tmp_path, _summary())
    result = write_direct_amplifier_deck(summary_path, tmp_path / "runs" / "amp_2ghz_direct_hpeesofsim")

    netlist = Path(result["netlist_path"])
    spec_json = Path(result["spec_path"])
    assert netlist.is_file()
    assert spec_json.is_file()
    assert netlist.parent.name == "amp_2ghz_direct_hpeesofsim"
    payload = json.loads(spec_json.read_text(encoding="utf-8"))
    assert payload["candidate_name"] == "n2_double_short"
    assert payload["device_spec"]["sources"][0]["port_node"] == "RF_MAIN"
