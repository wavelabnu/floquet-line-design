"""ADS OA amplifier scaffold export tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _summary() -> dict:
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
            "topology": "n2_double_short",
            "n_segments": 2,
            "via": "double",
            "termination": "short",
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
            "input_match": {
                "topology": "single_stub",
                "elements": [
                    {
                        "kind": "series_line",
                        "width_m": 3.5e-3,
                        "length_m": 6.0e-3,
                        "z0_ohm": 50.0,
                    }
                ],
            },
            "output_match": None,
        },
    }


def _write_fixture(tmp_path: Path) -> Path:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(_summary()), encoding="utf-8")
    candidates = [_summary()["best_candidate"], {**_summary()["best_candidate"], "rank": 2}]
    (tmp_path / "candidates.json").write_text(json.dumps(candidates), encoding="utf-8")
    return summary_path


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src = Path(__file__).resolve().parents[1] / "src"
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(src) if not existing else str(src) + os.pathsep + existing
    return env


def test_export_amplifier_oa_help_exits_successfully():
    proc = subprocess.run(
        [sys.executable, "-m", "tvtl_ads_builder_v2", "export-amplifier-oa", "--help"],
        capture_output=True,
        check=False,
        env=_subprocess_env(),
        text=True,
    )
    assert proc.returncode == 0
    assert "--input" in proc.stdout
    assert "--simulate" in proc.stdout
    assert "--gain-comparison-report" in proc.stdout
    assert "--pump-sweep-plot" in proc.stdout


def test_summary_is_parsed_and_selected_candidates_counted(tmp_path):
    from tvtl_ads_builder_v2.amplifier_oa_export import build_amplifier_schematic_plan
    from tvtl_ads_builder_v2.gui_export import load_optimizer_candidates

    summary_path = _write_fixture(tmp_path)
    summary, candidates = load_optimizer_candidates(summary_path)
    plan = build_amplifier_schematic_plan(
        summary, candidates[0], selected_count=len(candidates)
    )

    assert plan["candidate_summary"]["candidate_name"] == "n2_double_short"
    assert plan["candidate_summary"]["target_frequency_GHz"] == 2.0
    assert plan["candidate_summary"]["coherent_peak"] == 10.6
    assert plan["candidate_summary"]["Z_TL_ohm"] == 61.3
    assert plan["candidate_summary"]["main_width_mm"] == 2.49
    assert plan["candidate_summary"]["n_cells"] == 15
    assert plan["candidate_summary"]["selected_candidates"] == 2
    assert plan["coherent_peak_from_optimizer"] == 10.6
    assert plan["available_gain_metrics"]["analytical"]["coherent_gain_db"] == 10.6
    assert (
        plan["available_gain_metrics"]["analytical"]["normalization"]
        == "Stage-4 coherent_peak normalized to pump-off signal reference"
    )
    assert "sweep-pump-gain" in plan["pump_power_sweep_command"]
    assert plan["gain_artifacts"]["status"] == "suggested_paths_only"
    assert plan["schematic_annotation_status"] == "planned_for_ads_text_and_open_in_ads"
    assert "gain_comparison_report.md" in plan["schematic_annotation"]
    assert plan["sim_ready_status"]["full_amplifier_numerical_gain"] == "unavailable_preliminary"


def test_gain_artifacts_are_discovered_and_copied_into_project(tmp_path):
    from tvtl_ads_builder_v2.amplifier_oa_export import _prepare_gain_artifacts

    run_dir = tmp_path / "runs" / "amp_2ghz"
    compare_dir = tmp_path / "runs" / "amp_2ghz_gain_compare_ads"
    sweep_dir = tmp_path / "runs" / "amp_2ghz_pump_sweep_ads"
    out_dir = tmp_path / "ads_project"
    run_dir.mkdir(parents=True)
    compare_dir.mkdir()
    sweep_dir.mkdir()
    out_dir.mkdir()
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(_summary()), encoding="utf-8")
    (compare_dir / "gain_comparison_report.md").write_text("# compare\n", encoding="utf-8")
    (compare_dir / "gain_comparison_report.json").write_text("{}\n", encoding="utf-8")
    (sweep_dir / "pump_power_sweep_report.md").write_text("# sweep\n", encoding="utf-8")
    (sweep_dir / "pump_sweep.json").write_text("{}\n", encoding="utf-8")
    (sweep_dir / "gain_vs_pump_power.png").write_bytes(b"png")

    artifacts = _prepare_gain_artifacts(out_dir, summary_path, {})

    assert artifacts["status"] == "copied_all_gain_artifacts_to_project"
    assert artifacts["source_status"]["gain_comparison_report_md"] == "auto_discovered"
    assert artifacts["project_paths"]["gain_comparison_report_md"] == (
        str(Path("gain_reports") / "gain_comparison_report.md")
    )
    assert artifacts["project_paths"]["pump_power_sweep_png"] == (
        str(Path("gain_reports") / "gain_vs_pump_power.png")
    )
    assert (out_dir / "gain_reports" / "gain_comparison_report.md").read_text(
        encoding="utf-8"
    ) == "# compare\n"
    assert (out_dir / "gain_reports" / "gain_vs_pump_power.png").read_bytes() == b"png"


def test_amplifier_plan_is_self_pumped_same_line_scaffold(tmp_path):
    from tvtl_ads_builder_v2.amplifier_oa_export import build_amplifier_schematic_plan
    from tvtl_ads_builder_v2.gui_export import load_optimizer_candidates

    summary_path = _write_fixture(tmp_path)
    summary, candidates = load_optimizer_candidates(summary_path)
    plan = build_amplifier_schematic_plan(
        summary, candidates[0], selected_count=len(candidates)
    )

    sources = plan["sources"]
    assert [source["role"] for source in sources] == ["signal_fs", "pump_2fs"]
    assert {source["positive_net"] for source in sources} == {"RF_MAIN"}
    assert {source["negative_net"] for source in sources} == {"gnd!"}
    assert plan["architecture"]["forbidden_lane_status"] == "absent"
    assert plan["proof"]["no_forbidden_lane_exists"] is True
    assert plan["mode"] == "connected_amplifier_symbol_scaffold"
    assert plan["unit_cell_representation"] == "custom_unit_cell_symbol"
    assert plan["unit_cell_symbol_name"] == "UC_N2_DOUBLE_SHORT"
    assert plan["sim_ready"] is False
    assert "implemented unit-cell internals" in plan["simulation_claim"]
    assert "not a validated full amplifier hpeesofsim simulation" in plan["simulation_claim"]


def test_unit_cell_cascade_is_connected(tmp_path):
    from tvtl_ads_builder_v2.amplifier_oa_export import build_amplifier_schematic_plan
    from tvtl_ads_builder_v2.gui_export import load_optimizer_candidates

    summary_path = _write_fixture(tmp_path)
    summary, candidates = load_optimizer_candidates(summary_path)
    plan = build_amplifier_schematic_plan(
        summary, candidates[0], selected_count=len(candidates)
    )

    blocks = plan["unit_cell_cascade"]["blocks"]
    assert len(blocks) == 15
    assert plan["unit_cell_cascade"]["unit_cell_symbol_name"] == "UC_N2_DOUBLE_SHORT"
    assert plan["unit_cell_cascade"]["unit_cell_representation"] == "custom_unit_cell_symbol"
    assert plan["unit_cell_cascade"]["unit_cell_internal_status"] == "implemented"
    assert plan["unit_cell_cascade"]["internal_sim_ready"] is True
    assert plan["unit_cell_implementation"]["status"] == "implemented"
    assert plan["unit_cell_implementation"]["representation"] == "single_shunt_varactor_device_spec"
    assert plan["unit_cell_implementation"]["topology"]["key"] == "n2_double_short"
    assert plan["unit_cell_implementation"]["component_counts"] == {
        "I_Probe": 1,
        "L": 1,
        "MLIN": 4,
        "MTEE": 1,
        "R": 1,
        "SDD2P": 1,
        "VIAGND": 2,  # n2_double_short: two vias per cell since v1.0.1 (was 1)
    }
    assert "UC_TLIN_PLACEHOLDER" not in plan["unit_cell_implementation"]["element_ids"]
    assert "LVIA_C1" not in plan["unit_cell_implementation"]["element_ids"]
    assert "VIAGND_C1" in plan["unit_cell_implementation"]["element_ids"]
    assert plan["unit_cell_implementation"]["termination_models"]["via"] == "viagnd"
    assert plan["unit_cell_implementation"]["termination_models"]["open_end"] == "mlef"
    assert blocks[0]["input_net"] == "RF_MAIN"
    for left, right in zip(blocks, blocks[1:]):
        assert left["output_net"] == right["input_net"]
    assert plan["load"]["input_net"] == blocks[-1]["output_net"]
    assert all(block["role"] == "unit_cell_symbol" for block in blocks)
    assert all(block["ads_cell"] != "ads_tlines:TLIN:symbol" for block in blocks)
    assert all(block["role"] != "unit_cell_placeholder" for block in blocks)
    assert plan["proof"]["uc_cascade_connected"] is True
    assert plan["proof"]["uc_blocks_are_custom_symbols"] is True
    assert plan["proof"]["unit_cell_internals_implemented"] is True
    assert plan["proof"]["unit_cell_internals_use_placeholder"] is False
    assert plan["proof"]["output_load_connected"] is True


def test_open_end_unit_cell_plan_uses_mlef_not_mloc_or_lvia(tmp_path):
    from tvtl_ads_builder_v2.amplifier_oa_export import build_amplifier_schematic_plan

    summary = _summary()
    candidate = {
        **summary["best_candidate"],
        "topology": "n2_none_open",
        "via": "none",
        "termination": "open",
    }
    plan = build_amplifier_schematic_plan(summary, candidate, selected_count=1)
    plan_text = json.dumps(plan)

    assert "MLEF_C1" in plan["unit_cell_implementation"]["element_ids"]
    assert "VIAGND_C1" not in plan["unit_cell_implementation"]["element_ids"]
    assert "MLOC" not in plan_text
    assert "LVIA" not in plan_text


def test_ads_gated_oa_smoke_generates_self_pumped_netlist(tmp_path):
    import pytest

    # ADS-gated: the keysight.ads.* OpenAccess/AEL bindings only function under
    # the ADS Python interpreter (ADSPY). Under a plain CPython venv the
    # namespace can partially import (depending on collection order) but its
    # compiled AEL backend is missing symbols, so export_amplifier_oa would
    # crash deep inside AEL. importorskip only catches ImportError, not the
    # AttributeError that the partial AEL import raises, so acquire the ADS
    # runtime defensively and skip on ANY failure -- this keeps the test an
    # order-independent ADS-gated skip rather than an order-dependent failure.
    try:
        import keysight.ads.de  # noqa: F401
        from keysight.ads.ael import _ael

        if not hasattr(_ael, "AELValue"):
            raise RuntimeError("keysight AEL backend missing AELValue (needs ADSPY)")
    except Exception as exc:  # noqa: BLE001 - any failure means no usable ADS runtime
        pytest.skip(f"ADS Python runtime (ADSPY) unavailable: {exc}")
    from tvtl_ads_builder_v2.amplifier_oa_export import export_amplifier_oa

    summary_path = _write_fixture(tmp_path)
    manifest = export_amplifier_oa(summary_path, tmp_path / "amp_oa")
    plan = json.loads(Path(manifest["plan_path"]).read_text(encoding="utf-8"))
    build = json.loads(Path(manifest["build_result_path"]).read_text(encoding="utf-8"))
    netlist = Path(manifest["generated_netlist_path"]).read_text(encoding="utf-8")
    uc_lines = []
    for raw_line in netlist.splitlines():
        line = raw_line.strip()
        if not line.startswith("UC_N2_DOUBLE_SHORT:UC"):
            continue
        parts = line.split()
        uc_lines.append((parts[0], parts[1], parts[2]))

    assert Path(manifest["oa_workspace"]).exists()
    assert "schematic_annotation" in manifest
    assert "schematic_annotation_status" in manifest
    assert manifest["gain_artifacts"]["status"] == "suggested_paths_only"
    assert build["required_cellviews"]["unit_cell_symbol"] == (
        "tvtl_amp_oa_lib:UC_N2_DOUBLE_SHORT:symbol"
    )
    assert build["required_cellviews"]["unit_cell_schematic"] == (
        "tvtl_amp_oa_lib:UC_N2_DOUBLE_SHORT:schematic"
    )
    assert all(
        build["instance_cells"][f"UC{index:02d}"] == "UC_N2_DOUBLE_SHORT"
        for index in range(1, 16)
    )
    assert all(
        block["ads_cell"] == "tvtl_amp_oa_lib:UC_N2_DOUBLE_SHORT:symbol"
        for block in plan["unit_cell_cascade"]["blocks"]
    )
    assert "PORT_SIG  RF_MAIN 0" in netlist
    assert "PORT_PUMP  RF_MAIN 0" in netlist
    assert "Freq[1]=fs" in netlist
    assert "Freq[1]=2*fs" in netlist
    assert "TLIN:UC01" not in netlist
    assert "UC_TLIN_PLACEHOLDER" not in netlist
    assert "MTEE_ADS:TEE_C1" in netlist
    assert "MLIN2:LINE_LEFT_C1" in netlist
    assert "MLIN2:LINE_RIGHT_C1" in netlist
    assert "MLIN2:LINE_RIGHT_C1  RF_OUT RF_OUT" not in netlist
    assert "MLIN2:STUB_C1_0" in netlist
    assert "MLIN2:STUB_C1_1" in netlist
    assert "R:RPKG_C1" in netlist
    assert "L:LPKG_C1" in netlist
    assert "L:LVIA_C1" not in netlist
    assert "LVIA_C1" not in netlist
    assert "MLOC:" not in netlist
    assert "VIAGND:VIAGND_C1" in netlist
    assert "SDD:VARACTOR_C1" in netlist
    assert "UC_N2_DOUBLE_SHORT" in netlist
    assert len(uc_lines) == 15
    assert uc_lines[0][1] == "RF_MAIN"
    assert all(node_in != node_out for _inst, node_in, node_out in uc_lines)
    for left, right in zip(uc_lines, uc_lines[1:]):
        assert left[2] == right[1]
    assert " PUMP " not in f" {netlist} "
    assert " CONTROL " not in f" {netlist} "
    assert " BIAS " not in f" {netlist} "
