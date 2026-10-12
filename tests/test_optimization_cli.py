"""CLI smoke tests for the public optimizer entry points."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src = Path(__file__).resolve().parents[1] / "src"
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(src) if not existing else str(src) + os.pathsep + existing
    return env


def test_module_help_exits_successfully():
    proc = subprocess.run(
        [sys.executable, "-m", "tvtl.optimization", "--help"],
        capture_output=True,
        check=False,
        env=_subprocess_env(),
        text=True,
    )
    assert proc.returncode == 0
    assert "optimize-amplifier" in proc.stdout


def test_console_entry_point_function_prints_help(capsys):
    from tvtl.optimization.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "floquet-line-design" in out
    assert "optimize-amplifier" in out


def test_quick_optimizer_smoke_writes_outputs_without_ads_or_env(tmp_path, monkeypatch):
    from tvtl.optimization.cli import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HPEESOF_DIR", raising=False)
    monkeypatch.delenv("TVTL_AEL_REFERENCE_WORKSPACE", raising=False)
    monkeypatch.delenv("TVTL_MTEE_SURROGATE_JSON", raising=False)

    out_dir = tmp_path / "amp_2ghz"
    rc = main(
        [
            "optimize-amplifier",
            "--freq-ghz",
            "2.0",
            "--out",
            str(out_dir),
            "--quick",
        ]
    )

    assert rc == 0
    summary_path = out_dir / "summary.json"
    assert summary_path.exists()
    assert (out_dir / "candidates.csv").exists()
    assert (out_dir / "candidates.json").exists()
    assert (out_dir / "report.md").exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["env_required"] is False
    assert summary["ads_validation"]["required"] is False
    assert summary["ads_validation"]["run_ads"] is False
    assert summary["target_frequency_GHz"] == 2.0
    assert summary["counts"]["reports"] >= 1
    assert not (tmp_path / ".env").exists()

    # ------------------------------------------------------------------ #
    # Theoretical (predicted) leg: predicted fields + figures + labeling. #
    # Every addition here is an analytical prediction (run_ads=False).     #
    # ------------------------------------------------------------------ #
    # Prediction labeling + pending numerical slot at the summary level.
    assert summary["results_are_predictions"] is True
    assert summary["prediction_label"]
    assert summary["numerical_verification"]["status"] == "pending"

    # G7: operating capacitance (V_bias + C@bias) on the varactor summary.
    varac = summary["varactor"]
    assert "V_bias_v" in varac
    assert "C_at_bias_f" in varac and "C_at_bias_pF" in varac

    # Figures F1/F2/F3 generated, recorded in summary, and written to disk.
    figs = summary["figures"]
    for key in ("topology_png", "predicted_sparams_png", "predicted_nsweep_png"):
        name = figs[key]
        assert name, f"missing figure key {key}"
        assert (out_dir / name).exists(), f"figure file {name} not written"

    candidates = json.loads((out_dir / "candidates.json").read_text(encoding="utf-8"))
    assert candidates, "expected at least one candidate"
    best = candidates[0]
    # G3: optimistic-prediction qualifier on the candidate gain.
    assert "optimistic" in best["predicted_gain_note"].lower()
    # G1: predicted S-parameter sweep (full arrays + endpoints).
    sweep = best["predicted_s_sweep"]
    assert sweep is not None
    assert len(sweep["s11_dB"]) == sweep["n_points"]
    assert len(sweep["s21_dB"]) == sweep["n_points"]
    assert sweep["reference_impedance_ohm"] == 50.0
    # G5: coherence length N_coh (per-candidate heuristic).
    coh = best["coherence"]
    assert coh is not None
    assert coh["chosen_N"] == 15
    assert "N_coh" in coh["formula"]
    assert coh["N_coh_cells"] is None or coh["N_coh_cells"] > 0
    # G5 reconciliation: the candidate heuristic is scoped to the simplified
    # analytical cell and is distinguished from the measured robust optimum
    # (~15 cells at 2.0 GHz, by ADS pump/length sweeps). The 15.3-cell figure is
    # attributed as a reported value from prior band-edge analysis that this
    # public cell does not reproduce. The gain-max pick is not phase-matched.
    assert "static_cell_abcd" in coh["cell_model"]
    assert coh["is_phase_matched"] is False
    law = coh["validated_law"]
    assert law["f_s_anchored"] is True  # 2.0 GHz is a documented anchor
    assert law["predicted_N_coh_cells"] == 15.3  # reported prior-analysis value
    assert law["measured_optimum_N"] == 15  # measured robust optimum (ADS sweeps)
    # The reference leads with the measured optimum and flags that the reported
    # 15.3 figure is not reproduced by this public optimizer's simplified cell.
    assert "measured robust optimum" in law["reference"].lower()
    assert "not reproduced" in law["reference"].lower()
    # G4: analytical stability margin surfaced on the candidate.
    assert "margin" in best["analytical_stability"]
    # G2: electrical length in degrees on the stub segments.
    assert best["stub_parameters"]
    assert "length_deg" in best["stub_parameters"][0]

    # CSV gains appended predicted columns; legacy leading columns preserved.
    csv_header = (out_dir / "candidates.csv").read_text(encoding="utf-8").splitlines()[0]
    assert csv_header.startswith("rank,topology,a1_score,")
    assert "predicted_N_coh_cells" in csv_header
    assert "predicted_stability_margin" in csv_header

    # report.md: predictions labeled, every gap closed, figures embedded, slot.
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "PREDICTION ONLY" in report
    # The predicted-gain disclaimer: an analytical ranking is not a gain claim, the
    # proxy's optimism is family-dependent, and only a harmonic-balance record
    # certified as in docs/GAIN_DEFINITIONS.md is a gain claim. It quotes no
    # study-specific figure. Pin the honest wording, not a trivial substring.
    assert "optimistic" in report.lower()
    assert "family-dependent" in report
    assert "not a gain claim" in report
    assert "certified as in docs/GAIN_DEFINITIONS.md" in report
    assert "Characterized Performance Limits" in report
    assert "~20 dB" not in report  # no study-specific margin is quoted
    assert "COMBINED trap" not in report
    assert "Predicted S-parameters" in report
    assert "N_coh" in report
    assert "stability margin" in report.lower()
    assert "Operating varactor capacitance" in report
    assert "electrical length" in report.lower()
    assert "Numerical verification (ADS): pending" in report
    assert "fig_topology.png" in report
    assert "fig_predicted_sparams.png" in report
    assert "fig_predicted_nsweep.png" in report
    # G5 reconciliation in the report: the candidate heuristic is distinguished
    # from the measured robust optimum (~15 cells, by ADS pump/length sweeps),
    # with a cross-reference. The reported 15.3-cell figure still appears but is
    # attributed to prior band-edge analysis, not presented as a code-reproducible
    # number, so the artifact does not over-claim.
    assert "this candidate vs the measured optimum" in report
    assert "measured robust optimum" in report.lower()
    assert "15 cells" in report
    assert "15.3" in report and "reported by prior band-edge analysis" in report
    assert "LIMITATIONS.md" in report
