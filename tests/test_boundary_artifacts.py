"""Smoke/gating tests for the R10 model-boundary artifacts and SM/main figures.

Covers the adopted run records (loadable + key fields), the relocated stub
S-parameter tables (Touchstone header, point count, and Y_stub(3.1 GHz) matching
the value recorded in the m* arc), a machine-path hygiene guard, and that every
committed figure script runs headless (exit 0) and emits its PDF + PNG.

No ADS: every artifact here is pure post-processing of committed inputs.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "examples" / "real_parts" / "results"
STUBS = RESULTS / "stub_tables"
FIGSCRIPTS = REPO / "examples" / "real_parts"


# --------------------------------------------------------------------------- #
# 1) schema smoke on the adopted JSON run records                             #
# --------------------------------------------------------------------------- #
def _load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def test_headline_mavr_schema():
    d = _load("headline_mavr.json")
    for k in ("curve", "peak", "trough", "passive_pumpoff_Vout_V", "P_avs_W",
              "certified_reference_dB", "single_lobe", "all_phases_converged"):
        assert k in d, f"headline_mavr.json missing {k}"
    assert len(d["curve"]) == 24
    assert {"phase_deg", "G_mod_in_dB", "boost_dB"} <= d["curve"][0].keys()
    assert round(d["peak"]["boost_dB"], 2) == 18.96
    assert d["peak"]["phase_deg"] == 210.0


def test_bias_sweep_schema():
    d = _load("smv1245_bias_sweep.json")
    for k in ("conditions", "cj_srf_table", "rows", "baseline_operating_point"):
        assert k in d, f"bias_sweep missing {k}"
    assert d["conditions"]["n_cells"] == 15
    assert len(d["rows"]) == 16                     # 8 biases x 2 pumps
    assert len(d["cj_srf_table"]) == 8
    conv = [r for r in d["rows"] if r.get("all_phases_converged")]
    assert conv and "peak_boost_dB" in conv[0] and "boost_curve" in conv[0]


def test_mstar_arc_v2_schema():
    d = _load("smv1245_mstar_arc_v2.json")
    for k in ("step0b_stub_table", "step1_linear_parity_gate", "step2_msweep_3p1GHz",
              "D16", "step4_delivered_junction_depth", "step5_verdict", "provenance"):
        assert k in d, f"mstar_arc_v2 missing {k}"
    assert len(d["step2_msweep_3p1GHz"]) == 8
    assert d["D16"]["at_notch"]["bracketed"] is True
    assert d["D16"]["at_notch"]["D16_depth_eff_junction"] > 0.2


def test_branch_delivery_schema():
    d = _load("branch_delivery_vs_freq.json")
    for k in ("grid_GHz", "T", "Z_stub_abs_ohm", "spot_table", "T_global_max"):
        assert k in d, f"branch_delivery missing {k}"
    n = len(d["grid_GHz"])
    assert n == len(d["T"]) == len(d["Z_stub_abs_ohm"]) == 1201
    for f in ("3.1", "5.3", "6.2", "8.4", "10.6"):
        assert f in d["spot_table"] and "T_dB" in d["spot_table"][f]


def test_lobe2_schema():
    d = _load("smv1245_lobe2_prediction.json")
    for k in ("stub_table", "prescribed_leg_5p3GHz", "analytical_S21_off_4p8_5p7",
              "hb_leg", "junction_at_10p6_pump"):
        assert k in d, f"lobe2 missing {k}"
    assert d["stub_table"]["npts"] == 7996
    assert len(d["analytical_S21_off_4p8_5p7"]) == 19


# --------------------------------------------------------------------------- #
# 2) relocated stub S-parameter tables (Touchstone .s1p)                      #
# --------------------------------------------------------------------------- #
def _read_s1p(path):
    f, s = [], []
    header_seen = False
    for ln in Path(path).read_text(encoding="ascii").splitlines():
        ln = ln.strip()
        if ln.startswith("#"):
            header_seen = True
            assert ln.replace("  ", " ").upper().startswith("# HZ S RI R 50")
            continue
        if not ln or ln.startswith("!"):
            continue
        a, b, c = ln.split()
        f.append(float(a)); s.append(complex(float(b), float(c)))
    assert header_seen, f"{path} has no Touchstone option line"
    return f, s


@pytest.mark.parametrize("name,npts,fhi", [
    ("smv1245_stub_55GHz.s1p", 5496, 55e9),
    ("smv1245_stub_80GHz.s1p", 7996, 80e9),
])
def test_stub_table_shape(name, npts, fhi):
    f, s = _read_s1p(STUBS / name)
    assert len(f) == npts
    assert abs(f[0] - 0.05e9) < 1e3 and abs(f[-1] - fhi) < 1e6


def test_stub_table_matches_recorded_y():
    """Y_stub(3.1 GHz) from the 55 GHz .s1p must match the m* arc's record."""
    f, s = _read_s1p(STUBS / "smv1245_stub_55GHz.s1p")
    i = min(range(len(f)), key=lambda k: abs(f[k] - 3.1e9))
    y = (1.0 / 50.0) * (1 - s[i]) / (1 + s[i])
    rec = _load("smv1245_mstar_arc_v2.json")["step0b_stub_table"]["y_stub_3p1GHz_tabulated_S"]
    assert abs(y.real - rec[0]) < 1e-9 and abs(y.imag - rec[1]) < 1e-9


# --------------------------------------------------------------------------- #
# 3) machine-path hygiene (no user/temp absolute paths committed)             #
# --------------------------------------------------------------------------- #
_ABS = re.compile(r"[A-Za-z]:\\\\Users|[A-Za-z]:\\Users|Temp[\\/]+[^\\/]+|/Users/")


@pytest.mark.parametrize("rel", [
    "results/smv1245_bias_sweep.json",
    "results/smv1245_mstar_arc_v2.json",
    "results/branch_delivery_vs_freq.json",
    "results/smv1245_lobe2_prediction.json",
    "boundary/bias_sweep_runner.py",
    "boundary/mstar_arc_v2_runner.py",
    "boundary/branch_delivery_runner.py",
    "boundary/lobe2_runner.py",
    "boundary/_stub_table.py",
])
def test_no_machine_specific_paths(rel):
    text = (FIGSCRIPTS / rel).read_text(encoding="utf-8", errors="ignore")
    m = _ABS.search(text)
    assert m is None, f"{rel} contains a machine-specific path: {m.group(0)!r}"


# --------------------------------------------------------------------------- #
# 4) every figure script runs headless (exit 0) and emits PDF + PNG          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("script", [
    "fig2c_mavr_phase_boost.py",
    "figS11_mavr_recorded_sweep.py",
    "figS6_delivery_map.py",
    "figS7_model_boundary.py",
])
def test_figure_script_runs_headless(script, tmp_path):
    env = dict(os.environ, TVTL_FIG_OUTDIR=str(tmp_path), MPLBACKEND="Agg")
    r = subprocess.run([sys.executable, str(FIGSCRIPTS / script)],
                       cwd=str(REPO), env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"{script} failed:\n{r.stdout}\n{r.stderr}"
    stem = script[:-3]
    pdf = tmp_path / f"{stem}.pdf"
    png = tmp_path / f"{stem}.png"
    assert pdf.exists() and pdf.stat().st_size > 0, f"{script} did not write {pdf.name}"
    assert png.exists() and png.stat().st_size > 0, f"{script} did not write {png.name}"
