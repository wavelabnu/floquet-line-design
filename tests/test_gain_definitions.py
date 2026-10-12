from __future__ import annotations

import json
from pathlib import Path

import pytest

from tvtl_ads_builder_v2.validation.gain_definitions import (
    GainMetricReport,
    db10,
    db20,
    full_sum_gain,
    load_power_from_voltage,
    power_gain,
    transducer_gain,
    voltage_gain,
)


def test_db20_and_db10():
    assert db20(10.0) == pytest.approx(20.0)
    assert db10(10.0) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        db20(0.0)
    with pytest.raises(ValueError):
        db10(-1.0)


def test_voltage_gain_formula():
    assert voltage_gain(4.0 + 3.0j, 1.0j) == pytest.approx(5.0)
    assert voltage_gain(1.0, 0.0) is None


def test_power_gain_formula():
    assert power_gain(2.5, 0.5) == pytest.approx(5.0)
    assert power_gain(1.0, 0.0) is None
    with pytest.raises(ValueError):
        power_gain(-1.0, 1.0)


def test_load_power_from_voltage_rms_and_peak():
    assert load_power_from_voltage(10.0, 50.0, "rms") == pytest.approx(2.0)
    assert load_power_from_voltage(10.0, 50.0, "peak") == pytest.approx(1.0)
    with pytest.raises(ValueError):
        load_power_from_voltage(1.0, 0.0, "rms")


def test_transducer_gain_formula():
    assert transducer_gain(1.0e-3, 1.0e-4) == pytest.approx(10.0)
    assert transducer_gain(1.0e-3, 0.0) is None


def test_full_sum_gain_formula_and_missing_harmonic():
    powers = {1: 1.0e-3, 2: 2.0e-3, 3: 3.0e-3}
    assert full_sum_gain(powers, 1.0e-3, [1, 3]) == pytest.approx(4.0)
    assert full_sum_gain(powers, 1.0e-3, [1, 4]) is None
    with pytest.raises(ValueError):
        full_sum_gain({1: -1.0}, 1.0, [1])


def test_gain_metric_report_json_serializes():
    report = GainMetricReport(
        voltage_gain_db=6.0,
        load_power_gain_db=None,
        harmonics_in_sum=[1, 3],
        notes=["missing load current"],
    )
    encoded = json.dumps(report.to_dict(), sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["voltage_gain_db"] == 6.0
    assert decoded["load_power_gain_db"] is None
    assert decoded["harmonics_in_sum"] == [1, 3]
    assert decoded["notes"] == ["missing load current"]


def test_gain_definitions_doc_contains_major_equations():
    doc = Path(__file__).resolve().parents[1] / "docs" / "GAIN_DEFINITIONS.md"
    text = doc.read_text(encoding="utf-8")

    assert "G_V(fs) = |V_out(fs)| / |V_in(fs)|" in text
    assert "G_P(fs) = P_L(fs) / P_in(fs)" in text
    assert "G_T(fs) = P_L(fs) / P_avs(fs)" in text
    assert "G_S21(fs) = |S21(fs)|^2" in text
    assert "G_coh,V = |V_out(fs)| / |V_ref(fs)|" in text
    assert "G_sum = (sum_{k in K_out} P_out(k fs)) / P_in(fs)" in text
    assert "eta_sig = P_out(fs) / P_pump,av" in text
