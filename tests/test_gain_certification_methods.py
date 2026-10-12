"""Certification fields of the paper's Methods (v1.0.1): r_A, input-scaling slope, analytical-vs-HB peak
parity, bounded pump sweep and junction voltage, next to the 1.0.0 checks. New fields are optional on old
records: None is "not run" and never a pass."""
import math

import pytest

from tvtl_ads_builder_v2.oa.certification import NEW_FIELDS, GainRecord


def legacy():
    return dict(name='record', absolute_outin_dB=9.4, order_convergence={5: 9.383, 7: 9.399, 9: 9.399},
                device_tier_agreement=dict(sdd=9.401, diode=9.399, phase_deg=270),
                signal_off_pump_on=dict(signal_on_dBm=-30, signal_off_dBm=-90),
                phase_lobe=dict(phases_deg=list(range(0, 360, 45)),
                                absolute_dB=[1 + 8 * math.cos(math.radians(x - 270)) for x in range(0, 360, 45)],
                                passive_dB=0),
                idler=dict(detuning_Hz=50e6, signal_dBm=-27, idler_dBm=-27.2),
                soft_start=[dict(pump_dBm=x, converged=True) for x in (11, 14, 17, 20)], pump_slope=5.62)


def methods():
    return dict(r_A=6e-9, input_scaling_slope=1.0, peak_parity=dict(analytical_dB=10.2, hb_dB=9.4),
                pump_sweep=dict(pump_dBm=[11, 14, 17, 20, 23, 26], absolute_dB=[1.09, 2.95, 5.62, 9.40, 14.69, 22.26],
                                converged=[True] * 6, operating_pump_dBm=20),
                junction_voltage=dict(min_reverse_V=10.14, max_reverse_V=19.55, rating_V=22))


def test_old_record_keeps_its_verdict_and_lists_the_new_checks_as_not_run():
    out = GainRecord(**legacy()).evaluate()
    assert out['certified'] and out['missing'] == [] and out['failed'] == []
    assert out['not_run'] == list(NEW_FIELDS) and out['methods_complete'] is False


def test_complete_methods_record():
    out = GainRecord(**legacy(), **methods()).evaluate()
    assert out['certified'] and out['methods_complete'] and out['not_run'] == []


@pytest.mark.parametrize('field,value', [
    ('r_A', 0.5), ('r_A', 0.87), ('r_A', float('nan')),
    ('input_scaling_slope', 0.4999), ('input_scaling_slope', 0.083),
    ('peak_parity', dict(analytical_dB=35.8, hb_dB=5.4)),
    ('pump_sweep', dict(pump_dBm=[14, 17, 20], absolute_dB=[0.5, 1.0, 16.5], converged=[True] * 3, operating_pump_dBm=20)),
    ('pump_sweep', dict(pump_dBm=[14, 17, 20], absolute_dB=[0.5, 1.0, 9.4], converged=[True, False, True], operating_pump_dBm=20)),
    ('pump_sweep', dict(pump_dBm=[14, 20], absolute_dB=[0.5, 9.4], converged=[True, True], operating_pump_dBm=20)),
    ('junction_voltage', dict(min_reverse_V=-0.3, max_reverse_V=12.3, rating_V=22)),
    ('junction_voltage', dict(min_reverse_V=4.6, max_reverse_V=22.22, rating_V=22)),
])
def test_each_new_check_fails_the_record(field, value):
    d = {**legacy(), **methods(), field: value}
    out = GainRecord(**d).evaluate()
    assert field in out['failed'] and not out['certified'] and not out['methods_complete']


@pytest.mark.parametrize('field,value', [
    ('input_scaling_slope', 0.5), ('peak_parity', dict(analytical_dB=10.0, hb_dB=5.0)),
    ('pump_sweep', dict(pump_dBm=[17, 20], absolute_dB=[5.62, 20.61], converged=[True, True], operating_pump_dBm=20)),
    ('junction_voltage', dict(min_reverse_V=0.0, max_reverse_V=22.0, rating_V=22)),
])
def test_boundaries_that_pass(field, value):
    d = {**legacy(), **methods(), field: value}
    assert GainRecord(**d).evaluate()['methods_complete']


def test_a_present_failing_new_check_blocks_an_otherwise_certified_old_record():
    out = GainRecord(**legacy(), r_A=0.87).evaluate()
    assert not out['certified'] and out['failed'] == ['r_A']
    assert set(out['not_run']) == set(NEW_FIELDS) - {'r_A'}
    assert 'methods_complete' in GainRecord(**legacy()).to_dict()
