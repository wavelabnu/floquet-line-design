"""Tee-model range guard (v1.0.1): no silent drop of the MTEE correction above its validity edge."""
import sys
from pathlib import Path

import numpy as np
import pytest

from tvtl.design.feasibility import feasibility
from tvtl_ads_builder_v2.analytical.delivered_gain import (
    TeeModelOutOfRange, cell_functions_from_spec, delivered_gain)

ROOT = Path(__file__).resolve().parents[1]


def smv1245_cell(mode="raise"):
    sys.path[:0] = [str(ROOT / 'examples/real_parts'), str(ROOT / 'examples/real_parts/boundary')]
    from _stub_table import load_s1p, make_y_override, STUB_80GHz
    from real_part_designs import spec_smv1245
    from tvtl.varactor_palette import get_entry
    var = get_entry('SMV1245-079LF').to_realistic_varactor(v_bias_v=0)
    return cell_functions_from_spec(spec=spec_smv1245(n_cells=1), varactor=var,
                                    stub_y_fn=make_y_override(*load_s1p(STUB_80GHz)),
                                    tee_out_of_range=mode)[0]


def test_sample_above_the_tee_edge_raises():
    cell = smv1245_cell()
    cell(11.6e9)                                               # below the 11.64 GHz edge of this cell
    with pytest.raises(TeeModelOutOfRange, match='^tee_model_out_of_range'):
        cell(12e9)
    assert issubclass(TeeModelOutOfRange, ValueError)


def test_cascade_with_third_harmonic_at_12_ghz_is_refused():
    kwargs = dict(N=20, f_s=4e9, P_pump_dBm=19.5, P_sig_dBm=-56)
    with pytest.raises(TeeModelOutOfRange, match='12 GHz'):
        delivered_gain(cell_fn=smv1245_cell(), **kwargs)        # basis (1,3): 3 f_s = 12 GHz
    out = delivered_gain(cell_fn=smv1245_cell('reduced'), **kwargs)   # explicit 1.0.0 behaviour
    assert np.isfinite(out['absolute_outin_dB'])


def test_gate_refuses_when_the_pump_sample_is_out_of_range():
    out = feasibility(smv1245_cell(), 6e9, N=20)                 # 2 f_s = 12 GHz
    assert out['reasons'] == ['tee_model_out_of_range'] and not out['feasible']
    assert out['T_2fs'] is None and out['margin_delivery_dB'] is None
    assert out['tee_model_refusal'].startswith('tee_model_out_of_range')


def test_sensitivity_basis_beyond_the_edge_is_withheld():
    out = delivered_gain(cell_fn=smv1245_cell(), N=20, f_s=3.1e9, P_pump_dBm=19.5, P_sig_dBm=-56)
    assert np.isfinite(out['absolute_outin_dB'])                 # basis (1,3): 9.3 GHz is in range
    assert out['harmonic_sensitivity_dB'] is None                # (1,3,5,7) needs 15.5 and 21.7 GHz
    assert out['meta']['harmonic_sensitivity_unavailable'].startswith('tee_model_out_of_range')
    assert out['meta']['needs_native_HB_order_check'] is True
