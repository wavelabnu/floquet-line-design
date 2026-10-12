"""Gate D in the stage-0 gate (v1.0.1): signal transport alpha(f_s) <= alpha_max, with its margin."""
import numpy as np
import pytest

from tvtl.design.feasibility import ALPHA_MAX, T_REQ, feasibility
from tvtl_ads_builder_v2.analytical.delivered_gain import BranchCell

C_J = 1e-12
F_R = 1e9                                    # series resonance of the branch
L_S = 1 / ((2 * np.pi * F_R) ** 2 * C_J)


def resonant_cell(f):
    """Line halves around a branch in series resonance at F_R: the signal at F_R meets a near short."""
    t = .2 * f / 1e9
    line = np.array([[np.cos(t), 50j * np.sin(t)], [1j * np.sin(t) / 50, np.cos(t)]])
    return BranchCell(line, line, .5 + 2j * np.pi * f * L_S, C_J, 1, 3)


def test_cell_in_signal_stopband_is_refused_for_signal_transport():
    out = feasibility(resonant_cell, F_R, N=20)
    a = resonant_cell(F_R).abcd(frequency_hz=F_R)
    assert abs((a[0, 0] + a[1, 1]) / 2) > 1                       # evanescent signal (stopband)
    assert out['T_2fs'] >= T_REQ and out['alpha_2fs_Np_per_cell'] <= ALPHA_MAX   # A and B pass
    assert out['alpha_fs'] > 1                                    # Np/cell
    assert out['in_band_s']                                       # the real-part passband flag misses it
    assert out['reasons'] == ['signal_transport_above_budget'] and not out['feasible']
    assert out['margin_signal_transport_Np_per_cell'] == pytest.approx(ALPHA_MAX - out['alpha_fs'])
    assert out['margin_signal_transport_dB'] == pytest.approx(20 / np.log(10) * 20 * (ALPHA_MAX - out['alpha_fs']))
    assert out['margin_signal_transport_dB'] < 0 < out['margin_delivery_dB']


def test_r1_smv2019_gate_d_record_c3_is_admitted():
    """R1 rule point with gate D: SMV2019-079LF at 20 V, f_s 3.1 GHz, N 15, n1_single_short (4.824516, 8.914551) mm."""
    from tvtl.design.feasibility import candidate_cell_factory
    from tvtl.optimization.substrate import ROGERS_4003C
    from tvtl.optimization.topology_generator import Termination, TopologyFamily, ViaConfig
    from tvtl.varactor_palette import get_entry
    var = get_entry('SMV2019-079LF').to_realistic_varactor(v_bias_v=20.0)
    factory = candidate_cell_factory(varactor=var, substrate=ROGERS_4003C, f_s=3.1e9, N=15)
    topology = TopologyFamily(n_segments=1, via=ViaConfig.SINGLE, termination=Termination.SHORT)
    out = feasibility(factory(topology, (4.824516e-3, 8.914551e-3)), 3.1e9, N=15)
    assert out['feasible'] and out['reasons'] == []
    assert out['alpha_fs'] == pytest.approx(0.008271, abs=1e-5)
    assert out['margin_signal_transport_dB'] > 0 and out['margin_pump_transport_dB'] > 0
    assert out['margin_delivery_dB'] > 0
