"""Independent circuit checks and archived R5 reproduction; no simulator."""
from pathlib import Path
import json
import numpy as np
import pytest

from tvtl_ads_builder_v2.analytical.delivered_gain import (
    BranchCell, delivered_gain, delivered_pump_profile, modulation_profile,
    harmonic_cell, loaded_harmonic_response, cell_functions_from_spec,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = json.loads((ROOT/'tests/fixtures/r6_delivery_core/r5_reference_cells.json').read_text())


def decode(v):
    return np.asarray(v['re']) + 1j*np.asarray(v['im'])


def fixture_cell(row, M=None):
    values = {c['f_Hz']: c for c in row['cells']}
    def sample(f):
        c = values[f]
        return BranchCell(decode(c['left']), decode(c['right']), complex(decode(c['series'])),
                          row['C0'], row['M'] if M is None else M, row['Vj'], complex(decode(c['shunt'])))
    return sample


@pytest.mark.parametrize('row', REFERENCES, ids=lambda r:r['name'])
def test_r5_reference_gain_to_one_nanodecibel(row):
    out = delivered_gain(cell_fn=fixture_cell(row), N=row['N'], f_s=row['fs'],
                         P_pump_dBm=row['pump'], P_sig_dBm=row['signal'], Z0=row['Z0'])
    assert out['onoff_dB'] == pytest.approx(row['expected_onoff_dB'], abs=1e-9)
    assert out['meta']['certified'] is False
    assert np.isfinite(out['harmonic_sensitivity_dB'])


@pytest.mark.parametrize('N,Z0', [(20,50), (7,75), (1,25)])
def test_loaded_recursion_against_independent_nodal_solve(N,Z0):
    f=6.2e9
    # Non-identical cells, unrelated C/Vj/M and reference impedance.
    def cell(k):
        theta=.15+.01*k
        line=np.array([[np.cos(theta),1j*41*np.sin(theta)],[1j*np.sin(theta)/41,np.cos(theta)]])
        return lambda f:BranchCell(line,line,1.2+2j*np.pi*f*(.4+k*.01)*1e-9,1.1e-12,.8,2.2)
    funcs=[cell(k) for k in range(N)];mat=np.zeros((N+1,N+1),complex)
    for k,fn in enumerate(funcs):
        a,b,c,d=fn(f).abcd(frequency_hz=f).ravel()
        mat[k,k]+=d/b;mat[k+1,k+1]+=a/b
        mat[k,k+1]+=(b*c-a*d)/b;mat[k+1,k]+=-1/b
    mat[0,0]+=1/Z0;mat[-1,-1]+=1/Z0
    rhs=np.zeros(N+1,complex);rhs[0]=2*np.sqrt(2*Z0*.01)/Z0
    v=np.linalg.solve(mat,rhs);expected=[]
    for k,fn in enumerate(funcs):
        sample=fn(f);a,b,c,d=sample.abcd(frequency_hz=f).ravel()
        current=(v[k]-a*v[k+1])/b
        branch=sample.right[0,0]*v[k+1]+sample.right[0,1]*current
        zc=1/(2j*np.pi*f*sample.junction_capacitance_f)
        expected.append(branch*zc/(zc+sample.series_impedance_ohm))
    actual=delivered_pump_profile(cell_fn=funcs,N=N,f_p=f,P_inc_dBm=10,Z0=Z0)
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)


@pytest.mark.parametrize('N,Z0', [(20,50),(4,75)])
def test_zero_modulation_equals_passive(N,Z0):
    row=REFERENCES[1]
    out=delivered_gain(cell_fn=fixture_cell(row,M=0),N=N,f_s=row['fs'],P_pump_dBm=19.5,P_sig_dBm=-43,Z0=Z0)
    assert out['onoff_dB']==pytest.approx(0,abs=1e-9)
    assert out['absolute_outin_dB']==pytest.approx(out['passive_S21_dB'],abs=1e-9)


def test_uniform_prescribed_coefficients_bridge_to_existing_core():
    from tvtl.varactors import RealisticVaractor
    from tvtl_ads_builder_v2.parity import level10_realistic_c_canonical
    from tvtl_ads_builder_v2.analytical.realistic_c_gain import realistic_pumped_unit_cell_abcd
    from tvtl.floquet.harmonic_abcd import cascade_harmonic_abcd,harmonic_s_parameters
    var=RealisticVaractor(CJ0_f=.3e-12,VJ_v=3.5,M=1.7,CP_f=.03e-12,L_pkg_h=.3e-9,R_series_ohm=.5)
    spec=level10_realistic_c_canonical(varactor=var,stub_scale=1.,n_cells=5,f_signal_Hz=3.1e9,f_pump_Hz=6.2e9)
    y=lambda w:1/(2+1j*w*.2e-9)
    fns=cell_functions_from_spec(spec=spec,varactor=var,stub_y_fn=y,tee_reference_frequency_hz=3.1e9)
    coeff=var.capacitance_coeffs_from_junction_vpump(.2,phi=0)
    new=harmonic_cell(cell_fn=fns[0],f_s=3.1e9,indices=(-1,1,3),coefficients=coeff)
    old=realistic_pumped_unit_cell_abcd(spec,var,control_port_vpump_v=.2,truncation_N=1,stub_y_fn_override=y)
    for a,b in zip(new,old):np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
    direct=harmonic_s_parameters(*cascade_harmonic_abcd([old]*5),50,50)[1]
    stable=loaded_harmonic_response(cells=[new]*5,Z0=50)[0]
    for a,b in [(direct[1,1],stable[1,1]),(direct[1,0],stable[1,0])]:
        assert 20*np.log10(abs(a))==pytest.approx(20*np.log10(abs(b)),abs=1e-9)


def test_caps_and_explicit_device_parameters():
    m,p,cap=modulation_profile(V_j=[1,2j],M=[1,2],Vj=[4,2],m_cap=.8)
    np.testing.assert_allclose(m,[.25,.8]);np.testing.assert_allclose(p,[0,np.pi/2])
    assert cap.tolist()==[False,True]


@pytest.mark.parametrize('kwargs',[dict(N=0),dict(N=1.5),dict(Z0=0),dict(f_p=0),dict(P_inc_dBm=float('nan'))])
def test_invalid_physical_arguments_fail(kwargs):
    args=dict(cell_fn=fixture_cell(REFERENCES[0]),N=20,f_p=6.2e9,P_inc_dBm=19.5,Z0=50);args.update(kwargs)
    with pytest.raises(ValueError):delivered_pump_profile(**args)


def test_bare_abcd_cannot_invent_internal_delivery():
    with pytest.raises(TypeError,match='ABCD alone'):
        delivered_pump_profile(cell_fn=lambda f:np.eye(2),N=1,f_p=1e9,P_inc_dBm=0)


def test_signal_scaling_and_phase_output_consistency():
    row=REFERENCES[1];args=dict(cell_fn=fixture_cell(row),N=20,f_s=row['fs'],P_pump_dBm=19.5)
    a=delivered_gain(**args,P_sig_dBm=-56);b=delivered_gain(**args,P_sig_dBm=-36,phase=a['phase_rad'])
    assert a['absolute_outin_dB']==pytest.approx(b['absolute_outin_dB'],abs=1e-10)
    np.testing.assert_allclose(b['per_cell_signal'],10*a['per_cell_signal'],rtol=1e-10)
