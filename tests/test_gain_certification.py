import math,os,time
import pytest
from tvtl_ads_builder_v2.oa.certification import GainRecord
from tvtl_ads_builder_v2.simulate.runner import _own_dataset

def complete():
 return dict(name='complete',absolute_outin_dB=3,order_convergence={5:3,7:3.1,9:3.2},
 device_tier_agreement=dict(sdd=3,diode=3.2,phase_deg=0),signal_off_pump_on=dict(signal_on_dBm=-27,signal_off_dBm=-50),
 phase_lobe=dict(phases_deg=list(range(0,360,45)),absolute_dB=[1+2*math.cos(math.radians(x)) for x in range(0,360,45)],passive_dB=0),
 idler=dict(detuning_Hz=50e6,signal_dBm=-27,idler_dBm=-28),soft_start=[dict(pump_dBm=x,converged=True) for x in (0,3,6)],pump_slope=1.)

def test_complete_and_missing_evidence():
 assert GainRecord(**complete()).certified
 assert not GainRecord('old',absolute_outin_dB=10).certified
 assert 'order_convergence' in GainRecord('old').missing

def test_r5_order_collapse_cannot_certify():
 d=complete();d['order_convergence']={5:14.746546,7:-4.103114,9:-4.1}
 assert 'order_convergence' in GainRecord(**d).evaluate()['failed']

@pytest.mark.parametrize('field,value',[('pump_slope',float('nan')),('absolute_outin_dB',0),('tier','ANALYTICAL'),('soft_start',[dict(pump_dBm=0,converged=True),dict(pump_dBm=6,converged=True)])])
def test_invalid_evidence(field,value):
 d=complete();d[field]=value;assert not GainRecord(**d).certified

def test_own_dataset_two_decks(tmp_path):
 start=time.time()-1
 old=tmp_path/'a.ckt.ds';old.write_text('old');os.utime(old,(start-100,start-100))
 own=tmp_path/'b.ckt.ds';own.write_text('own')
 assert _own_dataset(tmp_path,tmp_path/'b.ckt',start)==str(own)
 other=tmp_path/'c.ckt.ds';other.write_text('other')
 assert _own_dataset(tmp_path,tmp_path/'b.ckt',start)==str(own)
 with pytest.raises(RuntimeError,match='Ambiguous'):_own_dataset(tmp_path,tmp_path/'d.ckt',start)


def test_absolute_is_default_and_legacy_is_explicit():
 from tvtl.optimization.workflow import WorkflowConfig
 assert WorkflowConfig().order_by=='absolute_outin_dB'
 assert WorkflowConfig(order_by='coherent_peak').order_by=='coherent_peak'


def test_six_published_records_unchanged():
 import json,hashlib
 from pathlib import Path
 root=Path(__file__).resolve().parents[1]
 hashes=json.loads((root/'tests/fixtures/r6_delivery_core/stage3_example_hashes.json').read_text())
 assert len(hashes)==4  # three headline files plus the three-proxy summary
 for name,digest in hashes.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
