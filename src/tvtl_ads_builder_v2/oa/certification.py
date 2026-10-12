"""Evidence-based gain certification. Missing evidence never implies a pass.

The checks the paper's Methods lists, with their pass rules (``docs/GAIN_DEFINITIONS.md``):

order_convergence       HB orders 5, 7, 9: consecutive absolute out/in within 0.5 dB.
device_tier_agreement   SDD vs full diode at the same phase within 1 dB.
signal_off_pump_on      signal-on output at least 20 dB above the signal-off output.
phase_lobe              eight pump phases 45 deg apart: one maximum, one minimum, 180 +- 15 deg apart, the
                        minimum below the pump-off reference.
idler                   detuned signal (50 MHz): signal and idler outputs within 3 dB.
soft_start, pump_slope  ascending converged pump ladder in steps <= 3 dB; absolute at pump - 3 dB finite.
r_A                     |out(signal off)| / |out(signal on)|, pump on in both: r_A < 0.5.
input_scaling_slope     d(output dBm)/d(input dBm) over the small-signal input sweep: slope >= 0.5.
peak_parity             |analytical best absolute out/in - HB best absolute out/in| <= 5 dB.
pump_sweep              bounded pump sweep (not SPIKE): every rung converged up to the operating pump, and
                        the lower 3 dB shoulder absolute(P_op) - absolute(P_op - 3 dB) < 15 dB. A turnover
                        inside the swept range is not required.
junction_voltage        per-cell junction voltage over the pump cycle: minimum reverse voltage >= 0 (no
                        forward-biased interval) and maximum reverse voltage <= the part's rating.

The last five are new in 1.0.1 and optional on older records: ``None`` means "not run". A not-run check
is listed in ``not_run`` and never counts as a pass; ``methods_complete`` is True only when every check
above is present and passes. ``certified`` keeps the 1.0.0 rule (the first six present and passing) and
additionally fails if any present 1.0.1 check fails.
"""
from dataclasses import dataclass,asdict
import math

R_A_MAX=0.5           # r_A < 0.5 (strict)
SLOPE_MIN=0.5         # slope >= 0.5
PEAK_PARITY_MAX_DB=5.0
SPIKE_SHOULDER_DB=15.0
NEW_FIELDS=('r_A','input_scaling_slope','peak_parity','pump_sweep','junction_voltage')

@dataclass(frozen=True)
class GainRecord:
    name: str
    absolute_outin_dB: float | None = None
    order_convergence: dict | None = None  # {5: absolute, 7: absolute, 9: absolute}
    device_tier_agreement: dict | None = None  # sdd, diode; same phase_deg
    signal_off_pump_on: dict | None = None  # signal_on_dBm, signal_off_dBm
    phase_lobe: dict | None = None  # phases_deg, absolute_dB, passive_dB
    idler: dict | None = None  # detuning_Hz, signal_dBm, idler_dBm
    soft_start: list | None = None  # [{pump_dBm, converged}]
    pump_slope: float | None = None  # absolute at pump -3 dB
    tier: str = 'ADS-REALIZED'
    r_A: float | None = None  # |out(signal off)| / |out(signal on)|
    input_scaling_slope: float | None = None  # d out_dBm / d in_dBm
    peak_parity: dict | None = None  # analytical_dB, hb_dB (best absolute out/in)
    pump_sweep: dict | None = None  # pump_dBm, absolute_dB, converged (lists), operating_pump_dBm
    junction_voltage: dict | None = None  # min_reverse_V, max_reverse_V, rating_V

    def evaluate(self):
        missing=[];failed=[];not_run=[]
        finite=lambda x:isinstance(x,(float,int)) and not isinstance(x,bool) and math.isfinite(x)
        for name in ('absolute_outin_dB','order_convergence','device_tier_agreement',
                     'signal_off_pump_on','phase_lobe','idler','soft_start','pump_slope'):
            if getattr(self,name) is None:missing.append(name)
        for name in NEW_FIELDS:
            if getattr(self,name) is None:not_run.append(name)
        if self.tier!='ADS-REALIZED':failed.append('tier')
        if self.absolute_outin_dB is not None and (not finite(self.absolute_outin_dB) or self.absolute_outin_dB<=0):failed.append('absolute_outin_dB')
        def check(name,fn):
            value=getattr(self,name)
            if value is None:return
            try:
                if not fn(value):failed.append(name)
            except (KeyError,TypeError,ValueError,IndexError):missing.append(name+'.required_fields')
        def orders(d):
            v=[d.get(k,d.get(str(k))) for k in (5,7,9)]
            return all(finite(x) for x in v) and abs(v[0]-v[1])<=.5 and abs(v[1]-v[2])<=.5
        check('order_convergence',orders)
        check('device_tier_agreement',lambda d:all(finite(d[k]) for k in ('sdd','diode','phase_deg')) and abs(d['sdd']-d['diode'])<=1)
        check('signal_off_pump_on',lambda d:all(finite(d[k]) for k in ('signal_on_dBm','signal_off_dBm')) and d['signal_on_dBm']-d['signal_off_dBm']>=20)
        def lobe(d):
            p=d['phases_deg'];v=d['absolute_dB'];off=d['passive_dB']
            if len(p)!=8 or len(v)!=8 or not all(finite(x) for x in p+v+[off]):return False
            pairs=sorted((x%360,y) for x,y in zip(p,v));p,v=map(list,zip(*pairs))
            if len(set(p))!=8:return False
            peaks=[i for i in range(8) if v[i]>v[i-1] and v[i]>v[(i+1)%8]]
            lows=[i for i in range(8) if v[i]<v[i-1] and v[i]<v[(i+1)%8]]
            if len(peaks)!=1 or len(lows)!=1:return False
            delta=(p[peaks[0]]-p[lows[0]])%360
            return abs(delta-180)<=15 and v[lows[0]]<off
        check('phase_lobe',lobe)
        check('idler',lambda d:all(finite(d[k]) for k in ('detuning_Hz','signal_dBm','idler_dBm')) and abs(d['detuning_Hz']-50e6)<=1 and abs(d['signal_dBm']-d['idler_dBm'])<=3)
        def start(v):
            return len(v)>=2 and all(x['converged'] is True and finite(x['pump_dBm']) for x in v) and all(0<b['pump_dBm']-a['pump_dBm']<=3 for a,b in zip(v,v[1:]))
        check('soft_start',start)
        check('pump_slope',finite)
        check('r_A',lambda x:finite(x) and 0<=x<R_A_MAX)
        check('input_scaling_slope',lambda x:finite(x) and x>=SLOPE_MIN)
        check('peak_parity',lambda d:finite(d['analytical_dB']) and finite(d['hb_dB']) and abs(d['analytical_dB']-d['hb_dB'])<=PEAK_PARITY_MAX_DB)
        def sweep(d):
            p,a,c,op=list(d['pump_dBm']),list(d['absolute_dB']),list(d['converged']),d['operating_pump_dBm']
            if not (len(p)==len(a)==len(c)>=2 and finite(op) and all(finite(x) for x in p)):return False
            if any(b<=x for x,b in zip(p,p[1:])):return False
            at=lambda q:next((i for i,x in enumerate(p) if abs(x-q)<=1e-6),None)
            i,j=at(op),at(op-3)
            if i is None or j is None:return False
            if not all(c[k] is True and finite(a[k]) for k in range(i+1)):return False
            return a[i]-a[j]<SPIKE_SHOULDER_DB
        check('pump_sweep',sweep)
        check('junction_voltage',lambda d:all(finite(d[k]) for k in ('min_reverse_V','max_reverse_V','rating_V')) and d['min_reverse_V']>=0 and d['max_reverse_V']<=d['rating_V'])
        certified=not missing and not failed
        return dict(certified=certified,missing=missing,failed=failed,not_run=not_run,
                    methods_complete=certified and not not_run)

    @property
    def certified(self):return self.evaluate()['certified']

    @property
    def missing(self):return self.evaluate()['missing']

    def to_dict(self):return dict(**asdict(self),**self.evaluate())

    def to_markdown(self):
        result=self.evaluate()
        return (f"{self.name}: certified={result['certified']}; methods_complete={result['methods_complete']}; "
                f"absolute={self.absolute_outin_dB} dB; missing={result['missing']}; failed={result['failed']}; "
                f"not_run={result['not_run']}")
