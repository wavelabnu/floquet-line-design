"""Passive delivery/attenuation feasibility; not an amplification claim.

Gate constants (tier ANALYTICAL; docs/GAIN_DEFINITIONS.md, "Stage-0 feasibility gate"):

``T_REQ = 0.17378`` -- adopted delivery threshold, cascade-anchored:
    T_req = m*_D / (k V_bt^max) = 0.235625 / (0.4547 x 2.98197) = 0.17378 (+1.645 dB vs 0.1438), with
    m*_D = 0.235625  the smallest uniform junction depth at which the N = 20 cascade of the as-built SMV1245
                     cell (0 V, stub scale 1) gives a 1.0 dB on/off boost at f_s (ideal phase matching, best of
                     24 global pump phases), minimised over f_s = 2.0-6.0 GHz on a 0.1 GHz grid restricted to
                     f_s that pass gate D; the minimum is at the first such f_s, 3.3 GHz;
    k = 0.4547 /V    |C1|/C0 per volt of the SMV1245 junction at 0 V, (M/V_J)(C_J0/C_0);
    V_bt^max = 2.98197 V  the largest branch-top pump amplitude over the N = 20 cells (cell 4), linear pump
                     solve at +19.5 dBm incident from a 50-ohm source at the board plane, f_p = 6.2 GHz.
``T_REQ_SCREEN_2026_09 = 0.1438`` -- the earlier screening threshold, reverse-engineered from the fabricated
    board (measured +16 dB at +29 dBm). The 2026-09 screens (R1 leg 1, the R6 maps, Fig. 2b,c) used it.
``ALPHA_MAX = 0.02`` Np/cell -- per-cell loss budget of pump transport (gate B, alpha(2 f_s)) and signal
    transport (gate D, alpha(f_s)).
"""
from __future__ import annotations
import numpy as np
from tvtl.theory.dispersion import bloch_dispersion

T_REQ = 0.17378
T_REQ_SCREEN_2026_09 = 0.1438
ALPHA_MAX = 0.02

def feasibility(cell_fn, f_s, T_req=T_REQ, alpha_max=ALPHA_MAX, N=None,
                board_slot_mm2=None, stub_area_mm2=None, *,
                period_mm=None, board_width_mm=None, main_width_mm=None,
                board_area_per_cell_mm2=None, phase_margin=False, P_pump_dBm=19.5, depth_ceiling=None):
    """Evaluate a BranchCell callable. Area inputs are both per cell, in mm².

    ``T_req`` defaults to the adopted ``T_REQ`` (0.17378); pass ``T_REQ_SCREEN_2026_09`` (0.1438) to
    reproduce the 2026-09 screens. Both are defined in the module docstring.

    ``phase_margin=True`` (off by default; on in the optimiser's ``cascade`` gate) adds the cascade
    condition ``0.5 m_del - m_EP >= 0`` at ``N`` cells and board-plane pump ``P_pump_dBm``, reason
    ``phase_margin_negative`` (``tvtl_ads_builder_v2.analytical.cascade_budget``). Its coupling fit
    samples 3 f_s; above the tee-model edge the gate refuses with ``tee_model_out_of_range``.

    ``depth_ceiling={'bias_V': ..., 'v_max_V': ...}`` (on in the optimiser's ``cascade`` gate) refuses a
    cell whose delivered depth at ``P_pump_dBm`` exceeds the C(V) ceiling at the bias point, reason
    ``delivered_depth_above_ceiling`` (``tvtl.design.depth_ceiling``).

    board_slot_mm2 is the K7 routing slot, never total board area / N.
    Alternatively provide period, board width and main-line width to derive K7.
    board_area_per_cell_mm2 is an independent optional upper bound.
    N sets survival and the dB margins only; no fixed board or varactor is assumed. Unknown area
    yields constructible=None and does not invent a construction violation.
    The quantitative alpha thresholds are mandatory, irrespective of trace: gate B on the pump,
    alpha(2 f_s) <= alpha_max, and gate D on the signal, alpha(f_s) <= alpha_max
    (``signal_transport_above_budget``). Gate D also refuses lossy cells whose half-trace real part
    stays inside [-1, 1], which the passband flag ``in_band_s`` cannot see.
    Margins: delivery 20 log10(T(2 f_s)/T_req) dB; transport (alpha_max - alpha) Np/cell and,
    with N, 8.686 N (alpha_max - alpha) dB. A negative margin fails its gate.
    """
    if not np.isfinite(f_s) or f_s<=0:raise ValueError('f_s must be positive')
    if not np.isfinite(T_req) or T_req<=0 or not np.isfinite(alpha_max) or alpha_max<0:raise ValueError('Invalid thresholds')
    if N is not None and (isinstance(N,bool) or int(N)!=N or N<1):raise ValueError('N must be a positive integer')
    if (phase_margin or depth_ceiling is not None) and N is None:raise ValueError('phase_margin and depth_ceiling need N')
    geometry=(period_mm,board_width_mm,main_width_mm)
    if any(x is not None for x in geometry):
        if not all(x is not None and np.isfinite(x) and x>0 for x in geometry):
            raise ValueError("Provide positive period, board width and main width together")
        if main_width_mm>=board_width_mm:raise ValueError("Main line must fit across board")
        geometric_slot=period_mm*(board_width_mm-main_width_mm)/2
        if board_slot_mm2 is not None and not np.isclose(board_slot_mm2,geometric_slot):
            raise ValueError("Explicit slot disagrees with K7 geometry")
        board_slot_mm2=geometric_slot
    for x in (board_slot_mm2,stub_area_mm2,board_area_per_cell_mm2):
        if x is not None and (not np.isfinite(x) or x<0):raise ValueError('Area must be nonnegative')
    def sample(f):
        c=cell_fn(f);a=c.abcd(frequency_hz=f)
        alpha=float(abs(complex(bloch_dispersion(tuple(a.ravel()),1.)).real))
        # Lossy traces can exceed unity slightly inside a propagating band.
        inband=bool(abs(((a[0,0]+a[1,1])/2).real)<=1)
        zc=1/(2j*np.pi*f*c.junction_capacitance_f)
        return float(abs(zc/(zc+c.series_impedance_ohm))),alpha,inband
    # A cell sample above its tee-model edge is a refusal (tee_model_out_of_range), never a number.
    try:
        ts,als,bs=sample(f_s);tp,alp,bp=sample(2*f_s);refused=None
    except ValueError as exc:
        if getattr(exc,'reason',None)!='tee_model_out_of_range':raise
        ts=als=bs=tp=alp=bp=None;refused=str(exc)
    build=None if board_slot_mm2 is None or stub_area_mm2 is None else bool(stub_area_mm2<=board_slot_mm2)
    board_area_ok=None if board_area_per_cell_mm2 is None or stub_area_mm2 is None else bool(stub_area_mm2<=board_area_per_cell_mm2)
    constructible=False if build is False or board_area_ok is False else build
    reasons=[] if refused is None else ['tee_model_out_of_range']
    if refused is None:
        if tp<T_req:reasons.append('pump_delivery_below_threshold')
        if alp>alpha_max:reasons.append('pump_attenuation_above_threshold')
        if als>alpha_max:reasons.append('signal_transport_above_budget')
        if not bs:reasons.append('signal_stopband')
    cascade=None
    if phase_margin and refused is None:
        from tvtl_ads_builder_v2.analytical.cascade_budget import cascade_budget
        try:
            b=cascade_budget(cell_fn,f_s,N=int(N),P_pump_dBm=P_pump_dBm)
            cascade={k:b[k] for k in ('phase_margin','m_del','m_EP','delta_theta_rad','kappa_per_m','P_pump_dBm')}
            if b['phase_margin']<0:reasons.append('phase_margin_negative')
        except ValueError as exc:
            if getattr(exc,'reason',None)!='tee_model_out_of_range':raise
            reasons.append('tee_model_out_of_range');cascade={'refusal':str(exc)}
    depth=None
    if depth_ceiling is not None and refused is None:
        from tvtl.design.depth_ceiling import depth_ceiling_check
        depth=depth_ceiling_check(cell_fn,f_s,N=int(N),P_pump_dBm=depth_ceiling.get('P_pump_dBm',P_pump_dBm),
                                  bias_V=depth_ceiling['bias_V'],v_max_V=depth_ceiling.get('v_max_V'))
        if not depth['ok']:reasons.append(depth['reason'])
    if build is False:reasons.append('stub_exceeds_cell_slot')
    if board_area_ok is False:reasons.append('stub_exceeds_board_area_per_cell')
    db=20/np.log(10);ok=refused is None
    return dict(routing_slot_mm2=board_slot_mm2,board_area_per_cell_mm2=board_area_per_cell_mm2,
        routing_slot_ok=build,board_area_ok=board_area_ok,T_2fs=tp,T_fs=ts,alpha_2fs_Np_per_cell=alp,alpha_fs=als,
        in_band_p=bp,in_band_s=bs,pump_survival_e_minus_N_alpha=None if N is None or not ok else float(np.exp(-N*alp)),
        signal_survival_e_minus_N_alpha=None if N is None or not ok else float(np.exp(-N*als)),
        margin_delivery_dB=None if not ok else float(20*np.log10(tp/T_req)) if tp>0 else -np.inf,
        margin_pump_transport_Np_per_cell=alpha_max-alp if ok else None,
        margin_signal_transport_Np_per_cell=alpha_max-als if ok else None,
        margin_pump_transport_dB=None if N is None or not ok else float(db*N*(alpha_max-alp)),
        margin_signal_transport_dB=None if N is None or not ok else float(db*N*(alpha_max-als)),
        tee_model_refusal=refused if refused else (cascade or {}).get('refusal'),
        phase_margin=(cascade or {}).get('phase_margin'),cascade=cascade,depth_ceiling=depth,
        constructible=constructible,feasible=not reasons,reasons=reasons,tier='ANALYTICAL',T_req=T_req,alpha_max=alpha_max)


def candidate_cell_factory(*, varactor, substrate, f_s, N, main_period_m=None, bias_return_ohm=None):
    """Return the physical Stage-2 cell builder used by the optional prefilter.

    Main width is chosen before scoring by the existing deterministic Bloch
    match. No new optimizer constants or native simulator calls are introduced.
    An OPEN-stub family has no DC path from the varactor's stub-side terminal: it is refused
    (``NoDCReturn``, reason ``no_dc_return_for_varactor``) unless ``bias_return_ohm`` names the
    bias-return resistor, which the cell then carries in parallel with the stub.
    """
    from tvtl.optimization.stage1_per_cell import StubCandidate
    from tvtl.optimization.stage2_main_tl import match_z_tl
    from tvtl.optimization.topology_generator import topology_y_stub_fn, Termination
    from tvtl_ads_builder_v2.parity.self_pumped_design import optimized_self_pumped_spec
    from tvtl_ads_builder_v2.models import ViagndVia, MlefOpen
    from tvtl_ads_builder_v2.analytical.delivered_gain import cell_functions_from_spec
    from tvtl_ads_builder_v2.analytical.floquet_stub import build_stub_y_fn_from_spec,extract_stub_segments
    from tvtl_ads_builder_v2.validation.dc_return import NoDCReturn, REASON
    def build(topology,params):
        if topology.termination is Termination.OPEN and bias_return_ohm is None:
            raise NoDCReturn(f"{REASON}: {topology.key} leaves the varactor's stub-side terminal without a DC path "
                             "(pass bias_return_ohm for a bias-return element)")
        cand=StubCandidate(topology,tuple(params),0.,f_s)
        match=match_z_tl(cand,varactor,substrate=substrate,f0_Hz=f_s)
        if not match.realizable:raise ValueError('Unrealizable through-line')
        kw={} if main_period_m is None else dict(main_period_m=main_period_m)
        spec=optimized_self_pumped_spec(topology=topology,params=tuple(params),varactor=varactor,
             substrate=substrate,n_cells=N,main_width_m=match.w_main_m,f_signal_Hz=f_s,f_pump_Hz=2*f_s,
             input_match=None,output_match=None,via_model=ViagndVia(),open_end_model=MlefOpen(),
             bias_return_ohm=bias_return_ohm if topology.termination is Termination.OPEN else None,**kw)
        if topology.termination is Termination.COMBINED:
            y=topology_y_stub_fn(topology,np.asarray(params),substrate)
        else:
            seg,_,term=extract_stub_segments(spec,0);sub=substrate.to_tvtl_substrate()
            # a DOUBLE family's two vias in parallel, as the emitted deck renders them
            load=ViagndVia().analytical_termination(substrate=sub,n_parallel=max(1,topology.n_via_parallel)) if term=='short' else MlefOpen().analytical_termination(width_m=seg[-1].width_m,substrate=sub,f_ref_Hz=f_s)
            y=build_stub_y_fn_from_spec(spec,0,termination_override=load)
            if topology.termination is Termination.OPEN:
                y=(lambda w,y=y,g=1/float(bias_return_ohm):y(w)+g)   # bias return across the stub input
        return cell_functions_from_spec(spec=spec,varactor=varactor,stub_y_fn=y)[0]
    return build
